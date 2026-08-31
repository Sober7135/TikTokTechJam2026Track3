#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kBatch = 64;
constexpr int kHeads = 4;
constexpr int kSequence = 1024;
constexpr int kHeadDim = 32;
constexpr int kRows = 16;
constexpr int kWarps = 8;

__device__ __forceinline__ uint32_t pack_bf16(__nv_bfloat16 low,
                                              __nv_bfloat16 high) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(low)) |
         (static_cast<uint32_t>(__bfloat16_as_ushort(high)) << 16);
}

__device__ __forceinline__ void
mma_m16n8k16(float (&acc)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};\n"
               : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                 "r"(b[1]));
}

template <int RowStart>
__device__ __forceinline__ void
load_query_fragment(const __nv_bfloat16 *query_tile, int reduction_start,
                    int lane, uint32_t (&a)[4]) {
  const int group = lane >> 2;
  const int thread = lane & 3;
  const int row0 = group;
  const int row8 = row0 + 8;
  const int column0 = reduction_start + thread * 2;
  const int column8 = column0 + 8;

  // PTX m16n8k16 row-major A order is row0-low, row8-low,
  // row0-high, row8-high. O74 exchanged the middle two registers.
  a[0] = pack_bf16(query_tile[row0 * kHeadDim + column0],
                   query_tile[row0 * kHeadDim + column0 + 1]);
  a[1] = pack_bf16(query_tile[row8 * kHeadDim + column0],
                   query_tile[row8 * kHeadDim + column0 + 1]);
  a[2] = pack_bf16(query_tile[row0 * kHeadDim + column8],
                   query_tile[row0 * kHeadDim + column8 + 1]);
  a[3] = pack_bf16(query_tile[row8 * kHeadDim + column8],
                   query_tile[row8 * kHeadDim + column8 + 1]);
}

__device__ __forceinline__ void
load_key_fragment(const __nv_bfloat16 *key, int64_t key_base,
                  int64_t key_stride_row, int matrix_column,
                  int reduction_start, int lane, uint32_t (&b)[2]) {
  const int group = lane >> 2;
  const int thread = lane & 3;
  const int key_row = matrix_column + group;
  const int reduction0 = reduction_start + thread * 2;
  const int reduction8 = reduction0 + 8;
  b[0] = pack_bf16(key[key_base + key_row * key_stride_row + reduction0],
                   key[key_base + key_row * key_stride_row + reduction0 + 1]);
  b[1] = pack_bf16(key[key_base + key_row * key_stride_row + reduction8],
                   key[key_base + key_row * key_stride_row + reduction8 + 1]);
}

template <int RowStart, int KeyCount>
__device__ __forceinline__ void
store_score_fragment(__half *scores, const float (&acc)[4],
                     int matrix_column, int row_block, float scale, int lane) {
  const int group = lane >> 2;
  const int thread = lane & 3;
  const int rows[4] = {group, group, group + 8, group + 8};
  const int columns[4] = {
      matrix_column + thread * 2, matrix_column + thread * 2 + 1,
      matrix_column + thread * 2, matrix_column + thread * 2 + 1};
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    const __nv_bfloat16 rounded_dot = __float2bfloat16_rn(acc[index]);
    const __nv_bfloat16 rounded_scaled = __float2bfloat16_rn(
        __bfloat162float(rounded_dot) * scale);
    const int query_row = RowStart + row_block * kRows + rows[index];
    const float transported =
        columns[index] <= query_row
            ? __bfloat162float(rounded_scaled)
            : -std::numeric_limits<float>::infinity();
    // Reproduce the established Case-13 BF16-to-FP16 conversion here.  BF16
    // mantissas that are in FP16's exponent range are exact; overflow and the
    // causal -infinity follow CUDA's ordinary RNE half conversion semantics.
    scores[rows[index] * KeyCount + columns[index]] =
        __float2half_rn(transported);
  }
}

template <int KeyCount>
__device__ __forceinline__ void native_softmax_rows(unsigned short *storage,
                                                    int warp, int lane) {
  // ATen's persistent template pads 768 to next_power_of_two=1024.  Keeping
  // those final eight -infinity iterations preserves its local comparison,
  // exp, and addition instruction order even though they cannot change a
  // finite row's result.
  constexpr int kPaddedCount = KeyCount == 768 ? 1024 : KeyCount;
  constexpr int kIterations = kPaddedCount / 32;
  __half *scores = reinterpret_cast<__half *>(storage);
  __nv_bfloat16 *probabilities =
      reinterpret_cast<__nv_bfloat16 *>(storage);
#pragma unroll
  for (int wave = 0; wave < kRows / kWarps; ++wave) {
    const int row = warp + wave * kWarps;
    float elements[kIterations];
#pragma unroll
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      const int column = lane + iteration * 32;
      elements[iteration] =
          column < KeyCount
              ? __half2float(scores[row * KeyCount + column])
              : -std::numeric_limits<float>::infinity();
    }

    float maximum = elements[0];
#pragma unroll
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      maximum = maximum > elements[iteration] ? maximum : elements[iteration];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other = __shfl_xor_sync(0xffffffffU, maximum, offset, 32);
      maximum = maximum < other ? other : maximum;
    }

    float sum = 0.0f;
#pragma unroll
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      elements[iteration] = std::exp(elements[iteration] - maximum);
      sum += elements[iteration];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_xor_sync(0xffffffffU, sum, offset, 32);
    }

#pragma unroll
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      const int column = lane + iteration * 32;
      const float result =
          sum == 0.0f ? std::numeric_limits<float>::quiet_NaN()
                      : elements[iteration] / sum;
      if (column < KeyCount) {
        probabilities[row * KeyCount + column] = __float2bfloat16_rn(result);
      }
    }
  }
}

template <int KeyCount>
__device__ __forceinline__ void
load_probability_fragment(const __nv_bfloat16 *probabilities,
                          int reduction_start, int lane, uint32_t (&a)[4]) {
  const int group = lane >> 2;
  const int thread = lane & 3;
  const int row0 = group;
  const int row8 = group + 8;
  const int column0 = reduction_start + thread * 2;
  const int column8 = column0 + 8;
  a[0] = pack_bf16(probabilities[row0 * KeyCount + column0],
                   probabilities[row0 * KeyCount + column0 + 1]);
  a[1] = pack_bf16(probabilities[row8 * KeyCount + column0],
                   probabilities[row8 * KeyCount + column0 + 1]);
  a[2] = pack_bf16(probabilities[row0 * KeyCount + column8],
                   probabilities[row0 * KeyCount + column8 + 1]);
  a[3] = pack_bf16(probabilities[row8 * KeyCount + column8],
                   probabilities[row8 * KeyCount + column8 + 1]);
}

__device__ __forceinline__ void
load_value_fragment(const __nv_bfloat16 *value, int64_t value_base,
                    int64_t value_stride_row, int matrix_column,
                    int reduction_start, int lane, uint32_t (&b)[2]) {
  const int group = lane >> 2;
  const int thread = lane & 3;
  const int output_column = matrix_column + group;
  const int key0 = reduction_start + thread * 2;
  const int key8 = key0 + 8;
  b[0] = pack_bf16(
      value[value_base + key0 * value_stride_row + output_column],
      value[value_base + (key0 + 1) * value_stride_row + output_column]);
  b[1] = pack_bf16(
      value[value_base + key8 * value_stride_row + output_column],
      value[value_base + (key8 + 1) * value_stride_row + output_column]);
}

template <int RowStart>
__device__ __forceinline__ void
store_context_fragment(__nv_bfloat16 *context, int64_t context_base,
                       int64_t context_stride_row, const float (&acc)[4],
                       int matrix_column, int row_block, int lane) {
  const int group = lane >> 2;
  const int thread = lane & 3;
  const int rows[4] = {group, group, group + 8, group + 8};
  const int columns[4] = {
      matrix_column + thread * 2, matrix_column + thread * 2 + 1,
      matrix_column + thread * 2, matrix_column + thread * 2 + 1};
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    const int query_row = RowStart + row_block * kRows + rows[index];
    context[context_base + query_row * context_stride_row + columns[index]] =
        __float2bfloat16_rn(acc[index]);
  }
}

template <int RowStart, int KeyCount>
__global__ void case13_exact_attention_kernel(
    const __nv_bfloat16 *query, const __nv_bfloat16 *key,
    const __nv_bfloat16 *value, __nv_bfloat16 *context, float scale,
    int64_t query_stride_batch, int64_t query_stride_head,
    int64_t query_stride_row, int64_t key_stride_batch, int64_t key_stride_head,
    int64_t key_stride_row, int64_t value_stride_batch,
    int64_t value_stride_head, int64_t value_stride_row,
    int64_t context_stride_batch, int64_t context_stride_head,
    int64_t context_stride_row) {
  extern __shared__ __align__(16) unsigned short storage[];
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int linear_thread = warp * 32 + lane;
  const int batch_head = blockIdx.x;
  const int row_block = blockIdx.y;
  const int batch = batch_head / kHeads;
  const int head = batch_head % kHeads;
  const int64_t query_base =
      batch * query_stride_batch + head * query_stride_head;
  const int64_t key_base = batch * key_stride_batch + head * key_stride_head;
  const int64_t value_base =
      batch * value_stride_batch + head * value_stride_head;
  const int64_t context_base =
      batch * context_stride_batch + head * context_stride_head;

  // The same M16xK32 A tile feeds every output N8 fragment. Stage it once per
  // CTA so the M16 decomposition does not multiply global Q loads by N/8.
  __nv_bfloat16 *query_tile =
      reinterpret_cast<__nv_bfloat16 *>(storage);
  unsigned short *scores = storage + kRows * kHeadDim;
#pragma unroll
  for (int index = linear_thread; index < kRows * kHeadDim; index += 256) {
    const int local_row = index / kHeadDim;
    const int column = index % kHeadDim;
    const int query_row = RowStart + row_block * kRows + local_row;
    query_tile[index] =
        query[query_base + query_row * query_stride_row + column];
  }
  __syncthreads();

  constexpr int kFragmentsPerWarp = KeyCount / 64;
  __half *score_half = reinterpret_cast<__half *>(scores);
  // Each score fragment is independent. Finish its two increasing K16 MMAs
  // and store it before starting the next fragment, retaining the exact
  // per-output dependency chain without keeping 4*fragments accumulators live.
#pragma unroll
  for (int fragment = 0; fragment < kFragmentsPerWarp; ++fragment) {
    float score_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int matrix_column = (warp * kFragmentsPerWarp + fragment) * 8;
#pragma unroll
    for (int reduction_start = 0; reduction_start < kHeadDim;
         reduction_start += 16) {
      uint32_t a[4];
      uint32_t b[2];
      load_query_fragment<RowStart>(query_tile, reduction_start, lane, a);
      load_key_fragment(key, key_base, key_stride_row, matrix_column,
                        reduction_start, lane, b);
      mma_m16n8k16(score_acc, a, b);
    }
    store_score_fragment<RowStart, KeyCount>(
        score_half, score_acc, matrix_column, row_block, scale, lane);
  }
  __syncthreads();

  native_softmax_rows<KeyCount>(scores, warp, lane);
  __syncthreads();

  if (warp < 4) {
    float context_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int matrix_column = warp * 8;
#pragma unroll
    for (int reduction_start = 0; reduction_start < KeyCount;
         reduction_start += 16) {
      uint32_t a[4];
      uint32_t b[2];
      load_probability_fragment<KeyCount>(
          reinterpret_cast<const __nv_bfloat16 *>(scores), reduction_start,
          lane, a);
      load_value_fragment(value, value_base, value_stride_row, matrix_column,
                          reduction_start, lane, b);
      mma_m16n8k16(context_acc, a, b);
    }
    store_context_fragment<RowStart>(
        context, context_base, context_stride_row, context_acc, matrix_column,
        row_block, lane);
  }
}

void check_tensor(const torch::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be BF16");
  TORCH_CHECK(tensor.dim() == 4, name, " must have rank four");
  TORCH_CHECK(tensor.size(0) == kBatch && tensor.size(1) == kHeads &&
                  tensor.size(2) == kSequence && tensor.size(3) == kHeadDim,
              name, " must have exact Case-13 BHSD shape");
  TORCH_CHECK(tensor.stride(3) == 1, name, " must have unit-stride HD");
}

void case13_exact_attention(torch::Tensor query, torch::Tensor key,
                            torch::Tensor value, torch::Tensor context,
                            double scale) {
  check_tensor(query, "query");
  check_tensor(key, "key");
  check_tensor(value, "value");
  check_tensor(context, "context");
  TORCH_CHECK(key.device() == query.device() &&
                  value.device() == query.device() &&
                  context.device() == query.device(),
              "Case-13 Q/K/V/context devices must match");

  const c10::cuda::CUDAGuard device_guard(query.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const dim3 grid(kBatch * kHeads, 16);
  const dim3 block(32, kWarps);
  const float scale_value = static_cast<float>(scale);

#define LAUNCH_CASE13_PREFIX(ROW_START, KEY_COUNT)                            \
  case13_exact_attention_kernel<ROW_START, KEY_COUNT>                         \
      <<<grid, block,                                                         \
         kRows * (kHeadDim + KEY_COUNT) * sizeof(unsigned short), stream>>>(  \
          reinterpret_cast<const __nv_bfloat16 *>(                            \
              query.data_ptr<c10::BFloat16>()),                               \
          reinterpret_cast<const __nv_bfloat16 *>(                            \
              key.data_ptr<c10::BFloat16>()),                                 \
          reinterpret_cast<const __nv_bfloat16 *>(                            \
              value.data_ptr<c10::BFloat16>()),                               \
          reinterpret_cast<__nv_bfloat16 *>(                                  \
              context.data_ptr<c10::BFloat16>()),                             \
          scale_value, query.stride(0), query.stride(1), query.stride(2),     \
          key.stride(0), key.stride(1), key.stride(2), value.stride(0),       \
          value.stride(1), value.stride(2), context.stride(0),                \
          context.stride(1), context.stride(2));                              \
  C10_CUDA_KERNEL_LAUNCH_CHECK()

  LAUNCH_CASE13_PREFIX(0, 256);
  LAUNCH_CASE13_PREFIX(256, 512);
  LAUNCH_CASE13_PREFIX(512, 768);
  LAUNCH_CASE13_PREFIX(768, 1024);
#undef LAUNCH_CASE13_PREFIX
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("case13_exact_attention", &case13_exact_attention,
             "Exact Case-13 shared-CTA causal attention");
}
