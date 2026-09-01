#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kHeads = 4;
constexpr int kSequence = 128;
constexpr int kHeadDim = 32;
constexpr int kRows = 32;
constexpr int kWarps = 4;

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
load_query_fragment(const __nv_bfloat16 *query, int64_t query_base,
                    int64_t query_stride_row, int matrix_row,
                    int reduction_start, int lane, uint32_t (&a)[4]) {
  const int group = lane >> 2;
  const int thread = lane & 3;
  const int row0 = RowStart + matrix_row + group;
  const int row1 = row0 + 8;
  const int column0 = reduction_start + thread * 2;
  const int column1 = column0 + 8;
  a[0] = pack_bf16(query[query_base + row0 * query_stride_row + column0],
                   query[query_base + row0 * query_stride_row + column0 + 1]);
  a[1] = pack_bf16(query[query_base + row1 * query_stride_row + column0],
                   query[query_base + row1 * query_stride_row + column0 + 1]);
  a[2] = pack_bf16(query[query_base + row0 * query_stride_row + column1],
                   query[query_base + row0 * query_stride_row + column1 + 1]);
  a[3] = pack_bf16(query[query_base + row1 * query_stride_row + column1],
                   query[query_base + row1 * query_stride_row + column1 + 1]);
}

template <int KeyCount>
__device__ __forceinline__ void
load_key_fragment(const __nv_bfloat16 *key, int64_t key_base,
                  int64_t key_stride_row, int matrix_column,
                  int reduction_start, int lane, uint32_t (&b)[2]) {
  const int group = lane >> 2;
  const int thread = lane & 3;
  const int key_row = matrix_column + group;
  const int reduction0 = reduction_start + thread * 2;
  const int reduction1 = reduction0 + 8;
  const bool valid = key_row < KeyCount;
  const __nv_bfloat16 zero = __float2bfloat16_rn(0.0f);
  b[0] =
      valid
          ? pack_bf16(key[key_base + key_row * key_stride_row + reduction0],
                      key[key_base + key_row * key_stride_row + reduction0 + 1])
          : pack_bf16(zero, zero);
  b[1] =
      valid
          ? pack_bf16(key[key_base + key_row * key_stride_row + reduction1],
                      key[key_base + key_row * key_stride_row + reduction1 + 1])
          : pack_bf16(zero, zero);
}

__device__ __forceinline__ void
store_score_fragment(__nv_bfloat16 *scores, const float (&acc)[4],
                     int matrix_row, int matrix_column, int row_start,
                     int key_count, float scale, int lane) {
  const int group = lane >> 2;
  const int thread = lane & 3;
  const int rows[4] = {matrix_row + group, matrix_row + group,
                       matrix_row + group + 8, matrix_row + group + 8};
  const int columns[4] = {
      matrix_column + thread * 2, matrix_column + thread * 2 + 1,
      matrix_column + thread * 2, matrix_column + thread * 2 + 1};
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    __nv_bfloat16 rounded_dot = __float2bfloat16_rn(acc[index]);
    __nv_bfloat16 rounded_scaled =
        __float2bfloat16_rn(__bfloat162float(rounded_dot) * scale);
    const bool valid =
        columns[index] < key_count && columns[index] <= row_start + rows[index];
    scores[rows[index] * kSequence + columns[index]] =
        valid ? rounded_scaled
              : __float2bfloat16_rn(-std::numeric_limits<float>::infinity());
  }
}

template <int KeyCount, int PaddedKeyCount>
__device__ __forceinline__ void native_softmax_rows(__nv_bfloat16 *scores,
                                                    int warp, int lane) {
  constexpr int kIterations = PaddedKeyCount / 32;
#pragma unroll
  for (int wave = 0; wave < kRows / kWarps; ++wave) {
    const int row = warp + wave * kWarps;
    float elements[kIterations];
#pragma unroll
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      const int column = lane + iteration * 32;
      elements[iteration] =
          column < KeyCount ? __bfloat162float(scores[row * kSequence + column])
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
      scores[row * kSequence + column] =
          column < KeyCount ? __float2bfloat16_rn(elements[iteration] / sum)
                            : __float2bfloat16_rn(0.0f);
    }
  }
}

__device__ __forceinline__ void
load_probability_fragment(const __nv_bfloat16 *probabilities, int matrix_row,
                          int reduction_start, int lane, uint32_t (&a)[4]) {
  const int group = lane >> 2;
  const int thread = lane & 3;
  const int row0 = matrix_row + group;
  const int row1 = row0 + 8;
  const int column0 = reduction_start + thread * 2;
  const int column1 = column0 + 8;
  a[0] = pack_bf16(probabilities[row0 * kSequence + column0],
                   probabilities[row0 * kSequence + column0 + 1]);
  a[1] = pack_bf16(probabilities[row1 * kSequence + column0],
                   probabilities[row1 * kSequence + column0 + 1]);
  a[2] = pack_bf16(probabilities[row0 * kSequence + column1],
                   probabilities[row0 * kSequence + column1 + 1]);
  a[3] = pack_bf16(probabilities[row1 * kSequence + column1],
                   probabilities[row1 * kSequence + column1 + 1]);
}

template <int KeyCount>
__device__ __forceinline__ void
load_value_fragment(const __nv_bfloat16 *value, int64_t value_base,
                    int64_t value_stride_row, int matrix_column,
                    int reduction_start, int lane, uint32_t (&b)[2]) {
  const int group = lane >> 2;
  const int thread = lane & 3;
  const int output_column = matrix_column + group;
  const int key0 = reduction_start + thread * 2;
  const int key1 = key0 + 8;
  const __nv_bfloat16 zero = __float2bfloat16_rn(0.0f);
  b[0] = key0 < KeyCount
             ? pack_bf16(
                   value[value_base + key0 * value_stride_row + output_column],
                   value[value_base + (key0 + 1) * value_stride_row +
                         output_column])
             : pack_bf16(zero, zero);
  b[1] = key1 < KeyCount
             ? pack_bf16(
                   value[value_base + key1 * value_stride_row + output_column],
                   value[value_base + (key1 + 1) * value_stride_row +
                         output_column])
             : pack_bf16(zero, zero);
}

__device__ __forceinline__ void
store_context_fragment(__nv_bfloat16 *context, int64_t context_base,
                       int64_t context_stride_row, const float (&acc)[4],
                       int matrix_row, int matrix_column, int row_start,
                       int lane) {
  const int group = lane >> 2;
  const int thread = lane & 3;
  const int rows[4] = {matrix_row + group, matrix_row + group,
                       matrix_row + group + 8, matrix_row + group + 8};
  const int columns[4] = {
      matrix_column + thread * 2, matrix_column + thread * 2 + 1,
      matrix_column + thread * 2, matrix_column + thread * 2 + 1};
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    context[context_base + (row_start + rows[index]) * context_stride_row +
            columns[index]] = __float2bfloat16_rn(acc[index]);
  }
}

template <int RowStart, int KeyCount, int PaddedKeyCount>
__global__ void case6_exact_attention_kernel(
    const __nv_bfloat16 *query, const __nv_bfloat16 *key,
    const __nv_bfloat16 *value, __nv_bfloat16 *context, float scale,
    int64_t query_stride_batch, int64_t query_stride_head,
    int64_t query_stride_row, int64_t key_stride_batch, int64_t key_stride_head,
    int64_t key_stride_row, int64_t value_stride_batch,
    int64_t value_stride_head, int64_t value_stride_row,
    int64_t context_stride_batch, int64_t context_stride_head,
    int64_t context_stride_row) {
  __shared__ __nv_bfloat16 scores[kRows * kSequence];
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int batch_head = blockIdx.x;
  const int batch = batch_head / kHeads;
  const int head = batch_head % kHeads;
  const int64_t query_base =
      batch * query_stride_batch + head * query_stride_head;
  const int64_t key_base = batch * key_stride_batch + head * key_stride_head;
  const int64_t value_base =
      batch * value_stride_batch + head * value_stride_head;
  const int64_t context_base =
      batch * context_stride_batch + head * context_stride_head;

  constexpr int kNTiles = PaddedKeyCount / 8;
  constexpr int kFragmentsPerWarp = (2 * kNTiles) / kWarps;
  constexpr int kFragmentBaseStep = kFragmentsPerWarp;
  float score_acc[kFragmentsPerWarp][4];
#pragma unroll
  for (int fragment = 0; fragment < kFragmentsPerWarp; ++fragment) {
#pragma unroll
    for (int output = 0; output < 4; ++output) {
      score_acc[fragment][output] = 0.0f;
    }
  }

  const int fragment_base = warp * kFragmentBaseStep;
  const int matrix_row = (fragment_base / kNTiles) * 16;
#pragma unroll
  for (int reduction_start = 0; reduction_start < kHeadDim;
       reduction_start += 16) {
    uint32_t a[4];
    load_query_fragment<RowStart>(query, query_base, query_stride_row,
                                  matrix_row, reduction_start, lane, a);
#pragma unroll
    for (int fragment = 0; fragment < kFragmentsPerWarp; ++fragment) {
      const int fragment_index = fragment_base + fragment;
      const int matrix_column = (fragment_index % kNTiles) * 8;
      uint32_t b[2];
      load_key_fragment<KeyCount>(key, key_base, key_stride_row, matrix_column,
                                  reduction_start, lane, b);
      mma_m16n8k16(score_acc[fragment], a, b);
    }
  }

#pragma unroll
  for (int fragment = 0; fragment < kFragmentsPerWarp; ++fragment) {
    const int fragment_index = fragment_base + fragment;
    const int fragment_row = (fragment_index / kNTiles) * 16;
    const int fragment_column = (fragment_index % kNTiles) * 8;
    store_score_fragment(scores, score_acc[fragment], fragment_row,
                         fragment_column, RowStart, KeyCount, scale, lane);
  }
  __syncthreads();

  native_softmax_rows<KeyCount, PaddedKeyCount>(scores, warp, lane);
  __syncthreads();

  constexpr int kPVFragmentsPerWarp = 2;
  constexpr int kPVNTiles = kHeadDim / 8;
  float context_acc[kPVFragmentsPerWarp][4];
#pragma unroll
  for (int fragment = 0; fragment < kPVFragmentsPerWarp; ++fragment) {
#pragma unroll
    for (int output = 0; output < 4; ++output) {
      context_acc[fragment][output] = 0.0f;
    }
  }
  const int pv_fragment_base = warp * kPVFragmentsPerWarp;
  const int pv_matrix_row = (pv_fragment_base / kPVNTiles) * 16;
#pragma unroll
  for (int reduction_start = 0; reduction_start < PaddedKeyCount;
       reduction_start += 16) {
    uint32_t a[4];
    load_probability_fragment(scores, pv_matrix_row, reduction_start, lane, a);
#pragma unroll
    for (int fragment = 0; fragment < kPVFragmentsPerWarp; ++fragment) {
      const int fragment_index = pv_fragment_base + fragment;
      const int matrix_column = (fragment_index % kPVNTiles) * 8;
      uint32_t b[2];
      load_value_fragment<KeyCount>(value, value_base, value_stride_row,
                                    matrix_column, reduction_start, lane, b);
      mma_m16n8k16(context_acc[fragment], a, b);
    }
  }

#pragma unroll
  for (int fragment = 0; fragment < kPVFragmentsPerWarp; ++fragment) {
    const int fragment_index = pv_fragment_base + fragment;
    const int fragment_row = (fragment_index / kPVNTiles) * 16;
    const int fragment_column = (fragment_index % kPVNTiles) * 8;
    store_context_fragment(context, context_base, context_stride_row,
                           context_acc[fragment], fragment_row, fragment_column,
                           RowStart, lane);
  }
}

void check_tensor(const torch::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be BF16");
  TORCH_CHECK(tensor.dim() == 4, name, " must have rank four");
  TORCH_CHECK(tensor.size(1) == 4 && tensor.size(2) == 128 &&
                  tensor.size(3) == 32,
              name, " must have exact Case-6 H/S/HD");
  TORCH_CHECK(tensor.stride(3) == 1, name, " must have unit-stride HD");
}

void case6_exact_attention(torch::Tensor query, torch::Tensor key,
                           torch::Tensor value, torch::Tensor context,
                           double scale) {
  check_tensor(query, "query");
  check_tensor(key, "key");
  check_tensor(value, "value");
  check_tensor(context, "context");
  TORCH_CHECK(query.size(0) > 0 && query.size(0) <= 512,
              "Case-6 batch slice must be in [1, 512]");
  TORCH_CHECK(key.sizes() == query.sizes() && value.sizes() == query.sizes() &&
                  context.sizes() == query.sizes(),
              "Case-6 Q/K/V/context shapes must match");
  TORCH_CHECK(key.device() == query.device() &&
                  value.device() == query.device() &&
                  context.device() == query.device(),
              "Case-6 Q/K/V/context devices must match");

  const c10::cuda::CUDAGuard device_guard(query.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const dim3 grid(query.size(0) * kHeads);
  const dim3 block(32, kWarps);
  const float scale_value = static_cast<float>(scale);

#define LAUNCH_CASE6_PREFIX(ROW_START, KEY_COUNT, PADDED_KEY_COUNT)            \
  case6_exact_attention_kernel<ROW_START, KEY_COUNT, PADDED_KEY_COUNT>         \
      <<<grid, block, 0, stream>>>(                                            \
          reinterpret_cast<const __nv_bfloat16 *>(                             \
              query.data_ptr<c10::BFloat16>()),                                \
          reinterpret_cast<const __nv_bfloat16 *>(                             \
              key.data_ptr<c10::BFloat16>()),                                  \
          reinterpret_cast<const __nv_bfloat16 *>(                             \
              value.data_ptr<c10::BFloat16>()),                                \
          reinterpret_cast<__nv_bfloat16 *>(                                   \
              context.data_ptr<c10::BFloat16>()),                              \
          scale_value, query.stride(0), query.stride(1), query.stride(2),      \
          key.stride(0), key.stride(1), key.stride(2), value.stride(0),        \
          value.stride(1), value.stride(2), context.stride(0),                 \
          context.stride(1), context.stride(2));                               \
  C10_CUDA_KERNEL_LAUNCH_CHECK()

  LAUNCH_CASE6_PREFIX(0, 32, 32);
  LAUNCH_CASE6_PREFIX(32, 64, 64);
  LAUNCH_CASE6_PREFIX(64, 96, 128);
  LAUNCH_CASE6_PREFIX(96, 128, 128);
#undef LAUNCH_CASE6_PREFIX
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("case6_exact_attention", &case6_exact_attention,
             "Exact Case-6 shared-CTA causal attention");
}
