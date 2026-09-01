#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/BFloat16.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kBatch = 10000;
constexpr int kSequence = 128;
constexpr int kWidth = 128;
constexpr int kRowsPerBlock = 4;
constexpr int kWarpSize = 32;
constexpr int kVecSize = 4;
constexpr int64_t kRowCount =
    static_cast<int64_t>(kBatch) * static_cast<int64_t>(kSequence);

// This is the pinned PyTorch 2.13 aligned-vector representation. Keeping the
// vector width fixed at four is part of the numerical schedule, not just a
// memory optimization.
template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

struct WelfordDataLN {
  float mean;
  float sigma2;
  float count;

  __device__ WelfordDataLN() : mean(0.0f), sigma2(0.0f), count(0.0f) {}
  __device__ WelfordDataLN(float mean_, float sigma2_, float count_)
      : mean(mean_), sigma2(sigma2_), count(count_) {}
};

// Copied without algebraic simplification from PyTorch
// cf30153c:aten/src/ATen/native/cuda/layer_norm_kernel.cu.
__device__ __forceinline__ WelfordDataLN cuWelfordOnlineSum(
    const float val, const WelfordDataLN& curr_sum) {
  float delta = val - curr_sum.mean;
  float new_count = curr_sum.count + 1.0f;
  auto fn_rcp_mul = [](auto a, auto b) { return a * (1.0f / b); };
  float new_mean = curr_sum.mean + fn_rcp_mul(delta, new_count);
  return {new_mean,
          curr_sum.sigma2 + delta * (val - new_mean),
          new_count};
}

// Argument order and expression order match the pinned native source. Empty
// native warps are intentionally not approximated by a different reduction.
__device__ __forceinline__ WelfordDataLN cuWelfordCombine(
    const WelfordDataLN dataB, const WelfordDataLN dataA) {
  using U = decltype(dataB.count);
  U delta = dataB.mean - dataA.mean;
  U count = dataA.count + dataB.count;
  U mean;
  U sigma2;
  if (count > decltype(dataB.count){0}) {
    auto fn_rcp = [](auto a) { return 1.0f / a; };
    auto coef = fn_rcp(count);
    auto nA = dataA.count * coef;
    auto nB = dataB.count * coef;
    mean = nA * dataA.mean + nB * dataB.mean;
    sigma2 = dataA.sigma2 + dataB.sigma2 +
        delta * delta * dataA.count * nB;
  } else {
    mean = U(0);
    sigma2 = U(0);
  }
  return {mean, sigma2, count};
}

__global__ void case6_exact_layer_norm_kernel(
    const c10::BFloat16* __restrict__ input,
    const c10::BFloat16* __restrict__ gamma,
    const c10::BFloat16* __restrict__ beta,
    float eps,
    c10::BFloat16* __restrict__ output,
    float* __restrict__ mean,
    float* __restrict__ rstd) {
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + warp;
  if (row >= kRowCount) {
    return;
  }

  using vec_t = aligned_vector<c10::BFloat16, kVecSize>;
  const vec_t* input_vec =
      reinterpret_cast<const vec_t*>(input + row * kWidth);

  // Native N=128 maps exactly one consecutive vector-of-four to each lane of
  // its sole nonempty warp. Preserve those four online updates in order.
  vec_t data = input_vec[lane];
  WelfordDataLN wd(0.0f, 0.0f, 0.0f);
#pragma unroll
  for (int ii = 0; ii < kVecSize; ++ii) {
    wd = cuWelfordOnlineSum(static_cast<float>(data.val[ii]), wd);
  }

  // Match native compute_stats: descending shuffle-down passes the running
  // state as dataB and the shuffled-down state as dataA.
#pragma unroll
  for (int offset = (kWarpSize >> 1); offset > 0; offset >>= 1) {
    WelfordDataLN wdB{WARP_SHFL_DOWN(wd.mean, offset),
                      WARP_SHFL_DOWN(wd.sigma2, offset),
                      WARP_SHFL_DOWN(wd.count, offset)};
    wd = cuWelfordCombine(wd, wdB);
  }

  wd.mean = WARP_SHFL(wd.mean, 0);
  wd.sigma2 = WARP_SHFL(wd.sigma2, 0) / float(kWidth);
  float rstd_val = c10::cuda::compat::rsqrt(wd.sigma2 + eps);

  // Native reloads the row for affine output after statistics. Keep that load
  // and the exact gamma * (rstd * (x - mean)) + beta expression order.
  data = input_vec[lane];
  const vec_t* gamma_vec = reinterpret_cast<const vec_t*>(gamma);
  const vec_t* beta_vec = reinterpret_cast<const vec_t*>(beta);
  vec_t* output_vec = reinterpret_cast<vec_t*>(output + row * kWidth);
  vec_t affine_weight = gamma_vec[lane];
  vec_t affine_bias = beta_vec[lane];
  vec_t out;
#pragma unroll
  for (int ii = 0; ii < kVecSize; ++ii) {
    out.val[ii] = static_cast<float>(affine_weight.val[ii]) *
            (rstd_val * (static_cast<float>(data.val[ii]) - wd.mean)) +
        static_cast<float>(affine_bias.val[ii]);
  }
  output_vec[lane] = out;

  if (lane == 0) {
    mean[row] = wd.mean;
    rstd[row] = rstd_val;
  }
}

void check_tensor(
    const torch::Tensor& tensor,
    const torch::Device& device,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.device() == device, name, " must share the input device");
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be BF16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::vector<torch::Tensor> case6_exact_layer_norm(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps) {
  TORCH_CHECK(
      input.sizes() == at::IntArrayRef({kBatch, kSequence, kWidth}),
      "exact Case-6 LayerNorm requires input [10000,128,128]");
  check_tensor(input, input.device(), "input");
  check_tensor(weight, input.device(), "weight");
  check_tensor(bias, input.device(), "bias");
  TORCH_CHECK(
      weight.sizes() == at::IntArrayRef({kWidth}),
      "exact Case-6 LayerNorm requires weight [128]");
  TORCH_CHECK(
      bias.sizes() == at::IntArrayRef({kWidth}),
      "exact Case-6 LayerNorm requires bias [128]");
  TORCH_CHECK(eps > 0.0, "exact Case-6 LayerNorm requires positive epsilon");

  const c10::cuda::CUDAGuard device_guard(input.device());
  auto output = torch::empty_like(input);
  auto stat_options = input.options().dtype(torch::kFloat32);
  auto mean = torch::empty({kRowCount}, stat_options);
  auto rstd = torch::empty({kRowCount}, stat_options);

  constexpr int kBlocks = static_cast<int>(kRowCount / kRowsPerBlock);
  const dim3 threads(kWarpSize, kRowsPerBlock, 1);
  case6_exact_layer_norm_kernel<<<
      kBlocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      input.const_data_ptr<c10::BFloat16>(),
      weight.const_data_ptr<c10::BFloat16>(),
      bias.const_data_ptr<c10::BFloat16>(),
      static_cast<float>(eps),
      output.data_ptr<c10::BFloat16>(),
      mean.data_ptr<float>(),
      rstd.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, mean, rstd};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "case6_exact_layer_norm",
      &case6_exact_layer_norm,
      "Exact four-rows-per-CTA Case-6 BF16 LayerNorm");
}
