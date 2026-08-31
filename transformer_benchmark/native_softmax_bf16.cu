#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <ATen/native/cuda/PersistentSoftmax.cuh>

#include <limits>

namespace {

void check_input(const torch::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "native BF16 softmax requires a CUDA tensor");
  TORCH_CHECK(
      input.is_contiguous(), "native BF16 softmax requires contiguous input");
  TORCH_CHECK(input.dim() == 4, "native BF16 softmax requires rank four");
  const bool is_case6_prefix =
      input.scalar_type() == at::kFloat && input.size(0) > 0 &&
      input.size(0) <= 512 && input.size(1) == 4 && input.size(2) == 32 &&
      (input.size(3) == 32 || input.size(3) == 64 || input.size(3) == 96 ||
       input.size(3) == 128);
  const bool is_case13_prefix =
      input.scalar_type() == at::kHalf && input.size(0) == 64 &&
      input.size(1) == 4 && input.size(2) == 256 &&
      (input.size(3) == 256 || input.size(3) == 512 ||
       input.size(3) == 768 || input.size(3) == 1024);
  TORCH_CHECK(
      is_case6_prefix || is_case13_prefix,
      "native BF16 softmax is restricted to exact Case-6/13 prefixes");
  TORCH_CHECK(
      input.numel() / input.size(-1) <= std::numeric_limits<int>::max(),
      "native BF16 softmax batch count exceeds int32");
}

}  // namespace

torch::Tensor native_softmax_bf16(torch::Tensor input) {
  check_input(input);
  const c10::cuda::CUDAGuard device_guard(input.device());
  auto output = torch::empty(
      input.sizes(), input.options().dtype(torch::kBFloat16));
  const int element_count = static_cast<int>(input.size(-1));
  const int batch_count = static_cast<int>(input.numel() / element_count);

  if (input.scalar_type() == at::kFloat) {
    dispatch_softmax_forward<float, c10::BFloat16, float, false, false>(
        output.data_ptr<c10::BFloat16>(),
        input.data_ptr<float>(),
        element_count,
        element_count,
        batch_count);
  } else {
    dispatch_softmax_forward<c10::Half, c10::BFloat16, float, false, false>(
        output.data_ptr<c10::BFloat16>(),
        input.data_ptr<c10::Half>(),
        element_count,
        element_count,
        batch_count);
  }
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "native_softmax_bf16",
      &native_softmax_bf16,
      "ATen persistent softmax with a fused BF16 output cast");
}
