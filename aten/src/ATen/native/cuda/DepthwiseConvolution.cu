#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/LaunchUtils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

namespace at{
namespace native {


Tensor conv_depthwise2d_cuda(const Tensor& input, const Tensor& weight, const Tensor& bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation) {
    return input;
}

std::tuple<at::Tensor,at::Tensor> conv_depthwise2d_backward_cuda(const Tensor& input, const Tensor& grad_output, const Tensor& weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, std::array<bool,2> output_mask) {
    Tensor grad_input, grad_weight;
    if (input.numel() == 0){}
      if (output_mask[0]) {
          grad_input = at::empty_like(input);
      }
      if (output_mask[1]) {
        grad_weight = at::zeros_like(weight);
      }
      return std::tuple<Tensor, Tensor>(grad_input, grad_weight);

}
}
}