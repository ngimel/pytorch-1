#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/BinaryOps.h>

// TODO: update to use lazynvrtc
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <torch/csrc/jit/resource_guard.h>
#include <sstream>
#include <chrono>
#include <torch/csrc/jit/frontend/code_template.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

template<typename scalar_t>
struct AddFunctor {
  AddFunctor(scalar_t a): alpha(a) {}
  __device__ __forceinline__ scalar_t operator() (const scalar_t a, const scalar_t b) const {
    return a + alpha * b;
  }
  private:
    scalar_t alpha;
};

void add_kernel_cuda(TensorIterator& iter, Scalar alpha_scalar) {
  // create template here
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kHalf, kBool, kBFloat16, iter.common_dtype(), "add_cuda/sub_cuda", [&]() {
    // NOTE: we don't need compile-time switching this does at all, so maybe use alternative?
    // Question: is instantiating worthwhile vs. just recompiling?
      // Cons of recompilation: string manipulation done every time
      // Cons of recompilation: need your own code template
      // Cons of instantiation: complicated
    // instantiate dispatched scalar types using the template here
    // this happens at runtime before the call
    // cache whether instantiated or not
    // call
    AddFunctor<scalar_t> f(alpha_scalar.to<scalar_t>());
    gpu_kernel_with_scalars(iter, f);
  });
}

static void sub_kernel_cuda(TensorIterator& iter, Scalar alpha_scalar) {
  add_kernel_cuda(iter, -alpha_scalar);
}

REGISTER_DISPATCH(add_stub, &add_kernel_cuda);
REGISTER_DISPATCH(sub_stub, &sub_kernel_cuda);

// TODO: update this
static void getMajorMinor(
    const cudaDeviceProp* const prop,
    int& major,
    int& minor) {
  int nvrtc_major, nvrtc_minor;
  AT_CUDA_NVRTC_CHECK(at::globalContext().getNVRTC().nvrtcVersion(&nvrtc_major, &nvrtc_minor));

  // Short-circuits if NVRTC version too low
  AT_ASSERT(nvrtc_major >= 6);

  // Major and minor is determined by device properties and
  // possibly "downcompiled" to a lower (compatible) compute architecture
  // based on the NVRTC version
  major = prop->major;
  minor = prop->minor;
  if (nvrtc_major <= 7 && prop->major > 5) { // 7 supports 2-5.x
    major = 5;
    minor = 0;
  } else if (nvrtc_major <= 8 && prop->major > 6) { // 8 supports 2-6.x
    major = 6;
    minor = 0;
  } else if (nvrtc_major <= 9 && prop->major >= 7) { // 9 supports 3-7.2
    major = 7;
    if (prop->major == 7 && prop->minor <= 2)
      minor = prop->minor;
    else
      minor = 0;
  } else if (nvrtc_major <= 10 && prop->major >= 7) { // 10 supports 3-7.5
    major = 7;
    if (prop->major == 7 && prop->minor <= 5) {
      minor = prop->minor;
    } else {
      minor = 0;
    }
  }
}

namespace {

  // TODO: update with IntDivider
  // TODO: possibly size specialize with template
struct TensorAccessor {
  TensorAccessor(
      const IntArrayRef shape,
      const IntArrayRef strides,
      const int64_t _element_size)
      : element_size_{_element_size}, ndims_(shape.size()) {

    std::copy(shape.cbegin(), shape.cend(), std::begin(sizes_));
    std::copy(strides.cbegin(), strides.cend(), std::begin(strides_));
  }

  C10_HOST_DEVICE int64_t index_to_offset(int32_t idx) const {
    int64_t offset = 0;

    #pragma unroll
    for (int32_t dim = 0; dim < 25; ++dim) {
      if (dim == ndims_) {
        break;
      }

      const auto quot = sizes_[dim] / idx;
      const auto rem = sizes_[dim] % idx;

      idx = quot;
      offset += rem * strides_[dim];
    }

    return offset;
  }

  int64_t element_size_;
  int32_t ndims_;
  int32_t sizes_[25];
  int64_t strides_[25];
};

static auto cuda_template = torch::jit::CodeTemplate(R"(
  struct TensorAccessor {
    TensorAccessor() = default;

    // TODO: add a real function here
    __host__ __device__ long index_to_offset(int idx) const {
      return idx;
    }

    long element_size_;
    int ndims_;
    int sizes_[25];
    long strides_[25];
  };

  ${function}

  extern "C" __global__
  void foo_kernel(
      long numel,
      TensorAccessor* out_accessor,
      TensorAccessor* a_accessor,
      TensorAccessor* b_accessor,
      float* out,
      float* a,
      float* b) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      for (int i = 0; i < 4; ++i) {
        int out_offset = out_accessor->index_to_offset(i);
        int a_offset = a_accessor->index_to_offset(i);
        int b_offset = b_accessor->index_to_offset(i);
        // TODO: allow for custom names
        out[out_offset] = foo<${dtype}>(a[a_offset], b[b_offset]);
      }
    }
  }

  #define NUM_THREADS (C10_WARP_SIZE * 2)
  #define THREAD_WORK_SIZE 4
  #define BLOCK_WORK_SIZE (THREAD_WORK_SIZE * num_threads)

  // extern "C" __global__
  // void vectorized_elementwise_kernel(int numel) {
  //   const int remaining = numel - BLOCK_WORK_SIZE * blockIdx.x;

  //   if (remaining < BLOCK_WORK_SIZE) {

  //   } else {
  //     int idx = blockIdx.x;
  //     using vec_t = aligned_vector<scalar_t, vec_size>;
  //     vec_t *from_ = reinterpret_cast<vec_t *>(from);
  //     int thread_idx = threadIdx.x;
  //     #pragma unroll
  //     for (int i = 0; i < loop_size; i++) {
  //       int index = thread_idx + i * num_threads;
  //       vec_t v = from_[index];
  //       #pragma unroll
  //       for (int j = 0; j < vec_size; j++) {
  //         to(vec_size * i + j) = v.val[j];
  //       }
  //     }
  //   }
  // }


// instantiations here
)");

static const std::unordered_map<ScalarType, std::string> dtype_map {{kFloat,"float"}, {kDouble, "double"}};
static const std::unordered_map<ScalarType, int> dtype_to_int {{kFloat,0}, {kDouble, 1}};

char inline compute_key(const ScalarType dtype, const bool contiguous, const bool no_dynamic_casting) {
  int key = 0;
  key += no_dynamic_casting ? 1 : 0;
  key += contiguous ? 2 : 0;
  int dtype_int;
  auto dtype_int_iter = dtype_to_int.find(dtype);
  if (dtype_int_iter != dtype_to_int.end()){
     dtype_int = dtype_int_iter->second;
  } else {
    TORCH_CHECK(false, "unexpected dype");
  }
  key += 4*dtype_int;
  return key;
}

} // anonymous namespace

Tensor foo_cuda(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);

  std::cout << "common_dtype: " << iter.common_dtype() << std::endl;

  // launch_vectorized_kernel path
  int64_t numel = iter.numel();
  int64_t grid = (numel + block_work_size - 1) / block_work_size;

  const auto ntensors = iter.ntensors();
  // at::detail::Array<char*, ntensors> data;
  // for (auto i = decltype(ntensors){0}; i < ntensors; i++) {
  //   data[i] = (char*)iter.data_ptr(i);
  // }
  // TODO: revise vectorize functions (see MemoryAccess.cuh) to work at runtime
  //   without array allocation
  // int32_t vec_size = memory::can_vectorize_up_to<func_t>(data);
  // TODO: for now assume in case 4
  auto stream = at::cuda::getCurrentCUDAStream();

  // vectorized_elementwise_kernel switch here


  std::vector<void*> args;
  args.push_back((void*)&numel);
  std::vector<TensorAccessor> accessors;
  for (auto i = decltype(iter.ntensors()){0}; i < iter.ntensors(); ++i) {
    accessors.emplace_back(iter.shape(), iter.strides(i), iter.element_size(i));
  }

  for (const auto& accessor : accessors) {
    args.push_back((void*)&accessor);
  }


  std::cout << "iter.ntensors(): " << iter.ntensors() << std::endl;

  const auto& nvrtc = at::globalContext().getNVRTC();
  CUfunction function;
  static std::unordered_map<char, CUfunction> kernelCache;
  //TODO use needs_dynamic_casting
  bool no_dynamic_casting = iter.dtype(0) == iter.common_dtype() && iter.dtype(1) == iter.common_dtype() && iter.dtype(2) == iter.common_dtype();

  auto key = compute_key(iter.common_dtype(), iter.is_contiguous(), no_dynamic_casting);
  auto start = std::chrono::high_resolution_clock::now();
  auto found_fn = kernelCache.find(key);
  if (found_fn != kernelCache.end()) {
    function = found_fn->second;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds elapsed_seconds = end-start;
    std::cout << "lookup time " << elapsed_seconds.count() << " ns \n";
  } else {
    std::cout << "building the kernel\n";

  #define stringify(...) std::string("template <typename T> __device__ __forceinline__ " #__VA_ARGS__)
  const auto s = stringify(
    T foo(T a, T b) {
      return a + b;
    }
  );
  #undef stringify

  torch::jit::TemplateEnv env;
  env.s("function", s);
  std::string string_dtype;
  auto dtype_iter = dtype_map.find(iter.common_dtype());
  if (dtype_iter != dtype_map.end()){
     string_dtype = dtype_iter->second;
  } else {
    TORCH_CHECK(false, "unexpected dype");
  }
  env.s("dtype", string_dtype);
  std::string code = cuda_template.format(env);

  std::cout << "code: " << code << "\n";

// Acquires device and NVRTC properties (for compile arch and occupancy
// calculations)
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  int major, minor;
  getMajorMinor(prop, major, minor);

  // Creates the NVRTC program
  nvrtcProgram program;
  AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcCreateProgram(
      &program, code.c_str(), nullptr, 0, nullptr, nullptr));

  // constructs nvrtc arguments
  const std::string compute = "--gpu-architecture=compute_" +
    std::to_string(major) + std::to_string(minor);
  const std::vector<const char*> build_args = {
    "--std=c++14", compute.c_str(), "-default-device"};

  const auto compilation_result =
        nvrtc.nvrtcCompileProgram(program, build_args.size(), build_args.data());

  if (compilation_result != NVRTC_SUCCESS) {
    size_t logsize;
    AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcGetProgramLogSize(program, &logsize));
    std::vector<char> log(logsize);
    AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcGetProgramLog(program, log.data()));
    std::stringstream cu;
    cu << log.data();
    throw std::runtime_error(cu.str());
  }

  CUmodule module;

  ::torch::jit::ResourceGuard holdProgram([&] { nvrtc.nvrtcDestroyProgram(&program); });
  std::vector<char> ptx;
  size_t ptx_size;
  AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcGetPTXSize(program, &ptx_size));
  ptx.resize(ptx_size);
  AT_CUDA_NVRTC_CHECK(nvrtc.nvrtcGetPTX(program, ptx.data()));

  AT_CUDA_DRIVER_CHECK(nvrtc.cuModuleLoadData(&module, ptx.data()));
  const std::string name = "foo_kernel";
  AT_CUDA_DRIVER_CHECK(
    nvrtc.cuModuleGetFunction(&function, module, name.c_str()));
  kernelCache.insert(std::make_pair(key,function));
  }
  //int maxBlocks;
  //AT_CUDA_DRIVER_CHECK(nvrtc.cuOccupancyMaxActiveBlocksPerMultiprocessor(
  //  &maxBlocks, function, 128, 0));
  //maxBlocks *= prop->multiProcessorCount;

  // const auto nBlocks = std::min(maxBlocks_, ceilDiv(numel, kBlockSize));
  const int nBlocks = 1;

  constexpr int32_t kBlockSize = 128;

  void* out_ptr = iter.output().data_ptr();
  void* self_ptr = self.data_ptr();
  void* other_ptr = other.data_ptr();

  // args.push_back(out_ptr);
  args.push_back((void*)&out_ptr);
  args.push_back((void*)&self_ptr);
  args.push_back((void*)&other_ptr);

  // Launches kernel on current stream (device was set by executor)

  AT_CUDA_DRIVER_CHECK(nvrtc.cuLaunchKernel(
    function,
    nBlocks,
    1,
    1,
    kBlockSize,
    1,
    1,
    0,
    stream,
    args.data(),
    nullptr));

  return iter.output();

  // NOTE: may need/want to initialize CUDA context here (refactor into nvrtc call)

  // void* out, void* a, void* b
  // TODO: provide code (a std::string)
  // const std::string name{"foo_kernel"};
  // const std::string code{R"foo(
  // extern "C" __global__
  // void foo_kernel(void* out, void* a, void* b) {
  //   // if (blockIdx.x == 0 && threadIdx.x == 0) {
  //   //   printf("%f\n", a);
  //   //   printf("%i\n", b);
  //   //   printf("%f\n", ((float*)ptr)[0]);
  //   // }
  //   float* out_float = static_cast<float*>(out);
  //   float* a_float = static_cast<float*>(a);
  //   float* b_float = static_cast<float*>(b);

  //   if (blockIdx.x == 0 && threadIdx.x == 0) {
  //     *out_float = *a_float + *b_float;
  //   }
  // })foo"};
}

}} // namespace at::native
