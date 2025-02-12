#pragma once
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <float.h>

template <typename T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}

// random utils

/**
 * Returns a random array of floats of size N, between [-1, 1]
 */
float *make_random_float(size_t N) {
  float *arr = (float *)malloc(N * sizeof(float));
  for (size_t i = 0; i < N; i++) {
    arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0;
  }
  return arr;
}

/**
 * Returns a random array of N ints in range [0, MAX).
 */
int *make_random_int(size_t N, int MAX) {
  int *arr = (int *)malloc(N * sizeof(int));
  for (size_t i = 0; i < N; i++) {
    arr[i] = rand() % MAX;
  }
  return arr;
}

// reduced/mixed precision utilities
#if defined(ENABLE_BF16)
typedef __nv_bfloat16 floatX;
typedef __nv_bfloat16 floatN;
#elif defined(ENABLE_FP16)
typedef half floatX;
typedef half floatN;
#else
typedef float floatX;
typedef float floatN;
#endif

/**
 * Packed128 data structure, which forces the compiler to use 128-bit loads/stores in GPUs
 * that support the LDG.128 and STS.128 instructions. This is similar to float4 used in
 * 32-bit floats, but supports arbitrary precision.
 */
template <typename T>
struct alignas(16) Packed128 {
  // e.g. sizeof(float4) is 16 (4x4 bytes) / sizeof(bfloat16) = 2, size = 8
  // Therefore, if T is bfloat16 type, 8 elements are stored.
  static constexpr int size = sizeof(int4) / sizeof(T);
  T payload[size];

  Packed128() = default;

  __device__ explicit Packed128(int4 bits) {
    static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
    memcpy(&payload, &bits, sizeof(bits));
  }

  // Some common fill utils
  __device__ static Packed128 constant(T value) {
    Packed128 result;
    for (int k = 0; k < size; k++) {
      result.payload[k] = value;
    }
    return result;
  }

  __device__ static Packed128 zeros() { return constant(0); }

  __device__ static Packed128 ones() { return constant(1); }

  // Operator overloads
  __device__ T &operator[](int index) {
    // returns reference to the index, allowing modifications
    return payload[index];
  }

  __device__ const T &operator[](int index) const {
    // returns const reference to the index, i.e, not allowing modifications.
    // second `const` means the function does not modify class state. It is mandatory.
    return payload[index];
  }

  __device__ int4 get_bits() const {
    int4 bits;
    static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
    memcpy(&bits, &payload, sizeof(bits));
    return bits;
  }
};

typedef Packed128<floatX> x128;

// Load a Packed128 from an aligned memory address
template <typename T>
__device__ Packed128<T> load128(const T *address) {
  return Packed128<T>{*reinterpret_cast<const int4 *>(address)};
}

// Load a Packed128 from an aligned memory address with streaming cache hint
template <typename T>
__device__ Packed128<T> load128cs(const T *address) {
  return Packed128<T>{__ldcs(reinterpret_cast<const int4*>(address))};
}

// Store a Packed128 to an aligned memory address
template <typename T>
__device__ void store128(T* target, Packed128<T> value) {
  *reinterpret_cast<int4*>(target) = value.get_bits();
}

// Store a Packed128 to an aligned memory address with streaming cache hint
template <typename T>
__device__ void store128cs(T* target, Packed128<T> value) {
  __stcs(reinterpret_cast<int4*>(target), value.get_bits());
}
/**
 * Copy from host to device with DType conversion.
 */
template <class TargetType>
cudaError_t memcpy_convert(TargetType *d_ptr, float *h_ptr, size_t count) {
  TargetType *converted = (TargetType *)malloc(count * sizeof(TargetType));
  for (int i = 0; i < count; i++) {
    converted[i] = (TargetType)h_ptr[i];
  }

  cudaError_t status = cudaMemcpy(d_ptr, converted, count * sizeof(TargetType), cudaMemcpyHostToDevice);
  free(converted);
  return status;
}

// Error check utils

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR]: %s:%s:%d\n", cudaGetErrorString(error), file, line);
    exit(EXIT_FAILURE);
  }
}
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

void cublasCheck(cublasStatus_t status, const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("[cuBLAS ERROR]: %d:%s:%d\n", status, file, line);
    exit(EXIT_FAILURE);
  }
}
#define cublasCheck(status) (cublasCheck(status, __FILE__, __LINE__));

// testing utils
template <typename D, typename T>
void validate_result(D *d_out, T *out, const char *name, size_t N, T tol) {
  // Copy data back to host.
  D *out_gpu = (D *)malloc(sizeof(D) * N);
  cudaCheck(cudaMemcpy(out_gpu, d_out, sizeof(D) * N, cudaMemcpyDeviceToHost));

#ifdef ENABLE_BF16
  float epsilon = 0.079;
#else
  float epsilon = FLT_EPSILON;
#endif

  // Check each element.
  for (int i = 0; i < N; i++) {
    if (i < 5) {
      printf("%f %f (delta: %f)\n", out[i], (T)out_gpu[i], out[i] - (T)out_gpu[i]);
    }
    // Machine epsilon scales with the absolute value of the number, because
    // bits available for exponent and mantissa are fixed.
    float t_eff = tol + epsilon * fabs(out[i]);
    if (fabs((float)out_gpu[i] - out[i]) > t_eff) {
      printf("Mismatch of `%s` at flat n=%d. Expected %f, got %f\n", name, i, out[i], (float)out_gpu[i]);
      free(out_gpu);
      exit(EXIT_FAILURE);
    }
  }
  free(out_gpu);
}

template <typename Kernel, typename... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs &&...kernel_args) {
  cudaEvent_t start, stop;
  // Prepare a buffer to scrub L2 cache between benchmarks by zeroing
  // a large dummy array.
  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));
  cudaDeviceProp deviceProp;
  cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
  void *flush_buffer;
  cudaCheck(cudaMalloc((void **)&flush_buffer, deviceProp.l2CacheSize));

  cudaCheck(cudaEventCreate(&start));
  cudaCheck(cudaEventCreate(&stop));
  float elapsed_time = 0.0f;
  for (int i = 0; i < repeats; i++) {
    // Clear L2
    cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
    // Start recording the timing of the kernel.
    cudaCheck(cudaEventRecord(start, nullptr));
    kernel(std::forward<KernelArgs>(kernel_args)...);
    cudaCheck(cudaEventRecord(stop, nullptr));
    cudaCheck(cudaEventSynchronize(start));
    cudaCheck(cudaEventSynchronize(stop));
    float single_call;
    cudaCheck(cudaEventElapsedTime(&single_call, start, stop));
    elapsed_time += single_call;
  }
  cudaCheck(cudaFree(flush_buffer));
  return elapsed_time / repeats;
}

static size_t cublaslt_workspace_size = 32 * 1024 * 1024;
static void *cublaslt_workspace = NULL;
static cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
int cuda_arch_major = 0;
int cuda_arch_minor = 0;
int cuda_num_SMs = 0;
int cuda_threads_per_SM = 0;

void setup_main() {
  srand(42);

  // Set up the device
  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, deviceIdx);
  cuda_num_SMs = deviceProp.multiProcessorCount;
  cuda_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
  cuda_arch_major = deviceProp.major;
  cuda_arch_minor = deviceProp.minor;

  // Setup cuBLAS and cuBLASLt
  cublasCheck(cublasCreate(&cublas_handle));
  cublasCheck(cublasLtCreate(&cublaslt_handle));
  cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

  // Enable TF32 if available.
  int enable_tf32 = (cuda_arch_major >= 8) ? 1 : 0;
  cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
  cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
  cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
}