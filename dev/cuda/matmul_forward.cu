#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "common.h"

/**
 * (B, T, C) @ (C, OC) + (OC,) = (B, T, OC)
 * (N, C) @ (C, OC) + (OC,) = (N, OC)
 */
void matmul_forward_cpu(float *out, const float *inp, const float *weight, float *bias, int B, int T, int C, int OC) {
  int N = B * T;
#pragma omp parallel for
  for (int n = 0; n < N; n++) {
    const float *x = inp + n * C;
    for (int oc = 0; oc < OC; oc++) {
      float sum = (bias == NULL) ? 0.0f : bias[oc];
      for (int c = 0; c < C; c++) {
        sum += x[c] * weight[oc * C + c];
      }
      out[n * OC + oc] = sum;
    }
  }
}

/**
 * Naive Matmul, each thread within a block is responsible for one value of the output matrix.
 * This kernel suffers from gigantic global memory accesses, no reuse between blocks.
 */
__global__ void matmul_forward_kernel1(float *out, const float *inp, const float *weight, const float *bias, int BT,
                                       int C, int OC) {
  int bt = blockIdx.x * blockDim.x + threadIdx.x;
  int oc = blockIdx.y * blockDim.y + threadIdx.y;
  if (bt < BT && oc < OC) {
    float sum = (bias == NULL) ? 0.0f : bias[oc];
    for (int c = 0; c < C; c++) {
      sum += inp[bt * C + c] * weight[oc * C + c];
    }
    out[bt * OC + oc] = sum;
  }
}

// GPU kernels
void matmul_forward1(float *out, const float *inp, const float *weight, const float *bias, int B, int T, int C, int OC,
                     int sqrt_block_size) {
  dim3 gridDim(ceil_div(B * T, sqrt_block_size), ceil_div(OC, sqrt_block_size));
  dim3 blockDim(sqrt_block_size, sqrt_block_size);
  matmul_forward_kernel1<<<gridDim, blockDim>>>(out, inp, weight, bias, B * T, C, OC);
  cudaCheck(cudaGetLastError());
}

/**
 * Matrix Vector add using grid-stride looping.
 */
__global__ void add_bias(float *out, const float *bias, int B, int T, int OC) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < B * T * OC; i += stride) {
    out[i] += bias[tid % OC];
  }
}

// Kernel 2 calls cuBLAS.
void matmul_forward2(float *out, const float *inp, const float *weight, const float *bias, int B, int T, int C, int OC,
                     int sqrt_block_size) {
  /**
   * Reference API:
   * cublasStatus_t cublasSgemm(cublasHandle_t handle,
   *                            cublasOperation_t transa, cublasOperation_t transb,
   *                            int m, int n, int k,
   *                            const float *alpha,
   *                            const float *A, int lda,
   *                            const float *B, int ldb,
   *                            const float *beta,
   *                            float *C, int ldc)
   *
   * inp = (B*T, C)
   * weight = (OC, C)
   * out = (B*T, OC)
   *
   * cuBLAS does C = alpha * A * B + beta * C; where A is mxk, B is kxn, C is mxn.
   * In our current storage format, we would do: out = inp @ weight.T
   * Since cuBLAS uses column-major format, we want out.T instead, as it translates to
   * out.T = weight @ inp.T; After the operation, out.T in column-major is the same as
   * out in row-major.
   * Since `weight` in the translated equation is column major, it means we must tell
   * cuBLAS to read it as transposed. So weight -> CUBLAS_OP_T
   * And because inp.T in cuBLAS means it is inp as row major, inp -> CUBLAS_OP_N
   */
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasCheck(
      cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B * T, C, &alpha, weight, C, inp, C, &beta, out, OC));
  if (bias != NULL) {
    int block_size = sqrt_block_size * sqrt_block_size;
    int grid_size = ceil_div(B * T * OC, block_size);
    add_bias<<<grid_size, block_size>>>(out, bias, B, T, OC);
    cudaCheck(cudaGetLastError());
  }
}

void matmul_forward3(float *out, const float *inp, const float *weight, const float *bias, int B, int T, int C,
                     int OC) {
  int has_bias = (bias != NULL);
  int has_gelu = 0;

  // Tensor Core usage mandates byte alignment
  // https://docs.nvidia.com/cuda/cublas/#tensor-core-usage
  if (((uintptr_t)bias % 16) != 0) {
    printf("Bias pointer is not aligned (tensor core requirement)\n");
    exit(EXIT_FAILURE);
  }

  int returnedResults = 0;
  cublasLtMatmulDesc_t operationDesc;
  cublasLtMatmulPreference_t preference;
  cublasLtMatrixLayout_t weightLayout;
  cublasLtMatrixLayout_t inputLayout;
  cublasLtMatrixLayout_t outputLayout;
  cublasLtMatrixLayout_t biasLayout;
  cublasLtMatmulHeuristicResult_t heuristic;

  // Create the operation descriptors.
  cublasOperation_t opNoTranspose = CUBLAS_OP_N;
  cublasOperation_t opTranspose = CUBLAS_OP_T;
  cublasLtEpilogue_t epilogueBias = CUBLASLT_EPILOGUE_DEFAULT;
  if (has_bias && has_gelu) {
    epilogueBias = CUBLASLT_EPILOGUE_GELU_BIAS;
  } else if (has_bias) {
    epilogueBias = CUBLASLT_EPILOGUE_BIAS;
  } else if (has_gelu) {
    epilogueBias = CUBLASLT_EPILOGUE_GELU;
  }
  cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute_type, CUDA_R_32F));
  cublasCheck(
      cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));
  cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opNoTranspose,
                                             sizeof(opNoTranspose)));
  cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogueBias,
                                             sizeof(epilogueBias)));
  cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

  // Define matrix layouts
  cublasCheck(cublasLtMatrixLayoutCreate(&weightLayout, CUDA_R_32F, C, OC, C));
  cublasCheck(cublasLtMatrixLayoutCreate(&inputLayout, CUDA_R_32F, C, B * T, C));
  cublasCheck(cublasLtMatrixLayoutCreate(&outputLayout, CUDA_R_32F, OC, B * T, OC));
  cublasCheck(cublasLtMatrixLayoutCreate(&biasLayout, CUDA_R_32F, OC, 1, OC));

  // Create a preference handle with specified max workspace.
  cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
  cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

  // Find a suitable algorithm.
  cublasCheck(cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc, weightLayout, inputLayout, outputLayout, outputLayout, preference, 1, &heuristic, &returnedResults));
  if (returnedResults == 0) {
    printf("No cuBLASLt algorithm found for B: %d, T: %d, C: %d, OC: %d, bias: %d, gelu: %d\n", B, T, C, OC, has_bias, has_gelu);
    exit(EXIT_FAILURE);
  }
  // Call the matmul
  const float alpha = 1.0f, beta = 0.0f;
  cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc, &alpha, weight, weightLayout, inp, inputLayout, &beta, out, outputLayout, out, outputLayout, &heuristic.algo, cublaslt_workspace, cublaslt_workspace_size, 0));

  // cleanups.
  cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
  cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
  cublasCheck(cublasLtMatrixLayoutDestroy(weightLayout));
  cublasCheck(cublasLtMatrixLayoutDestroy(inputLayout));
  cublasCheck(cublasLtMatrixLayoutDestroy(outputLayout));
  cublasCheck(cublasLtMatrixLayoutDestroy(biasLayout));
}

// Kernel dispatcher.
void matmul_forward(int kernel_num, float *out, const float *inp, const float *weight, const float *bias, int B, int T,
                    int C, int OC, int sqrt_block_size) {
  switch (kernel_num) {
    case 1:
      matmul_forward1(out, inp, weight, bias, B, T, C, OC, sqrt_block_size);
      break;
    case 2:
      matmul_forward2(out, inp, weight, bias, B, T, C, OC, sqrt_block_size);
      break;
    case 3:
      matmul_forward3(out, inp, weight, bias, B, T, C, OC);
      break;
    default:
      printf("Invalid kernel number\n");
      exit(1);
  }
}

int main(int argc, char **argv) {
  srand(42);

  int B = 32;
  int T = 1024;
  int C = 768;
  int OC = C * 4;

  // Device setup
  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, deviceIdx);
  printf("Device %d: %s\n", deviceIdx, deviceProp.name);

  // Setup cuBLAS and cuBLASLt
  cublasCheck(cublasCreate(&cublas_handle));
  cublasCheck(cublasLtCreate(&cublaslt_handle));
  int enable_tf32 = (deviceProp.major >= 8) ? 1 : 0;
  printf("Enable TF32: %d\n", enable_tf32);
  cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
  cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
  cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));

  // Setup the global cuBLASLt workspace.
  cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

  // Create host memory of random tensors.
  float *out = (float *)malloc(B * T * OC * sizeof(float));
  float *inp = make_random_float(B * T * C);
  float *weight = make_random_float(OC * C);
  float *bias = make_random_float(OC);

  // Move to GPU
  float *d_out;
  float *d_inp;
  float *d_weight;
  float *d_bias;
  cudaCheck(cudaMalloc(&d_out, B * T * OC * sizeof(float)));
  cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
  cudaCheck(cudaMalloc(&d_weight, OC * C * sizeof(float)));
  cudaCheck(cudaMalloc(&d_bias, OC * sizeof(float)));
  cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_weight, weight, OC * C * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_bias, bias, OC * sizeof(float), cudaMemcpyHostToDevice));

  int kernel_num;
  if (argc > 1) {
    kernel_num = std::atoi(argv[1]);
  }
  printf("Using kernel %d\n", kernel_num);

  // CPU reference.
  matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

  // Block sizes (sqrt of the total threads we can afford in each block)
  int sqrt_block_sizes[] = {4, 8, 16, 32};

  for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++) {
    int sqrt_block_size = sqrt_block_sizes[j];
    printf("Checking block size %dx%d\n", sqrt_block_size, sqrt_block_size);
    matmul_forward(kernel_num, d_out, d_inp, d_weight, d_bias, B, T, C, OC, sqrt_block_size);
    validate_result(d_out, out, "out", B * T * OC, 1e-1f);
  }
  printf("All results match, starting benchmarks.\n");

  for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++) {
    int sqrt_block_size = sqrt_block_sizes[j];
    int repeats = 100;
    float elapsed_time_ms = benchmark_kernel(repeats, matmul_forward, kernel_num, d_out, d_inp, d_weight, d_bias, B, T,
                                             C, OC, sqrt_block_size);

    // Estimate flops achieved.
    float tflops = (float)2 * B * T * C * OC / elapsed_time_ms * 1e3f / 1e12f;
    printf("sqrt_block_size %4d | time %.4f ms | tflops %.2f\n", sqrt_block_size, elapsed_time_ms, tflops);
  }

  // Cleanup
  free(out);
  free(inp);
  free(weight);
  free(bias);
  cudaCheck(cudaFree(d_out));
  cudaCheck(cudaFree(d_inp));
  cudaCheck(cudaFree(d_weight));
  cudaCheck(cudaFree(d_bias));
  cublasCheck(cublasDestroy(cublas_handle));
  cublasCheck(cublasLtDestroy(cublaslt_handle));
  return 0;
}