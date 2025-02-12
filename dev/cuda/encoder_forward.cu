/**
 * Encoder Forward Kernel Implementations.
 * nvcc --use_fast_math -arch=sm_86 -O3 -lcublas -lcublasLt encoder_forward.cu -o encoder_forward
 * Run: ./encoder_forward 2
 * Variants:
 *  1. Naive kernel that parallelizes over B*T, loops over C
 *  2. Optimized kernel that parallelizes over B*T*C
 *  3. Previous, but uses faster float4 reads/writes
 */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define ENABLE_BF16
#include "common.h"

// Kernels

/**
 * Naive kernel that parallelizes over B, T, loop over C.
 */
__global__ void encoder_forward_kernel1(floatX *out, const int *inp, const floatX *wte, const floatX *wpe, int B, int T,
                                        int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = B * T;

  if (idx < N) {
    int b = idx / T;
    int t = idx % T;
    floatX *out_bt = out + b * T * C + t * C;
    int ix = inp[b * T + t];
    for (int c = 0; c < C; c++) {
      out_bt[c] = (floatX)(wte[ix * C + c] + wpe[t * C + c]);
    }
  }
}

/**
 * Optimize over naive kernel by also parallelizing over C.
 */
__global__ void encoder_forward_kernel2(floatX *out, const int *inp, const floatX *wte, const floatX *wpe, int B, int T,
                                        int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = B * T * C;
  if (idx < N) {
    int bt = idx / C;
    int b = bt / T;
    int t = bt % T;
    int c = idx % C;
    int ix = inp[b * T + t];
    out[b * T * C + t * C + c] = wte[ix * C + c] + wpe[t * C + c];
  }
}

/**
 * Use vector load/store to optimize adds.
 */
__global__ void encoder_forward_kernel3(floatX *out, const int *inp, const floatX *wte, const floatX *wpe, int B, int T,
                                        int C) {
  int idx = (blockDim.x * blockIdx.x + threadIdx.x) * x128::size;
  int N = B * T * C;
  if (idx < N) {
    int bt = idx / C;
    int b = bt / T;
    int t = bt % T;
    int c = idx % C;
    int ix = inp[b * T + t];

    floatX *out_btc = out + (b * T * C + t * C + c);
    const floatX *wte_bt = wte + (ix * C + c);
    const floatX *wpe_bt = wpe + (t * C + c);

    // Load 128-byte vectors and add (local scope wpe/wte shadow variables)
    x128 packed_out;
    x128 wte = load128(wte_bt);
    x128 wpe = load128(wpe_bt);

    #pragma unroll
    for (int k = 0; k < x128::size; k++) {
      packed_out[k] = wte[k] + wpe[k];
    }
    store128(out_btc, packed_out);
  }
}

void encoder_forward1(floatX *out, const int *inp, const floatX *wte, const floatX *wpe, int B, int T, int C,
                      int block_size) {
  const int N = B * T;
  const int grid_size = ceil_div(N, block_size);
  encoder_forward_kernel1<<<grid_size, block_size>>>(out, inp, wte, wpe, B, T, C);
  cudaCheck(cudaGetLastError());
}

void encoder_forward2(floatX *out, const int *inp, const floatX *wte, const floatX *wpe, int B, int T, int C,
                      int block_size) {
  const int N = B * T * C;
  const int grid_size = ceil_div(N, block_size);
  encoder_forward_kernel2<<<grid_size, block_size>>>(out, inp, wte, wpe, B, T, C);
  cudaCheck(cudaGetLastError());
}

void encoder_forward3(floatX *out, const int *inp, const floatX *wte, const floatX *wpe, int B, int T, int C,
                      int block_size) {
  const int N = B * T * C;
  const int grid_size = ceil_div(N, block_size * x128::size);
  encoder_forward_kernel3<<<grid_size, block_size>>>(out, inp, wte, wpe, B, T, C);
  cudaCheck(cudaGetLastError());
}

/**
 * Naive CPU Kernel to generate reference output.
 */
void encoder_forward_cpu(float *out, const int *inp, const float *wte, const float *wpe, int B, int T, int C) {
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      for (int c = 0; c < C; c++) {
        int ix = inp[b * T + t];
        out[b * T * C + t * C + c] = wte[ix * C + c] + wpe[t * C + c];
      }
    }
  }
}

/**
 * Kernel launcher func
 */
void encoder_forward(int kernel_num, floatX *out, const int *inp, const floatX *wte, const floatX *wpe, int B, int T,
                     int C, int block_size) {
  switch (kernel_num) {
    case 1:
      encoder_forward1(out, inp, wte, wpe, B, T, C, block_size);
      break;
    case 2:
      encoder_forward2(out, inp, wte, wpe, B, T, C, block_size);
      break;
    case 3:
      encoder_forward3(out, inp, wte, wpe, B, T, C, block_size);
      break;
    default:
      printf("Invalid kernel number \n");
      exit(1);
  }
}

int main(int argc, char **argv) {
  setup_main();

  int B = 8;
  int T = 1024;
  int C = 768;
  int V = 50257;

  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));

  // Generate random tensors on host.
  float *out = (float *)malloc(B * T * C * sizeof(float));
  int *inp = make_random_int(B * T, V);
  float *wte = make_random_float(V * C);
  float *wpe = make_random_float(T * C);

  // Move to GPU
  floatX *d_out;
  int *d_inp;
  floatX *d_wte;
  floatX *d_wpe;
  cudaCheck(cudaMalloc(&d_out, sizeof(floatX) * B * T * C));
  cudaCheck(cudaMalloc(&d_wte, sizeof(floatX) * V * C));
  cudaCheck(cudaMalloc(&d_wpe, sizeof(floatX) * T * C));
  cudaCheck(cudaMalloc(&d_inp, sizeof(int) * B * T));
  cudaCheck(cudaMemcpy(d_inp, inp, sizeof(int) * B * T, cudaMemcpyHostToDevice));
  cudaCheck(memcpy_convert(d_wte, wte, V * C));
  cudaCheck(memcpy_convert(d_wpe, wpe, T * C));

  // Read kernel_num from CLI
  int kernel_num = 2;
  if (argc > 1) {
    kernel_num = atoi(argv[1]);
  }
  printf("Using kernel %d\n", kernel_num);

  // Generate ground truth data
  encoder_forward_cpu(out, inp, wte, wpe, B, T, C);

  // Verify kernel at different block sizes.
  int block_sizes[] = {32, 64, 128, 256, 512, 1024};
  for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
    int block_size = block_sizes[j];
    printf("Checking block size %d.\n", block_size);
    encoder_forward(kernel_num, d_out, d_inp, d_wte, d_wpe, B, T, C, block_size);
#ifdef ENABLE_BF16
    float tol = 1e-2f;
#else
    float tol = 1e-5f;
#endif
    validate_result(d_out, out, "out", B * T * C, tol);
  }

  // Now that validated, benchmark the kernel.
  printf("All results match. Starting benchmarks.\n");

  for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
    int block_size = block_sizes[j];
    int repeat_times = 1000;
    float elapsed_time =
        benchmark_kernel(repeat_times, encoder_forward, kernel_num, d_out, d_inp, d_wte, d_wpe, B, T, C, block_size);

// Estimate memory bandwidth achieved: Total memory_ops * bytes per value.
// * Read one inp token, one wte float, one wpe float
// * Write one out float
// Total = (1 + 3C) ops, 2 bytes each (4 bytes if f32)
#ifdef ENABLE_BF16
    int bytes_per_op = 2;
#else
    int bytes_per_op = 4;
#endif
    long memory_ops = B * T * (1 + 3 * C) * bytes_per_op;
    float memory_bandwidth = memory_ops / elapsed_time / 1e6;
    printf("block_size %d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
  }
}