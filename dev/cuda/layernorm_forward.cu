#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "common.h"

void layernorm_forward_cpu(float *out, float *mean, float *rstd, float *inp, const float *weight, const float *bias,
                           int B, int T, int C) {
  const float eps = 1e-5f;
#pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      const float *x = inp + b * T * C + t * C;

      // Compute mu/var/s
      float mu = 0.0f;
      for (int c = 0; c < C; c++) {
        mu += x[c];
      }
      mu /= C;

      float var = 0.0f;
      for (int c = 0; c < C; c++) {
        float xshift = (x[c] - mu);
        var += xshift * xshift;
      }
      var /= C;
      float s = 1 / sqrtf(var + eps);

      // Scale by weight, add bias.
      float *out_bt = out + b * T * C + t * C;
      for (int c = 0; c < C; c++) {
        out_bt[c] = ((x[c] - mu) * s) * weight[c] + bias[c];
      }

      // Cache mu/rstd
      mean[b * T + t] = mu;
      rstd[b * T + t] = s;
    }
  }
}

/**
 * Naive parallelism over N=B*T, loop over C - similar to CPU, but on GPU.
 */
__global__ void layernorm_forward_kernel1(float *out, float *mean, float *rstd, float *inp, const float *weight,
                                          const float *bias, int N, int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float eps = 1e-5f;

  if (idx < N) {
    const float *x = inp + idx * C;

    float mu = 0.0f;
    for (int c = 0; c < C; c++) {
      mu += x[c];
    }
    mu /= C;

    float var = 0.0f;
    for (int c = 0; c < C; c++) {
      float xshift = (x[c] - mu);
      var += xshift * xshift;
    }
    var /= C;
    float s = 1.0f / sqrtf(var + eps);

    float *out_bt = out + idx * C;
    for (int c = 0; c < C; c++) {
      float xshift = (x[c] - mu);
      out_bt[c] = (xshift * weight[c]) * s + bias[c];
    }

    mean[idx] = mu;
    rstd[idx] = s;
  }
}

__global__ void mean_kernel(float *mean, const float *inp, int N, int C, int block_size) {
  extern __shared__ float shared[];
  int idx = blockIdx.x;
  int tid = threadIdx.x;
  const float *x = inp + idx * C;
  // thread coarsening
  float sum = 0.0f;
  for (int i = tid; i < C; i += block_size) {
    sum += x[i];
  }
  shared[tid] = sum;
  __syncthreads();
  // reductions
  for (int stride = block_size / 2; stride >= 1; stride /= 2) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  // write the final result (at thread 0) to global memory
  if (tid == 0) {
    mean[idx] = shared[0] / C;
  }
}

__global__ void rstd_kernel(float *rstd, const float *inp, const float *mean, int N, int C, int block_size) {
  extern __shared__ float shared[];
  int idx = blockIdx.x;
  int tid = threadIdx.x;
  const float *x = inp + idx * C;
  // thread coarsening.
  float sum = 0.0f;
  float mu = mean[idx];
  for (int i = tid; i < C; i += block_size) {
    float xshift = x[i] - mu;
    sum += xshift * xshift;
  }
  shared[tid] = sum;
  __syncthreads();
  // reductions
  for (int stride = block_size / 2; stride >= 1; stride /= 2) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  // write the final result (at thread 0) to global memory
  if (tid == 0) {
    rstd[idx] = 1.0f / sqrtf(shared[0] / C + 1e-5f);
  }
}

__global__ void normalization_kernel(float *out, float *mean, float *rstd, const float *inp, const float *weight,
                                     const float *bias, int B, int T, int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int bt = idx / C;
  int c = idx % C;

  out[idx] = weight[c] * (rstd[bt] * (inp[idx] - mean[bt])) + bias[c];
}

/**
 * Cooperative groups to fuse 3 steps into one (eliminates launch overheads).
 * Each warp is responsible for layernorm of one row of C values.
 */
__global__ void layernorm_forward_kernel3(float *out, float *mean, float *rstd, const float *inp, const float *weight,
                                          const float *bias, int N, int C) {
  namespace cg = cooperative_groups;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

  // meta_group_size is the number of subgroups within a parent group. Here, number of warps in a block.
  // meta_group_rank is the rank of subgroup within a parent group. Here, warp index.
  // idx == global warp id.
  int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
  if (idx >= N) {
    return;
  }

  // Row of input that this warp will be responsible for.
  const float *x = inp + idx * C;

  // mean
  float sum = 0.0f;
  for (int i = warp.thread_rank(); i < C; i += warp.size()) {
    sum += x[i];
  }
  sum = cg::reduce(warp, sum, cg::plus<float>{});
  float m = sum / C;
  if (warp.thread_rank() == 0 && mean != nullptr) {
    __stcs(mean + idx, m);
  }

  // rstd
  sum = 0.0f;
  for (int i = warp.thread_rank(); i < C; i += warp.size()) {
    float diff = x[i] - m;
    sum += diff * diff;
  }
  sum = cg::reduce(warp, sum, cg::plus<float>{});
  float s = rsqrtf(sum / C + 1e-5f);
  if (warp.thread_rank() == 0 && rstd != nullptr) {
    __stcs(rstd + idx, s);
  }

  // Final normalization and rescaling
  float *o = out + idx * C;
  for (int c = warp.thread_rank(); c < C; c += warp.size()) {
    // load/store using cache-streaming hints ".cs" to the compiler. It is functionally
    // equivalent to the lines:
    // float n = s * (x[c] - m);
    // o[c] = n * weight[c] + bias[c];
    // streaming hints indicate the compiler that the data will not be reused soon.
    float n = s * (__ldcs(x + c) - m);
    __stcs(o + c, n * weight[c] + bias[c]);
  }
}

void layernorm_forward1(float *out, float *mean, float *rstd, float *inp, const float *weight, const float *bias, int B,
                        int T, int C, int block_size) {
  int N = B * T;
  int grid_size = ceil_div(N, block_size);
  layernorm_forward_kernel1<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
  cudaCheck(cudaGetLastError());
}

/**
 * Parallel Reduction for mean/rstd. Normalize in a separate kernel.
 * This requires three kernel invocations.
 */
void layernorm_forward2(float *out, float *mean, float *rstd, float *inp, const float *weight, const float *bias, int B,
                        int T, int C, int block_size) {
  int N = B * T;
  mean_kernel<<<N, block_size, block_size * sizeof(float)>>>(mean, inp, N, C, block_size);
  cudaCheck(cudaGetLastError());
  rstd_kernel<<<N, block_size, block_size * sizeof(float)>>>(rstd, inp, mean, N, C, block_size);
  cudaCheck(cudaGetLastError());
  int block_size2 = 256;
  int grid_size = ceil_div(B * T * C, block_size2);
  normalization_kernel<<<grid_size, block_size2>>>(out, mean, rstd, inp, weight, bias, B, T, C);
  cudaCheck(cudaGetLastError());
}

void layernorm_forward3(float *out, float *mean, float *rstd, float *inp, const float *weight, const float *bias, int B,
                        int T, int C, int block_size) {
  assert(block_size % 32 == 0);
  const int N = B * T;
  const int grid_size = ceil_div(N * 32, block_size);
  layernorm_forward_kernel3<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
  cudaCheck(cudaGetLastError());
}

void layernorm_forward(int kernel_num, float *out, float *mean, float *rstd, float *inp, const float *weight,
                       const float *bias, int B, int T, int C, int block_size) {
  switch (kernel_num) {
    case 1:
      layernorm_forward1(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
      break;
    case 2:
      layernorm_forward2(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
      break;
    case 3:
      layernorm_forward3(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
      break;
    default:
      printf("Invalid kernel number.");
      exit(1);
  }
}

int main(int argc, char **argv) {
  srand(42);

  int B = 8;
  int T = 1024;
  int C = 768;

  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));

  // Create host memory of random numbers.
  float *out = (float *)malloc(B * T * C * sizeof(float));
  float *mean = (float *)malloc(B * T * sizeof(float));
  float *rstd = (float *)malloc(B * T * sizeof(float));
  float *inp = make_random_float(B * T * C);
  float *weight = make_random_float(C);
  float *bias = make_random_float(C);

  // move to GPU
  float *d_out;
  float *d_mean;
  float *d_rstd;
  float *d_inp;
  float *d_weight;
  float *d_bias;
  cudaCheck(cudaMalloc(&d_out, sizeof(float) * B * T * C));
  cudaCheck(cudaMalloc(&d_mean, sizeof(float) * B * T));
  cudaCheck(cudaMalloc(&d_rstd, sizeof(float) * B * T));
  cudaCheck(cudaMalloc(&d_inp, sizeof(float) * B * T * C));
  cudaCheck(cudaMalloc(&d_weight, sizeof(float) * C));
  cudaCheck(cudaMalloc(&d_bias, sizeof(float) * C));
  cudaCheck(cudaMemcpy(d_inp, inp, sizeof(float) * B * T * C, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_weight, weight, sizeof(float) * C, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_bias, bias, sizeof(float) * C, cudaMemcpyHostToDevice));

  // read kernel_num from CLI
  int kernel_num = 2;
  if (argc > 1) {
    kernel_num = atoi(argv[1]);
  }
  printf("Using kernel %d\n", kernel_num);
  int block_sizes[] = {32, 64, 128, 256, 512, 1024};
  layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);

  // Check the correctness of kernel at all block sizes.
  for (int i = 0; i < sizeof(block_sizes) / sizeof(int); i++) {
    int block_size = block_sizes[i];
    printf("Checking block size %d\n", block_size);
    layernorm_forward(kernel_num, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, block_size);
    validate_result(d_out, out, "ln_out", B * T * C, 1e-5f);
    validate_result(d_mean, mean, "ln_mean", B * T, 1e-5f);
    validate_result(d_rstd, rstd, "ln_rstd", B * T, 1e-5f);
  }
  printf("All results match, starting benchmarks.\n");
  // return 0;
  // Time the kernel at all block sizes.
  int repeats = 1000;
  for (int i = 0; i < sizeof(block_sizes) / sizeof(int); i++) {
    int block_size = block_sizes[i];
    float elapsed_time = benchmark_kernel(repeats, layernorm_forward, kernel_num, d_out, d_mean, d_rstd, d_inp,
                                          d_weight, d_bias, B, T, C, block_size);

    // Estimate memory bandwidth achieved (4 bytes per float).
    long memory_ops = B * T * (2 * C) * 4;
    float memory_bandwidth = memory_ops / elapsed_time / 1e6;
    printf("block_size %4d | time %.4f | bandwidth %.2f GB/s \n", block_size, elapsed_time, memory_bandwidth);
  }

  // Cleanup
  free(out);
  free(mean);
  free(rstd);
  free(inp);
  free(weight);
  free(bias);
  cudaCheck(cudaFree(d_out));
  cudaCheck(cudaFree(d_mean));
  cudaCheck(cudaFree(d_rstd));
  cudaCheck(cudaFree(d_inp));
  cudaCheck(cudaFree(d_weight));
  cudaCheck(cudaFree(d_bias));
}