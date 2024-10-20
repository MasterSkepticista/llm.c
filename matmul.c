#include <stdio.h>
#include <time.h>

#include "llmc/rand.h"
#include "llmc/utils.h"

#ifdef OMP
#include <omp.h>
#endif

void print_matrix(float *m, int rows, int cols) {
  printf("_____________\n");
  for (int i = 0; i < rows; i++) {
    printf("[");
    for (int j = 0; j < cols; j++) {
      printf("%8.3f", m[i * cols + j]);
    }
    printf("]\n");
  }
  printf("_____________\n");
}

/**
 * Initialize an array with a constant.
 */
void constant_(float *m, int numel, float val) {
  for (int i = 0; i < numel; i++) {
    m[i] = val;
  }
}

void transpose_(float *m, int rows, int cols) {
#pragma omp parallel for
  for (int i = 0; i < rows; i++) {
    for (int j = i + 1; j < cols; j++) {
      if (i != j) {
        float temp = m[i * cols + j];
        m[i * cols + j] = m[j * rows + i];
        m[j * rows + i] = temp;
      }
    }
  }
}

void allclose(float *a, float *b, int numel, float rtol) {
  for (int i = 0; i < numel; i++) {
    if (fabsf(a[i] - b[i]) > 1e-3) {
      printf("mismatch at idx %d, %f != %f \n", i, a[i], b[i]);
      exit(1);
    }
  }
  printf("match\n");
}

void matmul(float *left, float *right, float *out, int rows, int inners, int cols) {
#pragma omp parallel for
  for (int row = 0; row < rows; row++) {
    for (int inner = 0; inner < inners; inner++) {
      for (int col = 0; col < cols; col++) {
        out[row * cols + col] += left[row * cols + inner] * right[inner * cols + col];
      }
    }
  }
}

void matmul_tiled(float *left, float *right, float *out, int rows, int inners, int cols) {
  const int TILE_SIZE = 8;

#pragma omp parallel for
  for (int i = 0; i < rows; i += TILE_SIZE) {
    for (int j = 0; j < inners; j++) {
      // initialize tile with zeros
      float tile[TILE_SIZE];
      for (int t = 0; t < TILE_SIZE; t++) {
        tile[t] = 0.0f;
      }

      for (int k = 0; k < cols; k++) {
        for (int t = 0; t < TILE_SIZE; t++) {
          int it = i + t;
          tile[t] += left[it * inners + k] * right[j * cols + k];
        }
      }

      // writeback
      for (int t = 0; t < TILE_SIZE; t++) {
        int it = i + t;
        out[it * cols + j] = tile[t];
      }
    }
  }
}

void matmul_tiled2(float *out, const float *inp, const float *weight, const float *bias, int BT, int C, int OC) {
  const int LOOP_UNROLL = 8;
#pragma omp parallel for
  for (int obt = 0; obt < BT; obt += LOOP_UNROLL) {
    for (int o = 0; o < OC; o++) {
      // we'll keep LOOP_UNROLL many results in registers
      float result[LOOP_UNROLL];
      // initialize the bias, if it exists
      for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
        result[ibt] = (bias != NULL) ? bias[o] : 0.0f;
      }
      // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
      // the value of weight[i + o * C] and reuse it.
      // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
      for (int i = 0; i < C; i++) {
        float w = weight[i + o * C];
        for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
          int bt = obt + ibt;
          result[ibt] += inp[bt * C + i] * w;
        }
      }
      // write back results to main memory
      for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
        int bt = obt + ibt;
        out[bt * OC + o] = result[ibt];
      }
    }
  }
}

#ifdef DEBUG
#define N 4
#else
#define N 4096
#endif

int main() {
  float *A = (float *)mallocCheck(sizeof(float) * N * N);
  float *B = (float *)mallocCheck(sizeof(float) * N * N);
  float *C = (float *)mallocCheck(sizeof(float) * N * N);
  float *val = (float *)mallocCheck(sizeof(float) * N * N);

  // initialize
  FILE *file = fopenCheck("/tmp/matmul", "rb");
  freadCheck(A, 1, sizeof(float) * N * N, file);
  freadCheck(B, 1, sizeof(float) * N * N, file);
  freadCheck(C, 1, sizeof(float) * N * N, file);
  fcloseCheck(file);

  // prints
  size_t total_flop = 2.0 * N * N * N;
  struct timespec start, end;

  transpose_(B, N, N);
  for (int i = 0; i < 10; i++) {
    constant_(val, N * N, 0.0f);

    clock_gettime(CLOCK_MONOTONIC, &start);
    // matmul(A, B, val, N, N, N);
    matmul_tiled(A, B, val, N, N, N);
    // matmul_tiled2(val, A, B, NULL, N, N, N);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double cpu_time_used = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("GFLOP/s: %f\n", (total_flop / 1e9) / cpu_time_used);
    allclose(val, C, N * N, 1e-5f);
  }

#ifdef DEBUG
  print_matrix(A, N, N);
  print_matrix(B, N, N);
  print_matrix(C, N, N);
  print_matrix(val, N, N);
#endif

  return 0;
}