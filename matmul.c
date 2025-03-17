#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "llmc/rand.h"
#include "llmc/utils.h"

#ifdef OMP
#include <omp.h>
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))

double tick() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

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
    if (fabsf(a[i] - b[i]) > rtol) {
      printf("mismatch at idx %d, %f != %f \n", i, a[i], b[i]);
      exit(1);
    }
  }
}

#define BLOCK 4

void matmul(const float *left, const float *right, float *out, int rows, int inners, int cols) {
  // #pragma omp parallel for collapse(2) shared(left, right, out)
  for (int by = 0; by < rows; by += BLOCK) {
    for (int bx = 0; bx < cols; bx += BLOCK) {
      float tc[BLOCK][BLOCK] = {};
      // Compute
      for (int k = 0; k < inners; k++) {
        for (int y = 0; y < BLOCK; y++) {
          for (int x = 0; x < BLOCK; x++) {
            tc[y][x] += left[(by + y) * inners + k] * right[(bx + x) * cols + k];
          }
        }
      }

      // Store
      for (int y = 0; y < BLOCK; y++) {
        for (int x = 0; x < BLOCK; x++) {
          out[(by + y) * cols + (bx + x)] = tc[y][x];
        }
      }
    }
  }
}

#ifdef DEBUG
#define N 4
#else
#define N 2048
#endif

float A[N * N] __attribute__((aligned(32)));
float B[N * N] __attribute__((aligned(32)));
float C[N * N] __attribute__((aligned(32)));
float val[N * N] __attribute__((aligned(32)));

int main() {
  printf("Starting...\n");
  /**
   * Xeon 6258R
   * 2 AVX-512 FMA units
   * = 2 * 16 * 2 = 64 FLOP/cycle
   * = 2.7 * 64 = 172.8 GFLOP/s at 2.7GHz
   */

  // initialize
  FILE *file = fopenCheck("/tmp/matmul", "rb");
  freadCheck(A, 1, sizeof(float) * N * N, file);
  freadCheck(B, 1, sizeof(float) * N * N, file);
  freadCheck(C, 1, sizeof(float) * N * N, file);
  fcloseCheck(file);
  memset(val, 0, sizeof(float) * N * N);

  // Validate result
  matmul(A, B, val, N, N, N);
  allclose(val, C, N*N, 1e-3f);
  printf("Results verified, starting benchmarks...\n");

  // prints
  int repeats = 2;
  for (int i = 0; i < repeats; i++) {
    uint64_t start = tick();
    matmul(A, B, val, N, N, N);
    uint64_t stop = tick();
    double elapsed_time = (stop - start) * 1e-3;
    printf("GFLOP/s: %f\n", (2.0 * N * N * N * 1e-9) / elapsed_time);
  }

#ifdef DEBUG
  print_matrix(A, N, N);
  print_matrix(B, N, N);
  print_matrix(C, N, N);
  print_matrix(val, N, N);
#endif

  return 0;
}