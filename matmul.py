#!/usr/bin/python3
import os
os.environ["OMP_NUM_THREADS"] = "64"
import time
import numpy as np

N = 4096
np.random.seed(42)
A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)
C = A @ B

with open("/tmp/matmul", "wb") as f:
  f.write(A.tobytes())
  f.write(B.tobytes())
  f.write(C.tobytes())

flop = N * N * 2 * N

for i in range(2):
  t0 = time.monotonic()
  C = A @ B
  dt = (time.monotonic() - t0) * 1e9

  print("GFLOPS:", flop / dt)