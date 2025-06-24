import cupy as cp
import numpy as np
import time

import numpy as np
import cupy as cp
import time

# Taille des matrices
N = 5

# --- CPU avec NumPy ---
A_cpu = np.random.rand(N, N)
B_cpu = np.random.rand(N, N)

start_cpu = time.time()
C_cpu = np.matmul(A_cpu, B_cpu)
end_cpu = time.time()
print(f"NumPy (CPU) time: {end_cpu - start_cpu:.4f} seconds")

# --- GPU avec CuPy ---
A_gpu = cp.random.rand(N, N)
B_gpu = cp.random.rand(N, N)

# Warm-up (CUDA init)
cp.matmul(A_gpu, B_gpu)

start_gpu = time.time()
C_gpu = cp.matmul(A_gpu, B_gpu)
cp.cuda.Device().synchronize()  # attendre la fin du calcul GPU
end_gpu = time.time()
print(f"CuPy (GPU) time: {end_gpu - start_gpu:.4f} seconds")
