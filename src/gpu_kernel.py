# src/gpu_kernel.py
from numba import cuda
import math
import numpy as np

@cuda.jit
def mat_add_kernel(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        C[i, j] = A[i, j] + B[i, j]

def add_on_gpu(A: np.ndarray, B: np.ndarray):
    """
    A, B: numpy float32 2D arrays with same shape
    returns: C (numpy array), elapsed_time_seconds (kernel only)
    """
    # allocate device arrays
    dA = cuda.to_device(A)
    dB = cuda.to_device(B)
    dC = cuda.device_array_like(A)

    threadsperblock = (16, 16)
    blockspergrid = (math.ceil(A.shape[0] / threadsperblock[0]),
                     math.ceil(A.shape[1] / threadsperblock[1]))

    # launch and time kernel (kernel time measured on host side; for precise timing use CUDA events)
    import time
    start = time.perf_counter()
    mat_add_kernel[blockspergrid, threadsperblock](dA, dB, dC)
    # Wait for kernel to finish
    cuda.synchronize()
    elapsed = time.perf_counter() - start

    C = dC.copy_to_host()
    return C, elapsed
