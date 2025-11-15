# make_test_matrices.py
import numpy as np

np.savez("matrix_a.npz", arr_0=np.random.rand(512,512).astype(np.float32))
np.savez("matrix_b.npz", arr_0=np.random.rand(512,512).astype(np.float32))
print("Saved matrix_a.npz and matrix_b.npz")
