# src/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import io
from src.gpu_kernel import add_on_gpu

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/add")
async def add_matrices(file_a: UploadFile = File(...), file_b: UploadFile = File(...)):

    # load matrices
    try:
        A = np.load(io.BytesIO(await file_a.read()))["arr_0"]
        B = np.load(io.BytesIO(await file_b.read()))["arr_0"]
    except:
        raise HTTPException(status_code=400, detail="Invalid NPZ files")

    if A.shape != B.shape:
        raise HTTPException(status_code=400, detail="Matrices must have the same shape")

    C, elapsed = add_on_gpu(A.astype(np.float32), B.astype(np.float32))

    return {
        "matrix_shape": list(A.shape),
        "elapsed_time": elapsed,
        "device": "GPU"
    }
