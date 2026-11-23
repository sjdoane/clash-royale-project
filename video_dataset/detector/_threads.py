"""Import first: one BLAS / OpenMP / OpenCV thread per process (shared GPU box;
parallelism comes from running several videos, not from threads)."""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"
try:
    import cv2
    cv2.setNumThreads(1)
except ImportError:
    pass
