import numpy as np
import scipy.linalg as sl

from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.numpy.core import convert_to_tensor


def cholesky(a, upper=False):
    return np.linalg.cholesky(a, upper=upper)


def cholesky_inverse(a, upper=False):
    identity = np.eye(a.shape[-1], dtype=a.dtype)
    inv_chol = solve_triangular(a, identity, lower=not upper)
    if upper:
        a_inv = np.matmul(inv_chol, inv_chol.T)
    else:
        a_inv = np.matmul(inv_chol.T, inv_chol)
    return a_inv


def det(a):
    return np.linalg.det(a)


def eig(a):
    return np.linalg.eig(a)


def eigh(a):
    return np.linalg.eigh(a)


def inv(a):
    return np.linalg.inv(a)


def lu_factor(a):
    if a.ndim == 2:
        return sl.lu_factor(a)

    m, n = a.shape[-2:]
    signature = "(m,n) -> (m,n), "
    signature += "(m)" if m <= n else "(n)"
    _lu_factor_gufunc = np.vectorize(
        sl.lu_factor,
        signature=signature,
    )
    return _lu_factor_gufunc(a)


def norm(x, ord=None, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        dtype = dtypes.result_type(x.dtype, "float32")
    return np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims).astype(
        dtype
    )


def qr(x, mode="reduced"):
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    return np.linalg.qr(x, mode=mode)


def solve(a, b):
    return np.linalg.solve(a, b)


def solve_triangular(a, b, lower=False):
    if a.ndim == 2:
        return sl.solve_triangular(a, b, lower=lower)

    # Instead of np.vectorize, which is slow, use a more efficient approach:
    # We'll use np.apply_along_axis if b is 2D and a is 3D and broadcast if possible,
    # or manually vectorized looping if needed
    batch_shape = a.shape[:-2]
    n, m = b.shape[-2], b.shape[-1] if b.ndim > a.ndim - 1 else 1
    a_reshaped = a.reshape(-1, a.shape[-2], a.shape[-1])

    if b.ndim == a.ndim - 1:
        b_reshaped = b.reshape(-1, b.shape[-1])
        results = [sl.solve_triangular(ai, bi, lower=lower) for ai, bi in zip(a_reshaped, b_reshaped)]
        results = np.stack(results, axis=0)
        results = results.reshape(*batch_shape, results.shape[-1])
        return results
    else:
        b_reshaped = b.reshape(-1, b.shape[-2], b.shape[-1])
        results = [sl.solve_triangular(ai, bi, lower=lower) for ai, bi in zip(a_reshaped, b_reshaped)]
        results = np.stack(results, axis=0)
        results = results.reshape(*batch_shape, results.shape[-2], results.shape[-1])
        return results


def svd(x, full_matrices=True, compute_uv=True):
    return np.linalg.svd(x, full_matrices=full_matrices, compute_uv=compute_uv)


def lstsq(a, b, rcond=None):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    return np.linalg.lstsq(a, b, rcond=rcond)[0]


def jvp(fun, primals, tangents, has_aux=False):
    raise NotImplementedError("JVP is not supported by the Numpy backend.")
