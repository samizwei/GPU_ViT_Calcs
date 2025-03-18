import numpy as np
import scipy
import time
import jax
import jax.numpy as jnp
from netket.jax import tree_ravel

def _qr_solver(A, b):
    error = 1e-5
    m, n = A.shape
    norm = jnp.linalg.norm(A)
    # A /= norm
    # b /= norm
    # print(A[:10])
    # Compute QR decomposition with column pivoting
    Q, R, P = scipy.linalg.qr(A, mode='economic', pivoting=True)
    print(n)
    rank = np.count_nonzero(np.abs(np.diag(R)) > error)
    print(rank)
    print(np.diag(R)[rank-10:rank+10])
    S = np.linalg.svd(A, compute_uv=False)
    print(np.count_nonzero(S > error))
    print(S[rank-10:rank+10])

    # Find the rank of R
    rank = np.count_nonzero(np.abs(np.diag(R)) > error)

    QT = Q.T[:rank]
    
    # Solve Rx = Q^T b
    x_temp = scipy.linalg.solve_triangular(R[:rank, :rank], QT @ b, lower=False)
    
    # Undo the column permutation
    x = np.empty(n)
    x[P] = np.pad(x_temp, (0, n - rank), 'constant')  # Apply permutation
    # print(rank)
    # print(x)

    return x

@jax.jit
def qr_solver(A, b, **kwargs):
    if not isinstance(A, jax.numpy.ndarray):
        A = A.to_dense()
    shape = A.shape[1]
    dtype = jax.numpy.result_type(A)
    out_type = jax.ShapeDtypeStruct((shape,), dtype)
    if not isinstance(b, jax.numpy.ndarray):
        b, unravel = tree_ravel(b)
        return unravel(jax.pure_callback(_qr_solver, out_type, A, b)), None
    else:
        return jax.pure_callback(_qr_solver, out_type, A, b), None

def svd_solver(A, b):
    # Compute SVD of A
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    
    # Solve Ax = b
    x = Vh.T @ np.diag(1/S), U.T @ b
    
    return x

# # Test case
# m, n = 5000, 5000
# A = np.random.randn(m, n)
# b = np.random.randn(m)
# 
# now = time.time()
# 
# x1 = np.linalg.lstsq(A, b, rcond=None)[0]
# print("Time taken by numpy lstsq:", time.time() - now)
# now = time.time()
# x2 = qr_solver(A, b)[0]
# print("Time taken by QR solver:", time.time() - now)
# now = time.time()
# x3 = svd_solver(A, b)
# print("Time taken by SVD solver:", time.time() - now)
# 
# print("Error:", np.linalg.norm(x1 - x2))  # Should be close to 0
