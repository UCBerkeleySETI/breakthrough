import numpy as np
import numba as nb
import math

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Fast Matrix Padding with Numba JIT for building B-matrix in DEMUD.
#
# Pads a Vector of shape (1, D) with 0's above to yield a Vector of shape (2, D).
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
@nb.jit(nopython=True)
def _pad(x, mu, C, out, D):
    for i in range(D):
        out[1, i] = C * (x[0, i] - mu[0, i])

@nb.jit(nopython=True)
def pad_top(x, mu, C):
    D = x.shape[1]
    out = np.zeros((2, D))
    _pad(x, mu, C, out, D)
    return out


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Numba Version of Building Transform Matrix, R for DEMUD.
# 
# Builds the matrix R = [[np.diag(np.sqrt(s)), 0], [B.dot(U), R.T]].
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
@nb.jit(nopython=True)
def _btm_helper(s, d, ll, R, out):
    for i in range(d):
        out[i, i] = np.sqrt(s[i])
    for i in range(d, d+2):
        for j in range(d):
            out[i, j] = ll[i-d, j]
        for j in range(d, d+2):
            out[i, j] = R[j-d, i-d]

@nb.jit(nopython=True)
def vec_build_xform_mat(B, Bc, R, U, s):
    d = s.shape[0]
    out = np.zeros((d+2, d+2))
    ll = np.dot(B, U)
    _btm_helper(s, d, ll, R, out)
    return out

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Helper Vectorized Numba Squared Sum Functions.
# 
# Sums along the 0th axis for input X of shape (D, N).
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
@nb.jit(nopython=True)
def _vec_sq_sum(X, R, N, D):
    for j in range(D):
        for i in range(N):
            R[i] += X[j, i] * X[j, i]

@nb.jit(nopython=True)
def _vec_argmax(X):
    D, N = X.shape
    R = np.zeros(N)
    _vec_sq_sum(X, R, N, D)
    return np.argmax(R)

@nb.jit(nopython=True)
def _get_recon_matrix(X, U):
    return np.dot(np.dot(U, U.T), X)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Fast Implementation of Anomaly Identification via Reconstruction Errors.
# 
# Used for DEMUD in utils.py.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def vec_find_anomaly(X, U):
    Xr = _get_recon_matrix(X, U)
    return _vec_argmax(X - Xr)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Fast Implementation of K-Nearest Neighbors.
#
# About 4x faster than pure NumPy implementation.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
@nb.jit(nopython=True)
def _kNN(X, Z, N, D):
    for i in range(N):
        for j in range(N):
            if j > i:
                for k in range(D):
                    Z[i, j] += (X[i, k] - X[j, k]) ** 2
            else:
                Z[i, j] = Z[j, i]
                    
@nb.jit(nopython=True)
def vec_kNN(X):
    N, D = X.shape
    Z = np.zeros((N, N))
    _kNN(X, Z, N, D)
    return Z


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Numba JIT Implementation of Nearest Neighbor Outlier Detection.
#
# Uses the average distance of kNN as the outlier score.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
@nb.jit(nopython=True)
def _average_dist_knn(S, V, K, N, tol):
    for i in range(N):
        for j in range(K):
            V[i] += S[i, j]
        for j in range(K):
            if S[i, j] - S[i, j-1] < tol:
                V[i] += S[i, j]
            else:
                V[i] /= j
                break
                
def vec_knn_anomaly(X, K, tol=1e-6):
    N, D = X.shape
    S = np.partition(vec_kNN(X), K, 1)
    V = np.zeros(N)
    _average_dist_knn(S, V, K+1, N, tol)
    return V


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Numba JIT Implementations of LOF Helper Functions.
#
# Computes Reachability Distance and Local Reachability Density.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
@nb.jit(nopython=True)
def _remove_self_index(U, V, M, N, K):
    """
    Subroutine to remove x from the set of x's nearest neighbors.
    
    Inputs:
        - U: Index matrix of shape (N, N).
        - V: Output matrix of shape (N, M).
        - M: Variable to cache the value N-1.
        - N: Number of data points.
        - K: The number of nearest neighbors.
    """
    for i in range(N):
        z = 0
        for j in range(K):
            if U[i, z] == i: z += 1
            V[i, j] = U[i, z]
            z += 1
        for j in range(K, M):
            V[i, j] = U[i, j+1]


@nb.jit(nopython=True)
def _compute_lrd_vector(V, S, M, N, K, L, Q, tol):
    """
    Subroutine to compute local reachability density for each
    data point.
    
    Inputs:
        - V  : Index matrix of nearest neighbors with shape (N, M).
        - S  : Similarity matrix of shape (N, N).
        - M  : Variable to cache the value N-1.
        - N  : Number of data points.
        - K  : The number of nearest neighbors.
        - L  : Local reachability density output vector of shape (N,).
        - Q  : Output vector with nearest neighbor set cardinalities
               for each data point. Has shape (N,).
        - tol: Tolerance measure for tie-breaking.
    """
    k = K-1
    for i in range(N):
        for j in range(K):
            b = V[i, j]
            L[i] += max(S[b, V[b, k]], S[i, b])
        for j in range(K, M):
            b = V[i, j]
            d = S[i, b]
            if d - S[i, V[i, j-1]] < tol:
                L[i] += max(S[b, V[b, k]], d)
            else:
                Q[i] = j
                break
        L[i] = Q[i]/L[i]


@nb.jit(nopython=True)
def _compute_lof_vector(V, Q, L, N, P):
    """
    Subroutine to compute local outlier factor for each data point.
    
    Inputs:
        - V  : Index matrix of nearest neighbors with shape (N, M).
        - Q  : Vector, of shape (N,), with nearest neighbor set 
               cardinalities for each data point.
        - L  : Vector, of shape (N,), with local reachability densities.
        - N  : Number of data points.
        - P  : Local outlier factor vector of shape (N,).
    """
    for i in range(N):
        for j in range(Q[i]):
            P[i] += L[V[i, j]]
        P[i] /= (Q[i] * L[i])


@nb.jit(nopython=True)
def _get_lof_vector(S, K, U, tol):
    """
    Subroutine to coordinate the computation of local outlier factor
    values for each data point.
    
    Inputs:
        - S  : Similarity matrix of shape (N, N).
        - K  : The number of nearest neighbors.
        - U  : Index matrix of shape (N, N).
        - tol: Tolerance measure for tie-breaking.
    
    Output:
        - P  : Vector of local outlier factors with shape (N,).
    """
    N       = U.shape[0]
    M       = N-1
    V       = np.empty((N, M), dtype=np.uint32)
    Q       = np.empty(N, dtype=np.uint32)
    L, P    = np.zeros(N), np.zeros(N)
    _remove_self_index(U, V, M, N, K)
    _compute_lrd_vector(V, S, M, N, K, L, Q, tol)
    _compute_lof_vector(V, Q, L, N, P)
    return P


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Fast Implementation of Local Outlier Factor.
#
# About 3x faster than Pure NumPy implementation.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def vec_lof(X, K, tol=1e-6):
    """
    Utility function to compute the Local Outlier Factor as proposed by
    Breunig, et. al. (2000).
    
    Inputs:
        - X  : Input dataset matrix of shape (N, D).
        - K  : The number of nearest neighbors.
        - tol: Tolerance measure for tie-breaking.
    
    Output: Vector of local outlier factors with shape (N,).
    """
    S = vec_kNN(X)
    U = np.argpartition(S, K, 1)
    return _get_lof_vector(S, K, U, tol)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Numba JIT Implementations of HBOS Helper Functions.
#
# Computes Bin Density and Feature Probabilities.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
@nb.jit(nopython=True)
def _bsearch_helper(b, x, z, N, B):
    for i in range(N):
        left, mid, right = 0, B/2, B
        while right > left:
            if x[i] < b[mid]: right = mid
            else: left = mid + 1
            mid = (left + right)/2
        z[i] = left-1

@nb.jit(nopython=True)
def _bin_search(b, X):
    N, B = X.shape[0], b.shape[0]
    Z = np.zeros(N, dtype=np.uint32)
    _bsearch_helper(b, X, Z, N, B)
    return Z

@nb.jit(nopython=True)
def _buildbins_helper(w, a, b, n):
    b[0] = a
    for i in range(1, n):
        b[i] = b[i-1] + w

@nb.jit(nopython=True)
def _build_bins(X, w):
    N = X.shape[0]
    mx, mn = np.max(X), np.min(X)
    bins = int((mx-mn)/w)
    b = np.zeros(bins)
    _buildbins_helper(w, mn, b, bins)
    return b

@nb.jit(nopython=True)
def _compute_hbos_scores(W, X, Y, N, D):
    for i in range(D):
        x, w = X[:, i], W[i]
        B = _build_bins(x, w)
        I = _bin_search(B, x)
        H = np.zeros(B.shape[0])
        for j in range(N):
            H[I[j]] += w
        for j in range(N):
            Y[j] *= 1/H[I[j]]

            
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Fast Implementation of Histogram Based Outlier Score.
#
# About 4x faster than Pure NumPy implementation.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def vec_hbos(X):
    N, D  = X.shape
    q1    = N/4
    q3    = 3*q1
    c     = 2.0/(N ** (1.0/3.0))
    Y     = np.ones(N)
    vfunc = np.vectorize(lambda i: np.partition(X[:,i], q3)[q3] - np.partition(X[:,i], q1)[q1])
    W     = vfunc(np.arange(D)) * c
    _compute_hbos_scores(W, X, Y, N, D)
    return Y


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Numba JIT Implementations of Gaussian NBOS Helper Functions.
#
# Computes Calculates Gaussian Probabilities for each Feature.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
@nb.jit(nopython=True)
def _calc_zscore(X, Y, M, S, N, D):
    for j in range(D):
        for i in range(N):
            M[j] += X[i, j]
        M[j] /= N
    for j in range(D):
        for i in range(N):
            Y[i, j] = X[i, j] - M[j]
            Y[i, j] *= Y[i, j]
            S[j] += Y[i, j]
        S[j] /= N
        S[j] *= 2
    for i in range(N):
        for j in range(D):
            Y[i, j] /= S[j]
            Y[i, j] = math.exp(-Y[i, j])
            Y[i, j] /= math.sqrt(math.pi * S[j])
            
@nb.jit(nopython=True)
def _get_gauss_probs(Y, P, N, D):
    for i in range(N):
        for j in range(D):
            P[i] *= Y[i, j]
            

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Fast Implementation of Gaussian Naive Bayes Outlier Score.
#
# About 5x faster than Pure NumPy implementation.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
@nb.jit(nopython=True)
def vec_nbos(X):
    N, D = X.shape
    Y = np.zeros_like(X)
    M, S = np.zeros(D), np.zeros(D)
    P = np.ones(N)
    _calc_zscore(X, Y, M, S, N, D)
    _get_gauss_probs(Y, P, N, D)
    return P