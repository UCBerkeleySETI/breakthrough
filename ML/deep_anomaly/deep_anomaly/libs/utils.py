import numpy as np
import scipy.linalg as sp
import deep_anomaly.libs.vec_utils as vc

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Basic Data Whitening.
# 
# Assumptions:
#   1. Data can be captured by a Multivariate Gaussian distribution.
#   2. Data either has homoscedastic or heteroscedastic covariance i.e.
#      feature dimensions are independent Gaussian distribution.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def whiten_basic(X, eps=1e-5):
    return center_data(X)/(X.var(0) + eps)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Center the data across each feature dimension.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def center_data(X):
    return X - X.mean(0)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Helper function for PCA and ZCA whitening.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def _decompose_components(X, tol=0.99, eps=1e-6, check_finite=False):
    """
    Helper function for calculating covariance, performing eigendecomposition
    and principal componnent selection.
    
    """
    N, D  = X.shape
    X     = center_data(X)
    C     = np.dot(X.T, X)/N
    d, U  = sp.eigh(C, overwrite_a=True, check_finite=check_finite)
    k     = np.argmin((1 - (d.cumsum()/d.sum())) > tol)
    ratio = 1 - float(k)/D
    d, U  = d[k:], U[:, k:]
    
    return np.dot(X, U/np.sqrt(d + eps)), U, ratio

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Regularized PCA Data Whitening: Global Feature Extraction.
# 
# Assumptions:
#   1. Data can be captured by a Multivariate Gaussian distribution.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def reg_pca_whitening(X, tol=0.99, eps=1e-6, check_finite=False):
    """
    A fast NumPy implementation of Regularized PCA Whitening for dimensionality
    reduction and sphering the dataset.
    
    Inputs:
       X            : Data matrix of shape (N, D).
       tol          : Percentage for capturing data variance in X. Defaults to 99%.
       eps          : Regularization value. Defaults to 1e-6.
       check_finite : Verify if matrix contains NaN or inf. Defaults to False for
                      possibly faster execution.
    
    Outputs:
      - Whitened data matrix of shape (N, k)
      - Reduction ratio: 0 < r <= 1, where r = k/D.
    
    Execution Speed:
      - Performs about 10% faster than SVD implementation.
      - Performs about 100% faster than sklearn's PCA implementation.
    
    """
    out, _, ratio = _decompose_components(X, tol=0.99, eps=1e-6, 
                                          check_finite=False)
    return out, ratio


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Regularized ZCA Data Whitening: Local Feature Extraction.
# --> Also known as the Mahalanobis Transformation.
# 
# Assumptions:
#   1. Data can be captured by a Multivariate Gaussian distribution.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def reg_zca_whitening(X, tol=0.99, eps=1e-6, check_finite=False):
    """
    A fast NumPy implementation of Regularized ZCA Whitening for dimensionality
    reduction and unique spherization of the dataset as proposed by Bell and 
    Sejnowski in 1996.
    
    Inputs:
       X            : Data matrix of shape (N, D).
       tol          : Percentage for capturing data variance in X. Defaults to 99%.
       eps          : Regularization value. Defaults to 1e-6.
       check_finite : Verify if matrix contains NaN or inf. Defaults to False for
                      possibly faster execution.
    
    Outputs:
      - ZCA Whitened data matrix of shape (N, k)
      - Reduction ratio: 0 < r <= 1, where r = k/D.
    
    """
    out, U, ratio = _decompose_components(X, tol=0.99, eps=1e-6, 
                                          check_finite=False)
    return np.dot(out, U.T), ratio
        
        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# DEMUD: Discovery via Eigenbasis Modeling of Uninteresting Data
#
# Based on the implementation by Kiri et. al. (2013) using the
# incremental SVD update proposed by Ross et. al. (2008).
#
# Alternate Implementation using Hermitian Matrices: Lower Overhead & Faster.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def _pooled_mean_fast(mu, x, n):
    T      = float(n + 1)
    C      = np.sqrt(n/T)
    mu_out = ((n * mu) + x)/T
    B      = vc.pad_top(x, mu, C)
    
    return mu_out, B


def _orthogonalize(X, overwrite=False, check_finite=False):
    return sp.qr(X, mode='economic', overwrite_a=overwrite, check_finite=check_finite)


def _build_transform_matrix(B, Bc, Q, R, U, d):
    uH, rW = d.shape[0], Q.shape[1]
    upper  = np.hstack((np.diag(np.sqrt(d)), np.zeros((uH, rW))))
    lower  = np.hstack((np.dot(B, U), R.T))
    
    return np.vstack((upper, lower))


def _incremental_svd_fast(U, mu, x, n, d, tol, k):
    mu, B    = _pooled_mean_fast(mu, x, n)
    Bc       = B - np.dot(B.dot(U), U.T)
    Q, R     = _orthogonalize(Bc.T)
    R        = vc.vec_build_xform_mat(B, Bc, R, U, d)
    Uc, _, d = _batch_svd_fast(R, tol, max(k-x.shape[1], -n-1))
    U        = np.dot(np.hstack((U, Q)), Uc)
    
    return U, np.maximum(0, d), mu


def _batch_svd_fast(X, tol, k=None, check_finite=False):
    d, U = sp.eigh(X.T.dot(X), overwrite_a=True, check_finite=check_finite)
    k    = np.argmin((1 - (d.cumsum()/d.sum())) > tol) if k is None else k
    return U[:, k:], k, d[k:]


def _extract_anomaly_fast(X, U, mu):
    Xc = (X-mu).T
    j = vc.vec_find_anomaly(Xc, U)
    return X[j].reshape(1, X.shape[1]), np.vstack((X[:j], X[j+1:]))
    

def demudify_fast(X, tol=0.99, u_lim=10, check_finite=False):
    N, D    = X.shape
    mu      = X.mean(0)
    Xf      = np.empty((u_lim, D))
    
    U, k, _ = _batch_svd_fast(X, tol)
    Xu, X   = _extract_anomaly_fast(X, U, mu)
    
    U, _, d = _batch_svd_fast(Xu, tol, -1)
    mu      = Xu
    Xf[0]   = Xu
    
    for i in xrange(1, u_lim):
        x, X     = _extract_anomaly_fast(X, U, mu)
        Xf[i]    = x
        U, d, mu = _incremental_svd_fast(U, mu, x, i, d, tol, k)
        
    return Xf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# DEMUD: Discovery via Eigenbasis Modeling of Uninteresting Data
#
# Based on the implementation by Kiri et. al. (2013) using the
# incremental SVD update proposed by Ross et. al. (2008).
#
# Optimized Original Implementation.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def _pooled_mean(mu, x, n):
    T      = float(n + 1)
    C      = np.sqrt(n/T)
    mu_out = ((n * mu) + x)/T
    B      = np.hstack((np.zeros((x.shape[0], 1)), C * (x - mu)))
    
    return mu_out, B


def _build_transform_matrix_tp(B, Bc, Q, U, d):
    uW, lH = d.shape[0], Q.shape[1]
    upper  = np.hstack((np.diag(d), np.dot(U.T, B)))
    lower  = np.hstack((np.zeros((lH, uW)), np.dot(Q.T, Bc)))
    
    return np.vstack((upper, lower))


def _incremental_svd(U, mu, x, n, d, tol, k):
    mu, B    = _pooled_mean(mu, x, n)
    Bc       = B - np.dot(U.dot(U.T), B)
    Q, _     = _orthogonalize(Bc)
    R        = _build_transform_matrix_tp(B, Bc, Q, U, d)
    Uc, _, d = _batch_svd(R, tol, min(k, n+1))
    U        = np.dot(np.hstack((U, Q)), Uc)
    
    return U, d, mu


def _batch_svd(X, tol, k=None, check_finite=False):
    U, d, _ = sp.svd(X, full_matrices=False, check_finite=check_finite)
    ds      = np.square(d)
    k       = np.argmax((ds.cumsum()/ds.sum()) > tol) + 1 if k is None else k
    return U[:, :k], k, d[:k]


def _extract_anomaly(X, U, mu):
    Xr = np.dot(U.dot(U.T), X-mu) + mu
    j  = np.argmax(np.square(X-Xr).sum(0))
    return X[:, j].reshape(X.shape[0], 1), np.delete(X, j, 1)
    

def demudify(X, tol=0.99, u_lim=10, check_finite=False):
    X = X.T
    D, N    = X.shape
    mu      = X.mean(1, keepdims=True)
    
    U, k, _ = _batch_svd(X, tol)
    Xu, X   = _extract_anomaly(X, U, mu)
    
    U, _, d = _batch_svd(Xu, tol, 1)
    mu      = Xu
    
    for i in xrange(1, u_lim):
        x, X     = _extract_anomaly(X, U, mu)
        Xu       = np.hstack((Xu, x))
        U, d, mu = _incremental_svd(U, mu, x, i, d, tol, k)
        
    return Xu