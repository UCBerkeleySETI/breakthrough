import numpy as np
from deep_anomaly.libs import vec_utils as vc

class NearestNeighborAnomaly(object):
    """
    Super class for nearest neighbor anomaly detection algorithms.
    """
    
    def __init__(self, tol=1e-6, dtype=np.float64):
        """
        Constructor method to initialize model parameters.
        
        Inputs:
            - tol  : Tolerance for tie-breaking. Defaults to 1e-6.
            - dtype: Datatype of data matrix. Defaults to 64-bit floats.
        """
        # Book-keeping and Useful Parameters.
        self._tol = tol
        self._dtype = dtype
        self._params = {'tolerance': tol, 'dtype': dtype}
        self._scores = None
        
        # Compile Numba Functions.
        self._compile()
        
        
    def _compile(self):
        raise NotImplementedError('Subclass needs to override compilation mechanism')
        
        
    def get_params(self):
        """
        Returns the model's parameters.
        """
        return self._params
    
    
    def set_params(self, new_tol, new_dtype):
        """
        Updates the model's initialization parameters.
        
        Inputs:
            - new_tol  : Tolerance for tie-breaking.
            - new_dtype: Datatype of data matrix.
        """
        self._tol = new_tol
        self._params['tolerance'] = new_tol
        
        if new_dtype is not self._dtype:
            self._dtype = new_dtype
            self._params['dtype'] = new_dtype
            self._compile()
            
            
    def fit_transform(self, X, y=None, k=5, func=None):
        """
        Run the anomaly model on the data matrix X.
        
        Inputs:
            - X        : Input 2D data matrix of shape (N, D).
            - y        : Class labels with shape (N, ) for evaluating per-class anomaly 
                         coverage.
            - k        : A list of ints or an int for k-nearest neighbors. A
                         list implies that a separate score vector is computed for
                         each k.
        """     
        # Initializations.
        scores = {}
        k_vals = list(k) if isinstance(k, list) or isinstance(k, tuple) else [k]
        
        # Assertions.
        min_k = min(k_vals)
        N, D  = X.shape
        assert min_k > 0 and min_k < N-1, 'Given value of k not within 0 < k < N-1.'
        assert N > 2 and D > 0, 'Input data dimensions not compatible. Need N > 2 and D > 0.'
        assert func is not None, 'Anomaly algorithm not provided.'
        if y is not None: 
            assert y.shape == (N,), 'Class labels dimensions mismatch.'
        
        # Run given anomaly function on data matrix.
        X = X.astype(self._dtype)
        self._params['data'] = X
        self._params['labels'] = y.astype(np.uint32)
        self._params['k_values'] = k_vals
        
        for k in k_vals:
            scores[k] = func(X, k, self._tol)
            
        self._scores = scores
        
        
    def get_scores(self, normalized=False):
        """
        Returns the dictionary of score vectors.
        
        Inputs:
            - normalized: Normalize the scores to standard gaussian distribution.
        
        Output: A dictionary of scores with k values supplied during fit_transform as
                keys and score vectors as values for each corresponding value of k.
        """
        scores = {}
        
        if normalized:
            for k, v in self._scores.iteritems():
                scores[k] = (v - v.mean())/v.std()
        else:
            for k, v in self._scores.iteritems():
                scores[k] = v.copy()
                
        return scores
    
    
    def get_indices(self, selection='auto', topk=50, ensemble=False, with_coverage=False):
        """
        Returns the dictionary of indices identified as anomalies.
        
        Inputs:
            - selection : [ 'auto' | 'topk' ].
            - topk      : Integer value for the top k highest anomaly scores. Ignored
                          when selection is 'auto'.
            - ensemble  : When True, a bootstrapped index vector is outputted.
            
        Output: A dictionary of indexes with k values supplied during fit_transform as
                keys and index vectors as values for each corresponding value of k.
        """
        N = self.params['data'].shape[0]
        K = len(self.params['k_values'])
        
        indices   = None
        normalize = lambda x   : (x - x.mean())/x.std()
        extract   = lambda x   : np.nonzero(x > np.abs(x.min()))
        partition = lambda x, k: np.argpartition(-x, k-1)[:k]
        
        if selection is 'topk':
            assert topk < N-1 and topk > 0, 'Value of topk not within range: 0 < topk < N-1.'
            indices = {k: partition(v, topk) for k, v in self._scores.iteritems()}
        elif selection is 'auto':
            indices = {k: extract(normalize(v)) for k, v in self._scores.iteritems()}
        else:
            raise AttributeError('Invalid argument for selection!')
            
        if ensemble:
            idxs = np.zeros(N, dtype=uint32)
            for v in indices.values():
                idxs += np.bincount(v, minlength=N)
            indices = idxs[idxs > (K/2)]
            
        if with_coverage:
            y   = self._params['labels']
            C   = y.max()
            R   = np.bincount(y, minlength=C).astype(np.float32)
            cov = lambda x: np.bincount(y[x], minlength=C).astype(np.float32)/R
            if ensemble:
                return indices, cov(indices)
            else:
                coverage = {k: cov(v) for k, v in indices.iteritems()}
                return indices, coverage
        else:
            return indices
    

class LocalOutlierFactor(NearestNeighborAnomaly):
    """
    An implementation of the Local Outlier Factor algorithm as proposed by 
    Breunig, et. al. (2000).
    
    Note:
        - The numba-based backend functions are compiled upon initialization.
        - The input parameters should be constrained as follows:
          [ N > 2 | D > 0 | 0 < K < N-1 ]
          where (N, D) is the shape of the input data matrix and K is
          the number of nearest neighbors.
    """
    
    def __init__(self, tol=1e-6, dtype=np.float64):
        super(LocalOutlierFactor, self).__init__(tol=tol, dtype=dtype)
        
        
    def _compile(self):
        """
        Internal: Compile numba-based backend functions with provided datatype.
        """
        compile_data = np.random.randn(3, 1).astype(self._dtype)
        _ = vc.vec_lof(compile_data, 1)
    
    
    def fit_transform(self, X, y=None, k=5):
        """
        Run the local outlier factor model on the data matrix X.
        
        See NearestNeighborAnomaly.fit_transform() for more info.
        """     
        super(LocalOutlierFactor, self).fit_transform(X, y=y, k=k, func=vc.vec_lof)
        
        
class KNNAnomaly(NearestNeighborAnomaly):
    """
    An implementation of the k-nearest neighbor outlier algorithm.
    
    Note:
        - The numba-based backend functions are compiled upon initialization.
        - The input parameters should be constrained as follows:
          [ N > 2 | D > 0 | 0 < K < N-1 ]
          where (N, D) is the shape of the input data matrix and K is
          the number of nearest neighbors.
    """
    
    def __init__(self, tol=1e-6, dtype=np.float64):
        super(KNNAnomaly, self).__init__(tol=tol, dtype=dtype)
        
        
    def _compile(self):
        """
        Internal: Compile numba-based backend functions with provided datatype.
        """
        compile_data = np.random.randn(3, 1).astype(self._dtype)
        _ = vc.vec_knn_anomaly(compile_data, 1)
    
    
    def fit_transform(self, X, y=None, k=5):
        """
        Run the k-nearest neighbor anomaly model on the data matrix X.
        
        See NearestNeighborAnomaly.fit_transform() for more info.
        """     
        super(KNNAnomaly, self).fit_transform(X, y=y, k=k, func=vc.vec_knn_anomaly)