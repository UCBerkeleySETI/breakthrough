DeepAnomaly
-----------

DeepAnomaly is a Python framework for anomaly and outlier detection
built upon Numba's JIT compilation backend.

As of v0.1, support for the following models and algorithms are included:

- Nearest Neighbor Based Techniques
    - k-NN Outlier Score
    - Local Outlier Factor

- Statistical Based Techniques
    - Histogram Based Outlier Score
    - Gaussian Kernel Outlier Score
    - Sigmoid Kernel Outlier Score

- Eigenbasis Subspace Techniques
    - DEMUD

- Deep Learning Based Techniques (TODO)
    - Autoencoder
    - Variational Autoencoder

- Data Preprocessing Tools
    - Regularlized PCA Whitening
    - Regularized ZCA Whitening

An example usage is as follows:

.. code:: python

    import numpy as np
    from deep_anomaly.models import LocalOutlierFactor

    # Generate some sample dataset.
    X = np.random.randn(1000, 100)

    # Initialize and execute model.
    model = LocalOutlierFactor()
    model.fit_transform(X, k=5)
    indices = model.get_indices(topk=25, ensemble=True)
    outliers = X[indices]
