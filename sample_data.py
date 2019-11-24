
import scipy.stats as stats
import numpy as np

class MultVarGaussianDataGenerator():
    def __init__(self, centroids:list, cov:np.ndarray, weights:list=None, k=None, d=None ):
        assert isinstance(centroids, list) or isinstance(centroids, np.ndarray)
        assert isinstance(weights, list) or isinstance(weights, np.ndarray)
        assert isinstance(cov, list) or isinstance(cov, np.ndarray)
        cov = np.array(cov)
        if k is None:
            k = len(centroids)
        if d is None:
            d = len(centroids[0])
        assert len(centroids) == k, "Need k centroids"
        if weights is None:
            weights = np.ones([k]) * 1/k
        assert cov.shape == (k, d, d)
        assert len(weights) == k, "Need k weights for centroids"
        assert all([isinstance(c, list) or isinstance(c, np.ndarray) for c in centroids])
        assert all([len(c)==d for c in centroids])
        self.centroids = centroids
        self.weights = weights
        self.k = k
        self.d = d
        self.cov = cov


    def sample(self, N=100):
        # sample centroid assignment
        C = np.random.choice(range(self.k), size=(N,), p=self.weights)
        # ...
        samples = np.zeros((N,self.d))
        for i, c in enumerate(C):
            rv = stats.multivariate_normal(mean=self.centroids[c], cov=self.cov[c])
            samples[i] = rv.rvs(size=1)

        return samples, C

