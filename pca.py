import numpy as np
from numpy.linalg import eig


class PCA:
    def __init__(self, n_component):
        self.n_component = n_component

    def mean(self, X):
        mean = np.sum(X, axis=0) / X.shape[0]
        return mean

    def std(self, X):
        std = np.sum(np.square(X - self.mean(X)), axis=0) / (X.shape[0] - 1)
        return np.sqrt(std)

    def Standardize_data(self, X):
        X_std = (X - self.mean(X)) / self.std(X)
        return X_std

    def covariance(self, X):
        cov = (X.T @ X) / (X.shape[0] - 1)

        return cov

    def fit(self, X):
        # standardize data
        mean = self.mean(X)
        std = self.std(X)
        X_std = self.Standardize_data(X)

        # eigenv decomposition of cov matrix
        cov = self.covariance(X)
        eigen_values, eigen_vectors = eig(cov)

        # rank the eigenvalues and their associated eigenvectors
        # in decreasing order
        idx = np.array([np.abs(i) for i in eigen_values]).argsort()[::-1]
        eigen_values_sorted = eigen_values[idx]
        self.eigen_vectors_sorted = eigen_vectors.T[:, idx]

        explained_variance = [
            (i / sum(eigen_values)) * 100 for i in eigen_values_sorted
        ]
        explained_variance = np.round(explained_variance, 2)
        cum_explained_variance = np.cumsum(explained_variance)

    def transform(self, X):
        P = self.eigen_vectors_sorted[: self.n_component, :]  # Projection matrix

        X_std = self.Standardize_data(X)

        X_proj = X_std.dot(P.T)
        # X_proj.shape
        return X_proj

