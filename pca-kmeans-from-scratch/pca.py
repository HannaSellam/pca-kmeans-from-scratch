"""Educational PCA adapted from university coursework."""
import numpy as np


class MyPCA:
    def __init__(self, n_components, standardize=False):
        self.n_components = n_components
        self.standardize = standardize
        self.components = None

    def fit(self, data):
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[0] < 2 or not np.isfinite(data).all():
            raise ValueError("Expected a finite 2D array with at least two rows.")
        if not 1 <= self.n_components <= data.shape[1]:
            raise ValueError("Invalid number of components.")
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        self.std[self.std == 0] = 1.0
        prepared = data - self.mean
        if self.standardize:
            prepared = prepared / self.std
        covariance = np.dot(prepared.T, prepared) / (len(data) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        # Largest eigenvalues first: prefixes now retain the leading components.
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        self.total_variance = eigenvalues.sum()
        if self.total_variance == 0:
            raise ValueError("PCA requires nonconstant data.")
        self.explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio = self.explained_variance / self.total_variance
        self.components = eigenvectors[:, order[:self.n_components]].T
        return self

    def transform(self, data):
        if self.components is None:
            raise ValueError("Fit PCA before transforming data.")
        prepared = np.asarray(data, dtype=float) - self.mean
        if self.standardize:
            prepared = prepared / self.std
        return np.dot(prepared, self.components.T)

    def inverse_transform(self, projected):
        if self.components is None:
            raise ValueError("Fit PCA before reconstructing data.")
        reconstructed = np.dot(np.asarray(projected, dtype=float), self.components)
        if self.standardize:
            reconstructed = reconstructed * self.std
        return reconstructed + self.mean
