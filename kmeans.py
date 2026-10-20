"""Educational Lloyd K-means adapted from university coursework."""
import numpy as np


class MyKMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, random_state=42):
        self.k = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    @staticmethod
    def _assign(data, centroids, batch_size=256):
        # Batches bound the temporary (samples, clusters, features) array.
        labels = np.empty(len(data), dtype=int)
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            distances = np.linalg.norm(batch[:, None, :] - centroids, axis=2)
            labels[start:start + batch_size] = np.argmin(distances, axis=1)
        return labels

    def fit(self, data):
        # uint8 subtraction would wrap negative pixel differences.
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or not len(data) or not np.isfinite(data).all():
            raise ValueError("Expected a nonempty finite 2D array.")
        if not 1 <= self.k <= len(data) or self.max_iter < 1 or self.tol < 0:
            raise ValueError("Invalid K-means parameters.")
        rng = np.random.default_rng(self.random_state)
        self.centroids = data[rng.choice(len(data), self.k, replace=False)].copy()
        self.converged = False
        for iteration in range(self.max_iter):
            labels = self._assign(data, self.centroids)
            new_centroids = np.empty_like(self.centroids)
            for cluster in range(self.k):
                members = data[labels == cluster]
                # Reinitialize empty clusters rather than producing NaNs.
                new_centroids[cluster] = members.mean(axis=0) if len(members) else data[rng.integers(len(data))]
            movement = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            self.n_iter = iteration + 1
            if movement < self.tol:
                self.converged = True
                break
        self.labels = self._assign(data, self.centroids)
        return self

    def predict(self, new_data):
        if self.centroids is None:
            raise ValueError("Fit K-means before predicting clusters.")
        data = np.asarray(new_data, dtype=float)
        if data.ndim != 2 or data.shape[1] != self.centroids.shape[1] or not np.isfinite(data).all():
            raise ValueError("Expected finite data with the fitted feature count.")
        return self._assign(data, self.centroids)
