import numpy as np
from src.models.base import BaseModel

class KNNClassifier(BaseModel):
    def __init__(self, k=5, metric='cosine'):
        """
        K-Nearest Neighbors Classifier from scratch.
        metric: 'cosine' or 'euclidean'
        """
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Stores training samples. X must be a NumPy array.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def _k_nearest_labels(self, X):
        """
        Returns the labels of the k nearest neighbors for each sample in X.
        """
        X = np.array(X)
        all_neighbor_labels = []
        
        # Determine predictions in batches to prevent memory blowup
        batch_size = 500
        n_samples = X.shape[0]
        k_eff = min(self.k, len(self.X_train))
        
        for i in range(0, n_samples, batch_size):
            X_batch = X[i : i + batch_size]
            
            # Compute distance matrix of shape (batch_size, X_train.shape[0])
            if self.metric == 'cosine':
                # Cosine distance = 1 - cosine_similarity
                norm_batch = np.linalg.norm(X_batch, axis=1, keepdims=True)
                norm_train = np.linalg.norm(self.X_train, axis=1, keepdims=True).T
                
                # Avoid division by zero
                norm_batch[norm_batch == 0] = 1e-8
                norm_train[norm_train == 0] = 1e-8
                
                dot_product = np.dot(X_batch, self.X_train.T)
                similarity = dot_product / (norm_batch * norm_train)
                dists = 1.0 - similarity
            else:
                # Euclidean distance: sqrt(||x||^2 + ||y||^2 - 2 x.y^T)
                x2 = np.sum(X_batch ** 2, axis=1, keepdims=True)
                y2 = np.sum(self.X_train ** 2, axis=1, keepdims=True).T
                xy = np.dot(X_batch, self.X_train.T)
                
                dists = np.sqrt(np.clip(x2 + y2 - 2 * xy, 0, None))
                
            k_nearest_indices = np.argpartition(dists, k_eff - 1, axis=1)[:, :k_eff]
            
            for row_idx, indices in enumerate(k_nearest_indices):
                row_dists = dists[row_idx, indices]
                sorted_k_indices = indices[np.argsort(row_dists)]
                all_neighbor_labels.append(self.y_train[sorted_k_indices])
                
        return all_neighbor_labels

    def predict_proba(self, X):
        """
        Predicts probability of class 1 (Fake) based on neighbor vote fraction.
        """
        neighbor_labels_list = self._k_nearest_labels(X)
        return np.array([float(np.mean(labels == 1)) for labels in neighbor_labels_list])

    def predict(self, X):
        """
        Predicts classes for the test samples in X.
        """
        neighbor_labels_list = self._k_nearest_labels(X)
        predictions = []
        for neighbor_labels in neighbor_labels_list:
            counts = np.bincount(neighbor_labels.astype(int))
            predictions.append(np.argmax(counts))
        return np.array(predictions)
