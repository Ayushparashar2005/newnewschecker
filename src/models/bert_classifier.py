import numpy as np
import os
import pickle
from src.models.base import BaseModel

# Check transformers and torch availability
_TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

class DistilBertFeatureExtractor:
    """
    Singleton wrapper for the pre-trained DistilBERT model.
    Extracts 768-dimensional contextual [CLS] embedding representations for input texts.
    The transformer encoder weights remain frozen.
    """
    _tokenizer = None
    _model = None
    _model_name = "distilbert-base-uncased"

    @classmethod
    def load(cls):
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError("torch and transformers must be installed to use DistilBERT.")
        if cls._tokenizer is None or cls._model is None:
            cls._tokenizer = AutoTokenizer.from_pretrained(cls._model_name)
            cls._model = AutoModel.from_pretrained(cls._model_name)
            cls._model.eval()

    @classmethod
    def extract_embeddings(cls, texts, max_length=256, batch_size=16):
        """
        Tokenizes and extracts [CLS] token representations on CPU in mini-batches.
        Returns: NumPy array of shape (N, 768)
        """
        cls.load()
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        n_samples = len(texts)

        for i in range(0, n_samples, batch_size):
            batch_texts = [str(t) if str(t).strip() else "[EMPTY]" for t in texts[i:i + batch_size]]
            inputs = cls._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            with torch.no_grad():
                outputs = cls._model(**inputs)
                # Use [CLS] representation (token index 0)
                cls_reps = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_reps)

        return np.vstack(all_embeddings).astype(np.float32)


class BERTClassifier(BaseModel):
    """
    Hybrid Deep Classifier:
      Frozen Pretrained DistilBERT contextual encoder (768 dims)
      + Custom Pure NumPy Multi-Layer Classification Head (768 -> hidden_dim -> 1)
      
    This delivers frontier transformer representations while maintaining the project's
    transparent algorithmic philosophy (NumPy trained classification head with zero sklearn).
    """
    def __init__(self, hidden_dim=128, lr=0.001, epochs=30, batch_size=32,
                 lambda_reg=0.0001, max_token_len=256, verbose=False):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.max_token_len = max_token_len
        self.verbose = verbose

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        self.loss_history = []
        self._is_fitted = False

    def _init_weights(self, in_features=768):
        np.random.seed(42)
        # He initialization for ReLU hidden layer
        self.W1 = np.random.randn(in_features, self.hidden_dim).astype(np.float32) * np.sqrt(2.0 / in_features)
        self.b1 = np.zeros((1, self.hidden_dim), dtype=np.float32)

        # Xavier initialization for Sigmoid output
        self.W2 = np.random.randn(self.hidden_dim, 1).astype(np.float32) * np.sqrt(2.0 / self.hidden_dim)
        self.b2 = np.zeros((1, 1), dtype=np.float32)

    @staticmethod
    def _sigmoid(z):
        z_clip = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z_clip))

    @staticmethod
    def _relu(z):
        return np.maximum(0.0, z)

    def fit_from_embeddings(self, X_emb, y):
        """
        Trains the NumPy classification head directly on precomputed 768-dim embeddings.
        """
        X = np.asarray(X_emb, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        n_samples, n_features = X.shape

        if self.W1 is None or not self._is_fitted:
            self._init_weights(in_features=n_features)

        # Adam optimizer state
        m_W1, v_W1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        m_b1, v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        m_W2, v_W2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        m_b2, v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t_step = 0

        self.loss_history = []

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            num_batches = int(np.ceil(n_samples / self.batch_size))

            for b in range(num_batches):
                start = b * self.batch_size
                end = min(start + self.batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                B = X_batch.shape[0]

                # Forward pass
                z1 = np.dot(X_batch, self.W1) + self.b1
                a1 = self._relu(z1)

                z2 = np.dot(a1, self.W2) + self.b2
                probs = self._sigmoid(z2)

                # Loss
                eps_clip = 1e-7
                bce = -np.mean(y_batch * np.log(probs + eps_clip) + (1.0 - y_batch) * np.log(1.0 - probs + eps_clip))
                l2 = 0.5 * self.lambda_reg * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
                epoch_loss += (bce + l2) * B

                # Backpropagation
                dz2 = (probs - y_batch) / B
                dW2 = np.dot(a1.T, dz2) + self.lambda_reg * self.W2
                db2 = np.sum(dz2, axis=0, keepdims=True)

                da1 = np.dot(dz2, self.W2.T)
                dz1 = da1 * (z1 > 0).astype(np.float32)
                dW1 = np.dot(X_batch.T, dz1) + self.lambda_reg * self.W1
                db1 = np.sum(dz1, axis=0, keepdims=True)

                # Adam updates
                t_step += 1
                for param, grad, m, v in [
                    (self.W1, dW1, m_W1, v_W1),
                    (self.b1, db1, m_b1, v_b1),
                    (self.W2, dW2, m_W2, v_W2),
                    (self.b2, db2, m_b2, v_b2)
                ]:
                    m[:] = beta1 * m + (1.0 - beta1) * grad
                    v[:] = beta2 * v + (1.0 - beta2) * (grad ** 2)
                    m_hat = m / (1.0 - beta1 ** t_step)
                    v_hat = v / (1.0 - beta2 ** t_step)
                    param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

            avg_loss = epoch_loss / n_samples
            self.loss_history.append(float(avg_loss))
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"DistilBERT Head Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")

        self._is_fitted = True
        return self

    def fit(self, texts, y):
        """
        Extracts DistilBERT contextual embeddings and fits the NumPy classification head.
        """
        X_emb = DistilBertFeatureExtractor.extract_embeddings(texts, max_length=self.max_token_len)
        return self.fit_from_embeddings(X_emb, y)

    def predict_proba(self, texts):
        """
        Predicts probabilities of Fake News (class 1) for input raw text strings.
        """
        if not self._is_fitted or self.W1 is None:
            raise RuntimeError("BERTClassifier head is not fitted.")

        # If already an embedding array
        if isinstance(texts, np.ndarray) and texts.ndim == 2 and texts.shape[1] == 768:
            X_emb = texts
        else:
            X_emb = DistilBertFeatureExtractor.extract_embeddings(texts, max_length=self.max_token_len)

        z1 = np.dot(X_emb, self.W1) + self.b1
        a1 = self._relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        probs = self._sigmoid(z2)
        return probs.ravel()

    def predict(self, texts, threshold=0.5):
        probs = self.predict_proba(texts)
        return (probs >= threshold).astype(int)

    def partial_fit(self, texts, y, lr_scale=0.3):
        """
        Online adaptation for user feedback.
        """
        X_emb = DistilBertFeatureExtractor.extract_embeddings(texts, max_length=self.max_token_len)
        original_lr = self.lr
        original_epochs = self.epochs
        self.lr = original_lr * lr_scale
        self.epochs = 3
        self.fit_from_embeddings(X_emb, y)
        self.lr = original_lr
        self.epochs = original_epochs
        return self

    def save_head(self, filepath):
        """
        Saves only the lightweight NumPy classification head weights (~100 KB).
        """
        weights = {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "hidden_dim": self.hidden_dim,
            "loss_history": self.loss_history
        }
        with open(filepath, "wb") as f:
            pickle.dump(weights, f)

    def load_head(self, filepath):
        """
        Loads the NumPy classification head weights from disk.
        """
        with open(filepath, "rb") as f:
            weights = pickle.load(f)
        self.W1 = weights["W1"]
        self.b1 = weights["b1"]
        self.W2 = weights["W2"]
        self.b2 = weights["b2"]
        self.hidden_dim = weights.get("hidden_dim", 128)
        self.loss_history = weights.get("loss_history", [])
        self._is_fitted = True
        return self
