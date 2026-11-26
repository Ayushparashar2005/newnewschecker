import numpy as np
from src.models.base import BaseModel

class LSTMClassifier(BaseModel):
    """
    Bidirectional / Unidirectional Long Short-Term Memory (LSTM) Recurrent Neural Network
    built from scratch using pure NumPy.
    
    Architecture:
      Input Integer Token Sequences (Batch, Seq_Len)
        -> Embedding Lookup Matrix (Vocab_Size, Embed_Dim)
        -> Forward LSTM (and optional Backward LSTM)
        -> Dense Linear Classification Head (Hidden_Dim [* 2] -> 1)
        -> Sigmoid Activation -> Probability of Fake News
    """
    def __init__(self, vocab_size=5000, embed_dim=48, hidden_dim=64,
                 lr=0.003, epochs=15, batch_size=32, bidirectional=True,
                 lambda_reg=0.0001, clip_norm=5.0, verbose=False):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.lambda_reg = lambda_reg
        self.clip_norm = clip_norm
        self.verbose = verbose

        self.loss_history = []
        self._is_fitted = False

        # Weight containers
        self.E = None  # Embeddings
        # Forward cell weights: [W_f, W_i, W_c, W_o] stacked into W_gates (embed_dim + hidden_dim, 4 * hidden_dim)
        self.W_fwd = None
        self.b_fwd = None
        # Backward cell weights (if bidirectional)
        self.W_bwd = None
        self.b_bwd = None
        # Dense classifier head
        self.W_clf = None
        self.b_clf = None

    def _init_weights(self):
        np.random.seed(42)
        H = self.hidden_dim
        D = self.embed_dim
        in_dim = D + H

        # He / Xavier style initialization
        self.E = np.random.randn(self.vocab_size, D).astype(np.float32) * 0.05

        # Forward LSTM weights
        limit_fwd = np.sqrt(6.0 / (in_dim + 4 * H))
        self.W_fwd = np.random.uniform(-limit_fwd, limit_fwd, (in_dim, 4 * H)).astype(np.float32)
        self.b_fwd = np.zeros((1, 4 * H), dtype=np.float32)
        # Initialize forget gate bias to 1.0 for improved gradient flow over long sequences
        self.b_fwd[0, :H] = 1.0

        if self.bidirectional:
            limit_bwd = np.sqrt(6.0 / (in_dim + 4 * H))
            self.W_bwd = np.random.uniform(-limit_bwd, limit_bwd, (in_dim, 4 * H)).astype(np.float32)
            self.b_bwd = np.zeros((1, 4 * H), dtype=np.float32)
            self.b_bwd[0, :H] = 1.0
            clf_in = 2 * H
        else:
            clf_in = H

        limit_clf = np.sqrt(6.0 / (clf_in + 1))
        self.W_clf = np.random.uniform(-limit_clf, limit_clf, (clf_in, 1)).astype(np.float32)
        self.b_clf = np.zeros((1, 1), dtype=np.float32)

    @staticmethod
    def _sigmoid(x):
        x_clipped = np.clip(x, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    @staticmethod
    def _tanh(x):
        return np.tanh(np.clip(x, -30.0, 30.0))

    def _forward_lstm_direction(self, X_emb, W, b, backward=False):
        """
        Runs one direction of the LSTM across all sequence timesteps.
        X_emb shape: (B, T, D)
        """
        B, T, D = X_emb.shape
        H = self.hidden_dim

        # Cache variables for BPTT
        h_seq = np.zeros((B, T, H), dtype=np.float32)
        c_seq = np.zeros((B, T, H), dtype=np.float32)
        f_seq = np.zeros((B, T, H), dtype=np.float32)
        i_seq = np.zeros((B, T, H), dtype=np.float32)
        c_tilde_seq = np.zeros((B, T, H), dtype=np.float32)
        o_seq = np.zeros((B, T, H), dtype=np.float32)
        u_seq = np.zeros((B, T, D + H), dtype=np.float32)

        h_prev = np.zeros((B, H), dtype=np.float32)
        c_prev = np.zeros((B, H), dtype=np.float32)

        time_indices = range(T - 1, -1, -1) if backward else range(T)

        for step_idx, t in enumerate(time_indices):
            x_t = X_emb[:, t, :]  # (B, D)
            u_t = np.hstack((x_t, h_prev))  # (B, D + H)
            u_seq[:, t, :] = u_t

            gates = np.dot(u_t, W) + b  # (B, 4H)
            f_t = self._sigmoid(gates[:, 0:H])
            i_t = self._sigmoid(gates[:, H:2*H])
            c_tilde_t = self._tanh(gates[:, 2*H:3*H])
            o_t = self._sigmoid(gates[:, 3*H:4*H])

            c_t = f_t * c_prev + i_t * c_tilde_t
            h_t = o_t * self._tanh(c_t)

            h_seq[:, t, :] = h_t
            c_seq[:, t, :] = c_t
            f_seq[:, t, :] = f_t
            i_seq[:, t, :] = i_t
            c_tilde_seq[:, t, :] = c_tilde_t
            o_seq[:, t, :] = o_t

            h_prev = h_t
            c_prev = c_t

        cache = {
            "h_seq": h_seq, "c_seq": c_seq,
            "f_seq": f_seq, "i_seq": i_seq,
            "c_tilde_seq": c_tilde_seq, "o_seq": o_seq,
            "u_seq": u_seq, "backward": backward
        }
        return h_prev, cache

    def _backward_lstm_direction(self, dh_last, cache, W, B, T, D, H):
        """
        Backpropagation Through Time (BPTT) for a single LSTM direction.
        """
        backward = cache["backward"]
        h_seq = cache["h_seq"]
        c_seq = cache["c_seq"]
        f_seq = cache["f_seq"]
        i_seq = cache["i_seq"]
        c_tilde_seq = cache["c_tilde_seq"]
        o_seq = cache["o_seq"]
        u_seq = cache["u_seq"]

        dW = np.zeros_like(W)
        db = np.zeros((1, 4 * H), dtype=np.float32)
        dX_emb = np.zeros((B, T, D), dtype=np.float32)

        dh_next = dh_last.copy()
        dc_next = np.zeros((B, H), dtype=np.float32)

        time_indices = range(0, T) if backward else range(T - 1, -1, -1)

        for t in time_indices:
            h_t = h_seq[:, t, :]
            c_t = c_seq[:, t, :]
            f_t = f_seq[:, t, :]
            i_t = i_seq[:, t, :]
            c_tilde_t = c_tilde_seq[:, t, :]
            o_t = o_seq[:, t, :]
            u_t = u_seq[:, t, :]

            # Previous c
            prev_t = t + 1 if backward else t - 1
            if (backward and prev_t < T) or (not backward and prev_t >= 0):
                c_prev = c_seq[:, prev_t, :]
            else:
                c_prev = np.zeros((B, H), dtype=np.float32)

            dh = dh_next
            tanh_c = np.tanh(c_t)
            do = dh * tanh_c * o_t * (1.0 - o_t)
            dc = dc_next + dh * o_t * (1.0 - tanh_c ** 2)

            dc_tilde = dc * i_t * (1.0 - c_tilde_t ** 2)
            di = dc * c_tilde_t * i_t * (1.0 - i_t)
            df = dc * c_prev * f_t * (1.0 - f_t)

            dc_next = dc * f_t

            d_gates = np.hstack((df, di, dc_tilde, do))  # (B, 4H)
            dW += np.dot(u_t.T, d_gates)
            db += np.sum(d_gates, axis=0, keepdims=True)

            du = np.dot(d_gates, W.T)
            dX_emb[:, t, :] = du[:, :D]
            dh_next = du[:, D:]

        return dW, db, dX_emb

    def fit(self, X, y):
        """
        Trains the LSTM model using mini-batch gradient descent with Adam optimizer.
        X: shape (N, T) token indices
        y: shape (N,) binary labels {0, 1}
        """
        X = np.asarray(X, dtype=np.int32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        n_samples, seq_len = X.shape

        # Initialize weights if not already done
        if not self._is_fitted or self.W_fwd is None:
            self._init_weights()

        H = self.hidden_dim
        D = self.embed_dim

        # Adam optimizer state variables
        m_params = [np.zeros_like(p) for p in self._get_params()]
        v_params = [np.zeros_like(p) for p in self._get_params()]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t_adam = 0

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

                # 1. Embedding lookup: (B, T) -> (B, T, D)
                X_emb = self.E[X_batch]

                # 2. Forward LSTM pass
                h_fwd_last, cache_fwd = self._forward_lstm_direction(X_emb, self.W_fwd, self.b_fwd, backward=False)

                if self.bidirectional:
                    h_bwd_last, cache_bwd = self._forward_lstm_direction(X_emb, self.W_bwd, self.b_bwd, backward=True)
                    h_final = np.hstack((h_fwd_last, h_bwd_last))
                else:
                    h_final = h_fwd_last

                # 3. Dense Classifier Head
                logits = np.dot(h_final, self.W_clf) + self.b_clf
                probs = self._sigmoid(logits)

                # Loss computation
                eps_clip = 1e-7
                bce_loss = -np.mean(y_batch * np.log(probs + eps_clip) + (1.0 - y_batch) * np.log(1.0 - probs + eps_clip))
                l2_loss = 0.5 * self.lambda_reg * (
                    np.sum(self.W_fwd ** 2) + np.sum(self.W_clf ** 2) +
                    (np.sum(self.W_bwd ** 2) if self.bidirectional else 0.0)
                )
                batch_loss = bce_loss + l2_loss
                epoch_loss += batch_loss * B

                # 4. Backward Pass
                d_logits = (probs - y_batch) / B  # (B, 1)
                dW_clf = np.dot(h_final.T, d_logits) + self.lambda_reg * self.W_clf
                db_clf = np.sum(d_logits, axis=0, keepdims=True)

                dh_final = np.dot(d_logits, self.W_clf.T)  # (B, clf_in)

                if self.bidirectional:
                    dh_fwd = dh_final[:, :H]
                    dh_bwd = dh_final[:, H:]
                    dW_bwd, db_bwd, dX_emb_bwd = self._backward_lstm_direction(
                        dh_bwd, cache_bwd, self.W_bwd, B, seq_len, D, H
                    )
                    dW_bwd += self.lambda_reg * self.W_bwd
                else:
                    dh_fwd = dh_final
                    dW_bwd, db_bwd, dX_emb_bwd = None, None, 0.0

                dW_fwd, db_fwd, dX_emb_fwd = self._backward_lstm_direction(
                    dh_fwd, cache_fwd, self.W_fwd, B, seq_len, D, H
                )
                dW_fwd += self.lambda_reg * self.W_fwd

                # Gradient for embeddings
                dX_emb = dX_emb_fwd + (dX_emb_bwd if self.bidirectional else 0.0)
                dE = np.zeros_like(self.E)
                np.add.at(dE, X_batch, dX_emb)

                # Collect gradients
                if self.bidirectional:
                    grads = [dE, dW_fwd, db_fwd, dW_bwd, db_bwd, dW_clf, db_clf]
                else:
                    grads = [dE, dW_fwd, db_fwd, dW_clf, db_clf]

                # Gradient clipping
                total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
                if total_norm > self.clip_norm:
                    scale = self.clip_norm / (total_norm + 1e-6)
                    grads = [g * scale for g in grads]

                # Adam parameter update
                t_adam += 1
                params = self._get_params()
                for i in range(len(params)):
                    m_params[i] = beta1 * m_params[i] + (1.0 - beta1) * grads[i]
                    v_params[i] = beta2 * v_params[i] + (1.0 - beta2) * (grads[i] ** 2)
                    m_hat = m_params[i] / (1.0 - beta1 ** t_adam)
                    v_hat = v_params[i] / (1.0 - beta2 ** t_adam)
                    params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

            avg_epoch_loss = epoch_loss / n_samples
            self.loss_history.append(float(avg_epoch_loss))
            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"LSTM Epoch {epoch+1}/{self.epochs} - BCE Loss: {avg_epoch_loss:.4f}")

        self._is_fitted = True
        return self

    def _get_params(self):
        if self.bidirectional:
            return [self.E, self.W_fwd, self.b_fwd, self.W_bwd, self.b_bwd, self.W_clf, self.b_clf]
        return [self.E, self.W_fwd, self.b_fwd, self.W_clf, self.b_clf]

    def predict_proba(self, X):
        """
        Returns probabilities of class 1 (Fake News) for input token sequences.
        """
        X = np.asarray(X, dtype=np.int32)
        X_emb = self.E[X]

        h_fwd_last, _ = self._forward_lstm_direction(X_emb, self.W_fwd, self.b_fwd, backward=False)
        if self.bidirectional:
            h_bwd_last, _ = self._forward_lstm_direction(X_emb, self.W_bwd, self.b_bwd, backward=True)
            h_final = np.hstack((h_fwd_last, h_bwd_last))
        else:
            h_final = h_fwd_last

        logits = np.dot(h_final, self.W_clf) + self.b_clf
        probs = self._sigmoid(logits)
        return probs.ravel()

    def predict(self, X, threshold=0.5):
        """
        Returns binary class predictions {0: Real, 1: Fake}.
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def partial_fit(self, X, y, lr_scale=0.3):
        """
        Online learning update from single or small batches of user feedback.
        """
        original_lr = self.lr
        original_epochs = self.epochs
        self.lr = original_lr * lr_scale
        self.epochs = 2
        self.fit(X, y)
        self.lr = original_lr
        self.epochs = original_epochs
        return self
