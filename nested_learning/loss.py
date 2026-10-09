"""Cross-entropy loss function with padding masking support in pure NumPy.

Reference:
- Behrouz et al., "Nested Learning: The Illusion of Deep Learning Architectures", NeurIPS 2025.
"""

from typing import Optional, Tuple
import numpy as np


class CrossEntropyLoss:
    """Categorical Cross-Entropy Loss with Softmax and padding mask support.

    Calculates:
        L = - (1 / N_valid) * sum_{i in valid} log(softmax(logits_i)[targets_i])

    Features:
    - Numerically stable log-softmax via max subtraction.
    - ignore_index to ignore padding tokens (e.g., pad_token_id).
    - Label smoothing support.
    """

    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0):
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

        self._cache: Optional[Tuple] = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Compute cross-entropy loss.

        Args:
            logits: Predicted logits [B, T, V] or [B, V].
            targets: Ground-truth target indices [B, T] or [B].

        Returns:
            Scalar loss value.
        """
        orig_shape = logits.shape
        V = orig_shape[-1]

        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1).astype(np.int64)

        # Numerically stable softmax
        max_logits = np.max(logits_flat, axis=-1, keepdims=True)
        exp_logits = np.exp(np.clip(logits_flat - max_logits, -35.0, 35.0))
        probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-12)

        valid_mask = (targets_flat != self.ignore_index)
        n_valid = int(np.sum(valid_mask))

        if n_valid == 0:
            self._cache = (probs, targets_flat, valid_mask, n_valid, orig_shape)
            return 0.0

        # Gather target probabilities
        valid_indices = np.where(valid_mask)[0]
        valid_targets = targets_flat[valid_indices]

        target_probs = probs[valid_indices, valid_targets]
        log_probs = np.log(np.clip(target_probs, 1e-12, 1.0))

        if self.label_smoothing > 0.0:
            smooth_loss = -np.mean(np.log(np.clip(probs[valid_indices], 1e-12, 1.0)))
            loss = (1.0 - self.label_smoothing) * (-np.mean(log_probs)) + self.label_smoothing * smooth_loss
        else:
            loss = float(-np.sum(log_probs) / n_valid)

        self._cache = (probs, targets_flat, valid_mask, n_valid, orig_shape)
        return float(loss)

    def backward(self) -> np.ndarray:
        """Compute analytical gradient of the cross-entropy loss w.r.t logits.

        Returns:
            Gradient tensor dlogits with the same shape as logits.
        """
        if self._cache is None:
            raise RuntimeError("CrossEntropyLoss.backward called before forward")

        probs, targets_flat, valid_mask, n_valid, orig_shape = self._cache
        V = orig_shape[-1]

        if n_valid == 0:
            return np.zeros(orig_shape, dtype=np.float32)

        dlogits_flat = np.zeros_like(probs, dtype=np.float32)
        valid_indices = np.where(valid_mask)[0]
        valid_targets = targets_flat[valid_indices]

        # For valid positions: dlogits = (probs - one_hot) / n_valid
        if self.label_smoothing > 0.0:
            smooth_target = self.label_smoothing / V
            one_hot_weight = 1.0 - self.label_smoothing

            dlogits_flat[valid_indices] = (probs[valid_indices] - smooth_target) / n_valid
            dlogits_flat[valid_indices, valid_targets] -= one_hot_weight / n_valid
        else:
            dlogits_flat[valid_indices] = probs[valid_indices] / n_valid
            dlogits_flat[valid_indices, valid_targets] -= 1.0 / n_valid

        return dlogits_flat.reshape(orig_shape).astype(np.float32)

    def __call__(self, logits: np.ndarray, targets: np.ndarray) -> float:
        return self.forward(logits, targets)
