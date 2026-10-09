"""Nested Learning Lite: Core Neural Network Layers (Pure NumPy).

Reference:
- Behrouz et al., "Nested Learning: The Illusion of Deep Learning Architectures", NeurIPS 2025.
- https://github.com/obekt/HOPE-nested-learning
- https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/
"""

import math
from typing import List, Optional, Tuple, Union
import numpy as np


class Parameter:
    """Trainable array parameter with gradient tracking."""

    def __init__(self, data: np.ndarray, name: str = ""):
        self.data: np.ndarray = np.asarray(data, dtype=np.float32)
        self.grad: np.ndarray = np.zeros_like(self.data, dtype=np.float32)
        self.name: str = name

    def zero_grad(self) -> None:
        self.grad.fill(0.0)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def __repr__(self) -> str:
        return f"Parameter(name='{self.name}', shape={self.data.shape}, dtype={self.data.dtype})"


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x_clip = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


class Module:
    """Base class for all neural network modules."""

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self) -> List[Parameter]:
        """Collect all trainable parameters in this module and submodules."""
        params = []
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            val = getattr(self, attr_name)
            if isinstance(val, Parameter):
                params.append(val)
            elif isinstance(val, Module):
                params.extend(val.parameters())
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Parameter):
                        params.append(item)
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    """Affine linear transformation: y = x @ W + b."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, name: str = "linear"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        # Xavier Uniform initialization
        limit = math.sqrt(6.0 / (in_features + out_features))
        w_init = np.random.uniform(-limit, limit, size=(in_features, out_features)).astype(np.float32)
        self.weight = Parameter(w_init, name=f"{name}.weight")

        if bias:
            b_init = np.zeros(out_features, dtype=np.float32)
            self.bias = Parameter(b_init, name=f"{name}.bias")
        else:
            self.bias = None

        self._cache_x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache_x = x
        orig_shape = x.shape
        if x.ndim > 2:
            x_flat = x.reshape(-1, self.in_features)
            out = np.matmul(x_flat, self.weight.data)
            if self.bias is not None:
                out = out + self.bias.data
            return out.reshape(orig_shape[:-1] + (self.out_features,))
        else:
            out = np.matmul(x, self.weight.data)
            if self.bias is not None:
                out = out + self.bias.data
            return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self._cache_x
        if x is None:
            raise RuntimeError("Linear.backward called before forward")

        x_orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        dout_flat = dout.reshape(-1, self.out_features)

        self.weight.grad += np.matmul(x_flat.T, dout_flat)
        if self.bias is not None:
            self.bias.grad += np.sum(dout_flat, axis=0)

        dx_flat = np.matmul(dout_flat, self.weight.data.T)
        return dx_flat.reshape(x_orig_shape)


class Embedding(Module):
    """Lookup table for token embeddings."""

    def __init__(self, num_embeddings: int, embedding_dim: int, name: str = "embedding"):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Standard normal initialization (mean 0.0, std 0.02)
        w_init = (np.random.randn(num_embeddings, embedding_dim) * 0.02).astype(np.float32)
        self.weight = Parameter(w_init, name=f"{name}.weight")
        self._cache_indices: Optional[np.ndarray] = None

    def forward(self, indices: np.ndarray) -> np.ndarray:
        self._cache_indices = indices.astype(np.int64)
        return self.weight.data[self._cache_indices]

    def backward(self, dout: np.ndarray) -> None:
        indices = self._cache_indices
        if indices is None:
            raise RuntimeError("Embedding.backward called before forward")

        flat_idx = indices.reshape(-1)
        flat_dout = dout.reshape(-1, self.embedding_dim)
        np.add.at(self.weight.grad, flat_idx, flat_dout)
        return None


class LayerNorm(Module):
    """Layer Normalization over the last feature dimension."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, name: str = "layernorm"):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = Parameter(np.ones(normalized_shape, dtype=np.float32), name=f"{name}.gamma")
        self.beta = Parameter(np.zeros(normalized_shape, dtype=np.float32), name=f"{name}.beta")
        self._cache: Optional[Tuple] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x - mean) * std_inv
        out = self.gamma.data * x_hat + self.beta.data
        self._cache = (x, x_hat, mean, var, std_inv)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("LayerNorm.backward called before forward")
        x, x_hat, mean, var, std_inv = self._cache
        D = self.normalized_shape

        # Gradient w.r.t gamma and beta
        reduction_axes = tuple(range(dout.ndim - 1))
        self.gamma.grad += np.sum(dout * x_hat, axis=reduction_axes)
        self.beta.grad += np.sum(dout, axis=reduction_axes)

        # Gradient w.r.t x
        dx_hat = dout * self.gamma.data
        dx = (1.0 / D) * std_inv * (
            D * dx_hat
            - np.sum(dx_hat, axis=-1, keepdims=True)
            - x_hat * np.sum(dx_hat * x_hat, axis=-1, keepdims=True)
        )
        return dx


class GELU(Module):
    """Gaussian Error Linear Unit (GELU) with standard tanh approximation."""

    def __init__(self):
        super().__init__()
        self._cache: Optional[Tuple] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        s = math.sqrt(2.0 / math.pi)
        u = s * (x + 0.044715 * (x ** 3))
        tanh_u = np.tanh(u)
        out = 0.5 * x * (1.0 + tanh_u)
        self._cache = (x, u, tanh_u, s)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("GELU.backward called before forward")
        x, u, tanh_u, s = self._cache
        du_dx = s * (1.0 + 3.0 * 0.044715 * (x ** 2))
        dx = dout * (0.5 * (1.0 + tanh_u) + 0.5 * x * (1.0 - tanh_u ** 2) * du_dx)
        return dx


class SelfModifyingLayer(Module):
    """Fast-Weight Memory Layer trained by an inner-loop Delta Rule.

    In the Nested Learning paradigm (Behrouz et al., NeurIPS 2025):
    The memory matrix M acts as an online associative memory that performs
    one step of gradient descent per token on the reconstruction loss:
        L_inner = 0.5 * ||k_t @ M_{t-1} - v_t||^2

    The recurrence is governed by the gated Delta Rule:
        M_t = alpha_t * M_{t-1} + beta_t * k_t^T @ (v_t - k_t @ M_{t-1})
        o_t = q_t @ M_{t-1}   (read-out before write at step t)

    Properties:
    - alpha_t: learned forget/retention gate (sigmoid(W_alpha @ x + b_alpha + 4.0))
               initialized near 1.0 (retain existing memories).
    - beta_t:  learned inner learning rate (sigmoid(W_beta @ x + b_beta - 2.0))
               initialized small for gentle, stable inner-loop writes.
    - Keys k_t are L2-normalized so beta_t acts as a true step size and bounds updates.
    - Prediction error (v_t - k_t @ M_{t-1}) ensures only novel surprise is written;
      re-presenting already learned associations produces near-zero writes.
    - Token generation carries state forward in O(1) time per token (O(N) total).
    """

    def __init__(self, dim: int, name: str = "fast_memory"):
        super().__init__()
        self.dim = dim

        self.proj_q = Linear(dim, dim, bias=True, name=f"{name}.proj_q")
        self.proj_k = Linear(dim, dim, bias=True, name=f"{name}.proj_k")
        self.proj_v = Linear(dim, dim, bias=True, name=f"{name}.proj_v")
        self.proj_out = Linear(dim, dim, bias=True, name=f"{name}.proj_out")

        # Learned gates
        self.gate_alpha = Linear(dim, 1, bias=True, name=f"{name}.gate_alpha")
        self.gate_beta = Linear(dim, 1, bias=True, name=f"{name}.gate_beta")

        # Custom bias offsets: alpha near 1.0 (+4.0), beta small (-2.0)
        self.alpha_bias_offset: float = 4.0
        self.beta_bias_offset: float = -2.0

        self._cache: Optional[Tuple] = None
    def forward(
        self,
        x: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass over a sequence x of shape [B, T, D].

        Args:
            x: Input tensor [B, T, D].
            state: Initial memory matrix [B, D, D]. If None, zero-initialized.
            mask: Optional float padding mask [B, T] where 1.0 = real token, 0.0 = pad.

        Returns:
            Tuple of (output [B, T, D], final_state [B, D, D]).
        """
        B, T, D = x.shape

        q = self.proj_q(x)
        k_raw = self.proj_k(x)
        k_norm = np.linalg.norm(k_raw, axis=-1, keepdims=True) + 1e-12
        k = k_raw / k_norm
        v = self.proj_v(x)

        # Gate pre-activations
        za = self.gate_alpha(x) + self.alpha_bias_offset
        alpha = sigmoid(za)
        zb = self.gate_beta(x) + self.beta_bias_offset
        beta = sigmoid(zb)

        if state is None:
            memory = np.zeros((B, D, D), dtype=np.float32)
        else:
            memory = np.array(state, dtype=np.float32, copy=True)

        memory_history = [memory]
        outputs = []

        for t in range(T):
            qt = q[:, t : t + 1, :]
            kt = k[:, t : t + 1, :]
            vt = v[:, t : t + 1, :]
            at = alpha[:, t : t + 1, :]
            bt = beta[:, t : t + 1, :]

            # Read-out before write
            ot = np.matmul(qt, memory)
            outputs.append(ot)

            # Error-driven write (Delta rule)
            v_hat = np.matmul(kt, memory)
            error = vt - v_hat
            update = bt * np.matmul(kt.transpose(0, 2, 1), error)
            new_memory = at * memory + update

            if mask is not None:
                # Padding masking: masked token preserves memory untouched
                mt = mask[:, t : t + 1, np.newaxis]
                memory = (1.0 - mt) * memory + mt * new_memory
            else:
                memory = new_memory

            memory_history.append(memory)

        o = np.concatenate(outputs, axis=1)
        y = self.proj_out(o)

        self._cache = (x, q, k_raw, k_norm, k, v, za, alpha, zb, beta, memory_history, o, mask)
        return y, memory

    def step(
        self,
        x_t: np.ndarray,
        state: np.ndarray,
        mask_t: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fast O(1) single-token inference step for autoregressive generation.

        Args:
            x_t: Single token tensor [B, 1, D] or [B, D].
            state: Prior memory matrix [B, D, D].
            mask_t: Optional mask [B, 1] or [B].

        Returns:
            Tuple of (output [B, 1, D], next_memory [B, D, D]).
        """
        if x_t.ndim == 2:
            x_t = x_t[:, np.newaxis, :]
        B, _, D = x_t.shape

        if state.ndim == 2:
            state = state[np.newaxis, :, :]
        if state.shape[0] != B:
            if state.shape[0] == 1 and B > 1:
                state = np.repeat(state, B, axis=0)
            else:
                raise ValueError(
                    f"Batch size mismatch: input x_t has batch size {B} but state has batch size {state.shape[0]}"
                )
        q = self.proj_q(x_t)
        k_raw = self.proj_k(x_t)
        k_norm = np.linalg.norm(k_raw, axis=-1, keepdims=True) + 1e-12
        k = k_raw / k_norm
        v = self.proj_v(x_t)

        za = self.gate_alpha(x_t) + self.alpha_bias_offset
        alpha = sigmoid(za)
        zb = self.gate_beta(x_t) + self.beta_bias_offset
        beta = sigmoid(zb)

        ot = np.matmul(q, state)
        v_hat = np.matmul(k, state)
        error = v - v_hat
        update = beta * np.matmul(k.transpose(0, 2, 1), error)
        new_state = alpha * state + update

        if mask_t is not None:
            if mask_t.ndim == 1:
                mask_t = mask_t[:, np.newaxis, np.newaxis]
            elif mask_t.ndim == 2:
                mask_t = mask_t[:, :, np.newaxis]
            next_state = (1.0 - mask_t) * state + mask_t * new_state
        else:
            next_state = new_state

        y = self.proj_out(ot)
        return y, next_state

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Exact analytical backpropagation through time (BPTT)."""
        if self._cache is None:
            raise RuntimeError("SelfModifyingLayer.backward called before forward")

        x, q, k_raw, k_norm, k, v, za, alpha, zb, beta, memory_history, o, mask = self._cache
        B, T, D = x.shape

        # Backward through proj_out
        do = self.proj_out.backward(dy)

        dq = np.zeros_like(q)
        dk = np.zeros_like(k)
        dv = np.zeros_like(v)
        dza = np.zeros_like(za)
        dzb = np.zeros_like(zb)
        dM = np.zeros((B, D, D), dtype=np.float32)

        for t in reversed(range(T)):
            M_prev = memory_history[t]
            qt = q[:, t : t + 1, :]
            kt = k[:, t : t + 1, :]
            vt = v[:, t : t + 1, :]
            at = alpha[:, t : t + 1, :]
            bt = beta[:, t : t + 1, :]
            dot = do[:, t : t + 1, :]

            # o_t = q_t @ M_prev => dq_t = do_t @ M_prev^T
            dq[:, t : t + 1, :] = np.matmul(dot, M_prev.transpose(0, 2, 1))

            if mask is not None:
                mt = mask[:, t : t + 1, np.newaxis]
                d_new_M = dM * mt
                dM_prev_direct = dM * (1.0 - mt)
            else:
                d_new_M = dM
                dM_prev_direct = 0.0

            # Gradient w.r.t alpha: M_t = alpha * M_prev + ...
            da_t = np.sum(d_new_M * M_prev, axis=(1, 2), keepdims=True)
            dza[:, t : t + 1, :] = da_t * at * (1.0 - at)

            # Gradient w.r.t beta: update = beta * (kt^T @ error)
            v_hat = np.matmul(kt, M_prev)
            error = vt - v_hat
            ktT_error = np.matmul(kt.transpose(0, 2, 1), error)
            db_t = np.sum(d_new_M * ktT_error, axis=(1, 2), keepdims=True)
            dzb[:, t : t + 1, :] = db_t * bt * (1.0 - bt)

            # Gradient w.r.t error and v
            det = bt * np.matmul(kt, d_new_M)
            dv[:, t : t + 1, :] = det
            dv_hat = -det

            # Gradient w.r.t k
            dkt_upd = bt * np.matmul(error, d_new_M.transpose(0, 2, 1))
            dkt_vhat = np.matmul(dv_hat, M_prev.transpose(0, 2, 1))
            dk[:, t : t + 1, :] = dkt_upd + dkt_vhat

            # Accumulate gradient back into M_prev
            # From M_t: at * d_new_M
            # From o_t = q_t @ M_prev: q_t^T @ dot
            # From v_hat = k_t @ M_prev: k_t^T @ dv_hat
            dM = (
                dM_prev_direct
                + at * d_new_M
                + np.matmul(qt.transpose(0, 2, 1), dot)
                + np.matmul(kt.transpose(0, 2, 1), dv_hat)
            )

        # Backward through key L2-normalization:
        # dk_raw = (dk - k * sum(k * dk)) / k_norm
        dk_raw = (dk - k * np.sum(k * dk, axis=-1, keepdims=True)) / k_norm

        # Backward through input projections and gate layers
        dx_q = self.proj_q.backward(dq)
        dx_k = self.proj_k.backward(dk_raw)
        dx_v = self.proj_v.backward(dv)
        dx_za = self.gate_alpha.backward(dza)
        dx_zb = self.gate_beta.backward(dzb)

        dx = dx_q + dx_k + dx_v + dx_za + dx_zb
        return dx


class ContinuumMemoryBlock(Module):
    """Continuum Memory System (CMS) block.

    Replaces standard feed-forward networks (FFN) with a multi-frequency
    consolidating block that stores long-term patterns and representations:
        h = x + Linear2(GELU(Linear1(x)))
        out = LayerNorm(h)

    In HOPE, CMS blocks are grouped into tiers with different update periods
    (e.g., period 1, 4, 16) to form a continuum of memory from fast to slow.
    """

    def __init__(self, dim: int, expansion: int = 4, name: str = "cms"):
        super().__init__()
        self.dim = dim
        self.expansion = expansion
        hidden_dim = dim * expansion

        self.fc1 = Linear(dim, hidden_dim, bias=True, name=f"{name}.fc1")
        self.act = GELU()
        self.fc2 = Linear(hidden_dim, dim, bias=True, name=f"{name}.fc2")
        self.norm = LayerNorm(dim, name=f"{name}.norm")

        self._cache_residual: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache_residual = x
        h1 = self.fc1(x)
        h2 = self.act(h1)
        h3 = self.fc2(h2)
        out = self.norm(x + h3)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._cache_residual is None:
            raise RuntimeError("ContinuumMemoryBlock.backward called before forward")

        # Backward through LayerNorm
        dh = self.norm.backward(dout)

        # Residual branch: dh splits to x and fc2
        dx_res = dh
        dh3 = dh

        dh2 = self.fc2.backward(dh3)
        dh1 = self.act.backward(dh2)
        dx_net = self.fc1.backward(dh1)

        dx = dx_res + dx_net
        return dx
