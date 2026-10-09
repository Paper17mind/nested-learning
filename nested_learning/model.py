"""HOPE Architecture (High-order Optimization & Perception Engine) in pure NumPy.

Reference:
- Behrouz et al., "Nested Learning: The Illusion of Deep Learning Architectures", NeurIPS 2025.
- https://github.com/obekt/HOPE-nested-learning
- https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from nested_learning.layers import (
    ContinuumMemoryBlock,
    Embedding,
    LayerNorm,
    Linear,
    Module,
    Parameter,
    SelfModifyingLayer,
)


class HOPE(Module):
    """HOPE: Self-Modifying Sequence Model with Continuum Memory System (CMS).

    Architecture:
    1. Token Embedding: E(x)
    2. Fast-Weight Memory: SelfModifyingLayer with online Delta Rule SGD per token
    3. Residual + Norm: LayerNorm(E(x) + FastMemory(E(x)))
    4. Continuum Memory System (CMS): Stack of multi-frequency MLP blocks
    5. Output Head: Linear projection to vocabulary logits

    Nested Learning Tiers (CMS Tiers):
    The layers are grouped into update periods, e.g. [[fast_layers, 1], [med_layers, 4], [slow_layers, 16]].
    Fast tiers adapt immediately, while slow tiers consolidate long-term knowledge,
    structurally preventing catastrophic forgetting.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        cms_tiers: Optional[List[List[int]]] = None,
        expansion: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.expansion = expansion

        # Validate / set CMS tiers: [[layer_count, period], ...]
        if cms_tiers is None:
            cms_tiers = [[n_layers, 1]]
        total_tier_layers = sum(count for count, _ in cms_tiers)
        if total_tier_layers != n_layers:
            raise ValueError(
                f"Sum of tier layer counts ({total_tier_layers}) must equal n_layers ({n_layers})."
            )
        self.cms_tiers = [list(t) for t in cms_tiers]

        # Model components
        self.embedding = Embedding(vocab_size, d_model, name="embedding")
        self.fast_memory = SelfModifyingLayer(d_model, name="fast_memory")
        self.norm_fast = LayerNorm(d_model, name="norm_fast")

        self.cms_layers: List[ContinuumMemoryBlock] = [
            ContinuumMemoryBlock(d_model, expansion=expansion, name=f"cms_{i}")
            for i in range(n_layers)
        ]

        self.head = Linear(d_model, vocab_size, bias=True, name="head")

        self._cache_fast_h: Optional[np.ndarray] = None
        self._cache_fast_out: Optional[np.ndarray] = None

    def tier_param_groups(self) -> List[Tuple[int, List[Parameter]]]:
        """Partition parameters into (period, parameters) groups for nested updates.

        Tier 0 (period = 1) owns:
        - embedding
        - fast_memory
        - norm_fast
        - head
        - and its assigned first n CMS layers.

        Subsequent tiers own their respective deeper CMS layers.
        """
        groups: List[Tuple[int, List[Parameter]]] = []
        layer_idx = 0
        for i, (n, period) in enumerate(self.cms_tiers):
            params: List[Parameter] = []
            if i == 0:
                params.extend(self.embedding.parameters())
                params.extend(self.fast_memory.parameters())
                params.extend(self.norm_fast.parameters())
                params.extend(self.head.parameters())
            for layer in self.cms_layers[layer_idx : layer_idx + n]:
                params.extend(layer.parameters())
            layer_idx += n
            groups.append((period, params))
        return groups

    def forward(
        self,
        x: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        last_only: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Full sequence forward pass.

        Args:
            x: Token IDs array of shape [B, T].
            state: Optional initial fast memory state [B, D, D].
            mask: Optional mask array [B, T] (1 = real token, 0 = padding).
            last_only: If True, runs CMS stack and head on the final position only
                       (valid for generation prefill because CMS blocks are position-wise).

        Returns:
            Tuple of (logits [B, T, V] or [B, 1, V], final_state [B, D, D]).
        """
        # 1. Embedding
        h = self.embedding(x)
        self._cache_fast_h = h

        # 2. Fast-weight self-modifying memory
        fast_out, new_state = self.fast_memory(h, state=state, mask=mask)
        self._cache_fast_out = fast_out

        # 3. Residual connection and LayerNorm
        h = self.norm_fast(h + fast_out)

        # 4. Optional last-only slicing for fast prefill
        if last_only:
            h = h[:, -1:, :]

        # 5. Continuum Memory Blocks (CMS stack)
        for layer in self.cms_layers:
            h = layer(h)

        # 6. LM Head
        logits = self.head(h)
        return logits, new_state

    def step(
        self,
        x_t: np.ndarray,
        state: np.ndarray,
        mask_t: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single-token autoregressive generation step in O(1) time per token.

        Args:
            x_t: Single token IDs array of shape [B, 1] or [B].
            state: Memory state [B, D, D] from previous token.
            mask_t: Optional mask for this step [B, 1] or [B].

        Returns:
            Tuple of (logits [B, 1, V], next_state [B, D, D]).
        """
        if x_t.ndim == 1:
            x_t = x_t[:, np.newaxis]

        h = self.embedding(x_t)
        fast_out, next_state = self.fast_memory.step(h, state, mask_t=mask_t)
        h = self.norm_fast(h + fast_out)

        for layer in self.cms_layers:
            h = layer(h)

        logits = self.head(h)
        return logits, next_state

    def backward(self, dlogits: np.ndarray) -> None:
        """Backpropagation through the entire HOPE model."""
        # 1. Backward through LM Head
        dh = self.head.backward(dlogits)

        # 2. Backward through CMS layers in reverse order
        for layer in reversed(self.cms_layers):
            dh = layer.backward(dh)

        # 3. Backward through norm_fast
        dh_norm = self.norm_fast.backward(dh)

        # Residual split: dh_norm splits into h and fast_out
        dh_fast_out = dh_norm
        dh_emb_direct = dh_norm

        # 4. Backward through fast memory
        dh_fast_in = self.fast_memory.backward(dh_fast_out)

        # 5. Total gradient arriving at embedding
        dh_emb_total = dh_emb_direct + dh_fast_in
        self.embedding.backward(dh_emb_total)

    def generate(
        self,
        prompt_tokens: Union[List[int], np.ndarray],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> List[int]:
        """Autoregressive text generation using O(N) state-passing inference.

        Args:
            prompt_tokens: List or 1D array of token IDs.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (higher = more random, 0 = greedy).
            top_k: Top-k filtering (0 = disabled).
            top_p: Top-p nucleus filtering (0.0 = disabled).
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty).
            eos_token_id: End of sequence token ID to terminate early.

        Returns:
            List of generated token IDs (including the prompt).
        """
        prompt = np.asarray(prompt_tokens, dtype=np.int64)
        if prompt.ndim == 1:
            prompt = prompt[np.newaxis, :]  # [1, T]

        # 1. Prefill phase: process full prompt and get initial memory state
        logits, state = self.forward(prompt, state=None, last_only=True)
        # logits is [1, 1, V]
        last_logits = logits[0, -1, :].copy()

        generated = prompt[0].tolist()

        for _ in range(max_new_tokens):
            cur_logits = last_logits.copy()

            # Apply repetition penalty
            if repetition_penalty != 1.0 and len(generated) > 0:
                for token_id in set(generated):
                    if cur_logits[token_id] > 0:
                        cur_logits[token_id] /= repetition_penalty
                    else:
                        cur_logits[token_id] *= repetition_penalty

            # Greedy or temperature sampling
            if temperature <= 1e-4:
                next_token = int(np.argmax(cur_logits))
            else:
                scaled_logits = cur_logits / temperature

                # Top-k filter
                if top_k > 0:
                    indices_to_remove = np.argsort(scaled_logits)[:-top_k]
                    scaled_logits[indices_to_remove] = -1e9

                # Softmax
                max_logit = np.max(scaled_logits)
                probs = np.exp(scaled_logits - max_logit)
                prob_sum = np.sum(probs)
                if prob_sum <= 0 or not np.isfinite(prob_sum):
                    probs = np.ones_like(probs) / len(probs)
                else:
                    probs = probs / prob_sum

                # Top-p (nucleus) filter
                if 0.0 < top_p < 1.0:
                    sorted_indices = np.argsort(probs)[::-1]
                    sorted_probs = probs[sorted_indices]
                    cumulative_probs = np.cumsum(sorted_probs)

                    cutoff_idx = np.searchsorted(cumulative_probs, top_p)
                    valid_indices = sorted_indices[: cutoff_idx + 1]
                    probs_filtered = np.zeros_like(probs)
                    probs_filtered[valid_indices] = probs[valid_indices]
                    filter_sum = np.sum(probs_filtered)
                    if filter_sum > 0:
                        probs = probs_filtered / filter_sum

                next_token = int(np.random.choice(len(probs), p=probs))

            generated.append(next_token)

            if eos_token_id is not None and next_token == eos_token_id:
                break

            # O(1) single-token step
            x_next = np.array([[next_token]], dtype=np.int64)
            logits_step, state = self.step(x_next, state)
            last_logits = logits_step[0, -1, :].copy()

        return generated

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.data.size for p in self.parameters())

    def get_config(self) -> Dict[str, Any]:
        """Return model architecture configuration dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "cms_tiers": self.cms_tiers,
            "expansion": self.expansion,
        }

    def save_checkpoint(self, filepath: str, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        """Save model weights and metadata to a compressed .npz archive."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        param_dict = {}
        for p in self.parameters():
            param_dict[p.name] = p.data

        meta = {
            "config": self.get_config(),
            "extra": extra_meta or {},
        }
        meta_json = json.dumps(meta)

        np.savez_compressed(filepath, __meta__=meta_json, **param_dict)

    @classmethod
    def load_checkpoint(cls, filepath: str) -> Tuple["HOPE", Dict[str, Any]]:
        """Load model from a compressed .npz archive."""
        data = np.load(filepath, allow_pickle=False)
        meta_json = str(data["__meta__"])
        meta = json.loads(meta_json)
        cfg = meta["config"]

        model = cls(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            cms_tiers=cfg.get("cms_tiers"),
            expansion=cfg.get("expansion", 4),
        )

        for p in model.parameters():
            if p.name in data:
                p.data[...] = data[p.name]
            else:
                raise KeyError(f"Parameter '{p.name}' missing from checkpoint {filepath}")

        return model, meta.get("extra", {})
