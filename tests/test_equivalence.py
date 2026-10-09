"""Equivalence tests: Full-sequence forward == token-by-token state-passing forward.

This validates the mathematical correctness of O(1) step / O(N) generation inference:
Carrying memory state forward step-by-step produces identical results to full sequence forward.
"""

import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nested_learning.model import HOPE


def run_equivalence_test():
    print("=" * 60)
    print("TEST: State-Passing Equivalence (O(N) Inference)")
    print("=" * 60)

    np.random.seed(42)
    vocab_size = 64
    d_model = 32
    n_layers = 3
    seq_len = 12

    model = HOPE(vocab_size, d_model, n_layers, cms_tiers=[[2, 1], [1, 2]])

    # Generate random test sequence
    tokens = np.random.randint(0, vocab_size, size=(2, seq_len))

    # 1. Full-sequence forward
    logits_full, state_full = model.forward(tokens)

    # 2. Token-by-token incremental step forward
    state = None
    step_logits_list = []
    for t in range(seq_len):
        tok_t = tokens[:, t : t + 1]
        if state is None:
            # First token initializes state
            l_t, state = model.step(tok_t, np.zeros((2, d_model, d_model), dtype=np.float32))
        else:
            l_t, state = model.step(tok_t, state)
        step_logits_list.append(l_t)

    logits_incremental = np.concatenate(step_logits_list, axis=1)

    # 3. Check differences
    logits_diff = np.max(np.abs(logits_full - logits_incremental))
    state_diff = np.max(np.abs(state_full - state))

    print(f"Logits max absolute difference:     {logits_diff:.2e}")
    print(f"Final state max absolute difference: {state_diff:.2e}")

    assert logits_diff < 1e-4, f"Logits mismatch! Diff: {logits_diff}"
    assert state_diff < 1e-4, f"Final memory state mismatch! Diff: {state_diff}"
    print("✓ Full sequence forward ≡ Token-by-token state passing: PASS\n")

    # 4. Check last_only prefill equivalence
    logits_last_only, _ = model.forward(tokens, last_only=True)
    expected_last_logits = logits_full[:, -1:, :]
    last_diff = np.max(np.abs(logits_last_only - expected_last_logits))
    print(f"Last-only prefill max diff:          {last_diff:.2e}")
    assert last_diff < 1e-5, f"Last-only prefill mismatch! Diff: {last_diff}"
    print("✓ Last-only prefill optimization equivalence: PASS\n")


if __name__ == "__main__":
    run_equivalence_test()
