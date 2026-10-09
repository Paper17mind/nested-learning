"""Delta Rule tests: Inner-loop learning and associative memory integrity.

Tests:
1. Inner-loop SGD convergence: Repeated presentation of (k -> v) causes prediction error to shrink.
2. Stability: Memory values remain strictly bounded.
3. Masking integrity: Masked (padding) tokens leave memory completely untouched.
4. Checkpoint roundtrip: Saving and loading preserves bit-identical predictions.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from nested_learning.layers import SelfModifyingLayer
from nested_learning.model import HOPE


def test_delta_rule_convergence():
    print("=" * 60)
    print("TEST: Delta Rule Inner-Loop Learning")
    print("=" * 60)

    np.random.seed(42)
    dim = 32
    layer = SelfModifyingLayer(dim)

    # Single token repeated 30 times
    x = np.random.randn(1, 1, dim).astype(np.float32)

    # Compute key and value for this token
    k_raw = layer.proj_k(x)
    k_norm = np.linalg.norm(k_raw, axis=-1, keepdims=True) + 1e-12
    k = k_raw / k_norm
    v = layer.proj_v(x)

    state = None
    errors = []

    for step in range(30):
        _, state = layer.forward(x, state=state)
        # Recall from memory: v_hat = k @ M
        v_hat = np.matmul(k, state)
        err = float(np.sum(np.abs(v - v_hat)))
        errors.append(err)

    initial_error = errors[0]
    final_error = errors[-1]
    max_state_val = float(np.max(np.abs(state)))

    print(f"Step  0 Prediction Error: {initial_error:.4f}")
    print(f"Step 29 Prediction Error: {final_error:.4f} (Ratio: {final_error / initial_error:.2%})")
    print(f"Max |M| memory element:   {max_state_val:.4f}")

    assert final_error < 0.6 * initial_error, (
        f"Inner loop failed to reduce error! err0={initial_error}, err29={final_error}"
    )
    assert max_state_val < 50.0, f"Memory state exploded! max |M| = {max_state_val}"
    print("✓ Error-driven inner loop converges and remains bounded: PASS\n")


def test_mask_integrity():
    print("=" * 60)
    print("TEST: Memory Mask Integrity (Padding Leakage Guard)")
    print("=" * 60)

    np.random.seed(42)
    dim = 32
    layer = SelfModifyingLayer(dim)

    # 3 tokens: token 0 is real, tokens 1 and 2 are padding
    x = np.random.randn(1, 3, dim).astype(np.float32)
    mask = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    _, memory_masked = layer.forward(x, state=None, mask=mask)
    _, memory_first_only = layer.forward(x[:, :1, :], state=None, mask=mask[:, :1])

    diff = float(np.max(np.abs(memory_masked - memory_first_only)))
    print(f"Memory difference between masked run and first-token-only: {diff:.2e}")

    assert diff < 1e-6, f"Padding leaked into memory! diff={diff}"
    print("✓ Masked tokens leave memory bit-identical: PASS\n")


def test_checkpoint_roundtrip():
    print("=" * 60)
    print("TEST: Checkpoint Serialization Roundtrip")
    print("=" * 60)

    np.random.seed(42)
    model = HOPE(vocab_size=32, d_model=16, n_layers=2)
    x = np.random.randint(0, 32, size=(2, 6))

    logits_orig, state_orig = model.forward(x)

    tmp_path = "/tmp/hope_test_ckpt.npz"
    model.save_checkpoint(tmp_path, extra_meta={"note": "roundtrip_test", "epoch": 5})

    loaded_model, meta = HOPE.load_checkpoint(tmp_path)
    assert meta["note"] == "roundtrip_test"
    assert meta["epoch"] == 5

    logits_loaded, state_loaded = loaded_model.forward(x)

    diff_logits = float(np.max(np.abs(logits_orig - logits_loaded)))
    diff_state = float(np.max(np.abs(state_orig - state_loaded)))

    print(f"Loaded model logits diff: {diff_logits:.2e}")
    print(f"Loaded model state diff:  {diff_state:.2e}")

    assert diff_logits == 0.0, "Checkpoint restored weights produce differing logits!"
    assert diff_state == 0.0, "Checkpoint restored weights produce differing memory state!"
    print("✓ Checkpoint serialization roundtrip bit-identical: PASS\n")

    if os.path.exists(tmp_path):
        os.remove(tmp_path)


if __name__ == "__main__":
    test_delta_rule_convergence()
    test_mask_integrity()
    test_checkpoint_roundtrip()
