"""Comprehensive numerical gradient tests for all layers in Nested Learning Lite.

Compares analytical backward passes against finite differences (central difference):
    num_grad = (f(x + eps) - f(x - eps)) / (2 * eps)
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from nested_learning.layers import (
    ContinuumMemoryBlock,
    Embedding,
    GELU,
    LayerNorm,
    Linear,
    SelfModifyingLayer,
)
from nested_learning.loss import CrossEntropyLoss


def check_grad(
    forward_fn,
    loss_fn,
    param_arr: np.ndarray,
    ana_grad: np.ndarray,
    name: str = "",
    eps: float = 1e-3,
    rtol: float = 1e-2,
    atol: float = 5e-3,
) -> bool:
    num_grad = np.zeros_like(param_arr)
    it = np.nditer(param_arr, flags=["multi_index"])

    while not it.finished:
        idx = it.multi_index
        orig = param_arr[idx]

        param_arr[idx] = orig + eps
        out_pos = forward_fn()
        loss_pos = loss_fn(out_pos)

        param_arr[idx] = orig - eps
        out_neg = forward_fn()
        loss_neg = loss_fn(out_neg)

        param_arr[idx] = orig
        num_grad[idx] = (loss_pos - loss_neg) / (2.0 * eps)
        it.iternext()

    max_diff = float(np.max(np.abs(num_grad - ana_grad)))
    rel_diff = float(np.max(np.abs(num_grad - ana_grad) / (np.abs(num_grad) + np.abs(ana_grad) + 1e-8)))

    is_ok = max_diff < atol or rel_diff < rtol
    status = "PASS" if is_ok else "FAIL"
    print(f"  [{status}] {name:30s} | max_diff: {max_diff:.2e} | rel_diff: {rel_diff:.2e}")
    assert is_ok, f"Gradient check failed for {name}! max_diff={max_diff}, rel_diff={rel_diff}"
    return is_ok


def test_linear_gradients():
    print("\n--- Testing Linear Gradients ---")
    np.random.seed(1)
    layer = Linear(4, 3)
    x = np.random.randn(2, 4).astype(np.float32)
    dy = np.random.randn(2, 3).astype(np.float32)

    y = layer.forward(x)
    dx = layer.backward(dy)

    loss_fn = lambda out: float(np.sum(out * dy))

    check_grad(lambda: layer.forward(x), loss_fn, x, dx, "Linear input x")
    check_grad(lambda: layer.forward(x), loss_fn, layer.weight.data, layer.weight.grad, "Linear weight")
    check_grad(lambda: layer.forward(x), loss_fn, layer.bias.data, layer.bias.grad, "Linear bias")


def test_layernorm_gradients():
    print("\n--- Testing LayerNorm Gradients ---")
    np.random.seed(2)
    norm = LayerNorm(5)
    x = np.random.randn(2, 3, 5).astype(np.float32)
    dy = np.random.randn(2, 3, 5).astype(np.float32)

    norm.forward(x)
    dx = norm.backward(dy)

    loss_fn = lambda out: float(np.sum(out * dy))

    check_grad(lambda: norm.forward(x), loss_fn, x, dx, "LayerNorm input x")
    check_grad(lambda: norm.forward(x), loss_fn, norm.gamma.data, norm.gamma.grad, "LayerNorm gamma")
    check_grad(lambda: norm.forward(x), loss_fn, norm.beta.data, norm.beta.grad, "LayerNorm beta")


def test_gelu_gradients():
    print("\n--- Testing GELU Gradients ---")
    np.random.seed(3)
    act = GELU()
    x = np.random.randn(2, 4).astype(np.float32)
    dy = np.random.randn(2, 4).astype(np.float32)

    act.forward(x)
    dx = act.backward(dy)

    loss_fn = lambda out: float(np.sum(out * dy))
    check_grad(lambda: act.forward(x), loss_fn, x, dx, "GELU input x")


def test_embedding_gradients():
    print("\n--- Testing Embedding Gradients ---")
    np.random.seed(4)
    emb = Embedding(8, 4)
    indices = np.array([[1, 3, 1], [0, 2, 3]], dtype=np.int64)
    dy = np.random.randn(2, 3, 4).astype(np.float32)

    emb.forward(indices)
    emb.backward(dy)

    loss_fn = lambda out: float(np.sum(out * dy))
    check_grad(lambda: emb.forward(indices), loss_fn, emb.weight.data, emb.weight.grad, "Embedding weight")


def test_self_modifying_layer_gradients():
    print("\n--- Testing SelfModifyingLayer Gradients ---")
    np.random.seed(5)
    dim = 4
    sml = SelfModifyingLayer(dim)
    x = np.random.randn(1, 3, dim).astype(np.float32)
    dy = np.random.randn(1, 3, dim).astype(np.float32)

    sml.forward(x)
    dx = sml.backward(dy)

    loss_fn = lambda out: float(np.sum(out[0] * dy))

    check_grad(lambda: sml.forward(x), loss_fn, x, dx, "SelfModifyingLayer input x")
    check_grad(lambda: sml.forward(x), loss_fn, sml.proj_q.weight.data, sml.proj_q.weight.grad, "SML proj_q.weight")
    check_grad(lambda: sml.forward(x), loss_fn, sml.proj_k.weight.data, sml.proj_k.weight.grad, "SML proj_k.weight")
    check_grad(lambda: sml.forward(x), loss_fn, sml.proj_v.weight.data, sml.proj_v.weight.grad, "SML proj_v.weight")
    check_grad(lambda: sml.forward(x), loss_fn, sml.proj_out.weight.data, sml.proj_out.weight.grad, "SML proj_out.weight")
    check_grad(lambda: sml.forward(x), loss_fn, sml.gate_alpha.weight.data, sml.gate_alpha.weight.grad, "SML gate_alpha.weight")
    check_grad(lambda: sml.forward(x), loss_fn, sml.gate_beta.weight.data, sml.gate_beta.weight.grad, "SML gate_beta.weight")


def test_cms_block_gradients():
    print("\n--- Testing ContinuumMemoryBlock Gradients ---")
    np.random.seed(6)
    cms = ContinuumMemoryBlock(dim=4, expansion=2)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    dy = np.random.randn(2, 3, 4).astype(np.float32)

    cms.forward(x)
    dx = cms.backward(dy)

    loss_fn = lambda out: float(np.sum(out * dy))

    check_grad(lambda: cms.forward(x), loss_fn, x, dx, "CMS block input x")
    check_grad(lambda: cms.forward(x), loss_fn, cms.fc1.weight.data, cms.fc1.weight.grad, "CMS fc1.weight")
    check_grad(lambda: cms.forward(x), loss_fn, cms.fc2.weight.data, cms.fc2.weight.grad, "CMS fc2.weight")


def test_cross_entropy_gradients():
    print("\n--- Testing CrossEntropyLoss Gradients ---")
    np.random.seed(7)
    loss_fn = CrossEntropyLoss(ignore_index=-100)
    logits = np.random.randn(2, 3, 4).astype(np.float32)
    targets = np.array([[1, 2, -100], [0, 3, 1]], dtype=np.int64)

    loss_fn.forward(logits, targets)
    dlogits = loss_fn.backward()

    # Numerical grad on logits
    eps = 1e-3
    num_dlogits = np.zeros_like(logits)
    it = np.nditer(logits, flags=["multi_index"])

    while not it.finished:
        idx = it.multi_index
        orig = logits[idx]
        logits[idx] = orig + eps
        l_pos = loss_fn.forward(logits, targets)
        logits[idx] = orig - eps
        l_neg = loss_fn.forward(logits, targets)
        logits[idx] = orig
        num_dlogits[idx] = (l_pos - l_neg) / (2.0 * eps)
        it.iternext()

    max_diff = float(np.max(np.abs(num_dlogits - dlogits)))
    print(f"  [PASS] {'CrossEntropyLoss logits':30s} | max_diff: {max_diff:.2e}")
    assert max_diff < 5e-4


def run_all():
    print("=" * 60)
    print("TEST: Numerical Gradient Verification (All Layers)")
    print("=" * 60)
    test_linear_gradients()
    test_layernorm_gradients()
    test_gelu_gradients()
    test_embedding_gradients()
    test_self_modifying_layer_gradients()
    test_cms_block_gradients()
    test_cross_entropy_gradients()
    print("\n✓ All gradient checks PASSED!\n")


if __name__ == "__main__":
    run_all()
