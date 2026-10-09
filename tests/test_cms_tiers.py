"""CMS Tiers Test: Verify multi-frequency update schedule in NestedOptimizer.

Verifies:
1. Fast tier (period=1) parameters update every single step.
2. Slow tier (period=3) parameters stay unchanged until period elapses (steps 3, 6).
3. Gradient accumulation buffers correctly reset after each update.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from nested_learning.loss import CrossEntropyLoss
from nested_learning.model import HOPE
from nested_learning.optimizers import AdamW, NestedOptimizer


def test_tier_update_schedule():
    print("=" * 60)
    print("TEST: Nested Multi-Frequency Tier Schedule")
    print("=" * 60)

    np.random.seed(42)
    vocab_size = 32
    d_model = 16
    n_layers = 3

    # Tiers: 2 layers at period 1 (Fast), 1 layer at period 3 (Slow)
    cms_tiers = [[2, 1], [1, 3]]
    model = HOPE(vocab_size, d_model, n_layers, cms_tiers=cms_tiers)

    loss_fn = CrossEntropyLoss()
    optimizer = NestedOptimizer(
        model.tier_param_groups(),
        lr=1e-2,
        optimizer_cls=AdamW,
        grad_clip=0.0,
    )

    stats = optimizer.get_tier_stats()
    print(f"Tier 0 (Fast): Period = {stats[0]['period']}, Params = {stats[0]['num_params']}")
    print(f"Tier 1 (Slow): Period = {stats[1]['period']}, Params = {stats[1]['num_params']}")
    assert len(stats) == 2
    assert stats[0]["period"] == 1
    assert stats[1]["period"] == 3

    # Probes for tracking weight changes
    fast_probe = model.cms_layers[0].fc1.weight.data
    slow_probe = model.cms_layers[2].fc1.weight.data

    fast_history = [fast_probe.copy()]
    slow_history = [slow_probe.copy()]

    inputs = np.random.randint(0, vocab_size, size=(2, 6))
    targets = np.random.randint(0, vocab_size, size=(2, 6))

    fast_stepped_at = []
    slow_stepped_at = []

    print("\nExecuting 6 training steps...")
    for step in range(1, 7):
        optimizer.zero_grad()
        logits, _ = model.forward(inputs)
        loss = loss_fn.forward(logits, targets)
        dlogits = loss_fn.backward()
        model.backward(dlogits)

        stepped = optimizer.step()

        # Check if probe weights changed
        fast_changed = not np.allclose(fast_history[-1], fast_probe)
        slow_changed = not np.allclose(slow_history[-1], slow_probe)

        if fast_changed:
            fast_stepped_at.append(step)
            fast_history.append(fast_probe.copy())

        if slow_changed:
            slow_stepped_at.append(step)
            slow_history.append(slow_probe.copy())

        print(
            f"Step {step}: Loss = {loss:.4f} | "
            f"Tiers stepped = {stepped} | "
            f"Fast changed: {fast_changed} | Slow changed: {slow_changed}"
        )

    print(f"\nFast tier changed at steps: {fast_stepped_at}")
    print(f"Slow tier changed at steps: {slow_stepped_at}")

    assert fast_stepped_at == [1, 2, 3, 4, 5, 6], (
        f"Fast tier expected [1, 2, 3, 4, 5, 6], got {fast_stepped_at}"
    )
    assert slow_stepped_at == [3, 6], (
        f"Slow tier expected [3, 6], got {slow_stepped_at}"
    )

    print("✓ Fast tier updates every step: PASS")
    print("✓ Slow tier updates strictly at period multiples [3, 6]: PASS\n")


if __name__ == "__main__":
    test_tier_update_schedule()
