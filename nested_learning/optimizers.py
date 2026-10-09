"""Deep Optimizers & Nested Multi-Tier Optimization in pure NumPy.

Reference:
- Behrouz et al., "Nested Learning: The Illusion of Deep Learning Architectures", NeurIPS 2025.
- https://github.com/obekt/HOPE-nested-learning
- https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from nested_learning.layers import Parameter


def clip_grad_norm(parameters: List[Parameter], max_norm: float = 1.0) -> float:
    """Clips parameter gradients by total L2 norm.

    Args:
        parameters: List of Parameter objects.
        max_norm: Maximum allowed L2 norm.

    Returns:
        Total norm before clipping.
    """
    total_sq = sum(np.sum(p.grad ** 2) for p in parameters)
    total_norm = float(np.sqrt(total_sq))
    if max_norm > 0 and total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in parameters:
            p.grad *= scale
    return total_norm


class Optimizer:
    """Base optimizer class."""

    def __init__(self, params: List[Parameter], lr: float = 1e-3, grad_clip: float = 1.0):
        self.params = params
        self.lr = lr
        self.grad_clip = grad_clip
        self.step_count = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        return {"step_count": self.step_count, "lr": self.lr}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.step_count = state.get("step_count", self.step_count)
        self.lr = state.get("lr", self.lr)


class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay."""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
    ):
        super().__init__(params, lr, grad_clip)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # State buffers
        self.m: List[np.ndarray] = [np.zeros_like(p.data) for p in params]
        self.v: List[np.ndarray] = [np.zeros_like(p.data) for p in params]

    def step(self) -> None:
        self.step_count += 1
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)

        b1, b2 = self.beta1, self.beta2
        lr = self.lr
        wd = self.weight_decay
        eps = self.eps
        t = self.step_count

        bias_correction1 = 1.0 - (b1 ** t)
        bias_correction2 = 1.0 - (b2 ** t)

        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad

            # Decoupled weight decay
            if wd > 0.0:
                p.data -= lr * wd * p.data

            # Moment estimates
            m[...] = b1 * m + (1.0 - b1) * g
            v[...] = b2 * v + (1.0 - b2) * (g ** 2)

            m_hat = m / bias_correction1
            v_hat = v / bias_correction2

            # Parameter update
            p.data -= lr * (m_hat / (np.sqrt(v_hat) + eps))

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["m"] = [arr.copy() for arr in self.m]
        state["v"] = [arr.copy() for arr in self.v]
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        if "m" in state and len(state["m"]) == len(self.m):
            for dest, src in zip(self.m, state["m"]):
                dest[...] = src
        if "v" in state and len(state["v"]) == len(self.v):
            for dest, src in zip(self.v, state["v"]):
                dest[...] = src


class SGD(Optimizer):
    """Stochastic Gradient Descent with momentum and weight decay."""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        grad_clip: float = 1.0,
    ):
        super().__init__(params, lr, grad_clip)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities: List[np.ndarray] = [np.zeros_like(p.data) for p in params]

    def step(self) -> None:
        self.step_count += 1
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)

        lr = self.lr
        mu = self.momentum
        wd = self.weight_decay

        for p, v in zip(self.params, self.velocities):
            g = p.grad
            if wd > 0.0:
                g = g + wd * p.data

            v[...] = mu * v + g
            p.data -= lr * v

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["velocities"] = [arr.copy() for arr in self.velocities]
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        if "velocities" in state and len(state["velocities"]) == len(self.velocities):
            for dest, src in zip(self.velocities, state["velocities"]):
                dest[...] = src


class CosineAnnealingLR:
    """Cosine Annealing Learning Rate Scheduler with optional linear warmup."""

    def __init__(
        self,
        base_lr: float,
        max_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 1e-6,
    ):
        self.base_lr = base_lr
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * float(step) / max(1.0, float(self.warmup_steps))

        progress = float(step - self.warmup_steps) / max(1.0, float(self.max_steps - self.warmup_steps))
        progress = min(1.0, max(0.0, progress))

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay


class NestedOptimizer:
    """Continuum Deep Optimizer for Nested Learning architectures.

    Concept:
    Instead of updating all layers at a single uniform rate, Nested Learning organizes
    the model into multi-frequency tiers (e.g., fast tier every step, medium every 4,
    slow every 16 steps).

    Mechanics:
    - Every step, current gradients are accumulated into per-tier buffers:
        buf_i += grad_i
    - When a tier's period elapses (counter_i == period_i):
        avg_grad = buf_i / period_i
        tier_opt_i.step(avg_grad)
        buf_i.fill(0.0)
        counter_i = 0
    - This allows slow tiers to consolidate structural knowledge over multiple steps,
      mitigating catastrophic forgetting while fast tiers quickly absorb recent context.
    """

    def __init__(
        self,
        tier_param_groups: List[Tuple[int, List[Parameter]]],
        lr: float = 1e-3,
        optimizer_cls: Callable = AdamW,
        scheduler: Optional[CosineAnnealingLR] = None,
        grad_clip: float = 1.0,
        **opt_kwargs: Any,
    ):
        self.tier_periods: List[int] = []
        self.tier_param_lists: List[List[Parameter]] = []
        self.optimizers: List[Optimizer] = []
        self.grad_buffers: List[List[np.ndarray]] = []
        self.tier_counters: List[int] = []
        self.global_step = 0
        self.scheduler = scheduler
        self.base_lr = lr
        self.grad_clip = grad_clip

        for period, params in tier_param_groups:
            self.tier_periods.append(period)
            self.tier_param_lists.append(params)
            opt = optimizer_cls(params, lr=lr, grad_clip=0.0, **opt_kwargs)  # clip handled globally
            self.optimizers.append(opt)
            self.grad_buffers.append([np.zeros_like(p.data) for p in params])
            self.tier_counters.append(0)

    def zero_grad(self) -> None:
        """Clear current step gradients across all parameters."""
        for params in self.tier_param_lists:
            for p in params:
                p.zero_grad()

    def step(self) -> List[bool]:
        """Perform one nested optimization step across all tiers.

        Returns:
            List of boolean flags indicating which tiers performed a weight update this step.
        """
        self.global_step += 1

        # Synchronize learning rate across all tier optimizers
        if self.scheduler is not None:
            current_lr = self.scheduler.get_lr(self.global_step)
            for opt in self.optimizers:
                opt.lr = current_lr

        # Global gradient clipping across all parameters
        if self.grad_clip > 0:
            all_params = [p for params in self.tier_param_lists for p in params]
            clip_grad_norm(all_params, self.grad_clip)

        tier_stepped = []

        for i, (period, params, opt) in enumerate(
            zip(self.tier_periods, self.tier_param_lists, self.optimizers)
        ):
            # Accumulate this step's gradient into tier buffer
            for buf, p in zip(self.grad_buffers[i], params):
                buf += p.grad

            self.tier_counters[i] += 1

            if self.tier_counters[i] >= period:
                # Elapse reached: apply averaged gradient
                scale = 1.0 / period
                for buf, p in zip(self.grad_buffers[i], params):
                    p.grad[...] = buf * scale

                opt.step()

                # Reset buffer and counter
                for buf in self.grad_buffers[i]:
                    buf.fill(0.0)
                self.tier_counters[i] = 0
                tier_stepped.append(True)
            else:
                tier_stepped.append(False)

        return tier_stepped

    def get_tier_stats(self) -> List[Dict[str, Any]]:
        """Return status of all tiers (period, counter, params count)."""
        stats = []
        for i, (period, params, counter) in enumerate(
            zip(self.tier_periods, self.tier_param_lists, self.tier_counters)
        ):
            stats.append({
                "tier_idx": i,
                "period": period,
                "counter": counter,
                "num_params": sum(p.data.size for p in params),
            })
        return stats

    def state_dict(self) -> Dict[str, Any]:
        """Serialize optimizer state."""
        return {
            "global_step": self.global_step,
            "tier_counters": list(self.tier_counters),
            "tier_periods": list(self.tier_periods),
            "opt_states": [opt.state_dict() for opt in self.optimizers],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore optimizer state."""
        self.global_step = state.get("global_step", self.global_step)
        if "tier_counters" in state:
            self.tier_counters = list(state["tier_counters"])
        if "opt_states" in state and len(state["opt_states"]) == len(self.optimizers):
            for opt, opt_state in zip(self.optimizers, state["opt_states"]):
                opt.load_state_dict(opt_state)
