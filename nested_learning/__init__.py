"""Nested Learning Lite: A Pure NumPy Implementation of Nested Learning & HOPE.

Zero-dependency framework based on:
- Behrouz et al., "Nested Learning: The Illusion of Deep Learning Architectures", NeurIPS 2025.
- Reference implementation: https://github.com/obekt/HOPE-nested-learning
- Google Research blog: https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/
"""

from nested_learning.layers import (
    ContinuumMemoryBlock,
    Embedding,
    GELU,
    LayerNorm,
    Linear,
    Module,
    Parameter,
    SelfModifyingLayer,
)
from nested_learning.loss import CrossEntropyLoss
from nested_learning.model import HOPE
from nested_learning.optimizers import (
    AdamW,
    CosineAnnealingLR,
    NestedOptimizer,
    Optimizer,
    SGD,
    clip_grad_norm,
)
from nested_learning.tokenizer import (
    BaseTokenizer,
    ByteTokenizer,
    CharTokenizer,
    get_tokenizer,
    load_tokenizer,
    resolve_tokenizer_for_checkpoint,
)
from nested_learning.memory_store import SQLiteMemoryStore, memory_to_vector
from nested_learning.trainer import TextDataset, Trainer
__all__ = [
    "Parameter",
    "Module",
    "Linear",
    "Embedding",
    "LayerNorm",
    "GELU",
    "SelfModifyingLayer",
    "ContinuumMemoryBlock",
    "HOPE",
    "CrossEntropyLoss",
    "Optimizer",
    "AdamW",
    "SGD",
    "NestedOptimizer",
    "CosineAnnealingLR",
    "clip_grad_norm",
    "BaseTokenizer",
    "ByteTokenizer",
    "CharTokenizer",
    "get_tokenizer",
    "load_tokenizer",
    "resolve_tokenizer_for_checkpoint",
    "TextDataset",
    "Trainer",
    "SQLiteMemoryStore",
    "memory_to_vector",
]

__version__ = "0.1.0"
