"""Training Demo for Nested Learning Lite (HOPE Architecture).

Trains a lightweight HOPE model on an educational text corpus using pure NumPy,
demonstrating the continuum memory system and saving a checkpoint.
"""

import os
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import nested_learning as nl

TRAINING_CORPUS = """
Nested Learning: The Illusion of Deep Learning Architectures.
Nested Learning is a machine learning paradigm introduced by Google Research in 2025.
Instead of treating a model as a static set of layers trained by an external optimizer,
Nested Learning views intelligence as a hierarchy of nested optimization problems.

Key components of the HOPE architecture:
1. Fast-Weight Memory Layer:
An inner-loop associative memory that performs online gradient descent on every token.
It follows the delta rule: M_t = alpha_t * M_{t-1} + beta_t * k_t^T * (v_t - k_t * M_{t-1}).
Only the prediction error (surprise) is stored in the memory matrix.
Already-known information produces zero update, preventing memory interference.

2. Continuum Memory System (CMS):
Replaces traditional feed-forward networks with a chain of multi-frequency blocks.
Earlier blocks update at high frequencies, while deeper blocks update at lower frequencies.
This multi-time-scale update mimics biological brain waves and memory consolidation.

3. Deep Optimizers:
Treats optimization algorithms as associative memory modules that compress gradients.
Nested optimizers manage multi-tier updates across fast, medium, and slow layers.

Benefits of Nested Learning:
- Mitigates catastrophic forgetting during continual learning.
- Enables unbounded in-context adaptation.
- State-passing autoregressive inference runs in O(N) total time and O(1) per token.
- Operates efficiently on everyday consumer hardware without massive GPU clusters.
""" * 8


def main():
    print("=" * 70)
    print("NESTED LEARNING LITE: TRAINING DEMO")
    print("Architecture: HOPE (Self-Modifying Fast Memory + CMS Tiers)")
    print("Zero PyTorch Dependency — Pure NumPy")
    print("=" * 70)

    # 1. Tokenizer
    tok = nl.ByteTokenizer()
    tokens = tok.encode(TRAINING_CORPUS)
    print(f"Tokenized corpus: {len(tokens)} tokens (Vocab size: {tok.vocab_size})")

    # 2. Datasets
    seq_len = 32
    split_idx = int(len(tokens) * 0.85)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    train_dataset = nl.TextDataset(train_tokens, seq_len=seq_len, stride=8)
    val_dataset = nl.TextDataset(val_tokens, seq_len=seq_len, stride=8)
    print(f"Dataset sequences: {len(train_dataset)} train, {len(val_dataset)} val")

    # 3. Model Architecture
    d_model = 48
    n_layers = 4
    # CMS Tiers: 2 fast layers (period 1), 2 slow layers (period 4)
    cms_tiers = [[2, 1], [2, 4]]
    model = nl.HOPE(
        vocab_size=tok.vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        cms_tiers=cms_tiers,
    )
    print(f"Model parameters: {model.count_parameters():,} trainable weights")
    print(f"CMS Tiers config: {cms_tiers}")

    # 4. Deep Optimizer & Scheduler
    max_steps = 60
    learning_rate = 8e-3
    scheduler = nl.CosineAnnealingLR(base_lr=learning_rate, max_steps=max_steps, warmup_steps=5)
    optimizer = nl.NestedOptimizer(
        model.tier_param_groups(),
        lr=learning_rate,
        optimizer_cls=nl.AdamW,
        scheduler=scheduler,
        grad_clip=1.0,
    )
    loss_fn = nl.CrossEntropyLoss(ignore_index=tok.pad_token_id)

    # 5. Training
    checkpoint_file = "models/hope_model.npz"
    trainer = nl.Trainer(model, optimizer, loss_fn, tok)

    print("\nStarting training loop...")
    result = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=5,
        max_steps=max_steps,
        batch_size=4,
        accumulate_grad=1,
        eval_every_steps=15,
        checkpoint_path=checkpoint_file,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print(f"Training completed! Total steps: {result['total_steps']}")
    print(f"Best validation loss: {result['best_val_loss']:.4f}")
    print(f"Checkpoint saved to: {checkpoint_file}")
    print("=" * 70)

    # 6. Sample Generations
    print("\n--- SAMPLE GENERATION ---")
    test_prompts = [
        "Nested Learning is",
        "Key components of",
        "Benefits of",
    ]

    for p in test_prompts:
        p_tokens = tok.encode(p)
        gen_ids = model.generate(
            p_tokens,
            max_new_tokens=40,
            temperature=0.6,
            top_k=20,
            repetition_penalty=1.1,
        )
        gen_text = tok.decode(gen_ids)
        print(f"\n[Prompt]: {p}")
        print(f"[Generated]:\n{gen_text}\n" + "-" * 50)


if __name__ == "__main__":
    main()
