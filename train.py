"""Generic CLI Training Script for Nested Learning Lite (HOPE Architecture).

Train on your own dataset (.txt, .jsonl, or QA pairs) without PyTorch.

Usage:
    # 1. Train on plain text file:
    python train.py --data data/contoh_teks.txt --epochs 5 --batch-size 4

    # 2. Train on JSONL dataset:
    python train.py --data data/contoh_qa.jsonl --data-type qa --epochs 10

    # 3. Custom architecture:
    python train.py --data data/contoh_teks.txt --d-model 64 --n-layers 4 --cms-tiers "2,1:2,4"
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import numpy as np
import nested_learning as nl


def parse_tiers(tier_str: str, n_layers: int):
    """Parse string format like '2,1:2,4' into [[2, 1], [2, 4]]."""
    if not tier_str:
        return [[n_layers, 1]]
    tiers = []
    for part in tier_str.split(":"):
        count, period = part.split(",")
        tiers.append([int(count), int(period)])
    total = sum(t[0] for t in tiers)
    if total != n_layers:
        raise ValueError(f"Sum of tier counts ({total}) does not match n_layers ({n_layers}).")
    return tiers


def parse_args():
    parser = argparse.ArgumentParser(description="Train HOPE model on custom datasets")
    # Dataset arguments
    parser.add_argument("--data", type=str, required=True, help="Path to text or JSONL dataset file")
    parser.add_argument("--data-type", type=str, default="auto", choices=["auto", "text", "jsonl", "qa"],
                        help="Data format: 'auto', 'text', 'jsonl', or 'qa'")
    parser.add_argument("--text-key", type=str, default="text", help="JSON key containing text (for jsonl)")
    parser.add_argument("--question-key", type=str, default="question", help="Question key (for qa)")
    parser.add_argument("--answer-key", type=str, default="answer", help="Answer key (for qa)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation data ratio (0.0 - 0.5)")
    parser.add_argument("--seq-len", type=int, default=32, help="Context sequence length")
    parser.add_argument("--stride", type=int, default=8, help="Sliding window stride")

    # Architecture arguments
    parser.add_argument("--d-model", type=int, default=48, help="Model width (feature dimension)")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of CMS layers")
    parser.add_argument("--cms-tiers", type=str, default="2,1:2,4", help="CMS tiers format 'count,period:count,period'")
    parser.add_argument("--tokenizer", type=str, default="byte", choices=["byte", "char"], help="Tokenizer type")

    # Optimization arguments
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum training steps (optional)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--accumulate-grad", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=6e-3, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Warmup steps")
    parser.add_argument("--eval-every", type=int, default=15, help="Evaluation and logging frequency")
    parser.add_argument("--save-path", type=str, default="hope_model.npz", help="Output checkpoint file (.npz)")

    return parser.parse_args()


def load_dataset(args, tokenizer):
    filepath = args.data
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    data_type = args.data_type
    if data_type == "auto":
        if filepath.endswith(".jsonl") or filepath.endswith(".json"):
            # Check first line
            with open(filepath, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            if "question" in first_line and "answer" in first_line:
                data_type = "qa"
            else:
                data_type = "jsonl"
        else:
            data_type = "text"

    print(f"Loading dataset: '{filepath}' as type '{data_type}'...")

    if data_type == "text":
        ds = nl.TextDataset.from_file(filepath, tokenizer=tokenizer, seq_len=args.seq_len, stride=args.stride)
    elif data_type == "jsonl":
        ds = nl.TextDataset.from_jsonl(filepath, tokenizer=tokenizer, text_key=args.text_key, seq_len=args.seq_len, stride=args.stride)
    elif data_type == "qa":
        import json
        pairs = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
        ds = nl.TextDataset.from_qa_pairs(
            pairs,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            stride=args.stride,
            question_key=args.question_key,
            answer_key=args.answer_key,
        )
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    return ds


def main():
    args = parse_args()
    print("=" * 70)
    print("TRAINING NESTED LEARNING (HOPE) ON CUSTOM DATASET")
    print("=" * 70)

    # 1. Tokenizer
    tok = nl.get_tokenizer(args.tokenizer)
    print(f"Tokenizer: {tok.__class__.__name__} (Vocab size: {tok.vocab_size})")

    # 2. Dataset
    full_ds = load_dataset(args, tok)
    print(f"Total tokens: {len(full_ds.tokens):,} | Sequences: {len(full_ds)}")

    if args.val_ratio > 0.0 and len(full_ds.tokens) > 100:
        train_ds, val_ds = full_ds.train_val_split(val_ratio=args.val_ratio)
        print(f"Split: {len(train_ds.tokens):,} train tokens, {len(val_ds.tokens):,} val tokens")
    else:
        train_ds = full_ds
        val_ds = None
        print("Training on 100% data (no validation split)")

    # 3. Model Architecture
    cms_tiers = parse_tiers(args.cms_tiers, args.n_layers)
    model = nl.HOPE(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        cms_tiers=cms_tiers,
    )
    print(f"HOPE Model: {model.count_parameters():,} params | d_model={args.d_model} | layers={args.n_layers}")
    print(f"CMS Tiers: {cms_tiers}")

    # 4. Optimizer & Scheduler
    scheduler = nl.CosineAnnealingLR(
        base_lr=args.lr,
        max_steps=args.max_steps if args.max_steps else (len(train_ds) * args.epochs // args.batch_size + 1),
        warmup_steps=args.warmup_steps,
    )
    optimizer = nl.NestedOptimizer(
        model.tier_param_groups(),
        lr=args.lr,
        optimizer_cls=nl.AdamW,
        scheduler=scheduler,
        grad_clip=1.0,
    )
    loss_fn = nl.CrossEntropyLoss(ignore_index=tok.pad_token_id)

    # 5. Training Loop
    trainer = nl.Trainer(model, optimizer, loss_fn, tok)
    print("\nStarting training...")
    result = trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        accumulate_grad=args.accumulate_grad,
        eval_every_steps=args.eval_every,
        checkpoint_path=args.save_path,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print(f"Training finished! Steps: {result['total_steps']}")
    if result.get("best_val_loss") and result["best_val_loss"] != float("inf"):
        print(f"Best Val Loss: {result['best_val_loss']:.4f}")
    print(f"Saved Checkpoint: {args.save_path}")
    print("=" * 70)

    # 6. Sample Generation Test
    sample_prompt = full_ds.tokens[:min(12, len(full_ds.tokens))]
    prompt_text = tok.decode(sample_prompt)
    print(f"\nGenerating preview from prompt: '{prompt_text}'")
    gen_ids = model.generate(sample_prompt, max_new_tokens=30, temperature=0.7)
    print(f"Output:\n{tok.decode(gen_ids)}\n")


if __name__ == "__main__":
    main()
