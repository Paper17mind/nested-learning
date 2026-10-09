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

import json
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
    parser.add_argument("--val-data", type=str, default=None, help="File JSONL QA terpisah untuk validasi")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to text or JSONL dataset file(s). Supports comma-separated files or directory.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to existing model checkpoint (.npz) to continue training (Continual Learning). "
                             "If omitted, a new model is initialized from scratch.")
    parser.add_argument("--data-type", type=str, default="auto", choices=["auto", "text", "jsonl", "qa", "pdf"],
                        help="Data format: 'auto', 'text', 'jsonl', 'qa', or 'pdf'")
    parser.add_argument("--text-key", type=str, default="text", help="JSON key containing text (for jsonl)")
    parser.add_argument("--question-key", type=str, default="question", help="Question key (for qa)")
    parser.add_argument("--answer-key", type=str, default="answer", help="Answer key (for qa)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation data ratio (0.0 - 0.5)")
    parser.add_argument("--seq-len", type=int, default=32, help="Context sequence length")
    parser.add_argument("--stride", type=int, default=None,
                        help="Sliding window stride (default: None = auto seq_len for non-overlapping fast training)")
    parser.add_argument("--threads", type=int, default=0,
                        help="Limit number of CPU threads used by NumPy BLAS (e.g. 2, 4). 0 = use all CPU cores")
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
    parser.add_argument("--save-path", type=str, default="models/hope_model.npz", help="Output checkpoint file (.npz)")

    return parser.parse_args()


def load_single_file(filepath, data_type, args, tokenizer):
    stride = args.stride if args.stride is not None else args.seq_len
    if data_type == "pdf" or filepath.lower().endswith(".pdf"):
        return nl.TextDataset.from_pdf(filepath, tokenizer=tokenizer, seq_len=args.seq_len, stride=stride)
    elif data_type == "text":
        return nl.TextDataset.from_file(filepath, tokenizer=tokenizer, seq_len=args.seq_len, stride=stride)
    elif data_type == "jsonl":
        return nl.TextDataset.from_jsonl(filepath, tokenizer=tokenizer, text_key=args.text_key, seq_len=args.seq_len, stride=stride)
    elif data_type == "qa":
        import json
        pairs = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
        return nl.TextDataset.from_qa_pairs(
            pairs,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            stride=stride,
            question_key=args.question_key,
            answer_key=args.answer_key,
        )
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def load_dataset(args, tokenizer):
    import glob
    raw_paths = [p.strip() for p in args.data.split(",") if p.strip()]
    file_list = []

    for p in raw_paths:
        if os.path.isdir(p):
            for ext in ("*.txt", "*.md", "*.jsonl", "*.pdf"):
                file_list.extend(glob.glob(os.path.join(p, ext)))
        elif "*" in p:
            file_list.extend(glob.glob(p))
        elif os.path.exists(p):
            file_list.append(p)
        else:
            raise FileNotFoundError(f"Dataset path not found: {p}")

    if not file_list:
        raise ValueError(f"No valid dataset files found for: {args.data}")

    datasets = []
    for fp in file_list:
        dt = args.data_type
        if dt == "auto":
            if fp.lower().endswith(".pdf"):
                dt = "pdf"
            elif fp.lower().endswith(".jsonl") or fp.lower().endswith(".json"):
                with open(fp, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                if "question" in first_line and "answer" in first_line:
                    dt = "qa"
                else:
                    dt = "jsonl"
            else:
                dt = "text"
        print(f"Loading dataset item: '{fp}' (type: {dt})")
        datasets.append(load_single_file(fp, dt, args, tokenizer))

    if len(datasets) == 1:
        return datasets[0]
    else:
        print(f"Merging {len(datasets)} datasets into one combined corpus...")
        return nl.TextDataset.concat(datasets)

def build_char_tokenizer_from_qa(path, question_key, answer_key):
    import json

    texts = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            q = str(item.get(question_key, ""))
            a = str(item.get(answer_key, ""))

            texts.append(
                f"Pertanyaan: {q}\nJawaban: {a}\n\n"
            )

    if not texts:
        raise ValueError("Dataset training kosong.")

    corpus = "".join(texts)
    return nl.CharTokenizer.train_from_text(corpus)

def main():
    args = parse_args()
    if args.threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
        print(f"Limiting CPU usage to {args.threads} thread(s).")
    print("=" * 70)
    print("TRAINING NESTED LEARNING (HOPE) ON CUSTOM DATASET")
    print("=" * 70)

    # 1. Tokenizer
    if args.tokenizer == "char":
        tok = build_char_tokenizer_from_qa(
            args.data,
            args.question_key,
            args.answer_key,
        )
    else:
        tok = nl.get_tokenizer(args.tokenizer)

    print(
        f"Tokenizer: {tok.__class__.__name__} "
        f"| Vocab: {tok.vocab_size}"
    )

    # 2. Dataset
    full_ds = load_dataset(args, tok)
    if args.val_data:
        val_args = argparse.Namespace(**vars(args))
        val_args.data = args.val_data
        val_args.data_type = "qa"

        train_ds = full_ds
        val_ds = load_dataset(val_args, tok)

        print(
            f"Train sequences: {len(train_ds)} "
            f"| Validation sequences: {len(val_ds)}"
        )
    else:
        train_ds, val_ds = full_ds.train_val_split(
            val_ratio=args.val_ratio
    )

    print(f"Total tokens: {len(full_ds.tokens):,} | Sequences: {len(full_ds)}")

    if args.val_ratio > 0.0 and len(full_ds.tokens) > 100:
        train_ds, val_ds = full_ds.train_val_split(val_ratio=args.val_ratio)
        print(f"Split: {len(train_ds.tokens):,} train tokens, {len(val_ds.tokens):,} val tokens")
    else:
        train_ds = full_ds
        val_ds = None
        print("Training on 100% data (no validation split)")

    # 3. Model Architecture & Checkpoint Resume
    if args.checkpoint:
        ckpt_path = args.checkpoint
        if not os.path.exists(ckpt_path) and os.path.exists(os.path.join("models", ckpt_path)):
            ckpt_path = os.path.join("models", ckpt_path)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint} (also checked models/{args.checkpoint})")
        print(f"\n[Continual Learning Mode] Resuming model from checkpoint: '{ckpt_path}'")
        model, meta = nl.HOPE.load_checkpoint(ckpt_path)
        cfg = model.get_config()
        print(f"HOPE Model: {model.count_parameters():,} params | d_model={cfg['d_model']} | layers={cfg['n_layers']}")
        print(f"CMS Tiers (retained from checkpoint): {cfg['cms_tiers']}")
    else:
        print("\n[Fresh Initialization] Initializing brand new HOPE model from scratch...")
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
    save_path = args.save_path
    tokenizer_path = save_path + ".tokenizer.json"
    if not os.path.dirname(save_path):
        save_path = os.path.join("models", save_path)
    
    tok.save(tokenizer_path)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    trainer = nl.Trainer(model, optimizer, loss_fn, tok)
    print(f"\nStarting training (output will save to '{save_path}')...")
    result = trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        accumulate_grad=args.accumulate_grad,
        eval_every_steps=args.eval_every,
        checkpoint_path=save_path,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print(f"Training finished! Steps: {result['total_steps']}")
    if result.get("best_val_loss") and result["best_val_loss"] != float("inf"):
        print(f"Best Val Loss: {result['best_val_loss']:.4f}")
    print(f"Saved Checkpoint: {save_path}")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": args.data,
        "total_tokens": len(full_ds.tokens),
        "total_steps": result["total_steps"],
        "epochs": args.epochs,
        "best_val_loss": result.get("best_val_loss") if result.get("best_val_loss") != float("inf") else None,
        "checkpoint": save_path,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
    }
    with open(os.path.join(log_dir, "training_runs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    print("Log dicatat ke: logs/training_runs.jsonl")
    print("=" * 70)

    # 6. Sample Generation Test
    sample_prompt = full_ds.tokens[:min(12, len(full_ds.tokens))]
    prompt_text = tok.decode(sample_prompt)
    print(f"\nGenerating preview from prompt: '{prompt_text}'")
    gen_ids = model.generate(sample_prompt, max_new_tokens=30, temperature=0.7)
    print(f"Output:\n{tok.decode(gen_ids)}\n")


if __name__ == "__main__":
    main()
