"""Command-line text generation for Nested Learning Lite (HOPE).

Usage:
    python generate.py --prompt "Nested Learning is" --max-tokens 50 --temperature 0.7
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import numpy as np
import nested_learning as nl


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using Nested Learning Lite (HOPE)")
    parser.add_argument("--checkpoint", type=str, default="hope_model.npz", help="Path to .npz model checkpoint")
    parser.add_argument("--prompt", type=str, default="Nested Learning is", help="Input prompt text")
    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K sampling (0 = disabled)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P nucleus sampling (0.0 = disabled)")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty factor")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint '{args.checkpoint}' not found.")
        print("Please train a model first with: python train_demo.py")
        sys.exit(1)

    print(f"Loading checkpoint from: {args.checkpoint}")
    model, meta = nl.HOPE.load_checkpoint(args.checkpoint)
    cfg = model.get_config()
    print(f"Model: {model.count_parameters():,} params | d_model={cfg['d_model']} | layers={cfg['n_layers']} | tiers={cfg['cms_tiers']}")

    tok = nl.ByteTokenizer()
    prompt_tokens = tok.encode(args.prompt)

    print(f"\nPrompt: '{args.prompt}' ({len(prompt_tokens)} tokens)")
    print(f"Sampling: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, rep_pen={args.repetition_penalty}")
    print("-" * 60)

    t0 = time.time()
    gen_ids = model.generate(
        prompt_tokens,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    t1 = time.time()

    gen_text = tok.decode(gen_ids)
    new_tokens_count = len(gen_ids) - len(prompt_tokens)
    elapsed = max(1e-4, t1 - t0)
    tok_per_sec = new_tokens_count / elapsed

    print(gen_text)
    print("-" * 60)
    print(f"Generated {new_tokens_count} tokens in {elapsed:.3f}s ({tok_per_sec:.1f} tokens/sec on CPU)")


if __name__ == "__main__":
    main()
