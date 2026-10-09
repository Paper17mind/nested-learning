"""Interactive Console Chat and Fast-Memory Inspector for Nested Learning Lite.

Features:
- State-passing conversation: carry inner-loop fast memory across dialogue turns.
- /memory command: inspect inner associative memory matrix (Frobenius norm, sparsity, values).
- /reset command: clear fast memory.
- /temp and /tokens commands: adjust generation parameters on the fly.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import numpy as np
import nested_learning as nl


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Chat with Nested Learning Lite")
    parser.add_argument("--checkpoint", type=str, default="hope_model.npz", help="Path to .npz model checkpoint")
    parser.add_argument("--temperature", type=float, default=0.7, help="Default sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=50, help="Default max new tokens per reply")
    return parser.parse_args()


def inspect_memory(state: np.ndarray) -> None:
    """Print statistical analysis of the fast-weight memory matrix."""
    if state is None:
        print("[Memory]: Empty / Not initialized (Zeros)")
        return

    m = state[0]  # shape [D, D]
    f_norm = float(np.linalg.norm(m, ord="fro"))
    mean_val = float(np.mean(m))
    max_val = float(np.max(np.abs(m)))
    sparsity = float(np.mean(np.abs(m) < 1e-4) * 100.0)

    print("\n--- FAST-WEIGHT MEMORY MATRIX (M) ---")
    print(f"Shape:               {m.shape}")
    print(f"Frobenius Norm:      {f_norm:.4f}")
    print(f"Mean Value:          {mean_val:.6f}")
    print(f"Max Absolute Value:  {max_val:.4f}")
    print(f"Near-Zero Sparsity:  {sparsity:.1f}%")
    print(f"Sub-matrix preview (top-left 4x4):\n{m[:4, :4]}")
    print("-------------------------------------\n")


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint '{args.checkpoint}' not found.")
        print("Please train a model first: python train_demo.py")
        sys.exit(1)

    print("=" * 65)
    print("NESTED LEARNING LITE: INTERACTIVE CHAT & MEMORY INSPECTOR")
    print("Zero PyTorch Dependency — Pure NumPy")
    print("=" * 65)

    model, meta = nl.HOPE.load_checkpoint(args.checkpoint)
    cfg = model.get_config()
    print(f"Loaded: {model.count_parameters():,} parameters | Tiers: {cfg['cms_tiers']}")
    print("Commands:")
    print("  /memory      - Inspect fast-weight associative memory matrix")
    print("  /reset       - Reset fast memory state")
    print("  /temp <val>  - Set sampling temperature (current: {})".format(args.temperature))
    print("  /tokens <n>  - Set max generation tokens (current: {})".format(args.max_tokens))
    print("  /help        - Show available commands")
    print("  quit / exit  - Exit session")
    print("-" * 65)

    tok = nl.ByteTokenizer()
    temperature = args.temperature
    max_tokens = args.max_tokens
    state: np.ndarray = None

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd == "/memory":
                inspect_memory(state)
                continue
            elif cmd == "/reset":
                state = None
                print("[Memory]: Fast memory state reset to zero.")
                continue
            elif cmd == "/temp" and len(parts) > 1:
                try:
                    temperature = float(parts[1])
                    print(f"[Config]: Temperature set to {temperature}")
                except ValueError:
                    print("Invalid temperature value.")
                continue
            elif cmd == "/tokens" and len(parts) > 1:
                try:
                    max_tokens = int(parts[1])
                    print(f"[Config]: Max tokens set to {max_tokens}")
                except ValueError:
                    print("Invalid max tokens value.")
                continue
            elif cmd in ("/help", "/?"):
                print("Commands: /memory, /reset, /temp <float>, /tokens <int>, quit")
                continue
            else:
                print(f"Unknown command '{cmd}'. Type /help for options.")
                continue

        # Process prompt through state-passing inference
        prompt_tokens = tok.encode(user_input)
        prompt_arr = np.asarray(prompt_tokens, dtype=np.int64)[np.newaxis, :]  # [1, T]

        # Prefill prompt using existing state
        logits, state = model.forward(prompt_arr, state=state, last_only=True)
        last_logits = logits[0, -1, :].copy()

        gen_tokens = []
        t0 = time.time()

        for _ in range(max_tokens):
            if temperature <= 1e-4:
                next_tok = int(np.argmax(last_logits))
            else:
                scaled = last_logits / temperature
                # Top-20 filter
                top_k = min(20, len(scaled))
                indices_to_remove = np.argsort(scaled)[:-top_k]
                scaled[indices_to_remove] = -1e9

                max_l = np.max(scaled)
                probs = np.exp(scaled - max_l)
                probs /= (np.sum(probs) + 1e-12)
                next_tok = int(np.random.choice(len(probs), p=probs))

            gen_tokens.append(next_tok)

            # Advance state by single-token step
            x_next = np.array([[next_tok]], dtype=np.int64)
            logits_step, state = model.step(x_next, state)
            last_logits = logits_step[0, -1, :].copy()

        t1 = time.time()
        reply_text = tok.decode(gen_tokens)
        speed = len(gen_tokens) / max(1e-4, t1 - t0)

        print(f"HOPE > {reply_text.strip()}")
        print(f"[Stats]: {len(gen_tokens)} tokens generated in {t1 - t0:.3f}s ({speed:.1f} tok/s) | Memory norm: {float(np.linalg.norm(state[0])): .3f}")


if __name__ == "__main__":
    main()
