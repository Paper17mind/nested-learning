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
    parser.add_argument("--checkpoint", type=str, default="models/hope_model.npz", help="Path to .npz model checkpoint")
    parser.add_argument("--temperature", type=float, default=0.2, help="Default sampling temperature")
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

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path) and os.path.exists(os.path.join("models", ckpt_path)):
        ckpt_path = os.path.join("models", ckpt_path)
    elif not os.path.exists(ckpt_path) and os.path.exists(os.path.basename(ckpt_path)):
        ckpt_path = os.path.basename(ckpt_path)

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint '{args.checkpoint}' not found.")
        print("Please train a model first: python scripts/train_demo.py or python train.py")
        sys.exit(1)

    model, meta = nl.HOPE.load_checkpoint(ckpt_path)
    current_checkpoint = ckpt_path

    print("=" * 65)
    print("NESTED LEARNING LITE: INTERACTIVE CHAT & MEMORY INSPECTOR")
    print("Zero PyTorch Dependency — Pure NumPy")
    print("=" * 65)

    cfg = model.get_config()
    print(f"Loaded: {model.count_parameters():,} parameters | Tiers: {cfg['cms_tiers']}")
    print("Commands:")
    print("  /memory         - Inspect fast-weight associative memory matrix (Frobenius norm, sparsity)")
    print("  /session <name> - Switch to or create a named memory session in SQLite (e.g., /session budi)")
    print("  /sessions       - List all active memory sessions in SQLite")
    print("  /save [path]    - Save current model weights AND memory matrix to file (.npz)")
    print("  /load <path>    - Load model weights AND restore saved memory state")
    print("  /learn <txt>    - Immediately train model on new fact/text (in-chat continual learning)")
    print("  /reset          - Reset fast memory state to zero")
    print("  /temp <val>     - Set sampling temperature (current: {})".format(args.temperature))
    print("  /tokens <n>     - Set max generation tokens (current: {})".format(args.max_tokens))
    print("  /help           - Show available commands")
    print("  quit / exit     - Exit session")
    print("-" * 65)

    tok = nl.ByteTokenizer()
    temperature = args.temperature
    max_tokens = args.max_tokens
    state: np.ndarray = meta.get("memory_state")
    if state is not None:
        print(f"[Memory]: Restored fast-weight memory state from checkpoint (norm: {float(np.linalg.norm(state[0])):.3f})")
    current_session = "default"
    db_store = nl.SQLiteMemoryStore("data/memory.db")
    optimizer = nl.NestedOptimizer(model.tier_param_groups(), lr=3e-3, optimizer_cls=nl.AdamW)
    loss_fn = nl.CrossEntropyLoss(ignore_index=tok.pad_token_id)
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
            elif cmd == "/session" and len(parts) > 1:
                new_sess = parts[1].strip()
                if state is not None:
                    db_store.save_memory(current_session, state, meta={"checkpoint": current_checkpoint})
                    print(f"[SQLite]: Saved active memory to session '{current_session}'.")
                current_session = new_sess
                res = db_store.load_memory(current_session)
                if res is not None:
                    state, _ = res
                    print(f"[SQLite]: Switched to session '{current_session}'. Restored memory (norm: {float(np.linalg.norm(state[0])):.3f}).")
                else:
                    state = None
                    print(f"[SQLite]: Switched to new session '{current_session}'. Starting fresh.")
                continue
            elif cmd == "/sessions":
                all_sess = db_store.list_sessions()
                print("\n--- ACTIVE SQLITE SESSIONS ---")
                if not all_sess:
                    print("No saved sessions in memory.db yet.")
                else:
                    for s in all_sess:
                        active_flag = " (ACTIVE)" if s["session_id"] == current_session else ""
                        print(f"- {s['session_id']:15s} | shape: {s['shape']} | norm: {s['norm']:.4f}{active_flag}")
                print("------------------------------\n")
                continue
            elif cmd == "/save":
                save_path = parts[1] if len(parts) > 1 else current_checkpoint
                if not os.path.dirname(save_path):
                    save_path = os.path.join("models", save_path)
                model.save_checkpoint(save_path, memory_state=state, extra_meta={"note": "chat_session"})
                print(f"[Saved]: Model weights and memory matrix saved to '{save_path}'.")
                continue
            elif cmd == "/load" and len(parts) > 1:
                load_path = parts[1]
                if not os.path.exists(load_path) and os.path.exists(os.path.join("models", load_path)):
                    load_path = os.path.join("models", load_path)
                if not os.path.exists(load_path):
                    print(f"File not found: '{parts[1]}' (also checked models/{parts[1]})")
                    continue
                model, meta = nl.HOPE.load_checkpoint(load_path)
                state = meta.get("memory_state")
                current_checkpoint = load_path
                print(f"[Loaded]: Checkpoint '{load_path}' loaded." + (f" Restored memory state (norm: {float(np.linalg.norm(state[0])):.3f})" if state is not None else " (Memory was empty)"))
                continue
            elif cmd == "/learn":
                learn_text = " ".join(parts[1:]).strip()
                if not learn_text:
                    print("Usage: /learn <text to memorize>")
                    continue
                learn_tokens = tok.encode(learn_text)
                if len(learn_tokens) < 3:
                    print("Text too short to train (minimum 3 characters).")
                    continue
                inp = np.array([learn_tokens[:-1]], dtype=np.int64)
                tgt = np.array([learn_tokens[1:]], dtype=np.int64)
                optimizer.zero_grad()
                logits, _ = model.forward(inp)
                loss = loss_fn.forward(logits, tgt)
                model.backward(loss_fn.backward())
                optimizer.step()
                print(f"[Learned]: Updated model weights on '{learn_text}' (Loss: {loss:.4f})")
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
                print("Commands: /memory, /save [path], /load <path>, /learn <text>, /reset, /temp <float>, /tokens <int>, quit")
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
