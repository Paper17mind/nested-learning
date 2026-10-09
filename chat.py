"""Interactive Console Chat and Fast-Memory Inspector for Nested Learning Lite.

Features:
- State-passing conversation: carry inner-loop fast memory across dialogue turns.
- /memory command: inspect inner associative memory matrix (Frobenius norm, sparsity, values).
- /reset command: clear fast memory.
- /temp and /tokens commands: adjust generation parameters on the fly.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import numpy as np
import nested_learning as nl


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Chat with Nested Learning Lite")
    parser.add_argument("--checkpoint", type=str, default="models/hope_tutored.npz", help="Path to .npz model checkpoint")
    parser.add_argument("--temperature", type=float, default=0.2, help="Default sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=250, help="Default max new tokens per reply")
    parser.add_argument("--session", type=str, default="default", help="Session ID in SQLite database (default: 'default')")
    parser.add_argument("--autolearn", action="store_true", default=True, help="Enable active auto-ask and auto-learn when uncertain (default: True)")
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
    print("  /autolearn [on|off] - Toggle auto-ask & auto-learn when model is uncertain (default: on)")
    print("  /reset          - Reset fast memory state to zero")
    print("  /temp <val>     - Set sampling temperature (current: {})".format(args.temperature))
    print("  /tokens <n>     - Set max generation tokens (current: {})".format(args.max_tokens))
    print("  /help           - Show available commands")
    print("  quit / exit     - Exit session")
    print("-" * 65)

    tok = nl.resolve_tokenizer_for_checkpoint(ckpt_path)
    print(f"Tokenizer: {tok.__class__.__name__} (Vocab size: {tok.vocab_size})")
    temperature = args.temperature
    max_tokens = args.max_tokens
    state: np.ndarray = meta.get("memory_state")
    if state is not None:
        if state.shape[-1] == model.d_model and state.shape[-2] == model.d_model:
            print(f"[Memory]: Restored fast-weight memory state from checkpoint (norm: {float(np.linalg.norm(state[0])):.3f})")
        else:
            print(f"[Memory]: Checkpoint memory state shape {state.shape} incompatible with d_model={model.d_model}. Starting fresh.")
            state = None

    current_session = args.session.strip() if args.session and args.session.strip() else "default"
    db_store = nl.SQLiteMemoryStore("data/memory.db")
    if state is None:
        saved_db = db_store.load_memory(current_session)
        if saved_db is not None:
            db_state, db_meta = saved_db
            if db_state.shape[-1] == model.d_model and db_state.shape[-2] == model.d_model:
                state = db_state
                print(f"[SQLite]: Restored active memory for session '{current_session}' from data/memory.db (norm: {float(np.linalg.norm(state[0])):.3f})")
            else:
                print(f"[SQLite]: Stored session '{current_session}' has dimension {db_state.shape[-1]}, but model has d_model={model.d_model}. Starting fresh memory.")
                state = None
    optimizer = nl.NestedOptimizer(model.tier_param_groups(), lr=3e-3, optimizer_cls=nl.AdamW)
    loss_fn = nl.CrossEntropyLoss(ignore_index=tok.pad_token_id)
    auto_learn_enabled = getattr(args, "autolearn", True)
    pending_question = None
    last_learned_record = None
    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            if state is not None:
                db_store.save_memory(current_session, state, meta={"last_action": "interrupt", "checkpoint": current_checkpoint})
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            if state is not None:
                db_store.save_memory(current_session, state, meta={"last_action": "exit", "checkpoint": current_checkpoint})
            print("Goodbye!")
            break

        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd == "/memory":
                inspect_memory(state)
            elif cmd == "/session":
                new_sess = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "default"
                if state is not None:
                    db_store.save_memory(current_session, state, meta={"checkpoint": current_checkpoint})
                    print(f"[SQLite]: Saved active memory to session '{current_session}'.")
                current_session = new_sess
                res = db_store.load_memory(current_session)
                if res is not None:
                    db_state, _ = res
                    if db_state.shape[-1] == model.d_model and db_state.shape[-2] == model.d_model:
                        state = db_state
                        print(f"[SQLite]: Switched to session '{current_session}'. Restored memory (norm: {float(np.linalg.norm(state[0])):.3f}).")
                    else:
                        state = None
                        print(f"[SQLite]: Switched to session '{current_session}'. Stored memory dimension {db_state.shape[-1]} mismatch with d_model={model.d_model}. Started fresh memory.")
                else:
                    state = None
                    print(f"[SQLite]: Switched to session '{current_session}' (starting fresh).")
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
                if state is not None and (state.shape[-1] != model.d_model or state.shape[-2] != model.d_model):
                    print(f"[Warning]: Checkpoint memory state shape {state.shape} incompatible with d_model={model.d_model}. Resetting memory.")
                    state = None
                current_checkpoint = load_path
                tok = nl.resolve_tokenizer_for_checkpoint(load_path)
                loss_fn = nl.CrossEntropyLoss(ignore_index=tok.pad_token_id)
                optimizer = nl.NestedOptimizer(model.tier_param_groups(), lr=3e-3, optimizer_cls=nl.AdamW)
                print(f"[Loaded]: Checkpoint '{load_path}' loaded. Tokenizer: {tok.__class__.__name__} ({tok.vocab_size} vocab)." + (f" Restored memory state (norm: {float(np.linalg.norm(state[0])):.3f})" if state is not None else " (Memory was empty)"))
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
                log_dir = "logs"
                os.makedirs(log_dir, exist_ok=True)
                with open(os.path.join(log_dir, "in_chat_learning.log"), "a", encoding="utf-8") as f:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] session={current_session} loss={loss:.4f} text='{learn_text}'\n")
                continue
            elif cmd == "/reset":
                state = None
                db_store.delete_memory(current_session)
                print("[Memory]: Fast memory state reset to zero and cleared from SQLite.")
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
            elif cmd == "/autolearn":
                if len(parts) > 1 and parts[1].lower() in ("off", "false", "0"):
                    auto_learn_enabled = False
                    pending_question = None
                    print("[Auto-Learn]: Dimatikan (Disabled).")
                else:
                    auto_learn_enabled = True
                    print("[Auto-Learn]: Diaktifkan (Enabled). Model akan bertanya balik saat belum tahu dan otomatis menyimpan jawaban Anda.")
                continue
            elif cmd in ("/help", "/?"):
                print("Commands: /memory, /session <name>, /sessions, /save [path], /load <path>, /learn <text>, /autolearn [on|off], /reset, /temp <float>, /tokens <int>, quit")
                continue
            else:
                print(f"Unknown command '{cmd}'. Type /help for options.")
                continue

        # 1. Alur Auto-Learning: Cek apakah user sedang menjawab pertanyaan model sebelumnya
        if pending_question is not None and auto_learn_enabled:
            answer_text = user_input.strip()
            qa_pair = f"Pertanyaan: {pending_question}\nJawaban: {answer_text}\n\n"
            learn_tokens = tok.encode(qa_pair)
            if len(learn_tokens) >= 4:
                inp = np.array([learn_tokens[:-1]], dtype=np.int64)
                tgt = np.array([learn_tokens[1:]], dtype=np.int64)

                # Micro-training cepat hingga konvergen (25 steps atau loss < 0.20)
                fast_opt = nl.NestedOptimizer(model.tier_param_groups(), lr=0.012, optimizer_cls=nl.AdamW)
                micro_loss = 0.0
                for step in range(25):
                    fast_opt.zero_grad()
                    logits, state = model.forward(inp, state=state)
                    loss_val = loss_fn.forward(logits, tgt)
                    model.backward(loss_fn.backward())
                    fast_opt.step()
                    micro_loss = float(loss_val)
                    if micro_loss < 0.20:
                        break

                # Simpan ke dataset permanen data/user_learned_qa.jsonl
                os.makedirs("data", exist_ok=True)
                entry = {
                    "question": pending_question,
                    "answer": answer_text,
                    "session": current_session,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                with open("data/user_learned_qa.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

                # Auto-save ke database SQLite
                if state is not None:
                    db_store.save_memory(
                        session_id=current_session,
                        state=state,
                        meta={
                            "type": "auto_learned",
                            "last_q": pending_question[:100],
                            "last_a": answer_text[:100],
                            "checkpoint": current_checkpoint,
                        },
                    )

                last_learned_record = {"q": pending_question, "a": answer_text}
                print(f"HOPE > Terima kasih! Aku sudah mencatat dan mempelajari jawabannya ke memoriku:")
                print(f"       • Pertanyaan : \"{pending_question}\"")
                print(f"       • Jawaban    : \"{answer_text}\"")
                print(f"[Auto-Learned]: Memory norm={float(np.linalg.norm(state[0])):.3f} | Loss: {micro_loss:.4f} (Langkah: {step+1})")
                print(f"                Tersimpan di 'data/user_learned_qa.jsonl' & 'data/memory.db'. Coba tanyakan lagi!")
                pending_question = None
                continue

        # 2. Proses Prompt Melalui State-Passing Inference
        import re
        is_question = "?" in user_input or any(user_input.lower().startswith(q) for q in [
            "siap", "siapa", "apa", "dimana", "di mana", "kapan", "mengapa", "kenapa", "bagaimana", "apakah", "berapa"
        ])

        if is_question:
            prompt_str = f"Pertanyaan: {user_input.strip()}\nJawaban: "
        else:
            prompt_str = user_input

        prompt_tokens = tok.encode(prompt_str)
        prompt_arr = np.asarray(prompt_tokens, dtype=np.int64)[np.newaxis, :]  # [1, T]

        # Prefill prompt using existing state
        logits, state = model.forward(prompt_arr, state=state, last_only=True)
        last_logits = logits[0, -1, :].copy()

        # Cek confidence skor awal
        l_exp = np.exp(last_logits - np.max(last_logits))
        probs_first = l_exp / (np.sum(l_exp) + 1e-12)
        top_prob = float(np.max(probs_first))

        gen_tokens = []
        t0 = time.time()
        rep_penalty = 1.35

        for _ in range(max_tokens):
            logits_penalized = last_logits.copy()
            if len(gen_tokens) > 0:
                recent_window = gen_tokens[-24:]
                for prev_tok in set(recent_window):
                    if logits_penalized[prev_tok] > 0:
                        logits_penalized[prev_tok] /= rep_penalty
                    else:
                        logits_penalized[prev_tok] *= rep_penalty

            if temperature <= 1e-4:
                next_tok = int(np.argmax(logits_penalized))
            else:
                scaled = logits_penalized / temperature
                # Top-20 filter
                top_k = min(20, len(scaled))
                indices_to_remove = np.argsort(scaled)[:-top_k]
                scaled[indices_to_remove] = -1e9

                max_l = np.max(scaled)
                probs = np.exp(scaled - max_l)
                probs /= (np.sum(probs) + 1e-12)
                next_tok = int(np.random.choice(len(probs), p=probs))

            gen_tokens.append(next_tok)

            if next_tok == tok.eos_token_id:
                break
            if len(gen_tokens) > 5:
                tail = tok.decode(gen_tokens[-8:])
                if "\n" in tail or "\nPertanyaan:" in tail:
                    break

            # Advance state by single-token step
            x_next = np.array([[next_tok]], dtype=np.int64)
            logits_step, state = model.step(x_next, state)
            last_logits = logits_step[0, -1, :].copy()

        t1 = time.time()
        raw_reply = tok.decode(gen_tokens)
        if is_question:
            # Ambil jawaban pertama sebelum baris baru
            first_line = raw_reply.split("\n")[0].strip()
            reply_text = first_line if first_line else raw_reply.strip()
        else:
            reply_text = raw_reply.strip()
        speed = len(gen_tokens) / max(1e-4, t1 - t0)

        # 3. Deteksi Ketidaktahuan: Jika user bertanya dan model ragu / meracau (degeneration)
        words = reply_text.lower().split()
        unique_ratio = (len(set(words)) / len(words)) if len(words) >= 4 else 1.0
        max_word_rep = max([words.count(w) for w in set(words)]) if words else 0
        is_repetitive_loop = (len(words) >= 6 and (unique_ratio < 0.50 or max_word_rep >= 4))

        # Cek apakah jawaban mengandung informasi dari yang baru saja diajarkan
        matches_recently_learned = False
        if last_learned_record is not None:
            ans_keywords = [w for w in last_learned_record["a"].lower().split() if len(w) >= 3]
            if any(kw in reply_text.lower() for kw in ans_keywords):
                matches_recently_learned = True

        is_uncertain = (
            (len(reply_text) < 3 or
             bool(re.search(r"(.)\1{4,}", reply_text)) or
             is_repetitive_loop or
             top_prob < 0.16 or
             reply_text.startswith("?") or
             "tidak tahu" in reply_text.lower())
            and not matches_recently_learned
        )

        if is_question and auto_learn_enabled and is_uncertain:
            clean_q = user_input.strip()
            print(f"HOPE > Maaf, aku belum tahu tentang itu. Boleh tolong beri tahu aku jawabannya agar aku bisa mengingatnya?")
            pending_question = clean_q
            continue

        print(f"HOPE > {reply_text}")
        print(f"[Stats]: {len(gen_tokens)} tokens generated in {t1 - t0:.3f}s ({speed:.1f} tok/s) | Memory norm: {float(np.linalg.norm(state[0])): .3f}")
        # Auto-persist memory state to SQLite on every turn
        if state is not None:
            db_store.save_memory(
                session_id=current_session,
                state=state,
                meta={
                    "last_prompt": user_input[:120],
                    "last_reply": reply_text[:120],
                    "checkpoint": current_checkpoint,
                },
            )

if __name__ == "__main__":
    main()
