#!/usr/bin/env python3
"""Script untuk menguji dan membandingkan performa model HOPE sebelum dan sesudah belajar matematika.
Menguji inferensi murni neural network (tanpa CompositionalReasoner / tanpa hardcode).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import nested_learning as nl


def generate_response(model, tok, prompt: str, max_tokens: int = 40, temperature: float = 0.0) -> str:
    # Format prompt jika pertanyaan
    is_question = "?" in prompt or any(prompt.lower().startswith(q) for q in [
        "siapa", "apa", "dimana", "kapan", "mengapa", "kenapa", "bagaimana", "apakah", "berapa"
    ])
    if is_question:
        prompt_str = f"Pertanyaan: {prompt.strip()}\nJawaban: "
    else:
        prompt_str = prompt

    tokens = tok.encode(prompt_str)
    arr = np.asarray(tokens, dtype=np.int64)[np.newaxis, :]
    logits, state = model.forward(arr, state=None, last_only=True)
    last_logits = logits[0, -1, :].copy()

    gen_tokens = []
    rep_penalty = 1.35

    for _ in range(max_tokens):
        penalized = last_logits.copy()
        if gen_tokens:
            for t in set(gen_tokens[-24:]):
                if penalized[t] > 0:
                    penalized[t] /= rep_penalty
                else:
                    penalized[t] *= rep_penalty

        if temperature <= 1e-4:
            next_tok = int(np.argmax(penalized))
        else:
            scaled = penalized / temperature
            top_k = min(20, len(scaled))
            indices_to_remove = np.argsort(scaled)[:-top_k]
            scaled[indices_to_remove] = -1e9
            max_l = np.max(scaled)
            probs = np.exp(scaled - max_l)
            probs /= (np.sum(probs) + 1e-12)
            next_tok = int(np.random.choice(len(probs), p=probs))

        gen_tokens.append(next_tok)
        if next_tok == tok.eos_token_id or next_tok == 10:  # newline or EOS
            break
        ch = tok.decode([next_tok])
        if len(gen_tokens) >= 5 and ch in (".", "!", "?"):
            break

        x_step = np.array([[next_tok]], dtype=np.int64)
        step_logits, state = model.step(x_step, state)
        last_logits = step_logits[0, -1, :].copy()

    return tok.decode(gen_tokens).strip()


def run_comparison(base_ckpt: str, trained_ckpt: str, test_queries: list):
    print("=" * 70)
    print("PENGUJIAN BELAJAR MODEL HOPE (RAW NEURAL NETWORK INFERENCE)")
    print("Komposisional Reasoner: NONAKTIF (murni bobot synaptic model)")
    print("=" * 70)

    # 1. Base Model
    print(f"\n[1] Memuat Base Model: {base_ckpt}")
    m_base, _ = nl.HOPE.load_checkpoint(base_ckpt)
    tok_base = nl.resolve_tokenizer_for_checkpoint(base_ckpt)

    # 2. Trained Model
    print(f"[2] Memuat Trained Model: {trained_ckpt}")
    m_trained, _ = nl.HOPE.load_checkpoint(trained_ckpt)
    tok_trained = nl.resolve_tokenizer_for_checkpoint(trained_ckpt)

    print("\n" + "-" * 70)
    print(f"{'Prompt / Pertanyaan':<32} | {'Base Model (Sebelum)':<20} | {'Trained Model (Sesudah)'}")
    print("-" * 70)

    for q in test_queries:
        ans_base = generate_response(m_base, tok_base, q, max_tokens=35, temperature=0.0)
        ans_trained = generate_response(m_trained, tok_trained, q, max_tokens=35, temperature=0.0)
        
        # Bersihkan newline untuk display tabel
        ans_base_clean = ans_base.replace("\n", " ")[:20]
        ans_trained_clean = ans_trained.replace("\n", " ")[:35]
        print(f"{q:<32} | {ans_base_clean:<20} | {ans_trained_clean}")

    print("-" * 70)


if __name__ == "__main__":
    queries = [
        "Satu itu",
        "Dua itu",
        "Tiga itu",
        "Empat itu",
        "Lima itu",
        "Setelah 1 adalah",
        "Setelah 2 adalah",
        "Setelah 3 adalah",
        "Setelah 4 adalah",
        "Sebelum 2 adalah",
        "Sebelum 3 adalah",
        "Mata kita ada",
        "Jari satu tangan ada",
        "Kaki kita ada",
    ]
    
    base_model = "models/hope_infant.npz"
    trained_model = "models/hope_tutored.npz"
    
    run_comparison(base_model, trained_model, queries)
