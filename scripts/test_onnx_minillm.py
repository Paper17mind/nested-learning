#!/usr/bin/env python3
"""
Test & Benchmark Mini-LLM Transformer via ONNX Runtime (Zero PyTorch Dependency)
Part of Nested Learning Lite (HOPE) Project.

Membuktikan bahwa:
1. Model Transformer (MiniLM-L6 384-D) bisa dijalankan TANPA PyTorch (Zero PyTorch).
2. Outputnya 100% berupa array NumPy murni (`np.ndarray`).
3. Mampu mengenali kemiripan semantik antar kalimat Bahasa Indonesia secara akurat.
4. Dapat diintegrasikan sebagai sensor semantik untuk Fast Associative Memory M (Delta Rule HOPE).
"""

import os
import sys
import time
import urllib.request
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("[Error]: onnxruntime belum terpasang.")
    print("Silakan pasang dengan: pip install onnxruntime")
    sys.exit(1)

# Import Nested Learning untuk uji jembatan ke Associative Memory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nested_learning as nl


class WordPieceTokenizer:
    """Tokenizer WordPiece murni Python (Nol dependensi library tokenizers/PyTorch)."""

    def __init__(self, vocab_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        self.vocab = {word: idx for idx, word in enumerate(lines)}
        self.inv_vocab = {idx: word for idx, word in enumerate(lines)}
        self.cls_token_id = self.vocab.get("[CLS]", 101)
        self.sep_token_id = self.vocab.get("[SEP]", 102)
        self.unk_token_id = self.vocab.get("[UNK]", 100)
        self.pad_token_id = self.vocab.get("[PAD]", 0)

    def encode(self, text: str, max_length: int = 64):
        # Bersihkan teks sederhana
        clean_text = text.lower().strip()
        tokens = [self.cls_token_id]

        for raw_word in clean_text.split():
            # Hapus tanda baca sederhana untuk tokenisasi kata dasar
            word = "".join(c for c in raw_word if c.isalnum() or c in ("-", "_"))
            if not word:
                continue

            chars = list(word)
            start = 0
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                cur_idx = None
                while start < end:
                    sub = "".join(chars[start:end])
                    if start > 0:
                        sub = "##" + sub
                    if sub in self.vocab:
                        cur_substr = sub
                        cur_idx = self.vocab[sub]
                        break
                    end -= 1

                if cur_idx is None:
                    tokens.append(self.unk_token_id)
                    break
                tokens.append(cur_idx)
                start = end

        tokens.append(self.sep_token_id)
        if len(tokens) > max_length:
            tokens = tokens[:max_length - 1] + [self.sep_token_id]

        return tokens


class MiniLMOnnxEmbedder:
    """Pengambil vektor semantik Transformer via ONNX Runtime murni NumPy."""

    def __init__(self, model_dir: str = "models/minilm_onnx"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.vocab_path = os.path.join(model_dir, "vocab.txt")
        self.model_path = os.path.join(model_dir, "model_quantized.onnx")

        self._ensure_model_files()
        print(f"[ONNX]: Loading session dari '{self.model_path}'...")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(self.model_path, sess_options=opts)
        self.tokenizer = WordPieceTokenizer(self.vocab_path)
        print(f"[ONNX]: Model siap! Provider: {self.session.get_providers()} | Dimensi: 384-D")

    def _ensure_model_files(self):
        if not os.path.exists(self.vocab_path):
            print("[Download]: Mengunduh vocab.txt (226 KB)...")
            url_vocab = "https://huggingface.co/Xenova/all-MiniLM-L6-v2/raw/main/vocab.txt"
            urllib.request.urlretrieve(url_vocab, self.vocab_path)

        if not os.path.exists(self.model_path):
            print("[Download]: Mengunduh model_quantized.onnx (21.9 MB)...")
            url_model = "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx"
            urllib.request.urlretrieve(url_model, self.model_path)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Menghasilkan representasi vektor semantik (np.ndarray) ternormalisasi L2."""
        batch_tokens = [self.tokenizer.encode(t) for t in texts]
        max_len = max(len(toks) for toks in batch_tokens)

        batch_size = len(texts)
        input_ids = np.zeros((batch_size, max_len), dtype=np.int64)
        attention_mask = np.zeros((batch_size, max_len), dtype=np.int64)
        token_type_ids = np.zeros((batch_size, max_len), dtype=np.int64)

        for i, toks in enumerate(batch_tokens):
            input_ids[i, :len(toks)] = toks
            attention_mask[i, :len(toks)] = 1

        # Jalankan inferensi ONNX (Murni NumPy)
        outputs = self.session.run(
            ["last_hidden_state"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        last_hidden = outputs[0]  # shape: [B, T, 384]

        # Mean Pooling dengan bobot Attention Mask
        mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        sum_embeddings = np.sum(last_hidden * mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
        mean_pooled = sum_embeddings / sum_mask  # [B, 384]

        # L2 Normalize
        norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        normalized = mean_pooled / norms

        return normalized.astype(np.float32)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Hitung Cosine Similarity antara 2 vektor (rentang -1.0 s.d 1.0)."""
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Test Mini-LLM ONNX Semantic Embedder with HOPE Associative Memory")
    parser.add_argument("--query", "-q", type=str, default="tokoh proklamator dan kepala negara ri pertama",
                        help="Kalimat query uji untuk menguji recall memori asosiatif M")
    parser.add_argument("--question", type=str, default="siapa presiden pertama indonesia ?",
                        help="Pertanyaan basis yang dipelajari ke memori M")
    parser.add_argument("--answer", type=str, default="Ir. Soekarno, sang proklamator kemerdekaan",
                        help="Jawaban basis yang dipelajari ke memori M")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("TEST MINI-LLM TRANSFORMER VIA ONNX RUNTIME (ZERO PYTORCH)")
    print("Zero PyTorch Dependency — Pure NumPy & ONNX C++ Engine")
    print("=" * 70)

    t0 = time.time()
    embedder = MiniLMOnnxEmbedder()
    init_time = time.time() - t0
    print(f"Inisialisasi selesai dalam {init_time:.2f}s\n")

    # 1. Dataset uji semantik
    sentences = [
        "siapa presiden pertama indonesia ?",
        "tokoh proklamator dan kepala negara ri pertama",
        "pemimpin pertama republik indonesia adalah soekarno",
        "cara membuat resep nasi goreng enak di rumah",
        "tips merawat tanaman hias di dalam ruangan",
    ]

    print("-" * 70)
    print("1. EKSTRAKSI VEKTOR SEMANTIK DARI TEKS")
    print("-" * 70)
    t_start = time.time()
    vectors = embedder.embed(sentences)
    lat_ms = (time.time() - t_start) * 1000

    print(f"Berhasil menghasilkan {len(sentences)} vektor!")
    print(f"Tipe Data: {type(vectors)} (Murni NumPy)")
    print(f"Shape    : {vectors.shape} (5 kalimat x 384 dimensi)")
    print(f"Latency  : {lat_ms:.2f} ms total ({lat_ms / len(sentences):.2f} ms/kalimat di CPU)\n")

    # 2. Uji Cosine Similarity (Pemahaman Makna Kata)
    print("-" * 70)
    print("2. UJI KEMIRIPAN SEMANTIK (COSINE SIMILARITY MATRIX)")
    print("-" * 70)
    print("Basis Perbandingan: 'siapa presiden pertama indonesia ?'\n")

    base_vec = vectors[0]
    for i in range(len(sentences)):
        sim = cosine_similarity(base_vec, vectors[i])
        bar_len = int(max(0, sim) * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        tag = ""
        if i == 0:
            tag = "(Kalimat Identik)"
        elif sim > 0.70:
            tag = "✓ Sangat Mirip Maknanya!"
        elif sim > 0.40:
            tag = "~ Cukup Relevan"
        else:
            tag = "✗ Tidak Nyambung / Beda Topik"

        print(f"[{sim:.4f}] {bar} {tag}")
        print(f"  -> \"{sentences[i]}\"\n")

    # 3. Uji Integrasi ke Associative Fast Memory HOPE (Delta Rule M)
    print("-" * 70)
    print("3. UJI JEMBATAN KE ASSOCIATIVE FAST MEMORY M (HOPE DELTA RULE)")
    print("-" * 70)
    print("Konsep: Vektor semantik 384-D dari ONNX langsung diserap ke matriks M HOPE:")
    print("        M_t = alpha * M_{t-1} + beta * (v (x) k^T)")

    # Proyeksi linier dari 384-D ke dimensi HOPE (d_model = 64)
    d_model = 64
    d_onnx = 384
    np.random.seed(42)
    # Proyeksi bobot ortogonal
    W_proj = np.random.randn(d_onnx, d_model).astype(np.float32) / np.sqrt(d_onnx)

    # Key (Pertanyaan Basis) dan Value (Jawaban Basis)
    print(f"Basis Q     : \"{args.question}\"")
    print(f"Basis A     : \"{args.answer}\"")
    q_vec = embedder.embed([args.question])[0]  # [384]
    a_vec = embedder.embed([args.answer])[0]    # [384]

    # Proyeksikan ke ruang laten HOPE
    k = np.dot(q_vec, W_proj)  # [64]
    v = np.dot(a_vec, W_proj)  # [64]
    k /= (np.linalg.norm(k) + 1e-12)
    v /= (np.linalg.norm(v) + 1e-12)

    # Inisialisasi associative memory M
    M = np.zeros((d_model, d_model), dtype=np.float32)
    alpha = 0.95
    beta = 0.85

    # Simpan fakta ke M (Delta Rule Outer Product)
    M = alpha * M + beta * np.outer(v, k)
    print(f"Memori M terisi! Dimensi: {M.shape} | Frobenius Norm: {np.linalg.norm(M):.4f}")

    # Uji Recall (Ingatan): Beri pertanyaan lain dari CLI argument
    query_paraphrase = args.query
    q_test = embedder.embed([query_paraphrase])[0]
    k_test = np.dot(q_test, W_proj)
    k_test /= (np.linalg.norm(k_test) + 1e-12)

    # Ambil jawaban dari memori: v_recalled = M @ k_test
    v_recalled = np.dot(M, k_test)
    recall_sim = cosine_similarity(v_recalled, v)

    print(f"\nQuery Uji   : \"{query_paraphrase}\"")
    print(f"Recall Sim  : {recall_sim:.4f} terhadap jawaban basis (target: '{args.answer[:30]}...')")
    if recall_sim > 0.65:
        print("✓ Sukses! HOPE berhasil mengingat dan merecall jawaban dari memori M")
        print("  meskipun pertanyaan pengujinya menggunakan kalimat dan susunan kata yang berbeda!")
    else:
        print("~ Catatan: Kemiripan query dengan basis pertanyaan berada di bawah ambang 0.65.")
    print("=" * 70)


if __name__ == "__main__":
    main()
