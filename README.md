# Nested Learning Lite 🧠⚡

> **Implementasi eksperimental murni NumPy (Zero PyTorch Dependency) yang *terinspirasi oleh* paradigma *Nested Learning* dan arsitektur *HOPE* (Google Research, NeurIPS 2025).**

[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Dependency: Pure NumPy](https://img.shields.io/badge/dependencies-NumPy%20only-green.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> [!NOTE]
> **Disclaimer & Batasan:**
> Proyek ini **bukan merupakan implementasi resmi atau replika 1:1** dari arsitektur HOPE maupun kode rilis Google Research. Proyek ini adalah **interpretasi independen & eksplorasi edukasional murni NumPy** yang mengadopsi prinsip-prinsip inti Nested Learning: perpaduan *Fast-Weight Associative Memory* (melalui inner-loop SGD delta rule per token) dan *Slow-Weight Continuum Memory System* (pembaruan gradien multi-frekuensi). Seluruh layer dirancang seringan mungkin agar dapat dipelajari, dilatih, dan dijalankan di CPU biasa tanpa pustaka deep learning eksternal.

Referensi ilmiah & riset inspirasi:
1. **Google Research Blog:** [Introducing Nested Learning: A new ML paradigm for continual learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)
2. **Paper Riset:** *Nested Learning: The Illusion of Deep Learning Architectures* (Behrouz, Razaviyayn, Zhong, Mirrokni — NeurIPS 2025)
3. **Reference Repo (PyTorch):** [obekt/HOPE-nested-learning](https://github.com/obekt/HOPE-nested-learning)

---

## 🌟 Mengapa Nested Learning Lite?

Model bahasa standar (LLM) mengalami **"Anterograde Amnesia"** dan **"Catastrophic Forgetting"** (lupa katastropik):
- Sekali selesai pre-training, parameter model dibekukan (*frozen*).
- Ketika di-fine-tune dengan data atau tugas baru, representasi lama tertimpa secara agresif.

**Nested Learning membalik paradigma ini:**
Arsitektur neural network dan algoritma optimasi bukanlah dua hal terpisah, melainkan **masalah optimasi multi-level yang saling bersarang (nested) dengan frekuensi pembaruan yang berbeda**.

Versi **Lite** ini dibuat sebagai studi kasus / eksplorasi agar:
- ✅ **100% Bebas PyTorch / CUDA:** Hanya menggunakan Python standar dan NumPy.
- ✅ **Sangat Ringan & Portabel:** Berjalan mulus di laptop standar, Raspberry Pi, VPS spek rendah, atau CPU apa pun tanpa instalasi gigabyte dependencies.
- ✅ **Backpropagation Analitis Lengkap:** Turunan eksak (*closed-form exact analytical backward*) untuk semua layer, termasuk BPTT (Backpropagation Through Time) pada *gated delta rule* memori dinamis.
- ✅ **Inferensi $O(1)$ per Token ($O(N)$ Total):** Menggunakan *state-passing* autoregressive inference dengan kecepatan **>800–1300 tokens/detik di CPU**.
- ✅ **Mitigasi Catastrophic Forgetting:** Dilengkapi sistem *Continuum Memory System (CMS)* multi-tier yang terbukti menahan degradasi performa pada tugas sebelumnya.

---

## 📐 Desain Arsitektur (Terinspirasi Konsep HOPE)

Model mengadopsi konsep pembagian pemrosesan menjadi dua lapisan memori:

```
Input Tokens: x_t
      │
      ▼
┌──────────────┐
│  Embedding   │
└──────┬───────┘
       │
       ├─────────────────────────────────────────────┐ (residual)
       ▼                                             │
┌───────────────────────────────────────────────┐    │
│  Fast-Weight Memory Layer (Self-Modifying)    │    │
│  - Online SGD Delta Rule per token            │    │
│  - alpha_t: forget/retention gate             │    │
│  - beta_t: inner learning rate                │    │
│  - Only writes prediction error (surprise)    │    │
└──────────────────────┬────────────────────────┘    │
                       ▼                             │
              [ + (Residual) ] ◄─────────────────────┘
                       │
                       ▼
             [ LayerNorm (Fast) ]
                       │
                       ▼
┌───────────────────────────────────────────────┐
│     Continuum Memory System (CMS Tiers)       │
│                                               │
│  ┌─────────────────────────────────────────┐  │
│  │ Tier 0 (Fast): update setiap 1 step     │  │
│  │ FFN Blocks 1..2                         │  │
│  └─────────────────────────────────────────┘  │
│                       ▼                       │
│  ┌─────────────────────────────────────────┐  │
│  │ Tier 1 (Slow): update setiap 4..16 step │  │
│  │ FFN Blocks 3..4 (Konsolidasi Pola Lama) │  │
│  └─────────────────────────────────────────┘  │
└──────────────────────┬────────────────────────┘
                       ▼
                ┌─────────────┐
                │   LM Head   │
                └──────┬──────┘
                       ▼
                Logits: P(x_{t+1})
```

### 1. Fast Weights: Inner-Loop Delta Rule SGD
Memori dinamis $M_t \in \mathbb{R}^{D \times D}$ melakukan satu langkah *online gradient descent* untuk setiap token $t$ terhadap loss rekonstruksi $L_{inner} = \frac{1}{2} \| k_t M_{t-1} - v_t \|^2$:

$$\hat{v}_t = k_t M_{t-1} \quad \text{(apa yang diingat memori untuk key } k_t\text{)}$$
$$e_t = v_t - \hat{v}_t \quad \text{(prediction error / surprise)}$$
$$M_t = \alpha_t M_{t-1} + \beta_t \, k_t^T e_t$$
$$o_t = q_t M_{t-1} \quad \text{(read-out sebelum menulis)}$$

- $\alpha_t = \sigma(W_\alpha x_t + b_\alpha + 4.0) \in (0, 1)$: Gerbang retensi (diinisialisasi mendekati 1.0).
- $\beta_t = \sigma(W_\beta x_t + b_\beta - 2.0) \in (0, 1)$: *Inner learning rate* (langkah adaptasi dinamis).
- $k_t = \frac{k_{raw}}{\|k_{raw}\|_2}$: Kunci dinormalisasi $L_2$ agar pembaruan stabil dan terbatas.
- **Sifat Inti:** Jika memori sudah memprediksi asosiasi dengan benar ($k_t M_{t-1} \approx v_t$), maka error $e_t = 0$, sehingga informasi yang sudah dipahami **tidak akan ditimpa**!

### 2. Slow Weights: Continuum Memory System (CMS) & Deep Optimizer
Lapisan Feed-Forward Network (FFN) dipartisi ke dalam tier frekuensi yang diperbarui oleh *Deep Optimizer* pada kecepatan berbeda:
- **Tier 0 (Fast, Period = 1):** Diperbarui setiap langkah pelatihan untuk menangkap konteks dan adaptasi cepat.
- **Tier 1 (Medium, Period = 4):** Mengakumulasi dan merata-ratakan gradien selama 4 langkah sebelum melangkah.
- **Tier 2 (Slow, Period = 16):** Mengakumulasi gradien selama 16 langkah untuk konsolidasi struktur jangka panjang.

Mekanisme multi-frekuensi ini meniru osilasi gelombang otak manusia (gelombang gamma cepat untuk perhatian sesaat, gelombang delta/theta lambat untuk konsolidasi memori permanen saat tidur).

---

## 📁 Struktur Direktori

```
nested-learning/
├── nested_learning/               # Core Package (Pure NumPy)
│   ├── __init__.py                # Public API
│   ├── layers.py                  # Linear, LayerNorm, GELU, Embedding, SelfModifyingLayer, CMS
│   ├── model.py                   # HOPE architecture, state-passing generation, NPZ checkpointing
│   ├── optimizers.py              # AdamW, SGD, NestedOptimizer (Deep Optimizer), CosineAnnealingLR
│   ├── loss.py                    # CrossEntropyLoss dengan padding masking analitis
│   ├── tokenizer.py               # ByteTokenizer (100% UTF-8) & CharTokenizer (zero dependency)
│   ├── memory_store.py            # SQLite BLOB memory store & Vector DB embedding helper
│   └── trainer.py                 # TextDataset & Trainer dengan multi-tier scheduling
│
├── data/                          # Folder Penyimpanan Data & Database
│   ├── memory.db                  # Database SQLite penyimpan Fast-Weight Memory per user
│   ├── fakta_transportasi.txt     # Kamus fakta transportasi darat, laut, dan udara
│   ├── id_foundation_corpus.txt   # Korpus fondasi bahasa Indonesia (sains, geografi, AI)
│   ├── id_instruction_qa.jsonl    # Dataset instruksi & tanya-jawab format JSONL
│   └── wikipedia_id_articles.txt  # Artikel ensiklopedia resmi Wikipedia Indonesia
│
├── models/                        # Folder Penyimpanan Model Checkpoint (.npz)
│   ├── hope_model.npz             # Model checkpoint hasil pelatihan demo
│   └── hope_id.npz                # Model checkpoint hasil pelatihan bahasa Indonesia
│
├── scripts/                       # Skrip Pendukung & Eksperimen
│   ├── download_datasets.py       # Pengunduh dataset MediaWiki Wikipedia ID (zero dependency)
│   ├── train_demo.py              # Skrip demo latihan cepat (~4 detik di CPU)
│   └── demo_continual_learning.py # Benchmark mitigasi Catastrophic Forgetting
│
├── tests/                         # Behavioral & Mathematical Verification
│   ├── test_equivalence.py        # O(N) state-passing == full sequence forward equivalence
│   ├── test_delta_rule.py         # Konvergensi inner-loop & integritas mask padding
│   ├── test_cms_tiers.py          # Verifikasi jadwal pembaruan multi-frekuensi optimizer
│   ├── test_gradients.py          # Uji gradien numerik (finite differences) seluruh layer
│   └── run_all_tests.py           # Master runner seluruh test suite
│
├── docs/                          # Dokumentasi Lengkap & Riwayat Sesi
├── train.py                       # CLI Utama: Pelatihan fleksibel dataset kustom
├── generate.py                    # CLI Utama: Generasi teks autoregresif O(N)
├── chat.py                        # CLI Utama: Konsol chat interaktif + multi-session SQLite
├── requirements.txt               # numpy>=1.22.0
├── pyproject.toml                 # Package configuration
└── README.md                      # Dokumentasi teknis
```

---

## 🚀 Panduan Penggunaan (Quickstart)

### 1. Instalasi
Pastikan Python 3.9+ dan NumPy terpasang:
```bash
pip install -r requirements.txt
```

### 2. Menjalankan Rangkaian Pengujian (Master Test Suite)
Verifikasi ekuivalensi matematika, gradien analitis, dan mekanisme tier:
```bash
python tests/run_all_tests.py
```
Hasil:
```text
======================================================================
SUMMARY: 6/6 Test Suites PASSED in 0.13 seconds
✓ ALL BEHAVIORAL & MATHEMATICAL TESTS PASSED SUCCESSFULLY!
======================================================================
```

### 3. Eksperimen Continual Learning (Mitigasi Lupa Katastropik)
Uji perbandingan antara model baseline monolitik vs model HOPE bertingkat:
```bash
python scripts/demo_continual_learning.py
```
Output benchmark:
```text
========================================================================
RESULTS SUMMARY:
========================================================================
Model Architecture       | Task A Init | Task B Final | Task A Retained | Catastrophic Forgetting (+Loss)
------------------------------------------------------------------------
Monolithic Baseline      | 2.3765      | 2.5066       | 3.1683          | +0.7918
Nested Learning (HOPE)   | 2.2929      | 2.4525       | 2.9473          | +0.6544
========================================================================
Retention Improvement with Nested Learning: 17.4% reduction in catastrophic forgetting!
✓ The multi-frequency continuum memory system successfully consolidates past knowledge.
```

### 4. Melatih Model Demo
Latih model HOPE di CPU dalam hitungan detik:
```bash
python scripts/train_demo.py
```
Model dan metadata arsitektur akan disimpan ke `hope_model.npz`.

### 5. Generasi Teks Non-Interaktif
Hasilkan teks secara langsung dari terminal:
```bash
python generate.py --prompt "Nested Learning is" --max-tokens 50 --temperature 0.7
```
Output:
```text
Generated 50 tokens in 0.045s (1110.2 tokens/sec on CPU)
```

### 6. Chat Interaktif & Inspeksi Matriks Memori
Jalankan sesi konsol interaktif:
```bash
python chat.py
```
Perintah khusus dalam chat:
- Masukkan teks biasa untuk bercakap dengan model.
- `/memory` : Memeriksa statistik matriks memori dinamis $M$ (Frobenius norm, rata-rata, nilai maks, sparsity, dan preview sub-matriks).
- `/reset` : Mengosongkan memori dinamis ke nol.
- `/temp <val>` : Mengubah temperatur sampling secara dinamis (misal `/temp 0.5`).
- `/tokens <n>` : Mengubah panjang token balasan (misal `/tokens 60`).
- `quit` atau `exit` : Keluar dari sesi.

### 7. Melatih Dataset Kustom Anda Sendiri
Gunakan skrip serbaguna `train.py` untuk melatih model pada dataset Anda:
```bash
# Melatih dari file teks (.txt / .md)
python train.py --data data/contoh_teks.txt --epochs 5 --batch-size 4 --save-path model_saya.npz

# Melatih dari dataset Tanya-Jawab (Q&A JSONL)
python train.py --data data/contoh_qa.jsonl --data-type qa --epochs 8 --save-path model_qa.npz
```
Panduan mendalam cara membuat format data, pembagian train/val split, dan fine-tuning lanjutan tersedia di **[docs/DATASET_GUIDE.md](./docs/DATASET_GUIDE.md)**.
Dokumentasi lengkap penjelasan arsitektur, pemrosesan data, dan bedah seluruh kode package `nested_learning` tersedia di **[docs/BEDAH_KODE_DAN_ARSITEKTUR.md](./docs/BEDAH_KODE_DAN_ARSITEKTUR.md)**.
---

## 💻 Contoh Penggunaan API Python

```python
import numpy as np
import nested_learning as nl

# 1. Inisialisasi Tokenizer (ByteTokenizer melingkupi 100% UTF-8 tanpa token <unk>)
tok = nl.ByteTokenizer()

# 2. Definisikan Model HOPE
# cms_tiers: 2 layer cepat (period 1), 2 layer lambat (period 4)
model = nl.HOPE(
    vocab_size=tok.vocab_size,
    d_model=64,
    n_layers=4,
    cms_tiers=[[2, 1], [2, 4]],
)

# 3. Setup Nested Deep Optimizer & Loss
loss_fn = nl.CrossEntropyLoss(ignore_index=tok.pad_token_id)
optimizer = nl.NestedOptimizer(
    model.tier_param_groups(),
    lr=5e-3,
    optimizer_cls=nl.AdamW,
)

# 4. Pelatihan 1 Step
x_tokens = np.array([[10, 20, 30, 40]], dtype=np.int64)
y_targets = np.array([[20, 30, 40, 50]], dtype=np.int64)

optimizer.zero_grad()
logits, state = model.forward(x_tokens)
loss = loss_fn.forward(logits, y_targets)
dlogits = loss_fn.backward()
model.backward(dlogits)
tier_updates = optimizer.step()

print(f"Loss: {loss:.4f} | Tier updates: {tier_updates}")

# 5. Generasi Autoregresif O(N)
prompt = tok.encode("Nested Learning")
generated = model.generate(prompt, max_new_tokens=30, temperature=0.7)
print("Output:", tok.decode(generated))

# 6. Simpan & Muat Checkpoint
model.save_checkpoint("model_saya.npz", extra_meta={"version": 1})
loaded_model, meta = nl.HOPE.load_checkpoint("model_saya.npz")
```

---

## 📊 Ringkasan Fitur Teknis

| Fitur | PyTorch Standar | Nested Learning Lite |
|---|---|---|
| **Dependensi Eksternal** | Torch, CUDA, CuDNN (~3GB+) | **0 dependensi** (hanya Python + NumPy) |
| **Ukuran Memori Runtime** | Ratusan MB - beberapa GB | **< 30 MB** |
| **Mekanisme Delta Rule** | Autograd graf dinamis | **Turunan analitis eksak (BPTT tertutup)** |
| **Pembaruan Multi-Frekuensi** | Butuh hook optimizer rumit | **Bawaan terintegrasi (`NestedOptimizer`)** |
| **Kecepatan Inferensi CPU** | ~100-300 tok/detik | **>800-1300 tok/detik (state-passing)** |
| **Portabilitas** | Tergantung binary wheel | **Dapat berjalan di OS/arsitektur mana pun** |

---

## 📜 Sitasi & Referensi

Konsep orisinal dikembangkan oleh tim Google Research:

```bibtex
@inproceedings{behrouz2025nested,
  title={Nested Learning: The Illusion of Deep Learning Architectures},
  author={Behrouz, Ali and Razaviyayn, Meisam and Zhong, Peilin and Mirrokni, Vahab},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```
Repository referensi PyTorch: [obekt/HOPE-nested-learning](https://github.com/obekt/HOPE-nested-learning).

---

## 📄 Lisensi

Proyek ini dirilis di bawah lisensi [MIT License](LICENSE).
