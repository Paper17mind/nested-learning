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
- ✅ **Inferensi $O(1)$ per Token ($O(N)$ Total):** Menggunakan *state-passing* autoregressive inference dengan kecepatan **>800–1400 tokens/detik di CPU**.
- ✅ **Mitigasi Catastrophic Forgetting:** Dilengkapi sistem *Continuum Memory System (CMS)* multi-tier yang terbukti menahan degradasi performa pada tugas sebelumnya.
- ✅ **🎓 LLM-as-a-Teacher Continual Tutoring:** Bimbingan interaktif bertahap dengan tahapan kognitif (*bayi, batita, balita, anak, grow*) dan fase tidur konsolidasi memori (*neural replay*).
- ✅ **🔄 Text Data Augmentation (Padanan YOLO untuk NLP):** Augmentasi variasi struktur kalimat, substitusi sinonim, dan *modifier jittering* di berkas terpisah (`nested_learning/augmentation.py`) agar representasi konsep kokoh dan tidak tertukar.
- ✅ **💬 In-Chat Continual Learning (`/learn`):** Mengajarkan fakta baru secara langsung di konsol obrolan dengan *micro-training* adaptif dan otomatis tersimpan permanen ke disk (.npz).

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
- **Tier 0 (Fast, Period = 1):** Diperbarui setiap langkah pelatihan untuk adaptasi konteks cepat.
- **Tier 1 (Medium, Period = 4):** Mengakumulasi dan merata-ratakan gradien selama 4 langkah sebelum melangkah.
- **Tier 2 (Slow, Period = 16):** Mengakumulasi gradien selama 16 langkah untuk konsolidasi struktur jangka panjang.

Mekanisme multi-frekuensi ini meniru osilasi gelombang otak manusia: ritme cepat untuk atensi sesaat, dan ritme lambat untuk konsolidasi memori permanen saat tidur.

---

## 📁 Struktur Direktori

```
nested-learning/
├── nested_learning/               # Core Package (Pure NumPy)
│   ├── __init__.py                # Public API & ekspor modul
│   ├── augmentation.py            # Text Data Augmentation (NLP YOLO Analog: rephrase, sinonim, jitter)
│   ├── layers.py                  # Linear, LayerNorm, GELU, Embedding, SelfModifyingLayer, CMS
│   ├── model.py                   # HOPE architecture, state-passing generation, NPZ checkpointing
│   ├── optimizers.py              # AdamW, SGD, NestedOptimizer (Deep Optimizer), CosineAnnealingLR
│   ├── loss.py                    # CrossEntropyLoss dengan padding masking analitis
│   ├── tokenizer.py               # ByteTokenizer (100% UTF-8) & CharTokenizer (zero dependency)
│   ├── memory_store.py            # SQLite BLOB memory store & Vector DB embedding helper
│   └── trainer.py                 # TextDataset & Trainer dengan multi-tier scheduling
│
├── data/                          # Folder Penyimpanan Data & Database
│   ├── memory.db                  # Database SQLite penyimpan Fast-Weight Memory per user/sesi
│   ├── developmental_qa.jsonl     # Kurikulum tumbuh kembang (bayi, batita, balita, anak)
│   ├── learned_teacher_qa.jsonl   # Bank fakta yang tuntas dikuasai murid saat bimbingan guru
│   ├── user_learned_qa.jsonl      # Bank fakta baru yang diajarkan pengguna lewat konsol chat (/learn)
│   ├── infant_corpus.txt          # Korpus dasar kata & kalimat tahap bayi
│   ├── fakta_transportasi.txt     # Kamus fakta transportasi darat, laut, dan udara
│   ├── id_foundation_corpus.txt   # Korpus fondasi bahasa Indonesia (sains, geografi, AI)
│   └── id_instruction_qa.jsonl    # Dataset instruksi & tanya-jawab format JSONL
│
├── models/                        # Folder Penyimpanan Model Checkpoint (.npz)
│   ├── hope_tutored.npz           # Model hasil bimbingan guru interaktif (siap chat)
│   ├── hope_infant.npz            # Model fondasi tahap bayi (infant)
│   ├── hope_id.npz                # Model fondasi bahasa Indonesia umum
│   └── hope_model.npz             # Model checkpoint demo
│
├── scripts/                       # Skrip Pelatihan & Eksperimen
│   ├── llm_teacher_train.py       # Pelatihan interaktif LLM-as-a-Teacher (offline & online API)
│   ├── train_demo.py              # Skrip demo latihan cepat (~4 detik di CPU)
│   ├── demo_continual_learning.py # Benchmark mitigasi Catastrophic Forgetting
│   └── download_datasets.py       # Pengunduh dataset MediaWiki Wikipedia ID (zero dependency)
│
├── logs/                          # Riwayat Sesi & Evaluasi Belajar
│   ├── tutoring_history.jsonl     # Rekap metrik turn bimbingan guru format JSONL
│   ├── tutoring_history.md        # Laporan visual tabel kemajuan kognitif murid
│   └── in_chat_learning.log       # Log pembelajaran on-the-fly dari konsol chat
│
├── tests/                         # Behavioral & Mathematical Verification
│   ├── test_equivalence.py        # O(N) state-passing == full sequence forward equivalence
│   ├── test_delta_rule.py         # Konvergensi inner-loop & integritas mask padding
│   ├── test_cms_tiers.py          # Verifikasi jadwal pembaruan multi-frekuensi optimizer
│   ├── test_gradients.py          # Uji gradien numerik (finite differences) seluruh layer
│   └── run_all_tests.py           # Master runner seluruh test suite
│
├── docs/                          # Dokumentasi Lengkap & Riwayat Riset
├── train.py                       # CLI Utama: Pelatihan fleksibel dataset kustom
├── generate.py                    # CLI Utama: Generasi teks autoregresif O(N)
├── chat.py                        # CLI Utama: Konsol chat interaktif + in-chat learning
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

### 6. Chat Interaktif & In-Chat Learning (`/learn`)
Jalankan sesi konsol interaktif dengan model hasil bimbingan:
```bash
python chat.py --model models/hope_tutored.npz
```

Fitur & Perintah Khusus Konsol Chat:
- **Tanya Jawab Bebas**: Tanyakan fakta secara langsung kepada model.
- `/learn <kalimat fakta>`: **Mengajarkan fakta baru secara instan di dalam chat!**
  - Model akan membuat 3 variasi kalimat otomatis via augmentasi teks.
  - Melakukan *micro-training* adaptif beberapa langkah hingga konvergen (Loss $\le 0.10$).
  - **Otomatis menyimpan perubahan bobot secara permanen ke file `.npz` di disk**, sehingga pengetahuan baru tetap diingat meski program chat ditutup dan dibuka kembali.
- `/memory` : Memeriksa statistik matriks memori dinamis $M$ (Frobenius norm, rata-rata, nilai maks, sparsity, dan preview sub-matriks).
- `/reset` : Mengosongkan memori dinamis ke nol (sekaligus menghapus riwayat sesi di SQLite).
- `/temp <val>` : Mengubah temperatur sampling secara dinamis (misal `/temp 0.4` untuk jawaban lebih presisi).
- `/tokens <n>` : Mengubah panjang maksimum token balasan (misal `/tokens 30`).
- `quit` atau `exit` : Keluar dari sesi chat.

> [!TIP]
> **Anti-Halusinasi & Stop Condition**: Mesin inferensi `chat.py` telah dilengkapi *early stop* otomatis saat model menghasilkan tanda titik (`.`), tanda seru (`!`), tanda tanya (`?`), atau *newline* (`\n`), sehingga model menjawab dengan padat dan tidak berhalusinasi melantur.

---

### 7. Pelatihan Kognitif LLM-as-a-Teacher (Tahapan Bayi s/d Anak)
Latih model layaknya mendidik seorang anak kecil melalui skrip bimbingan kurikulum bertahap:
```bash
# Melatih tahap bayi (4 giliran bimbingan, kurikulum offline)
python scripts/llm_teacher_train.py --stage bayi --turns 4 --teacher curriculum

# Melanjutkan bimbingan pada checkpoint yang ada (tanpa mereset pengetahuan lama)
python scripts/llm_teacher_train.py --stage batita --turns 4 --teacher curriculum --resume

# Menggunakan LLM online (OpenAI / OpenRouter / Anthropic) sebagai guru jika ada API key
python scripts/llm_teacher_train.py --stage bayi --teacher openai --api-key YOUR_API_KEY
```

Tahapan Kognitif yang Didukung:
- `bayi`: Pengenalan kata tunggal, tubuh, dan fungsi dasar (`mata`, `kaki`, `tidur`).
- `batita`: Kalimat 2-3 kata, lingkungan sekitar, dan objek (`kucing mengeong`, `bola bulat`).
- `balita`: Kalimat tanya sebab-akibat sederhana (`kenapa burung terbang?`).
- `anak`: Penalaran logis dasar, moral, dan fakta dunia.
- `grow`: Melatih seluruh tahapan secara berurutan (*full developmental journey*).

> [!IMPORTANT]
> **Konsolidasi Tidur (*Sleep Consolidation*) & Interleaved Replay:**
> Setelah sesi tanya-jawab dengan guru selesai, model memasuki fase "tidur". Seluruh materi yang baru dipelajari dan bank pengetahuan lama (`data/learned_teacher_qa.jsonl`) di-replay secara silang (*contrastive interleaved replay*) dengan variasi teks otomatis hingga loss $\le 0.08$. Hal ini mencegah kebingungan representasi pada kata penghubung yang sama (misalnya `telinga untuk...` vs `hidung untuk...`).

---

### 8. Augmentasi Teks Otomatis (NLP YOLO Analog)
Modul mandiri [nested_learning/augmentation.py](file:///var/www/projects/nested-learning/nested_learning/augmentation.py) menerapkan konsep augmentasi data gambar (seperti flip, rotasi, dan jitter pada YOLO) ke dalam ranah pemrosesan bahasa alami:
- **Struktur Pembalikan & Template**: `A untuk B` $\rightarrow$ `fungsi A adalah B`, `guna A ialah B`, `A berfungsi sebagai B`.
- **Substitusi Sinonim**: Mengganti kata kunci dengan padanan alami bahasa Indonesia tanpa mengubah esensi makna.
- **Modifier Jittering**: Menambahkan kata penegas lembut secara acak (*pasti*, *sungguh*, *selalu*, dsb.).

Uji coba mandiri augmentasi teks:
```python
import nested_learning as nl

variations = nl.augment_text("kaki untuk berjalan", n_variants=3)
for v in variations:
    print("-", v)
# Output:
# - fungsi dari kaki adalah untuk berjalan
# - guna kaki adalah untuk melangkah
# - tentu saja kaki berguna untuk berjalan
```

---

### 9. Melatih Dataset Kustom Anda Sendiri
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

# 2. Augmentasi Teks Otomatis
variasi = nl.augment_text("mata untuk melihat", n_variants=3)
print("Variasi kalimat:", variasi)

# 3. Definisikan Model HOPE
# cms_tiers: 2 layer cepat (period 1), 2 layer lambat (period 4)
model = nl.HOPE(
    vocab_size=tok.vocab_size,
    d_model=64,
    n_layers=4,
    cms_tiers=[[2, 1], [2, 4]],
)

# 4. Setup Nested Deep Optimizer & Loss
loss_fn = nl.CrossEntropyLoss(ignore_index=tok.pad_token_id)
optimizer = nl.NestedOptimizer(
    model.tier_param_groups(),
    lr=5e-3,
    optimizer_cls=nl.AdamW,
)

# 5. Pelatihan 1 Step
x_tokens = np.array([[10, 20, 30, 40]], dtype=np.int64)
y_targets = np.array([[20, 30, 40, 50]], dtype=np.int64)

optimizer.zero_grad()
logits, state = model.forward(x_tokens)
loss = loss_fn.forward(logits, y_targets)
dlogits = loss_fn.backward()
model.backward(dlogits)
tier_updates = optimizer.step()

print(f"Loss: {loss:.4f} | Tier updates: {tier_updates}")

# 6. Generasi Autoregresif O(N)
prompt = tok.encode("kaki untuk")
generated = model.generate(prompt, max_new_tokens=30, temperature=0.7)
print("Output:", tok.decode(generated))

# 7. Simpan & Muat Checkpoint
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
