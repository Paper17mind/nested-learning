# Panduan Menambahkan & Melatih Dataset Kustom

Panduan lengkap cara menambahkan dataset sendiri ke dalam **Nested Learning Lite**, baik melalui antarmuka skrip CLI (`train.py`) maupun melalui kode Python secara langsung.

---

## 1. Pilihan Format Dataset yang Didukung

Nested Learning Lite mendukung beberapa format dataset:

### Format A: File Teks Biasa (`.txt` / `.md`)
Format paling sederhana untuk pre-training atau membaca buku, artikel, dokumen, atau korpus teks besar.
- **Contoh berkas:** `data/contoh_teks.txt`
```text
Nested Learning adalah paradigma pembelajaran mesin baru...
Dalam paradigma ini, model dipandang sebagai sistem optimasi bertingkat...
```

### Format B: File JSONL Korpus Teks (`.jsonl`)
Format standar umum (seperti OpenWebText, Wikipedia dumps, atau CommonCrawl) di mana setiap baris adalah satu objek JSON:
```json
{"text": "Paragraf pertama artikel atau dokumen..."}
{"text": "Paragraf kedua artikel atau dokumen berikutnya..."}
```

### Format C: File JSONL Tanya-Jawab / Instruksi (`.jsonl`)
Format untuk melatih kemampuan interaksi tanya-jawab (seperti Alpaca atau format Q&A di `obekt/HOPE-nested-learning`).
- **Contoh berkas:** `data/contoh_qa.jsonl`
```json
{"question": "Apa itu Nested Learning?", "answer": "Nested Learning adalah paradigma..."}
{"question": "Bagaimana arsitektur HOPE bekerja?", "answer": "HOPE membagi memori..."}
```
Ketika dimuat, format ini secara otomatis diformat menjadi:
```text
Pertanyaan: Apa itu Nested Learning?
Jawaban: Nested Learning adalah paradigma...
```

---

## 2. Cara Melatih via Terminal (Skrip `train.py`)

Anda dapat melatih model langsung dari terminal tanpa perlu menulis kode Python tambahan:

### A. Melatih dari File Teks Biasa
```bash
python train.py --data data/contoh_teks.txt --epochs 5 --batch-size 4 --save-path model_teks.npz
```

### B. Melatih dari Dataset Tanya-Jawab (Q&A JSONL)
```bash
python train.py --data data/contoh_qa.jsonl --data-type qa --epochs 10 --save-path model_qa.npz
```

### C. Melatih dengan Pengaturan Arsitektur Kustom
```bash
python train.py \
    --data data/contoh_teks.txt \
    --d-model 64 \
    --n-layers 4 \
    --cms-tiers "2,1:2,4" \
    --seq-len 48 \
    --lr 5e-3 \
    --epochs 6 \
    --save-path hope_custom.npz
```

### Penjelasan Parameter CLI:
| Argumen | Default | Deskripsi |
|---|---|---|
| `--data` | *Wajib* | Jalur lokasi file dataset (`.txt` atau `.jsonl`). |
| `--data-type` | `auto` | Format data: `auto`, `text`, `jsonl`, atau `qa`. |
| `--text-key` | `text` | Nama kunci JSON jika menggunakan format JSONL biasa. |
| `--question-key` | `question` | Nama kunci pertanyaan untuk format Q&A. |
| `--answer-key` | `answer` | Nama kunci jawaban untuk format Q&A. |
| `--val-ratio` | `0.15` | Porsi data untuk validasi (misal 0.15 = 15%). |
| `--seq-len` | `32` | Panjang konteks token (*context window*). |
| `--stride` | `8` | Jarak pergeseran *sliding window* saat ekstraksi urutan. |
| `--d-model` | `48` | Dimensi representasi tersembunyi model (*width*). |
| `--n-layers` | `4` | Jumlah lapisan CMS FFN (*depth*). |
| `--cms-tiers` | `"2,1:2,4"` | Konfigurasi tier: `jumlah_layer,periode:jumlah_layer,periode`. |
| `--epochs` | `5` | Jumlah perulangan seluruh dataset. |
| `--batch-size` | `4` | Ukuran batch pelatihan. |
| `--lr` | `6e-3` | Laju pemelajaran awal (*peak learning rate*). |
| `--save-path` | `hope_model.npz` | Lokasi penyimpanan checkpoint bobot. |

---

## 3. Cara Memuat & Melatih via Kode Python

Jika ingin mengintegrasikan dataset ke dalam skrip Python sendiri:

### Contoh 1: Memuat dari File Teks Biasa
```python
import nested_learning as nl

# 1. Inisialisasi Tokenizer (ByteTokenizer melingkupi 100% karakter UTF-8)
tok = nl.ByteTokenizer()

# 2. Muat dataset langsung dari file
dataset = nl.TextDataset.from_file(
    filepath="data/contoh_teks.txt",
    tokenizer=tok,
    seq_len=32,
    stride=8,
)

# 3. Bagi menjadi data latih dan data validasi
train_ds, val_ds = dataset.train_val_split(val_ratio=0.15)
print(f"Train sequences: {len(train_ds)}, Val sequences: {len(val_ds)}")
```

### Contoh 2: Memuat dari File JSONL Q&A
```python
import nested_learning as nl

tok = nl.ByteTokenizer()

dataset = nl.TextDataset.from_jsonl(
    filepath="data/dataset_anda.jsonl",
    tokenizer=tok,
    text_key="text", # atau gunakan from_qa_pairs
    seq_len=48,
)
```

### Contoh 3: Melatih Model Lengkap dengan Dataset Baru
```python
import nested_learning as nl

tok = nl.ByteTokenizer()
dataset = nl.TextDataset.from_file("data/contoh_teks.txt", tokenizer=tok, seq_len=32)
train_ds, val_ds = dataset.train_val_split(val_ratio=0.15)

# Definisikan model HOPE
model = nl.HOPE(
    vocab_size=tok.vocab_size,
    d_model=48,
    n_layers=4,
    cms_tiers=[[2, 1], [2, 4]], # 2 layer cepat, 2 layer lambat
)

loss_fn = nl.CrossEntropyLoss(ignore_index=tok.pad_token_id)
scheduler = nl.CosineAnnealingLR(base_lr=5e-3, max_steps=100, warmup_steps=10)
optimizer = nl.NestedOptimizer(
    model.tier_param_groups(),
    lr=5e-3,
    optimizer_cls=nl.AdamW,
    scheduler=scheduler,
)

trainer = nl.Trainer(model, optimizer, loss_fn, tok)
trainer.train(
    train_dataset=train_ds,
    val_dataset=val_ds,
    epochs=5,
    batch_size=4,
    checkpoint_path="model_baru.npz",
)

# Coba uji generasi
prompt = tok.encode("Nested Learning adalah")
hasil = model.generate(prompt, max_new_tokens=30, temperature=0.7)
print("Hasil Generasi:", tok.decode(hasil))
```

---

## 4. Cara Melakukan Continual Learning / Fine-Tuning Lanjutan

Salah satu keunggulan terbesar Nested Learning adalah kemampuan **melanjutkan pelatihan pada domain baru tanpa merusak pengetahuan lama**:

```python
import nested_learning as nl

# 1. Muat checkpoint yang sudah dilatih sebelumnya
model, meta = nl.HOPE.load_checkpoint("hope_model.npz")
tok = nl.ByteTokenizer()

# 2. Muat dataset baru (misalnya domain kedokteran, sains, atau Q&A baru)
dataset_baru = nl.TextDataset.from_file("data/domain_baru.txt", tokenizer=tok, seq_len=32)

# 3. Lanjutkan pelatihan dengan laju belajar yang lebih halus
optimizer = nl.NestedOptimizer(
    model.tier_param_groups(),
    lr=2e-3, # laju belajar lebih kecil untuk fine-tuning
    optimizer_cls=nl.AdamW,
)
loss_fn = nl.CrossEntropyLoss(ignore_index=tok.pad_token_id)

trainer = nl.Trainer(model, optimizer, loss_fn, tok)
trainer.train(
    train_dataset=dataset_baru,
    epochs=3,
    batch_size=4,
    checkpoint_path="hope_model_finetuned.npz",
)
```
Karena tier yang lebih dalam (Slow Tiers) memiliki periode pembaruan yang lambat (misal setiap 4–16 langkah), pengetahuan lama yang telah dikonsolidasikan tidak akan terhapus seketika saat mempelajari domain baru.

---

## 5. Tips Pemilihan Hyperparameter

1. **`seq_len` (Panjang Konteks):**
   - Untuk data teks umum pendek atau tanya-jawab ringkas: gunakan `32` atau `48`.
   - Untuk teks berparagraf panjang: gunakan `64` atau `128`.
2. **`stride` (Pergeseran Window):**
   - Nilai lebih kecil dari `seq_len` (misal `stride = seq_len // 4`) menghasilkan lebih banyak sampel latihan dengan tumpang tindih (*overlap*), sangat membantu jika korpus teks Anda relatif sedikit.
3. **`cms_tiers`:**
   - Default seimbang: `[[2, 1], [2, 4]]` untuk 4 layer (2 layer cepat, 2 layer lambat).
   - Untuk tugas yang sering berganti domain: gunakan interval lebih lebar, misal `[[2, 1], [2, 8]]` agar lapisan lambat lebih kokoh menahan *catastrophic forgetting*.
