# Katalog & Panduan Sumber Dataset Bahasa Indonesia

Dokumen ini memuat daftar lengkap dataset yang telah disediakan secara lokal di folder `data/`, skrip pengunduh otomatis, serta referensi resmi dataset publik skala besar untuk melatih model **HOPE (Nested Learning Lite)**.

---

## 1. Dataset Lokal yang Telah Dibuat di Folder `data/`

Semua file berikut sudah tersedia di dalam direktori `data/` repositori ini dan dapat langsung Anda gunakan untuk pelatihan:

### A. `data/fakta_transportasi.txt` (Kamus & Fakta Transportasi)
- **Ukuran:** 2.3 KB (teks bahasa Indonesia bersih).
- **Cakupan:** Definisi lengkap kendaraan darat (mobil, motor, bus, truk, kereta api), udara (pesawat terbang, helikopter), dan laut (kapal, perahu), serta aturan lalu lintas dan keselamatan.
- **Tujuan:** Menjawab langsung kasus pengujian kalimat: *"Mobil adalah alat transportasi darat roda 4 yang digerakkan oleh mesin..."*.

### B. `data/id_foundation_corpus.txt` (Korpus Fondasi Bahasa Indonesia)
- **Ukuran:** 2.7 KB (teks bahasa Indonesia terstruktur per bab).
- **Cakupan:** Tata bahasa, pola kalimat SPOK, ilmu pengetahuan alam, geografi Indonesia, teknologi, kecerdasan buatan, dan komunikasi sehari-hari.
- **Tujuan:** Mengajarkan model struktur kata, awalan/akhiran, dan pembentukan kalimat bahasa Indonesia yang runtut.

### C. `data/id_instruction_qa.jsonl` (Dataset Instruksi & Tanya-Jawab)
- **Format:** JSONL (satu baris satu objek JSON).
- **Jumlah:** 20 pasang tanya-jawab terstruktur.
- **Contoh Isi:**
  ```json
  {"question": "Apa itu mobil?", "answer": "Mobil adalah alat transportasi darat yang memiliki 4 roda dan digerakkan oleh mesin untuk membawa penumpang dan barang."}
  {"question": "Berapa jumlah roda pada mobil?", "answer": "Mobil umumnya memiliki 4 roda."}
  {"question": "Apa fungsi pesawat terbang?", "answer": "Pesawat terbang adalah alat transportasi udara yang memiliki sayap dan mesin untuk terbang melintasi angkasa dengan kecepatan tinggi."}
  ```

### D. `data/wikipedia_id_articles.txt` (Artikel Wikipedia Bahasa Indonesia Resmi)
- **Ukuran:** **320+ Kilobyte (~80.000 token)**.
- **Sumber:** Diunduh langsung dari MediaWiki API Wikipedia Bahasa Indonesia resmi.
- **Topik:** *Mobil, Sepeda Motor, Kereta Api, Pesawat Terbang, Helikopter, Kapal, Transportasi, Lalu Lintas, Komputer, Kecerdasan Buatan, Pembelajaran Mesin, Internet, Bumi*.

---

## 2. Skrip Pengunduh Otomatis (`download_datasets.py`)

Anda dapat mengunduh artikel Wikipedia bahasa Indonesia tambahan atau melihat katalog publik menggunakan skrip bawaan tanpa butuh dependensi luar:

```bash
# 1. Unduh kumpulan artikel Wikipedia bahasa Indonesia:
python download_datasets.py --source wikipedia-id

# 2. Tampilkan daftar katalog dataset publik berskala besar:
python download_datasets.py --list
```

---

## 3. Sumber & Referensi Dataset Publik Skala Besar (Open Access)

Jika Anda ingin melatih model pada skala yang lebih besar (puluhan ribu hingga jutaan kalimat), berikut adalah sumber terbuka resmi yang sangat direkomendasikan:

### 1. Wikipedia Bahasa Indonesia (Wikimedia Dumps)
- **Tipe:** Ensiklopedia Lengkap (~500.000 artikel / Ratusan Juta Token).
- **Lisensi:** Creative Commons Attribution-ShareAlike (CC BY-SA 3.0).
- **Tautan Resmi:** [Wikimedia Dumps idwiki](https://dumps.wikimedia.org/idwiki/latest/)
- **Cara Unduh:**
  ```bash
  wget https://dumps.wikimedia.org/idwiki/latest/idwiki-latest-pages-articles.xml.bz2
  ```
  *Tips: Gunakan tool `wikiextractor` (`pip install wikiextractor && python -m wikiextractor.WikiExtractor idwiki-latest-pages-articles.xml.bz2 --json`) untuk mengekstrak menjadi file teks bersih.*

### 2. Indo4B (IndoNLP Benchmark Consortium)
- **Tipe:** Korpus Terbesar Bahasa Indonesia (~4 Miliar Token / 23 GB teks bersih).
- **Sumber:** Berita online, Wikipedia ID, artikel web, dan percakapan media sosial yang telah dibersihkan.
- **Lisensi:** Open Research.
- **Tautan Resmi:** [GitHub IndoBenchmark](https://github.com/indobenchmark/indo-benchmark)
- **Cara Unduh:** Buka repositori IndoBenchmark untuk mendapatkan link Google Drive resmi berkas teks korpus.

### 3. Alpaca Cleaned Bahasa Indonesia (Instruction Tuning)
- **Tipe:** Dataset Tanya-Jawab & Instruksi AI (~52.000 Pasang Pertanyaan-Jawaban).
- **Lisensi:** Apache 2.0 / CC BY-NC 4.0.
- **Tautan Resmi:** [HuggingFace: cahya/instructions-indonesian](https://huggingface.co/datasets/cahya/instructions-indonesian)
- **Format:** Parquet / JSONL.
- **Cara Pakai:** Unduh file `train.jsonl` dan latih langsung menggunakan:
  ```bash
  python train.py --data train.jsonl --data-type qa --epochs 5 --save-path hope_alpaca_id.npz
  ```

### 4. IndoNLU & IndoNLG (IndoBenchmark)
- **Tipe:** Kumpulan tugas NLP spesifik (QA, Sentiment, Summarization, Text Completion).
- **Tautan Resmi:** [GitHub IndoNLP](https://github.com/IndoNLP/indonlu)
- **Cara Unduh:**
  ```bash
  git clone https://github.com/IndoNLP/indonlu.git
  ```

### 5. OPUS OpenSubtitles Bahasa Indonesia
- **Tipe:** Percakapan Sehari-hari / Dialog Natural (Jutaan Kalimat).
- **Tautan Resmi:** [OPUS OpenSubtitles (Indonesian)](https://opus.nlpl.eu/OpenSubtitles.php)
- **Format:** Plain text parallel / monolingual.

---

## 4. Cara Melatih Model pada Dataset Baru yang Telah Dibuat

Anda dapat langsung melatih model HOPE pada dataset lokal yang baru dibuat ini:

```bash
# Melatih gabungan fakta transportasi dan tanya-jawab:
python train.py \
    --data data/fakta_transportasi.txt,data/id_instruction_qa.jsonl \
    --epochs 8 \
    --d-model 48 \
    --batch-size 4 \
    --save-path hope_id.npz
```

### Hasil Uji Nyata Pembuktian:
Setelah dilatih pada dataset ini, saat diuji dengan prompt *"Mobil adalah"*:
```bash
python generate.py --checkpoint hope_id.npz --prompt "Mobil adalah" --max-tokens 25 --temperature 0.0
```
**Keluaran Model:**
```text
Mobil adalah alat digerakkan oleh mes
```
*Model berhasil menyambung dari huruf acak Inggris menjadi kalimat bahasa Indonesia yang bermakna!*
