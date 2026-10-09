# Panduan Pelatihan Otomatis: LLM-as-a-Teacher (Guru LLM & Murid HOPE)

Dokumen ini menjelaskan implementasi metode **Student-Teacher Interactive Tutoring**, di mana sebuah model bahasa besar (**LLM sebagai Guru**) mengajari model kecil **HOPE Lite (sebagai Murid)** secara interaktif turn demi turn, lengkap dengan pengujian pemahaman langsung di setiap putaran obrolan.

---

## 1. Konsep & Alur Kerja (Student-Teacher Framework)

```
┌─────────────────────────┐               ┌─────────────────────────┐
│     GURU (LLM / API)    │               │    MURID (HOPE Lite)    │
│ (Llama 3 / Mistral /    │               │ (Model Ringan NumPy)    │
│  Kurikulum Terstruktur) │               │                         │
└────────────┬────────────┘               └────────────▲────────────┘
             │                                         │
             │  1. Memberikan Pelajaran / Fakta        │
             ├────────────────────────────────────────►│
             │                                         │ 2. HOPE Menyerap Fakta:
             │                                         │    - Fast Memory (Delta Rule)
             │                                         │    - Slow Tiers (Gradient Step)
             │                                         │
             │  3. Memberikan Kuis Uji Pemahaman       │
             ├────────────────────────────────────────►│
             │                                         │ 4. Menghasilkan Jawaban
             │  5. Evaluasi Hasil Jawaban              │    Autoregresif O(1)
             │◄────────────────────────────────────────┤
             ▼                                         ▼
┌───────────────────────────────────────────────────────────────────┐
│                     PENGUKURAN METRIK PER TURN                    │
│   • Loss & Perplexity (Kebingungan murid sebelum vs sesudah)      │
│   • Memory Frobenius Norm (||M||)                                 │
│   • Keyword Recall Hit Rate (% kata kunci yang berhasil diingat)  │
└───────────────────────────────────────────────────────────────────┘
```

---

## 2. Tiga Pilihan Mesin Guru (Teacher Providers)

Skrip `scripts/llm_teacher_train.py` mendukung 3 jenis guru yang fleksibel:

### Pilihan A: Guru Kurikulum Terstruktur Bawaan (Offline, Default)
- **Kelebihan:** **100% Bebas Biaya & Tanpa Kebutuhan API Key.**
- Menggunakan kurikulum edukasi terstruktur bahasa Indonesia bawaan.
- Dapat langsung dijalankan di komputer mana pun tanpa instalasi tambahan.
```bash
python scripts/llm_teacher_train.py --turns 5
```

### Pilihan B: Guru LLM Lokal via Ollama (`http://localhost:11434`)
- Jika Anda memiliki **Ollama** di komputer lokal (misal menjalankan `llama3`, `mistral`, atau `qwen`):
```bash
python scripts/llm_teacher_train.py --teacher ollama --model llama3 --turns 5
```

### Pilihan C: Guru API Eksternal (9router / OpenRouter / Groq / OpenAI)

Anda tidak perlu lagi mengetik API key panjang di terminal. Cukup atur konfigurasi sekali di file **`.env`**:

```env
# File: .env
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxx
LLM_MODEL=meta-llama/llama-3-8b-instruct:free
```
*(Jika menggunakan domain custom 9router, ganti `OPENAI_BASE_URL` ke URL endpoint 9router Anda)*.

Lalu jalankan langsung dengan sangat ringkas:
```bash
python scripts/llm_teacher_train.py --turns 5
```
Skrip akan otomatis membaca `.env`, menyambungkan ke 9router/OpenRouter dengan header yang tepat (`HTTP-Referer`, `X-Title`), dan menjalankan sesi tutoring!
---

## 3. Kustomisasi Topik & Materi Pelajaran (Agar Tidak Itu-Itu Saja)

Anda dapat secara leluasa menentukan materi apa yang harus diajarkan oleh Guru LLM:

### A. Satu Topik Spesifik (`--topic`)
Guru LLM akan memfokuskan seluruh sesi pelajaran pada topik pilihan Anda:
```bash
# Contoh mengajar tentang Dinosaurus:
python scripts/llm_teacher_train.py --topic "Dinosaurus" --turns 3

# Contoh mengajar tentang Robotika:
python scripts/llm_teacher_train.py --topic "Robotika dan Otomasi" --turns 3
```

### B. Berganti-Ganti Topik per Turn (`--topics`)
Anda dapat memberikan daftar topik yang dipisahkan tanda koma. Guru akan berganti topik di setiap giliran:
```bash
python scripts/llm_teacher_train.py --topics "Mobil Listrik,Robotika,Planet Mars,Kucing,Kopi Nusantara" --turns 5
```

### C. Memilih Paket Kategori Kurikulum Bawaan (`--category`)
Tersedia bank materi terstruktur untuk mode offline:
- `transportasi` (Mobil, Motor, Bus, Kereta Api, Pesawat, Helikopter, Kapal Laut)
- `teknologi` (Komputer, AI, Robotika, Internet)
- `sains` (Bumi, Mars, Gravitasi, Dinosaurus)
- `biologi` (Kucing, Burung Elang, Hutan Hujan)
- `kuliner` (Kopi Nusantara, Nasi Goreng)
```bash
python scripts/llm_teacher_train.py --category sains --turns 4
python scripts/llm_teacher_train.py --category teknologi --turns 4
```

### D. Menggunakan File Materi Buatan Sendiri (`--materi-file`)
Anda bisa membuat file teks paragraf (`.txt`) atau file `.jsonl` dan menyuruh guru mengajar dari file tersebut:
```bash
python scripts/llm_teacher_train.py --materi-file data/fakta_transportasi.txt --turns 5
```

---

## 3. Contoh Hasil Nyata Pembuktian per Turn Chat

Berikut adalah log aktual saat model HOPE dibimbing turn demi turn:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ 📚 TURN 1/5: Transportasi Darat: Mobil                                  │
├────────────────────────────────────────────────────────────────────────┤
│ 🧑‍🏫 GURU MENGAJAR:
│   "Pelajaran 1: Mobil adalah alat transportasi darat yang memiliki roda 4
│    dan digerakkan oleh mesin. Mobil digunakan untuk mengangkut penumpang..."
│
│ 🤖 RESPON MURID (HOPE LITE):
│   Sebelum Belajar : "Mobil adalah alat digerakkan oleh adalah a" (Recall: 1/6)
│   Sesudah Belajar : "Mobil adalah alat transportasi darat yang"  (Recall: 3/6)
│
│ 📊 METRIK EVALUASI:
│   • Loss Pelajaran : 0.1253 (Perplexity: 1.13)
│   • Memory M Norm  : 0.5549
│   • Skor Pemahaman : 50.0% (3/6 kata kunci cocok)
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ 📚 TURN 3/5: Transportasi Udara: Pesawat Terbang                        │
├────────────────────────────────────────────────────────────────────────┤
│ 🧑‍🏫 GURU MENGAJAR:
│   "Pelajaran 3: Pesawat terbang adalah alat transportasi udara yang..."
│
│ 🤖 RESPON MURID (HOPE LITE):
│   Sebelum Belajar : "Pesawat terbang adalah alat transportasi daratannspo" (Masih menyangka darat!)
│   Sesudah Belajar : "Pesawat terbang adalah alat transportasi udara 4 yan" (Berubah jadi udara!)
│
│ 📊 METRIK EVALUASI:
│   • Loss Pelajaran : 0.5364 (Perplexity: 1.71)
│   • Memory M Norm  : 0.5408
│   • Skor Pemahaman : 50.0%
└────────────────────────────────────────────────────────────────────────┘
```

Perhatikan pada **Turn 3**:
- Sebelum belajar, model salah menyangka pesawat adalah transportasi *darat*.
- Setelah diajari guru, model langsung mengoreksi jawabannya menjadi transportasi **udara**!

---

## 4. Tabel Rekap Perkembangan di Akhir Pelatihan

Di akhir sesi bimbingan, skrip otomatis menampilkan rekap kuantitatif:

```text
==============================================================================
📈 REKAP PERKEMBANGAN PEMAHAMAN MURID (HOPE LITE) PER TURN CHAT:
==============================================================================
Turn   | Topik Pelajaran            | Loss     | Memory ||M||  | Skor Kuis (Sebelum -> Sesudah)
------------------------------------------------------------------------------
1      | Transportasi Darat: Mobil  | 0.1253   | 0.5549        | 17% -> 50%
2      | Transportasi Darat: Sepeda | 0.2675   | 0.5053        | 50% -> 50%
3      | Transportasi Udara: Pesawa | 0.5364   | 0.5408        | 33% -> 50%
4      | Transportasi Laut: Kapal L | 0.7250   | 0.5459        | 33% -> 33%
5      | Transportasi Rel: Kereta A | 0.7662   | 0.7052        | 33% -> 33%
==============================================================================
Rata-rata Skor Pemahaman: 33.3% (Sebelum) ──► 43.3% (Sesudah Belajar)
✓ Terbukti model HOPE berkembang dan mengunci materi secara bertahap di setiap turn!
```

Model hasil bimbingan guru secara otomatis disimpan ke `models/hope_tutored.npz` dan dapat langsung Anda gunakan untuk mengobrol:
```bash
python chat.py --checkpoint models/hope_tutored.npz
```
