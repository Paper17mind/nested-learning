# Dokumentasi Nested Learning Lite

Selamat datang di dokumentasi resmi **Nested Learning Lite**.

Repositori ini adalah implementasi murni **Python + NumPy (Zero PyTorch Dependency)** dari paradigma **Nested Learning** dan arsitektur **HOPE** (*High-order Optimization & Perception Engine*), yang dipublikasikan oleh peneliti Google Research pada NeurIPS 2025.

---

## 📚 Daftar Isi Dokumentasi

1. **[Riwayat Percakapan & Sesi Pembuatan (CHAT_HISTORY.md)](./CHAT_HISTORY.md)**
   - Catatan lengkap seluruh proses sesi kerja saat ini: latar belakang, penurunan rumus matematika, log pengujian terminal lengkap, dan panduan lengkap yang sebelumnya terpotong di terminal.
2. **[Arsitektur & Penurunan Rumus Matematika (ARCHITECTURE.md)](./ARCHITECTURE.md)**
   - Penjelasan mendalam mengenai *Fast Weights* vs *Slow Weights*, aturan *Gated Delta Rule*, *Continuum Memory System (CMS)*, *Deep Optimizer*, serta turunan eksak analitis *Backpropagation Through Time (BPTT)*.
3. **[Referensi API & Modul (API_REFERENCE.md)](./API_REFERENCE.md)**
   - Dokumentasi lengkap untuk setiap kelas dan fungsi di `nested_learning` (`layers`, `model`, `optimizers`, `loss`, `tokenizer`, `trainer`).
4. **[Panduan Menambahkan Dataset (DATASET_GUIDE.md)](./DATASET_GUIDE.md)**
   - Tutorial lengkap cara memasukkan data kustom (.txt, .jsonl, Q&A), melatih via CLI `train.py`, dan teknik fine-tuning lanjutan / continual learning.
5. **[Penyimpanan Memori di SQLite & Vector DB (MEMORY_STORAGE_GUIDE.md)](./MEMORY_STORAGE_GUIDE.md)**
   - Penjelasan teknis dan panduan integrasi matriks Fast-Weight Memory M ke database relasional (SQLite BLOB) maupun Vector DB (Chroma, Qdrant) untuk multi-user chatbot & semantic memory recall.
6. **[Perbedaan Mendasar HOPE vs LLM Biasa (HOPE_VS_LLM.md)](./HOPE_VS_LLM.md)**
   - Analisis mendalam mengapa HOPE berbeda 180 derajat dari LLM konvensional (Transformer), mengapa penambahan dataset tetap membutuhkan pelatihan, dan bagaimana Continuum Memory System mencegah lupa katastropik.
7. **[Memahami Perkembangan Model & Kualitas Jawaban (MODEL_PROGRESS_AND_TRAINING.md)](./MODEL_PROGRESS_AND_TRAINING.md)**
   - Analisis objektif mengapa model demo menghasilkan teks acak pada bahasa Indonesia, bagaimana cara mengukur perkembangan model, serta panduan dataset fondasi & instruksi yang diperlukan.
8. **[Katalog Sumber & Unduhan Dataset (DATASET_DOWNLOAD_SOURCES.md)](./DATASET_DOWNLOAD_SOURCES.md)**
   - Daftar dataset lokal bahasa Indonesia yang telah dibuat di folder `data/`, skrip pengunduh otomatis MediaWiki `download_datasets.py`, dan tautan resmi dataset publik skala besar (Wikipedia Dumps, Indo4B, Alpaca ID, IndoNLU).
9. **[Parameter Decoding & Temperature (SAMPLING_AND_TEMPERATURE.md)](./SAMPLING_AND_TEMPERATURE.md)**
   - Penjelasan teknis parameter decoding: Temperature, Top-K, Top-P, dan Repetition Penalty, serta pengaruhnya pada kreativitas vs keakuratan jawaban.
10. **[Nested Learning & Teori "Dreaming" (NESTED_LEARNING_AND_DREAMING.md)](./NESTED_LEARNING_AND_DREAMING.md)**
   - Analisis ilmiah hubungan antara Nested Learning, ritme gelombang otak biologis (gamma hingga delta), teori Complementary Learning Systems (CLS), dan bedanya dengan metode AI Generative Dreaming / Replay biasa.
11. **[Pelatihan Otomatis Guru LLM (LLM_TEACHER_TUTORING.md)](./LLM_TEACHER_TUTORING.md)**
   - Metode interaktif di mana LLM (Ollama/OpenAI/Kurikulum Offline) bertindak sebagai guru yang mengajari murid (HOPE) turn demi turn, menguji kuis, dan mengukur perkembangan pemahaman secara otomatis.
12. **[Eksperimen Continual Learning (BENCHMARKS.md)](./BENCHMARKS.md)**
   - Analisis hasil eksperimen mitigasi *Catastrophic Forgetting*, perbandingan dengan baseline monolitik, dan hasil uji kecepatan inferensi di CPU.

---

## ⚡ Ringkasan Cepat Menjalankan Perintah

```bash
# 1. Jalankan master test suite (ekuivalensi, gradien, tier, delta rule)
python tests/run_all_tests.py

# 2. Jalankan benchmark mitigasi Catastrophic Forgetting
python scripts/demo_continual_learning.py

# 3. Latih model demo (~4 detik di CPU)
python scripts/train_demo.py

# 4. Hasilkan teks non-interaktif
python generate.py --prompt "Nested Learning is" --max-tokens 50

# 5. Konsol chat interaktif dan inspeksi matriks memori dinamis
python chat.py
```
