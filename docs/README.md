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
5. **[Eksperimen Continual Learning (BENCHMARKS.md)](./BENCHMARKS.md)**
   - Analisis hasil eksperimen mitigasi *Catastrophic Forgetting*, perbandingan dengan baseline monolitik, dan hasil uji kecepatan inferensi di CPU.
---

## ⚡ Ringkasan Cepat Menjalankan Perintah

```bash
# 1. Jalankan master test suite (ekuivalensi, gradien, tier, delta rule)
python tests/run_all_tests.py

# 2. Jalankan benchmark mitigasi Catastrophic Forgetting
python demo_continual_learning.py

# 3. Latih model demo (~4 detik di CPU)
python train_demo.py

# 4. Hasilkan teks non-interaktif
python generate.py --prompt "Nested Learning is" --max-tokens 50

# 5. Konsol chat interaktif dan inspeksi matriks memori dinamis
python chat.py
```
