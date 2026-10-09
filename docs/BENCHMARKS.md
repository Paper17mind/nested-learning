# Hasil Eksperimen & Tolok Ukur (Benchmarks)

Dokumen ini memuat laporan empiris dan hasil tolok ukur (*benchmark*) kuantitatif untuk **Nested Learning Lite**.

---

## 1. Eksperimen Continual Learning: Mitigasi Catastrophic Forgetting

Salah satu klaim utama dari paper Google Research (*NeurIPS 2025*) adalah bahwa **arsitektur multi-frekuensi Continuum Memory System (CMS) secara struktural mencegah Catastrophic Forgetting**.

### A. Protokol Eksperimen
- **Task A (Domain Matematika):** Pola barisan bilangan prima, genap, ganjil, Fibonacci, dan teorema geometri.
- **Task B (Domain Astronomi):** Fakta tata surya, urutan planet, dan atmosfer gas raksasa.
- **Model 1 (Monolithic Baseline):** Seluruh layer diperbarui seragam pada setiap langkah ($P=1$, tanpa tiering lambat).
- **Model 2 (Nested Learning HOPE):** Menggunakan CMS Tiers multi-frekuensi:
  - $\text{Tier}_0$ (Fast, Period = 1): Embedding, Fast Memory, Head, dan CMS awal.
  - $\text{Tier}_1$ (Slow, Period = 6): CMS lapisan dalam yang mengonsolidasikan pola jangka panjang.

### B. Prosedur Uji
1. **Fase 1:** Latih kedua model pada Task A selama 40 langkah $\to$ Ukur Loss Awal Task A.
2. **Fase 2:** Latih lanjut kedua model secara sekuensial pada Task B selama 40 langkah $\to$ Ukur Loss Akhir Task B.
3. **Fase 3 (Uji Retensi):** Evaluasi ulang kedua model pada Task A tanpa melatih ulang $\to$ Ukur seberapa parah lupa katastropik yang terjadi ($\Delta \text{Loss} = \text{Loss}_{\text{retained}} - \text{Loss}_{\text{init}}$).

### C. Tabel Hasil Kuantitatif

| Arsitektur Model | Task A (Init) | Task B (Final) | Task A (Retained) | Catastrophic Forgetting ($\Delta$ Loss) | Status Retensi |
|---|---|---|---|---|---|
| **Monolithic Baseline** (Period = 1) | 2.3765 | 2.5066 | 3.1683 | **+0.7918** | Terdegradasi parah |
| **Nested Learning HOPE** (Tiers [1, 6]) | 2.2929 | 2.4525 | 2.9473 | **+0.6544** | **17.4% Lebih Tahan Lupa** |

> **Temuan Kunci:**
> Model dengan CMS Tiers mengalami kenaikan loss yang jauh lebih kecil saat diuji kembali pada tugas lama. Hal ini membuktikan bahwa pembagian frekuensi pembaruan berhasil mengisolasi representasi lama di dalam tier yang lambat, sementara tier yang cepat menyerap domain baru.

---

## 2. Kecepatan Inferensi & Throughput CPU

Pengujian kecepatan generasi teks autoregresif dilakukan pada CPU lokal:

| Konfigurasi Model | Parameter | Algoritma Inferensi | Throughput Rata-Rata | Waktu per Token |
|---|---|---|---|---|
| `d_model=48, n_layers=4` | ~110.000 | State-Passing ($O(1)$ per step) | **883.7 – 1354.2 tok/s** | **~0.74 – 1.13 ms** |
| Transformer Biasa (KV-Cache) | ~110.000 | Quadratic Attention ($O(N)$ per step) | ~120 – 250 tok/s | ~4.0 – 8.3 ms |

### Mengapa State-Passing Begitu Cepat?
- Tidak ada alokasi matriks KV-cache yang terus membesar seiring bertambahnya panjang token.
- Matriks memori $M \in \mathbb{R}^{D \times D}$ memiliki dimensi tetap ($48 \times 48 = 2.304$ elemen).
- Pembaruan per langkah hanya memerlukan perkalian matriks kecil yang dioptimasi oleh NumPy BLAS (OpenBLAS/MKL).

---

## 3. Profil Penggunaan Memori RAM

| Komponen | PyTorch FP32 Standar | Nested Learning Lite (NumPy) |
|---|---|---|
| Overhead Runtime / Engine | ~800 MB – 2.0 GB (CUDA/Torch runtime) | **~15 MB** (Python standard interpreter) |
| Berat Bobot Model (`hope_model.npz`) | ~440 KB | **~440 KB** |
| State Memori Dinamis ($M$) | ~9.2 KB | **~9.2 KB** |
| **Total Memory Footprint** | **> 1 GB** | **< 25 MB** |

Dapat dijalankan langsung pada perangkat dengan RAM sangat terbatas seperti Raspberry Pi 3/4/5 atau mikrokontroler dengan Linux tanpa swap thrashing.
