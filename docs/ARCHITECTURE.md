# Arsitektur & Teori Matematika: Nested Learning & HOPE

Dokumen ini menjelaskan fondasi teori dan penurunan matematis dari paradigma **Nested Learning** dan arsitektur **HOPE** (*High-order Optimization & Perception Engine*) sebagaimana dipublikasikan oleh tim Google Research (Behrouz et al., NeurIPS 2025).

---

## 1. Paradigma Nested Learning

Dalam arsitektur pembelajaran mendalam konvensional (misal Transformer standar):
- **Arsitektur jaringan** (lapisan self-attention, feed-forward network, layer norm) dipandang sebagai pemroses representasi statis.
- **Algoritma optimasi** (AdamW, SGD, momentum) dipandang sebagai prosedur eksternal yang hanya memperbarui bobot saat fase *backward pass*.

Paradigma **Nested Learning (NL)** membuktikan bahwa pemisahan tersebut bersifat semu (*illusion*):
> *"Real learning is a set of nested, multi-level optimization problems operating across different update frequencies."*

Setiap komponen pemrosesan dalam otak manusia dan jaringan cerdas dapat dimodelkan sebagai **memori asosiatif** (*associative memory*) yang menyelesaikan masalah optimasi lokal pada skala waktu (*timescale*) tertentu.

---

## 2. Fast Weights vs Slow Weights

Nested Learning menyusun komputasi ke dalam spektrum skala waktu:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SPEKTRUM SKALA WAKTU                            │
├──────────────────────────┬─────────────────────────────────────────────┤
│ Fast Weights             │ Slow Weights                                │
│ (Adaptasi Konteks Cepat) │ (Konsolidasi Jangka Panjang)                │
├──────────────────────────┼─────────────────────────────────────────────┤
│ Self-Modifying Layer     │ Continuum Memory Blocks (CMS Tiers)         │
│ Frekuensi: Setiap token  │ Frekuensi: Setiap 1, 4, 16 langkah optimasi │
│ Inner-Loop Online SGD    │ Outer-Loop Deep Optimizer (AdamW)           │
│ Rekonstruksi (k -> v)    │ Representasi Semantik & Tata Bahasa Global  │
└──────────────────────────┴─────────────────────────────────────────────┘
```

---

## 3. Self-Modifying Layer: Inner-Loop Gated Delta Rule

### A. Formulasi Matematika
Diberikan representasi token $x_t \in \mathbb{R}^D$ pada langkah $t$:

1. **Proyeksi Linier:**
   $$q_t = x_t W_q + b_q \quad \in \mathbb{R}^D$$
   $$k_{raw, t} = x_t W_k + b_k \quad \in \mathbb{R}^D$$
   $$v_t = x_t W_v + b_v \quad \in \mathbb{R}^D$$

2. **Normalisasi Kunci $L_2$:**
   $$k_t = \frac{k_{raw, t}}{\|k_{raw, t}\|_2 + \epsilon}$$
   Normalisasi ini krusial: membuat $\|k_t\| = 1$, sehingga $\beta_t$ berfungsi sebagai *true bounded step size*.

3. **Gerbang Dinamis Per-Token:**
   - **Retention/Forget Gate ($\alpha_t$):**
     $$z_{\alpha, t} = x_t W_\alpha + b_\alpha + 4.0$$
     $$\alpha_t = \sigma(z_{\alpha, t}) \in (0, 1)$$
     Bias $+4.0$ memastikan $\alpha_t \approx 0.98$ pada inisialisasi (memori sangat awet dan lambat melupakan).
   - **Inner Learning Rate ($\beta_t$):**
     $$z_{\beta, t} = x_t W_\beta + b_\beta - 2.0$$
     $$\beta_t = \sigma(z_{\beta, t}) \in (0, 1)$$
     Bias $-2.0$ memastikan $\beta_t \approx 0.11$ pada inisialisasi (penulisan ke memori berlangsung lembut dan stabil).

4. **Pembacaan Memori (Read-Out):**
   $$o_t = q_t M_{t-1} \quad \in \mathbb{R}^D$$
   Pembacaan dilakukan terhadap memori sebelum langkah pembaruan di token $t$.

5. **Pembaruan Memori: Error-Driven Delta Rule:**
   - Rekonstruksi apa yang diingat memori untuk key $k_t$:
     $$\hat{v}_t = k_t M_{t-1}$$
   - *Prediction Error* (Kejutan / Surprise):
     $$e_t = v_t - \hat{v}_t = v_t - k_t M_{t-1}$$
   - Pembaruan matriks memori:
     $$M_t = \alpha_t M_{t-1} + \beta_t \, k_t^T e_t$$

6. **Proyeksi Output:**
   $$y_t = o_t W_{out} + b_{out}$$

---

### B. Mengapa Delta Rule Unggul Dibanding Hebbian Accumulation?

Pada memori asosiatif konvensional (Hopfield / Hebbian sederhana):
$$M_t = M_{t-1} + k_t^T v_t$$
Setiap ada token baru, asosiasi $k_t^T v_t$ langsung dijumlahkan. Akibatnya:
- Nilai matriks memori $M$ terus membesar (*blow up*).
- Jika fakta yang sama muncul berulang kali, asosiasi tersebut mendominasi memori secara berlebihan (*cross-talk interference*).

Sebaliknya, pada **Delta Rule**:
Pembaruan didorong oleh gradien dari loss kuadrat lokal:
$$L_{inner}(M) = \frac{1}{2} \| k_t M - v_t \|^2$$
Turunannya terhadap $M$:
$$\nabla_M L_{inner} = - k_t^T (v_t - k_t M) = - k_t^T e_t$$
Langkah *gradient descent* dengan step size $\beta_t$:
$$M \leftarrow M - \beta_t \nabla_M L_{inner} = M + \beta_t k_t^T e_t$$

**Konsekuensi Penting:**
Jika memori sudah mempelajari asosiasi $(k_t \to v_t)$, maka $\hat{v}_t = v_t \implies e_t = 0$. Akibatnya:
$$\beta_t k_t^T e_t = 0$$
Artinya, **tidak ada perubahan pada memori**. Memori hanya menyerap informasi yang baru atau mengejutkan (*novelty/surprise*).

---

## 4. Continuum Memory System (CMS) & Deep Optimizer

Dalam arsitektur Transformer biasa, semua bobot FFN diperbarui pada frekuensi yang persis sama. Jika model diberi data baru, parameter FFN akan bergeser drastis, menghapus ingatan masa lalu (*catastrophic forgetting*).

Dalam **Continuum Memory System (CMS)**:
1. Rantai blok FFN dibagi ke dalam *tiers*:
   - $\text{Tier}_0$: diperbarui setiap period $P_0 = 1$ langkah.
   - $\text{Tier}_1$: diperbarui setiap period $P_1 = 4$ langkah.
   - $\text{Tier}_2$: diperbarui setiap period $P_2 = 16$ langkah.
2. Setiap langkah optimasi, gradien diakumulasikan ke buffer khusus tier:
   $$\text{Buffer}_i \leftarrow \text{Buffer}_i + \nabla_\theta L$$
   $$\text{Counter}_i \leftarrow \text{Counter}_i + 1$$
3. Ketika $\text{Counter}_i = P_i$:
   $$\bar{g}_i = \frac{1}{P_i} \text{Buffer}_i$$
   $$\theta_i \leftarrow \text{AdamW}(\theta_i, \bar{g}_i)$$
   $$\text{Buffer}_i \leftarrow 0, \quad \text{Counter}_i \leftarrow 0$$

Hal ini memungkinkan lapisan yang dalam untuk mengonsolidasikan pola-pola abstrak secara stabil tanpa terdistorsi oleh fluktuasi data sesaat.

---

## 5. Penurunan Analitis Backpropagation Through Time (BPTT)

Seluruh gradien diturunkan secara eksak dalam bentuk tertutup:

### A. Backward pada `SelfModifyingLayer`
Diberikan $\partial L / \partial y_t = d y_t$:
1. **Output Projection:**
   $$d o_t = d y_t W_{out}^T$$
   $$d W_{out} = \sum_t o_t^T d y_t, \quad d b_{out} = \sum_t d y_t$$
2. **Rekursi Waktu Mundur ($t = T-1 \dots 0$):**
   - Dari $o_t = q_t M_{t-1}$:
     $$d q_t = d o_t M_{t-1}^T$$
   - Dari pembaruan memori $M_t = \alpha_t M_{t-1} + \beta_t k_t^T e_t$ dengan gradien masuk $d M_t$:
     $$d \alpha_t = \text{tr}(d M_t^T M_{t-1}) = \sum_{i,j} d M_t[i,j] \cdot M_{t-1}[i,j]$$
     $$d z_{\alpha, t} = d \alpha_t \cdot \alpha_t (1 - \alpha_t)$$
     $$d \beta_t = \sum_{i,j} d M_t[i,j] \cdot (k_t^T e_t)[i,j]$$
     $$d z_{\beta, t} = d \beta_t \cdot \beta_t (1 - \beta_t)$$
     $$d e_t = \beta_t (k_t \cdot d M_t)$$
     $$d v_t = d e_t, \quad d \hat{v}_t = -d e_t$$
     $$d k_t = \beta_t (e_t \cdot d M_t^T) + d \hat{v}_t M_{t-1}^T$$
   - Akumulasi balik ke $M_{t-1}$:
     $$d M_{t-1} = \alpha_t d M_t + q_t^T d o_t + k_t^T d \hat{v}_t$$
3. **Normalisasi Kunci Mundur:**
   $$d k_{raw, t} = \frac{1}{\|k_{raw, t}\|_2} \left( d k_t - k_t (k_t \cdot d k_t) \right)$$
4. **Input Projections:**
   $$d x_t = d q_t W_q^T + d k_{raw, t} W_k^T + d v_t W_v^T + d z_{\alpha, t} W_\alpha^T + d z_{\beta, t} W_\beta^T$$

---

## 6. Kompleksitas Komputasi: Inferensi Autoregresif $O(N)$

Pada Transformer standar:
- Menghasilkan token ke-$N$ membutuhkan perhatian (*attention*) ke seluruh $N-1$ token sebelumnya (KV Cache berukuran $O(N \cdot D)$).
- Kompleksitas total untuk menghasilkan $N$ token adalah $O(N^2 \cdot D)$.

Pada HOPE (Nested Learning):
- Seluruh riwayat masa lalu terkompresi secara dinamis di dalam matriks memori $M_t \in \mathbb{R}^{D \times D}$.
- Komputasi per langkah:
  $$M_t = \alpha_t M_{t-1} + \beta_t k_t^T e_t$$
  Kompleksitas per token adalah $O(D^2)$, yang bersifat **konstan $O(1)$** terhadap panjang urutan waktu $t$.
- Akibatnya, kompleksitas total untuk menghasilkan urutan panjang $N$ adalah **$O(N \cdot D^2)$ linier**, bukan kuadratik!
