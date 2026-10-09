# Referensi API Lengkap (API Reference)

Dokumentasi teknis untuk semua kelas, metode, parameter, dan nilai balik di dalam pustaka `nested_learning`.

---

## 1. Modul `nested_learning.layers`

### `Parameter`
Objek pembungkus array NumPy untuk menyimpan data bobot dan gradien yang dapat dilatih.
```python
p = Parameter(data: np.ndarray, name: str = "")
```
- **Atribut:**
  - `p.data` (`np.ndarray` bertipe `float32`): Nilai bobot parameter.
  - `p.grad` (`np.ndarray` bertipe `float32`): Akumulasi gradien turunan loss terhadap parameter.
  - `p.name` (`str`): Nama parameter identifikasi (misal `"fast_memory.proj_q.weight"`).
- **Metode:**
  - `zero_grad()`: Mengatur seluruh elemen gradien ke 0.0.

---

### `Linear`
Transformasi afina linier: $y = x W + b$.
```python
layer = Linear(in_features: int, out_features: int, bias: bool = True, name: str = "linear")
```
- **Inisialisasi:** Bobot diinisialisasi menggunakan *Xavier Uniform*.
- **Metode:**
  - `forward(x: np.ndarray) -> np.ndarray`: Menerima input berdimensi `[B, in_features]` atau `[B, T, in_features]`.
  - `backward(dout: np.ndarray) -> np.ndarray`: Menghitung $dW, db$, dan mengembalikan $dx$.

---

### `Embedding`
Tabel pencarian embedding token.
```python
emb = Embedding(num_embeddings: int, embedding_dim: int, name: str = "embedding")
```
- **Metode:**
  - `forward(indices: np.ndarray) -> np.ndarray`: Menerima indeks bilangan bulat berdimensi `[B, T]`. Mengembalikan tensor embedding `[B, T, embedding_dim]`.
  - `backward(dout: np.ndarray) -> None`: Mengakumulasikan gradien ke `self.weight.grad` via operasi *scatter-add*.

---

### `LayerNorm`
Normalisasi lapisan sepanjang dimensi fitur terakhir.
```python
norm = LayerNorm(normalized_shape: int, eps: float = 1e-5, name: str = "layernorm")
```
- **Parameter Terlatih:** `gamma` (skala, inisialisasi 1.0) dan `beta` (pergeseran, inisialisasi 0.0).
- **Metode:**
  - `forward(x: np.ndarray) -> np.ndarray`: Melakukan normalisasi berdasar rata-rata dan varians.
  - `backward(dout: np.ndarray) -> np.ndarray`: Menghitung turunan analitis terhadap $x, \gamma, \beta$.

---

### `GELU`
Aktivasi *Gaussian Error Linear Unit* dengan pendekatan analitis tangens hiperbolik.
```python
act = GELU()
```
- **Metode:**
  - `forward(x: np.ndarray) -> np.ndarray`: $0.5 x (1 + \tanh(\sqrt{2/\pi}(x + 0.044715 x^3)))$.
  - `backward(dout: np.ndarray) -> np.ndarray`: Menghitung turunan analitis eksak terhadap $x$.

---

### `SelfModifyingLayer`
Lapisan memori cepat dinamis yang dilatih menggunakan *inner-loop Gated Delta Rule* per token.
```python
layer = SelfModifyingLayer(dim: int, name: str = "fast_memory")
```
- **Komponen Internal:**
  - `proj_q`, `proj_k`, `proj_v`, `proj_out`: Proyeksi linier dimensi `dim -> dim`.
  - `gate_alpha`: Proyeksi gerbang retensi (offset bias $+4.0$).
  - `gate_beta`: Proyeksi gerbang laju belajar lokal (offset bias $-2.0$).
- **Metode:**
  - `forward(x: np.ndarray, state: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]`:
    - Menjalankan komputasi sekuensial pada input urutan $x \in \mathbb{R}^{B \times T \times D}$.
    - Mengembalikan `(output, final_memory_state)`.
  - `step(x_t: np.ndarray, state: np.ndarray, mask_t: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]`:
    - Inferensi satu token dalam kompleksitas waktu $O(1)$.
  - `backward(dy: np.ndarray) -> np.ndarray`:
    - Backpropagation Through Time (BPTT) eksak dari akhir urutan ke awal.

---

### `ContinuumMemoryBlock`
Blok sistem memori kontinum (CMS) yang menggantikan FFN konvensional.
```python
cms = ContinuumMemoryBlock(dim: int, expansion: int = 4, name: str = "cms")
```
- **Struktur:** `norm(x + fc2(gelu(fc1(x))))`.
- **Metode:**
  - `forward(x: np.ndarray) -> np.ndarray`.
  - `backward(dout: np.ndarray) -> np.ndarray`.

---

## 2. Modul `nested_learning.model`

### `HOPE`
Arsitektur lengkap penggabungan Embedding, Self-Modifying Layer, CMS Tiers, dan LM Head.
```python
model = HOPE(
    vocab_size: int,
    d_model: int,
    n_layers: int,
    cms_tiers: Optional[List[List[int]]] = None,
    expansion: int = 4,
)
```
- **Parameter Inisialisasi:**
  - `cms_tiers`: Daftar pasangan `[jumlah_layer, periode]`, contoh `[[2, 1], [2, 4]]`. Jumlah layer harus sama dengan `n_layers`.
- **Metode:**
  - `tier_param_groups() -> List[Tuple[int, List[Parameter]]]`:
    - Mempartisi seluruh bobot model ke dalam kelompok tier frekuensi optimasi.
  - `forward(x: np.ndarray, state=None, mask=None, last_only: bool = False) -> Tuple[np.ndarray, np.ndarray]`:
    - Forward pass sekuensial penuh. Jika `last_only=True`, hanya posisi terakhir yang diproses oleh CMS dan Head (optimasi prefill).
  - `step(x_t: np.ndarray, state: np.ndarray, mask_t=None) -> Tuple[np.ndarray, np.ndarray]`:
    - Satu langkah autoregresif inferensi dalam kompleksitas waktu $O(1)$.
  - `backward(dlogits: np.ndarray) -> None`:
    - Melakukan backpropagation lengkap melalui Head, CMS stack, LayerNorm, Fast Memory, dan Embedding.
  - `generate(prompt_tokens, max_new_tokens=50, temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0, eos_token_id=None) -> List[int]`:
    - Menghasilkan token baru secara autoregresif menggunakan algoritma *state-passing* berkecepatan tinggi.
  - `save_checkpoint(filepath: str, extra_meta: Optional[Dict] = None) -> None`:
    - Menyimpan bobot dan konfigurasi ke dalam arsip terkompresi `.npz`.
  - `HOPE.load_checkpoint(filepath: str) -> Tuple[HOPE, Dict]`:
    - Membaca dan membangun model kembali dari arsip `.npz`.

---

## 3. Modul `nested_learning.optimizers`

### `NestedOptimizer`
Deep Optimizer pengelola pembaruan multi-frekuensi Continuum Memory System.
```python
opt = NestedOptimizer(
    tier_param_groups: List[Tuple[int, List[Parameter]]],
    lr: float = 1e-3,
    optimizer_cls: Callable = AdamW,
    scheduler: Optional[CosineAnnealingLR] = None,
    grad_clip: float = 1.0,
    **opt_kwargs,
)
```
- **Metode:**
  - `zero_grad()`: Mengosongkan gradien per langkah.
  - `step() -> List[bool]`: Mengakumulasi gradien ke buffer tier, dan mengeksekusi pembaruan bobot jika periode tier terpenuhi. Mengembalikan daftar boolean status pembaruan masing-masing tier.
  - `get_tier_stats() -> List[Dict]`: Mengembalikan informasi jumlah parameter dan status pencacah periode tiap tier.

---

### `AdamW`
Pengoptimal momentum orde kedua dengan *decoupled weight decay*.
```python
opt = AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, grad_clip=1.0)
```

---

### `CosineAnnealingLR`
Penjadwal laju pemelajaran dengan kurva kosinus dan fase *warmup* linier.
```python
sched = CosineAnnealingLR(base_lr: float, max_steps: int, warmup_steps: int = 0, min_lr: float = 1e-6)
```

---

## 4. Modul `nested_learning.loss`

### `CrossEntropyLoss`
Loss fungsi klasifikasi token dengan penanganan padding mask.
```python
loss_fn = CrossEntropyLoss(ignore_index: int = -100, label_smoothing: float = 0.0)
```
- **Metode:**
  - `forward(logits: np.ndarray, targets: np.ndarray) -> float`: Menghitung loss skalar rata-rata pada posisi token yang valid.
  - `backward() -> np.ndarray`: Menghitung gradien analitis terhadap tensor `logits`.

---

## 5. Modul `nested_learning.tokenizer`

### `ByteTokenizer`
Tokenizer tingkat bita dengan cakupan 100% UTF-8 tanpa token `<unk>`.
```python
tok = ByteTokenizer()
```
- **Ukuran Kosakata:** 260 token (0–255: literal bytes, 256: `<pad>`, 257: `<eos>`, 258: `<bos>`, 259: `<sep>`).
- **Metode:**
  - `encode(text: str, add_bos=False, add_eos=False) -> List[int]`.
  - `decode(token_ids: List[int], skip_special_tokens=True) -> str`.

---

## 6. Modul `nested_learning.trainer`

### `TextDataset`
Generator batch sekuensial untuk pemodelan bahasa kausal autoregresif.
```python
dataset = TextDataset(token_ids: List[int], seq_len: int, pad_token_id: int = 0, stride: Optional[int] = None)
```
- **Metode:**
  - `get_batches(batch_size: int, shuffle: bool = True) -> Iterator[Tuple[inputs, targets, mask]]`.

### `Trainer`
Orkestrator pelatihan model HOPE dengan nested optimizer.
```python
trainer = Trainer(model: HOPE, optimizer: NestedOptimizer, loss_fn: Optional[CrossEntropyLoss] = None, tokenizer=None)
```
- **Metode:**
  - `train(train_dataset, val_dataset=None, epochs=5, max_steps=None, batch_size=4, accumulate_grad=1, eval_every_steps=50, checkpoint_path=None, verbose=True) -> Dict`.
  - `evaluate(dataset, batch_size=4) -> Dict[str, float]`.
