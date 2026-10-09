# Panduan Parameter Decoding: Temperature, Top-K, Top-P & Repetition Penalty

Dokumen ini menjelaskan bagaimana mekanisme **Temperature** dan parameter decoding generasi teks lainnya bekerja di dalam **HOPE (Nested Learning Lite)**.

---

## 1. Apakah HOPE Memiliki Pengaturan Temperature?

**YA, ADA dan SUDAH TERPASANG LENGKAP!**

Sama persis seperti model bahasa modern (Llama, GPT, Claude), HOPE Lite mengonversi logit keluaran menjadi probabilitas menggunakan rumus **Softmax dengan Skala Temperature**:

$$P(\text{token}_i) = \frac{\exp\left(\frac{z_i}{T}\right)}{\sum_j \exp\left(\frac{z_j}{T}\right)}$$

Di mana:
- $z_i$ adalah skor logit mentah untuk token ke-$i$ yang dihasilkan oleh lapisan `head`.
- $T$ adalah nilai **Temperature**.

---

## 2. Pengaruh Nilai Temperature terhadap Karakter Jawaban

| Nilai Temperature ($T$) | Karakter Respon | Penggunaan Ideal |
|---|---|---|
| **$T = 0.0$** (Greedy Search) | **100% Deterministik / Pasti**<br>Model selalu memilih token dengan probabilitas mutlak tertinggi ($\text{argmax}$). | Menjawab definisi pasti, fakta eksak, rumus matematika, atau kode program. |
| **$T = 0.2 - 0.5$** | **Fokus & Akurat**<br>Sangat sedikit keacakan, variasi kata minim, fokus pada fakta yang dipelajari. | Tugas Q&A faktual, asisten tugas sekolah/kuliah, ringkasan dokumen. |
| **$T = 0.7 - 0.8$** *(Default)* | **Seimbang & Alami**<br>Kreatif namun tetap koheren dan masuk akal. | Percakapan santai, chat asisten umum. |
| **$T \ge 1.0$** | **Sangat Acak & Liar**<br>Distribusi probabilitas merata, model sering memilih kata-kata tak terduga. | Menulis puisi, cerita fiksi, *brainstorming* ide liar. |

---

## 3. Parameter Decoding Pendukung Lainnya di HOPE Lite

Selain Temperature, HOPE Lite juga dilengkapi 3 parameter kendali generasi mutakhir:

### A. Top-K Sampling (`--top-k`)
- Membatasi pencarian hanya pada **$K$ token teratas** dengan nilai probabilitas tertinggi (misal $K=20$).
- Token di luar $K$ teratas langsung diberi probabilitas nol ($-\infty$).
- **Fungsi:** Mencegah model memilih kata-kata aneh atau *outlier* yang probabilitasnya sangat kecil di ekor distribusi.

### B. Top-P / Nucleus Sampling (`--top-p`)
- Memilih token dari kelompok kecil teratas yang **jumlah kumulatif probabilitasnya mencapai $P$** (misal $P=0.9$ atau $90\%$).
- Jumlah kandidat token dinamis: menyempit saat model yakin, melebar saat model bimbang.
- **Fungsi:** Menghasilkan bahasa yang luwes tanpa keluar dari topik.

### C. Repetition Penalty (`--repetition-penalty`)
- Jika sebuah token sudah pernah dihasilkan di kalimat sebelumnya, logit token tersebut diturunkan (dibagi dengan faktor penalti, misal $1.1$ atau $1.2$).
- **Fungsi:** Mencegah model terjebak dalam *looping* pengulangan kata yang sama (misal *"dan dan dan..."*).

---

## 4. Cara Menggunakan di Berbagai Mode

### Cara 1: Mengatur di Konsol Chat (`chat.py`)

#### Saat Menjalankan Program:
```bash
# Menjalankan chat dengan temperature rendah (faktual/fokus):
python chat.py --temperature 0.2

# Menjalankan chat dengan temperature kreatif:
python chat.py --temperature 0.8
```

#### Mengubah Suhu Secara Instan Saat Chat Sedang Berjalan:
Anda bisa mengubah suhu sewaktu-waktu di tengah percakapan tanpa perlu me-restart chat menggunakan perintah **/temp**:
```text
You > /temp 0.1
[Config]: Temperature set to 0.1

You > /temp 0.7
[Config]: Temperature set to 0.7
```

---

### Cara 2: Mengatur di Skrip Generasi CLI (`generate.py`)
```bash
# Mode Greedy (Faktual Eksak):
python generate.py --prompt "Mobil adalah" --temperature 0.0

# Mode Faktual Terarah:
python generate.py --prompt "Mobil adalah" --temperature 0.3 --top-k 20 --top-p 0.9

# Mode Eksplorasi Kreatif:
python generate.py --prompt "Ceritakan tentang mobil" --temperature 0.8 --repetition-penalty 1.15
```

---

### Cara 3: Mengatur di Kode Python Murni (`model.generate`)

```python
import nested_learning as nl

model, _ = nl.HOPE.load_checkpoint("models/hope_id.npz")
tok = nl.ByteTokenizer()

prompt = tok.encode("Mobil adalah")

generated_tokens = model.generate(
    prompt,
    max_new_tokens=40,
    temperature=0.2,          # Atur temperatur di sini (0.0 untuk greedy)
    top_k=20,                 # Batasi 20 pilihan teratas
    top_p=0.9,                # Nucleus filtering 90%
    repetition_penalty=1.1,   # Cegah pengulangan kata
)

print(tok.decode(generated_tokens))
```
