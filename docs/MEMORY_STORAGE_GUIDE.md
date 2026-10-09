# Panduan Penyimpanan Fast-Weight Memory di SQLite & Vector DB

Dokumen ini menjelaskan bagaimana matriks **Fast-Weight Memory ($M$)** pada arsitektur HOPE dapat disimpan dan dikelola secara efisien menggunakan **SQLite** (database relasional) maupun **Vector Database** (ChromaDB, Qdrant, Milvus, pgvector).

---

## 1. Karakteristik Data Fast-Weight Memory ($M$)

Dalam arsitektur HOPE / Nested Learning, memori kerja dinamis bukanlah teks mentah, melainkan sebuah **matriks numerik berdimensi tetap**:
$$M \in \mathbb{R}^{D \times D}$$

| Dimensi Model ($D$) | Ukuran Elemen Matriks | Ukuran Memori Biner (Float32) | Waktu Baca/Tulis SQLite |
|---|---|---|---|
| **$d = 48$** (Lite/Demo) | $48 \times 48 = 2.304$ elemen | **~9.2 KB** | **< 0.05 ms** |
| **$d = 256$** (Small) | $256 \times 256 = 65.536$ elemen | **~262 KB** | **< 0.20 ms** |
| **$d = 512$** (Base) | $512 \times 512 = 262.144$ elemen | **~1.05 MB** | **< 0.80 ms** |

Karena ukurannya **tetap (fixed size)** dan tidak membengkak seiring bertambahnya token (berbeda dengan KV-Cache Transformer biasa), matriks ini **sangat ideal disimpan ke dalam database**!

---

## 2. Penyimpanan di SQLite (Sangat Direkomendasikan untuk Multi-User)

SQLite adalah pilihan paling ringkas, cepat, dan tanpa dependensi eksternal untuk menyimpan memori per pengguna (*multi-user / multi-session management*).

### A. Mengapa SQLite?
- **Zero Configuration:** Database berupa 1 file `.db` lokal portabel.
- **Kecepatan Sub-Milidetik:** Memuat dan menyimpan $M$ (9 KB - 1 MB) membutuhkan waktu kurang dari 1 milidetik.
- **Isolasi Memori Sempurna:** Setiap pengguna atau sesi chat memiliki barisnya sendiri, sehingga memori percakapan satu pengguna tidak akan bocor ke pengguna lain.

### B. Skema Tabel SQLite
Pustaka `nested_learning` menyediakan kelas bawaan `SQLiteMemoryStore`:
```sql
CREATE TABLE fast_memory (
    session_id TEXT PRIMARY KEY,
    shape TEXT NOT NULL,
    dtype TEXT NOT NULL,
    norm REAL NOT NULL,
    state_blob BLOB NOT NULL,
    meta TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
```

### C. Contoh Penggunaan Python (`SQLiteMemoryStore`)
```python
import numpy as np
import nested_learning as nl

# 1. Inisialisasi Database
store = nl.SQLiteMemoryStore("user_memories.db")

# 2. Asumsikan kita punya matriks memori dari obrolan dengan Budi
M_budi = np.random.randn(1, 48, 48).astype(np.float32)

# Simpan ke SQLite
store.save_memory(
    session_id="user_budi_123",
    state=M_budi,
    meta={"nama": "Budi", "lokasi": "Bandung", "topik_terakhir": "Sains AI"}
)
print("Memori Budi berhasil disimpan ke SQLite!")

# 3. Di kemudian hari, Budi kembali membuka chat:
state_termuat, meta = store.load_memory("user_budi_123")
print(f"Memori Budi dimuat kembali! Frobenius norm: {np.linalg.norm(state_termuat):.4f}")

# 4. Melihat daftar semua sesi aktif:
for sesi in store.list_sessions():
    print(f"- Sesi: {sesi['session_id']} | Norm: {sesi['norm']:.3f} | Metadata: {sesi['meta']}")
```

### D. Penggunaan Langsung di Konsol Chat (`chat.py`)
Dalam `chat.py`, fitur SQLite sudah terpasang secara langsung:
```text
You > /session budi
[SQLite]: Switched to new session 'budi'. Starting fresh.

You > Namaku Budi dan aku tinggal di Bandung.
HOPE > ...

You > /session alice
[SQLite]: Saved active memory to session 'budi'.
[SQLite]: Switched to new session 'alice'. Starting fresh.

You > /sessions
--- ACTIVE SQLITE SESSIONS ---
- budi            | shape: (1, 48, 48) | norm: 0.3821
- alice           | shape: (1, 48, 48) | norm: 0.0000 (ACTIVE)
------------------------------

You > /session budi
[SQLite]: Switched to session 'budi'. Restored memory (norm: 0.382).
```

---

## 3. Penyimpanan di Vector Database (Chroma, Qdrant, Milvus)

Jika Anda ingin menerapkan **Semantic Memory Retrieval / Swapping**, Anda bisa menggunakan Vector Database.

### A. Kapan Menggunakan Vector Database?
Jika agen cerdas Anda memiliki **ratusan riwayat topik percakapan**, dan Anda ingin memanggil kembali memori spesifik berdasarkan *kemiripan makna/topik*, misalnya:
- Pengguna bertanya: *"Bagaimana perkembangan proyek website kita bulan lalu?"*
- Vector DB melakukan *similarity search* terhadap topik tersebut, menemukan matriks memori $M$ yang sesuai, dan menyuntikkannya ke model HOPE!

### B. Konversi Matriks Memori ke Vektor Embedding
Pustaka `nested_learning` menyediakan fungsi helper `memory_to_vector`:
```python
import nested_learning as nl

# Matriks memori HOPE [1, 48, 48]
M = ...

# Konversi menjadi 1D vector embedding [48] yang dinormalisasi L2
vec = nl.memory_to_vector(M, method="mean_pool") # atau method="diag" / "flatten"
```

### C. Contoh Integrasi dengan ChromaDB / Qdrant
```python
import chromadb
import nested_learning as nl

# 1. Inisialisasi Chroma Client
client = chromadb.Client()
collection = client.get_or_create_collection("hope_memories")

# 2. Simpan Memori ke Vector DB
# Embedding dapat berupa vektor topik percakapan atau memory_to_vector
embedding_vektor = nl.memory_to_vector(M, method="mean_pool").tolist()
blob_string = M.tobytes().hex() # simpan biner sebagai hex string dalam metadata

collection.add(
    ids=["topik_liburan_bali"],
    embeddings=[embedding_vektor],
    metadatas=[{"topik": "Liburan ke Bali", "memory_hex": blob_string}],
    documents=["Percakapan mengenai rencana liburan ke Bali tahun lalu."]
)

# 3. Pencarian Semantik Saat Pengguna Bertanya
query_vector = embedding_vektor # atau embedding dari model teks
results = collection.query(query_embeddings=[query_vector], n_results=1)

# 4. Pulihkan Matriks M dari Hasil Pencarian
hex_data = results["metadatas"][0][0]["memory_hex"]
restored_M = np.frombuffer(bytes.fromhex(hex_data), dtype=np.float32).reshape(1, 48, 48)

# Inject restored_M kembali ke HOPE model untuk melanjutkan obrolan!
logits, next_state = model.forward(prompt_tokens, state=restored_M)
```

---

## 4. Arsitektur Rekomendasi (Hybrid SQLite + Vector DB)

Untuk sistem produksi berskala besar:
```
                      ┌───────────────────────┐
                      │    Pertanyaan User    │
                      └──────────┬────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       [ User ID Check ]                 [ Semantic Search ]
                 │                               │
                 ▼                               ▼
         ┌──────────────┐                ┌──────────────┐
         │   SQLite     │                │  Vector DB   │
         │ (Working M   │                │ (Topik Arsip │
         │  per User)   │                │  M Jangka    │
         │  Latency:    │                │  Panjang)    │
         │  < 0.05 ms   │                │              │
         └───────┬──────┘                └───────┬──────┘
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                     ┌───────────────────────┐
                     │    Model HOPE Lite    │
                     │  (State-Passing M_t)  │
                     └───────────────────────┘
```

1. **SQLite:** Menangani memori aktif sesi pengguna yang sedang berlangsung secara real-time (*working memory*).
2. **Vector DB:** Mengarsipkan topik-topik lama, memungkinkan agen melakukan *recall* topik masa lalu secara cerdas.
