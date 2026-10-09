"""Zero-Dependency Dataset Downloader & Reference Catalog for Nested Learning Lite.

Downloads public Indonesian corpora directly using Python standard library (urllib),
without requiring PyTorch or heavy Hugging Face libraries.

Usage:
    # 1. Download real Indonesian Wikipedia encyclopedic articles:
    python download_datasets.py --source wikipedia-id

    # 2. View all public dataset references & download links (Indo4B, HuggingFace, etc):
    python download_datasets.py --list
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

WIKIPEDIA_TOPICS = [
    # Transportasi & Kendaraan
    "Mobil",
    "Sepeda_motor",
    "Bus_(kendaraan)",
    "Kereta_api",
    "Pesawat_terbang",
    "Helikopter",
    "Kapal",
    "Transportasi",
    "Lalu_lintas",
    # Teknologi & AI
    "Komputer",
    "Kecerdasan_buatan",
    "Pembelajaran_mesin",
    "Internet",
    # Sains & Alam
    "Bumi",
    "Matahari",
    "Tata_Surya",
    "Fisika",
    # Geografi & Bahasa
    "Indonesia",
    "Bahasa_Indonesia",
]

PUBLIC_REFERENCES = [
    {
        "name": "Wikipedia Bahasa Indonesia (Wikimedia Dumps)",
        "type": "Korpus Ensiklopedia Lengkap (Ratusan Juta Token)",
        "license": "CC BY-SA 3.0",
        "url": "https://dumps.wikimedia.org/idwiki/latest/idwiki-latest-pages-articles.xml.bz2",
        "description": "Dump resmi seluruh artikel Wikipedia bahasa Indonesia. Dapat diekstrak menggunakan wikiextractor menjadi file .txt bersih.",
        "how_to_get": "wget https://dumps.wikimedia.org/idwiki/latest/idwiki-latest-pages-articles.xml.bz2",
    },
    {
        "name": "Indo4B (IndoNLP Dataset)",
        "type": "Korpus Fondasi Bahasa Indonesia Terbesar (4 Miliar Token / 23 GB)",
        "license": "Open Research",
        "url": "https://github.com/indobenchmark/indo-benchmark",
        "description": "Koleksi teks bahasa Indonesia bersih terbesar dari berita, Wikipedia, artikel web, dan media sosial, dibuat oleh konsorsium peneliti IndoNLP.",
        "how_to_get": "Kunjungi https://github.com/indobenchmark/indo-benchmark untuk tautan unduhan Google Drive / server resmi.",
    },
    {
        "name": "Alpaca Cleaned Bahasa Indonesia (Instruction Dataset)",
        "type": "Tanya Jawab & Instruksi Chat (~52.000 Pasang)",
        "license": "Apache 2.0 / CC BY-NC 4.0",
        "url": "https://huggingface.co/datasets/cahya/instructions-indonesian",
        "description": "Dataset instruksi bergaya asisten AI yang diterjemahkan dan diselaraskan ke bahasa Indonesia. Sangat cocok untuk fine-tuning Q&A.",
        "how_to_get": "Download raw parquet/jsonl dari HuggingFace: https://huggingface.co/datasets/cahya/instructions-indonesian/tree/main",
    },
    {
        "name": "IndoNLU & IndoNLG Benchmarks",
        "type": "Dataset Klasifikasi, Ringkasan, dan Dialog",
        "license": "MIT / Creative Commons",
        "url": "https://github.com/IndoNLP/indonlu",
        "description": "Kumpulan dataset benchmark standar untuk pemahaman dan pembentukan teks bahasa Indonesia (sentiment, QA, summarization).",
        "how_to_get": "git clone https://github.com/IndoNLP/indonlu.git",
    },
    {
        "name": "Indonesian Open Subtitles / Conversational Corpus",
        "type": "Teks Percakapan Bahasa Sehari-hari (Jutaan Kalimat)",
        "license": "Open Data",
        "url": "https://opus.nlpl.eu/OpenSubtitles.php",
        "description": "Subtitle film berbahasa Indonesia yang memuat gaya bahasa santai dan percakapan natural.",
        "how_to_get": "Pilih bahasa 'id' di situs OPUS: https://opus.nlpl.eu/",
    },
]


def download_wikipedia_id(output_path="data/wikipedia_id_articles.txt"):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print("=" * 70)
    print("MENGUNDUH ARTIKEL WIKIPEDIA BAHASA INDONESIA RESMI")
    print("Menggunakan MediaWiki API (Pure Python - Zero Dependency)")
    print("=" * 70)

    all_articles_text = []
    total_chars = 0

    headers = {"User-Agent": "NestedLearningLiteBot/1.0 (https://github.com/nested-learning; contact@example.com)"}

    for idx, title in enumerate(WIKIPEDIA_TOPICS, 1):
        encoded_title = urllib.parse.quote(title)
        api_url = (
            f"https://id.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext&titles={encoded_title}&format=json"
        )
        req = urllib.request.Request(api_url, headers=headers)

        success = False
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    pages = data.get("query", {}).get("pages", {})
                    for pid, pdata in pages.items():
                        extract = pdata.get("extract", "").strip()
                        display_title = pdata.get("title", title)
                        if extract:
                            header = f"=== {display_title} ===\n"
                            article_content = f"{header}{extract}\n\n"
                            all_articles_text.append(article_content)
                            chars = len(article_content)
                            total_chars += chars
                            print(f"[{idx:2d}/{len(WIKIPEDIA_TOPICS)}] Berhasil: {display_title:25s} ({chars:,} karakter)")
                        else:
                            print(f"[{idx:2d}/{len(WIKIPEDIA_TOPICS)}] Kosong / Dialihkan: {display_title}")
                success = True
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait_sec = (attempt + 1) * 2.0
                    time.sleep(wait_sec)
                else:
                    print(f"[{idx:2d}/{len(WIKIPEDIA_TOPICS)}] Gagal mengunduh {title}: {e}")
                    break
            except Exception as e:
                print(f"[{idx:2d}/{len(WIKIPEDIA_TOPICS)}] Gagal mengunduh {title}: {e}")
                break
        time.sleep(0.8)  # Polite delay between Wikipedia API calls
    full_corpus = "".join(all_articles_text)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_corpus)

    print("=" * 70)
    print(f"Selesai! Berhasil mengunduh {len(all_articles_text)} artikel.")
    print(f"Total karakter: {total_chars:,} (~{total_chars // 4:,} token)")
    print(f"File tersimpan di: {output_path}")
    print("=" * 70)
    return output_path


def print_public_references():
    print("=" * 75)
    print("KATALOG REFERENSI DATASET PUBLIK BAHASA INDONESIA (OPEN ACCESS)")
    print("=" * 75)
    for i, ref in enumerate(PUBLIC_REFERENCES, 1):
        print(f"\n{i}. {ref['name']}")
        print(f"   Tipe        : {ref['type']}")
        print(f"   Lisensi     : {ref['license']}")
        print(f"   URL Sumber  : {ref['url']}")
        print(f"   Keterangan  : {ref['description']}")
        print(f"   Cara Unduh  : {ref['how_to_get']}")
    print("\n" + "=" * 75)
    print("TIPS INTEGRASI KE NESTED LEARNING LITE:")
    print("1. Jika Anda mengunduh file teks besar (.txt):")
    print("   python train.py --data jalur/file_hasil_unduh.txt --epochs 5")
    print("2. Jika Anda mengunduh file JSONL instruksi:")
    print("   python train.py --data jalur/dataset.jsonl --data-type qa --epochs 8")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Downloader & Reference Guide Dataset Bahasa Indonesia")
    parser.add_argument(
        "--source",
        type=str,
        default="wikipedia-id",
        choices=["wikipedia-id", "list"],
        help="Pilihan sumber: 'wikipedia-id' atau 'list'",
    )
    parser.add_argument("--list", action="store_true", help="Tampilkan daftar katalog dataset publik")
    parser.add_argument("--output", type=str, default="data/wikipedia_id_articles.txt", help="Jalur file keluaran")
    args = parser.parse_args()

    if args.list or args.source == "list":
        print_public_references()
    elif args.source == "wikipedia-id":
        download_wikipedia_id(args.output)


if __name__ == "__main__":
    main()
