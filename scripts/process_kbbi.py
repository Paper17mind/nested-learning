"""KBBI PDF Parser and Cleaner for Nested Learning.

Extracts dictionary entries from Kamus Besar Bahasa Indonesia (KBBI) PDF,
expands standard abbreviations (yg -> yang, dng -> dengan, etc.),
removes dictionary artifacts, and converts entries into:
1. Natural narrative sentences (data/kbbi_definisi.txt) for language pre-training.
2. Question-Answer pairs (data/kbbi_qa.jsonl) for task fine-tuning.
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Standard KBBI abbreviations mapping to full words
ABBREVIATIONS: Dict[str, str] = {
    r"\byg\b": "yang",
    r"\bdng\b": "dengan",
    r"\bkpd\b": "kepada",
    r"\bdl\b": "dalam",
    r"\bdlm\b": "dalam",
    r"\bpd\b": "pada",
    r"\bsbg\b": "sebagai",
    r"\bdr\b": "dari",
    r"\butk\b": "untuk",
    r"\bkrn\b": "karena",
    r"\btsb\b": "tersebut",
    r"\bspt\b": "seperti",
    r"\btt\b": "tentang",
    r"\bdp\b": "daripada",
    r"\bbkn\b": "bukan",
    r"\bsdh\b": "sudah",
    r"\btjd\b": "terjadi",
    r"\btdk\b": "tidak",
    r"\borg\b": "orang",
    r"\bbrg\b": "barang",
    r"\bmjd\b": "menjadi",
    r"\bthn\b": "tahun",
    r"\bbln\b": "bulan",
    r"\bhr\b": "hari",
    r"\bdsb\b": "dan sebagainya",
    r"\bdst\b": "dan seterusnya",
    r"\bdll\b": "dan lain-lain",
    r"\bmsl\b": "misalnya",
    r"\bum\b": "umum",
}

# Part of speech labels in KBBI to skip / understand
POS_LABELS = {
    "n", "v", "a", "adv", "p", "num", "pron",
    "kl", "cak", "ki", "ark", "hor", "kas", "kp",
}


def clean_text(text: str) -> str:
    """Normalize whitespace and expand abbreviations."""
    # Replace line breaks inside sentences
    text = re.sub(r"\s+", " ", text).strip()
    for pattern, replacement in ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def parse_kbbi_text(raw_text: str, max_entries: Optional[int] = None) -> List[Dict[str, str]]:
    """Parse raw KBBI text into structured headword and clean definition sentences."""
    lines = raw_text.splitlines()
    entries: List[Dict[str, str]] = []

    # Regex for start of a headword entry, e.g.:
    # "abdi n 1 orang yg menjadi milik..."
    # "mengabdi v menghamba; berbakti..."
    # "1 abjad n sistem aksara..."
    entry_start_pattern = re.compile(
        r"^(?:[0-9]+\s+)?([a-zA-Z\(\)\-]+)\s+(?:(?:\([a-zA-Z\s]+\)\s+)?(?:(?:n|v|a|adv|p|num|pron|kl|cak|ki|ark)\s+)+)(.*)$"
    )

    current_word: Optional[str] = None
    current_pos: str = ""
    current_def_chunks: List[str] = []

    def commit_entry():
        nonlocal current_word, current_def_chunks, current_pos
        if not current_word or not current_def_chunks:
            return

        combined_def = " ".join(current_def_chunks)
        combined_def = clean_text(combined_def)

        # Remove example quotes starting with colon e.g. ": ayahnya -- seorang dokter;"
        # Keep the definition parts before colons or split by semicolons
        def_parts = re.split(r"[;:]", combined_def)
        clean_definitions = []
        for p in def_parts:
            p = p.strip()
            # Remove leading numbers e.g. "1 orang", "2 hamba"
            p = re.sub(r"^[0-9]+\s+", "", p)
            # Remove tilde ~ or double hyphens --
            p = p.replace("~", current_word).replace("--", current_word).replace("-", current_word)
            p = clean_text(p)
            if len(p) >= 10 and not p.startswith("ĺ") and not p.startswith("lihat"):
                clean_definitions.append(p)

        if clean_definitions:
            # Create a rich natural Indonesian definition sentence
            primary_def = clean_definitions[0]
            # Form natural definition
            word_clean = re.sub(r"[\(\)]", "", current_word).strip()
            if not word_clean:
                return

            word_cap = word_clean.capitalize()
            # If verb, "Mengabdi artinya...", if noun/adj, "Abdi adalah..."
            connector = "artinya" if current_pos == "v" else "adalah"
            sentence = f"{word_cap} {connector} {primary_def}."
            if not sentence.endswith("."):
                sentence += "."

            entries.append({
                "word": word_clean,
                "pos": current_pos,
                "definition": primary_def,
                "sentence": sentence,
            })

        current_word = None
        current_pos = ""
        current_def_chunks = []

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue

        # Skip header/footer noise
        if "KAMUS BAHASA INDONESIA" in line_str or line_str.isdigit() or len(line_str) <= 2:
            continue

        match = entry_start_pattern.match(line_str)
        if match:
            commit_entry()
            if max_entries and len(entries) >= max_entries:
                break
            raw_word, rest = match.groups()
            current_word = raw_word.lower()
            # Check part of speech
            parts = line_str.split()
            for p in parts[1:4]:
                if p in POS_LABELS:
                    current_pos = p
                    break
            current_def_chunks.append(rest)
        elif current_word is not None:
            # Continuation line
            current_def_chunks.append(line_str)

    commit_entry()
    return entries


def main():
    parser = argparse.ArgumentParser(description="Process KBBI PDF into clean training datasets")
    parser.add_argument("--pdf", type=str, default="data/KAMUS-BESAR-BAHASA-INDONESIA_Mutatis-Mutandis-Hal-1078.pdf",
                        help="Path to KBBI PDF file")
    parser.add_argument("--out-txt", type=str, default="data/kbbi_definisi.txt",
                        help="Path to save narrative definitions text file")
    parser.add_argument("--out-qa", type=str, default="data/kbbi_qa.jsonl",
                        help="Path to save QA JSONL file")
    parser.add_argument("--max-entries", type=int, default=10000,
                        help="Max entries to extract (default: 10000 for balanced CPU training)")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"[Error]: File PDF '{args.pdf}' tidak ditemukan.")
        sys.exit(1)

    print(f"📖 Membaca dan mengekstrak teks dari '{args.pdf}'...")
    from nested_learning.trainer import extract_pdf_text
    raw_text = extract_pdf_text(args.pdf)
    print(f"✓ Berhasil mengekstrak {len(raw_text):,} karakter teks mentah.")

    print(f"🔍 Memparsing dan membersihkan entri kamus (Target: max {args.max_entries} lema)...")
    entries = parse_kbbi_text(raw_text, max_entries=args.max_entries)
    print(f"✓ Berhasil mengekstrak {len(entries):,} entri definisi bersih!")

    # 1. Simpan ke Teks Naratif (untuk Language Pre-training)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_txt)), exist_ok=True)
    with open(args.out_txt, "w", encoding="utf-8") as f:
        # Tulis per kelompok paragraf (5 kalimat per paragraf untuk aliran konteks yang bagus)
        current_para = []
        for idx, entry in enumerate(entries, 1):
            current_para.append(entry["sentence"])
            if idx % 5 == 0:
                f.write(" ".join(current_para) + "\n\n")
                current_para = []
        if current_para:
            f.write(" ".join(current_para) + "\n\n")

    print(f"💾 File korpus naratif tersimpan di: '{args.out_txt}' ({os.path.getsize(args.out_txt):,} bytes)")

    # 2. Simpan ke Format Q&A JSONL (untuk Task Fine-tuning)
    if args.out_qa:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_qa)), exist_ok=True)
        with open(args.out_qa, "w", encoding="utf-8") as f:
            for entry in entries:
                q = f"Apa arti dari kata {entry['word']}?"
                a = entry["sentence"]
                f.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
        print(f"💾 File tanya-jawab tersimpan di: '{args.out_qa}' ({os.path.getsize(args.out_qa):,} bytes)")

    # Print sample entries
    print("\n" + "=" * 60)
    print(" CONTOH HASIL EKSTRAKSI KORPUS KBBI:")
    print("=" * 60)
    for sample in entries[:5]:
        print(f"• Lema     : {sample['word']}")
        print(f"  Definisi : {sample['sentence']}\n")


if __name__ == "__main__":
    main()
