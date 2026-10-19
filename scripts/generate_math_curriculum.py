"""
Math Curriculum Generator untuk HOPE Lite.

Menghasilkan dataset JSONL pelatihan matematika bertahap berdasarkan usia perkembangan:
  - Usia 1 (bayi)   : Mengenal angka 1-5, urutan, anggota tubuh
  - Usia 2 (batita) : Angka 6-10, urutan lanjut, generatif, perbandingan
  - Usia 3 (balita) : Penjumlahan & pengurangan dasar, soal cerita
  - Usia 4+ (anak)  : Perkalian, pembagian, pecahan, bilangan lebih besar

Usage:
    python scripts/generate_math_curriculum.py --output data/math_curriculum/generated.jsonl
    python scripts/generate_math_curriculum.py --age 3 --output data/math_curriculum/age3_only.jsonl
    python scripts/generate_math_curriculum.py --combine-all
"""

import argparse
import json
import os
import random
from typing import Optional

# Kosakata angka Indonesia
NUM_WORDS = {
    0: "nol", 1: "satu", 2: "dua", 3: "tiga", 4: "empat",
    5: "lima", 6: "enam", 7: "tujuh", 8: "delapan", 9: "sembilan",
    10: "sepuluh", 11: "sebelas", 12: "dua belas", 13: "tiga belas",
    14: "empat belas", 15: "lima belas", 16: "enam belas",
    17: "tujuh belas", 18: "delapan belas", 19: "sembilan belas",
    20: "dua puluh", 25: "dua puluh lima", 30: "tiga puluh",
}

STAGE_MAP = {1: "bayi", 2: "batita", 3: "balita", 4: "anak"}

OBJECTS = {
    "bayi":   ["apel", "bola", "balon", "kucing", "buku"],
    "batita": ["pensil", "kelereng", "cokelat", "ikan", "buah"],
    "balita": ["kelereng", "pensil", "apel", "buku", "buah mangga"],
    "anak":   ["kelereng", "pensil", "buku", "uang", "biskuit"],
}


def w(n: int) -> str:
    """Konversi angka ke kata Indonesia."""
    return NUM_WORDS.get(n, str(n))


def make_item(stage, age, topic, question, prompt, answer, keywords):
    return {
        "stage": stage, "age": age, "topic": topic,
        "question": question, "prompt": prompt,
        "answer": answer, "keywords": keywords,
    }


# ──────────────────────────────────────────────────────────────────────────────
# USIA 1 — BAYI: Mengenal angka 1-5, urutan dasar, anggota tubuh
# ──────────────────────────────────────────────────────────────────────────────
def generate_age1():
    items = []
    stage, age = "bayi", 1

    for n in range(1, 6):
        items.append(make_item(stage, age,
            topic=f"Angka: {w(n).capitalize()}",
            question=f"{w(n).capitalize()} itu angka berapa?",
            prompt=f"{w(n).capitalize()} itu angka",
            answer=f"{w(n).capitalize()} itu angka {n}.",
            keywords=[w(n), str(n), "angka"],
        ))

    for n in range(1, 6):
        nxt = n + 1
        items.append(make_item(stage, age,
            topic=f"Urutan: Setelah {n}",
            question=f"Setelah {n} adalah?",
            prompt=f"Setelah {n} adalah",
            answer=f"Setelah {n} adalah {nxt}.",
            keywords=[str(nxt), "setelah"],
        ))

    for n in range(2, 6):
        prv = n - 1
        items.append(make_item(stage, age,
            topic=f"Urutan: Sebelum {n}",
            question=f"Sebelum {n} adalah?",
            prompt=f"Sebelum {n} adalah",
            answer=f"Sebelum {n} adalah {prv}.",
            keywords=[str(prv), "sebelum"],
        ))

    body = [("jari satu tangan", 5), ("mata kita", 2), ("kaki kita", 2),
            ("telinga kita", 2), ("hidung kita", 1), ("mulut kita", 1)]
    for part, count in body:
        items.append(make_item(stage, age,
            topic=f"Hitung: {part.title()}",
            question=f"Berapa {part}?",
            prompt=f"{part.capitalize()} ada",
            answer=f"{part.capitalize()} ada {count}.",
            keywords=[str(count), part.split()[-1]],
        ))

    items.append(make_item(stage, age,
        topic="Generatif: Angka 1 sampai 5",
        question="Sebutkan angka 1 sampai 5!",
        prompt="Angka 1 sampai 5 adalah",
        answer="Angka 1 sampai 5 adalah 1 2 3 4 5.",
        keywords=["1", "2", "3", "4", "5"],
    ))
    return items


# ──────────────────────────────────────────────────────────────────────────────
# USIA 2 — BATITA: Angka 6-10, generatif, perbandingan
# ──────────────────────────────────────────────────────────────────────────────
def generate_age2():
    items = []
    stage, age = "batita", 2

    for n in range(6, 11):
        items.append(make_item(stage, age,
            topic=f"Angka: {w(n).capitalize()}",
            question=f"{w(n).capitalize()} itu angka berapa?",
            prompt=f"{w(n).capitalize()} itu angka",
            answer=f"{w(n).capitalize()} itu angka {n}.",
            keywords=[w(n), str(n)],
        ))

    for n in range(5, 10):
        nxt = n + 1
        items.append(make_item(stage, age,
            topic=f"Urutan: Setelah {n}",
            question=f"Setelah {n} adalah?",
            prompt=f"Setelah {n} adalah",
            answer=f"Setelah {n} adalah {nxt}.",
            keywords=[str(nxt), "setelah"],
        ))

    sequences = [
        ([1, 2, 3], [4, 5, 6]), ([4, 5, 6], [7, 8, 9]),
        ([5, 6, 7], [8, 9, 10]), ([7, 8, 9], [10]),
    ]
    for start, cont in sequences:
        seq_str = " ".join(str(x) for x in start)
        cont_str = " ".join(str(x) for x in cont)
        items.append(make_item(stage, age,
            topic=f"Generatif: Lanjutkan {seq_str}",
            question=f"{seq_str}, lanjutnya apa?",
            prompt=f"{seq_str} lanjutnya",
            answer=f"{seq_str} lanjutnya {cont_str}.",
            keywords=[str(c) for c in cont],
        ))

    pairs = [(3, 5), (2, 8), (7, 4), (1, 10), (6, 9)]
    for a, b in pairs:
        items.append(make_item(stage, age,
            topic=f"Perbandingan: {a} vs {b}",
            question=f"{a} atau {b}, mana yang lebih besar?",
            prompt=f"{max(a,b)} lebih besar dari",
            answer=f"{max(a,b)} lebih besar dari {min(a,b)}.",
            keywords=[str(max(a, b)), "besar", str(min(a, b))],
        ))

    items.append(make_item(stage, age,
        topic="Hitung: Semua Jari Dua Tangan",
        question="Berapa semua jari kedua tangan?",
        prompt="Semua jari kedua tangan ada",
        answer="Semua jari kedua tangan ada 10.",
        keywords=["10", "jari", "tangan"],
    ))
    return items


# ──────────────────────────────────────────────────────────────────────────────
# USIA 3 — BALITA: Penjumlahan & pengurangan 1-10, soal cerita, loncat
# ──────────────────────────────────────────────────────────────────────────────
def generate_age3():
    items = []
    stage, age = "balita", 3

    for a in range(1, 6):
        for b in range(1, 6):
            if a + b <= 10:
                total = a + b
                items.append(make_item(stage, age,
                    topic=f"Penjumlahan: {a}+{b}",
                    question=f"{a} ditambah {b} sama dengan?",
                    prompt=f"{a} ditambah {b} sama dengan",
                    answer=f"{a} ditambah {b} sama dengan {total}.",
                    keywords=[str(total), "ditambah"],
                ))

    for b in range(2, 11):
        for a in range(1, b):
            if a <= 5:
                result = b - a
                items.append(make_item(stage, age,
                    topic=f"Pengurangan: {b}-{a}",
                    question=f"{b} dikurangi {a} sama dengan?",
                    prompt=f"{b} dikurangi {a} sama dengan",
                    answer=f"{b} dikurangi {a} sama dengan {result}.",
                    keywords=[str(result), "dikurangi"],
                ))

    obj_list = OBJECTS["balita"]
    story_ops = [
        (3, 2, "tambah"), (4, 2, "tambah"), (5, 3, "tambah"),
        (6, 2, "kurang"), (7, 3, "kurang"), (8, 4, "kurang"),
    ]
    for a, b, op in story_ops:
        obj = random.choice(obj_list)
        if op == "tambah":
            result = a + b
            items.append(make_item(stage, age,
                topic=f"Soal Cerita: {a}+{b} {obj}",
                question=f"Ada {a} {obj}, ditambah {b} lagi. Sekarang berapa?",
                prompt=f"{a} {obj} ditambah {b} menjadi",
                answer=f"{a} {obj} ditambah {b} menjadi {result} {obj}.",
                keywords=[str(result), obj],
            ))
        else:
            result = a - b
            items.append(make_item(stage, age,
                topic=f"Soal Cerita: {a}-{b} {obj}",
                question=f"Ada {a} {obj}, diambil {b}. Sisa berapa?",
                prompt=f"{a} {obj} diambil {b} sisa",
                answer=f"{a} {obj} diambil {b} sisa {result} {obj}.",
                keywords=[str(result), obj, "sisa"],
            ))

    for step in [2, 5]:
        seq = list(range(step, step * 6 + 1, step))
        shown, cont = seq[:3], seq[3:5]
        seq_str = " ".join(str(x) for x in shown)
        cont_str = " ".join(str(x) for x in cont)
        items.append(make_item(stage, age,
            topic=f"Generatif: Loncat {step}",
            question=f"{seq_str}, lanjutnya apa?",
            prompt=f"{seq_str} lanjutnya",
            answer=f"{seq_str} lanjutnya {cont_str}.",
            keywords=[str(c) for c in cont],
        ))
    return items


# ──────────────────────────────────────────────────────────────────────────────
# USIA 4 — ANAK: Operasi lebih besar, perkalian, pembagian, pecahan
# ──────────────────────────────────────────────────────────────────────────────
def generate_age4():
    items = []
    stage, age = "anak", 4

    for a in [6, 7, 8, 9, 10, 11, 12]:
        for b in [5, 6, 7, 8, 9]:
            total = a + b
            if 10 <= total <= 25:
                items.append(make_item(stage, age,
                    topic=f"Penjumlahan: {a}+{b}",
                    question=f"{a} ditambah {b} sama dengan?",
                    prompt=f"{a} ditambah {b} sama dengan",
                    answer=f"{a} ditambah {b} sama dengan {total}.",
                    keywords=[str(total), "ditambah"],
                ))

    for b in range(10, 21):
        for a in range(3, 10):
            result = b - a
            if result >= 0:
                items.append(make_item(stage, age,
                    topic=f"Pengurangan: {b}-{a}",
                    question=f"{b} dikurangi {a} sama dengan?",
                    prompt=f"{b} dikurangi {a} sama dengan",
                    answer=f"{b} dikurangi {a} sama dengan {result}.",
                    keywords=[str(result), "dikurangi"],
                ))

    for a in range(2, 6):
        for b in range(1, 11):
            product = a * b
            items.append(make_item(stage, age,
                topic=f"Perkalian: {a}x{b}",
                question=f"{a} dikali {b} sama dengan?",
                prompt=f"{a} dikali {b} sama dengan",
                answer=f"{a} dikali {b} sama dengan {product}.",
                keywords=[str(product), "dikali"],
            ))

    for divisor in range(2, 6):
        for quotient in range(1, 11):
            dividend = divisor * quotient
            items.append(make_item(stage, age,
                topic=f"Pembagian: {dividend}:{divisor}",
                question=f"{dividend} dibagi {divisor} sama dengan?",
                prompt=f"{dividend} dibagi {divisor} sama dengan",
                answer=f"{dividend} dibagi {divisor} sama dengan {quotient}.",
                keywords=[str(quotient), "dibagi"],
            ))

    fractions = [
        ("setengah", 2, [10, 8, 6, 4, 20]),
        ("seperempat", 4, [8, 12, 16, 20]),
        ("sepertiga", 3, [9, 12, 15, 6]),
    ]
    for frac_name, divisor, numbers in fractions:
        for n in numbers:
            result = n // divisor
            items.append(make_item(stage, age,
                topic=f"Pecahan: {frac_name} dari {n}",
                question=f"{frac_name.capitalize()} dari {n} adalah?",
                prompt=f"{frac_name.capitalize()} dari {n} adalah",
                answer=f"{frac_name.capitalize()} dari {n} adalah {result}.",
                keywords=[str(result), frac_name],
            ))

    for n in range(1, 21):
        parity = "ganjil" if n % 2 == 1 else "genap"
        items.append(make_item(stage, age,
            topic=f"Ganjil Genap: {n}",
            question=f"Apakah {n} itu ganjil atau genap?",
            prompt=f"{n} adalah bilangan",
            answer=f"{n} adalah bilangan {parity}.",
            keywords=[parity, str(n)],
        ))

    obj_list = OBJECTS["anak"]
    for _ in range(10):
        a = random.randint(8, 20)
        b = random.randint(3, 9)
        obj = random.choice(obj_list)
        total = a + b
        items.append(make_item(stage, age,
            topic=f"Soal Cerita: {a}+{b} {obj}",
            question=f"Ada {a} {obj}, dapat {b} lagi. Totalnya berapa?",
            prompt=f"{a} {obj} ditambah {b} totalnya",
            answer=f"{a} {obj} ditambah {b} totalnya {total} {obj}.",
            keywords=[str(total), obj],
        ))

    for step in [5, 10]:
        seq = list(range(step, step * 6 + 1, step))
        shown, cont = seq[:3], seq[3:5]
        seq_str = " ".join(str(x) for x in shown)
        cont_str = " ".join(str(x) for x in cont)
        items.append(make_item(stage, age,
            topic=f"Generatif: Loncat {step}",
            question=f"{seq_str}, lanjutnya apa?",
            prompt=f"{seq_str} lanjutnya",
            answer=f"{seq_str} lanjutnya {cont_str}.",
            keywords=[str(c) for c in cont],
        ))
    return items


# ──────────────────────────────────────────────────────────────────────────────
GENERATORS = {1: generate_age1, 2: generate_age2, 3: generate_age3, 4: generate_age4}
OUTPUT_FILES = {
    1: "data/math_curriculum/age_1_numbers.jsonl",
    2: "data/math_curriculum/age_2_numbers.jsonl",
    3: "data/math_curriculum/age_3_addition_subtraction.jsonl",
    4: "data/math_curriculum/age_4_operations.jsonl",
}


def parse_args():
    p = argparse.ArgumentParser(description="Generate math curriculum datasets for HOPE Lite")
    p.add_argument("--age", type=int, default=None, choices=[1, 2, 3, 4])
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--combine-all", action="store_true")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def write_jsonl(items, path, shuffle=False, verbose=False):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if shuffle:
        random.shuffle(items)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    if verbose:
        print(f"  + {len(items):>4} item  ->  {path}")


def main():
    args = parse_args()
    ages = [args.age] if args.age else [1, 2, 3, 4]

    print("=" * 60)
    print("MATH CURRICULUM GENERATOR - HOPE LITE")
    print("=" * 60)

    all_items = []
    for age in ages:
        items = GENERATORS[age]()
        print(f"  Usia {age} ({STAGE_MAP[age]:7s}): {len(items):>4} item")
        all_items.extend(items)

    if args.combine_all:
        out = args.output or "data/math_curriculum/all_ages.jsonl"
        write_jsonl(all_items, out, shuffle=args.shuffle, verbose=True)
        print(f"\n+ Total {len(all_items)} item digabung ke: {out}")
    elif args.output:
        write_jsonl(all_items, args.output, shuffle=args.shuffle, verbose=True)
        print(f"\n+ Total {len(all_items)} item disimpan ke: {args.output}")
    else:
        print("\nMenyimpan per file usia:")
        for age in ages:
            items = GENERATORS[age]()
            write_jsonl(items, OUTPUT_FILES[age], shuffle=args.shuffle, verbose=True)
        print(f"\n+ Selesai. {len(ages)} file dataset tersimpan.")

    print("=" * 60)


if __name__ == "__main__":
    main()
