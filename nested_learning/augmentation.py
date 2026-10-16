"""Text Data Augmentation for Nested Learning & NLP.

Analogous to Image Augmentations in Computer Vision (YOLO flip, rotate, crop, color jitter):
- Template / Structural Rephrasing (Flip / Rotate)
- Synonym Substitution (Color Jitter)
- Context & Modifier Jittering (Random Padding / Crop)
- Dual Active-Passive & Subject-Action transformations

Designed specifically for Indonesian language, developmental learning, and continual memory retention.
"""

import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple


# Kamus sinonim untuk augmentasi berbasis makna (Semantic Jitter)
SYNONYM_MAP: Dict[str, List[str]] = {
    # Panca indra & aksi
    "melihat": ["memandang", "melihat", "menatap"],
    "mendengar": ["menyimak", "mendengarkan", "mendengar"],
    "mencium": ["membaui", "mencium", "menghirup"],
    "berjalan": ["melangkah", "berjalan"],
    "melangkah": ["berjalan", "melangkah"],
    
    # Sifat & objek indra
    "suara": ["bunyi", "suara", "alunan"],
    "bunyi": ["suara", "bunyi"],
    "bau": ["aroma", "wangi", "bau"],
    "aroma": ["wangi", "bau harum", "aroma"],
    "cahaya": ["sinar", "cahaya"],
    "terang": ["cerah", "berkilau", "terang"],
    "cerah": ["terang", "bercahaya", "cerah"],
    "lembut": ["empuk", "halus", "lembut"],
    "empuk": ["nyaman", "lembut", "empuk"],
    "manis": ["enak", "lezat", "manis"],
    
    # Hewan & aksi
    "mengeong": ["bersuara meong", "berbunyi meong", "mengeong"],
    "kwek": ["kwek kwek", "suara bebek kwek", "kwek"],
    "berputar": ["menggelinding", "berputar cepat", "berputar"],
    "menggelinding": ["berputar", "meluncur", "menggelinding"],
    
    # Kata bantu & fungsional
    "sangat": ["sekali", "amat", ""],
    "sekali": ["sangat", "amat", ""],

    # Kuantitas, Angka & Berhitung
    "1": ["satu", "1"],
    "2": ["dua", "2"],
    "3": ["tiga", "3"],
    "4": ["empat", "4"],
    "5": ["lima", "5"],
    "6": ["enam", "6"],
    "7": ["tujuh", "7"],
    "8": ["delapan", "8"],
    "9": ["sembilan", "9"],
    "10": ["sepuluh", "10"],
    "satu": ["1", "satu"],
    "dua": ["2", "dua"],
    "tiga": ["3", "tiga"],
    "empat": ["4", "empat"],
    "lima": ["5", "lima"],
    "enam": ["6", "enam"],
    "tujuh": ["7", "tujuh"],
    "delapan": ["8", "delapan"],
    "sembilan": ["9", "sembilan"],
    "sepuluh": ["10", "sepuluh"],
    "jumlah": ["total", "banyaknya", "jumlah"],
    "total": ["jumlah", "seluruhnya", "total"],
    "banyaknya": ["jumlah", "total", "banyaknya"],
    "semuanya": ["totalnya", "jumlahnya", "semuanya"],
}

# Pola kalimat terstruktur anak / pengetahuan dasar
STRUCTURAL_PATTERNS: List[Tuple[re.Pattern, List[str]]] = [
    # Pola: "[Subjek] untuk [Aksi] [Objek]."
    (
        re.compile(r"^([A-Za-z0-9]+)\s+untuk\s+([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*?)\.?$", re.IGNORECASE),
        [
            "{0} untuk {1}.",
            "{0} adik berfungsi {1}.",
            "{0} dipakai {1}.",
            "Guna {0} yaitu {1}.",
            "{0} membantu {1}.",
        ]
    ),
    # Pola: "[Subjek] membantu [Aksi] [Objek]."
    (
        re.compile(r"^([A-Za-z0-9]+)\s+membantu\s+([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*?)\.?$", re.IGNORECASE),
        [
            "{0} membantu {1}.",
            "{0} untuk {1}.",
            "{0} berguna {1}.",
            "Fungsi {0} adalah {1}.",
        ]
    ),
    # Pola: "[Subjek] melihat/mendengar/mencium [Objek]."
    (
        re.compile(r"^([A-Za-z0-9]+)\s+(melihat|mendengar|mencium|merasakan)\s+([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*?)\.?$", re.IGNORECASE),
        [
            "{0} {1} {2}.",
            "{0} adik {1} {2}.",
            "{0} untuk {1} {2}.",
            "Dengan {0}, adik bisa {1} {2}.",
        ]
    ),
    # Pola: "[Subjek] berbunyi/bersuara/mengeluarkan [Objek]."
    (
        re.compile(r"^([A-Za-z0-9]+)\s+(berbunyi|bersuara|mengeluarkan suara|mengeong)\s+([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*?)\.?$", re.IGNORECASE),
        [
            "{0} {1} {2}.",
            "{0} mengeluarkan bunyi {2}.",
            "{0} berbunyi {2}.",
            "Suara {0} berbunyi {2}.",
        ]
    ),
    # Pola: "[Subjek] rasanya/berwarna/berbentuk [Objek]."
    (
        re.compile(r"^([A-Za-z0-9]+)\s+(rasanya|berwarna|berbentuk)\s+([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*?)\.?$", re.IGNORECASE),
        [
            "{0} {1} {2}.",
            "{0} itu {1} {2}.",
            "{0} memiliki sifat {1} {2}.",
        ]
    ),
    # Pola: "[Objek A] [Jumlah A] dan [Objek B] [Jumlah B] totalnya/jumlahnya [Jumlah Total]."
    (
        re.compile(
            r"^([A-Za-z0-9\s]+?)\s+(\d+|satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh)\s+(?:jari|buah|biji|butir|ekor)?\s*dan\s+([A-Za-z0-9\s]+?)\s+(\d+|satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh)\s+(?:jari|buah|biji|butir|ekor)?\s*(?:totalnya|jumlahnya|semuanya|maka totalnya)\s+(?:ada\s+)?(\d+|satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh)\s*(?:jari|buah|biji|butir|ekor)?\.?$",
            re.IGNORECASE,
        ),
        [
            "Jumlah {0} {1} dan {2} {3} adalah {4}.",
            "{0} ada {1} dan {2} ada {3}, total keseluruhannya adalah {4}.",
            "Jika {0} berjumlah {1} serta {2} berjumlah {3}, semuanya berjumlah {4}.",
            "Gabungan {0} {1} dengan {2} {3} menghasilkan {4}.",
        ]
    ),
]


class TextAugmenter:
    """Automatic Text Data Augmenter for Continual Learning and Neural Replay.
    
    Transforms a single source sentence into diverse syntactic variations
    while preserving exact underlying semantic ground-truth.
    """

    def __init__(self, synonym_prob: float = 0.35, max_variations: int = 4):
        self.synonym_prob = synonym_prob
        self.max_variations = max_variations

    def substitute_synonyms(self, text: str) -> str:
        """Replace eligible words with semantic equivalents (analogous to color jitter)."""
        words = text.split()
        new_words = []
        for w in words:
            clean_w = w.lower().strip(".,!?:;")
            punct = w[len(clean_w):] if len(w) > len(clean_w) else ""
            if clean_w in SYNONYM_MAP and random.random() < self.synonym_prob:
                candidates = SYNONYM_MAP[clean_w]
                chosen = random.choice(candidates)
                if chosen:
                    if w and w[0].isupper():
                        chosen = chosen.capitalize()
                    new_words.append(chosen + punct)
                else:
                    # Dropped word (analogous to dropout/crop)
                    continue
            else:
                new_words.append(w)
        return " ".join(new_words)

    def structural_variations(self, text: str) -> List[str]:
        """Apply structural rephrasing templates (analogous to rotate / flip)."""
        clean_text = text.strip()
        variations = []
        for pattern, templates in STRUCTURAL_PATTERNS:
            m = pattern.match(clean_text)
            if m:
                groups = m.groups()
                for tmpl in templates:
                    try:
                        v = tmpl.format(*groups).strip()
                        if not v.endswith("."):
                            v += "."
                        variations.append(v)
                    except Exception:
                        pass
                break
        return variations

    def augment(self, text: str, n_variations: Optional[int] = None) -> List[str]:
        """Generate diverse augmented sentences from a single input fact.
        
        Guaranteed to include:
        1. The original clean sentence
        2. Structural flips (if pattern matches)
        3. Synonym-jittered variants
        """
        limit = n_variations or self.max_variations
        clean_origin = text.strip()
        if not clean_origin.endswith("."):
            clean_origin += "."

        results: Set[str] = {clean_origin}

        # 1. Structural pattern flips
        struct_variants = self.structural_variations(clean_origin)
        for sv in struct_variants:
            results.add(sv)
            if len(results) >= limit:
                break

        # 2. Synonym substitutions
        base_list = list(results)
        for base in base_list:
            if len(results) >= limit:
                break
            syn_v = self.substitute_synonyms(base)
            if syn_v and syn_v != base:
                if not syn_v.endswith("."):
                    syn_v += "."
                results.add(syn_v)

        # 3. Contextual child-friendly modifier injection
        if len(results) < limit:
            first_word = clean_origin.split()[0] if clean_origin.split() else ""
            if first_word:
                results.add(f"Lihat, {clean_origin[0].lower() + clean_origin[1:]}")
                results.add(f"Wah, {clean_origin[0].lower() + clean_origin[1:]}")

        out = list(results)[:limit]
        return out


# Global singleton instance for easy import and call
_default_augmenter = TextAugmenter()

def augment_text(text: str, max_variations: int = 4) -> List[str]:
    """Convenience helper to augment a text into multiple semantic variations."""
    return _default_augmenter.augment(text, n_variations=max_variations)


NUMBER_WORDS: Dict[int, str] = {
    0: "nol", 1: "satu", 2: "dua", 3: "tiga", 4: "empat",
    5: "lima", 6: "enam", 7: "tujuh", 8: "delapan", 9: "sembilan", 10: "sepuluh",
}


def generate_counting_samples(
    item_type: str = "jari tangan",
    part_a: str = "kanan",
    part_b: str = "kiri",
    num_a: int = 5,
    num_b: int = 5,
    unit: str = "jari",
) -> List[Dict[str, Any]]:
    """Generate structured arithmetic & counting QA pairs and facts with diverse phrasing.

    Example:
        generate_counting_samples("jari tangan", "kanan", "kiri", 5, 5, "jari")
        -> Returns diverse QA pairs like:
           - "Berapa jumlah jari tangan kanan dan kiri jika kanan ada 5 dan kiri ada 5?"
             "Jumlah jari tangan kanan 5 dan kiri 5 sehingga totalnya adalah 10 jari."
    """
    total = num_a + num_b
    word_a = NUMBER_WORDS.get(num_a, str(num_a))
    word_b = NUMBER_WORDS.get(num_b, str(num_b))
    word_total = NUMBER_WORDS.get(total, str(total))

    samples: List[Dict[str, Any]] = [
        {
            "question": f"Berapa jumlah {item_type} {part_a} dan {part_b} jika {part_a} ada {num_a} dan {part_b} ada {num_b}?",
            "answer": f"Jumlah {item_type} {part_a} {num_a} dan {part_b} {num_b} adalah {total} {unit}.",
            "prompt": f"Jumlah {item_type} {part_a} {num_a} dan {part_b} {num_b} adalah",
            "keywords": [str(total), word_total, unit],
        },
        {
            "question": f"Jika {item_type} {part_a} berjumlah {word_a} dan {item_type} {part_b} berjumlah {word_b}, berapa totalnya?",
            "answer": f"Total {item_type} {part_a} dan {part_b} adalah {word_total} {unit}.",
            "prompt": f"Total {item_type} {part_a} dan {part_b} adalah",
            "keywords": [str(total), word_total, unit],
        },
        {
            "question": f"Berapa total {item_type} {part_a} {num_a} ditambah {item_type} {part_b} {num_b}?",
            "answer": f"{num_a} ditambah {num_b} hasilnya adalah {total} {unit}.",
            "prompt": f"{num_a} ditambah {num_b} hasilnya adalah",
            "keywords": [str(total), word_total],
        },
        {
            "question": f"Ada {num_a} {unit} di {item_type} {part_a} dan {num_b} {unit} di {item_type} {part_b}. Berapa semuanya?",
            "answer": f"Semuanya berjumlah {total} {unit}.",
            "prompt": f"Semuanya berjumlah",
            "keywords": [str(total), word_total, unit],
        },
        (
            {
                "question": "Berapa jumlah jari tangan kiri dan kanan?",
                "answer": "Jari tangan kanan 5 dan jari tangan kiri 5, totalnya 10 jari.",
                "prompt": "Jari tangan kanan 5 dan jari tangan kiri 5, totalnya",
                "keywords": ["10", "sepuluh", "jari"],
            }
            if item_type == "jari tangan" and num_a == 5 and num_b == 5
            else {
                "question": f"Berapa hasil {num_a} ditambah {num_b}?",
                "answer": f"Hasil {num_a} ditambah {num_b} adalah {total}.",
                "prompt": f"Hasil {num_a} ditambah {num_b} adalah",
                "keywords": [str(total), word_total],
            }
        ),
    ]
    return samples


def generate_counting_dataset(max_sum: int = 10) -> List[Dict[str, Any]]:
    """Generate developmental arithmetic and counting QA dataset for cognitive training."""
    items: List[Dict[str, Any]] = []
    scenarios = [
        ("jari tangan", "kanan", "kiri", "jari"),
        ("buah apel", "merah", "hijau", "apel"),
        ("balok susun", "merah", "kuning", "balok"),
        ("bola mainan", "besar", "kecil", "bola"),
        ("kue manis", "cokelat", "keju", "kue"),
    ]
    for item_type, part_a, part_b, unit in scenarios:
        for a in range(1, 6):
            for b in range(1, 6):
                if a + b <= max_sum:
                    for s in generate_counting_samples(item_type, part_a, part_b, a, b, unit):
                        items.append({
                            "stage": 5,
                            "stage_name": "berhitung",
                            "topic": f"Hitung {item_type.title()} ({a}+{b})",
                            **s,
                        })
    return items

