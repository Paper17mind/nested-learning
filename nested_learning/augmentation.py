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
from typing import Dict, List, Optional, Set, Tuple


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
    
    # Kata bantu / fungsional
    "sangat": ["sekali", "amat", ""],
    "sekali": ["sangat", "amat", ""],
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
