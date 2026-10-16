"""Compositional Reasoning and Working Memory Deduction for Nested Learning Lite.

Implements:
1. Composite Query Decomposition: breaks questions like
   "Berapa jumlah jari tangan kanan dan kiri?" into sub-queries.
2. Multi-Hop Step Retrieval: queries the model/memory for each component.
3. Chain-of-Thought (CoT) Deduction: performs arithmetic synthesis (n1 + n2 = n_total)
   and produces structured step-by-step reasoning.
4. Cross-Fact Auto-Synthesis: automatically infers composite relationships
   when separate facts are learned in continual learning.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


WORD_TO_NUM: Dict[str, int] = {
    "nol": 0, "satu": 1, "dua": 2, "tiga": 3, "empat": 4,
    "lima": 5, "enam": 6, "tujuh": 7, "delapan": 8, "sembilan": 9,
    "sepuluh": 10, "sebelas": 11, "dua belas": 12, "tiga belas": 13,
    "empat belas": 14, "lima belas": 15, "enam belas": 16, "tujuh belas": 17,
    "delapan belas": 18, "sembilan belas": 19, "dua puluh": 20,
}


def extract_number(text: str) -> Optional[int]:
    """Extract an integer quantity from numeric digits or Indonesian number words."""
    # 1. Match literal digits
    m = re.search(r"\b(\d+)\b", text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass

    # 2. Match two-word numbers like 'dua puluh', 'tiga belas'
    text_lower = text.lower()
    for word_num, val in sorted(WORD_TO_NUM.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{word_num}\b", text_lower):
            return val

    return None


OP_MAP: Dict[str, Tuple[str, str, str]] = {
    "+": ("tambah", "+", "ditambah"),
    "plus": ("tambah", "+", "ditambah"),
    "ditambah": ("tambah", "+", "ditambah"),
    "tambah": ("tambah", "+", "ditambah"),
    "-": ("kurang", "-", "dikurangi"),
    "minus": ("kurang", "-", "dikurangi"),
    "dikurangi": ("kurang", "-", "dikurangi"),
    "kurang": ("kurang", "-", "dikurangi"),
    "*": ("kali", "x", "dikali"),
    "x": ("kali", "x", "dikali"),
    "dikali": ("kali", "x", "dikali"),
    "kali": ("kali", "x", "dikali"),
    "/": ("bagi", "/", "dibagi"),
    "dibagi": ("bagi", "/", "dibagi"),
    "bagi": ("bagi", "/", "dibagi"),
}


class CompositionalReasoner:
    """Engine for Multi-Hop Compositional Reasoning and Arithmetic Deduction."""

    @staticmethod
    def parse_direct_arithmetic(query: str) -> Optional[Tuple[int, int, str, str, str]]:
        """Parse direct arithmetic questions like:
        - 'berapa jumlah 10 + 5'
        - 'berapa 10 + 5'
        - '10 + 5 berapa'
        - '10 ditambah 5'
        - 'berapa hasil dari 10 + 5'
        Returns: (val_a, val_b, op_name, symbol, verb)
        """
        clean_q = query.strip().rstrip("?.!").lower()

        # Regex matching numbers (digits or word numbers) and arithmetic operators
        num_pattern = r"(\d+|satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh|sebelas|dua belas|tiga belas|empat belas|lima belas|enam belas|tujuh belas|delapan belas|sembilan belas|dua puluh)"
        op_pattern = r"(\+|\-|x|\*|/|ditambah|dikurangi|dikali|dibagi|plus|minus|tambah|kurang|kali|bagi)"

        pattern = re.compile(
            rf"^(?:berapa|hitung|apakah)?\s*(?:jumlah|total|hasil|hasil\s+dari|hasil\s+penjumlahan)?\s*(?:dari\s+)?{num_pattern}\s*{op_pattern}\s*{num_pattern}\s*(?:berapa|sama\s+dengan\s+berapa)?$",
            re.IGNORECASE,
        )
        m = pattern.match(clean_q)
        if not m:
            return None

        str_a, str_op, str_b = m.groups()
        val_a = extract_number(str_a)
        val_b = extract_number(str_b)
        if val_a is None or val_b is None:
            return None

        op_info = OP_MAP.get(str_op.strip().lower(), ("tambah", "+", "ditambah"))
        op_name, symbol, verb = op_info
        return val_a, val_b, op_name, symbol, verb

    @classmethod
    def is_reasoning_query(cls, query: str) -> bool:
        """Check if query is either a direct arithmetic problem or an entity composite query."""
        return (cls.parse_direct_arithmetic(query) is not None) or (cls.decompose_query(query) is not None)

    @staticmethod
    def decompose_query(query: str) -> Optional[Tuple[str, str, str]]:
        """Decompose a composite question into sub-queries.

        Example:
            'Berapa jumlah jari tangan kanan dan kiri?'
            -> ('jari tangan kanan', 'jari tangan kiri', 'penjumlahan')
        """
        clean_q = query.strip().rstrip("?.!").lower()

        # Pola: 'berapa (jumlah|total|banyaknya) [subjek_a] dan [subjek_b]'
        pattern = re.compile(
            r"^(?:berapa|hitung|apakah)\s+(?:jumlah|total|banyaknya|hasil\s+dari|hasil\s+penjumlahan)\s+([A-Za-z0-9\s]+?)\s+dan\s+([A-Za-z0-9\s]+?)$",
            re.IGNORECASE,
        )
        m = pattern.match(clean_q)
        if not m:
            # Pola alternatif: 'jika [subjek_a] ... dan [subjek_b] ... berapa totalnya'
            pattern_alt = re.compile(
                r"^(?:jika\s+)?([A-Za-z0-9\s]+?)\s+dan\s+([A-Za-z0-9\s]+?)\s+(?:berapa|maka)\s+(?:jumlahnya|totalnya|hasilnya)$",
                re.IGNORECASE,
            )
            m = pattern_alt.match(clean_q)

        if not m:
            return None

        part_a = m.group(1).strip()
        part_b = m.group(2).strip()

        # Handle distributed noun phrases: e.g. 'jari tangan kanan' and 'kiri' -> 'jari tangan kiri'
        words_a = part_a.split()
        words_b = part_b.split()

        if len(words_a) > 1 and len(words_b) == 1:
            # Check if words_b is a modifier like 'kiri' when words_a ends with 'kanan'
            modifier_b = words_b[0]
            base_noun = " ".join(words_a[:-1])
            part_b = f"{base_noun} {modifier_b}"
        elif len(words_b) > 1 and len(words_a) == 1:
            modifier_a = words_a[0]
            base_noun = " ".join(words_b[:-1])
            part_a = f"{base_noun} {modifier_a}"

        return part_a, part_b, "penjumlahan"

    @classmethod
    def answer_composite(
        cls,
        query: str,
        query_model_fn: Optional[Callable[[str], str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Perform multi-hop reasoning or direct arithmetic deduction.

        1. Handles direct arithmetic expressions like '10 + 5' or 'berapa jumlah 10 + 5'.
        2. Decomposes multi-hop entity queries and retrieves knowledge for each sub-query.
        3. Extracts quantities and computes deduction with Chain-of-Thought (CoT).
        """
        # 1. Check direct arithmetic expression
        direct = cls.parse_direct_arithmetic(query)
        if direct:
            val_a, val_b, op_name, symbol, verb = direct
            if op_name == "tambah":
                total = val_a + val_b
            elif op_name == "kurang":
                total = val_a - val_b
            elif op_name == "kali":
                total = val_a * val_b
            elif op_name == "bagi":
                total = val_a // val_b if val_b != 0 else 0
            else:
                total = val_a + val_b

            cot_explanation = (
                f"Perhitungan aritmatika langsung:\n"
                f"1. Nilai pertama : {val_a}\n"
                f"2. Operasi       : {verb} ({symbol}) {val_b}\n"
                f"Langkah hitung: {val_a} {symbol} {val_b} = {total}."
            )
            concise_answer = f"Hasil dari {val_a} {symbol} {val_b} adalah {total}."
            synthesized_fact = f"{val_a} {verb} {val_b} hasilnya adalah {total}."

            return {
                "type": "direct_arithmetic",
                "part_a": str(val_a),
                "part_b": str(val_b),
                "val_a": val_a,
                "val_b": val_b,
                "total": total,
                "cot_explanation": cot_explanation,
                "concise_answer": concise_answer,
                "synthesized_fact": synthesized_fact,
            }

        # 2. Check multi-hop entity composite query
        decomp = cls.decompose_query(query)
        if not decomp or query_model_fn is None:
            return None

        part_a, part_b, operation = decomp

        # Sub-query 1
        q_a = f"Berapa jumlah {part_a}?"
        ans_a = query_model_fn(q_a)
        val_a = extract_number(ans_a)

        # Sub-query 2
        q_b = f"Berapa jumlah {part_b}?"
        ans_b = query_model_fn(q_b)
        val_b = extract_number(ans_b)

        if val_a is None or val_b is None:
            # Cannot deduce if either premise is unknown
            return None

        total = val_a + val_b

        # Construct Chain-of-Thought (CoT) explanation
        cot_explanation = (
            f"Berdasarkan informasi yang dipelajari:\n"
            f"1. {part_a.capitalize()} berjumlah {val_a}.\n"
            f"2. {part_b.capitalize()} berjumlah {val_b}.\n"
            f"Jika dijumlahkan ({val_a} + {val_b}), maka total {part_a} dan {part_b} adalah {total}."
        )

        concise_answer = f"{part_a.capitalize()} ada {val_a} dan {part_b} ada {val_b}, sehingga total keseluruhannya adalah {total}."

        return {
            "type": "multi_hop_composite",
            "part_a": part_a,
            "part_b": part_b,
            "val_a": val_a,
            "val_b": val_b,
            "total": total,
            "cot_explanation": cot_explanation,
            "concise_answer": concise_answer,
            "synthesized_fact": f"{part_a.capitalize()} {val_a} dan {part_b} {val_b} totalnya {total}.",
        }

    @staticmethod
    def extract_fact_entity_value(fact: str) -> Optional[Tuple[str, str, int]]:
        """Extract entity base, modifier/part, and quantity from a factual sentence.

        Example:
            'Jari tangan kanan berjumlah 10.' -> ('jari tangan', 'kanan', 10)
            'Jari tangan kiri 10.' -> ('jari tangan', 'kiri', 10)
        """
        val = extract_number(fact)
        if val is None:
            return None

        clean = fact.strip().rstrip(".").lower()
        # Remove verbs and numbers
        clean = re.sub(r"\b(\d+|berjumlah|ada|memiliki|mempunyai|totalnya|sebanyak)\b", "", clean)
        clean = " ".join(clean.split())

        words = clean.split()
        if len(words) >= 2:
            base_entity = " ".join(words[:-1])
            part = words[-1]
            return base_entity, part, val
        elif len(words) == 1:
            return words[0], "", val

        return None

    @classmethod
    def synthesize_cross_facts(
        cls,
        new_fact: str,
        existing_facts: List[str],
    ) -> List[str]:
        """Automatically synthesize cross-fact compositional relationships.

        When a new fact (e.g. 'Jari tangan kiri berjumlah 10.') is learned,
        and an existing fact (e.g. 'Jari tangan kanan berjumlah 10.') is present,
        synthesize:
        'Jari tangan kanan 10 dan jari tangan kiri 10 totalnya 20.'
        """
        new_info = cls.extract_fact_entity_value(new_fact)
        if not new_info:
            return []

        base_new, part_new, val_new = new_info
        synthesized = []

        for old_fact in existing_facts:
            old_info = cls.extract_fact_entity_value(old_fact)
            if not old_info:
                continue
            base_old, part_old, val_old = old_info

            # Same entity, different parts (e.g. kanan vs kiri, merah vs hijau)
            if base_new == base_old and part_new != part_old and part_new and part_old:
                tot = val_old + val_new
                syn1 = f"{base_new.capitalize()} {part_old} {val_old} dan {base_new} {part_new} {val_new} totalnya {tot}."
                syn2 = f"Jumlah {base_new} {part_old} dan {part_new} adalah {tot}."
                synthesized.extend([syn1, syn2])

        return synthesized
