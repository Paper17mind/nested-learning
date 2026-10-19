"""Compositional Reasoning and Working Memory Deduction for Nested Learning Lite.

Features:
1. Universal Numeric Intent Resolution:
   Extracts quantities (digits or Indonesian number words) and infers operations
   (sum, subtraction, multiplication, division) without fragile regex templates.
   Handles typos (e.g., 'jumla', 'totall') and arbitrary nouns ('kelereng', 'apel', 'jari').
2. Multi-Hop Entity Decomposition:
   When numbers are not in the prompt (e.g., "Berapa jumlah jari tangan kanan dan kiri?"),
   decomposes into sub-queries and aggregates values retrieved from memory.
3. Cross-Fact Auto-Synthesis:
   Infers relational links when separate premises are taught in continual learning.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple


WORD_TO_NUM: Dict[str, int] = {
    "nol": 0, "satu": 1, "dua": 2, "tiga": 3, "empat": 4,
    "lima": 5, "enam": 6, "tujuh": 7, "delapan": 8, "sembilan": 9,
    "sepuluh": 10, "sebelas": 11, "dua belas": 12, "tiga belas": 13,
    "empat belas": 14, "lima belas": 15, "enam belas": 16, "tujuh belas": 17,
    "delapan belas": 18, "sembilan belas": 19, "dua puluh": 20,
    "tiga puluh": 30, "empat puluh": 40, "lima puluh": 50, "seratus": 100,
}


def extract_all_numbers(text: str) -> List[int]:
    """Extract all numbers from text in order of appearance (digits & word numbers)."""
    found = []
    # 1. Digits
    for m in re.finditer(r"\b\d+\b", text):
        found.append((m.start(), int(m.group(0))))

    # 2. Number words
    text_lower = text.lower()
    for word_num, val in sorted(WORD_TO_NUM.items(), key=lambda x: -len(x[0])):
        for m in re.finditer(rf"\b{word_num}\b", text_lower):
            found.append((m.start(), val))

    # Sort by character position in the input string
    found.sort(key=lambda x: x[0])
    # Filter overlapping matches
    result = []
    last_pos = -1
    for pos, val in found:
        if pos > last_pos:
            result.append(val)
            last_pos = pos + 1
    return result


def extract_number(text: str) -> Optional[int]:
    """Extract the first integer quantity found in text."""
    nums = extract_all_numbers(text)
    return nums[0] if nums else None


class CompositionalReasoner:
    """Universal Reasoning and Arithmetic Deduction Engine."""

    @classmethod
    def is_reasoning_query(cls, query: str) -> bool:
        """Check if query requires arithmetic or compositional reasoning."""
        nums = extract_all_numbers(query)
        # If query has 2 or more numbers, it's an arithmetic query
        if len(nums) >= 2:
            return True
        # If query asks to combine entities (e.g., 'kanan dan kiri')
        if cls.decompose_query(query) is not None:
            return True
        return False

    @staticmethod
    def detect_arithmetic_operation(text: str) -> Tuple[str, str, str]:
        """Detect mathematical operation from contextual keywords without rigid grammar.

        Returns: (op_name, symbol, verb)
        """
        t = text.lower()
        if any(k in t for k in ["kurang", "dikurangi", "sisa", "minus", "berkurang", "ambil", "dimakan", "-"]):
            return "kurang", "-", "dikurangi"
        if any(k in t for k in ["kali", "dikali", "x", "*"]):
            return "kali", "x", "dikali"
        if any(k in t for k in ["bagi", "dibagi", "/", ":"]):
            return "bagi", "/", "dibagi"
        # Default operation is addition (penjumlahan / penggabungan)
        return "tambah", "+", "ditambah"

    @classmethod
    def answer_composite(
        cls,
        query: str,
        query_model_fn: Optional[Callable[[str], str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Resolve arithmetic deduction or multi-hop entity reasoning.

        1. Universal Arithmetic: Solves questions with explicit numbers (any noun, typos allowed).
        2. Multi-hop Entity: Solves questions referencing entities stored in memory.
        """
        clean_q = query.strip()
        nums = extract_all_numbers(clean_q)

        # CASE 1: Query contains at least 2 numbers (Direct Arithmetic with ANY noun or wording)
        if len(nums) >= 2:
            op_name, symbol, verb = cls.detect_arithmetic_operation(clean_q)
            val_a, val_b = nums[0], nums[1]

            if op_name == "tambah":
                total = sum(nums)
                step_str = " + ".join(str(n) for n in nums)
            elif op_name == "kurang":
                total = val_a - sum(nums[1:])
                step_str = f"{val_a} - {val_b}"
            elif op_name == "kali":
                total = val_a * val_b
                step_str = f"{val_a} x {val_b}"
            elif op_name == "bagi":
                total = val_a // val_b if val_b != 0 else 0
                step_str = f"{val_a} / {val_b}"
            else:
                total = sum(nums)
                step_str = f"{val_a} + {val_b}"

            cot_explanation = (
                f"Perhitungan aritmatika:\n"
                f"1. Nilai yang ditemukan: {', '.join(str(n) for n in nums)}\n"
                f"2. Operasi             : {verb} ({symbol})\n"
                f"Langkah hitung: {step_str} = {total}."
            )
            concise_answer = f"{total}."
            synthesized_fact = f"{step_str} adalah {total}."

            return {
                "type": "direct_arithmetic",
                "val_a": val_a,
                "val_b": val_b,
                "total": total,
                "cot_explanation": cot_explanation,
                "concise_answer": concise_answer,
                "synthesized_fact": synthesized_fact,
            }

        # CASE 2: Multi-hop Entity Decomposition (No explicit numbers in query)
        decomp = cls.decompose_query(clean_q)
        if decomp and query_model_fn is not None:
            part_a, part_b, _ = decomp
            ans_a = query_model_fn(f"Berapa jumlah {part_a}?")
            val_a = extract_number(ans_a)

            ans_b = query_model_fn(f"Berapa jumlah {part_b}?")
            val_b = extract_number(ans_b)

            if val_a is not None and val_b is not None:
                total = val_a + val_b
                cot = (
                    f"Berdasarkan memori:\n"
                    f"1. {part_a.capitalize()} = {val_a}\n"
                    f"2. {part_b.capitalize()} = {val_b}\n"
                    f"Langkah hitung: {val_a} + {val_b} = {total}."
                )
                concise = f"{part_a.capitalize()} ada {val_a} dan {part_b} ada {val_b}, totalnya {total}."
                return {
                    "type": "multi_hop_composite",
                    "part_a": part_a,
                    "part_b": part_b,
                    "val_a": val_a,
                    "val_b": val_b,
                    "total": total,
                    "cot_explanation": cot,
                    "concise_answer": concise,
                    "synthesized_fact": f"{part_a} {val_a} dan {part_b} {val_b} totalnya {total}.",
                }

        return None

    @staticmethod
    def decompose_query(query: str) -> Optional[Tuple[str, str, str]]:
        """Split a composite question into two entities using 'dan' connector.

        Example:
            'Berapa jumlah jari tangan kanan dan kiri?'
            -> ('jari tangan kanan', 'jari tangan kiri', 'penjumlahan')
        """
        clean_q = query.strip().rstrip("?.!:").lower()

        # Split flexibly on ' dan '
        if " dan " not in clean_q:
            return None

        # Clean common question prefixes
        clean_core = re.sub(
            r"^(?:berapa|hitung|apakah|jika)?\s*(?:jumlah|jumla|total|totall|banyaknya|hasil\s+dari)?\s*",
            "",
            clean_q,
        ).strip()

        parts = clean_core.split(" dan ", 1)
        if len(parts) != 2:
            return None

        part_a = parts[0].strip()
        part_b = re.sub(r"\s*(?:adalah|totalnya|jumlahnya|berapa|maka|hasilnya)\s*$", "", parts[1]).strip()

        # Handle distributed nouns: e.g. 'jari tangan kanan' dan 'kiri' -> 'jari tangan kiri'
        words_a = part_a.split()
        words_b = part_b.split()
        if len(words_a) > 1 and len(words_b) == 1:
            base_noun = " ".join(words_a[:-1])
            part_b = f"{base_noun} {words_b[0]}"

        return part_a, part_b, "penjumlahan"

    @staticmethod
    def extract_fact_entity_value(fact: str) -> Optional[Tuple[str, str, int]]:
        """Extract entity base, modifier/part, and quantity from a factual sentence."""
        val = extract_number(fact)
        if val is None:
            return None

        clean = fact.strip().rstrip(".").lower()
        clean = re.sub(r"\b(\d+|berjumlah|ada|memiliki|mempunyai|totalnya|sebanyak)\b", "", clean)
        words = clean.split()
        if len(words) >= 2:
            return " ".join(words[:-1]), words[-1], val
        elif len(words) == 1:
            return words[0], "", val
        return None

    @classmethod
    def synthesize_cross_facts(
        cls,
        new_fact: str,
        existing_facts: List[str],
    ) -> List[str]:
        """Automatically synthesize cross-fact compositional relationships."""
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

            if base_new == base_old and part_new != part_old and part_new and part_old:
                tot = val_old + val_new
                syn = f"{base_new.capitalize()} {part_old} {val_old} dan {base_new} {part_new} {val_new} totalnya {tot}."
                synthesized.append(syn)

        return synthesized
