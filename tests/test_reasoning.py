"""Unit tests for Compositional Reasoning and Multi-Hop Deduction."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import nested_learning as nl


def run_reasoning_tests():
    print("=" * 60)
    print("TEST: Compositional Reasoning & Multi-Hop Deduction")
    print("=" * 60)

    # 1. Test Query Decomposition
    query = "Berapa jumlah jari tangan kanan dan kiri?"
    decomp = nl.CompositionalReasoner.decompose_query(query)
    assert decomp is not None, "Failed to decompose query"
    part_a, part_b, op = decomp
    print(f"Query: \"{query}\"")
    print(f"Decomposition: part_a='{part_a}', part_b='{part_b}', op='{op}'")
    assert "kanan" in part_a and "kiri" in part_b, "Sub-parts extraction mismatch"

    # 2. Test Multi-Hop Composition with mock retriever
    def mock_model_ask(q: str) -> str:
        if "kanan" in q:
            return "berjumlah 5."
        elif "kiri" in q:
            return "berjumlah 5."
        return "tidak tahu."

    result = nl.CompositionalReasoner.answer_composite(query, mock_model_ask)
    assert result is not None, "Failed to perform composite reasoning"
    assert result["val_a"] == 5 and result["val_b"] == 5, "Extracted values mismatch"
    assert result["total"] == 10, f"Expected 10, got {result['total']}"

    print("\nChain-of-Thought Result:")
    print(result["cot_explanation"])
    print(f"Concise Answer: {result['concise_answer']}")

    # 3. Test Cross-Fact Auto Synthesis
    history = ["Jari tangan kanan berjumlah 5."]
    new_fact = "Jari tangan kiri berjumlah 5."
    syn = nl.CompositionalReasoner.synthesize_cross_facts(new_fact, history)
    assert len(syn) >= 1, "Failed to synthesize cross facts"
    print(f"\nCross-Fact Synthesis from '{history[0]}' + '{new_fact}':")
    for s in syn:
        print(f"  • {s}")
        assert "10" in s, "Synthesized fact must contain 10"

    # 4. Test Direct Arithmetic Expression Deduction (e.g. 10 + 5)
    direct_query = "berapa jumlah 10 + 5"
    direct_res = nl.CompositionalReasoner.answer_composite(direct_query)
    assert direct_res is not None, "Failed to compute direct arithmetic"
    assert direct_res["val_a"] == 10 and direct_res["val_b"] == 5, "Operands mismatch"
    assert direct_res["total"] == 15, f"Expected 15, got {direct_res['total']}"
    # 5. Test Arbitrary Nouns & Typos (e.g. 'jumla kelereng 5 dan 15 adalah')
    kelereng_q = "jumla kelereng 5 dan 15 adalah"
    kel_res = nl.CompositionalReasoner.answer_composite(kelereng_q)
    assert kel_res is not None, "Failed to resolve kelereng query"
    assert kel_res["total"] == 20, f"Expected 20, got {kel_res['total']}"
    print(f"\nArbitrary Noun Query: \"{kelereng_q}\"")
    print(f"Result: {kel_res['concise_answer']}")

    print("\n✓ Compositional Reasoning tests PASSED!\n")


if __name__ == "__main__":
    run_reasoning_tests()
