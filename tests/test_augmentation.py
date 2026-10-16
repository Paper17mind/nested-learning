"""Unit tests for Text Data Augmentation and Counting / Arithmetic generation."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import nested_learning as nl


def run_augmentation_tests():
    print("=" * 60)
    print("TEST: Text Augmentation & Counting Capabilities")
    print("=" * 60)

    # 1. Test counting structural patterns
    fact_text = "Jari tangan kanan 5 dan jari tangan kiri 5 totalnya 10."
    variants = nl.augment_text(fact_text, max_variations=4)
    print(f"Original: {fact_text}")
    print(f"Augmented variants ({len(variants)}):")
    for v in variants:
        print(f"  - {v}")

    assert len(variants) >= 2, f"Expected at least 2 variants, got {len(variants)}"
    assert any("Jumlah" in v or "total" in v.lower() or "keseluruhannya" in v for v in variants), "Structural pattern flip missing"

    # 2. Test counting sample generator (specific hand fingers test)
    samples = nl.generate_counting_samples("jari tangan", "kanan", "kiri", 5, 5, "jari")
    assert len(samples) >= 4, f"Expected at least 4 sample variations, got {len(samples)}"
    print("\nCounting QA Samples for 5+5 Fingers:")
    for s in samples:
        print(f"  Q: {s['question']}")
        print(f"  A: {s['answer']}")
        assert "10" in s["answer"] or "sepuluh" in s["answer"], "Answer must contain total 10"

    # 3. Test general counting dataset generator
    ds = nl.generate_counting_dataset(max_sum=10)
    assert len(ds) > 50, f"Expected large dataset, got {len(ds)} items"
    print(f"\nSuccessfully generated {len(ds)} developmental arithmetic dataset items.")

    print("\n✓ Text Augmentation and Counting tests PASSED!\n")


if __name__ == "__main__":
    run_augmentation_tests()
