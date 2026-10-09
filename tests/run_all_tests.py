"""Master Test Runner for Nested Learning Lite.

Executes all test suites:
1. State-Passing Equivalence (O(N) inference)
2. Delta Rule & Associative Memory Integrity
3. CMS Multi-Frequency Tier Schedule
4. Numerical Gradient Checks (All Layers)
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_cms_tiers import test_tier_update_schedule
from tests.test_delta_rule import (
    test_checkpoint_roundtrip,
    test_delta_rule_convergence,
    test_mask_integrity,
)
from tests.test_equivalence import run_equivalence_test
from tests.test_gradients import run_all as run_gradient_tests


def main():
    print("=" * 70)
    print("NESTED LEARNING LITE: MASTER TEST SUITE")
    print("Reference: Google Research NeurIPS 2025 (Behrouz et al.)")
    print("=" * 70)

    t0 = time.time()
    passed = 0
    total = 6

    # 1. State-Passing Equivalence
    try:
        run_equivalence_test()
        passed += 1
    except Exception as e:
        print(f"FAILED: State-Passing Equivalence ({e})")

    # 2. Delta Rule Convergence
    try:
        test_delta_rule_convergence()
        passed += 1
    except Exception as e:
        print(f"FAILED: Delta Rule Convergence ({e})")

    # 3. Memory Mask Integrity
    try:
        test_mask_integrity()
        passed += 1
    except Exception as e:
        print(f"FAILED: Mask Integrity ({e})")

    # 4. Checkpoint Serialization
    try:
        test_checkpoint_roundtrip()
        passed += 1
    except Exception as e:
        print(f"FAILED: Checkpoint Roundtrip ({e})")

    # 5. CMS Tiers Schedule
    try:
        test_tier_update_schedule()
        passed += 1
    except Exception as e:
        print(f"FAILED: CMS Tiers Schedule ({e})")

    # 6. Numerical Gradients
    try:
        run_gradient_tests()
        passed += 1
    except Exception as e:
        print(f"FAILED: Numerical Gradients ({e})")

    elapsed = time.time() - t0
    print("=" * 70)
    print(f"SUMMARY: {passed}/{total} Test Suites PASSED in {elapsed:.2f} seconds")
    print("=" * 70)

    if passed == total:
        print("✓ ALL BEHAVIORAL & MATHEMATICAL TESTS PASSED SUCCESSFULLY!")
        return 0
    else:
        print("✗ SOME TESTS FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
