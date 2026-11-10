"""
SpendScope Unified Test Runner.

Executes all unit and integration test suites cleanly across platforms,
reporting status, timing, and pass/fail counts.
"""

from __future__ import annotations

import sys
import time
import traceback
from typing import Callable, List, Tuple

import tests.test_analytics as ta
import tests.test_anomaly_and_recurring as tar
import tests.test_behavior as tbeh
import tests.test_budget as tb
import tests.test_categorization as tc
import tests.test_forecasting as tfore
import tests.test_ingestion as ti
import tests.test_layman as tlay
import tests.test_pipeline as tpipe
import tests.test_preprocessing as tp
import tests.test_shifts as tshift


def collect_test_functions(module) -> List[Tuple[str, Callable]]:
    """Inspect module and return list of (function_name, callable)."""
    tests = []
    for attr_name in sorted(dir(module)):
        if attr_name.startswith("test_"):
            attr = getattr(module, attr_name)
            if callable(attr):
                tests.append((attr_name, attr))
    return tests


def run_all_tests() -> int:
    """Run all SpendScope tests and report results."""
    modules = [
        ("Ingestion Suite", ti),
        ("Preprocessing Suite", tp),
        ("Categorization Suite", tc),
        ("Pipeline Integration Suite", tpipe),
        ("Analytics & Insights Suite", ta),
        ("Anomaly Detection & DB Suite", tar),
        ("Budget Goals Suite", tb),
        ("Spending Behaviour Suite", tbeh),
        ("Spending Shift Suite", tshift),
        ("Financial Outlook Suite", tfore),
        ("Layman Intelligence & Health Suite", tlay),
    ]

    print("=" * 70)
    print("SpendScope Test Suite Runner")
    print("=" * 70)

    total_run = 0
    total_passed = 0
    total_failed = 0
    start_all = time.perf_counter()

    for suite_name, mod in modules:
        print(f"\n--- Running: {suite_name} ({mod.__name__}) ---")
        tests = collect_test_functions(mod)

        for name, func in tests:
            total_run += 1
            t0 = time.perf_counter()
            try:
                func()
                dt = (time.perf_counter() - t0) * 1000.0
                print(f"  [PASS] {name} ({dt:.1f}ms)")
                total_passed += 1
            except Exception as e:
                dt = (time.perf_counter() - t0) * 1000.0
                print(f"  [FAIL] {name} ({dt:.1f}ms)")
                print(f"         Error: {e}")
                traceback.print_exc(file=sys.stdout)
                total_failed += 1

    total_time = (time.perf_counter() - start_all) * 1000.0
    print("\n" + "=" * 70)
    print(
        f"TEST RESULTS: {total_passed}/{total_run} PASSED "
        f"({total_failed} failed) in {total_time:.1f}ms"
    )
    print("=" * 70)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
