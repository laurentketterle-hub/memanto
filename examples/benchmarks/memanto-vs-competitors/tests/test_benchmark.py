"""Unit tests for the memanto-vs-competitors benchmark.

Validates:
- All adapters implement the MemorySystem protocol
- All scenarios produce valid data
- Scoring/computation is correct
- Dry runs succeed for all adapters
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the benchmark package is importable
_BENCH_DIR = Path(__file__).resolve().parents[1]
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from adapters import ADAPTER_REGISTRY
from adapters.base import (
    DialogueTurn,
    EvidenceSpan,
    MemorySystem,
    QAPair,
    RetrievedItem,
    compute_scores,
    evaluate_query,
)
from benchmark import run_single_system
from datasets import ALL_SCENARIOS


def test_all_adapters_registered() -> bool:
    """Verify all five adapters are registered."""
    expected = {"memanto", "mem0", "zep", "letta", "langmem"}
    actual = set(ADAPTER_REGISTRY.keys())
    assert expected == actual, f"Expected {expected}, got {actual}"
    print("  [PASS] All 5 adapters registered")
    return True


def test_adapter_interface() -> bool:
    """Verify each adapter implements the MemorySystem protocol."""
    for name, cls in ADAPTER_REGISTRY.items():
        adapter = cls()
        assert isinstance(adapter, MemorySystem), f"{name} does not implement MemorySystem"
        assert hasattr(adapter, "name"), f"{name} missing name()"
        assert hasattr(adapter, "setup"), f"{name} missing setup()"
        assert hasattr(adapter, "store_turns"), f"{name} missing store_turns()"
        assert hasattr(adapter, "search"), f"{name} missing search()"
        assert hasattr(adapter, "teardown"), f"{name} missing teardown()"
        assert hasattr(adapter, "cleanup"), f"{name} missing cleanup()"
        assert hasattr(adapter, "dry_run"), f"{name} missing dry_run()"
        print(f"  [PASS] {name} implements MemorySystem protocol")
    return True


def test_adapter_dry_runs() -> bool:
    """Verify all adapters pass dry-run smoke test (in-memory mode)."""
    for name, cls in ADAPTER_REGISTRY.items():
        adapter = cls()
        ok = adapter.dry_run()
        assert ok, f"{name} dry_run failed"
        print(f"  [PASS] {name} dry-run OK")
    return True


def test_scenario_data_integrity() -> bool:
    """Verify all scenarios produce valid data."""
    for name, fn in ALL_SCENARIOS.items():
        scenario = fn()
        assert scenario.name == name, f"Scenario name mismatch: {scenario.name} != {name}"
        assert len(scenario.conversations) > 0, f"{name} has no conversations"
        for conv in scenario.conversations:
            assert len(conv.turns) > 0, f"{name}/{conv.sample_id} has no turns"
            for turn in conv.turns:
                assert turn.dia_id, f"Empty dia_id in {conv.sample_id}"
                assert turn.text, f"Empty text in {conv.sample_id}"
            for qa in conv.qa_pairs:
                assert qa.question, f"Empty question in {conv.sample_id}"
                assert qa.answer, f"Empty answer in {conv.sample_id}"
                assert qa.category, f"Empty category in {conv.sample_id}"
        print(f"  [PASS] {name}: {len(scenario.conversations)} convs, "
              f"{scenario.total_turns} turns, {scenario.total_qa_pairs} QAs")
    return True


def test_scoring_math() -> bool:
    """Verify scoring computations are correct."""

    # Perfect recall case
    qa = QAPair(
        question="What color is the sky?",
        answer="blue",
        category="basic",
        evidence=[EvidenceSpan(session_id="s1", turn_ids=["T1"])],
    )
    retrieved = [RetrievedItem(dia_id="T1", session_id="s1", text="The sky is blue", score=1.0)]
    recall, precision, has_stale, stale_count = compute_scores(retrieved, qa, k=10)
    assert recall == 1.0, f"Perfect recall should be 1.0, got {recall}"
    assert precision == 1.0, f"Perfect precision should be 1.0, got {precision}"
    print("  [PASS] Perfect recall scoring")

    # Zero recall case
    retrieved2 = [RetrievedItem(dia_id="T99", session_id="s99", text="Unrelated", score=0.0)]
    recall2, precision2, has_stale2, sc2 = compute_scores(retrieved2, qa, k=10)
    assert recall2 == 0.0, f"Zero recall should be 0.0, got {recall2}"
    assert precision2 == 0.0, f"Zero precision should be 0.0, got {precision2}"
    print("  [PASS] Zero recall scoring")

    return True


def test_benchmark_runs() -> bool:
    """Verify the benchmark runs end-to-end with a single scenario and adapter."""
    scenario_fn = ALL_SCENARIOS["basic_recall"]
    scenario = scenario_fn()
    adapter_cls = ADAPTER_REGISTRY["memanto"]
    adapter = adapter_cls()

    score = run_single_system(adapter, scenario, k=10, judge=None, verbose=False)

    assert score.system_name == "memanto"
    assert score.total_qas > 0, f"Expected >0 QAs, got {score.total_qas}"
    assert 0.0 <= score.avg_recall <= 1.0, f"Recall out of range: {score.avg_recall}"
    assert 0.0 <= score.avg_precision <= 1.0, f"Precision out of range: {score.avg_precision}"

    print(f"  [PASS] Benchmark runs: {score.total_qas} QAs, "
          f"recall={score.avg_recall:.2f}, precision={score.avg_precision:.2f}")
    return True


def run_tests() -> bool:
    """Run all tests and return True if all pass."""
    print("\n" + "=" * 60)
    print("  Memanto vs Competitors — Test Suite")
    print("=" * 60)

    tests = [
        ("Adapter Registration", test_all_adapters_registered),
        ("Adapter Interface", test_adapter_interface),
        ("Adapter Dry Runs", test_adapter_dry_runs),
        ("Scenario Integrity", test_scenario_data_integrity),
        ("Scoring Math", test_scoring_math),
        ("Benchmark Execution", test_benchmark_runs),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 60}")

    return failed == 0


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
