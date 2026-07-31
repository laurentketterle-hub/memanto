#!/usr/bin/env python3
"""CLI entry point for Memanto vs Competitors benchmark.

Usage:
    python run_benchmark.py
    python run_benchmark.py --adapters memanto mem0
    python run_benchmark.py --scenarios basic_recall temporal
    python run_benchmark.py --k 10 --output results.json
    python run_benchmark.py --list
    python run_benchmark.py --test
"""

from __future__ import annotations

import argparse
import sys

from adapters import ADAPTER_REGISTRY
from benchmark import format_report, run_benchmark, save_results
from datasets import ALL_SCENARIOS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Memanto vs Competitors — Memory System Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py                              # Run all adapters, all scenarios
  python run_benchmark.py --adapters memanto mem0      # Compare Memanto vs Mem0
  python run_benchmark.py --scenarios temporal         # Only temporal scenario
  python run_benchmark.py --output results.json        # Save to JSON
  python run_benchmark.py --test                       # Run self-test
  python run_benchmark.py --list                       # List available adapters/scenarios
        """,
    )
    parser.add_argument(
        "--adapters", nargs="+", default=None,
        help="Memory systems to test (default: all)",
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Scenarios to run (default: all)",
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Top-k for recall/precision (default: 10)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path for results",
    )
    parser.add_argument(
        "--judge-model", type=str, default=None,
        help="LLM model for judging (e.g. gpt-4o-mini). Requires OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available adapters and scenarios, then exit",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run self-test to validate benchmark integrity",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # --list
    if args.list:
        print("\nAvailable Adapters:")
        for name in sorted(ADAPTER_REGISTRY.keys()):
            print(f"  {name}")
        print("\nAvailable Scenarios:")
        for name in sorted(ALL_SCENARIOS.keys()):
            fn = ALL_SCENARIOS[name]
            scenario = fn()
            print(f"  {name}: {scenario.description}")
            print(f"    ({scenario.total_turns} turns, {scenario.total_qa_pairs} QA pairs)")
        return

    # --test
    if args.test:
        print("\nRunning self-test...")
        from tests.test_benchmark import run_tests
        ok = run_tests()
        sys.exit(0 if ok else 1)

    # Normal benchmark
    print("\n" + "=" * 70)
    print("  Memanto vs Competitors — Memory System Benchmark")
    print("=" * 70)

    try:
        result = run_benchmark(
            adapters=args.adapters,
            scenarios=args.scenarios,
            k=args.k,
            judge_model=args.judge_model,
            verbose=not args.quiet,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(format_report(result))

    if args.output:
        path = save_results(result, args.output)
        print(f"Results saved to: {path}")


if __name__ == "__main__":
    main()
