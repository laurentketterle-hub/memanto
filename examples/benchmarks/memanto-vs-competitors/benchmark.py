"""Benchmark orchestrator — runs all adapters through all scenarios.

Drives each memory system through synthetic scenarios, collects metrics,
and produces a comparative report.

Usage:
    python benchmark.py
    python benchmark.py --adapters memanto mem0
    python benchmark.py --scenarios basic_recall temporal
    python benchmark.py --k 10 --output results.json
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone
from typing import Any

from adapters import ADAPTER_REGISTRY
from adapters.base import (
    BenchmarkResult,
    MemorySystem,
    Scenario,
    SystemScore,
    evaluate_query,
)
from datasets import ALL_SCENARIOS
from evaluator import LLMJudge


def run_single_system(
    adapter: MemorySystem,
    scenario: Scenario,
    k: int = 10,
    judge: LLMJudge | None = None,
    verbose: bool = True,
) -> SystemScore:
    """Run one memory system through a full scenario."""

    system_name = adapter.name()
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  {system_name} on {scenario.name}")
        print(f"{'=' * 60}")

    try:
        adapter.setup()
    except Exception as e:
        print(f"  [{system_name}] setup failed: {e}")
        return SystemScore(
            system_name=system_name,
            total_qas=0,
            avg_recall=0.0,
            avg_precision=0.0,
            accuracy_score=0,
            staleness_score=0,
            precision_score=0,
            max_possible=scenario.total_qa_pairs * 15,
            total_tokens=0,
            total_latency_s=0.0,
            p50_latency_s=0.0,
            p95_latency_s=0.0,
        )

    all_latencies: list[float] = []
    total_tokens = 0
    total_accuracy = 0
    total_staleness = 0
    total_precision = 0
    total_recall = 0.0
    total_prec = 0.0
    qa_count = 0

    try:
        for conv in scenario.conversations:
            ns = f"{scenario.name}-{conv.sample_id}"

            # Store all turns
            if verbose:
                print(f"  [{system_name}] Storing {len(conv.turns)} turns for {ns}...")
            adapter.store_turns(conv.turns, ns)

            # Brief wait for indexing
            time.sleep(0.3)

            # Query all QA pairs
            if verbose:
                print(f"  [{system_name}] Running {len(conv.qa_pairs)} queries...")
            for qa in conv.qa_pairs:
                result = evaluate_query(adapter, qa, ns, k=k)
                all_latencies.append(result.latency_seconds)
                total_tokens += result.token_count
                total_recall += result.recall_at_k
                total_prec += result.precision_at_k
                qa_count += 1

                # Judge scoring
                if judge:
                    recalled_text = " ".join(r.text for r in result.retrieved[:3])
                    stale = [r.text for r in result.retrieved if r.dia_id not in result.evidence_turn_ids]
                    current = [qa.answer]
                    score = judge.score(
                        system_name=system_name,
                        query_id=qa.question_id,
                        query=qa.question,
                        golden_answer=qa.answer,
                        stale_signals=stale[:3],
                        current_signals=current,
                        recalled_answer=recalled_text,
                    )
                    total_accuracy += score.accuracy
                    total_staleness += score.staleness_avoidance
                    total_precision += score.precision

                if verbose:
                    r = result.recall_at_k
                    print(f"    [{system_name}] {qa.question_id}: recall={r:.2f}, "
                          f"latency={result.latency_seconds:.3f}s")

    finally:
        if verbose:
            print(f"  [{system_name}] Tearing down...")
        try:
            adapter.teardown()
        except Exception:
            pass

    # Compute aggregated metrics
    latencies_sorted = sorted(all_latencies) if all_latencies else [0.0]
    p50 = statistics.median(latencies_sorted) if latencies_sorted else 0.0
    p95_idx = max(0, int(len(latencies_sorted) * 0.95) - 1)
    p95 = latencies_sorted[p95_idx] if latencies_sorted else 0.0

    avg_recall = total_recall / max(qa_count, 1)
    avg_precision = total_prec / max(qa_count, 1)

    return SystemScore(
        system_name=system_name,
        total_qas=qa_count,
        avg_recall=avg_recall,
        avg_precision=avg_precision,
        accuracy_score=total_accuracy,
        staleness_score=total_staleness,
        precision_score=total_precision,
        max_possible=qa_count * 15,
        total_tokens=total_tokens,
        total_latency_s=sum(all_latencies),
        p50_latency_s=p50,
        p95_latency_s=p95,
    )


def run_benchmark(
    adapters: list[str] | None = None,
    scenarios: list[str] | None = None,
    k: int = 10,
    judge_model: str | None = None,
    verbose: bool = True,
) -> BenchmarkResult:
    """Run the full benchmark across all specified adapters and scenarios."""

    # Resolve adapters
    if adapters is None:
        adapter_names = list(ADAPTER_REGISTRY.keys())
    else:
        adapter_names = [a.lower() for a in adapters if a.lower() in ADAPTER_REGISTRY]

    if not adapter_names:
        raise ValueError(f"No valid adapters. Available: {list(ADAPTER_REGISTRY.keys())}")

    # Resolve scenarios
    if scenarios is None:
        scenario_names = list(ALL_SCENARIOS.keys())
    else:
        scenario_names = [s for s in scenarios if s in ALL_SCENARIOS]

    if not scenario_names:
        raise ValueError(f"No valid scenarios. Available: {list(ALL_SCENARIOS.keys())}")

    # Initialize judge
    judge = LLMJudge(model=judge_model or "", api_key="") if judge_model else None
    if judge and verbose:
        print(f"\nUsing LLM judge: {judge.model}")

    # Run all scenarios
    print(f"\n{'#' * 70}")
    print(f"  MEMANTO VS COMPETITORS BENCHMARK")
    print(f"  Scenarios: {', '.join(scenario_names)}")
    print(f"  Systems: {', '.join(adapter_names)}")
    print(f"  k={k}")
    print(f"{'#' * 70}")

    all_scores: dict[str, SystemScore] = {}

    for aname in adapter_names:
        adapter_cls = ADAPTER_REGISTRY[aname]
        adapter = adapter_cls()

        # Aggregate across scenarios
        scenario_scores: list[SystemScore] = []
        for sname in scenario_names:
            scenario_fn = ALL_SCENARIOS[sname]
            scenario = scenario_fn()
            score = run_single_system(adapter, scenario, k=k, judge=judge, verbose=verbose)
            scenario_scores.append(score)

        # Merge scores across scenarios
        merged = SystemScore(
            system_name=aname,
            total_qas=sum(s.total_qas for s in scenario_scores),
            avg_recall=statistics.mean([s.avg_recall for s in scenario_scores]) if scenario_scores else 0,
            avg_precision=statistics.mean([s.avg_precision for s in scenario_scores]) if scenario_scores else 0,
            accuracy_score=sum(s.accuracy_score for s in scenario_scores),
            staleness_score=sum(s.staleness_score for s in scenario_scores),
            precision_score=sum(s.precision_score for s in scenario_scores),
            max_possible=sum(s.max_possible for s in scenario_scores),
            total_tokens=sum(s.total_tokens for s in scenario_scores),
            total_latency_s=sum(s.total_latency_s for s in scenario_scores),
            p50_latency_s=statistics.mean([s.p50_latency_s for s in scenario_scores]) if scenario_scores else 0,
            p95_latency_s=statistics.mean([s.p95_latency_s for s in scenario_scores]) if scenario_scores else 0,
        )
        all_scores[aname] = merged

    # Determine winner
    winner = max(all_scores, key=lambda k: all_scores[k].overall_pct)

    return BenchmarkResult(
        scenario_name="+".join(scenario_names),
        scenario_title=f"Memanto vs Competitors ({'+'.join(scenario_names)})",
        scores=all_scores,
        winner=winner,
        timestamp=datetime.now(timezone.utc).isoformat(),
        metadata={
            "adapters": adapter_names,
            "scenarios": scenario_names,
            "k": k,
            "judge_model": judge.model if judge else "keyword-fallback",
        },
    )


def format_report(result: BenchmarkResult) -> str:
    """Format a human-readable report from benchmark results."""

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"  BENCHMARK REPORT: {result.scenario_title}")
    lines.append(f"  Timestamp: {result.timestamp}")
    lines.append(f"  Winner: {result.winner.upper()}")
    lines.append("=" * 80)

    # Summary table
    lines.append("")
    lines.append(f"{'System':<12} {'QAs':>5} {'Recall':>8} {'Prec':>8} "
                 f"{'Acc':>5} {'Stale':>6} {'PrecJ':>6} {'Score':>7} {'%':>6} "
                 f"{'p50':>8} {'p95':>8}")
    lines.append("-" * 90)

    for name, score in sorted(result.scores.items()):
        total_judge = score.accuracy_score + score.staleness_score + score.precision_score
        lines.append(
            f"{name:<12} {score.total_qas:>5} {score.avg_recall:>7.3f} {score.avg_precision:>7.3f} "
            f"{score.accuracy_score:>5} {score.staleness_score:>6} {score.precision_score:>6} "
            f"{total_judge:>7} {score.overall_pct:>5.1f}% "
            f"{score.p50_latency_s:>7.3f}s {score.p95_latency_s:>7.3f}s"
        )

    lines.append("")
    lines.append(f"Winner: {result.winner}")
    lines.append(f"All systems evaluated on {result.scores[result.winner].total_qas} "
                 f"QA pairs across scenarios.")
    lines.append("")

    # Per-scenario detail
    lines.append("-" * 80)
    lines.append("  METHODOLOGY")
    lines.append("-" * 80)
    lines.append("")
    lines.append("Each memory system is evaluated on four scenarios:")
    lines.append("  1. Basic Recall — Simple fact storage and retrieval")
    lines.append("  2. Temporal Reasoning — Tracking preference changes over time")
    lines.append("  3. Contradiction Handling — Resolving conflicting information")
    lines.append("  4. Multi-Session Persistence — Cross-session memory recall")
    lines.append("")
    lines.append("Metrics:")
    lines.append("  - Recall@k: Fraction of evidence turns found in top-k results")
    lines.append("  - Precision@k: Fraction of retrieved items that are evidence")
    lines.append("  - Accuracy (0-5): Did the system recall correct information?")
    lines.append("  - Staleness (0-5): Did it avoid stale/outdated information?")
    lines.append("  - Precision Judge (0-5): Was the answer concise and on-topic?")
    lines.append("  - p50/p95 latency: Response time percentiles in seconds")
    lines.append("")

    return "\n".join(lines)


def save_results(result: BenchmarkResult, path: str) -> str:
    """Save benchmark results to JSON."""

    payload: dict[str, Any] = {
        "timestamp": result.timestamp,
        "scenario": result.scenario_name,
        "winner": result.winner,
        "systems": {},
        "metadata": result.metadata,
    }

    for name, score in result.scores.items():
        payload["systems"][name] = {
            "total_qas": score.total_qas,
            "avg_recall": score.avg_recall,
            "avg_precision": score.avg_precision,
            "accuracy_score": score.accuracy_score,
            "staleness_score": score.staleness_score,
            "precision_score": score.precision_score,
            "overall_pct": score.overall_pct,
            "total_tokens": score.total_tokens,
            "total_latency_s": score.total_latency_s,
            "p50_latency_s": score.p50_latency_s,
            "p95_latency_s": score.p95_latency_s,
        }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    return path


if __name__ == "__main__":
    result = run_benchmark()
    print(format_report(result))
