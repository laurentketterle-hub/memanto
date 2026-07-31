"""
Base adapter interface and dataclasses for memory system benchmark.

All adapters implement the MemorySystem protocol and return
standardized result types for fair comparison.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------


@dataclass
class DialogueTurn:
    """A single turn in a conversation."""

    dia_id: str
    speaker: str
    text: str
    session_id: str


@dataclass
class EvidenceSpan:
    """Ground-truth evidence linking a QA pair to source turns."""

    session_id: str
    turn_ids: list[str]


@dataclass
class QAPair:
    """A question-answer pair with ground-truth evidence and category."""

    question: str
    answer: str
    category: str
    evidence: list[EvidenceSpan]
    question_id: str = ""


@dataclass
class RetrievedItem:
    """An item returned by a memory system search."""

    dia_id: str
    session_id: str
    text: str
    score: float = 0.0


@dataclass
class QueryResult:
    """Result of evaluating one QA pair against a memory system."""

    query: str
    ground_truth_answer: str
    category: str
    question_id: str
    retrieved: list[RetrievedItem]
    latency_seconds: float
    token_count: int

    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    has_stale: bool = False
    stale_count: int = 0

    evidence_session_ids: set[str] = field(default_factory=set)
    evidence_turn_ids: set[str] = field(default_factory=set)

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """A full conversation: turns + QA pairs."""

    sample_id: str
    turns: list[DialogueTurn]
    qa_pairs: list[QAPair]

    @property
    def session_ids(self) -> list[str]:
        return sorted(
            {t.session_id for t in self.turns},
            key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 0,
        )


@dataclass
class Scenario:
    """A loaded benchmark scenario ready for evaluation."""

    name: str
    title: str
    description: str
    conversations: list[Conversation]

    @property
    def total_qa_pairs(self) -> int:
        return sum(len(c.qa_pairs) for c in self.conversations)

    @property
    def total_turns(self) -> int:
        return sum(len(c.turns) for c in self.conversations)


@dataclass
class SystemScore:
    """Aggregated scores for one memory system on the full benchmark."""

    system_name: str
    total_qas: int
    avg_recall: float
    avg_precision: float
    accuracy_score: int
    staleness_score: int
    precision_score: int
    max_possible: int
    total_tokens: int
    total_latency_s: float
    p50_latency_s: float
    p95_latency_s: float

    @property
    def overall_pct(self) -> float:
        return (self.accuracy_score / max(self.max_possible, 1)) * 100


@dataclass
class BenchmarkResult:
    """Full benchmark result with all systems compared."""

    scenario_name: str
    scenario_title: str
    scores: dict[str, SystemScore]
    winner: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MemorySystem Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MemorySystem(Protocol):
    """Interface every memory adapter must implement."""

    def name(self) -> str:
        """Human-readable system name (e.g. 'memanto', 'mem0')."""
        ...

    def setup(self) -> None:
        """Initialize connection, auth, and resources."""
        ...

    def store_turns(self, turns: list[DialogueTurn], namespace: str) -> None:
        """Ingest a sequence of dialogue turns into the memory system."""
        ...

    def search(
        self, query: str, namespace: str, k: int = 10
    ) -> list[RetrievedItem]:
        """Search for relevant memories given a query."""
        ...

    def teardown(self) -> None:
        """Clean up resources for one run."""
        ...

    def cleanup(self) -> None:
        """Delete ALL resources created by this adapter (across runs)."""
        ...

    def dry_run(self) -> bool:
        """End-to-end smoke test: setup -> store -> search -> teardown."""
        ...


# ---------------------------------------------------------------------------
# Scoring Utilities
# ---------------------------------------------------------------------------


def compute_scores(
    retrieved: list[RetrievedItem],
    qa: QAPair,
    k: int = 10,
) -> tuple[float, float, bool, int]:
    """Compute recall, precision, staleness from retrieved items vs evidence."""

    top_k = retrieved[:k]
    evidence_turn_ids: set[str] = set()
    for ev in qa.evidence:
        evidence_turn_ids.update(ev.turn_ids)

    retrieved_turn_ids: set[str] = {r.dia_id for r in top_k}

    if not evidence_turn_ids:
        evidence_session_ids = {ev.session_id for ev in qa.evidence}
        retrieved_session_ids = {r.session_id for r in top_k}
        if evidence_session_ids:
            recall = len(retrieved_session_ids & evidence_session_ids) / len(
                evidence_session_ids
            )
            precision = (
                len(retrieved_session_ids & evidence_session_ids)
                / len(retrieved_session_ids)
                if retrieved_session_ids
                else 0.0
            )
        else:
            recall = precision = 0.0
    else:
        recall = (
            len(retrieved_turn_ids & evidence_turn_ids) / len(evidence_turn_ids)
            if evidence_turn_ids
            else 0.0
        )
        precision = (
            len(retrieved_turn_ids & evidence_turn_ids) / len(retrieved_turn_ids)
            if retrieved_turn_ids
            else 0.0
        )

    has_stale = False
    stale_count = 0
    evidence_session_ids = {ev.session_id for ev in qa.evidence}
    if evidence_turn_ids:
        stale_count = sum(1 for r in top_k if r.dia_id not in evidence_turn_ids)
    else:
        stale_count = sum(
            1 for r in top_k if r.session_id not in evidence_session_ids
        )
    has_stale = stale_count > 0

    return recall, precision, has_stale, stale_count


def evaluate_query(
    memory: MemorySystem,
    qa: QAPair,
    namespace: str,
    k: int = 10,
) -> QueryResult:
    """Run one QA pair through a memory system."""
    start = time.perf_counter()
    retrieved = memory.search(qa.question, namespace, k=k)
    elapsed = time.perf_counter() - start

    recall, precision, has_stale, stale_count = compute_scores(retrieved, qa, k=k)

    evidence_turn_ids: set[str] = set()
    evidence_session_ids: set[str] = set()
    for ev in qa.evidence:
        evidence_turn_ids.update(ev.turn_ids)
        evidence_session_ids.add(ev.session_id)

    return QueryResult(
        query=qa.question,
        ground_truth_answer=qa.answer,
        category=qa.category,
        question_id=qa.question_id,
        retrieved=retrieved,
        latency_seconds=elapsed,
        token_count=sum(len(r.text.split()) for r in retrieved),
        recall_at_k=recall,
        precision_at_k=precision,
        has_stale=has_stale,
        stale_count=stale_count,
        evidence_session_ids=evidence_session_ids,
        evidence_turn_ids=evidence_turn_ids,
    )
