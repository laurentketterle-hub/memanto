"""Synthetic benchmark datasets for offline memory system testing.

Each scenario tests different memory capabilities:
- basic_recall: Simple fact retrieval
- temporal: Timeline-aware memory with updates
- contradiction: Handling conflicting information
- multi_session: Cross-session memory persistence
"""

from __future__ import annotations

from adapters.base import (
    Conversation,
    DialogueTurn,
    EvidenceSpan,
    QAPair,
    Scenario,
)


def create_basic_recall_scenario() -> Scenario:
    """Tests simple fact storage and retrieval across one conversation."""
    conversations = []
    for user_id in range(3):
        turns = [
            DialogueTurn(
                dia_id=f"B{user_id}_T1",
                speaker="user",
                text=f"Hi, my name is TestUser{user_id}.",
                session_id="session_1",
            ),
            DialogueTurn(
                dia_id=f"B{user_id}_T2",
                speaker="assistant",
                text=f"Hello TestUser{user_id}! How can I help you?",
                session_id="session_1",
            ),
            DialogueTurn(
                dia_id=f"B{user_id}_T3",
                speaker="user",
                text=f"My favorite color is {'blue' if user_id % 2 == 0 else 'green'}.",
                session_id="session_1",
            ),
            DialogueTurn(
                dia_id=f"B{user_id}_T4",
                speaker="user",
                text=f"I live in City{user_id}.",
                session_id="session_1",
            ),
            DialogueTurn(
                dia_id=f"B{user_id}_T5",
                speaker="assistant",
                text="Got it! I've noted your preferences.",
                session_id="session_1",
            ),
        ]

        qa_pairs = [
            QAPair(
                question_id=f"B{user_id}_Q1",
                question="What is my name?",
                answer=f"TestUser{user_id}",
                category="basic",
                evidence=[EvidenceSpan(session_id="session_1", turn_ids=[f"B{user_id}_T1"])],
            ),
            QAPair(
                question_id=f"B{user_id}_Q2",
                question="What is my favorite color?",
                answer="blue" if user_id % 2 == 0 else "green",
                category="basic",
                evidence=[EvidenceSpan(session_id="session_1", turn_ids=[f"B{user_id}_T3"])],
            ),
        ]

        conversations.append(
            Conversation(sample_id=f"user_{user_id}", turns=turns, qa_pairs=qa_pairs)
        )

    return Scenario(
        name="basic_recall",
        title="Basic Recall",
        description="Simple fact storage and retrieval — tests fundamental memory accuracy.",
        conversations=conversations,
    )


def create_temporal_scenario() -> Scenario:
    """Tests memory with time-based preference changes."""
    conversations = []
    for user_id in range(2):
        turns = [
            # Session 1: Initial state
            DialogueTurn(
                dia_id=f"T{user_id}_S1T1",
                speaker="user",
                text=f"I prefer Python for backend development.",
                session_id="session_1",
            ),
            DialogueTurn(
                dia_id=f"T{user_id}_S1T2",
                speaker="user",
                text=f"My team uses Django for web projects.",
                session_id="session_1",
            ),
            # Session 2: Preference shift
            DialogueTurn(
                dia_id=f"T{user_id}_S2T1",
                speaker="user",
                text=f"Actually, I've switched to Go for backend. Performance matters more now.",
                session_id="session_2",
            ),
            DialogueTurn(
                dia_id=f"T{user_id}_S2T2",
                speaker="user",
                text=f"We're migrating away from Django to a Go + HTMX stack.",
                session_id="session_2",
            ),
            # Session 3: Another update
            DialogueTurn(
                dia_id=f"T{user_id}_S3T1",
                speaker="user",
                text=f"Update: keeping Go for services but using Python for ML pipelines.",
                session_id="session_3",
            ),
        ]

        qa_pairs = [
            QAPair(
                question_id=f"T{user_id}_Q1",
                question="What language do I currently use for backend?",
                answer="Go",
                category="temporal",
                evidence=[EvidenceSpan(session_id="session_2", turn_ids=[f"T{user_id}_S2T1"])],
            ),
            QAPair(
                question_id=f"T{user_id}_Q2",
                question="What web framework does my team use?",
                answer="Go + HTMX stack",
                category="temporal",
                evidence=[EvidenceSpan(session_id="session_2", turn_ids=[f"T{user_id}_S2T2"])],
            ),
            QAPair(
                question_id=f"T{user_id}_Q3",
                question="What do I use Python for now?",
                answer="ML pipelines",
                category="temporal",
                evidence=[EvidenceSpan(session_id="session_3", turn_ids=[f"T{user_id}_S3T1"])],
            ),
        ]

        conversations.append(
            Conversation(sample_id=f"temporal_{user_id}", turns=turns, qa_pairs=qa_pairs)
        )

    return Scenario(
        name="temporal",
        title="Temporal Reasoning",
        description="Tests ability to track preference changes over time — critical for long-term memory.",
        conversations=conversations,
    )


def create_contradiction_scenario() -> Scenario:
    """Tests how the system handles directly contradictory information."""
    conversations = []
    for user_id in range(2):
        turns = [
            # Initial info
            DialogueTurn(
                dia_id=f"C{user_id}_T1",
                speaker="user",
                text=f"The project deadline is March 15th, 2026.",
                session_id="session_1",
            ),
            DialogueTurn(
                dia_id=f"C{user_id}_T2",
                speaker="user",
                text=f"The budget for this project is $50,000.",
                session_id="session_1",
            ),
            # Contradiction
            DialogueTurn(
                dia_id=f"C{user_id}_T3",
                speaker="user",
                text=f"Correction: the deadline has been moved to April 1st, 2026.",
                session_id="session_1",
            ),
            DialogueTurn(
                dia_id=f"C{user_id}_T4",
                speaker="user",
                text=f"The budget was increased to $75,000 after the latest review.",
                session_id="session_1",
            ),
            # More contradiction in later session
            DialogueTurn(
                dia_id=f"C{user_id}_T5",
                speaker="user",
                text=f"Final budget is now $60,000 — we had to cut some scope.",
                session_id="session_2",
            ),
        ]

        qa_pairs = [
            QAPair(
                question_id=f"C{user_id}_Q1",
                question="What is the current project deadline?",
                answer="April 1st, 2026",
                category="contradiction",
                evidence=[EvidenceSpan(session_id="session_1", turn_ids=[f"C{user_id}_T3"])],
            ),
            QAPair(
                question_id=f"C{user_id}_Q2",
                question="What is the final budget for the project?",
                answer="$60,000",
                category="contradiction",
                evidence=[EvidenceSpan(session_id="session_2", turn_ids=[f"C{user_id}_T5"])],
            ),
        ]

        conversations.append(
            Conversation(sample_id=f"contra_{user_id}", turns=turns, qa_pairs=qa_pairs)
        )

    return Scenario(
        name="contradiction",
        title="Contradiction Handling",
        description="Tests ability to resolve conflicting information and surface the most recent/correct data.",
        conversations=conversations,
    )


def create_multi_session_scenario() -> Scenario:
    """Tests cross-session memory persistence and retrieval."""
    conversations = []
    for user_id in range(2):
        turns = [
            # Session 1: Profile setup
            DialogueTurn(
                dia_id=f"M{user_id}_S1T1",
                speaker="user",
                text=f"I'm a senior data engineer working on ETL pipelines.",
                session_id="session_1",
            ),
            DialogueTurn(
                dia_id=f"M{user_id}_S1T2",
                speaker="user",
                text=f"My main tools are Apache Spark and Airflow.",
                session_id="session_1",
            ),
            # Session 2: Project context
            DialogueTurn(
                dia_id=f"M{user_id}_S2T1",
                speaker="user",
                text=f"For the new project, we need to process 10TB of logs daily.",
                session_id="session_2",
            ),
            DialogueTurn(
                dia_id=f"M{user_id}_S2T2",
                speaker="user",
                text=f"Latency requirements: under 5 minutes end-to-end.",
                session_id="session_2",
            ),
            # Session 3: Status update
            DialogueTurn(
                dia_id=f"M{user_id}_S3T1",
                speaker="user",
                text=f"We've hit the latency target — now at 4.2 minutes average.",
                session_id="session_3",
            ),
            DialogueTurn(
                dia_id=f"M{user_id}_S3T2",
                speaker="user",
                text=f"Next: adding real-time anomaly detection to the pipeline.",
                session_id="session_3",
            ),
        ]

        qa_pairs = [
            QAPair(
                question_id=f"M{user_id}_Q1",
                question="What is my role and what tools do I use?",
                answer="Senior data engineer using Apache Spark and Airflow",
                category="multi-session",
                evidence=[EvidenceSpan(session_id="session_1", turn_ids=[f"M{user_id}_S1T1", f"M{user_id}_S1T2"])],
            ),
            QAPair(
                question_id=f"M{user_id}_Q2",
                question="What are the latency requirements for the current project?",
                answer="Under 5 minutes end-to-end",
                category="multi-session",
                evidence=[EvidenceSpan(session_id="session_2", turn_ids=[f"M{user_id}_S2T2"])],
            ),
            QAPair(
                question_id=f"M{user_id}_Q3",
                question="What is the current average latency and what's the next step?",
                answer="4.2 minutes average; next step is real-time anomaly detection",
                category="multi-session",
                evidence=[EvidenceSpan(session_id="session_3", turn_ids=[f"M{user_id}_S3T1", f"M{user_id}_S3T2"])],
            ),
        ]

        conversations.append(
            Conversation(sample_id=f"multi_{user_id}", turns=turns, qa_pairs=qa_pairs)
        )

    return Scenario(
        name="multi_session",
        title="Multi-Session Persistence",
        description="Tests ability to recall information across multiple separate sessions — simulates long-running agent memory.",
        conversations=conversations,
    )


ALL_SCENARIOS: dict[str, callable] = {
    "basic_recall": create_basic_recall_scenario,
    "temporal": create_temporal_scenario,
    "contradiction": create_contradiction_scenario,
    "multi_session": create_multi_session_scenario,
}
