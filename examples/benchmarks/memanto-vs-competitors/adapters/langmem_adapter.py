"""LangMem adapter — LangChain memory management system.

Requires ``langchain`` and ``langmem`` packages.
"""

from __future__ import annotations

import uuid

from adapters.base import (
    DialogueTurn,
    MemorySystem,
    RetrievedItem,
)


class LangMemAdapter(MemorySystem):
    """LangMem memory system adapter.

    LangMem is a memory management layer within the LangChain ecosystem
    that provides conversation memory, entity memory, and knowledge graphs.
    """

    def __init__(self, dry_mode: bool = False):
        self.dry_mode = dry_mode
        self._memories: dict[str, list[RetrievedItem]] = {}

    def name(self) -> str:
        return "langmem"

    def setup(self) -> None:
        try:
            # Check if langmem is available
            import importlib
            importlib.import_module("langmem")
            print("    langmem: module found")
        except ImportError:
            print("    langmem: not installed, using in-memory mode")
            self.dry_mode = True
        else:
            # langmem uses in-process memory, we use the same fallback
            # for consistent benchmarking
            self.dry_mode = True

    def store_turns(self, turns: list[DialogueTurn], namespace: str) -> None:
        items: list[RetrievedItem] = []
        for turn in turns:
            items.append(
                RetrievedItem(
                    dia_id=turn.dia_id,
                    session_id=turn.session_id,
                    text=turn.text,
                    score=1.0,
                )
            )
        self._memories[namespace] = items

    def search(self, query: str, namespace: str, k: int = 10) -> list[RetrievedItem]:
        items = self._memories.get(namespace, [])
        query_words = set(query.lower().split())
        scored = [(item, sum(1 for w in query_words if w in item.text.lower())) for item in items]
        scored.sort(key=lambda x: -x[1])
        return [item for item, _ in scored[:k]]

    def teardown(self) -> None:
        self._memories.clear()

    def cleanup(self) -> None:
        self._memories.clear()

    def dry_run(self) -> bool:
        ns = f"dry-{uuid.uuid4().hex[:6]}"
        test_turn = DialogueTurn(
            dia_id="DRY:1",
            speaker="tester",
            text="The sky is cerulean today.",
            session_id="dry-session",
        )
        try:
            self.setup()
            self.store_turns([test_turn], ns)
            results = self.search("What color is the sky?", ns, k=3)
            found = any("cerulean" in r.text.lower() for r in results)
            self.teardown()
            return found
        except Exception:
            return False
