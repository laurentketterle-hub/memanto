"""Letta adapter — stateful agent platform with advanced memory.

Requires ``letta_client`` and a running Letta server.
Default: localhost:8283
"""

from __future__ import annotations

import re
import uuid

from adapters.base import (
    DialogueTurn,
    MemorySystem,
    RetrievedItem,
)


class LettaAdapter(MemorySystem):
    """Letta memory system adapter."""

    def __init__(self, base_url: str = "http://localhost:8283", dry_mode: bool = False):
        self.base_url = base_url
        self.dry_mode = dry_mode
        self._client = None
        self._agent_ids: dict[str, str] = {}
        self._in_memory: dict[str, list[RetrievedItem]] = {}

    def name(self) -> str:
        return "letta"

    def setup(self) -> None:
        try:
            from letta_client import Letta

            self._client = Letta(
                base_url=self.base_url,
                environment="local",
                api_key="",
            )
            # Quick health check
            try:
                self._client.agents.list(limit=1)
                print(f"    letta: connected to {self.base_url}")
            except Exception:
                print(f"    letta: server at {self.base_url} not responding, using in-memory mode")
                self.dry_mode = True
                self._client = None
        except ImportError:
            print("    letta: letta_client not installed, using in-memory mode")
            self.dry_mode = True

    def store_turns(self, turns: list[DialogueTurn], namespace: str) -> None:
        if self.dry_mode or self._client is None:
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
            self._in_memory[namespace] = items
            return

        if namespace not in self._agent_ids:
            agent_id = f"bench-{namespace}-{uuid.uuid4().hex[:8]}"
            self._client.agents.create(
                name=agent_id,
                memory_blocks=[{
                    "label": "human",
                    "value": f"Benchmark user {namespace}",
                    "limit": 2000,
                }],
            )
            self._agent_ids[namespace] = agent_id

        agent_id = self._agent_ids[namespace]
        for turn in turns:
            try:
                self._client.archives.create(
                    agent_id=agent_id,
                    content=f"[dia_id={turn.dia_id}] [{turn.speaker}] {turn.text}",
                )
            except Exception as e:
                print(f"    letta archive error: {e}")

    def search(self, query: str, namespace: str, k: int = 10) -> list[RetrievedItem]:
        if self.dry_mode or self._client is None:
            items = self._in_memory.get(namespace, [])
            query_words = set(query.lower().split())
            scored = [(item, sum(1 for w in query_words if w in item.text.lower())) for item in items]
            scored.sort(key=lambda x: -x[1])
            return [item for item, _ in scored[:k]]

        agent_id = self._agent_ids.get(namespace)
        if not agent_id:
            return []

        try:
            results = self._client.passages.search(query=query, agent_id=agent_id)
        except Exception:
            return []

        items = []
        for p in (results or [])[:k]:
            text = getattr(p, "content", str(p))
            dia_match = re.search(r"\[dia_id=([^\]]+)\]", text)
            dia_id = dia_match.group(1) if dia_match else ""
            clean = re.sub(r"\[dia_id=[^\]]*\]\s*", "", text)
            items.append(
                RetrievedItem(
                    dia_id=dia_id,
                    session_id="",
                    text=clean,
                    score=getattr(p, "score", 0.0),
                )
            )
        return items

    def teardown(self) -> None:
        if self._client:
            for agent_id in self._agent_ids.values():
                try:
                    self._client.agents.delete(id=agent_id)
                except Exception:
                    pass
        self._agent_ids.clear()
        self._in_memory.clear()

    def cleanup(self) -> None:
        import shutil
        from pathlib import Path

        for p in [Path.home() / ".letta"]:
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)

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
