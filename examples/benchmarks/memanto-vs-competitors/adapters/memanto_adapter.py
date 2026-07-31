"""Memanto adapter — connects to the Moorcheh memory backend.

Requires ``MOORCHEH_API_KEY`` in the environment for live mode.
For dry-run/testing, falls back to an in-memory store.
"""

from __future__ import annotations

import os
import time
import uuid

from adapters.base import (
    DialogueTurn,
    MemorySystem,
    RetrievedItem,
)


class MemantoAdapter(MemorySystem):
    """Memanto memory system adapter (Moorcheh backend)."""

    def __init__(
        self,
        api_key: str = "",
        pacing: float = 0.01,
        max_retries: int = 3,
        dry_mode: bool = False,
    ):
        self.api_key = api_key
        self.pacing = pacing
        self.max_retries = max_retries
        self.dry_mode = dry_mode
        self._client = None
        self._agent_ids: dict[str, str] = {}
        self._in_memory: dict[str, list[RetrievedItem]] = {}

    def name(self) -> str:
        return "memanto"

    def setup(self) -> None:
        api_key = self.api_key or os.environ.get("MOORCHEH_API_KEY", "")
        if not api_key:
            print("    memanto: no API key, using in-memory mode")
            self.dry_mode = True
            return
        try:
            from memanto.cli.client.sdk_client import SdkClient
            self._client = SdkClient(api_key=api_key)
            print("    memanto: connected to Moorcheh backend")
        except Exception as e:
            print(f"    memanto: SDK init failed ({e}), using in-memory mode")
            self.dry_mode = True

    def store_turns(self, turns: list[DialogueTurn], namespace: str) -> None:
        if self.dry_mode or self._client is None:
            # In-memory fallback
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

        # Live mode: create agent and store via SDK
        if namespace not in self._agent_ids:
            agent_id = f"bench-{namespace}-{uuid.uuid4().hex[:6]}"
            self._client.create_agent(agent_id)
            self._client.activate_agent(agent_id)
            self._agent_ids[namespace] = agent_id

        agent_id = self._agent_ids[namespace]
        for turn in turns:
            for attempt in range(self.max_retries + 1):
                try:
                    self._client.remember(
                        agent_id=agent_id,
                        memory_type="fact",
                        title=f"Turn {turn.dia_id}",
                        content=turn.text,
                        tags=[turn.session_id, turn.dia_id, turn.speaker],
                    )
                    time.sleep(self.pacing)
                    break
                except Exception as e:
                    if attempt < self.max_retries:
                        time.sleep(1)
                    else:
                        print(f"    memanto remember error ({turn.dia_id}): {e}")

    def search(
        self, query: str, namespace: str, k: int = 10
    ) -> list[RetrievedItem]:
        if self.dry_mode or self._client is None:
            items = self._in_memory.get(namespace, [])
            # Simple keyword-based retrieval fallback
            query_words = set(query.lower().split())
            scored = [
                (item, sum(1 for w in query_words if w in item.text.lower()))
                for item in items
            ]
            scored.sort(key=lambda x: -x[1])
            return [item for item, _ in scored[:k]]

        agent_id = self._agent_ids.get(namespace)
        if not agent_id:
            return []

        try:
            self._client.activate_agent(agent_id)
        except Exception:
            pass

        try:
            result = self._client.recall(agent_id, query, limit=k)
        except Exception:
            return []

        items: list[RetrievedItem] = []
        if result is None:
            return items
        for mem in result.get("memories", [])[:k]:
            tags = mem.get("tags", []) or []
            dia_id = tags[1] if len(tags) > 1 else ""
            session_id = tags[0] if tags else ""
            items.append(
                RetrievedItem(
                    dia_id=dia_id,
                    session_id=session_id,
                    text=mem.get("content", mem.get("memory", "")),
                    score=mem.get("score", mem.get("similarity", 0.0)),
                )
            )
        return items

    def teardown(self) -> None:
        self._agent_ids.clear()
        self._in_memory.clear()

    def cleanup(self) -> None:
        import shutil
        from pathlib import Path

        # Delete local sessions
        sessions = Path.home() / ".memanto" / "sessions"
        if sessions.exists():
            shutil.rmtree(sessions, ignore_errors=True)

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
