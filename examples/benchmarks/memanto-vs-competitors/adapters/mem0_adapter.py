"""Mem0 adapter — local or cloud memory system.

Requires ``MEM0_API_KEY`` for cloud, or runs local with pip install mem0ai.
"""

from __future__ import annotations

import re
import uuid

from adapters.base import (
    DialogueTurn,
    MemorySystem,
    RetrievedItem,
)


class Mem0Adapter(MemorySystem):
    """Mem0 memory system adapter."""

    def __init__(self, api_key: str = "", dry_mode: bool = False):
        self.api_key = api_key
        self.dry_mode = dry_mode
        self._memory = None
        self._user_ids: dict[str, str] = {}
        self._in_memory: dict[str, list[RetrievedItem]] = {}

    def name(self) -> str:
        return "mem0"

    def setup(self) -> None:
        import os

        api_key = self.api_key or os.environ.get("MEM0_API_KEY", "")
        if not api_key:
            print("    mem0: no API key, using in-memory mode")
            self.dry_mode = True
            return
        try:
            from mem0 import Memory

            os.environ.setdefault("OPENAI_API_KEY", api_key)
            config = {
                "vector_store": {"provider": "qdrant", "config": {"collection_name": "mem0_bench", "embedding_model_dims": 1536}},
                "version": "v1.1",
            }
            self._memory = Memory.from_config(config)
            print("    mem0: connected")
        except Exception as e:
            print(f"    mem0: init failed ({e}), using in-memory mode")
            self.dry_mode = True

    def store_turns(self, turns: list[DialogueTurn], namespace: str) -> None:
        if self.dry_mode or self._memory is None:
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

        if namespace not in self._user_ids:
            self._user_ids[namespace] = f"bench-{namespace}"
        user_id = self._user_ids[namespace]

        for turn in turns:
            msg = {"role": "user", "content": f"[{turn.speaker}] ({turn.session_id}): {turn.text}"}
            meta = {"dia_id": turn.dia_id, "session_id": turn.session_id}
            try:
                self._memory.add([msg], user_id=user_id, metadata=meta)
            except Exception as e:
                print(f"    mem0 add error: {e}")

    def search(self, query: str, namespace: str, k: int = 10) -> list[RetrievedItem]:
        if self.dry_mode or self._memory is None:
            items = self._in_memory.get(namespace, [])
            query_words = set(query.lower().split())
            scored = [(item, sum(1 for w in query_words if w in item.text.lower())) for item in items]
            scored.sort(key=lambda x: -x[1])
            return [item for item, _ in scored[:k]]

        user_id = self._user_ids.get(namespace)
        if not user_id:
            return []

        try:
            result = self._memory.search(query=query, filters={"user_id": user_id}, limit=k)
        except Exception:
            return []

        memories = result.get("results", []) if isinstance(result, dict) else (result or [])
        items: list[RetrievedItem] = []
        for mem in memories:
            text = mem.get("memory", "")
            meta = mem.get("metadata", {}) or {}
            dia_id = meta.get("dia_id", "")
            session_id = meta.get("session_id", "")
            if not dia_id:
                m = re.search(r"\[dia_id=([^\]]+)\]", text)
                if m:
                    dia_id = m.group(1)
            clean = re.sub(r"\[dia_id=[^\]]*\]\s*", "", text)
            clean = re.sub(r"\[\w+\]\s*", "", clean)
            items.append(
                RetrievedItem(
                    dia_id=dia_id,
                    session_id=session_id,
                    text=clean,
                    score=mem.get("score", 0.0),
                )
            )
        return items

    def teardown(self) -> None:
        for user_id in self._user_ids.values():
            if self._memory:
                try:
                    self._memory.delete_all(user_id=user_id)
                except Exception:
                    pass
        self._user_ids.clear()
        self._in_memory.clear()

    def cleanup(self) -> None:
        import shutil
        from pathlib import Path

        for p in [Path.home() / ".mem0"]:
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
