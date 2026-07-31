"""Zep adapter — Zep Cloud memory platform.

Requires ``ZEP_API_KEY`` in the environment.
"""

from __future__ import annotations

import json
import os
import uuid

from adapters.base import (
    DialogueTurn,
    MemorySystem,
    RetrievedItem,
)


class ZepAdapter(MemorySystem):
    """Zep Cloud memory system adapter."""

    def __init__(self, api_key: str = "", dry_mode: bool = False):
        self.api_key = api_key
        self.dry_mode = dry_mode
        self._client = None
        self._user_ids: dict[str, str] = {}
        self._graph_ids: dict[str, str] = {}
        self._in_memory: dict[str, list[RetrievedItem]] = {}

    def name(self) -> str:
        return "zep"

    def setup(self) -> None:
        api_key = self.api_key or os.environ.get("ZEP_API_KEY", "")
        if not api_key:
            print("    zep: no API key, using in-memory mode")
            self.dry_mode = True
            return
        try:
            from zep_cloud import Zep

            self._client = Zep(api_key=api_key)
            print("    zep: connected to Zep Cloud")
        except Exception as e:
            print(f"    zep: init failed ({e}), using in-memory mode")
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

        if namespace not in self._user_ids:
            uid = f"bench-{namespace}-{uuid.uuid4().hex[:8]}"
            gid = f"graph-{uid}"
            self._client.user.add(user_id=uid)
            self._client.graph.create(graph_id=gid)
            self._user_ids[namespace] = uid
            self._graph_ids[namespace] = gid

        uid = self._user_ids[namespace]

        # Store turns as graph episodes via batch
        from zep_cloud.types import BatchAddItem

        items = []
        for turn in turns:
            items.append(
                BatchAddItem(
                    user_id=uid,
                    type="graph_episode",
                    data_type="message",
                    data=json.dumps({
                        "role": turn.speaker,
                        "content": f"[dia_id={turn.dia_id}] {turn.text}",
                    }),
                )
            )

        try:
            batch = self._client.batch.create()
            self._client.batch.add(batch.batch_id, items=items)
            self._client.batch.process(batch.batch_id)
        except Exception as e:
            print(f"    zep batch error: {e}")

    def search(self, query: str, namespace: str, k: int = 10) -> list[RetrievedItem]:
        if self.dry_mode or self._client is None:
            import re

            items = self._in_memory.get(namespace, [])
            query_words = set(query.lower().split())
            scored = [(item, sum(1 for w in query_words if w in item.text.lower())) for item in items]
            scored.sort(key=lambda x: -x[1])
            results = [item for item, _ in scored[:k]]

            cleaned: list[RetrievedItem] = []
            seen = set()
            for r in results:
                dia = re.search(r"\[dia_id=([^\]]+)\]", r.text)
                dia_id = dia.group(1) if dia else r.dia_id
                if dia_id in seen:
                    continue
                seen.add(dia_id)
                cleaned.append(r)
            return cleaned[:k]

        import re

        uid = self._user_ids.get(namespace)
        if not uid:
            return []

        try:
            result = self._client.graph.search(
                user_id=uid, query=query, scope="episodes", limit=k
            )
        except Exception:
            return []

        items: list[RetrievedItem] = []
        seen: set[str] = set()
        for ep in (getattr(result, "episodes", None) or [])[:k]:
            text = ep.content or ""
            dia = re.search(r"\[dia_id=([^\]]+)\]", text)
            dia_id = dia.group(1) if dia else ep.uuid_ or ""
            if dia_id in seen:
                continue
            seen.add(dia_id)
            clean = re.sub(r"\[dia_id=[^\]]*\]\s*", "", text)
            items.append(
                RetrievedItem(
                    dia_id=dia_id,
                    session_id=uid,
                    text=clean,
                    score=ep.score or 0.0,
                )
            )
        return items[:k]

    def teardown(self) -> None:
        if self._client:
            for gid in self._graph_ids.values():
                try:
                    self._client.graph.delete(graph_id=gid)
                except Exception:
                    pass
            for uid in self._user_ids.values():
                try:
                    self._client.user.delete(uid)
                except Exception:
                    pass
        self._user_ids.clear()
        self._graph_ids.clear()
        self._in_memory.clear()

    def cleanup(self) -> None:
        import os as _os

        key = _os.environ.get("ZEP_API_KEY", "")
        if key and self._client:
            for g in self._client.graph.list_all() or []:
                gid = g.graph_id if hasattr(g, "graph_id") else str(g)
                if "bench-" in gid:
                    try:
                        self._client.graph.delete(graph_id=gid)
                    except Exception:
                        pass

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
