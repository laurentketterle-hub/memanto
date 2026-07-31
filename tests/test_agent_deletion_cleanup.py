"""
Tests for agent deletion cleanup.

Verifies that `AgentService.delete_agent()` removes all artifacts:
- Agent metadata JSON
- Session files ({agent_id}.json and {agent_id}_* summary files)
- Conflict report files ({agent_id}_*)
- Active session marker

See: https://github.com/moorcheh-ai/memanto/issues/1741
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from memanto.app.services.agent_service import AgentService
from memanto.app.utils.errors import AgentNotFoundError


class TestAgentDeletionCleanup:
    """Ensure delete_agent() leaves no agent artifacts behind."""

    TEST_AGENT_ID = "test-del-cleanup-agent"

    @pytest.fixture(autouse=True)
    def setup_temp_env(self, monkeypatch, tmp_path):
        """Route all memanto I/O into a fresh temp directory."""
        # Use a dedicated subdir under tmp_path as the pretend home
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        # Patch get_data_dir so all paths are rooted under our temp home
        data_dir = fake_home / ".memanto"
        data_dir.mkdir(parents=True)

        with patch(
            "memanto.app.services.agent_service.get_data_dir",
            return_value=data_dir,
        ):
            # Create the agent service pointing at our temp dir
            service = AgentService(agents_dir=data_dir / "agents")
            service.agents_dir.mkdir(parents=True, exist_ok=True)

            # Pre-create the directories where artifacts will live
            self.sessions_dir = data_dir / "sessions"
            self.conflicts_dir = data_dir / "conflicts"
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
            self.conflicts_dir.mkdir(parents=True, exist_ok=True)

            self.service = service
            self.data_dir = data_dir
            yield

    # ── helpers ──────────────────────────────────────────────────────────

    def _create_agent_metadata(self, agent_id: str | None = None):
        """Write a minimal agent .json so the agent 'exists'."""
        agent_id = agent_id or self.TEST_AGENT_ID
        agent_data = {
            "agent_id": agent_id,
            "namespace": f"memanto_agent_{agent_id}",
            "pattern": "support",
            "description": "Test agent",
            "created_at": "2026-07-31T12:00:00Z",
            "memory_count": 5,
            "session_count": 2,
            "status": "ready",
        }
        agent_file = self.service.agents_dir / f"{agent_id}.json"
        agent_file.write_text(json.dumps(agent_data))
        return agent_file

    def _create_session_file(self, agent_id: str | None = None):
        """Create the main {agent_id}.json session file."""
        agent_id = agent_id or self.TEST_AGENT_ID
        p = self.sessions_dir / f"{agent_id}.json"
        p.write_text(json.dumps({"agent_id": agent_id, "status": "active"}))
        return p

    def _create_summary_file(self, agent_id: str | None = None):
        """Create a {agent_id}_{date}_{sid}_summary.md in sessions/."""
        agent_id = agent_id or self.TEST_AGENT_ID
        p = self.sessions_dir / f"{agent_id}_2026-07-31_sess_abc123_summary.md"
        p.write_text("# Summary\n\nTest content.")
        return p

    def _create_conflict_file(self, agent_id: str | None = None):
        """Create a {agent_id}_{date}_conflicts.json in conflicts/."""
        agent_id = agent_id or self.TEST_AGENT_ID
        p = self.conflicts_dir / f"{agent_id}_2026-07-31_conflicts.json"
        p.write_text("[]")
        return p

    def _set_active_session(self, agent_id: str | None = None):
        """Write the active marker pointing at our agent."""
        agent_id = agent_id or self.TEST_AGENT_ID
        active = self.sessions_dir / "active"
        active.write_text(agent_id)
        return active

    # ── tests ────────────────────────────────────────────────────────────

    def test_delete_agent_cleans_metadata_file(self):
        """The agent .json must be removed."""
        meta = self._create_agent_metadata()
        assert meta.exists()

        self.service.delete_agent(self.TEST_AGENT_ID)

        assert not meta.exists()

    def test_delete_agent_cleans_main_session_file(self):
        """{agent_id}.json in sessions/ must be removed."""
        meta = self._create_agent_metadata()
        sess = self._create_session_file()
        assert sess.exists()

        self.service.delete_agent(self.TEST_AGENT_ID)

        assert not meta.exists()
        assert not sess.exists()

    def test_delete_agent_cleans_summary_files(self):
        """Wildcard summary files ({agent_id}_*) must be removed."""
        meta = self._create_agent_metadata()
        summ = self._create_summary_file()
        assert summ.exists()

        self.service.delete_agent(self.TEST_AGENT_ID)

        assert not summ.exists()

    def test_delete_agent_cleans_conflict_files(self):
        """Conflict report files must be removed."""
        meta = self._create_agent_metadata()
        conf = self._create_conflict_file()
        assert conf.exists()

        self.service.delete_agent(self.TEST_AGENT_ID)

        assert not conf.exists()

    def test_delete_agent_clears_active_marker(self):
        """If the deleted agent was active, the marker must be cleared."""
        meta = self._create_agent_metadata()
        self._create_session_file()
        active = self._set_active_session()
        assert active.exists()
        assert active.read_text().strip() == self.TEST_AGENT_ID

        self.service.delete_agent(self.TEST_AGENT_ID)

        assert not active.exists()

    def test_delete_agent_preserves_active_marker_for_other_agent(self):
        """Active marker must survive when a different agent is deleted."""
        other_id = "other-agent"
        # Create both agents
        self._create_agent_metadata(self.TEST_AGENT_ID)
        self._create_agent_metadata(other_id)
        self._create_session_file(other_id)
        active = self._set_active_session(other_id)

        self.service.delete_agent(self.TEST_AGENT_ID)

        assert active.exists()
        assert active.read_text().strip() == other_id

    def test_delete_agent_removes_all_artifacts(self):
        """Integration: all artifact types must be gone after deletion."""
        meta = self._create_agent_metadata()
        sess = self._create_session_file()
        summ = self._create_summary_file()
        conf = self._create_conflict_file()
        active = self._set_active_session()

        # Sanity: everything exists before deletion
        assert meta.exists()
        assert sess.exists()
        assert summ.exists()
        assert conf.exists()
        assert active.exists()

        self.service.delete_agent(self.TEST_AGENT_ID)

        # Assert all gone
        assert not meta.exists()
        assert not sess.exists()
        assert not summ.exists()
        assert not conf.exists()
        assert not active.exists()

    def test_delete_nonexistent_agent_raises(self):
        """Deleting a non-existent agent must raise AgentNotFoundError."""
        with pytest.raises(AgentNotFoundError):
            self.service.delete_agent("nonexistent-agent-id")

    def test_delete_agent_survives_missing_optional_dirs(self):
        """Gracefully handle missing sessions/conflicts dirs."""
        meta = self._create_agent_metadata()

        # Remove the optional directories
        import shutil

        shutil.rmtree(self.sessions_dir)
        shutil.rmtree(self.conflicts_dir)

        # Must not crash
        self.service.delete_agent(self.TEST_AGENT_ID)
        assert not meta.exists()

    def test_delete_agent_with_namespace(self):
        """When delete_namespace=True, the Moorcheh namespace is deleted."""
        meta = self._create_agent_metadata()

        with patch(
            "memanto.app.services.agent_service.get_moorcheh_client"
        ) as mock_get:
            mock_client = mock_get.return_value
            self.service.delete_agent(
                self.TEST_AGENT_ID, delete_namespace=True
            )

            mock_client.namespaces.delete.assert_called_once_with(
                namespace_name=f"memanto_agent_{self.TEST_AGENT_ID}"
            )

    def test_delete_agent_namespace_failure_is_silent(self):
        """Namespace deletion failures must not propagate (best-effort)."""
        meta = self._create_agent_metadata()

        with patch(
            "memanto.app.services.agent_service.get_moorcheh_client"
        ) as mock_get:
            mock_client = mock_get.return_value
            mock_client.namespaces.delete.side_effect = ValueError("boom")

            # Must not raise
            self.service.delete_agent(
                self.TEST_AGENT_ID, delete_namespace=True
            )

        assert not meta.exists()
