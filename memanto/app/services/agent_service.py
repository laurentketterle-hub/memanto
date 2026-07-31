"""
Agent Service for MEMANTO

Handles agent creation, listing, and lifecycle management.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from moorcheh_sdk.exceptions import ConflictError

from memanto.app.clients.moorcheh import get_moorcheh_client
from memanto.app.config import get_data_dir
from memanto.app.core import agent_namespace
from memanto.app.models.session import AgentCreate, AgentInfo, AgentList
from memanto.app.utils.errors import AgentAlreadyExistsError, AgentNotFoundError
from memanto.app.utils.temporal_helpers import as_utc_aware
from memanto.app.utils.validation import validate_safe_id


class AgentService:
    """Service for managing agents"""

    def __init__(self, agents_dir: Path | None = None):
        """
        Initialize agent service

        Args:
            agents_dir: Directory for agent metadata storage (defaults to ~/.memanto/agents/)
        """
        self.agents_dir = agents_dir or get_data_dir() / "agents"

    def _generate_namespace(self, agent_id: str) -> str:
        """
        Generate the Moorcheh namespace for an agent.

        Format: memanto_agent_{agent_id}
        """
        return agent_namespace(agent_id)

    def _get_agent_file(self, agent_id: str) -> Path:
        """Get file path for agent metadata"""
        validate_safe_id(agent_id, "agent_id")
        return self.agents_dir / f"{agent_id}.json"

    def create_agent(
        self, agent_create: AgentCreate, moorcheh_api_key: str
    ) -> AgentInfo:
        """
        Create a new agent

        Args:
            agent_create: Agent creation request
            moorcheh_api_key: Moorcheh API key for namespace creation

        Returns:
            AgentInfo object

        Raises:
            AgentAlreadyExistsError: If agent already exists
        """
        agent_file = self._get_agent_file(agent_create.agent_id)
        if agent_file.exists():
            raise AgentAlreadyExistsError(
                f"Agent '{agent_create.agent_id}' already exists"
            )

        namespace = self._generate_namespace(agent_create.agent_id)

        # Create namespace in Moorcheh - CRITICAL: Must succeed.
        # ``moorcheh_api_key`` is honored on cloud; ignored on on-prem.
        client = get_moorcheh_client()

        try:
            # Use Moorcheh SDK to create namespace with type="text"
            client.namespaces.create(namespace, type="text")
            print(f"[OK] Namespace created in Moorcheh: {namespace}")
        except ConflictError:
            # Namespace already exists - this is OK, agent might have been created before
            print(f"[OK] Namespace already exists in Moorcheh: {namespace}")
        except Exception as e:
            # On-prem raises moorcheh.errors.MoorchehApiError (HTTP 409) rather
            # than the cloud SDK's typed ConflictError when the namespace
            # already exists. Match on message so both backends behave the same.
            msg = str(e).lower()
            if ("namespace" in msg and "already exists" in msg) or "conflict" in msg:
                print(f"[OK] Namespace already exists in Moorcheh: {namespace}")
            else:
                raise Exception(
                    f"Failed to create namespace '{namespace}' in Moorcheh: {str(e)}"
                )

        # Create agent metadata
        agent = AgentInfo(
            agent_id=agent_create.agent_id,
            namespace=namespace,
            pattern=agent_create.pattern,
            description=agent_create.description,
            created_at=datetime.now(timezone.utc),
            memory_count=0,
            session_count=0,
            status="ready",
        )

        # Save agent metadata
        self._save_agent(agent)

        return agent

    def get_agent(self, agent_id: str) -> AgentInfo | None:
        """
        Get agent by ID

        Args:
            agent_id: Agent identifier

        Returns:
            AgentInfo or None if not found
        """
        agent_file = self._get_agent_file(agent_id)
        if not agent_file.exists():
            return None

        with open(agent_file) as f:
            data = json.load(f)
            return AgentInfo(**data)

    def list_agents(self) -> AgentList:
        """
        List all agents

        Returns:
            AgentList with all agents
        """
        agents: list[AgentInfo] = []
        if not self.agents_dir.exists():
            return AgentList(agents=agents, count=0)

        for agent_file in self.agents_dir.glob("*.json"):
            with open(agent_file) as f:
                data = json.load(f)
                agents.append(AgentInfo(**data))

        # Sort by created_at (newest first); normalize for legacy naive timestamps.
        agents.sort(key=lambda a: as_utc_aware(a.created_at), reverse=True)

        return AgentList(agents=agents, count=len(agents))

    def update_agent_stats(
        self,
        agent_id: str,
        last_session: datetime | None = None,
        increment_session_count: bool = False,
    ) -> AgentInfo:
        """
        Update agent statistics

        Args:
            agent_id: Agent identifier
            last_session: Last session timestamp
            increment_session_count: Whether to increment session count

        Returns:
            Updated AgentInfo

        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        agent = self.get_agent(agent_id)
        if not agent:
            raise AgentNotFoundError(f"Agent '{agent_id}' not found")

        if last_session:
            agent.last_session = last_session

        if increment_session_count:
            agent.session_count += 1

        self._save_agent(agent)
        return agent

    def delete_agent(self, agent_id: str, delete_namespace: bool = False) -> None:
        """
        Delete agent and all associated artifacts.

        Cleans up:
        - Agent metadata file (~/.memanto/agents/{agent_id}.json)
        - Session files (~/.memanto/sessions/{agent_id}.json and
          ~/.memanto/sessions/{agent_id}_* summary/state files)
        - Conflict report files (~/.memanto/conflicts/{agent_id}_*)
        - Active session marker if this agent was active

        Args:
            agent_id: Agent identifier
            delete_namespace: If True, also delete the Moorcheh namespace
                (best-effort; failures are silently ignored).

        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        agent_file = self._get_agent_file(agent_id)
        if not agent_file.exists():
            raise AgentNotFoundError(f"Agent '{agent_id}' not found")

        # 1. Delete the agent metadata file
        agent_file.unlink()

        # 2. Clean up session files in ~/.memanto/sessions/
        data_dir = get_data_dir()
        sessions_dir = data_dir / "sessions"
        if sessions_dir.exists():
            # Delete main session file ({agent_id}.json)
            session_file = sessions_dir / f"{agent_id}.json"
            if session_file.exists():
                session_file.unlink()

            # Delete summary/state files matching the agent_id pattern
            # (e.g. {agent_id}_{date}_{session_id}_summary.md)
            for f in sessions_dir.glob(f"{agent_id}_*"):
                try:
                    f.unlink()
                except OSError:
                    pass

            # Clear the active-session marker if it points to this agent
            active_marker = sessions_dir / "active"
            if active_marker.exists():
                try:
                    if active_marker.is_symlink():
                        if active_marker.readlink().stem == agent_id:
                            active_marker.unlink()
                    else:
                        content = active_marker.read_text().strip()
                        if content == agent_id:
                            active_marker.unlink()
                except OSError:
                    pass

        # 3. Clean up conflict report files in ~/.memanto/conflicts/
        conflicts_dir = data_dir / "conflicts"
        if conflicts_dir.exists():
            for f in conflicts_dir.glob(f"{agent_id}_*"):
                try:
                    f.unlink()
                except OSError:
                    pass

        # 4. Optionally delete the Moorcheh namespace (best-effort)
        if delete_namespace:
            namespace = self._generate_namespace(agent_id)
            try:
                client = get_moorcheh_client()
                client.namespaces.delete(namespace_name=namespace)
            except Exception:
                pass

    def agent_exists(self, agent_id: str) -> bool:
        """
        Check if agent exists

        Args:
            agent_id: Agent identifier

        Returns:
            True if agent exists
        """
        return self._get_agent_file(agent_id).exists()

    def _save_agent(self, agent: AgentInfo) -> None:
        """Save agent metadata to file"""
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        agent_file = self._get_agent_file(agent.agent_id)
        with open(agent_file, "w") as f:
            json.dump(agent.model_dump(mode="json"), f, indent=2)
