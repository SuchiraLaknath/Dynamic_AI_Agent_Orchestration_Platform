"""Loads agents.yaml, validates it, and embeds each role for planner retrieval.

Owns the in-memory agent registry: the set of agent ids that legally exist. The
planner's output is validated against this object, so a malformed or ambiguous
registry has to fail loudly at startup rather than at request time.

It does not build runnable agents (that is app.agents.factory) and it does not
know which MCP tools exist (that is app.mcp.registry).
"""

from pathlib import Path

import yaml

from app.agents.models import AgentSpec
from app.embeddings import cosine_similarity, embed_text


class AgentRegistryError(ValueError):
    """Raised when agents.yaml is missing, malformed, or internally inconsistent."""


class AgentRegistry:
    """The loaded agent specs, plus similarity search over their roles."""

    def __init__(self, specs: list[AgentSpec]) -> None:
        self._specs: dict[str, AgentSpec] = {spec.id: spec for spec in specs}
        self._role_vectors: dict[str, list[float]] = {
            spec.id: embed_text(f"{spec.id} {spec.role}") for spec in specs
        }

    def __len__(self) -> int:
        return len(self._specs)

    @property
    def agent_ids(self) -> set[str]:
        """The only agent ids a plan may legally name."""
        return set(self._specs)

    def all_specs(self) -> list[AgentSpec]:
        return list(self._specs.values())

    def get(self, agent_id: str) -> AgentSpec:
        """Look up one spec, or raise naming what was actually available."""
        try:
            return self._specs[agent_id]
        except KeyError:
            raise AgentRegistryError(
                f"Unknown agent id {agent_id!r}. Registered agents: "
                f"{', '.join(sorted(self._specs))}"
            ) from None

    def search_by_role(self, goal: str, limit: int) -> list[AgentSpec]:
        """Return the agents whose roles are most similar to the goal, best first.

        This is the menu the planner chooses from. Agents scoring at or below
        zero similarity are dropped rather than padded in -- offering the
        planner an agent that has nothing to do with the goal invites it to use
        one.
        """
        goal_vector = embed_text(goal)
        scored = [
            (cosine_similarity(goal_vector, self._role_vectors[spec_id]), spec_id)
            for spec_id in self._specs
        ]
        ranked = sorted(scored, key=lambda pair: pair[0], reverse=True)
        selected = [self._specs[spec_id] for score, spec_id in ranked[:limit] if score > 0.0]
        # A goal with no lexical overlap at all must still get a menu, or the
        # planner has nothing to plan with.
        return selected or [self._specs[spec_id] for _, spec_id in ranked[:limit]]


def load_agent_registry(path: Path) -> AgentRegistry:
    """Read and validate agents.yaml into an AgentRegistry.

    Every failure mode here names the file and the offending record, because
    this runs at startup and the message is the only debugging the operator gets.
    """
    if not path.exists():
        raise AgentRegistryError(f"Agent registry not found at {path}")

    try:
        document = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise AgentRegistryError(f"{path} is not valid YAML: {error}") from error

    if not isinstance(document, dict) or "agents" not in document:
        raise AgentRegistryError(f"{path} must be a mapping with a top-level 'agents' key")

    records = document["agents"]
    if not isinstance(records, list) or not records:
        raise AgentRegistryError(f"{path}: 'agents' must be a non-empty list")

    specs: list[AgentSpec] = []
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise AgentRegistryError(f"{path}: agent at position {position} is not a mapping")
        try:
            specs.append(AgentSpec.model_validate(record))
        except ValueError as error:
            label = record.get("id", f"position {position}")
            raise AgentRegistryError(f"{path}: agent {label!r} is invalid: {error}") from error

    duplicates = sorted({spec.id for spec in specs if [s.id for s in specs].count(spec.id) > 1})
    if duplicates:
        raise AgentRegistryError(
            f"{path}: duplicate agent ids {duplicates}. Ids address agents in a plan, "
            "so they have to be unique."
        )

    return AgentRegistry(specs)
