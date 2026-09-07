"""Loads agents.yaml, validates it, and embeds each capability for retrieval.

Owns the in-memory capability registry: the set of envelope ids that legally
exist. The planner's output is validated against this object, so a malformed or
ambiguous registry has to fail loudly at startup rather than at request time.

It does not build agents (that is app.agents.composer and app.agents.factory)
and it does not know which MCP tools exist (that is app.mcp.registry).
"""

from pathlib import Path

import yaml

from app.agents.models import CapabilitySpec
from app.embeddings import cosine_similarity, embed_text


class CapabilityRegistryError(ValueError):
    """Raised when agents.yaml is missing, malformed, or internally inconsistent."""


class CapabilityRegistry:
    """The loaded capability envelopes, plus similarity search over them."""

    def __init__(self, specs: list[CapabilitySpec]) -> None:
        self._specs: dict[str, CapabilitySpec] = {spec.id: spec for spec in specs}
        self._vectors: dict[str, list[float]] = {
            spec.id: embed_text(f"{spec.id} {spec.description}") for spec in specs
        }

    def __len__(self) -> int:
        return len(self._specs)

    @property
    def capability_ids(self) -> set[str]:
        """The only capability ids a plan may legally name."""
        return set(self._specs)

    def all_specs(self) -> list[CapabilitySpec]:
        return list(self._specs.values())

    def get(self, capability_id: str) -> CapabilitySpec:
        """Look up one envelope, or raise naming what was actually available."""
        try:
            return self._specs[capability_id]
        except KeyError:
            raise CapabilityRegistryError(
                f"Unknown capability id {capability_id!r}. Registered capabilities: "
                f"{', '.join(sorted(self._specs))}"
            ) from None

    def search_by_capability(self, goal: str, limit: int) -> list[CapabilitySpec]:
        """Return the envelopes most similar to the goal, best first.

        This is the menu the planner designs agents against. Envelopes scoring
        at or below zero similarity are dropped rather than padded in -- an
        envelope with nothing to do with the goal invites the planner to use it.
        """
        goal_vector = embed_text(goal)
        scored = [
            (cosine_similarity(goal_vector, self._vectors[spec_id]), spec_id)
            for spec_id in self._specs
        ]
        ranked = sorted(scored, key=lambda pair: pair[0], reverse=True)
        selected = [self._specs[spec_id] for score, spec_id in ranked[:limit] if score > 0.0]
        # A goal with no lexical overlap at all must still get a menu, or the
        # planner has nothing to plan with.
        return selected or [self._specs[spec_id] for _, spec_id in ranked[:limit]]


def load_capability_registry(path: Path) -> CapabilityRegistry:
    """Read and validate agents.yaml into a CapabilityRegistry.

    Every failure mode here names the file and the offending record, because
    this runs at startup and the message is the only debugging the operator gets.
    """
    if not path.exists():
        raise CapabilityRegistryError(f"Capability registry not found at {path}")

    try:
        document = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise CapabilityRegistryError(f"{path} is not valid YAML: {error}") from error

    if not isinstance(document, dict) or "capabilities" not in document:
        raise CapabilityRegistryError(
            f"{path} must be a mapping with a top-level 'capabilities' key"
        )

    records = document["capabilities"]
    if not isinstance(records, list) or not records:
        raise CapabilityRegistryError(f"{path}: 'capabilities' must be a non-empty list")

    specs: list[CapabilitySpec] = []
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise CapabilityRegistryError(
                f"{path}: capability at position {position} is not a mapping"
            )
        try:
            specs.append(CapabilitySpec.model_validate(record))
        except ValueError as error:
            label = record.get("id", f"position {position}")
            raise CapabilityRegistryError(
                f"{path}: capability {label!r} is invalid: {error}"
            ) from error

    duplicates = sorted({spec.id for spec in specs if [s.id for s in specs].count(spec.id) > 1})
    if duplicates:
        raise CapabilityRegistryError(
            f"{path}: duplicate capability ids {duplicates}. Ids address envelopes in a "
            "plan, so they have to be unique."
        )

    return CapabilityRegistry(specs)
