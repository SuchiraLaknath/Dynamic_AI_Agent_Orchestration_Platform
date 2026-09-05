"""The AgentSpec: the pydantic shape of one record in agents.yaml.

This is the contract between the YAML registry and everything that consumes it
-- the factory, the planner's menu, and the /agents endpoint. It validates
shape and internal consistency only; it does not know whether the tools an
agent selects actually exist, because tool discovery happens later, at startup.
"""

import re
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

AGENT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class AgentSpec(BaseModel):
    """One specialist agent, as configured. Immutable once loaded."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Stable snake_case identifier the planner emits.")
    role: str = Field(
        min_length=20,
        description="Capability description. Embedded and matched against the user's goal.",
    )
    system_prompt: str = Field(min_length=1, description="Instructions given to the worker.")
    model: str = Field(description="Chat model id this agent runs on.")
    tool_selectors: list[str] = Field(
        default_factory=list,
        description="fnmatch globs over `<server>.<tool>` names. Empty means no tools.",
    )
    max_iterations: Annotated[int, Field(ge=1, le=20)] = 6
    requires_approval: bool = Field(
        default=False,
        description="If true the graph interrupts and waits for a human before running this agent.",
    )

    @field_validator("id")
    @classmethod
    def _check_id_is_snake_case(cls, value: str) -> str:
        if not AGENT_ID_PATTERN.match(value):
            raise ValueError(
                f"agent id {value!r} must be snake_case: lowercase, starting with a letter"
            )
        return value

    @field_validator("role")
    @classmethod
    def _check_role_is_a_capability(cls, value: str) -> str:
        """A role is retrieved by similarity, so a bare job title retrieves badly."""
        if len(value.split()) < 5:
            raise ValueError(
                f"role {value!r} is too short to retrieve on. Describe the capability "
                "-- what this agent can find out or work out -- not the job title."
            )
        return value
