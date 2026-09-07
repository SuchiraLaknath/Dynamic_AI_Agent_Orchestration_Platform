"""The two agent shapes: what is configured, and what is built at runtime.

`CapabilitySpec` is a record in agents.yaml. It is an *envelope*: the tools a
class of agent may reach, the models it may run on, its iteration ceiling, and
the policy it must obey. It deliberately contains no system prompt.

`AgentSpec` is one concrete agent, assembled per task by
`app.agents.composer.realize_agent` from an envelope plus the agent the planner
synthesized. It is the factory's input contract, and nothing loads it from disk.

The split is the point: configuration owns *permissions*, the planner owns
*instructions*. Neither can do the other's job -- a synthesized agent cannot
widen its envelope, and an envelope cannot dictate wording.
"""

import re
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AGENT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _reject_bare_job_title(value: str) -> str:
    """A description is retrieved by similarity, so a job title retrieves badly."""
    if len(value.split()) < 5:
        raise ValueError(
            f"{value!r} is too short to retrieve on. Describe the capability "
            "-- what can be found out or worked out -- not the job title."
        )
    return value


class CapabilitySpec(BaseModel):
    """One capability envelope, as configured in agents.yaml. Immutable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Stable snake_case identifier the planner selects.")
    description: str = Field(
        min_length=20,
        description="What this capability can do. Embedded and matched against the goal.",
    )
    allowed_tool_selectors: list[str] = Field(
        default_factory=list,
        description=(
            "fnmatch globs over `<server>.<tool>`. The permission ceiling: a synthesized "
            "agent may request any subset of what these resolve to, and nothing else."
        ),
    )
    allowed_models: list[str] = Field(
        min_length=1, description="Chat model ids an agent in this envelope may run on."
    )
    default_model: str = Field(description="Model used when the planner does not choose one.")
    max_iterations_limit: Annotated[int, Field(ge=1, le=20)] = Field(
        default=8, description="Ceiling on a synthesized agent's tool-calling iterations."
    )
    requires_approval: bool = Field(
        default=False,
        description=(
            "If true, every agent built in this envelope pauses for a human. Set here and "
            "never on the synthesized agent, so a generated agent cannot approve itself."
        ),
    )
    policy: str = Field(
        default="",
        description=(
            "Non-negotiable constraints appended after the synthesized prompt. This is "
            "where a rule that must survive any wording belongs."
        ),
    )

    @field_validator("id")
    @classmethod
    def _check_id_is_snake_case(cls, value: str) -> str:
        if not AGENT_ID_PATTERN.match(value):
            raise ValueError(
                f"capability id {value!r} must be snake_case: lowercase, starting with a letter"
            )
        return value

    @field_validator("description")
    @classmethod
    def _check_description_is_a_capability(cls, value: str) -> str:
        return _reject_bare_job_title(value)

    @model_validator(mode="after")
    def _check_default_model_is_allowed(self) -> "CapabilitySpec":
        if self.default_model not in self.allowed_models:
            raise ValueError(
                f"capability {self.id!r}: default_model {self.default_model!r} is not in "
                f"allowed_models {self.allowed_models}"
            )
        return self


class SynthesizedAgent(BaseModel):
    """An agent the planner invented, before it has been checked or realized.

    This is structured LLM output, so it is untrusted: every field here is a
    *request*, and `app.agents.composer.realize_agent` decides what is granted.
    """

    name: str = Field(
        description="snake_case name for this agent, e.g. 'sprint_history_reader'."
    )
    role: str = Field(
        min_length=20, description="One sentence on what this agent was created to do."
    )
    system_prompt: str = Field(
        min_length=1,
        description=(
            "The agent's instructions, written for this task. Say how to work and what "
            "to produce. Do not restate the objective."
        ),
    )
    tool_selectors: list[str] = Field(
        default_factory=list,
        description="Tools this agent needs. Must be a subset of its capability's allowance.",
    )
    model: str = Field(default="", description="Optional; the capability's default is used if empty.")
    max_iterations: Annotated[int, Field(ge=1, le=20)] = 4

    @field_validator("name")
    @classmethod
    def _check_name_is_snake_case(cls, value: str) -> str:
        if not AGENT_ID_PATTERN.match(value):
            raise ValueError(
                f"agent name {value!r} must be snake_case: lowercase, starting with a letter"
            )
        return value

    @field_validator("role")
    @classmethod
    def _check_role_is_a_capability(cls, value: str) -> str:
        return _reject_bare_job_title(value)


class AgentSpec(BaseModel):
    """One concrete agent, realized for a single task. Built, never loaded."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="The synthesized agent's name, used in the trace.")
    role: str = Field(min_length=20, description="What this agent was created to do.")
    system_prompt: str = Field(min_length=1, description="Instructions given to the worker.")
    model: str = Field(description="Chat model id this agent runs on.")
    tool_selectors: list[str] = Field(
        default_factory=list,
        description="fnmatch globs, already validated to sit inside the envelope.",
    )
    max_iterations: Annotated[int, Field(ge=1, le=20)] = 6
    requires_approval: bool = Field(
        default=False, description="Carried from the envelope, never from the synthesized agent."
    )
    capability_id: str = Field(
        default="",
        description="The envelope this agent was realized from, for the trace and audit.",
    )

    @field_validator("id")
    @classmethod
    def _check_id_is_snake_case(cls, value: str) -> str:
        if not AGENT_ID_PATTERN.match(value):
            raise ValueError(
                f"agent name {value!r} must be snake_case: lowercase, starting with a letter"
            )
        return value

    @field_validator("role")
    @classmethod
    def _check_role_is_a_capability(cls, value: str) -> str:
        return _reject_bare_job_title(value)
