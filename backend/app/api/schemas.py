"""Request and response models for the HTTP API.

These shapes are what FastAPI turns into the OpenAPI document at /docs, so the
descriptions here are the API documentation. Nothing in this module has
behaviour -- it is the boundary contract and nothing else.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CreateRunRequest(BaseModel):
    """A natural-language goal to orchestrate."""

    goal: str = Field(
        min_length=3,
        max_length=4000,
        description="What the user wants, in plain language.",
        examples=["Predict the velocity for our next sprint based on previous Jira sprints."],
    )


class CreateRunResponse(BaseModel):
    """Returned immediately; the run itself continues in the background."""

    run_id: str
    status: str = "running"


class ApprovalRequest(BaseModel):
    """A human's decision on a run paused at an approval interrupt."""

    approved: bool = Field(description="True runs the pending agent, false skips it.")
    reason: str = Field(default="", max_length=1000, description="Optional note, recorded in the trace.")


class UsageResponse(BaseModel):
    """Tokens and dollars for a run."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class TraceEntry(BaseModel):
    """One durable trace event, as stored in `steps`."""

    sequence: int
    name: str
    data: dict[str, Any]
    created_at: datetime


class RunResponse(BaseModel):
    """The full record of one run."""

    run_id: str
    goal: str
    status: str = Field(description="running, awaiting_approval, completed or failed.")
    plan: list[dict[str, Any]] = Field(default_factory=list)
    agent_ids: list[str] = Field(default_factory=list)
    final_answer: str | None = None
    failure: str | None = None
    usage: UsageResponse = Field(default_factory=UsageResponse)
    created_at: datetime | None = None
    completed_at: datetime | None = None
    trace: list[TraceEntry] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """One configured capability envelope, as the UI sees it.

    Field names are kept from when this described a fixed agent, because they
    still read correctly for an envelope and the UI is built on them.
    """

    id: str
    role: str = Field(description="What this capability can do. Embedded for planner retrieval.")
    model: str = Field(description="Default model for agents created here.")
    tool_selectors: list[str] = Field(description="The envelope's allowed selectors.")
    bound_tools: list[str] = Field(
        description="Tools those selectors resolve to -- the ceiling a created agent may request."
    )
    allowed_models: list[str] = Field(default_factory=list)
    max_iterations: int = Field(description="Iteration ceiling for agents created here.")
    requires_approval: bool


class ToolResponse(BaseModel):
    """One tool discovered from an MCP server at startup."""

    name: str
    server: str
    description: str
    input_schema: dict[str, Any]
