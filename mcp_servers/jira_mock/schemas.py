"""Pydantic models for what the Jira tools return.

Annotating a FastMCP tool with one of these makes the server advertise a real
`outputSchema` in its tool listing and return `structuredContent` alongside the
text, instead of an opaque `dict[str, Any]`. It also means the fixture is
validated on the way out: a field renamed in sprints.json fails here rather
than reaching an agent as a silently missing key.

What these do NOT do is tell the model anything. The Anthropic tool-use API
carries only `name`, `input_schema` and `description`, so an outputSchema stops
at the MCP layer -- it is read by MCP clients and by code, never by the agent
deciding what to do with a result. That is why the tools in server.py still
describe their return shape in prose: the docstring is the only channel that
reaches the model. Two audiences, two representations.

Literal types are used for the small closed vocabularies (`state`, `status`,
`role`, issue `type`) because they make the published schema self-explanatory
and turn a malformed fixture into an immediate validation error.
"""

from typing import Literal

from pydantic import BaseModel, Field

IssueStatus = Literal["Done", "Rolled over"]
IssueType = Literal["Story", "Bug", "Task", "Spike"]
TeamRole = Literal["Backend", "Frontend", "QA"]


class Sprint(BaseModel):
    """One closed sprint, as recorded on the board."""

    id: str = Field(description='Sprint id, e.g. "SPR-38". Pass to get_sprint_issues.')
    name: str = Field(description='Display name, e.g. "Sprint 38".')
    state: Literal["closed"] = Field(description="Only closed sprints are returned.")
    start_date: str = Field(description='ISO date, "YYYY-MM-DD".')
    end_date: str = Field(description='ISO date, "YYYY-MM-DD".')
    committed_points: int = Field(description="Points committed to at sprint start.")
    completed_points: int = Field(
        description="Points actually delivered. Average these to compute velocity."
    )


class SprintsResult(BaseModel):
    """Return type of get_sprints."""

    count: int = Field(description="How many sprints are in `sprints`.")
    sprints: list[Sprint] = Field(description="Oldest first, newest last.")


class Issue(BaseModel):
    """One issue inside a sprint."""

    key: str = Field(description='Issue key, e.g. "SPR-38-1".')
    summary: str = Field(description="One-line description of the work.")
    type: IssueType = Field(description="Issue type.")
    status: IssueStatus = Field(
        description='"Done" if delivered this sprint, "Rolled over" if carried to a later one.'
    )
    story_points: int = Field(description="The estimate, in points.")
    assignee: str = Field(description="Full name of the assigned engineer.")


class SprintIssuesResult(BaseModel):
    """Return type of get_sprint_issues."""

    sprint_id: str = Field(description="Echo of the requested sprint.")
    count: int = Field(description="How many issues are in `issues`.")
    completed_points: int = Field(
        description=(
            "Sum of story_points over issues whose status is \"Done\". Reconciles "
            "exactly with completed_points for the same sprint in get_sprints."
        )
    )
    issues: list[Issue] = Field(description="Every issue in the sprint.")


class TeamMember(BaseModel):
    """One person on the team, for the upcoming sprint."""

    name: str = Field(description="Full name.")
    role: TeamRole = Field(description="What they work on.")
    days_available: int = Field(
        description="Days this person can work, already net of their planned time off."
    )


class TeamCapacityResult(BaseModel):
    """Return type of get_team_capacity."""

    next_sprint_id: str = Field(description='The upcoming sprint, e.g. "SPR-39".')
    start_date: str = Field(description='ISO date, "YYYY-MM-DD".')
    end_date: str = Field(description='ISO date, "YYYY-MM-DD".')
    working_days: int = Field(description="Working days in the sprint, per person.")
    members: list[TeamMember] = Field(description="The roster.")
    planned_time_off_days: int = Field(description="Total days off across the whole team.")
    notes: str = Field(description="Free-text caveats a planner should read.")
    team_size: int = Field(description="Number of members. Derived, not stored in the fixture.")
    total_member_days: int = Field(
        description=(
            "Sum of days_available across members. Divide by (team_size * working_days) "
            "for an availability ratio to scale a baseline velocity by. Derived."
        )
    )


class PublishResult(BaseModel):
    """Return type of publish_sprint_plan."""

    published: bool = Field(description="True when the commitment was accepted.")
    sprint_id: str = Field(description="The sprint that was written to.")
    committed_points: int = Field(description="The total that was published.")
    summary: str = Field(description="The rationale that was published.")
    url: str = Field(description="Link to the sprint board showing the new commitment.")
