"""Mock Jira MCP server, speaking MCP over stdio.

Serves three read-only tools and one write tool backed by a JSON fixture so the platform can be
demonstrated without Jira credentials. It does not authenticate, paginate, or
write -- swapping it for a real Jira MCP server is an edit to
`backend/config/mcp_servers.yaml`, not a code change here.
"""

import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from schemas import (
    Issue,
    PublishResult,
    Sprint,
    SprintIssuesResult,
    SprintsResult,
    TeamCapacityResult,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sprints.json"

mcp = FastMCP("jira-mock")


def _fixture() -> dict[str, Any]:
    """Read on every call so editing the fixture does not need a restart."""
    return json.loads(FIXTURE_PATH.read_text())


class SprintNotFoundError(ValueError):
    """Raised when a caller asks for a sprint id that is not in the fixture."""


@mcp.tool()
def get_sprints(limit: int = 6) -> SprintsResult:
    """Return the most recently closed Jira sprints, newest last.

    Each sprint carries its committed and completed story points, which is the
    raw material for a velocity calculation.

    Args:
        limit: How many of the most recent sprints to return.

    Returns:
        An object with:
          count (int): how many sprints are in `sprints`.
          sprints (list): oldest first, newest last. Each entry has:
            id (str): sprint id, e.g. "SPR-38". Pass this to get_sprint_issues.
            name (str): display name, e.g. "Sprint 38".
            state (str): always "closed" -- only finished sprints are returned.
            start_date (str): ISO date, "YYYY-MM-DD".
            end_date (str): ISO date, "YYYY-MM-DD".
            committed_points (int): points the team committed to at sprint start.
            completed_points (int): points actually delivered. This is the
              figure to average when computing velocity.
    """
    sprints = _fixture()["sprints"]
    selected = sprints[-limit:] if limit > 0 else sprints
    return SprintsResult(
        count=len(selected), sprints=[Sprint(**sprint) for sprint in selected]
    )


@mcp.tool()
def get_sprint_issues(sprint_id: str) -> SprintIssuesResult:
    """Return every issue in one sprint, with status, type and story points.

    Args:
        sprint_id: A sprint id such as "SPR-38", as returned by get_sprints.

    Returns:
        An object with:
          sprint_id (str): echo of the requested sprint.
          count (int): how many issues are in `issues`.
          completed_points (int): sum of story_points over issues whose status
            is "Done". Reconciles exactly with the completed_points that
            get_sprints reports for the same sprint.
          issues (list): every issue in the sprint. Each entry has:
            key (str): issue key, e.g. "SPR-38-1".
            summary (str): one-line description of the work.
            type (str): one of "Story", "Bug", "Task", "Spike".
            status (str): "Done" if delivered in this sprint, otherwise
              "Rolled over" -- meaning it was carried into a later sprint.
            story_points (int): the estimate, in points.
            assignee (str): full name of the assigned engineer.

        The issue list is internally consistent with the sprint record, so
        these two identities always hold and can be used to check a figure:
          sum of every story_points          == committed_points (get_sprints)
          sum of story_points where "Done"   == completed_points (get_sprints)

    Raises:
        SprintNotFoundError: if sprint_id is not a known sprint. The message
            lists the valid ids.
    """
    issues_by_sprint = _fixture()["issues"]
    if sprint_id not in issues_by_sprint:
        raise SprintNotFoundError(
            f"Unknown sprint id {sprint_id!r}. Known ids: {', '.join(sorted(issues_by_sprint))}"
        )
    issues = issues_by_sprint[sprint_id]
    return SprintIssuesResult(
        sprint_id=sprint_id,
        count=len(issues),
        completed_points=sum(i["story_points"] for i in issues if i["status"] == "Done"),
        issues=[Issue(**issue) for issue in issues],
    )


@mcp.tool()
def get_team_capacity() -> TeamCapacityResult:
    """Return the upcoming sprint's team roster, working days and planned time off.

    Use this to adjust a historical velocity for who is actually available.
    Takes no arguments -- it always describes the next sprint.

    Returns:
        An object with:
          next_sprint_id (str): the upcoming sprint, e.g. "SPR-39".
          start_date (str): ISO date, "YYYY-MM-DD".
          end_date (str): ISO date, "YYYY-MM-DD".
          working_days (int): working days in the sprint, per person.
          members (list): the roster. Each entry has:
            name (str): full name.
            role (str): one of "Backend", "Frontend", "QA".
            days_available (int): days this person can actually work, already
              net of their planned time off.
          planned_time_off_days (int): total days off across the whole team.
          notes (str): free-text caveats a planner should read.
          team_size (int): number of members.
          total_member_days (int): sum of days_available across members.
            Divide by (team_size * working_days) for an availability ratio to
            scale a baseline velocity by.
    """
    capacity = _fixture()["team_capacity"]
    return TeamCapacityResult(
        **capacity,
        team_size=len(capacity["members"]),
        total_member_days=sum(m["days_available"] for m in capacity["members"]),
    )


@mcp.tool()
def publish_sprint_plan(
    sprint_id: str, committed_points: int, summary: str
) -> PublishResult:
    """Publish a committed sprint plan back to Jira. This writes to the board.

    Args:
        sprint_id: The sprint to publish the plan against, e.g. "SPR-39".
        committed_points: Story points the team is committing to.
        summary: A short human-readable rationale for the commitment.

    Returns:
        An object with:
          published (bool): True when the commitment was accepted.
          sprint_id (str): the sprint that was written to.
          committed_points (int): the total that was published.
          summary (str): the rationale that was published.
          url (str): link to the sprint board showing the new commitment.
    """
    # Deliberately not persisted: the fixture is a read model, and the point of
    # this tool is to be a write that the approval gate has to stop.
    return PublishResult(
        published=True,
        sprint_id=sprint_id,
        committed_points=committed_points,
        summary=summary,
        url=f"https://jira.example.com/secure/RapidBoard.jspa?sprint={sprint_id}",
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
