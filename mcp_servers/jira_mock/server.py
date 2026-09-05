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

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sprints.json"

mcp = FastMCP("jira-mock")


def _fixture() -> dict[str, Any]:
    """Read on every call so editing the fixture does not need a restart."""
    return json.loads(FIXTURE_PATH.read_text())


class SprintNotFoundError(ValueError):
    """Raised when a caller asks for a sprint id that is not in the fixture."""


@mcp.tool()
def get_sprints(limit: int = 6) -> dict[str, Any]:
    """Return the most recently closed Jira sprints, newest last.

    Each sprint carries its committed and completed story points, which is the
    raw material for a velocity calculation.

    Args:
        limit: How many of the most recent sprints to return.
    """
    sprints = _fixture()["sprints"]
    selected = sprints[-limit:] if limit > 0 else sprints
    return {"count": len(selected), "sprints": selected}


@mcp.tool()
def get_sprint_issues(sprint_id: str) -> dict[str, Any]:
    """Return every issue in one sprint, with status, type and story points.

    Args:
        sprint_id: A sprint id such as "SPR-38", as returned by get_sprints.
    """
    issues_by_sprint = _fixture()["issues"]
    if sprint_id not in issues_by_sprint:
        raise SprintNotFoundError(
            f"Unknown sprint id {sprint_id!r}. Known ids: {', '.join(sorted(issues_by_sprint))}"
        )
    issues = issues_by_sprint[sprint_id]
    return {
        "sprint_id": sprint_id,
        "count": len(issues),
        "completed_points": sum(i["story_points"] for i in issues if i["status"] == "Done"),
        "issues": issues,
    }


@mcp.tool()
def get_team_capacity() -> dict[str, Any]:
    """Return the upcoming sprint's team roster, working days and planned time off.

    Use this to adjust a historical velocity for who is actually available.
    """
    capacity = _fixture()["team_capacity"]
    total_member_days = sum(m["days_available"] for m in capacity["members"])
    return {
        **capacity,
        "team_size": len(capacity["members"]),
        "total_member_days": total_member_days,
    }


@mcp.tool()
def publish_sprint_plan(sprint_id: str, committed_points: int, summary: str) -> dict[str, Any]:
    """Publish a committed sprint plan back to Jira. This writes to the board.

    Args:
        sprint_id: The sprint to publish the plan against, e.g. "SPR-39".
        committed_points: Story points the team is committing to.
        summary: A short human-readable rationale for the commitment.
    """
    # Deliberately not persisted: the fixture is a read model, and the point of
    # this tool is to be a write that the approval gate has to stop.
    return {
        "published": True,
        "sprint_id": sprint_id,
        "committed_points": committed_points,
        "summary": summary,
        "url": f"https://jira.example.com/secure/RapidBoard.jspa?sprint={sprint_id}",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
