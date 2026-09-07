"""Regenerates fixtures/sprints.json, the data the mock Jira server serves.

The committed artifact is the JSON file, not this script -- the server reads
the JSON and never imports this module. This exists so the fixture can be
explained and extended rather than being an opaque blob: add a sprint to
SPRINT_SHAPES or a person to MEMBERS, re-run, and the issue lists are rebuilt
around them.

It is deterministic. The seed is fixed, so running it without changing the
inputs rewrites the same bytes and produces no diff. Change an input and only
the affected data moves.

The one property worth protecting is that the fixture is internally
consistent -- an agent can cross-check get_sprint_issues against get_sprints
and the numbers agree:

    sum of every story_points        == committed_points
    sum of story_points where "Done" == completed_points

`build_issues` is what enforces that, by partitioning each sprint's points
rather than inventing issues and hoping they add up.

Run with:  python mcp_servers/jira_mock/generate_fixtures.py
"""

import json
import random
from pathlib import Path
from typing import Any

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sprints.json"

# Changing this reshuffles every issue list. It is pinned so the committed
# fixture is reproducible.
RANDOM_SEED = 11

# (id, start_date, end_date, committed_points, completed_points)
SPRINT_SHAPES: list[tuple[str, str, str, int, int]] = [
    ("SPR-31", "2025-04-07", "2025-04-18", 42, 34),
    ("SPR-32", "2025-04-21", "2025-05-02", 40, 38),
    ("SPR-33", "2025-05-05", "2025-05-16", 45, 31),
    ("SPR-34", "2025-05-19", "2025-05-30", 40, 40),
    ("SPR-35", "2025-06-02", "2025-06-13", 44, 37),
    ("SPR-36", "2025-06-16", "2025-06-27", 42, 39),
    ("SPR-37", "2025-06-30", "2025-07-11", 46, 35),
    ("SPR-38", "2025-07-14", "2025-07-25", 43, 41),
]

MEMBERS: list[dict[str, Any]] = [
    {"name": "Ana Duarte", "role": "Backend", "days_available": 9},
    {"name": "Ben Oyelaran", "role": "Backend", "days_available": 10},
    {"name": "Chika Mori", "role": "Frontend", "days_available": 8},
    {"name": "Dmitri Vasilev", "role": "Frontend", "days_available": 10},
    {"name": "Elena Roche", "role": "QA", "days_available": 7},
]

# `team_size` and `total_member_days` are deliberately absent: the server
# derives both from `members` on every call, so storing them here would be a
# second copy of a computed value, free to drift from the roster above.
TEAM_CAPACITY: dict[str, Any] = {
    "next_sprint_id": "SPR-39",
    "start_date": "2025-07-28",
    "end_date": "2025-08-08",
    "working_days": 10,
    "members": MEMBERS,
    "planned_time_off_days": 6,
    "notes": "Two engineers are on leave for part of the sprint; QA is at 70 percent.",
}

# Fibonacci-ish estimates, the convention most Jira boards use. It includes 1,
# so any whole number of points can be partitioned exactly.
STORY_POINT_PALETTE = [1, 2, 3, 5, 8]

ISSUE_TYPES = ["Story", "Bug", "Task", "Spike"]

ISSUE_SUMMARIES = [
    "Add pagination to the audit log endpoint",
    "Fix flaky checkout integration test",
    "Migrate session store off in-memory cache",
    "Investigate p99 latency on /search",
    "Ship the invoice PDF export",
    "Harden the webhook retry policy",
    "Rewrite the onboarding empty state",
    "Reduce cold-start time for the worker pool",
    "Backfill missing tenant metadata",
    "Upgrade the Postgres driver",
    "Split the billing service deploy",
    "Add tracing spans to the ingest path",
]


class InconsistentFixtureError(AssertionError):
    """Raised when generated issues do not reconcile with their sprint record."""


def partition_points(total: int) -> list[int]:
    """Split `total` into story point estimates summing to exactly `total`.

    Biased toward the larger usable estimates, otherwise a 40-point sprint
    tends to come out as a long tail of one-pointers, which no real board
    looks like.
    """
    parts: list[int] = []
    remaining = total
    while remaining > 0:
        usable = [points for points in STORY_POINT_PALETTE if points <= remaining]
        parts.append(random.choice(usable[-3:] if len(usable) >= 3 else usable))
        remaining -= parts[-1]
    return parts


def build_issues(sprint_id: str, committed: int, completed: int) -> list[dict[str, Any]]:
    """Build one sprint's issue list so that its points reconcile by construction.

    Completed points are partitioned into "Done" issues and the shortfall into
    "Rolled over" ones, so the two identities in this module's docstring hold
    without needing to be checked afterwards.
    """
    done = [(points, "Done") for points in partition_points(completed)]
    rolled_over = [(points, "Rolled over") for points in partition_points(committed - completed)]

    bucket = done + rolled_over
    random.shuffle(bucket)

    return [
        {
            "key": f"{sprint_id}-{index + 1}",
            "summary": random.choice(ISSUE_SUMMARIES),
            "type": random.choice(ISSUE_TYPES),
            "status": status,
            "story_points": points,
            "assignee": random.choice([member["name"] for member in MEMBERS]),
        }
        for index, (points, status) in enumerate(bucket)
    ]


def build_fixture() -> dict[str, Any]:
    """Assemble the whole fixture: sprints, their issues, and team capacity."""
    random.seed(RANDOM_SEED)

    sprints: list[dict[str, Any]] = []
    issues: dict[str, list[dict[str, Any]]] = {}

    for sprint_id, start_date, end_date, committed, completed in SPRINT_SHAPES:
        sprints.append({
            "id": sprint_id,
            "name": f"Sprint {sprint_id.split('-')[1]}",
            "state": "closed",
            "start_date": start_date,
            "end_date": end_date,
            "committed_points": committed,
            "completed_points": completed,
        })

    # Issues are generated in a second pass so the random stream depends only on
    # the sprint shapes, not on how the sprint records happen to be built.
    for sprint_id, _, _, committed, completed in SPRINT_SHAPES:
        issues[sprint_id] = build_issues(sprint_id, committed, completed)

    return {"sprints": sprints, "issues": issues, "team_capacity": TEAM_CAPACITY}


def check_consistency(fixture: dict[str, Any]) -> None:
    """Fail loudly if the fixture would ship with numbers that disagree."""
    for sprint in fixture["sprints"]:
        issues = fixture["issues"][sprint["id"]]
        total = sum(issue["story_points"] for issue in issues)
        delivered = sum(
            issue["story_points"] for issue in issues if issue["status"] == "Done"
        )
        if total != sprint["committed_points"]:
            raise InconsistentFixtureError(
                f"{sprint['id']}: issues total {total} points but the sprint committed "
                f"{sprint['committed_points']}"
            )
        if delivered != sprint["completed_points"]:
            raise InconsistentFixtureError(
                f"{sprint['id']}: Done issues total {delivered} points but the sprint "
                f"completed {sprint['completed_points']}"
            )


def write_fixture() -> Path:
    """Generate, verify and write the fixture. Returns the path written."""
    fixture = build_fixture()
    check_consistency(fixture)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    return FIXTURE_PATH


if __name__ == "__main__":
    path = write_fixture()
    print(
        f"Wrote {path} -- {len(SPRINT_SHAPES)} sprints, "
        f"{sum(len(i) for i in json.loads(path.read_text())['issues'].values())} issues."
    )
