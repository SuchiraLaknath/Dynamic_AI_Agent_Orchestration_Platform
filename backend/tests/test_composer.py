"""The envelope check is the security boundary of runtime agent creation.

The planner writes agents, and an LLM writes what it likes. These tests are the
reason that is safe: every one of them is an agent asking for more than it may
have, and every one must be refused. They run against the real discovered tools
so the glob resolution is the same code the app uses.
"""

import pytest

from app.agents.composer import POLICY_HEADER, EnvelopeViolationError, realize_agent
from app.agents.models import CapabilitySpec, SynthesizedAgent

READ_ONLY = CapabilitySpec(
    id="jira_reporting",
    description="Retrieves and summarizes historical Jira sprint records and story points",
    allowed_tool_selectors=["jira.get_sprints", "jira.get_sprint_issues"],
    allowed_models=["claude-sonnet-5", "claude-haiku-4-5"],
    default_model="claude-sonnet-5",
    max_iterations_limit=6,
    policy="Never state a figure you did not retrieve.",
)

NO_TOOLS = CapabilitySpec(
    id="analysis_only",
    description="Reasons over figures that earlier tasks already produced, with no tools",
    allowed_tool_selectors=[],
    allowed_models=["claude-sonnet-5"],
    default_model="claude-sonnet-5",
    max_iterations_limit=3,
)

WRITEBACK = CapabilitySpec(
    id="jira_writeback",
    description="Writes a finalized sprint commitment back to the Jira board for the team",
    allowed_tool_selectors=["jira.publish_sprint_plan"],
    allowed_models=["claude-sonnet-5"],
    default_model="claude-sonnet-5",
    requires_approval=True,
)


def agent(**overrides) -> SynthesizedAgent:
    """A well-formed synthesized agent, before whatever the test wants to break."""
    return SynthesizedAgent(
        **{
            "name": "sprint_history_reader",
            "role": "Reads closed sprints and reports committed versus completed points",
            "system_prompt": "Fetch the last six sprints and tabulate their points.",
            "tool_selectors": ["jira.get_sprints"],
            "model": "",
            "max_iterations": 4,
            **overrides,
        }
    )


async def test_a_well_formed_agent_is_realized(tools):
    spec = realize_agent(READ_ONLY, agent(), tools)

    assert spec.id == "sprint_history_reader"
    assert spec.tool_selectors == ["jira.get_sprints"]
    assert spec.capability_id == "jira_reporting"
    assert spec.model == "claude-sonnet-5", "empty model falls back to the capability default"


async def test_agent_name_is_not_in_any_config_file(tools, capabilities):
    """The created agent is genuinely new, not a configured record under another name."""
    spec = realize_agent(READ_ONLY, agent(), tools)
    assert spec.id not in capabilities.capability_ids


# --- escalation: the tests this design exists for --------------------------


async def test_requesting_a_tool_outside_the_envelope_is_refused(tools):
    """The core case: a read-only agent tries to award itself the write tool."""
    with pytest.raises(EnvelopeViolationError, match="jira.publish_sprint_plan"):
        realize_agent(READ_ONLY, agent(tool_selectors=["jira.publish_sprint_plan"]), tools)


async def test_a_glob_cannot_widen_the_envelope(tools):
    """'jira.*' is not textually a subset of the allowed list, and must not resolve past it.

    This is why the check compares resolved tool names rather than glob strings.
    """
    with pytest.raises(EnvelopeViolationError, match="jira.publish_sprint_plan"):
        realize_agent(READ_ONLY, agent(tool_selectors=["jira.*"]), tools)


async def test_a_toolless_capability_grants_no_tools(tools):
    with pytest.raises(EnvelopeViolationError, match="does not permit"):
        realize_agent(NO_TOOLS, agent(tool_selectors=["jira.get_sprints"]), tools)


async def test_a_selector_matching_nothing_is_refused(tools):
    """A typo would otherwise run toolless and die on the iteration limit instead."""
    with pytest.raises(EnvelopeViolationError, match="matches no discovered tool"):
        realize_agent(READ_ONLY, agent(tool_selectors=["jira.get_sprintz"]), tools)


async def test_a_model_outside_the_allowlist_is_refused(tools):
    with pytest.raises(EnvelopeViolationError, match="claude-opus-5"):
        realize_agent(READ_ONLY, agent(model="claude-opus-5"), tools)


async def test_iterations_above_the_ceiling_are_refused(tools):
    with pytest.raises(EnvelopeViolationError, match="above capability"):
        realize_agent(READ_ONLY, agent(max_iterations=20), tools)


async def test_an_agent_cannot_clear_its_own_approval_requirement(tools):
    """requires_approval comes from the envelope, so a generated agent cannot opt out."""
    spec = realize_agent(
        WRITEBACK, agent(tool_selectors=["jira.publish_sprint_plan"]), tools
    )
    assert spec.requires_approval is True


async def test_partial_escalation_is_still_refused(tools):
    """One legitimate tool does not buy an illegitimate one alongside it."""
    with pytest.raises(EnvelopeViolationError, match="jira.get_team_capacity"):
        realize_agent(
            READ_ONLY, agent(tool_selectors=["jira.get_sprints", "jira.get_team_capacity"]), tools
        )


# --- prompt composition ----------------------------------------------------


async def test_policy_is_appended_after_the_synthesized_prompt(tools):
    """Ordering matters: later instructions dominate, so policy must come last."""
    spec = realize_agent(READ_ONLY, agent(), tools)

    assert "Fetch the last six sprints" in spec.system_prompt
    assert READ_ONLY.policy in spec.system_prompt
    assert POLICY_HEADER in spec.system_prompt
    assert spec.system_prompt.index("Fetch the last six sprints") < spec.system_prompt.index(
        READ_ONLY.policy
    )


async def test_a_capability_without_policy_adds_no_header(tools):
    spec = realize_agent(NO_TOOLS, agent(tool_selectors=[], max_iterations=3), tools)

    assert POLICY_HEADER not in spec.system_prompt
    assert spec.system_prompt == "Fetch the last six sprints and tabulate their points."
