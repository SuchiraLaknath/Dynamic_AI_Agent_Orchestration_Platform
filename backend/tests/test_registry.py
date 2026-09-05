"""The agent registry is the interface everything else depends on.

These tests pin the failure messages, not just the failures: a malformed
agents.yaml is something an operator has to fix from the startup log alone.
"""

import pytest

from app.agents.registry import AgentRegistryError, load_agent_registry

VALID_AGENT = """
agents:
  - id: jira_analyst
    role: Retrieves and summarizes historical Jira sprint data and story points
    system_prompt: You analyze sprint data.
    model: claude-sonnet-5
    tool_selectors: ["jira.*"]
"""


def test_loads_a_valid_registry(write_agents_yaml):
    registry = load_agent_registry(write_agents_yaml(VALID_AGENT))

    assert registry.agent_ids == {"jira_analyst"}
    spec = registry.get("jira_analyst")
    assert spec.tool_selectors == ["jira.*"]
    assert spec.max_iterations == 6, "documented default should apply when omitted"
    assert spec.requires_approval is False


def test_the_projects_own_registry_is_valid(agents):
    """The shipped agents.yaml must load, or the app cannot start."""
    assert len(agents) >= 2
    assert "jira_analyst" in agents.agent_ids


def test_missing_file_names_the_path(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(AgentRegistryError, match=str(missing)):
        load_agent_registry(missing)


def test_invalid_yaml_says_so(write_agents_yaml):
    with pytest.raises(AgentRegistryError, match="not valid YAML"):
        load_agent_registry(write_agents_yaml("agents: [unclosed"))


def test_missing_agents_key_is_rejected(write_agents_yaml):
    with pytest.raises(AgentRegistryError, match="top-level 'agents' key"):
        load_agent_registry(write_agents_yaml("specialists: []"))


def test_empty_registry_is_rejected(write_agents_yaml):
    with pytest.raises(AgentRegistryError, match="non-empty list"):
        load_agent_registry(write_agents_yaml("agents: []"))


def test_unknown_field_is_rejected_naming_the_agent(write_agents_yaml):
    """A typo'd key must fail loudly, not be silently ignored."""
    content = VALID_AGENT + "    temprature: 0.7\n"
    with pytest.raises(AgentRegistryError, match="jira_analyst"):
        load_agent_registry(write_agents_yaml(content))


def test_a_job_title_role_is_rejected(write_agents_yaml):
    """Roles are embedded and searched, so a bare title cannot be retrieved on."""
    content = VALID_AGENT.replace(
        "Retrieves and summarizes historical Jira sprint data and story points", "Jira Guy"
    )
    with pytest.raises(AgentRegistryError, match="role"):
        load_agent_registry(write_agents_yaml(content))


def test_non_snake_case_id_is_rejected(write_agents_yaml):
    content = VALID_AGENT.replace("id: jira_analyst", "id: JiraAnalyst")
    with pytest.raises(AgentRegistryError, match="snake_case"):
        load_agent_registry(write_agents_yaml(content))


def test_duplicate_ids_are_rejected(write_agents_yaml):
    with pytest.raises(AgentRegistryError, match="duplicate agent ids"):
        load_agent_registry(write_agents_yaml(VALID_AGENT + VALID_AGENT.replace("agents:\n", "")))


def test_unknown_agent_lookup_lists_what_exists(agents):
    with pytest.raises(AgentRegistryError, match="jira_analyst"):
        agents.get("no_such_agent")


def test_role_search_ranks_the_relevant_agent_above_an_irrelevant_one(agents):
    ranked = agents.search_by_role("predict our next sprint velocity from past sprints", limit=5)

    assert ranked, "search must always return a menu"
    ids = [spec.id for spec in ranked]
    assert "velocity_forecaster" in ids
    # The publisher writes to Jira; it has no business ranking above the forecaster.
    if "sprint_plan_publisher" in ids:
        assert ids.index("velocity_forecaster") < ids.index("sprint_plan_publisher")
