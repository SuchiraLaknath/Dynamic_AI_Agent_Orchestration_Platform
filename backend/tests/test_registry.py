"""The capability registry is the interface everything else depends on.

These tests pin the failure messages, not just the failures: a malformed
agents.yaml is something an operator has to fix from the startup log alone.
"""

import pytest

from app.agents.registry import CapabilityRegistryError, load_capability_registry

VALID_CAPABILITY = """
capabilities:
  - id: jira_reporting
    description: Retrieves and summarizes historical Jira sprint data and story points
    allowed_tool_selectors: ["jira.get_*"]
    allowed_models: ["claude-sonnet-5"]
    default_model: claude-sonnet-5
"""


def test_loads_a_valid_registry(write_agents_yaml):
    registry = load_capability_registry(write_agents_yaml(VALID_CAPABILITY))

    assert registry.capability_ids == {"jira_reporting"}
    spec = registry.get("jira_reporting")
    assert spec.allowed_tool_selectors == ["jira.get_*"]
    assert spec.max_iterations_limit == 8, "documented default should apply when omitted"
    assert spec.requires_approval is False
    assert spec.policy == ""


def test_the_projects_own_registry_is_valid(capabilities):
    """The shipped agents.yaml must load, or the app cannot start."""
    assert len(capabilities) >= 2
    assert "jira_reporting" in capabilities.capability_ids


def test_capabilities_declare_no_system_prompt(capabilities):
    """The whole point: prompts are written at runtime, never configured."""
    for spec in capabilities.all_specs():
        assert not hasattr(spec, "system_prompt")


def test_missing_file_names_the_path(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(CapabilityRegistryError, match=str(missing)):
        load_capability_registry(missing)


def test_invalid_yaml_says_so(write_agents_yaml):
    with pytest.raises(CapabilityRegistryError, match="not valid YAML"):
        load_capability_registry(write_agents_yaml("capabilities: [unclosed"))


def test_missing_capabilities_key_is_rejected(write_agents_yaml):
    with pytest.raises(CapabilityRegistryError, match="top-level 'capabilities' key"):
        load_capability_registry(write_agents_yaml("specialists: []"))


def test_empty_registry_is_rejected(write_agents_yaml):
    with pytest.raises(CapabilityRegistryError, match="non-empty list"):
        load_capability_registry(write_agents_yaml("capabilities: []"))


def test_unknown_field_is_rejected_naming_the_capability(write_agents_yaml):
    """A typo'd key must fail loudly, not be silently ignored."""
    content = VALID_CAPABILITY + "    temprature: 0.7\n"
    with pytest.raises(CapabilityRegistryError, match="jira_reporting"):
        load_capability_registry(write_agents_yaml(content))


def test_a_system_prompt_in_config_is_rejected(write_agents_yaml):
    """Guards the architecture: a prompt in YAML is the thing this design removed."""
    content = VALID_CAPABILITY + "    system_prompt: You analyze sprint data.\n"
    with pytest.raises(CapabilityRegistryError, match="jira_reporting"):
        load_capability_registry(write_agents_yaml(content))


def test_a_job_title_description_is_rejected(write_agents_yaml):
    """Descriptions are embedded and searched, so a bare title cannot be retrieved on."""
    content = VALID_CAPABILITY.replace(
        "Retrieves and summarizes historical Jira sprint data and story points", "Jira Guy"
    )
    with pytest.raises(CapabilityRegistryError, match="description"):
        load_capability_registry(write_agents_yaml(content))


def test_default_model_outside_allowed_models_is_rejected(write_agents_yaml):
    content = VALID_CAPABILITY.replace("default_model: claude-sonnet-5", "default_model: gpt-9")
    with pytest.raises(CapabilityRegistryError, match="default_model"):
        load_capability_registry(write_agents_yaml(content))


def test_non_snake_case_id_is_rejected(write_agents_yaml):
    content = VALID_CAPABILITY.replace("id: jira_reporting", "id: JiraReporting")
    with pytest.raises(CapabilityRegistryError, match="snake_case"):
        load_capability_registry(write_agents_yaml(content))


def test_duplicate_ids_are_rejected(write_agents_yaml):
    doubled = VALID_CAPABILITY + VALID_CAPABILITY.replace("capabilities:\n", "")
    with pytest.raises(CapabilityRegistryError, match="duplicate capability ids"):
        load_capability_registry(write_agents_yaml(doubled))


def test_unknown_capability_lookup_lists_what_exists(capabilities):
    with pytest.raises(CapabilityRegistryError, match="jira_reporting"):
        capabilities.get("no_such_capability")


def test_search_ranks_the_relevant_capability_above_an_irrelevant_one(capabilities):
    ranked = capabilities.search_by_capability(
        "predict our next sprint velocity from past sprints", limit=5
    )

    assert ranked, "search must always return a menu"
    ids = [spec.id for spec in ranked]
    assert "jira_reporting" in ids
    # The writeback envelope publishes to Jira; it has no business ranking above
    # the envelope that can actually read the sprint history.
    if "jira_writeback" in ids:
        assert ids.index("jira_reporting") < ids.index("jira_writeback")
