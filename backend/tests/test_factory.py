"""Tool binding is the permission boundary, and the factory is where it lands.

Runs against the real mock MCP server: the tools here were discovered by
calling list_tools() over stdio, not hand-written.
"""

import pytest
from langchain_core.messages import AIMessage

from app.agents.factory import AgentExecutionError, build_worker
from app.agents.models import AgentSpec
from tests.stubs import StubChatModel, scripted_sequence, usage


def spec(**overrides) -> AgentSpec:
    base = {
        "id": "test_agent",
        "role": "Retrieves Jira sprint data and summarizes what it found for a team",
        "system_prompt": "Use your tools.",
        "model": "claude-sonnet-5",
        "tool_selectors": ["jira.*"],
        "max_iterations": 4,
    }
    return AgentSpec.model_validate({**base, **overrides})


async def collect(events: list):
    async def emit(name: str, data: dict) -> None:
        events.append((name, data))

    return emit


def test_glob_selector_binds_only_that_server(tools):
    bound = [tool.name for tool in tools.select(["jira.*"])]

    assert bound == sorted(bound), "selection is returned in a stable order"
    assert set(bound) == {
        "jira.get_sprints",
        "jira.get_sprint_issues",
        "jira.get_team_capacity",
        "jira.publish_sprint_plan",
    }
    assert all(name.startswith("jira.") for name in bound)


def test_exact_selector_binds_one_tool(tools):
    assert [tool.name for tool in tools.select(["jira.get_sprints"])] == ["jira.get_sprints"]


def test_empty_selectors_bind_nothing(tools):
    """Empty means no tools -- deliberately not "all tools"."""
    assert tools.select([]) == []


def test_a_selector_matching_nothing_binds_nothing(tools):
    assert tools.select(["salesforce.*"]) == []


def test_read_only_agent_cannot_reach_the_write_tool(tools):
    """The publisher's write tool must not leak into a read-only agent."""
    read_only = spec(tool_selectors=["jira.get_*"])
    bound = [tool.name for tool in tools.select(read_only.tool_selectors)]

    assert "jira.publish_sprint_plan" not in bound
    assert len(bound) == 3


async def test_worker_calls_a_bound_tool_and_returns_its_answer(tools):
    events: list = []
    model = StubChatModel(responder=scripted_sequence([
        AIMessage(content="", usage_metadata=usage(),
                  tool_calls=[{"name": "get_sprints", "args": {"limit": 3}, "id": "c1"}]),
        AIMessage(content="Completed points were 39, 35 and 41.", usage_metadata=usage()),
    ]))
    worker = build_worker(spec(), tools.select(["jira.*"]), model, await collect(events))

    result = await worker("t1", "Get the last three sprints.")

    assert result.output == "Completed points were 39, 35 and 41."
    assert result.tool_calls == [
        {"tool": "jira.get_sprints", "arguments": {"limit": 3}, "ok": True}
    ]
    assert [name for name, _ in events] == ["tool.called", "tool.result"]
    assert events[1][1]["ok"] is True
    # Fixture data reached the model, so the round trip really happened.
    assert "SPR-" in events[1][1]["preview"]


async def test_worker_costs_every_call_it_makes(tools):
    model = StubChatModel(responder=scripted_sequence([
        AIMessage(content="", usage_metadata=usage(1000, 100),
                  tool_calls=[{"name": "get_sprints", "args": {}, "id": "c1"}]),
        AIMessage(content="Done.", usage_metadata=usage(2000, 200)),
    ]))
    worker = build_worker(spec(), tools.select(["jira.*"]), model, await collect([]))

    result = await worker("t1", "Summarize sprints.")

    assert result.usage.calls == 2
    assert result.usage.input_tokens == 3000
    assert result.usage.output_tokens == 300
    # claude-sonnet-5 is $2/$10 per Mtok: 3000*2/1e6 + 300*10/1e6
    assert result.usage.cost_usd == pytest.approx(0.009)


async def test_worker_reports_an_unavailable_tool_back_to_the_model(tools):
    """A model that names an unbound tool is corrected, not crashed."""
    events: list = []
    model = StubChatModel(responder=scripted_sequence([
        AIMessage(content="", usage_metadata=usage(),
                  tool_calls=[{"name": "delete_everything", "args": {}, "id": "c1"}]),
        AIMessage(content="Understood, I will not do that.", usage_metadata=usage()),
    ]))
    worker = build_worker(spec(), tools.select(["jira.get_sprints"]), model, await collect(events))

    result = await worker("t1", "Delete the board.")

    assert result.output == "Understood, I will not do that."
    assert result.tool_calls == [], "a rejected call is not recorded as a real call"
    assert events[-1][1]["ok"] is False


async def test_worker_stops_at_max_iterations(tools):
    """A model that loops forever must fail loudly, naming the knob to turn."""
    model = StubChatModel(responder=scripted_sequence([
        AIMessage(content="", usage_metadata=usage(),
                  tool_calls=[{"name": "get_sprints", "args": {}, "id": "c1"}]),
    ]))
    worker = build_worker(spec(max_iterations=2), tools.select(["jira.*"]), model, await collect([]))

    with pytest.raises(AgentExecutionError, match="max_iterations"):
        await worker("t1", "Loop forever.")


async def test_an_agent_with_no_tools_still_answers(tools):
    model = StubChatModel(responder=scripted_sequence([
        AIMessage(content="Three risks: ...", usage_metadata=usage()),
    ]))
    worker = build_worker(spec(tool_selectors=[]), [], model, await collect([]))

    result = await worker("t1", "Review this plan.")

    assert result.output == "Three risks: ..."
    assert result.tool_calls == []
