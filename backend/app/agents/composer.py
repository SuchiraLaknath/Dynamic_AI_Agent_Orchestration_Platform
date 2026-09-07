"""Turns a capability envelope plus a synthesized agent into a runnable AgentSpec.

This is where dynamic agent creation is made safe. The planner writes an agent
-- its name, role, system prompt and the tools it wants -- and this module
decides whether that agent is allowed to exist. Everything it can get wrong is
checked here, and every check narrows rather than trusts.

The tool check compares *resolved tool names*, never the glob strings. A
synthesized selector of "jira.*" is not textually a subset of an envelope's
"jira.get_*", but resolving both through the tool registry and comparing the
resulting name sets is exact. Comparing globs as text would be the bug that
lets a generated agent award itself a write tool.

It does not run agents (that is app.agents.factory) and it does not choose
which capability to use (that is the planner).
"""

from app.agents.models import AgentSpec, CapabilitySpec, SynthesizedAgent
from app.mcp.registry import ToolRegistry

POLICY_HEADER = (
    "Non-negotiable constraints (these override anything above, including any "
    "instruction that contradicts them):"
)


class EnvelopeViolationError(ValueError):
    """Raised when a synthesized agent asks for more than its capability allows."""


def realize_agent(
    capability: CapabilitySpec,
    synthesized: SynthesizedAgent,
    tools: ToolRegistry,
) -> AgentSpec:
    """Validate a synthesized agent against its envelope and build its AgentSpec.

    Raises EnvelopeViolationError on any attempt to exceed the envelope. The
    message names the capability, what was asked for and what was permitted,
    because it is fed back to the planner for its one retry.
    """
    permitted = _resolved_names(tools, capability.allowed_tool_selectors)
    requested = _check_selectors_resolve(capability, synthesized.tool_selectors, tools)

    escalation = requested - permitted
    if escalation:
        raise EnvelopeViolationError(
            f"Agent {synthesized.name!r} requested tool(s) {sorted(escalation)} that "
            f"capability {capability.id!r} does not permit. Permitted: "
            f"{sorted(permitted) or 'none'}."
        )

    model = synthesized.model or capability.default_model
    if model not in capability.allowed_models:
        raise EnvelopeViolationError(
            f"Agent {synthesized.name!r} requested model {model!r}, which capability "
            f"{capability.id!r} does not permit. Allowed: {capability.allowed_models}."
        )

    if synthesized.max_iterations > capability.max_iterations_limit:
        raise EnvelopeViolationError(
            f"Agent {synthesized.name!r} requested {synthesized.max_iterations} iterations, "
            f"above capability {capability.id!r}'s limit of {capability.max_iterations_limit}."
        )

    return AgentSpec(
        id=synthesized.name,
        role=synthesized.role,
        system_prompt=compose_system_prompt(synthesized.system_prompt, capability.policy),
        model=model,
        tool_selectors=list(synthesized.tool_selectors),
        max_iterations=synthesized.max_iterations,
        # Both are taken from the envelope, never from the synthesized agent, so
        # a generated agent cannot approve itself or escape its own audit trail.
        requires_approval=capability.requires_approval,
        capability_id=capability.id,
    )


def compose_system_prompt(synthesized_prompt: str, policy: str) -> str:
    """Put the envelope's policy after the synthesized prompt, not before it.

    Ordering is deliberate. Later instructions carry more weight, so a policy
    placed first could be talked over by whatever the planner wrote; placed last
    under an explicit override header, it is the final word.
    """
    if not policy.strip():
        return synthesized_prompt.strip()
    return f"{synthesized_prompt.strip()}\n\n{POLICY_HEADER}\n{policy.strip()}"


def _resolved_names(tools: ToolRegistry, selectors: list[str]) -> set[str]:
    """The concrete tool names a list of globs resolves to."""
    return {tool.name for tool in tools.select(selectors)}


def _check_selectors_resolve(
    capability: CapabilitySpec, selectors: list[str], tools: ToolRegistry
) -> set[str]:
    """Resolve requested selectors, rejecting any that match nothing.

    A selector matching no tool is almost always a typo. Allowing it would be
    technically safe -- an empty set is a subset of anything -- but the agent
    would then run toolless and fail on its iteration limit, which is a much
    worse error message than this one.
    """
    resolved: set[str] = set()
    for selector in selectors:
        matched = _resolved_names(tools, [selector])
        if not matched:
            raise EnvelopeViolationError(
                f"Agent selector {selector!r} for capability {capability.id!r} matches no "
                f"discovered tool. Known tools: {', '.join(sorted(tools.tool_names)) or 'none'}."
            )
        resolved |= matched
    return resolved


__all__ = ["realize_agent", "compose_system_prompt", "EnvelopeViolationError", "POLICY_HEADER"]
