"""Turns the workers' task outputs into one answer for the user.

Makes a single LLM call over the goal and every task's output, and produces both
the user-facing answer and a short summary that is embedded into semantic memory
for future planning.

It does not call tools or re-derive figures. If the workers did not retrieve
something, the synthesizer must say so rather than fill the gap.
"""

from typing import Any, Awaitable, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.costs import Usage, price_response

SYNTHESIZER_SYSTEM_PROMPT = """\
You write the final answer for a user whose goal was handled by several \
specialist agents. You are given the goal and each agent's output.

Rules:
1. Answer the goal directly, in the first sentence. Lead with the number or the \
conclusion the user asked for.
2. Use ONLY figures that appear in the agent outputs. You have no tools and no \
data of your own. If something was not retrieved, say it was not retrieved.
3. Show which agent contributed what, briefly, so the answer is auditable.
4. Carry forward stated caveats and risks. If a task failed or a human declined \
it, say so plainly and explain what that leaves missing.
5. Be concise. Markdown, a few short sections at most. No preamble about being \
an AI or about the process of synthesis."""

SUMMARY_SYSTEM_PROMPT = """\
Summarize, in at most two sentences, what this run was asked for and what it \
concluded. This is stored as memory and shown to a planner deciding how to \
handle a similar goal later, so favour the shape of the answer and the approach \
over the exact numbers."""


class SynthesisError(RuntimeError):
    """Raised when the synthesizer cannot produce a final answer."""


def render_task_outputs(task_outputs: dict[str, dict[str, Any]], plan: list[dict[str, Any]]) -> str:
    """Lay the outputs out in plan order, so the answer reads in dependency order."""
    ordered_ids = [task["task_id"] for task in plan if task["task_id"] in task_outputs]
    ordered_ids += [task_id for task_id in task_outputs if task_id not in ordered_ids]

    if not ordered_ids:
        return "(no agent produced any output)"

    blocks = []
    for task_id in ordered_ids:
        result = task_outputs[task_id]
        blocks.append(
            f"--- task {task_id!r} | agent {result.get('agent_id')} | "
            f"status {result.get('status')} ---\n{result.get('output', '')}"
        )
    return "\n\n".join(blocks)


async def synthesize_answer(
    goal: str,
    plan: list[dict[str, Any]],
    task_outputs: dict[str, dict[str, Any]],
    model: BaseChatModel,
    model_id: str,
) -> tuple[str, str, Usage]:
    """Return the user-facing answer, a memory summary, and what both cost."""
    rendered = render_task_outputs(task_outputs, plan)
    answer_response = await model.ainvoke([
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
        HumanMessage(content=f"USER GOAL:\n{goal}\n\nAGENT OUTPUTS:\n{rendered}"),
    ])
    usage = price_response(model_id, getattr(answer_response, "usage_metadata", None))
    answer = _text_of(answer_response)
    if not answer:
        raise SynthesisError("Synthesizer returned an empty answer.")

    summary_response = await model.ainvoke([
        SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
        HumanMessage(content=f"GOAL:\n{goal}\n\nFINAL ANSWER:\n{answer}"),
    ])
    usage = usage + price_response(model_id, getattr(summary_response, "usage_metadata", None))
    return answer, _text_of(summary_response) or answer[:400], usage


def build_synthesizer_node(
    model_builder: Callable[[str], BaseChatModel], model_id: str
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Return the LangGraph node that writes the final answer."""

    async def run_synthesizer(state: dict[str, Any]) -> dict[str, Any]:
        answer, summary, usage = await synthesize_answer(
            state["goal"],
            state.get("plan", []),
            state.get("task_outputs", {}),
            model_builder(model_id),
            model_id,
        )
        return {
            "final_answer": answer,
            "usage_records": [{"task_id": "synthesizer", "agent_id": "synthesizer",
                               "summary": summary, **usage.model_dump()}],
        }

    return run_synthesizer


def _text_of(message: Any) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(part for part in parts if part).strip()
