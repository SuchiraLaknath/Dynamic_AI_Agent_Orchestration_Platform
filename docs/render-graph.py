"""Renders the compiled LangGraph to docs/graph.mmd and docs/graph.png.

Every other diagram in this repository is drawn by hand, which means it can
drift out of step with the code -- docs/architecture.svg did exactly that. This
one is generated from `compiled.get_graph()`, so it shows the topology the
application actually compiled and cannot describe a graph that does not exist.

The graph is built with stub registries and no checkpointer, so this needs no
database, no MCP server and no API key. Only the shape is being read; nothing is
executed.

Run:  .venv/bin/python docs/render-graph.py
"""

import sys
import urllib.error
from pathlib import Path

DOCS = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS.parent / "backend"))

from app.agents.models import CapabilitySpec  # noqa: E402
from app.agents.registry import CapabilityRegistry  # noqa: E402
from app.graph.build import build_orchestration_graph  # noqa: E402
from app.mcp.registry import ToolRegistry  # noqa: E402
from app.settings import get_settings  # noqa: E402

MERMAID_PATH = DOCS / "graph.mmd"
PNG_PATH = DOCS / "graph.png"

# One placeholder envelope. The graph's shape does not depend on the registry
# contents -- it has four nodes whatever is configured -- but the builder needs
# something well-formed to close over.
_PLACEHOLDER = CapabilitySpec(
    id="placeholder",
    description="Stands in for a real capability so the graph can be compiled and drawn",
    allowed_tool_selectors=[],
    allowed_models=["claude-sonnet-5"],
    default_model="claude-sonnet-5",
)


def build_graph_for_drawing():
    """Compile the real orchestration graph with nothing wired behind it."""

    async def no_memory(_goal: str) -> list:
        return []

    async def ignore_plan(_run_id: str, _plan: list) -> None:
        return None

    async def ignore_event(*_args, **_kwargs) -> None:
        return None

    return build_orchestration_graph(
        capabilities=CapabilityRegistry([_PLACEHOLDER]),
        tools=ToolRegistry({}),
        settings=get_settings(),
        emit=ignore_event,
        find_similar_runs=no_memory,
        record_plan=ignore_plan,
        checkpointer=None,
        chat_model_builder=lambda model_id: None,
    )


def write_mermaid(compiled) -> str:
    """Write the Mermaid source. This is the artifact that cannot go stale."""
    source = compiled.get_graph().draw_mermaid()
    MERMAID_PATH.write_text(source)
    return source


def write_png(compiled) -> bool:
    """Render a PNG for the PDF, where Mermaid source is not enough.

    Returns False rather than raising when the renderer is unreachable: the
    Mermaid source is committed and GitHub renders it natively, so a missing
    image degrades the docs slightly instead of breaking the build.
    """
    try:
        PNG_PATH.write_bytes(compiled.get_graph().draw_mermaid_png())
        return True
    except (urllib.error.URLError, OSError, ValueError) as error:
        print(f"  PNG not rendered ({type(error).__name__}: {error}).")
        print(f"  {MERMAID_PATH.name} is still written; render it manually if the image is needed.")
        return False


if __name__ == "__main__":
    graph = build_graph_for_drawing()
    mermaid = write_mermaid(graph)
    nodes = sorted(graph.get_graph().nodes)

    print(f"Wrote {MERMAID_PATH.relative_to(DOCS.parent)} ({len(mermaid.splitlines())} lines)")
    print(f"  nodes compiled: {', '.join(nodes)}")
    if write_png(graph):
        print(f"Wrote {PNG_PATH.relative_to(DOCS.parent)} ({PNG_PATH.stat().st_size // 1024} KB)")
