"""The index of discovered MCP tools: namespacing, selector matching, search.

Holds every tool the manager discovered under a `<server>.<tool>` name, answers
"which tools does this agent's selectors bind to", and ranks tools against a
goal for the planner's menu.

It does not open sessions or call tools -- that is app.mcp.manager.
"""

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any

from langchain_core.tools import BaseTool

from app.embeddings import cosine_similarity, embed_text, embed_texts


class ToolNotFoundError(KeyError):
    """Raised when a namespaced tool name is not in the discovered index."""


@dataclass(frozen=True)
class DiscoveredTool:
    """One tool as advertised by an MCP server, plus where it came from."""

    name: str  # namespaced, e.g. "jira.get_sprints"
    server: str
    description: str
    input_schema: dict[str, Any]
    tool: BaseTool

    def describe(self) -> dict[str, Any]:
        """The JSON-safe view, for the planner prompt and GET /tools."""
        return {
            "name": self.name,
            "server": self.server,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """Namespaced index over discovered tools."""

    def __init__(self, tools_by_server: dict[str, list[BaseTool]]) -> None:
        self._tools: dict[str, DiscoveredTool] = {}
        for server, tools in tools_by_server.items():
            for tool in tools:
                namespaced = f"{server}.{tool.name}"
                self._tools[namespaced] = DiscoveredTool(
                    name=namespaced,
                    server=server,
                    description=(tool.description or "").strip(),
                    input_schema=_input_schema_of(tool),
                    tool=tool,
                )
        # One batched call rather than one per tool, for the same reason as the
        # capability registry: this runs once at startup.
        self._vectors = dict(
            zip(
                self._tools,
                embed_texts(
                    [f"{name} {tool.description}" for name, tool in self._tools.items()]
                ),
            )
        )

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def tool_names(self) -> set[str]:
        """The only tool names a plan or an agent selector may resolve to."""
        return set(self._tools)

    def all_tools(self) -> list[DiscoveredTool]:
        return list(self._tools.values())

    def get(self, name: str) -> DiscoveredTool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(
                f"Unknown tool {name!r}. Discovered tools: {', '.join(sorted(self._tools))}"
            ) from None

    def select(self, selectors: list[str]) -> list[DiscoveredTool]:
        """Resolve an agent's fnmatch selectors to concrete tools.

        This is the permission boundary: an agent is handed exactly what its
        selectors match and cannot reach anything else. An empty selector list
        means no tools, which is different from meaning all of them.
        """
        if not selectors:
            return []
        matched = [
            discovered
            for name, discovered in self._tools.items()
            if any(fnmatch(name, selector) for selector in selectors)
        ]
        return sorted(matched, key=lambda discovered: discovered.name)

    def search(self, goal: str, limit: int) -> list[DiscoveredTool]:
        """Rank tools by similarity to the goal, for the planner's menu."""
        goal_vector = embed_text(goal)
        ranked = sorted(
            self._tools.values(),
            key=lambda discovered: cosine_similarity(goal_vector, self._vectors[discovered.name]),
            reverse=True,
        )
        return ranked[:limit]


def _input_schema_of(tool: BaseTool) -> dict[str, Any]:
    """Read the JSON schema the MCP server advertised for a tool's arguments.

    The adapter exposes it either as a plain dict or as a pydantic model,
    depending on how the server declared the tool.
    """
    schema = tool.args_schema
    if isinstance(schema, dict):
        return schema
    if schema is not None and hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    return {"type": "object", "properties": {}}
