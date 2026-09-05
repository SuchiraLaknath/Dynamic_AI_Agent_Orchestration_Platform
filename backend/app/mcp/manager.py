"""Opens MCP sessions, discovers tools, and enforces timeout and retry policy.

Reads mcp_servers.yaml, holds one long-lived session per configured server for
the process lifetime, and calls `list_tools()` on each to discover what exists.
Tool schemas are never written in Python -- whatever a server advertises is what
the platform has.

It does not decide which agent gets which tool (that is app.mcp.registry plus
each agent's selectors) and it does not run agents.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

import yaml
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from app.settings import REPO_ROOT, get_settings


class McpConfigError(ValueError):
    """Raised when mcp_servers.yaml is missing or malformed."""


class McpToolCallError(RuntimeError):
    """Raised when a tool call exhausts its retries or times out."""


def load_server_connections(path: Path) -> dict[str, dict[str, Any]]:
    """Turn mcp_servers.yaml into the connection dict the MCP client expects.

    Relative stdio paths are resolved against the repository root so the same
    config works whether uvicorn runs from backend/ or the container's /srv.
    """
    if not path.exists():
        raise McpConfigError(f"MCP server registry not found at {path}")

    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict) or not isinstance(document.get("servers"), list):
        raise McpConfigError(f"{path} must be a mapping with a top-level 'servers' list")

    connections: dict[str, dict[str, Any]] = {}
    for record in document["servers"]:
        name = record.get("name")
        transport = record.get("transport")
        if not name or not transport:
            raise McpConfigError(f"{path}: every server needs a 'name' and a 'transport'")
        if name in connections:
            raise McpConfigError(f"{path}: duplicate server name {name!r}")

        if transport == "stdio":
            args = [str((REPO_ROOT / argument).resolve()) if argument.endswith(".py") else argument
                    for argument in record.get("args", [])]
            connections[name] = {
                "transport": "stdio",
                "command": _resolve_interpreter(record.get("command", "python")),
                "args": args,
                "env": record.get("env") or None,
            }
        else:
            connections[name] = {
                "transport": transport,
                "url": record["url"],
                "headers": record.get("headers") or None,
            }
    return connections


def _resolve_interpreter(command: str) -> str:
    """Run a Python stdio server on the interpreter that is running us.

    A bare "python" in the config resolves against PATH, which outside a
    container is usually a different interpreter than the one holding the
    backend's dependencies -- the server then fails to import `mcp` and the
    integration silently has no tools.
    """
    if command in {"python", "python3"}:
        return sys.executable
    return command


def _describe_failure(error: BaseException) -> str:
    """Flatten an ExceptionGroup so the operator sees the real cause.

    The MCP stdio client runs inside a task group, so a failed server start
    surfaces as "unhandled errors in a TaskGroup" with the actual ImportError
    or ENOENT nested inside it.
    """
    if isinstance(error, BaseExceptionGroup):
        return "; ".join(_describe_failure(inner) for inner in error.exceptions)
    return f"{type(error).__name__}: {error}"


class McpManager:
    """Owns MCP sessions for the process lifetime.

    Sessions are opened once at startup rather than per tool call: a stdio
    server is a subprocess, and reconnecting per call would pay process spawn
    cost on every tool use.

    Each session is held open inside its own dedicated task, which is the part
    that is not obvious. The MCP stdio client is built on anyio task groups, and
    an anyio cancel scope must be exited by the task that entered it -- but a
    web server runs lifespan startup and lifespan shutdown in *different* tasks.
    Entering the session in startup and closing it in shutdown therefore fails
    with "attempted to exit cancel scope in a different task". Giving each
    session a task that opens it, waits for a stop signal, and closes it keeps
    both ends in one task.
    """

    def __init__(self, connections: dict[str, dict[str, Any]]) -> None:
        self._connections = connections
        self._client = MultiServerMCPClient(connections)
        self._stop = asyncio.Event()
        self._holders: list[asyncio.Task[None]] = []
        self._tools_by_server: dict[str, list[BaseTool]] = {}

    @property
    def server_names(self) -> list[str]:
        return list(self._connections)

    async def discover_tools(self) -> dict[str, list[BaseTool]]:
        """Open a session per server and list what each one offers.

        A server that fails to start is reported and skipped, not fatal: one
        broken integration should not take the whole platform down, and the
        planner simply never sees its tools.
        """
        ready: dict[str, asyncio.Future[list[BaseTool]]] = {}
        loop = asyncio.get_running_loop()

        for name in self._connections:
            ready[name] = loop.create_future()
            self._holders.append(asyncio.create_task(self._hold_session(name, ready[name])))

        for name, future in ready.items():
            try:
                self._tools_by_server[name] = await future
            except Exception as error:  # noqa: BLE001 - reported, then degraded past
                self._tools_by_server[name] = []
                print(
                    f"[mcp] server {name!r} failed to start, its tools are unavailable: "
                    f"{_describe_failure(error)}"
                )
        return dict(self._tools_by_server)

    async def _hold_session(self, name: str, ready: asyncio.Future[list[BaseTool]]) -> None:
        """Open one server's session, publish its tools, and hold it until shutdown."""
        try:
            async with self._client.session(name) as session:
                tools = await load_mcp_tools(session, server_name=name)
                ready.set_result(tools)
                await self._stop.wait()
        except Exception as error:  # noqa: BLE001 - handed to whoever awaits `ready`
            if not ready.done():
                ready.set_exception(error)

    async def aclose(self) -> None:
        """Signal every session task to close, and wait for them to finish."""
        self._stop.set()
        if self._holders:
            await asyncio.gather(*self._holders, return_exceptions=True)
        self._holders.clear()
        self._tools_by_server.clear()


async def call_tool_with_policy(tool: BaseTool, arguments: dict[str, Any]) -> Any:
    """Invoke one MCP tool under the configured timeout and retry policy.

    Retries are for transport-level failures and timeouts. A tool that returns
    an error result has answered the question -- retrying it just spends the
    same money to get the same answer -- so the result is handed back as-is and
    the agent decides what to do about it.

    A free function rather than a BaseTool subclass, so the policy is visible at
    the call site instead of hidden inside the tool object.
    """
    settings = get_settings()
    last_error: Exception | None = None

    for attempt in range(1, settings.mcp_tool_max_attempts + 1):
        try:
            return await asyncio.wait_for(
                tool.ainvoke(arguments), timeout=settings.mcp_tool_timeout_seconds
            )
        except asyncio.TimeoutError as error:
            last_error = error
            message = f"timed out after {settings.mcp_tool_timeout_seconds}s"
        except Exception as error:  # noqa: BLE001 - re-raised below as a named error
            last_error = error
            message = repr(error)

        if attempt < settings.mcp_tool_max_attempts:
            await asyncio.sleep(0.5 * 2 ** (attempt - 1))

    raise McpToolCallError(
        f"Tool {tool.name!r} failed after {settings.mcp_tool_max_attempts} attempts: {message}"
    ) from last_error
