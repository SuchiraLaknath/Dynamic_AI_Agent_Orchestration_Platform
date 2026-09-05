# Architecture

![Architecture](architecture.svg)

The one-sentence version: **a goal is planned into a task DAG by an LLM that may
only choose from a retrieved menu of YAML-defined agents; each task runs an agent
built at runtime and equipped with exactly the MCP tools its selectors match; the
whole thing streams a trace.**

---

## 1. The idea the design is organised around

A monolithic agent — one big prompt with every tool bolted on — fails in a
specific way: adding a capability means editing a prompt, the tool list grows
until selection degrades, and there is no point at which you can say *why* a
particular tool was used. This project inverts that.

**Agents are data.** There is no `JiraAgent` class anywhere in the codebase.
An agent is a record in [`backend/config/agents.yaml`](../backend/config/agents.yaml):

```yaml
- id: velocity_forecaster
  role: >-
    Predicts and forecasts sprint velocity: the story points a team is likely
    to complete in the next upcoming sprint, computed as a rolling average...
  system_prompt: |
    You forecast sprint velocity from completed story points...
  model: claude-sonnet-5
  tool_selectors: ["jira.get_sprints"]
  max_iterations: 4
  requires_approval: false
```

`app/agents/factory.py` turns that record plus a list of tools into a coroutine.
The test that matters here is a negative one: **you can add a specialist agent to
this platform without writing any Python.** Adding an integration is likewise an
edit to `mcp_servers.yaml`.

---

## 2. Agent architecture and routing strategy

### Why a planner rather than a router

A classifier that maps a goal to one agent cannot express *"fetch the history,
then forecast and check capacity from it in parallel, then review the result."*
That is a DAG, not a label. So the routing decision produces a plan.

### The routing pipeline

1. **Embed the goal.** `app/embeddings.py`.
2. **Retrieve a menu.** Top-k agents ranked by cosine similarity between the goal
   and each agent's `role`, top-k tools by their MCP-advertised description, and
   the top-k most similar *completed* past runs from pgvector.
3. **One structured-output call.** The planner LLM returns a `TaskPlan`:
   a list of `{task_id, agent_id, objective, depends_on[]}`.
4. **Validate against the live registry.** `validate_plan` in
   `app/graph/state.py` rejects an unknown `agent_id`, a dangling or self
   dependency, a duplicate task id, and any dependency cycle.
5. **On rejection, retry exactly once**, feeding the validation error back into
   the conversation so the model can see which id it invented. A second failure
   fails the run loudly.

The rule that keeps this honest: **the planner picks from a menu, it never
invents.** An id that is not in the registry cannot reach execution. That is
tested in `tests/test_planner.py`.

### Execution: layered fan-out

The compiled graph has **four nodes regardless of how many agents run**:

```
START → planner → dispatcher ⇄ worker(s) → synthesizer → END
```

`dispatcher` is the scheduler. After the planner and after *every* worker
superstep, `route_ready_tasks` recomputes the set of tasks whose dependencies are
all satisfied and returns one LangGraph `Send` per task. Independent tasks
therefore fan out concurrently, and when nothing is left it routes to the
synthesizer instead.

The consequence worth stating out loud: **parallelism is a property of the plan,
not of the graph topology.** A plan with four independent tasks runs four workers
in one superstep without any change to the graph.

### Tool permissions

`tool_selectors` are fnmatch globs over namespaced `<server>.<tool>` names.
`ToolRegistry.select` resolves them, and the worker binds *only* those tools to
the model. An empty list means no tools — deliberately different from meaning
all of them. This is the role-based tool permission boundary, and it is why
`risk_reviewer` physically cannot call Jira and `jira_analyst` physically cannot
reach the write tool.

---

## 3. LangGraph and LangChain — what each is actually used for

| Library | Used for | Not used for |
|---|---|---|
| **LangGraph** | The graph runtime: `Send` fan-out, the Postgres checkpointer, and `interrupt()` for human approval | Anything prebuilt. `create_react_agent` is deliberately not used |
| **LangChain** | Two narrow things: `langchain-mcp-adapters` to turn MCP tools into bindable tools, and `ChatAnthropic` as the model client | Chains, agents, memory, retrievers |

Three LangGraph features carry real weight and are the reason it was chosen over
hand-rolling an executor:

- **`Send`** gives dynamic map-reduce fan-out. Without it, a runtime-decided
  number of parallel branches means writing a scheduler.
- **`interrupt()` + checkpointer** makes human approval *durable*. The run pauses
  in Postgres, not in memory, so the process can restart and the run is still
  resumable. This is the difference between an approval gate and a modal dialog.
- **Reducers on state** make concurrent worker writes safe by construction.

**The worker's tool loop is hand-written** (`app/agents/factory.py`) rather than
delegated to a prebuilt ReAct agent, for three reasons: `max_iterations` is
enforced per agent spec, every tool call emits `tool.called`/`tool.result` for the
trace, and token usage is attributed per call. A prebuilt agent hides all three
inside a subgraph.

### LangFlow: considered and rejected

LangFlow's flow definition lives in GUI state. This project's entire premise is
that routing is defined by version-controlled configuration and decided at
runtime by a planner. A GUI-authored static flow contradicts both halves. It
would also make the "add an agent without writing code" property *worse*, not
better — a YAML diff reviews; a serialized canvas does not.

### Redis: considered and rejected

The event bus is in-process. A single API process has one publisher and a handful
of SSE subscribers, so there is nothing for a broker to do. Adding Redis would
put a box on this diagram that nothing justifies. The cost is stated plainly in
the limitations: the live bus does not fan out across replicas. The durable copy
of every trace is in Postgres, which is what makes that acceptable.

---

## 4. MCP integration

**Discovery is at startup, and schemas are never hardcoded.** `McpManager` reads
`mcp_servers.yaml`, opens a session per server, and calls `list_tools()`. Whatever
comes back *is* the platform's tool surface. `GET /tools` returns exactly that,
which is the evidence.

### Sessions are held open, each inside its own task

This is the one genuinely subtle piece of the backend and it is worth being able
to explain. A stdio MCP server is a subprocess; reconnecting per tool call would
pay process-spawn cost on every use. But the MCP stdio client is built on anyio
task groups, and **an anyio cancel scope must be exited by the task that entered
it** — while a web server runs lifespan startup and lifespan shutdown in
*different* tasks. Opening the session at startup and closing it at shutdown
therefore fails with `attempted to exit cancel scope in a different task`.

The fix is to give each session a dedicated task that opens it, publishes its
tools, waits on a stop event, and closes it — both ends in one task.

### Policy

`call_tool_with_policy` applies a configurable timeout and bounded retries with
exponential backoff. Retries are for *transport* failures and timeouts only. A
tool that returns an error result has answered the question, so retrying it just
spends the same money for the same answer; the result goes back to the agent,
which decides what to do about it.

Failure is graded, not binary:

| Failure | Consequence |
|---|---|
| A server won't start | Reported; its tools are absent; the rest of the platform runs |
| A tool call exhausts retries | Returned to the model as an error `ToolMessage`; the agent can recover |
| An agent exceeds `max_iterations` | Task marked `failed`; siblings keep their work; the synthesizer reports the gap |
| The planner can't produce a valid plan | The run fails loudly — there is nothing to degrade to |

### Swapping the mock for real Jira

`mcp_servers/jira_mock/` exists because the assignment has no Jira credentials.
Replacing it is a config edit — the commented block in `mcp_servers.yaml` shows
the shape. No application code refers to Jira by name.

---

## 5. Memory design and persistence

One Postgres instance, three tiers with genuinely different jobs.

| Tier | Store | Written | Read | Job |
|---|---|---|---|---|
| **Working** | LangGraph Postgres checkpointer (`thread_id = run_id`) | Every superstep | On resume | Resume a run paused at an approval, across a restart |
| **Episodic** | `runs` + `steps` tables | As each event is published | `GET /runs/{id}` | Trace replay and cost history after the in-process bus has forgotten |
| **Semantic** | `goal_embedding vector(384)` on `runs` | At run start | At plan time | Inject similar completed runs into the planner prompt |

Every event goes to **both** the SSE bus and the `steps` table in the same call
(`build_event_recorder`). A trace that exists in one but not the other is worse
than having neither.

### The embedding model, stated honestly

`app/embeddings.py` is a **signed hashing vectorizer over word and character
n-grams with sublinear term frequency**. It is lexical, not neural: it captures
shared vocabulary and morphology, not paraphrase. Two reasons that is the right
call *here*:

1. **At this scale, retrieval is ranking, not recall.** With five agents and
   `top_k = 5`, the menu is the whole registry either way. The retrieval path
   exists so the design scales to hundreds of agents.
2. **It has no dependencies and no network call**, so `docker compose up` works
   offline and tests are deterministic.

It does measurably separate relevant from irrelevant — an unrelated role scores
*negative* against a sprint goal — but it ranks closely-related agents roughly.
`embed_text` is the single seam to swap for a hosted embedding model.

One consequence is worth internalising because it is a real lesson of the design:
**`role` text is retrieval surface, not documentation.** `velocity_forecaster`
originally ranked fourth for a velocity goal because its role never used the word
"velocity". Fixing the *config* fixed the routing.

---

## 6. Cost tracking

Every LLM response's `usage_metadata` is priced against a table of per-MTok rates
in `app/costs.py` and accumulated in graph state as one record per call, tagged
with the task and agent that spent it. An unpriced model is recorded in
`unpriced_models` rather than silently costed at zero. Per-run totals land on the
`runs` row and in the `run.completed` event.

---

## 7. The event contract

Seven names, fixed, and nothing else:

```
run.started   task.started   tool.called   tool.result
approval.required   run.completed   run.failed
```

**There is deliberately no `task.completed`.** The frontend therefore *derives*
per-task completion (`frontend/src/hooks/runState.js`): a task is done when a
task that depends on it starts — which is sound because the dispatcher only
releases a task once all its dependencies have returned — or when `run.completed`
arrives carrying each task's final status.

This is a real trade-off and worth naming: it keeps the wire contract minimal and
the client slightly cleverer. An eighth event name would make the client dumber.
If this contract were reopened, `task.completed` is the change I would argue for.

---

## 8. Notable decisions

**Failure inside a run is graded rather than fatal.** A worker that fails records
a failed task; siblings keep their work and the synthesizer is instructed to
report what is missing rather than paper over it. A partial answer that says what
it is missing beats a 500.

**Approval interrupts are detected in the orchestrator, not the worker.** A node
that emitted its own `approval.required` would emit it a second time, because
LangGraph replays the node when the graph resumes.

**The synthesizer has no tools, by design.** It can only use figures the workers
retrieved, and is instructed to say when something was not retrieved. Giving it
tools would let it quietly paper over a failed task.

**Modules that aren't in the assignment's suggested layout.** Four small ones
were added rather than growing others past the ~200-line limit:
`embeddings.py`, `costs.py`, `chat_models.py` (a single-seam model constructor),
and `orchestrator.py` (the run driver, which is genuinely its own job and would
otherwise bloat the route handlers).

---

## 9. What I would change for production

| Area | Now | Production |
|---|---|---|
| Embeddings | Lexical hashing vectorizer | A hosted embedding model behind `embed_text` |
| Event bus | In-process | Postgres `LISTEN/NOTIFY` before reaching for a broker |
| Schema | `create_all` at boot | Alembic migrations |
| Auth | None | Per-tenant auth; MCP credentials scoped per user, not per process |
| Planner quality | Untested | A labelled goal→plan eval set, scored on agent selection |
| Approval | Any caller can approve | Approver identity recorded and authorised |
| Forecasting | Rolling average | An actual model — and the honest note that the rolling average is a baseline, not a forecast |
