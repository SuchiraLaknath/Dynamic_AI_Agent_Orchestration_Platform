# Dynamic AI Agent Orchestration Platform

Submit a natural-language goal. The platform plans which specialist agents should
handle it, spawns those agents at runtime from a YAML registry, equips each one
with exactly the MCP tools it is allowed to use, runs them as a DAG, and returns a
synthesized answer with a full execution trace.

> **"Predict the velocity for our next sprint based on previous Jira sprints."**
>
> → planner selects `jira_analyst` → `velocity_forecaster` ∥ `capacity_planner`
> → `risk_reviewer`, each equipped from `agents.yaml`, tools discovered from an MCP
> server at startup, streamed back to the browser as it happens.

**The point of the project is that routing is dynamic and configuration-driven.**
Adding a specialist agent, or a whole new integration, is an edit to a YAML file.
There is no `JiraAgent` class in this repository.

![Architecture](docs/architecture.svg)

Full design rationale: **[docs/architecture.md](docs/architecture.md)**.

---

## Quick start

### With Docker (recommended)

```bash
cp .env.example .env
# Put a real key in ANTHROPIC_API_KEY — everything else has working defaults.

docker compose up --build
```

`.env` is passed wholesale to the backend container, so **any setting you add
there reaches the app without editing `docker-compose.yml`**. Two deliberate
exceptions: `DATABASE_URL` is overridden to `postgres:5432` (a localhost URL
would point the container at itself), and the database and frontend containers
get only the two or three values they need rather than the API key. The file is
optional — with no `.env`, everything falls back to the defaults in
`backend/app/settings.py`.

| | |
|---|---|
| UI | http://localhost:5173 |
| API docs (Swagger) | http://localhost:8000/docs |
| Health | http://localhost:8000/health |

The mock Jira MCP server runs as a stdio subprocess of the backend container, so
there is no fourth service to start.

### Without Docker

```bash
docker compose up -d postgres          # Postgres + pgvector, published on 5433

python3.12 -m venv .venv && source .venv/bin/activate
pip install -e "backend[dev]"
cp .env.example .env                   # DATABASE_URL already points at localhost:5433

cd backend && uvicorn app.main:app --reload    # :8000
cd frontend && npm install && npm run dev      # :5173
```

Postgres is published on **5433**, not 5432, so it cannot collide with a Postgres
you may already be running.

### Running the tests

Tests stub the LLM and **need no API key**. They start the real mock MCP server
over stdio, so tool discovery and invocation are genuinely exercised.

```bash
cd backend && pytest          # 47 tests, ~5s
docker compose exec backend pytest      # or inside the container
```

---

## Demonstration scenarios

Each is a one-click example in the UI.

| # | Prompt | What it demonstrates |
|---|---|---|
| 1 | *Predict the velocity for our next sprint based on previous Jira sprints.* | The core scenario: multi-agent plan, MCP tool use, synthesis |
| 2 | *How many story points did we complete in the last three sprints, and what was our completion rate?* | **Small goals get small plans** — one agent, not four |
| 3 | *Two independent things about our next sprint: a velocity forecast from sprint history, and the team capacity available. Then review the combined picture for delivery risk.* | **Parallel fan-out** — the forecast and capacity tasks are dispatched in one superstep, then converge on the reviewer |
| 4 | *Work out our next sprint commitment and publish it to the Jira board.* | **Human approval** — the run pauses before the write tool and waits for you |

Scenario 3 is worth watching in the trace: both branches report `task.started`
before either one's tool call returns. Phrasing matters — ask instead to
*"forecast velocity and **adjust it** for availability"* and the planner correctly
produces a sequential chain, because the second task then genuinely depends on the
first. The plan is a real decision, not a fixed pipeline.

Scenario 4 is the one to try deliberately: `sprint_plan_publisher` is the only
agent that can write to Jira, and it is marked `requires_approval: true`. The
graph interrupts *before* the agent runs and checkpoints to Postgres. Decline it
and the write never happens; the run still completes and says what was skipped.

### Watching the routing change

The interesting demo is not any single run — it is editing
`backend/config/agents.yaml`, restarting, and watching a different plan come out
of the same goal. Comment out `velocity_forecaster` and the planner reroutes.
The "Registry" panel in the UI shows what the planner is choosing from.

---

## API

Interactive docs and the OpenAPI schema are generated at **`/docs`** and
`/openapi.json`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/runs` | Submit a goal; returns `run_id` immediately (202) |
| `GET` | `/runs/{id}` | Final state, plan, answer, cost and durable trace |
| `GET` | `/runs/{id}/events` | SSE trace stream |
| `POST` | `/runs/{id}/approve` | Resume an interrupted run |
| `GET` | `/agents` | The loaded agent registry, selectors resolved to real tools |
| `GET` | `/tools` | Tools discovered from MCP servers at startup |

```bash
RUN=$(curl -s -X POST localhost:8000/runs -H 'content-type: application/json' \
  -d '{"goal":"Predict the velocity for our next sprint."}' | jq -r .run_id)

curl -N localhost:8000/runs/$RUN/events      # live trace
curl -s localhost:8000/runs/$RUN | jq        # final state
```

The SSE stream **replays everything already published** before going live, so
attaching after `POST /runs` — or after the run has already finished — still
renders the full trace.

---

## Adding an agent (the whole point)

Append to `backend/config/agents.yaml` and restart. No Python.

```yaml
  - id: release_notes_writer
    role: >-
      Drafts release notes from the issues completed in a sprint, grouped by
      change type and written for a non-technical audience.
    system_prompt: |
      You draft release notes from completed sprint issues. Group by type.
      Never describe work that is not in the issue list you were given.
    model: claude-sonnet-5
    tool_selectors: ["jira.get_sprint_issues"]
    max_iterations: 4
    requires_approval: false
```

`role` is embedded and matched against the user's goal, so **write it as a
description of capability using the words a user would use**. This is not
cosmetic: `velocity_forecaster` initially ranked poorly for velocity goals
because its role never contained the word "velocity".

Adding an integration is the same kind of edit to
[`backend/config/mcp_servers.yaml`](backend/config/mcp_servers.yaml).

---

## Technology choices

| Choice | Why |
|---|---|
| **FastAPI** | Async, native SSE, and OpenAPI docs for free — a required deliverable |
| **LangGraph** | `Send` for runtime fan-out, `interrupt()` for durable human approval, Postgres checkpointing. Used for the runtime only — no prebuilt agents |
| **LangChain** | Only `langchain-mcp-adapters` and `ChatAnthropic`. No chains, no agents, no memory |
| **Postgres + pgvector** | One store for all three memory tiers. A second datastore would need to earn its place |
| **React + Vite** | Fast dev server, no framework overhead |
| **React Flow** | Renders the run DAG without hand-drawn SVG |
| **Claude** | `claude-opus-5` plans and synthesizes; workers run `claude-sonnet-5` — set per agent in YAML |

**Not used, deliberately:**

- **LangFlow** — its flow definition lives in GUI state, which contradicts the
  premise that routing is version-controlled configuration decided at runtime.
- **Redis** — a single API process has one publisher and a few SSE subscribers.
  There is nothing for a broker to do, and the durable trace is in Postgres.

---

## How it works, briefly

1. **Startup** loads `agents.yaml` (embedding each `role`), opens a session to
   every server in `mcp_servers.yaml` and calls `list_tools()`, prepares Postgres,
   and compiles the graph. Tool schemas are never hardcoded.
2. **Planning** embeds the goal, retrieves a menu of candidate agents, relevant
   tools and similar past runs, then makes one structured-output call returning a
   task DAG. Every `agent_id` is validated against the registry — an invented id
   is retried once with the error fed back, then fails the run.
3. **Execution** dispatches every task whose dependencies are satisfied, in
   parallel, via LangGraph `Send`. The graph has four nodes no matter how many
   agents run; parallelism comes from the plan.
4. **Each worker** binds only the tools its `tool_selectors` match, runs a tool
   loop bounded by `max_iterations`, and emits a trace event per tool call.
5. **Synthesis** produces the answer plus a summary that is embedded for future
   planning.

---

## Assumptions

- **Jira is mocked.** No credentials were available, so
  `mcp_servers/jira_mock/` serves fixture data over real MCP/stdio. Swapping in a
  real server is a config edit; see the commented block in `mcp_servers.yaml`.
- **A rolling average is the forecast.** `velocity_forecaster` computes a mean of
  recent completed points and is instructed to say that is what it is. A real
  forecasting model is out of scope, and dressing a mean up as one would be worse
  than saying so.
- **Single tenant, single process.** No auth, one shared set of MCP sessions.
- **The demo fixture is synthetic** — eight closed sprints with plausible
  committed/completed spreads.

## Known limitations

- **The live event bus is in-process.** It does not fan out across replicas and
  does not survive a restart. The durable trace in `steps` does, and
  `GET /runs/{id}` serves it.
- **Embeddings are lexical, not semantic.** A hashing vectorizer over word and
  character n-grams — good enough to rank a handful of agents and to keep the
  stack dependency-free and offline, but it does not understand paraphrase.
  `embed_text` is the one function to replace. See
  [architecture.md §5](docs/architecture.md#5-memory-design-and-persistence).
- **No authentication**, and any caller can approve a paused run.
- **Schema is created with `create_all`** at boot, not migrations.
- **Planner quality is untested.** There is no eval set scoring whether the
  planner picks the right agents; the tests pin the *contract* (it cannot invent
  an id, it cannot emit a cycle), not the judgement.
- **MCP sessions are process-wide**, so tools cannot carry per-user credentials.
- **No `task.completed` event**, by contract. The client derives per-task
  completion; see [architecture.md §7](docs/architecture.md#7-the-event-contract).

## What I would improve for production

Hosted embeddings behind the existing seam; Postgres `LISTEN/NOTIFY` before
reaching for a broker; Alembic migrations; per-tenant auth with MCP credentials
scoped per user rather than per process; a labelled goal→plan eval set so planner
changes can be measured; and recording *who* approved a sensitive run.

---

## Repository layout

```
backend/
  config/agents.yaml         the agent registry — add agents here
  config/mcp_servers.yaml    the MCP server registry — add integrations here
  app/
    agents/                  AgentSpec model, registry (+ role embedding), factory
    graph/                   state & plan contract, planner, worker, synthesizer, build
    mcp/                     session manager (timeouts, retries), tool index & search
    memory/                  engine, tables, pgvector retrieval
    api/                     routes and response schemas
    orchestrator.py          drives one run: events out, state persisted
    events.py  costs.py  embeddings.py  chat_models.py  settings.py
  tests/                     registry, factory, planner, end-to-end
mcp_servers/jira_mock/       mock Jira MCP server over stdio + fixtures
frontend/src/                api/client.js, hooks/, components/
docs/architecture.md         design rationale
```
