/**
 * The page: goal in, plan and trace out.
 *
 * Holds no run logic -- useRunStream owns everything derived from the event
 * stream. This component only lays the pieces out and shows the registry the
 * planner is choosing from, which is the visible evidence that routing is
 * configuration-driven.
 */

import { useEffect, useState } from 'react'
import { listAgents, listTools } from './api/client'
import { useRunStream } from './hooks/useRunStream'
import ApprovalDialog from './components/ApprovalDialog'
import CostPanel from './components/CostPanel'
import GoalInput from './components/GoalInput'
import Markdown from './components/Markdown'
import RunGraph from './components/RunGraph'
import TraceTimeline from './components/TraceTimeline'

const BUSY_STATUSES = new Set(['starting', 'running'])

function Registry({ agents, tools }) {
  const [open, setOpen] = useState(false)

  return (
    <section className="panel registry">
      <button type="button" className="registry__toggle" onClick={() => setOpen(!open)}>
        {open ? '▾' : '▸'} Registry: {agents.length} agents, {tools.length} MCP tools
      </button>
      {open && (
        <div className="registry__body">
          <p className="muted small">
            Loaded from <code>backend/config/agents.yaml</code> and discovered by calling{' '}
            <code>list_tools()</code> on each server in <code>mcp_servers.yaml</code> at startup.
          </p>
          <table>
            <thead>
              <tr>
                <th>Agent</th>
                <th>Can do</th>
                <th>Tools bound</th>
              </tr>
            </thead>
            <tbody>
              {agents.map((agent) => (
                <tr key={agent.id}>
                  <td>
                    <code>{agent.id}</code>
                    {agent.requires_approval && <span className="tag tag--warn">approval</span>}
                  </td>
                  <td className="registry__role">{agent.role}</td>
                  <td>
                    {agent.bound_tools.length ? (
                      agent.bound_tools.map((tool) => (
                        <div key={tool}>
                          <code>{tool}</code>
                        </div>
                      ))
                    ) : (
                      <span className="muted">none</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

export default function App() {
  const run = useRunStream()
  const [agents, setAgents] = useState([])
  const [tools, setTools] = useState([])
  const [registryError, setRegistryError] = useState('')

  useEffect(() => {
    Promise.all([listAgents(), listTools()])
      .then(([loadedAgents, loadedTools]) => {
        setAgents(loadedAgents)
        setTools(loadedTools)
      })
      .catch((error) => setRegistryError(error.message))
  }, [])

  const busy = BUSY_STATUSES.has(run.status)

  return (
    <div className="app">
      <header className="app__header">
        <h1>Agent Orchestration</h1>
        <p className="muted">
          A goal is planned into a task DAG, specialist agents are spawned from YAML and equipped
          with MCP tools at runtime, and every step is traced.
        </p>
      </header>

      {registryError && (
        <p className="error">Could not reach the backend: {registryError}</p>
      )}

      <Registry agents={agents} tools={tools} />

      <GoalInput onSubmit={run.start} busy={busy} />

      <ApprovalDialog approval={run.approval} onDecide={run.decide} />

      <div className="app__columns">
        <div className="app__main">
          <RunGraph plan={run.plan} taskState={run.taskState} />
          <TraceTimeline
            plan={run.plan}
            taskState={run.taskState}
            reasoning={run.reasoning}
            status={run.status}
            error={run.error}
          />
        </div>

        <aside className="app__side">
          <CostPanel
            usage={run.usage}
            plan={run.plan}
            consideredAgents={run.consideredAgents}
            recalledRuns={run.recalledRuns}
            status={run.status}
          />
        </aside>
      </div>

      {run.finalAnswer && (
        <section className="panel answer">
          <h2>Answer</h2>
          <Markdown>{run.finalAnswer}</Markdown>
        </section>
      )}
    </div>
  )
}
