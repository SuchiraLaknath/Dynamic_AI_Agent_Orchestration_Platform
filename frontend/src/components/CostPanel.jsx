/** Token usage and dollar cost for the run, plus the agents it selected. */

function money(value) {
  if (value === null || value === undefined) return '--'
  return value < 0.01 ? `$${value.toFixed(5)}` : `$${value.toFixed(4)}`
}

function count(value) {
  return (value ?? 0).toLocaleString()
}

export default function CostPanel({ usage, plan, consideredAgents, recalledRuns, status }) {
  const selected = [...new Set(plan.map((task) => task.agent_id))]
  const notSelected = consideredAgents.filter((agentId) => !selected.includes(agentId))

  return (
    <section className="panel cost-panel">
      <h2>Cost &amp; routing</h2>

      <dl className="cost-panel__figures">
        <div>
          <dt>Input tokens</dt>
          <dd>{count(usage?.input_tokens)}</dd>
        </div>
        <div>
          <dt>Output tokens</dt>
          <dd>{count(usage?.output_tokens)}</dd>
        </div>
        <div>
          <dt>Cost</dt>
          <dd className="cost-panel__cost">{money(usage?.cost_usd)}</dd>
        </div>
      </dl>
      {!usage && status !== 'idle' && (
        <p className="muted small">Totalled when the run completes.</p>
      )}

      <h3>Agents selected</h3>
      <ul className="taglist">
        {selected.length === 0 && <li className="muted">--</li>}
        {selected.map((agentId) => (
          <li key={agentId} className="tag tag--on">
            {agentId}
          </li>
        ))}
      </ul>

      {notSelected.length > 0 && (
        <>
          <h3>Offered but not chosen</h3>
          <ul className="taglist">
            {notSelected.map((agentId) => (
              <li key={agentId} className="tag">
                {agentId}
              </li>
            ))}
          </ul>
        </>
      )}

      {recalledRuns.length > 0 && (
        <p className="muted small">
          Planner recalled {recalledRuns.length} similar past run
          {recalledRuns.length === 1 ? '' : 's'}.
        </p>
      )}
    </section>
  )
}
