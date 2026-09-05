/** The execution trace, newest activity last: one block per task, tools nested. */

import Markdown from './Markdown'

const STATUS_LABEL = {
  pending: 'queued',
  running: 'running',
  done: 'done',
  completed: 'done',
  failed: 'failed',
  declined: 'declined',
}

function ToolCall({ call }) {
  return (
    <li className={`toolcall toolcall--${call.pending ? 'pending' : call.ok ? 'ok' : 'error'}`}>
      <div className="toolcall__head">
        <code>{call.tool}</code>
        <span className="toolcall__args">{JSON.stringify(call.arguments)}</span>
      </div>
      {call.preview && <pre className="toolcall__preview">{call.preview}</pre>}
    </li>
  )
}

export default function TraceTimeline({ plan, taskState, reasoning, status, error }) {
  if (status === 'idle') {
    return (
      <section className="panel">
        <h2>Trace</h2>
        <p className="muted">Submit a goal to see the plan and every step it takes.</p>
      </section>
    )
  }

  return (
    <section className="panel trace">
      <h2>Trace</h2>

      {reasoning && (
        <p className="trace__reasoning">
          <strong>Why this plan:</strong> {reasoning}
        </p>
      )}
      {plan.length === 0 && status === 'running' && <p className="muted">Planning...</p>}
      {error && <p className="error">{error}</p>}

      <ol className="trace__tasks">
        {plan.map((task) => {
          const state = taskState[task.task_id] || { status: 'pending', toolCalls: [] }
          return (
            <li key={task.task_id} className={`task task--${state.status}`}>
              <header className="task__head">
                <span className="task__id">{task.task_id}</span>
                <span className="task__agent">{task.agent_id}</span>
                <span className={`badge badge--${state.status}`}>
                  {STATUS_LABEL[state.status] || state.status}
                </span>
              </header>

              <p className="task__objective">{task.objective}</p>
              {task.depends_on?.length > 0 && (
                <p className="muted small">after: {task.depends_on.join(', ')}</p>
              )}

              {state.toolCalls?.length > 0 && (
                <ul className="task__tools">
                  {state.toolCalls.map((call, index) => (
                    <ToolCall key={`${call.tool}-${index}`} call={call} />
                  ))}
                </ul>
              )}

              {state.output && (
                <div className="task__output">
                  <Markdown className="markdown--compact">{state.output}</Markdown>
                </div>
              )}
            </li>
          )
        })}
      </ol>
    </section>
  )
}
