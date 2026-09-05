/**
 * Pure reduction of a run's event stream into renderable state.
 *
 * Kept free of React so the derivation can be reasoned about and tested on its
 * own -- it is the only non-trivial logic in the frontend.
 *
 * Per-task completion is *derived*, not received. The backend's event
 * vocabulary is a fixed seven names with no `task.completed`, so a task is
 * marked done when a task depending on it starts, or when the run completes and
 * reports each task's final status.
 */

export const EMPTY_RUN = {
  runId: null,
  status: 'idle',
  goal: '',
  plan: [],
  reasoning: '',
  consideredAgents: [],
  recalledRuns: [],
  events: [],
  taskState: {},
  approval: null,
  finalAnswer: '',
  usage: null,
  error: '',
}

/** Every task the given task transitively depends on. */
export function ancestorsOf(taskId, plan) {
  const byId = Object.fromEntries(plan.map((task) => [task.task_id, task]))
  const found = new Set()
  const queue = [...(byId[taskId]?.depends_on || [])]

  while (queue.length) {
    const current = queue.pop()
    if (found.has(current)) continue
    found.add(current)
    queue.push(...(byId[current]?.depends_on || []))
  }
  return found
}

function onRunStarted(state, data) {
  return {
    ...state,
    status: 'running',
    goal: data.goal,
    plan: data.plan || [],
    reasoning: data.reasoning || '',
    consideredAgents: data.considered_agents || [],
    recalledRuns: data.recalled_runs || [],
    taskState: Object.fromEntries(
      (data.plan || []).map((task) => [task.task_id, { status: 'pending', toolCalls: [] }]),
    ),
  }
}

function onTaskStarted(state, data) {
  const taskState = { ...state.taskState }
  taskState[data.task_id] = {
    ...(taskState[data.task_id] || { toolCalls: [] }),
    status: 'running',
    agentId: data.agent_id,
    model: data.model,
    tools: data.tools || [],
  }

  ancestorsOf(data.task_id, state.plan).forEach((ancestor) => {
    if (taskState[ancestor]?.status === 'running') {
      taskState[ancestor] = { ...taskState[ancestor], status: 'done' }
    }
  })
  return { ...state, taskState }
}

function onToolEvent(state, name, data) {
  const taskState = { ...state.taskState }
  const task = taskState[data.task_id] || { status: 'running', toolCalls: [] }

  const toolCalls =
    name === 'tool.called'
      ? [...task.toolCalls, { tool: data.tool, arguments: data.arguments, pending: true }]
      : task.toolCalls.map((call, index) =>
          index === task.toolCalls.length - 1
            ? { ...call, pending: false, ok: data.ok, preview: data.preview }
            : call,
        )

  taskState[data.task_id] = { ...task, toolCalls }
  return { ...state, taskState }
}

function onRunCompleted(state, data) {
  const outputs = data.task_outputs || {}
  return {
    ...state,
    status: 'completed',
    finalAnswer: data.final_answer || '',
    usage: data.usage || null,
    approval: null,
    taskState: Object.fromEntries(
      Object.entries(state.taskState).map(([taskId, task]) => [
        taskId,
        { ...task, status: outputs[taskId]?.status || 'done', output: outputs[taskId]?.output },
      ]),
    ),
  }
}

/** Fold one event into the run state. Unknown names pass through untouched. */
export function reduceEvent(state, event) {
  const { name, data } = event
  const withEvent = { ...state, events: [...state.events, event] }

  switch (name) {
    case 'run.started':
      return onRunStarted(withEvent, data)
    case 'task.started':
      return onTaskStarted(withEvent, data)
    case 'tool.called':
    case 'tool.result':
      return onToolEvent(withEvent, name, data)
    case 'approval.required':
      return { ...withEvent, status: 'awaiting_approval', approval: data }
    case 'run.completed':
      return onRunCompleted(withEvent, data)
    case 'run.failed':
      return { ...withEvent, status: 'failed', error: data.error || 'The run failed.' }
    default:
      return withEvent
  }
}
