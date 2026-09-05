/**
 * Every network call the app makes. Components never call fetch themselves.
 *
 * Requests go to /api, which the Vite dev server proxies to the backend, so
 * the browser only ever uses same-origin relative URLs and there is no API
 * base URL to configure per environment.
 */

const API_BASE = '/api'

/** The event names the backend publishes. The UI switches on exactly these. */
export const EVENT_NAMES = [
  'run.started',
  'task.started',
  'tool.called',
  'tool.result',
  'approval.required',
  'run.completed',
  'run.failed',
]

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!response.ok) {
    const body = await response.text()
    throw new Error(`${options.method || 'GET'} ${path} failed (${response.status}): ${body}`)
  }
  return response.json()
}

/** Submit a goal. Resolves as soon as the run is accepted, not when it finishes. */
export function createRun(goal) {
  return request('/runs', { method: 'POST', body: JSON.stringify({ goal }) })
}

/** The full record of a run: plan, answer, cost and durable trace. */
export function getRun(runId) {
  return request(`/runs/${runId}`)
}

/** Resume a run paused at an approval interrupt. */
export function approveRun(runId, approved, reason = '') {
  return request(`/runs/${runId}/approve`, {
    method: 'POST',
    body: JSON.stringify({ approved, reason }),
  })
}

export function listAgents() {
  return request('/agents')
}

export function listTools() {
  return request('/tools')
}

/**
 * Subscribe to a run's trace over server-sent events.
 *
 * The backend replays everything already published before going live, so it is
 * safe to attach at any point -- including after the run has already finished.
 * Returns a function that closes the stream.
 */
export function streamRun(runId, onEvent, onError) {
  const source = new EventSource(`${API_BASE}/runs/${runId}/events`)

  // The server names every event, so the default `message` handler never fires
  // and each name needs its own listener.
  EVENT_NAMES.forEach((name) => {
    source.addEventListener(name, (message) => {
      onEvent(JSON.parse(message.data))
    })
  })

  source.onerror = () => {
    // EventSource also fires onerror on a normal server-side close, so this is
    // only surfaced while the run is still expected to be producing events.
    if (source.readyState === EventSource.CLOSED && onError) onError()
  }

  return () => source.close()
}
