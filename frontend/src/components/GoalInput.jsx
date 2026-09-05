/** The goal box, with the demo scenarios as one-click examples. */

import { useState } from 'react'

const EXAMPLES = [
  'Predict the velocity for our next sprint based on previous Jira sprints.',
  'How many story points did we complete in the last three sprints, and what was our completion rate?',
  'Two independent things about our next sprint: a velocity forecast from sprint history, and the team capacity available. Then review the combined picture for delivery risk.',
  'Work out our next sprint commitment and publish it to the Jira board.',
]

export default function GoalInput({ onSubmit, busy }) {
  const [goal, setGoal] = useState(EXAMPLES[0])

  function submit(event) {
    event.preventDefault()
    const trimmed = goal.trim()
    if (trimmed && !busy) onSubmit(trimmed)
  }

  return (
    <form className="panel goal-input" onSubmit={submit}>
      <label htmlFor="goal">Goal</label>
      <textarea
        id="goal"
        rows={3}
        value={goal}
        disabled={busy}
        onChange={(event) => setGoal(event.target.value)}
        placeholder="Describe what you want in plain language..."
      />
      <div className="goal-input__actions">
        <button type="submit" disabled={busy || !goal.trim()}>
          {busy ? 'Running...' : 'Run'}
        </button>
        <div className="goal-input__examples">
          {EXAMPLES.map((example, index) => (
            <button
              key={example}
              type="button"
              className="chip"
              disabled={busy}
              title={example}
              onClick={() => setGoal(example)}
            >
              Example {index + 1}
            </button>
          ))}
        </div>
      </div>
    </form>
  )
}
