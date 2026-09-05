/** Blocks a sensitive agent until a human approves it. */

import { useState } from 'react'

export default function ApprovalDialog({ approval, onDecide }) {
  const [reason, setReason] = useState('')
  if (!approval) return null

  return (
    <div className="approval" role="alertdialog" aria-labelledby="approval-title">
      <h2 id="approval-title">Approval required</h2>
      <p>
        Task <code>{approval.task_id}</code> would run <strong>{approval.agent_id}</strong>, which
        holds {(approval.tools || []).join(', ') || 'no tools'}. It is paused until you decide.
      </p>
      <p className="approval__objective">{approval.objective}</p>

      <input
        type="text"
        value={reason}
        onChange={(event) => setReason(event.target.value)}
        placeholder="Optional note, recorded in the trace"
      />
      <div className="approval__actions">
        <button type="button" className="danger" onClick={() => onDecide(false, reason)}>
          Decline
        </button>
        <button type="button" onClick={() => onDecide(true, reason)}>
          Approve and run
        </button>
      </div>
    </div>
  )
}
