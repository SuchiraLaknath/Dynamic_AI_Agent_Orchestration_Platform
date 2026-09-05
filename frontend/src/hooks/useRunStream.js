/**
 * Subscribes to a run's event stream and exposes the state the UI renders.
 *
 * Owns only the React and network concerns -- starting a run, attaching the
 * stream, sending an approval decision. The derivation of state from events
 * lives in runState.js.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { approveRun, createRun, streamRun } from '../api/client'
import { EMPTY_RUN, reduceEvent } from './runState'

export function useRunStream() {
  const [state, setState] = useState(EMPTY_RUN)
  const closeStream = useRef(null)

  useEffect(() => () => closeStream.current?.(), [])

  const start = useCallback(async (goal) => {
    closeStream.current?.()
    setState({ ...EMPTY_RUN, status: 'starting', goal })
    try {
      const { run_id: runId } = await createRun(goal)
      setState((current) => ({ ...current, runId }))
      closeStream.current = streamRun(runId, (event) =>
        setState((current) => reduceEvent(current, event)),
      )
    } catch (error) {
      setState((current) => ({ ...current, status: 'failed', error: error.message }))
    }
  }, [])

  const decide = useCallback(
    async (approved, reason) => {
      if (!state.runId) return
      setState((current) => ({ ...current, approval: null, status: 'running' }))
      try {
        await approveRun(state.runId, approved, reason)
      } catch (error) {
        setState((current) => ({ ...current, status: 'failed', error: error.message }))
      }
    },
    [state.runId],
  )

  return { ...state, start, decide }
}
