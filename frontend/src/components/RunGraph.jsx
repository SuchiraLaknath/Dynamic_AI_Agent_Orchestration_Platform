/**
 * The plan as a DAG, laid out in dependency layers.
 *
 * Layout is computed here rather than by a layout library: the plan is a
 * shallow DAG of a handful of nodes, so longest-path layering is both correct
 * and about ten lines. React Flow draws it.
 */

import { useMemo } from 'react'
import ReactFlow, { Background, Controls, MarkerType } from 'reactflow'
import 'reactflow/dist/style.css'

const NODE_WIDTH = 190
const COLUMN_GAP = 250
const ROW_GAP = 96

const STATUS_COLOR = {
  pending: { border: '#c9ccd6', background: '#f7f7fa', color: '#5b6070' },
  running: { border: '#2f6df6', background: '#e9f0ff', color: '#12306f' },
  done: { border: '#1f9d63', background: '#e8f7ef', color: '#0f5537' },
  completed: { border: '#1f9d63', background: '#e8f7ef', color: '#0f5537' },
  failed: { border: '#d64541', background: '#fdecec', color: '#8c2320' },
  declined: { border: '#c8912a', background: '#fdf4e0', color: '#7a5a12' },
}

/** Depth of each task = longest path from a root, which is its column. */
function layerOf(plan) {
  const byId = Object.fromEntries(plan.map((task) => [task.task_id, task]))
  const depth = {}

  const resolve = (taskId, seen = new Set()) => {
    if (depth[taskId] !== undefined) return depth[taskId]
    if (seen.has(taskId)) return 0
    seen.add(taskId)
    const parents = byId[taskId]?.depends_on || []
    depth[taskId] = parents.length
      ? Math.max(...parents.map((parent) => resolve(parent, seen))) + 1
      : 0
    return depth[taskId]
  }

  plan.forEach((task) => resolve(task.task_id))
  return depth
}

export default function RunGraph({ plan, taskState }) {
  const { nodes, edges } = useMemo(() => {
    if (plan.length === 0) return { nodes: [], edges: [] }

    const depth = layerOf(plan)
    const seenPerColumn = {}

    const nodes = plan.map((task) => {
      const column = depth[task.task_id]
      const row = seenPerColumn[column] ?? 0
      seenPerColumn[column] = row + 1
      const status = taskState[task.task_id]?.status || 'pending'
      const palette = STATUS_COLOR[status] || STATUS_COLOR.pending

      return {
        id: task.task_id,
        position: { x: column * COLUMN_GAP, y: row * ROW_GAP },
        data: {
          label: (
            <div className="node">
              <strong>{task.agent_id}</strong>
              <span>{task.task_id}</span>
            </div>
          ),
        },
        style: {
          width: NODE_WIDTH,
          borderRadius: 10,
          border: `2px solid ${palette.border}`,
          background: palette.background,
          color: palette.color,
          fontSize: 12,
          padding: 8,
        },
      }
    })

    const edges = plan.flatMap((task) =>
      (task.depends_on || []).map((parent) => ({
        id: `${parent}->${task.task_id}`,
        source: parent,
        target: task.task_id,
        animated: taskState[task.task_id]?.status === 'running',
        markerEnd: { type: MarkerType.ArrowClosed },
      })),
    )

    return { nodes, edges }
  }, [plan, taskState])

  return (
    <section className="panel run-graph">
      <h2>Run graph</h2>
      {plan.length === 0 ? (
        <p className="muted">The DAG appears once the planner has chosen the agents.</p>
      ) : (
        <div className="run-graph__canvas">
          <ReactFlow nodes={nodes} edges={edges} fitView proOptions={{ hideAttribution: true }}>
            <Background gap={16} />
            <Controls showInteractive={false} />
          </ReactFlow>
        </div>
      )}
    </section>
  )
}
