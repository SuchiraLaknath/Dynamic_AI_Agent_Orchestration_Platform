/**
 * Renders model output as Markdown.
 *
 * Agent and synthesizer output is Markdown -- headings, tables of sprint
 * figures, bolded conclusions -- so showing it verbatim wastes the structure
 * the model already produced.
 *
 * GFM is enabled because the tables are the point: a velocity breakdown is far
 * easier to check as a table than as pipe characters. Raw HTML is deliberately
 * NOT enabled: this text comes from an LLM by way of tool output, and
 * react-markdown escapes HTML unless rehype-raw is added, which keeps a
 * prompt-injected `<script>` inert.
 */

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const COMPONENTS = {
  // Wide tables scroll inside their own box rather than widening the page.
  table: ({ node, ...props }) => (
    <div className="markdown__scroll">
      <table {...props} />
    </div>
  ),
  a: ({ node, ...props }) => <a target="_blank" rel="noopener noreferrer" {...props} />,
}

export default function Markdown({ children, className = '' }) {
  return (
    <div className={`markdown ${className}`.trim()}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={COMPONENTS}>
        {children || ''}
      </ReactMarkdown>
    </div>
  )
}
