import { useState } from 'react'

const SNIPPET = `<script type="module" src="https://cdn.jsdelivr.net/npm/@bring-shrubbery/sharp-splat-viewer/dist/sharp-splat-viewer.iife.js"></script>
<sharp-splat src="YOUR_HOSTED_PLY_URL.ply" style="width:600px;height:400px"></sharp-splat>`

export function EmbedSnippet() {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(SNIPPET)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // Clipboard write can fail in non-secure contexts; fall back to a manual selection prompt.
      setCopied(false)
    }
  }

  return (
    <div className="embed-snippet">
      <div className="embed-snippet-header">
        <span className="embed-snippet-title">Embed on the web</span>
        <button type="button" className="btn embed-snippet-copy" onClick={() => void handleCopy()}>
          {copied ? 'Copied!' : 'Copy snippet'}
        </button>
      </div>
      <textarea className="embed-snippet-text" readOnly value={SNIPPET} rows={3} spellCheck={false} />
      <small className="embed-snippet-hint">
        Host the downloaded `.ply` somewhere with public CORS access (R2, S3, etc.) and paste this
        snippet into your HTML. The baked-in camera and render settings come along automatically.
      </small>
    </div>
  )
}
