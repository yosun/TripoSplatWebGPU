interface SplatPreviewControlsProps {
  bgColor: string
  onBgColor: (next: string) => void
  fov: number
  onFov: (next: number) => void
  maxScreenSize: number
  onMaxScreenSize: (next: number) => void
  autoRotate: boolean
  onAutoRotate: (next: boolean) => void
  onSaveDefaults: () => void
  saveDisabled?: boolean
  saveStatus?: string
}

export function SplatPreviewControls({
  bgColor,
  onBgColor,
  fov,
  onFov,
  maxScreenSize,
  onMaxScreenSize,
  autoRotate,
  onAutoRotate,
  onSaveDefaults,
  saveDisabled,
  saveStatus,
}: SplatPreviewControlsProps) {
  return (
    <div className="preview-controls">
      <div className="preview-controls-row">
        <label className="preview-control">
          <span>BG</span>
          <input
            type="color"
            value={bgColor}
            onChange={(event) => onBgColor(event.currentTarget.value)}
            aria-label="Background color"
          />
        </label>
        <label className="preview-control preview-control-grow">
          <span>FOV {Math.round(fov)}°</span>
          <input
            type="range"
            min={20}
            max={120}
            step={1}
            value={fov}
            onChange={(event) => onFov(Number(event.currentTarget.value))}
            aria-label="Field of view in degrees"
          />
        </label>
        <label className="preview-control">
          <span>Max splat px</span>
          <input
            type="number"
            min={256}
            max={8192}
            step={64}
            value={maxScreenSize}
            onChange={(event) => {
              const n = Math.max(64, Math.floor(Number(event.currentTarget.value) || 0))
              onMaxScreenSize(n)
            }}
          />
        </label>
        <label className="preview-control preview-control-checkbox">
          <input
            type="checkbox"
            checked={autoRotate}
            onChange={(event) => onAutoRotate(event.currentTarget.checked)}
          />
          <span>Auto-rotate</span>
        </label>
      </div>
      <div className="preview-controls-row preview-controls-actions">
        <button
          type="button"
          className="btn btn-primary"
          onClick={onSaveDefaults}
          disabled={saveDisabled}
        >
          Save current view as defaults
        </button>
        {saveStatus ? <span className="preview-save-status">{saveStatus}</span> : null}
      </div>
    </div>
  )
}
