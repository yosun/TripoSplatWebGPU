interface SplatPreviewControlsProps {
  bgColor: string
  onBgColor: (next: string) => void
  fov: number
  onFov: (next: number) => void
  maxScreenSize: number
  onMaxScreenSize: (next: number) => void
  autoRotate: boolean
  onAutoRotate: (next: boolean) => void
  splatPosition: [number, number, number]
  onSplatPosition: (next: [number, number, number]) => void
  splatRotation: [number, number, number]
  onSplatRotation: (next: [number, number, number]) => void
  splatFlip: [boolean, boolean, boolean]
  onSplatFlip: (next: [boolean, boolean, boolean]) => void
  onResetTransform: () => void
  onSaveDefaults: () => void
  saveDisabled?: boolean
  saveStatus?: string
}

const AXES: Array<{ label: string; index: 0 | 1 | 2 }> = [
  { label: 'X', index: 0 },
  { label: 'Y', index: 1 },
  { label: 'Z', index: 2 },
]

export function SplatPreviewControls({
  bgColor,
  onBgColor,
  fov,
  onFov,
  maxScreenSize,
  onMaxScreenSize,
  autoRotate,
  onAutoRotate,
  splatPosition,
  onSplatPosition,
  splatRotation,
  onSplatRotation,
  splatFlip,
  onSplatFlip,
  onResetTransform,
  onSaveDefaults,
  saveDisabled,
  saveStatus,
}: SplatPreviewControlsProps) {
  const updatePosition = (i: number, value: number) => {
    const next: [number, number, number] = [...splatPosition]
    next[i] = value
    onSplatPosition(next)
  }
  const updateRotation = (i: number, value: number) => {
    const next: [number, number, number] = [...splatRotation]
    next[i] = value
    onSplatRotation(next)
  }
  const toggleFlip = (i: number) => {
    const next: [boolean, boolean, boolean] = [...splatFlip]
    next[i] = !next[i]
    onSplatFlip(next)
  }
  const nudgeRotation = (i: number, delta: number) => {
    const next: [number, number, number] = [...splatRotation]
    let v = (next[i] + delta) % 360
    if (v > 180) v -= 360
    if (v < -180) v += 360
    next[i] = v
    onSplatRotation(next)
  }

  const transformDirty =
    splatPosition.some((n) => n !== 0) ||
    splatRotation.some((n) => n !== 0) ||
    splatFlip.some(Boolean)

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

      <div className="preview-transform">
        <div className="preview-transform-row">
          <span className="preview-transform-label">Position</span>
          {AXES.map(({ label, index }) => (
            <label key={label} className="preview-transform-axis">
              <span>{label}</span>
              <input
                type="number"
                step={0.05}
                value={splatPosition[index]}
                onChange={(event) => updatePosition(index, Number(event.currentTarget.value) || 0)}
              />
            </label>
          ))}
        </div>

        <div className="preview-transform-row">
          <span className="preview-transform-label">Rotation</span>
          {AXES.map(({ label, index }) => (
            <label key={label} className="preview-transform-axis">
              <span>{label}°</span>
              <input
                type="number"
                step={5}
                value={splatRotation[index]}
                onChange={(event) => updateRotation(index, Number(event.currentTarget.value) || 0)}
              />
              <button
                type="button"
                className="btn preview-transform-nudge"
                onClick={() => nudgeRotation(index, -90)}
                aria-label={`Rotate ${label} -90°`}
              >
                −90
              </button>
              <button
                type="button"
                className="btn preview-transform-nudge"
                onClick={() => nudgeRotation(index, 90)}
                aria-label={`Rotate ${label} +90°`}
              >
                +90
              </button>
            </label>
          ))}
        </div>

        <div className="preview-transform-row">
          <span className="preview-transform-label">Flip</span>
          {AXES.map(({ label, index }) => (
            <button
              key={label}
              type="button"
              className={`btn preview-transform-flip${splatFlip[index] ? ' is-active' : ''}`}
              onClick={() => toggleFlip(index)}
              aria-pressed={splatFlip[index]}
            >
              {label}
            </button>
          ))}
          <button
            type="button"
            className="btn preview-transform-reset"
            onClick={onResetTransform}
            disabled={!transformDirty}
          >
            Reset transform
          </button>
        </div>
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
