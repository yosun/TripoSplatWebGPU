import { useEffect, useRef, useState, type ReactNode } from 'react'

import type { Viewer } from '@mkkellogg/gaussian-splats-3d'

export interface CameraSnapshot {
  position: [number, number, number]
  target: [number, number, number]
  up: [number, number, number]
}

interface SplatPreviewProps {
  plyUrl: string | null
  generationKey: number
  initialCameraPosition?: [number, number, number]
  initialCameraTarget?: [number, number, number]
  initialCameraUp?: [number, number, number]
  bgColor: string
  fov: number
  autoRotate: boolean
  maxScreenSize: number
  onCameraChange?: (snap: CameraSnapshot) => void
  children?: ReactNode
}

export function SplatPreview({
  plyUrl,
  generationKey,
  initialCameraPosition,
  initialCameraTarget,
  initialCameraUp,
  bgColor,
  fov,
  autoRotate,
  maxScreenSize,
  onCameraChange,
  children,
}: SplatPreviewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const viewerRef = useRef<Viewer | null>(null)
  const [viewerStatus, setViewerStatus] = useState<string>('Waiting for a generated splat…')

  const onCameraChangeRef = useRef(onCameraChange)
  onCameraChangeRef.current = onCameraChange

  useEffect(() => {
    let cancelled = false
    let localViewer: Viewer | null = null

    const mount = async () => {
      if (!plyUrl || !containerRef.current) {
        setViewerStatus('Waiting for a generated splat…')
        return
      }

      setViewerStatus('Loading splat preview…')

      const module = await import('@mkkellogg/gaussian-splats-3d')
      if (cancelled || !containerRef.current) {
        return
      }

      const viewerOptions: Record<string, unknown> = {
        rootElement: containerRef.current,
        selfDrivenMode: true,
        useBuiltInControls: true,
        sharedMemoryForWorkers: false,
        ignoreDevicePixelRatio: false,
        inMemoryCompressionLevel: 0,
        maxScreenSpaceSplatSize: maxScreenSize,
      }
      if (initialCameraPosition) viewerOptions.initialCameraPosition = initialCameraPosition
      if (initialCameraTarget) viewerOptions.initialCameraLookAt = initialCameraTarget
      if (initialCameraUp) viewerOptions.cameraUp = initialCameraUp

      localViewer = new module.Viewer(viewerOptions)
      viewerRef.current = localViewer

      try {
        localViewer.start()
        await localViewer.addSplatScene(plyUrl, {
          format: module.SceneFormat.Ply,
          showLoadingUI: false,
          splatAlphaRemovalThreshold: 1,
        })
        if (cancelled) return

        applyLiveSettings(localViewer, { bgColor, fov, autoRotate })
        attachCameraReporter(localViewer, () => onCameraChangeRef.current)

        setViewerStatus('Preview ready. Drag to orbit, scroll to zoom.')
      } catch (error) {
        if (!cancelled) {
          setViewerStatus(`Preview failed: ${error instanceof Error ? error.message : String(error)}`)
        }
      }
    }

    void mount()

    return () => {
      cancelled = true
      const viewer = localViewer ?? viewerRef.current
      viewerRef.current = null
      if (viewer) {
        void viewer.dispose().catch(() => undefined)
      }
    }
    // maxScreenSize is in deps because it's a constructor option that can't be
    // updated live; changing it forces a viewer recreate.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [plyUrl, generationKey, maxScreenSize])

  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer) return
    applyLiveSettings(viewer, { bgColor, fov, autoRotate })
  }, [bgColor, fov, autoRotate])

  return (
    <section className="panel preview-panel">
      <div className="preview-toolbar">
        <span className="preview-title">Gaussian Splat Preview</span>
        <span className="preview-status">{viewerStatus}</span>
      </div>
      <div className="splat-canvas-shell">
        <div ref={containerRef} className="splat-canvas-host" />
        {!plyUrl ? <div className="splat-empty">Generate a splat to preview it here.</div> : null}
      </div>
      {children}
    </section>
  )
}

function applyLiveSettings(
  viewer: Viewer,
  settings: { bgColor: string; fov: number; autoRotate: boolean },
): void {
  const cam = (viewer as unknown as { camera?: { fov?: number; updateProjectionMatrix?: () => void } }).camera
  if (cam && typeof cam.fov === 'number' && Number.isFinite(settings.fov)) {
    cam.fov = settings.fov
    cam.updateProjectionMatrix?.()
  }
  const renderer = (viewer as unknown as { renderer?: { setClearColor?: (c: string) => void } }).renderer
  renderer?.setClearColor?.(settings.bgColor)

  const controls = (viewer as unknown as { controls?: { autoRotate?: boolean; autoRotateSpeed?: number } }).controls
  if (controls) {
    controls.autoRotate = settings.autoRotate
    if (settings.autoRotate) controls.autoRotateSpeed = 1.0
  }
}

function attachCameraReporter(viewer: Viewer, getCallback: () => ((snap: CameraSnapshot) => void) | undefined): void {
  const controls = (
    viewer as unknown as {
      controls?: {
        addEventListener?: (type: string, listener: () => void) => void
        target?: { x: number; y: number; z: number }
      }
    }
  ).controls
  const cam = (
    viewer as unknown as {
      camera?: {
        position?: { x: number; y: number; z: number }
        up?: { x: number; y: number; z: number }
      }
    }
  ).camera
  if (!controls?.addEventListener || !cam?.position || !cam?.up || !controls.target) return

  const emit = () => {
    const cb = getCallback()
    if (!cb) return
    cb({
      position: [cam.position!.x, cam.position!.y, cam.position!.z],
      target: [controls.target!.x, controls.target!.y, controls.target!.z],
      up: [cam.up!.x, cam.up!.y, cam.up!.z],
    })
  }
  controls.addEventListener('change', emit)
  emit()
}
