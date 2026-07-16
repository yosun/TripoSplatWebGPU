import { useEffect, useRef, useState, type ReactNode } from 'react'

import type { Viewer } from '@mkkellogg/gaussian-splats-3d'

export interface CameraSnapshot {
  position: [number, number, number]
  target: [number, number, number]
  up: [number, number, number]
}

export type SplatPreviewState = 'waiting' | 'loading' | 'ready' | 'failed'

export interface SplatPreviewStatus {
  state: SplatPreviewState
  message: string
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
  splatPosition: [number, number, number]
  splatRotation: [number, number, number]
  splatFlip: [boolean, boolean, boolean]
  onCameraChange?: (snap: CameraSnapshot) => void
  onViewerStateChange?: (status: SplatPreviewStatus) => void
  /** Called only after this PLY URL no longer has an attached viewer. */
  onViewerDisposed?: (plyUrl: string) => void
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
  splatPosition,
  splatRotation,
  splatFlip,
  onCameraChange,
  onViewerStateChange,
  onViewerDisposed,
  children,
}: SplatPreviewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const viewerRef = useRef<Viewer | null>(null)
  const [viewerStatus, setViewerStatus] = useState<SplatPreviewStatus>({
    state: 'waiting',
    message: 'Waiting for a generated splat…',
  })

  const onCameraChangeRef = useRef(onCameraChange)
  onCameraChangeRef.current = onCameraChange
  const onViewerStateChangeRef = useRef(onViewerStateChange)
  onViewerStateChangeRef.current = onViewerStateChange
  const onViewerDisposedRef = useRef(onViewerDisposed)
  onViewerDisposedRef.current = onViewerDisposed

  // Refs hold the latest transform values so the mount effect can apply them
  // when the viewer is recreated, without restarting on every transform change.
  const splatPositionRef = useRef(splatPosition)
  const splatRotationRef = useRef(splatRotation)
  const splatFlipRef = useRef(splatFlip)
  splatPositionRef.current = splatPosition
  splatRotationRef.current = splatRotation
  splatFlipRef.current = splatFlip

  useEffect(() => {
    let cancelled = false
    let localViewer: Viewer | null = null

    const publishStatus = (state: SplatPreviewState, message: string) => {
      if (cancelled) return
      const status = { state, message }
      setViewerStatus(status)
      onViewerStateChangeRef.current?.(status)
    }

    const mount = async () => {
      if (!plyUrl || !containerRef.current) {
        publishStatus('waiting', 'Waiting for a generated splat…')
        return
      }

      publishStatus('loading', 'Loading splat preview…')
      try {
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
          dynamicScene: true,
        }
        if (initialCameraPosition) viewerOptions.initialCameraPosition = initialCameraPosition
        if (initialCameraTarget) viewerOptions.initialCameraLookAt = initialCameraTarget
        if (initialCameraUp) viewerOptions.cameraUp = initialCameraUp

        localViewer = new module.Viewer(viewerOptions)
        viewerRef.current = localViewer
        localViewer.start()
        const initialPos = splatPositionRef.current
        const initialRot = splatRotationRef.current
        const initialFlip = splatFlipRef.current
        const sceneOptions: Record<string, unknown> = {
          format: module.SceneFormat.Ply,
          showLoadingUI: false,
          splatAlphaRemovalThreshold: 1,
        }
        if (initialPos.some((n) => n !== 0)) sceneOptions.position = initialPos
        if (initialRot.some((n) => n !== 0)) {
          sceneOptions.rotation = eulerDegToQuaternion(initialRot)
        }
        if (initialFlip.some(Boolean)) {
          sceneOptions.scale = flipsToScale(initialFlip)
        }

        await localViewer.addSplatScene(plyUrl, sceneOptions)
        if (cancelled) return

        applyLiveSettings(localViewer, { bgColor, fov, autoRotate })
        applySplatTransform(localViewer, {
          position: splatPositionRef.current,
          rotation: splatRotationRef.current,
          flip: splatFlipRef.current,
        })
        attachCameraReporter(localViewer, () => onCameraChangeRef.current)

        publishStatus('ready', 'Preview ready. Drag to orbit, scroll to zoom.')
      } catch (error) {
        if (!cancelled) {
          publishStatus(
            'failed',
            `Preview failed: ${error instanceof Error ? error.message : String(error)}`,
          )
        }
      }
    }

    void mount()

    return () => {
      cancelled = true
      const viewer = localViewer ?? viewerRef.current
      viewerRef.current = null
      const notifyDisposed = () => {
        if (plyUrl) onViewerDisposedRef.current?.(plyUrl)
      }
      if (viewer) {
        void Promise.resolve()
          .then(() => viewer.dispose())
          .catch(() => undefined)
          .finally(notifyDisposed)
      } else {
        notifyDisposed()
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

  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer) return
    applySplatTransform(viewer, {
      position: splatPosition,
      rotation: splatRotation,
      flip: splatFlip,
    })
  }, [splatPosition, splatRotation, splatFlip])

  return (
    <section className="panel preview-panel">
      <div className="preview-toolbar">
        <span className="preview-title">Gaussian Splat Preview</span>
        <span
          className="preview-status"
          data-testid="splat-preview-status"
          data-viewer-state={viewerStatus.state}
        >
          {viewerStatus.message}
        </span>
      </div>
      <div className="splat-canvas-shell">
        <div ref={containerRef} className="splat-canvas-host" data-testid="splat-canvas-host" />
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

interface SplatTransform {
  position: [number, number, number]
  rotation: [number, number, number]
  flip: [boolean, boolean, boolean]
}

interface Vec3Like {
  set(x: number, y: number, z: number): void
}

interface QuatLike {
  setFromEuler(euler: { x: number; y: number; z: number; order?: string }): void
}

interface SceneObject {
  position?: Vec3Like
  quaternion?: QuatLike
  scale?: Vec3Like
}

interface SplatMeshLike {
  scenes?: SceneObject[]
  getScene?(index: number): SceneObject | null | undefined
  updateTransforms?(): void
}

function applySplatTransform(viewer: Viewer, transform: SplatTransform): void {
  const splatMesh = (viewer as unknown as { splatMesh?: SplatMeshLike }).splatMesh
  if (!splatMesh) return
  const scene = splatMesh.getScene?.(0) ?? splatMesh.scenes?.[0]
  if (!scene) return

  scene.position?.set(transform.position[0], transform.position[1], transform.position[2])

  if (scene.quaternion) {
    const radX = (transform.rotation[0] * Math.PI) / 180
    const radY = (transform.rotation[1] * Math.PI) / 180
    const radZ = (transform.rotation[2] * Math.PI) / 180
    scene.quaternion.setFromEuler({ x: radX, y: radY, z: radZ, order: 'XYZ' } as never)
  }

  const sx = transform.flip[0] ? -1 : 1
  const sy = transform.flip[1] ? -1 : 1
  const sz = transform.flip[2] ? -1 : 1
  scene.scale?.set(sx, sy, sz)

  splatMesh.updateTransforms?.()
}

function eulerDegToQuaternion(rot: [number, number, number]): [number, number, number, number] {
  // ZYX order to match Three.js's default Euler (XYZ extrinsic == ZYX intrinsic).
  const cx = Math.cos((rot[0] * Math.PI) / 360)
  const sx = Math.sin((rot[0] * Math.PI) / 360)
  const cy = Math.cos((rot[1] * Math.PI) / 360)
  const sy = Math.sin((rot[1] * Math.PI) / 360)
  const cz = Math.cos((rot[2] * Math.PI) / 360)
  const sz = Math.sin((rot[2] * Math.PI) / 360)
  // XYZ Euler -> quaternion (Three.js convention)
  const x = sx * cy * cz + cx * sy * sz
  const y = cx * sy * cz - sx * cy * sz
  const z = cx * cy * sz + sx * sy * cz
  const w = cx * cy * cz - sx * sy * sz
  return [x, y, z, w]
}

function flipsToScale(flip: [boolean, boolean, boolean]): [number, number, number] {
  return [flip[0] ? -1 : 1, flip[1] ? -1 : 1, flip[2] ? -1 : 1]
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
