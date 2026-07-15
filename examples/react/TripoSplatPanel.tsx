/** Alpha lifecycle example; opaque photos additionally require a local removeBackground callback. */
import { useEffect, useRef, useState } from 'react'
import {
  CancelledError,
  TripoSplatWebGPU,
  type GaussianScene,
  type GenerationProgress,
} from '@ai3d/triposplat-webgpu'

export function TripoSplatPanel({ modelBaseUrl }: { modelBaseUrl: string }) {
  const modelRef = useRef<TripoSplatWebGPU | null>(null)
  const sceneRef = useRef<GaussianScene | null>(null)
  const operationRef = useRef<AbortController | null>(null)
  const [status, setStatus] = useState('Checking WebGPU…')
  const [ready, setReady] = useState(false)

  useEffect(() => {
    const lifecycle = new AbortController()
    const model = new TripoSplatWebGPU({ modelBaseUrl, cache: 'opfs' })
    modelRef.current = model

    void (async () => {
      const report = await TripoSplatWebGPU.checkCompatibility()
      if (!report.supported) throw new Error(report.blockers.join('\n'))
      await model.load({
        signal: lifecycle.signal,
        onProgress: ({ message }) => setStatus(message),
      })
      if (!lifecycle.signal.aborted) {
        setReady(true)
        setStatus('Ready')
      }
    })().catch((error: unknown) => {
      if (!lifecycle.signal.aborted) setStatus(String(error))
    })

    return () => {
      lifecycle.abort()
      operationRef.current?.abort()
      sceneRef.current?.dispose()
      modelRef.current = null
      sceneRef.current = null
      void model.dispose()
    }
  }, [modelBaseUrl])

  async function generate(file: File) {
    const model = modelRef.current
    if (!model) return

    operationRef.current?.abort()
    sceneRef.current?.dispose()
    const controller = new AbortController()
    operationRef.current = controller

    const onProgress = (progress: GenerationProgress) => {
      const calls = progress.totalInvocations
        ? ` (${progress.invocation}/${progress.totalInvocations} DiT calls)`
        : ''
      setStatus(`${progress.message}${calls}`)
    }

    try {
      const scene = await model.generate(file, {
        steps: 20,
        guidanceScale: 3,
        gaussianCount: 262144,
        seed: 42,
        signal: controller.signal,
        onProgress,
      })
      sceneRef.current = scene
      setStatus(`Generated ${scene.count.toLocaleString()} Gaussians`)
    } catch (error) {
      setStatus(error instanceof CancelledError ? 'Cancelled' : String(error))
    }
  }

  return (
    <section>
      <p>{status}</p>
      <input
        aria-label="Source image"
        type="file"
        accept="image/*"
        disabled={!ready}
        onChange={(event) => {
          const file = event.currentTarget.files?.[0]
          if (file) void generate(file)
        }}
      />
      <button type="button" onClick={() => operationRef.current?.abort()}>
        Cancel
      </button>
    </section>
  )
}
