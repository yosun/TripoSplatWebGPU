'use client'

/** Alpha client-component example; point the model URL at the complete immutable manifest. */
import { useEffect, useRef, useState } from 'react'
import { TripoSplatWebGPU } from '@ai3d/triposplat-webgpu'

export default function TripoSplatClient() {
  const modelRef = useRef<TripoSplatWebGPU | null>(null)
  const [message, setMessage] = useState('Loading…')

  useEffect(() => {
    const controller = new AbortController()
    const model = new TripoSplatWebGPU({
      modelBaseUrl: process.env.NEXT_PUBLIC_TRIPOSPLAT_MODEL_BASE_URL!,
      cache: 'opfs',
    })
    modelRef.current = model

    void model.load({
      signal: controller.signal,
      onProgress: ({ message: next }) => setMessage(next),
    }).then(() => setMessage('Ready'), (error: unknown) => {
      if (!controller.signal.aborted) setMessage(String(error))
    })

    return () => {
      controller.abort()
      modelRef.current = null
      void model.dispose()
    }
  }, [])

  return (
    <main>
      <p>{message}</p>
      <input
        type="file"
        accept="image/*"
        onChange={(event) => {
          const file = event.currentTarget.files?.[0]
          const model = modelRef.current
          if (!file || !model) return
          void model.generate(file, {
            steps: 20,
            onProgress: ({ message: next }) => setMessage(next),
          }).then((scene) => {
            setMessage(`Generated ${scene.count.toLocaleString()} Gaussians`)
            scene.dispose()
          }, (error: unknown) => setMessage(String(error)))
        }}
      />
    </main>
  )
}
