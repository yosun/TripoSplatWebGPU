'use client'

import { useEffect, useRef, useState } from 'react'
import { TripoSplatWebGPU } from '@ai3d/triposplat-webgpu'
import { exportSplat } from '@ai3d/triposplat-webgpu/export'

const MODEL_BASE_URL = 'https://models.example.invalid/triposplat/v1/'

/**
 * Minimal App Router client component used to typecheck the packed package's
 * public lifecycle and export surfaces without installing the full Next.js CLI.
 */
export default function TripoSplatClient() {
  const modelRef = useRef<TripoSplatWebGPU | null>(null)
  const [status, setStatus] = useState('Idle')

  useEffect(() => {
    const model = new TripoSplatWebGPU({
      modelBaseUrl: MODEL_BASE_URL,
      cache: 'opfs',
    })
    modelRef.current = model

    return () => {
      modelRef.current = null
      void model.dispose()
    }
  }, [])

  return (
    <main>
      <p>{status}</p>
      <input
        aria-label="Source image"
        type="file"
        accept="image/*"
        onChange={(event) => {
          const file = event.currentTarget.files?.[0]
          const model = modelRef.current
          if (!file || !model) return

          setStatus('Generating')
          void model.generate(file, {
            steps: 4,
            onProgress: ({ message }) => setStatus(message),
          }).then(async (scene) => {
            try {
              const splat = await exportSplat(scene)
              setStatus(`Generated ${scene.count.toLocaleString()} Gaussians (${splat.size} bytes)`)
            } finally {
              scene.dispose()
            }
          }, (error: unknown) => setStatus(error instanceof Error ? error.message : String(error)))
        }}
      />
    </main>
  )
}
