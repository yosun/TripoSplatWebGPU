import { OrtWorkerClient } from './runtime/OrtWorkerClient'
import { createTensorPayload } from './runtime/tensors'

const fixture = '/fixtures/generated/context-attention-mha-inv08'
const status = document.querySelector<HTMLPreElement>('#status')!

async function fetchF32(path: string, count: number): Promise<Float32Array> {
  const response = await fetch(path)
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}: ${path}`)
  const buffer = await response.arrayBuffer()
  if (buffer.byteLength !== count * 4) {
    throw new Error(`${path} has ${buffer.byteLength} bytes; expected ${count * 4}`)
  }
  return new Float32Array(buffer)
}

function metrics(reference: Float32Array, candidate: Float32Array) {
  let maxAbsoluteError = 0
  let squaredError = 0
  let meanAbsoluteError = 0
  for (let index = 0; index < reference.length; index += 1) {
    const error = Math.abs(reference[index] - candidate[index])
    maxAbsoluteError = Math.max(maxAbsoluteError, error)
    meanAbsoluteError += error
    squaredError += error * error
  }
  return {
    maxAbsoluteError,
    meanAbsoluteError: meanAbsoluteError / reference.length,
    rmse: Math.sqrt(squaredError / reference.length),
  }
}

async function run() {
  const client = new OrtWorkerClient({ onStatus: (event) => { status.textContent = event.message } })
  try {
    const [q, k, v, official, canonicalOrt] = await Promise.all([
      fetchF32(`${fixture}/q.f32`, 4 * 1024),
      fetchF32(`${fixture}/k.f32`, 16 * 4101 * 64),
      fetchF32(`${fixture}/v.f32`, 16 * 4101 * 64),
      fetchF32(`${fixture}/official.f32`, 4 * 1024),
      fetchF32(`${fixture}/canonical_ort.f32`, 4 * 1024),
    ])
    const loaded = await client.loadSession({
      sessionId: 'context-attention-mha',
      manifest: { graphUrl: '/models/triposplat/context_attention_mha_probe.onnx' },
      options: { allowWasmFallback: false, graphOptimizationLevel: 'disabled' },
    })
    const result = await client.runSession({
      sessionId: 'context-attention-mha',
      inputs: {
        Q: createTensorPayload('float32', q, [1, 4, 1024]),
        K: createTensorPayload('float32', k, [1, 16, 4101, 64]),
        V: createTensorPayload('float32', v, [1, 16, 4101, 64]),
      },
      outputs: ['Y'],
      tag: 'context-attention-mha-inv08',
    })
    const output = result.outputs.Y
    if (!output || output.type !== 'float32' || !(output.data instanceof Float32Array)) {
      throw new Error('MultiHeadAttention returned no float32 Y output')
    }
    const report = {
      executionProvider: loaded.executionProvider,
      loadMs: loaded.loadMs,
      timings: result.timings,
      vsOfficial: metrics(official, output.data),
      vsCanonicalOrt: metrics(canonicalOrt, output.data),
    }
    ;(window as typeof window & { __CONTEXT_ATTENTION_MHA_RESULT__?: unknown })
      .__CONTEXT_ATTENTION_MHA_RESULT__ = report
    status.textContent = JSON.stringify(report, null, 2)
  } finally {
    await client.dispose().catch(() => undefined)
  }
}

run().catch((error) => {
  const message = error instanceof Error ? `${error.message}\n${error.stack ?? ''}` : String(error)
  ;(window as typeof window & { __CONTEXT_ATTENTION_MHA_ERROR__?: string })
    .__CONTEXT_ATTENTION_MHA_ERROR__ = message
  status.textContent = message
})
