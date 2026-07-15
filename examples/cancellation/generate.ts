/** Alpha example: cancellation/retry has unit coverage; real-browser stress qualification remains open. */
import { CancelledError, TripoSplatWebGPU } from '@ai3d/triposplat-webgpu'

export async function generateWithCancellation(input: Blob, modelBaseUrl: string) {
  const model = new TripoSplatWebGPU({ modelBaseUrl })
  let controller = new AbortController()

  try {
    await model.load({ signal: controller.signal })

    const firstAttempt = model.generate(input, {
      steps: 20,
      signal: controller.signal,
      onProgress: ({ invocation, totalInvocations, message }) => {
        console.info(message, { invocation, totalInvocations })
      },
    })

    document.querySelector('#cancel')?.addEventListener(
      'click',
      () => controller.abort('Cancelled by the user'),
      { once: true },
    )

    try {
      return await firstAttempt
    } catch (error) {
      if (!(error instanceof CancelledError)) throw error

      // The package recreates its worker after hard cancellation. Use a fresh signal.
      controller = new AbortController()
      return await model.generate(input, {
        steps: 4,
        signal: controller.signal,
      })
    }
  } finally {
    controller.abort()
    await model.dispose()
  }
}
