/** Alpha custom-fetch example; production-origin CDN behavior is not yet qualified. */
import { TripoSplatWebGPU } from '@ai3d/triposplat-webgpu'

const authenticatedFetch: typeof fetch = async (input, init = {}) => {
  const token = await getShortLivedModelToken()
  const headers = new Headers(init.headers)
  headers.set('Authorization', `Bearer ${token}`)
  return fetch(input, { ...init, headers, credentials: 'omit' })
}

const model = new TripoSplatWebGPU({
  modelBaseUrl: 'https://models.example.com/triposplat-webgpu/v1/',
  // This endpoint may redirect to or return a manifest containing signed graph URLs.
  manifestUrl: 'https://auth.example.com/model-manifests/triposplat-v1',
  fetch: authenticatedFetch,
  cache: 'opfs',
})

await model.load()

async function getShortLivedModelToken(): Promise<string> {
  const response = await fetch('/api/model-token', { credentials: 'same-origin' })
  if (!response.ok) throw new Error(`Token request failed: HTTP ${response.status}`)
  const body: unknown = await response.json()
  if (!body || typeof body !== 'object' || !('token' in body) || typeof body.token !== 'string') {
    throw new Error('Token response did not contain a string token.')
  }
  return body.token
}
