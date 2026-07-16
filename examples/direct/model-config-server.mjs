import { createServer } from 'node:http'
import { pathToFileURL } from 'node:url'

/**
 * Optional same-origin configuration endpoint for server-rendered platforms.
 * It exposes a public CDN URL only; image generation still runs in the browser.
 */
export function createTripoSplatConfigHandler({
  modelBaseUrl = process.env.TRIPOSPLAT_MODEL_BASE_URL,
} = {}) {
  if (!modelBaseUrl) {
    throw new Error('Set TRIPOSPLAT_MODEL_BASE_URL to the public model directory URL.')
  }

  const body = JSON.stringify({
    modelBaseUrl: new URL(modelBaseUrl).href,
    manifestUrl: 'manifest.json',
  })

  return function handleTripoSplatConfig(request, response) {
    if (request.method !== 'GET' || request.url !== '/api/triposplat-config') {
      response.writeHead(404).end('Not found')
      return
    }
    response.writeHead(200, {
      'cache-control': 'public, max-age=300',
      'content-length': Buffer.byteLength(body),
      'content-type': 'application/json; charset=utf-8',
    })
    response.end(body)
  }
}

const isMain = process.argv[1]
  && import.meta.url === pathToFileURL(process.argv[1]).href

if (isMain) {
  const port = Number.parseInt(process.env.PORT ?? '3000', 10)
  const server = createServer(createTripoSplatConfigHandler())
  server.listen(port, '127.0.0.1', () => {
    console.info(`TripoSplat config: http://127.0.0.1:${port}/api/triposplat-config`)
  })
}
