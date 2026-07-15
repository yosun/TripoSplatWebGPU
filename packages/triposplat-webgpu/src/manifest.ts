import { ManifestError, ModelDownloadError } from './errors.js'
import type { Precision, TripoSplatGraphName } from './types.js'

export interface IntegrityDescriptor {
  algorithm: 'sha256'
  /** Lowercase hexadecimal digest. */
  digest: string
}

export interface ExternalDataDescriptor {
  /** Must exactly match the external-data location embedded in the ONNX graph. */
  path: string
  url: string
  byteLength?: number
  integrity?: IntegrityDescriptor
}

export interface GraphManifestEntry {
  url: string
  /** Internal parameter/compute precision recorded by the exporter. */
  precision?: Precision
  /** Public floating-point input dtype. TripoSplat exports default to fp32. */
  inputPrecision?: Precision
  byteLength?: number
  integrity?: IntegrityDescriptor
  externalData?: ExternalDataDescriptor[]
}

export interface TripoSplatModelManifest {
  name: 'triposplat-webgpu' | string
  version: string
  modelRevision: string
  precision: Precision
  estimatedModelBytes?: number
  estimatedPeakBytes?: number
  graphs: Partial<Record<TripoSplatGraphName, GraphManifestEntry>>
}

export interface ResolvedExternalDataDescriptor extends ExternalDataDescriptor {
  url: string
}

export interface ResolvedGraphManifestEntry extends GraphManifestEntry {
  url: string
  externalData?: ResolvedExternalDataDescriptor[]
}

export interface ResolvedTripoSplatModelManifest extends Omit<TripoSplatModelManifest, 'graphs'> {
  sourceUrl: string
  graphs: Partial<Record<TripoSplatGraphName, ResolvedGraphManifestEntry>>
}

const GRAPH_NAMES: readonly TripoSplatGraphName[] = [
  'dino',
  'vae',
  'dit',
  'octree',
  'gaussianDecoder',
]

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function stringField(value: unknown, label: string): string {
  if (typeof value !== 'string' || value.trim().length === 0 || value.includes('\0')) {
    throw new ManifestError(`${label} must be a non-empty string.`)
  }
  return value
}

function optionalByteLength(value: unknown, label: string): number | undefined {
  if (value === undefined) return undefined
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new ManifestError(`${label} must be a non-negative safe integer.`)
  }
  return value as number
}

function integrity(value: unknown, label: string): IntegrityDescriptor | undefined {
  if (value === undefined) return undefined
  if (!isRecord(value) || value.algorithm !== 'sha256') {
    throw new ManifestError(`${label} must use the sha256 algorithm.`)
  }
  const digest = stringField(value.digest, `${label}.digest`).toLowerCase()
  if (!/^[0-9a-f]{64}$/.test(digest)) {
    throw new ManifestError(`${label}.digest must be a 64-character hexadecimal SHA-256 digest.`)
  }
  return { algorithm: 'sha256', digest }
}

function externalData(value: unknown, label: string): ExternalDataDescriptor[] | undefined {
  if (value === undefined) return undefined
  if (!Array.isArray(value)) throw new ManifestError(`${label} must be an array.`)
  const paths = new Set<string>()
  return value.map((item, index) => {
    const itemLabel = `${label}[${index}]`
    if (!isRecord(item)) throw new ManifestError(`${itemLabel} must be an object.`)
    const path = stringField(item.path, `${itemLabel}.path`)
    if (paths.has(path)) throw new ManifestError(`${label} contains duplicate virtual path '${path}'.`)
    paths.add(path)
    const result: ExternalDataDescriptor = {
      path,
      url: stringField(item.url, `${itemLabel}.url`),
    }
    const length = optionalByteLength(item.byteLength, `${itemLabel}.byteLength`)
    const hash = integrity(item.integrity, `${itemLabel}.integrity`)
    if (length !== undefined) result.byteLength = length
    if (hash !== undefined) result.integrity = hash
    return result
  })
}

function graphEntry(value: unknown, label: string): GraphManifestEntry {
  if (!isRecord(value)) throw new ManifestError(`${label} must be an object.`)
  const result: GraphManifestEntry = { url: stringField(value.url, `${label}.url`) }
  if (value.precision !== undefined) {
    if (value.precision !== 'fp16' && value.precision !== 'fp32') {
      throw new ManifestError(`${label}.precision must be 'fp16' or 'fp32'.`)
    }
    result.precision = value.precision
  }
  if (value.inputPrecision !== undefined) {
    if (value.inputPrecision !== 'fp16' && value.inputPrecision !== 'fp32') {
      throw new ManifestError(`${label}.inputPrecision must be 'fp16' or 'fp32'.`)
    }
    result.inputPrecision = value.inputPrecision
  }
  const length = optionalByteLength(value.byteLength, `${label}.byteLength`)
  const hash = integrity(value.integrity, `${label}.integrity`)
  const data = externalData(value.externalData, `${label}.externalData`)
  if (length !== undefined) result.byteLength = length
  if (hash !== undefined) result.integrity = hash
  if (data !== undefined) result.externalData = data
  return result
}

/** Parse untrusted JSON without accepting unknown graph keys or ambiguous aliases. */
export function parseModelManifest(value: unknown): TripoSplatModelManifest {
  if (!isRecord(value)) throw new ManifestError('TripoSplat manifest must be an object.')
  if (!isRecord(value.graphs)) throw new ManifestError('manifest.graphs must be an object.')
  for (const key of Object.keys(value.graphs)) {
    if (!GRAPH_NAMES.some((name) => name === key)) {
      throw new ManifestError(`manifest.graphs contains unsupported graph '${key}'.`)
    }
  }
  if (value.precision !== 'fp16' && value.precision !== 'fp32') {
    throw new ManifestError("manifest.precision must be 'fp16' or 'fp32'.")
  }
  const graphs: Partial<Record<TripoSplatGraphName, GraphManifestEntry>> = {}
  for (const name of GRAPH_NAMES) {
    const entry = value.graphs[name]
    if (entry !== undefined) graphs[name] = graphEntry(entry, `manifest.graphs.${name}`)
  }
  const result: TripoSplatModelManifest = {
    name: stringField(value.name, 'manifest.name'),
    version: stringField(value.version, 'manifest.version'),
    modelRevision: stringField(value.modelRevision, 'manifest.modelRevision'),
    precision: value.precision,
    graphs,
  }
  const modelBytes = optionalByteLength(value.estimatedModelBytes, 'manifest.estimatedModelBytes')
  const peakBytes = optionalByteLength(value.estimatedPeakBytes, 'manifest.estimatedPeakBytes')
  if (modelBytes !== undefined) result.estimatedModelBytes = modelBytes
  if (peakBytes !== undefined) result.estimatedPeakBytes = peakBytes
  return result
}

export function resolveModelManifest(
  manifest: TripoSplatModelManifest,
  sourceUrl: string | URL,
): ResolvedTripoSplatModelManifest {
  const base = new URL(sourceUrl).href
  const graphs: ResolvedTripoSplatModelManifest['graphs'] = {}
  for (const name of GRAPH_NAMES) {
    const entry = manifest.graphs[name]
    if (!entry) continue
    const resolved: ResolvedGraphManifestEntry = {
      ...entry,
      url: new URL(entry.url, base).href,
    }
    if (entry.externalData !== undefined) {
      resolved.externalData = entry.externalData.map((descriptor) => ({
        ...descriptor,
        url: new URL(descriptor.url, base).href,
      }))
    }
    graphs[name] = resolved
  }
  return { ...manifest, sourceUrl: base, graphs }
}

export interface FetchModelManifestOptions {
  fetch?: typeof globalThis.fetch
  requestInit?: Omit<RequestInit, 'signal'>
  signal?: AbortSignal
}

export async function fetchModelManifest(
  url: string | URL,
  options: FetchModelManifestOptions = {},
): Promise<ResolvedTripoSplatModelManifest> {
  const sourceUrl = new URL(url).href
  const fetchImplementation = options.fetch ?? globalThis.fetch
  if (typeof fetchImplementation !== 'function') {
    throw new ModelDownloadError('No Fetch implementation is available for the model manifest.', {
      diagnostics: { url: sourceUrl },
    })
  }
  let response: Response
  try {
    const init: RequestInit = { ...options.requestInit }
    if (options.signal !== undefined) init.signal = options.signal
    response = await fetchImplementation(sourceUrl, init)
  } catch (cause) {
    throw new ModelDownloadError(`Could not download TripoSplat manifest from '${sourceUrl}'.`, {
      cause,
      diagnostics: { url: sourceUrl },
    })
  }
  if (!response.ok) {
    throw new ModelDownloadError(
      `TripoSplat manifest request failed with HTTP ${response.status} ${response.statusText}.`,
      { diagnostics: { url: sourceUrl, status: response.status } },
    )
  }
  let json: unknown
  try {
    json = await response.json()
  } catch (cause) {
    throw new ManifestError('TripoSplat manifest is not valid JSON.', {
      cause,
      diagnostics: { url: sourceUrl },
    })
  }
  return resolveModelManifest(parseModelManifest(json), sourceUrl)
}

export function configuredGraphNames(
  manifest: Pick<TripoSplatModelManifest, 'graphs'>,
): TripoSplatGraphName[] {
  return GRAPH_NAMES.filter((name) => manifest.graphs[name] !== undefined)
}

export const REQUIRED_GENERATION_GRAPHS = GRAPH_NAMES
