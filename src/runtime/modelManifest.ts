/**
 * The URL and the virtual ONNX path of one external tensor-data file.
 *
 * `path` must exactly match the `location` stored in the ONNX graph. `url` is
 * where the browser should fetch that file. They are deliberately separate:
 * large exports commonly use a short relative ONNX path while hosting the
 * bytes at a versioned CDN URL.
 */
export interface OnnxExternalDataDescriptor {
  path: string
  url: string
}

/** A single ONNX graph and every external-data file it requires. */
export interface OnnxModelManifest {
  graphUrl: string
  externalData?: readonly OnnxExternalDataDescriptor[]
}

export type ModelManifest = OnnxModelManifest
export type ExternalDataDescriptor = OnnxExternalDataDescriptor

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function assertNonEmptyString(value: unknown, label: string): asserts value is string {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new TypeError(`${label} must be a non-empty string.`)
  }
  if (value.includes('\0')) {
    throw new TypeError(`${label} must not contain a null character.`)
  }
}

/** Validate a manifest received from configuration or a structured-clone boundary. */
export function assertModelManifest(value: unknown): asserts value is OnnxModelManifest {
  if (!isRecord(value)) {
    throw new TypeError('ONNX model manifest must be an object.')
  }

  assertNonEmptyString(value.graphUrl, 'manifest.graphUrl')

  if (value.externalData === undefined) {
    return
  }
  if (!Array.isArray(value.externalData)) {
    throw new TypeError('manifest.externalData must be an array when provided.')
  }

  const paths = new Set<string>()
  for (let index = 0; index < value.externalData.length; index += 1) {
    const descriptor: unknown = value.externalData[index]
    if (!isRecord(descriptor)) {
      throw new TypeError(`manifest.externalData[${index}] must be an object.`)
    }

    assertNonEmptyString(descriptor.path, `manifest.externalData[${index}].path`)
    assertNonEmptyString(descriptor.url, `manifest.externalData[${index}].url`)

    if (paths.has(descriptor.path)) {
      throw new TypeError(`Duplicate ONNX external-data path '${descriptor.path}'.`)
    }
    paths.add(descriptor.path)
  }
}

/** Return a mutable, structured-clone-safe copy after validating the descriptor. */
export function copyModelManifest(manifest: OnnxModelManifest): OnnxModelManifest {
  assertModelManifest(manifest)
  return {
    graphUrl: manifest.graphUrl,
    externalData: manifest.externalData?.map(({ path, url }) => ({ path, url })),
  }
}

/** Resolve graph and data URLs without changing the virtual external-data paths. */
export function resolveModelManifest(manifest: OnnxModelManifest, baseUrl: string | URL): OnnxModelManifest {
  assertModelManifest(manifest)
  const base = baseUrl instanceof URL ? baseUrl.href : baseUrl
  assertNonEmptyString(base, 'baseUrl')

  return {
    graphUrl: new URL(manifest.graphUrl, base).href,
    externalData: manifest.externalData?.map(({ path, url }) => ({
      path,
      url: new URL(url, base).href,
    })),
  }
}

export interface ConventionalExternalDataOptions {
  /** Defaults to `<graph filename>.data`, matching common ONNX exporters. */
  externalDataPath?: string
  /** Defaults to the graph URL with `.data` appended before its query/hash. */
  externalDataUrl?: string
}

function appendBeforeQueryAndHash(url: string, suffix: string): string {
  const queryIndex = url.indexOf('?')
  const hashIndex = url.indexOf('#')
  const firstSuffixIndex = Math.min(
    queryIndex === -1 ? url.length : queryIndex,
    hashIndex === -1 ? url.length : hashIndex,
  )
  return `${url.slice(0, firstSuffixIndex)}${suffix}${url.slice(firstSuffixIndex)}`
}

function graphFileName(graphUrl: string): string {
  const withoutQueryOrHash = graphUrl.split(/[?#]/, 1)[0]
  const fileName = withoutQueryOrHash.slice(withoutQueryOrHash.lastIndexOf('/') + 1)
  if (fileName.length === 0) {
    throw new TypeError(`Cannot infer an external-data path from graph URL '${graphUrl}'.`)
  }
  try {
    return decodeURIComponent(fileName)
  } catch {
    return fileName
  }
}

/**
 * Opt-in compatibility helper for the conventional `model.onnx.data` sidecar.
 *
 * The generic loader never calls this helper itself. Prefer an explicit
 * `externalData` array for exported models, especially multi-file exports.
 */
export function createConventionalExternalDataManifest(
  graphUrl: string,
  options: ConventionalExternalDataOptions = {},
): OnnxModelManifest {
  assertNonEmptyString(graphUrl, 'graphUrl')
  const externalDataPath = options.externalDataPath ?? `${graphFileName(graphUrl)}.data`
  const externalDataUrl = options.externalDataUrl ?? appendBeforeQueryAndHash(graphUrl, '.data')
  assertNonEmptyString(externalDataPath, 'options.externalDataPath')
  assertNonEmptyString(externalDataUrl, 'options.externalDataUrl')

  const manifest: OnnxModelManifest = {
    graphUrl,
    externalData: [{ path: externalDataPath, url: externalDataUrl }],
  }
  assertModelManifest(manifest)
  return manifest
}
