import { WebGPUUnavailableError } from './errors.js'
import type { CompatibilityOptions, CompatibilityReport } from './types.js'

interface AdapterLike {
  limits: Record<string, number> & { maxStorageBufferBindingSize?: number }
  info?: { description?: string; vendor?: string; architecture?: string }
}

interface GpuLike {
  requestAdapter(options?: CompatibilityOptions['requestAdapterOptions']): Promise<AdapterLike | null>
}

function browserDescription(): string {
  return typeof navigator === 'undefined' ? 'unknown' : navigator.userAgent
}

function gpu(): GpuLike | undefined {
  if (typeof navigator === 'undefined') return undefined
  return (navigator as Navigator & { gpu?: GpuLike }).gpu
}

function numericLimits(adapter: AdapterLike): Record<string, number> {
  const limits: Record<string, number> = {}
  const names = [
    'maxBufferSize',
    'maxStorageBufferBindingSize',
    'maxComputeWorkgroupStorageSize',
    'maxComputeInvocationsPerWorkgroup',
    'maxStorageBuffersPerShaderStage',
    'maxBindGroups',
  ]
  for (const name of names) {
    const value = adapter.limits[name]
    if (typeof value === 'number') limits[name] = value
  }
  return limits
}

export async function checkCompatibility(
  options: CompatibilityOptions = {},
): Promise<CompatibilityReport> {
  const estimatedModelBytes = options.estimatedModelBytes ?? 0
  const measuredPeak = options.estimatedPeakBytes === undefined
    ? {}
    : { estimatedPeakBytes: options.estimatedPeakBytes }
  const warnings = [
    'Browser APIs do not expose reliable total GPU or unified-memory capacity.',
    'A successful adapter check does not guarantee that every exported graph will fit in memory.',
  ]
  const blockers: string[] = []
  const webgpu = gpu()
  if (!webgpu) {
    blockers.push('navigator.gpu is unavailable; use a secure context and a WebGPU-capable browser.')
    return {
      supported: false,
      level: 'unsupported',
      browser: browserDescription(),
      webgpu: false,
      limits: {},
      estimatedModelBytes,
      ...measuredPeak,
      warnings,
      blockers,
    }
  }
  let adapter: AdapterLike | null
  try {
    adapter = await webgpu.requestAdapter(options.requestAdapterOptions)
  } catch (cause) {
    blockers.push(`WebGPU adapter request failed: ${cause instanceof Error ? cause.message : String(cause)}`)
    adapter = null
  }
  if (!adapter) {
    blockers.push('No WebGPU adapter was returned.')
    return {
      supported: false,
      level: 'unsupported',
      browser: browserDescription(),
      webgpu: true,
      limits: {},
      estimatedModelBytes,
      ...measuredPeak,
      warnings,
      blockers,
    }
  }
  const limits = numericLimits(adapter)
  if (options.minimumStorageBufferBindingSize !== undefined) {
    const actual = limits.maxStorageBufferBindingSize ?? 0
    if (actual < options.minimumStorageBufferBindingSize) {
      blockers.push(
        `maxStorageBufferBindingSize ${actual} is below the graph requirement ${options.minimumStorageBufferBindingSize}.`,
      )
    }
  }
  const adapterName = adapter.info?.description
    || [adapter.info?.vendor, adapter.info?.architecture].filter(Boolean).join(' ')
    || undefined
  const report: CompatibilityReport = {
    supported: blockers.length === 0,
    level: blockers.length === 0 ? 'experimental' : 'unsupported',
    browser: browserDescription(),
    webgpu: true,
    limits,
    estimatedModelBytes,
    ...measuredPeak,
    warnings,
    blockers,
  }
  if (adapterName) report.adapterName = adapterName
  return report
}

export async function assertCompatible(options?: CompatibilityOptions): Promise<CompatibilityReport> {
  const report = await checkCompatibility(options)
  if (!report.webgpu) throw new WebGPUUnavailableError(undefined, { diagnostics: { report } })
  if (!report.supported) {
    throw new WebGPUUnavailableError(report.blockers.join(' '), { diagnostics: { report } })
  }
  return report
}
