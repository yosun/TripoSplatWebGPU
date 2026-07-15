export const TRIPOSPLAT_E2E_GAUSSIAN_COUNT = 262_144

export interface NumericArraySummary {
  present: boolean
  length: number
  expectedLength: number
  lengthMatches: boolean
  finite: boolean
  nonFiniteCount: number
  minimum: number | null
  maximum: number | null
}

export interface GaussianSceneStructuralQualification {
  passed: boolean
  scope: 'scene-structure-and-finiteness'
  count: number
  expectedCount: number
  countMatches: boolean
  arrays: {
    positions: NumericArraySummary
    scales: NumericArraySummary
    rotations: NumericArraySummary
    opacities: NumericArraySummary
    sphericalHarmonics: NumericArraySummary
  }
  metadata: {
    fourStepSchedule: boolean
    preparedInputRecorded: boolean
    degreeZeroSphericalHarmonics: boolean
    linearScales: boolean
    linearOpacities: boolean
  }
  sceneWasLive: boolean
}

export interface GaussianSceneQualificationInput {
  readonly count: number
  readonly positions: Float32Array
  readonly scales: Float32Array
  readonly rotations: Float32Array
  readonly opacities: Float32Array
  readonly sphericalHarmonics?: Float32Array
  readonly metadata: {
    readonly scaleEncoding: string
    readonly opacityEncoding: string
    readonly sphericalHarmonicsSemantics: string | null
    readonly generationSettings: Readonly<Record<string, unknown>>
  }
  readonly isDisposed: boolean
}

export interface GaussianViewerObservation {
  status: 'ready' | 'failed'
  message: string
  loadMs: number
  timeoutMs: number
  canvas: {
    present: boolean
    width: number
    height: number
    clientWidth: number
    clientHeight: number
  }
}

export interface GaussianViewerQualification extends GaussianViewerObservation {
  passed: boolean
  scope: 'exported-ply-viewer-load-and-canvas-sanity'
  canvas: GaussianViewerObservation['canvas'] & {
    drawingBufferDimensionsPresent: boolean
    displayDimensionsPresent: boolean
  }
}

function summarizeArray(
  values: Float32Array | undefined,
  expectedLength: number,
): NumericArraySummary {
  if (values === undefined) {
    return {
      present: false,
      length: 0,
      expectedLength,
      lengthMatches: false,
      finite: false,
      nonFiniteCount: 0,
      minimum: null,
      maximum: null,
    }
  }

  let minimum = Number.POSITIVE_INFINITY
  let maximum = Number.NEGATIVE_INFINITY
  let nonFiniteCount = 0
  for (const value of values) {
    if (!Number.isFinite(value)) {
      nonFiniteCount += 1
      continue
    }
    minimum = Math.min(minimum, value)
    maximum = Math.max(maximum, value)
  }
  return {
    present: true,
    length: values.length,
    expectedLength,
    lengthMatches: values.length === expectedLength,
    finite: nonFiniteCount === 0,
    nonFiniteCount,
    minimum: minimum === Number.POSITIVE_INFINITY ? null : minimum,
    maximum: maximum === Number.NEGATIVE_INFINITY ? null : maximum,
  }
}

/** Pure structural gate shared by the expensive browser lab and fast Node tests. */
export function qualifyGaussianSceneStructure(
  scene: GaussianSceneQualificationInput,
  expectedCount = TRIPOSPLAT_E2E_GAUSSIAN_COUNT,
): GaussianSceneStructuralQualification {
  const arrays = {
    positions: summarizeArray(scene.positions, expectedCount * 3),
    scales: summarizeArray(scene.scales, expectedCount * 3),
    rotations: summarizeArray(scene.rotations, expectedCount * 4),
    opacities: summarizeArray(scene.opacities, expectedCount),
    sphericalHarmonics: summarizeArray(scene.sphericalHarmonics, expectedCount * 3),
  }
  const metadata = {
    fourStepSchedule: scene.metadata.generationSettings.steps === 4,
    preparedInputRecorded: scene.metadata.generationSettings.inputIsPrepared === true,
    degreeZeroSphericalHarmonics:
      scene.metadata.sphericalHarmonicsSemantics === 'degree-0-rgb',
    linearScales: scene.metadata.scaleEncoding === 'linear',
    linearOpacities: scene.metadata.opacityEncoding === 'linear',
  }
  const countMatches = scene.count === expectedCount
  const sceneWasLive = !scene.isDisposed
  const passed = countMatches
    && sceneWasLive
    && Object.values(arrays).every((summary) =>
      summary.present && summary.lengthMatches && summary.finite,
    )
    && Object.values(metadata).every(Boolean)

  return {
    passed,
    scope: 'scene-structure-and-finiteness',
    count: scene.count,
    expectedCount,
    countMatches,
    arrays,
    metadata,
    sceneWasLive,
  }
}

/** Pure viewer gate shared by the browser lab and fast Node tests. */
export function qualifyGaussianViewer(
  observation: GaussianViewerObservation,
): GaussianViewerQualification {
  const drawingBufferDimensionsPresent = observation.canvas.width > 0
    && observation.canvas.height > 0
  const displayDimensionsPresent = observation.canvas.clientWidth > 0
    && observation.canvas.clientHeight > 0
  const canvas = {
    ...observation.canvas,
    drawingBufferDimensionsPresent,
    displayDimensionsPresent,
  }
  return {
    ...observation,
    passed: observation.status === 'ready'
      && canvas.present
      && drawingBufferDimensionsPresent
      && displayDimensionsPresent,
    scope: 'exported-ply-viewer-load-and-canvas-sanity',
    canvas,
  }
}
