/** Correctness-first host octree sampler translated from TripoSplat's decoder. */

export const TRIPOSPLAT_OCTREE_LEVELS = 8
export const OCTREE_CHILDREN = 8

/** Must return a uniform variate in [0, 1). */
export type RandomSource = () => number

export interface OctreeFrontier {
  /** xyz-interleaved integer voxel coordinates at the frontier's level. */
  coordinates: Int32Array
  /** Number of final samples represented by each occupied voxel. */
  counts: Uint32Array
  /** Accumulated model log probability for each occupied voxel. */
  logProbabilities: Float32Array
}

export interface OctreeChildDistribution {
  /** Parent-major arrays with exactly eight entries per parent. */
  probabilities: Float32Array
  logProbabilities: Float32Array
}

export interface OctreeOccupancyInvocation<Condition> {
  /** `[1, parentCount, 3]`, represented as xyz-interleaved float32 centers. */
  parentCenters: Float32Array
  parentCount: number
  /** One-based level being expanded. */
  level: number
  /** Child grid resolution (`2 ** level`), matching the official `l` input. */
  resolution: number
  /** Fixed sample count, matching the official `l2` input. */
  targetPointCount: number
  condition: Condition
  signal?: AbortSignal
}

export interface OctreeOccupancyOutput {
  /** `[1, parentCount, 8]` logits from the occupancy decoder. */
  logits: Float32Array
}

export type OctreeOccupancyPredictor<Condition> = (
  invocation: OctreeOccupancyInvocation<Condition>,
) => OctreeOccupancyOutput | Promise<OctreeOccupancyOutput>

export interface OctreeLevelProgress {
  level: number
  totalLevels: number
  resolution: number
  occupiedVoxels: number
  targetPointCount: number
}

export interface OctreeSampleOptions<Condition> {
  condition: Condition
  numPoints: number
  levels?: number
  temperature?: number
  rng?: RandomSource
  signal?: AbortSignal
  onLevel?: (progress: OctreeLevelProgress) => void
}

export interface OctreeSampleResult {
  /** `[1, numPoints, 3]`, xyz-interleaved and normalized to [0, 1). */
  points: Float32Array
  /** `[1, numPoints]`; the leaf path log probability repeated by leaf count. */
  logProbabilities: Float32Array
  /** Compact occupied leaf representation retained for debugging/WGSL parity. */
  leaves: OctreeFrontier
  resolution: number
}

function assertPositiveInteger(value: number, label: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive integer, got ${value}.`)
  }
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw signal.reason instanceof Error
      ? signal.reason
      : new DOMException('Operation aborted', 'AbortError')
  }
}

function randomUnit(rng: RandomSource): number {
  const value = rng()
  if (!Number.isFinite(value) || value < 0 || value >= 1) {
    throw new Error(`Octree RNG must return a finite value in [0, 1), got ${value}.`)
  }
  return value
}

function assertFrontier(frontier: Readonly<OctreeFrontier>): number {
  const nodeCount = frontier.counts.length
  if (frontier.coordinates.length !== nodeCount * 3) {
    throw new Error(
      `Octree frontier has ${frontier.coordinates.length} coordinate values for ${nodeCount} nodes.`,
    )
  }
  if (frontier.logProbabilities.length !== nodeCount) {
    throw new Error(
      `Octree frontier has ${frontier.logProbabilities.length} log probabilities for ` +
        `${nodeCount} nodes.`,
    )
  }
  return nodeCount
}

function totalFrontierCount(frontier: Readonly<OctreeFrontier>): number {
  let total = 0
  for (const count of frontier.counts) total += count
  return total
}

/**
 * Official `sample_probs(..., algo="systematic")` for one probability row.
 * Negative values are clamped, mass above one is normalized down, positive
 * underfull rows are retained, and an all-zero row becomes uniform.
 */
export function systematicResample(
  probabilities: ArrayLike<number>,
  count: number,
  rng: RandomSource = Math.random,
): Uint32Array {
  if (!Number.isInteger(count) || count < 0) {
    throw new Error(`count must be a non-negative integer, got ${count}.`)
  }
  if (probabilities.length === 0) {
    throw new Error('Systematic resampling requires at least one probability.')
  }
  // Upstream casts the row to float32 before every normalization/sampling
  // operation. Preserve those rounding points so CDF boundary behavior agrees
  // with torch.searchsorted rather than JavaScript's float64 arithmetic.
  const normalized = new Float32Array(probabilities.length)
  let sum = 0
  for (let index = 0; index < probabilities.length; index += 1) {
    const value = probabilities[index]
    if (!Number.isFinite(value)) {
      throw new Error(`Probability at index ${index} is not finite: ${value}.`)
    }
    const clamped = Math.max(0, value)
    normalized[index] = clamped
    sum = Math.fround(sum + normalized[index])
  }
  if (sum === 0) {
    normalized.fill(Math.fround(1 / probabilities.length))
  } else {
    // Official sample_probs divides by row_sums.clamp_min_(1): it scales rows
    // down when their mass exceeds one, but deliberately does not scale a
    // positive underfull row up. Targets beyond its CDF land in the final bin.
    const denominator = Math.max(sum, 1)
    for (let index = 0; index < normalized.length; index += 1) {
      normalized[index] = Math.fround(normalized[index] / denominator)
    }
  }

  const output = new Uint32Array(probabilities.length)
  if (count === 0) return output
  const cdf = new Float32Array(probabilities.length)
  let cumulative = 0
  const maximum = Math.fround(1 - 1e-12)
  for (let index = 0; index < normalized.length; index += 1) {
    cumulative = Math.fround(cumulative + normalized[index])
    cdf[index] = Math.min(cumulative, maximum)
  }

  const initial = Math.fround(Math.fround(randomUnit(rng)) / count)
  for (let sample = 0; sample < count; sample += 1) {
    const grid = Math.fround(sample / count)
    const target = Math.min(Math.fround(initial + grid), maximum)
    // `torch.searchsorted(..., right=False)`: first CDF value >= target.
    let low = 0
    let high = cdf.length
    while (low < high) {
      const middle = low + Math.floor((high - low) / 2)
      if (cdf[middle] < target) low = middle + 1
      else high = middle
    }
    output[Math.min(low, output.length - 1)] += 1
  }
  return output
}

/** Stable row-wise softmax and log-softmax over the eight occupancy logits. */
export function octreeSoftmax(
  logits: Float32Array,
  parentCount: number,
  temperature = 1,
): OctreeChildDistribution {
  if (logits.length !== parentCount * OCTREE_CHILDREN) {
    throw new Error(
      `Occupancy decoder returned ${logits.length} logits for ${parentCount} parents; ` +
        `expected ${parentCount * OCTREE_CHILDREN}.`,
    )
  }
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new Error(`temperature must be a positive finite number, got ${temperature}.`)
  }
  const probabilities = new Float32Array(logits.length)
  const logProbabilities = new Float32Array(logits.length)
  for (let parent = 0; parent < parentCount; parent += 1) {
    const offset = parent * OCTREE_CHILDREN
    let maximum = -Infinity
    for (let child = 0; child < OCTREE_CHILDREN; child += 1) {
      const value = logits[offset + child] / temperature
      if (!Number.isFinite(value)) {
        throw new Error(`Occupancy logit at index ${offset + child} is not finite: ${value}.`)
      }
      maximum = Math.max(maximum, value)
    }
    let exponentialSum = 0
    for (let child = 0; child < OCTREE_CHILDREN; child += 1) {
      exponentialSum += Math.exp(logits[offset + child] / temperature - maximum)
    }
    const logDenominator = maximum + Math.log(exponentialSum)
    for (let child = 0; child < OCTREE_CHILDREN; child += 1) {
      const logProbability = logits[offset + child] / temperature - logDenominator
      logProbabilities[offset + child] = logProbability
      probabilities[offset + child] = Math.exp(logProbability)
    }
  }
  return { probabilities, logProbabilities }
}

/**
 * Resamples each parent's eight children, expands integer Morton-style child
 * coordinates, and compacts children whose sampled count is zero.
 */
export function expandOctreeFrontier(
  frontier: Readonly<OctreeFrontier>,
  distribution: Readonly<OctreeChildDistribution>,
  rng: RandomSource = Math.random,
): OctreeFrontier {
  const parentCount = assertFrontier(frontier)
  const scoreCount = parentCount * OCTREE_CHILDREN
  if (
    distribution.probabilities.length !== scoreCount ||
    distribution.logProbabilities.length !== scoreCount
  ) {
    throw new Error(
      `Octree child distribution must contain ${scoreCount} probabilities and log probabilities.`,
    )
  }

  const coordinates: number[] = []
  const counts: number[] = []
  const logProbabilities: number[] = []
  // `torch.unique` orders the distinct parent counts before sample_probs draws
  // one systematic offset for each row in a count group. Match that RNG
  // assignment rather than consuming offsets in frontier order.
  const systematicOffsets = new Float32Array(parentCount)
  const parentsByCount = new Map<number, number[]>()
  for (let parent = 0; parent < parentCount; parent += 1) {
    const count = frontier.counts[parent]
    if (count === 0) continue
    const parents = parentsByCount.get(count)
    if (parents) parents.push(parent)
    else parentsByCount.set(count, [parent])
  }
  for (const count of [...parentsByCount.keys()].sort((left, right) => left - right)) {
    for (const parent of parentsByCount.get(count) ?? []) {
      systematicOffsets[parent] = randomUnit(rng)
    }
  }
  for (let parent = 0; parent < parentCount; parent += 1) {
    const scoreOffset = parent * OCTREE_CHILDREN
    const sampled = systematicResample(
      distribution.probabilities.subarray(scoreOffset, scoreOffset + OCTREE_CHILDREN),
      frontier.counts[parent],
      () => systematicOffsets[parent],
    )
    const coordinateOffset = parent * 3
    for (let child = 0; child < OCTREE_CHILDREN; child += 1) {
      const childCount = sampled[child]
      if (childCount === 0) continue
      // Official order: x changes fastest, then y, then z.
      coordinates.push(
        frontier.coordinates[coordinateOffset] * 2 + (child & 1),
        frontier.coordinates[coordinateOffset + 1] * 2 + ((child >> 1) & 1),
        frontier.coordinates[coordinateOffset + 2] * 2 + ((child >> 2) & 1),
      )
      counts.push(childCount)
      logProbabilities.push(
        frontier.logProbabilities[parent] + distribution.logProbabilities[scoreOffset + child],
      )
    }
  }

  const expanded: OctreeFrontier = {
    coordinates: Int32Array.from(coordinates),
    counts: Uint32Array.from(counts),
    logProbabilities: Float32Array.from(logProbabilities),
  }
  if (totalFrontierCount(expanded) !== totalFrontierCount(frontier)) {
    throw new Error('Systematic octree expansion did not preserve the requested sample count.')
  }
  return expanded
}

/** Builds normalized parent voxel centers exactly as the occupancy graph consumes them. */
export function buildOctreeParentCenters(
  frontier: Readonly<OctreeFrontier>,
  parentResolution: number,
): Float32Array {
  const parentCount = assertFrontier(frontier)
  assertPositiveInteger(parentResolution, 'parentResolution')
  const centers = new Float32Array(parentCount * 3)
  for (let index = 0; index < centers.length; index += 1) {
    centers[index] = (frontier.coordinates[index] + 0.5) / parentResolution
  }
  return centers
}

/** Repeats compact leaves by count and applies independent xyz jitter within each voxel. */
export function expandOctreeLeavesToPoints(
  leaves: Readonly<OctreeFrontier>,
  resolution: number,
  rng: RandomSource = Math.random,
): Pick<OctreeSampleResult, 'points' | 'logProbabilities'> {
  const leafCount = assertFrontier(leaves)
  assertPositiveInteger(resolution, 'resolution')
  const pointCount = totalFrontierCount(leaves)
  const points = new Float32Array(pointCount * 3)
  const logProbabilities = new Float32Array(pointCount)
  let point = 0
  for (let leaf = 0; leaf < leafCount; leaf += 1) {
    const coordinateOffset = leaf * 3
    for (let repeat = 0; repeat < leaves.counts[leaf]; repeat += 1) {
      const pointOffset = point * 3
      points[pointOffset] = (leaves.coordinates[coordinateOffset] + randomUnit(rng)) / resolution
      points[pointOffset + 1] =
        (leaves.coordinates[coordinateOffset + 1] + randomUnit(rng)) / resolution
      points[pointOffset + 2] =
        (leaves.coordinates[coordinateOffset + 2] + randomUnit(rng)) / resolution
      logProbabilities[point] = leaves.logProbabilities[leaf]
      point += 1
    }
  }
  return { points, logProbabilities }
}

/**
 * Runs all eight dynamic occupancy/resampling levels on the host. The callback
 * is the only model-specific piece and can wrap an ONNX Runtime WebGPU session.
 */
export async function sampleOctree<Condition>(
  predictOccupancy: OctreeOccupancyPredictor<Condition>,
  options: OctreeSampleOptions<Condition>,
): Promise<OctreeSampleResult> {
  assertPositiveInteger(options.numPoints, 'numPoints')
  const levels = options.levels ?? TRIPOSPLAT_OCTREE_LEVELS
  if (!Number.isInteger(levels) || levels < 1 || levels > TRIPOSPLAT_OCTREE_LEVELS) {
    throw new Error(
      `levels must be an integer in [1, ${TRIPOSPLAT_OCTREE_LEVELS}], got ${levels}.`,
    )
  }
  const temperature = options.temperature ?? 1
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new Error(`temperature must be a positive finite number, got ${temperature}.`)
  }
  const rng = options.rng ?? Math.random

  let frontier: OctreeFrontier = {
    coordinates: Int32Array.of(0, 0, 0),
    counts: Uint32Array.of(options.numPoints),
    logProbabilities: Float32Array.of(0),
  }

  for (let level = 1; level <= levels; level += 1) {
    throwIfAborted(options.signal)
    const parentResolution = 2 ** (level - 1)
    const resolution = 2 ** level
    const parentCount = frontier.counts.length
    const output = await predictOccupancy({
      parentCenters: buildOctreeParentCenters(frontier, parentResolution),
      parentCount,
      level,
      resolution,
      targetPointCount: options.numPoints,
      condition: options.condition,
      signal: options.signal,
    })
    throwIfAborted(options.signal)
    const distribution = octreeSoftmax(output.logits, parentCount, temperature)
    frontier = expandOctreeFrontier(frontier, distribution, rng)
    options.onLevel?.({
      level,
      totalLevels: levels,
      resolution,
      occupiedVoxels: frontier.counts.length,
      targetPointCount: options.numPoints,
    })
  }

  const resolution = 2 ** levels
  const expanded = expandOctreeLeavesToPoints(frontier, resolution, rng)
  if (expanded.logProbabilities.length !== options.numPoints) {
    throw new Error(
      `Octree produced ${expanded.logProbabilities.length} points; expected ${options.numPoints}.`,
    )
  }
  return {
    ...expanded,
    leaves: frontier,
    resolution,
  }
}
