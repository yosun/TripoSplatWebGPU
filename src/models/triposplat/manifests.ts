import {
  createConventionalExternalDataManifest,
  type OnnxModelManifest,
} from '../../runtime/modelManifest.ts'

export type TripoSplatGraphName =
  | 'dinov3'
  | 'vaeEncoder'
  | 'dit'
  | 'octree'
  | 'gaussianDecoder'

export type TripoSplatGraphPrecision = 'float32' | 'float16'

export interface TripoSplatGraphDescriptor {
  manifest: OnnxModelManifest
  /** Internal parameter/compute precision; distinct from the public input dtype. */
  internalPrecision?: TripoSplatGraphPrecision
  precision: TripoSplatGraphPrecision
}

export type TripoSplatModelGraphs = Partial<Record<TripoSplatGraphName, TripoSplatGraphDescriptor>>

export interface TripoSplatManifestOptions {
  /** Versioned CDN prefix. Defaults to the Vite environment setting, then local public/models. */
  baseUrl?: string
  /** Only the VAE exporter optionally changes public inputs to FP16. */
  vaeInputPrecision?: TripoSplatGraphPrecision
}

const GRAPH_FILES: Record<TripoSplatGraphName, string> = {
  // ORT WebGPU's FP16 ViT-H reductions drift outside the official parity gate.
  // The staged FP32 export is the validated browser baseline.
  dinov3: 'dinov3_encoder_fp32.onnx',
  vaeEncoder: 'flux2_vae_encoder.onnx',
  dit: 'dit_step_webgpu_fp32.onnx',
  octree: 'octree_occupancy_decoder_fp32.onnx',
  gaussianDecoder: 'gaussian_decoder_fp32.onnx',
}

function slashTerminated(value: string): string {
  return value.endsWith('/') ? value : `${value}/`
}

function defaultBaseUrl(): string {
  const configured = import.meta.env.VITE_TRIPOSPLAT_MODEL_BASE_URL as string | undefined
  return slashTerminated(configured?.trim() || `${import.meta.env.BASE_URL}models/triposplat/`)
}

/**
 * Build explicit graph + sidecar descriptors. Hosting may use different URLs,
 * but `path` must continue to match the ONNX external-data `location` exactly.
 */
export function createTripoSplatModelManifest(
  options: TripoSplatManifestOptions = {},
): TripoSplatModelGraphs {
  const baseUrl = slashTerminated(options.baseUrl?.trim() || defaultBaseUrl())
  const names: TripoSplatGraphName[] = [
    'dinov3',
    'vaeEncoder',
    'dit',
    'octree',
    'gaussianDecoder',
  ]

  const graphs: TripoSplatModelGraphs = {}
  for (const name of names) {
    const file = GRAPH_FILES[name]
    graphs[name] = {
      internalPrecision: 'float32',
      precision: name === 'vaeEncoder' ? options.vaeInputPrecision ?? 'float32' : 'float32',
      manifest: {
        graphUrl: `${baseUrl}${file}`,
        externalData: [
          {
            path: `${file}.data`,
            url: `${baseUrl}${file}.data`,
          },
        ],
      },
    }
  }
  return graphs
}

export function createVaeEncoderSliceManifest(
  graphUrl = `${import.meta.env.BASE_URL}models/triposplat/flux2_vae_encoder.onnx`,
  precision: TripoSplatGraphPrecision = 'float32',
): TripoSplatModelGraphs {
  return {
    vaeEncoder: {
      precision,
      manifest: createConventionalExternalDataManifest(graphUrl, {
        externalDataPath: 'flux2_vae_encoder.onnx.data',
      }),
    },
  }
}
