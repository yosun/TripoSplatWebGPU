# Model hosting and browser-local caching

TripoSplat model files are static browser assets. Put them on S3 with CloudFront, GCS with Cloud CDN, or an equivalent object CDN. Do not pass the 6.47 GB artifact set through Vercel functions, application route handlers, or an inference server. The browser downloads model files directly; user images remain local.

## Canonical fp32 release

[`public/models/triposplat/manifest.json`](../public/models/triposplat/manifest.json) is the deployable manifest for the current fp32 WebGPU artifacts. It contains the exact byte length and SHA-256 digest for each ONNX graph and external-data file.

| Graph role | Graph | External data | Combined bytes |
| --- | --- | --- | ---: |
| DINOv3 | `dinov3_encoder_fp32.onnx` | `dinov3_encoder_fp32.onnx.data` | 3,367,847,196 |
| Flux VAE | `flux2_vae_encoder.onnx` | `flux2_vae_encoder.onnx.data` | 137,800,544 |
| DiT step | `dit_step_webgpu_fp32.onnx` | `dit_step_webgpu_fp32.onnx.data` | 1,643,895,982 |
| Octree occupancy | `octree_occupancy_decoder_fp32.onnx` | `octree_occupancy_decoder_fp32.onnx.data` | 221,419,396 |
| Gaussian decoder | `gaussian_decoder_fp32.onnx` | `gaussian_decoder_fp32.onnx.data` | 1,094,219,284 |
| **Total** |  |  | **6,465,182,402** |

`estimatedPeakBytes` is intentionally absent. The browser does not expose reliable process-level GPU/unified-memory usage, and a target-machine peak has not been isolated and measured.

The manifest identifies a validated artifact set, not a completed end-to-end quality claim. DINOv3 and one DiT invocation pass their strict fp32 parity gates. The four-step flow loop meets the recorded qualification envelope but misses its stricter gate. The complete eight-level octree trajectory passes, including final points, and the Gaussian graph passes its raw neural boundary; final live activated and packed Gaussian-scene parity remains open.

## Immutable object layout

Copy the manifest and exactly the ten files it references into a revisioned prefix:

```text
triposplat-webgpu/0.1.0-fp32.20260715/
  manifest.json
  dinov3_encoder_fp32.onnx
  dinov3_encoder_fp32.onnx.data
  flux2_vae_encoder.onnx
  flux2_vae_encoder.onnx.data
  dit_step_webgpu_fp32.onnx
  dit_step_webgpu_fp32.onnx.data
  octree_occupancy_decoder_fp32.onnx
  octree_occupancy_decoder_fp32.onnx.data
  gaussian_decoder_fp32.onnx
  gaussian_decoder_fp32.onnx.data
```

All manifest URLs are relative, so no rewrite is required when the directory moves to a CDN. Each external-data `path` exactly matches the `location` embedded in its ONNX graph. The `url` may be replaced with an absolute signed URL, but the `path` must not change.

Publish model objects before the manifest. Never overwrite an object beneath an immutable version URL; make a new prefix and manifest version instead.

## S3 and CloudFront

Upload the binary objects with immutable caching, then upload the manifest with revalidation:

```sh
PREFIX=s3://example-models/triposplat-webgpu/0.1.0-fp32.20260715
FILES=(
  dinov3_encoder_fp32.onnx dinov3_encoder_fp32.onnx.data
  flux2_vae_encoder.onnx flux2_vae_encoder.onnx.data
  dit_step_webgpu_fp32.onnx dit_step_webgpu_fp32.onnx.data
  octree_occupancy_decoder_fp32.onnx octree_occupancy_decoder_fp32.onnx.data
  gaussian_decoder_fp32.onnx gaussian_decoder_fp32.onnx.data
)

for FILE in "${FILES[@]}"; do
  aws s3 cp "public/models/triposplat/$FILE" "$PREFIX/$FILE" \
    --content-type application/octet-stream \
    --cache-control 'public,max-age=31536000,immutable'
done

aws s3 cp public/models/triposplat/manifest.json "$PREFIX/manifest.json" \
  --content-type application/json \
  --cache-control 'public,max-age=300,must-revalidate'
```

The explicit list matters: the local model directory may also contain experimental exports.

Configure the bucket or CloudFront response-headers policy for the real application origin. A minimal S3 CORS rule is:

```json
[
  {
    "AllowedOrigins": ["https://app.example.com"],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["Authorization", "Range"],
    "ExposeHeaders": ["Accept-Ranges", "Content-Length", "Content-Range", "ETag"],
    "MaxAgeSeconds": 3600
  }
]
```

Use CloudFront signed URLs or signed cookies for private models. Do not put authorization tokens into logs. Keep signature query parameters out of the cache key where the CloudFront policy permits it so refreshed signatures still address the same immutable object.

## GCS and Cloud CDN

Upload the same revisioned directory, with long-lived binary caching and a short-lived manifest:

```sh
PREFIX=gs://example-models/triposplat-webgpu/0.1.0-fp32.20260715
FILES=(
  dinov3_encoder_fp32.onnx dinov3_encoder_fp32.onnx.data
  flux2_vae_encoder.onnx flux2_vae_encoder.onnx.data
  dit_step_webgpu_fp32.onnx dit_step_webgpu_fp32.onnx.data
  octree_occupancy_decoder_fp32.onnx octree_occupancy_decoder_fp32.onnx.data
  gaussian_decoder_fp32.onnx gaussian_decoder_fp32.onnx.data
)

for FILE in "${FILES[@]}"; do
  gcloud storage cp "public/models/triposplat/$FILE" "$PREFIX/$FILE" \
    --content-type=application/octet-stream \
    --cache-control='public,max-age=31536000,immutable'
done

gcloud storage cp public/models/triposplat/manifest.json "$PREFIX/manifest.json" \
  --content-type=application/json \
  --cache-control='public,max-age=300,must-revalidate'
```

Apply a bucket CORS file such as:

```json
[
  {
    "origin": ["https://app.example.com"],
    "method": ["GET", "HEAD"],
    "responseHeader": ["Accept-Ranges", "Content-Length", "Content-Range", "ETag"],
    "maxAgeSeconds": 3600
  }
]
```

```sh
gcloud storage buckets update gs://example-models --cors-file=cors.json
```

Use Cloud CDN signed URLs or signed cookies when the objects are private. Keep the origin bucket private and grant the CDN origin identity only the access it needs.

## Browser and CDN requirements

The CDN must allow the application origin to fetch the manifest and model objects. A representative response policy is:

```http
Access-Control-Allow-Origin: https://app.example.com
Access-Control-Allow-Methods: GET, HEAD, OPTIONS
Access-Control-Expose-Headers: Accept-Ranges, Content-Length, Content-Range, ETag
Cross-Origin-Resource-Policy: cross-origin
```

`Access-Control-Allow-Methods` is normally returned on the preflight response. `Cross-Origin-Resource-Policy: cross-origin` is needed when the application uses `Cross-Origin-Embedder-Policy: require-corp`; otherwise valid CORS is sufficient. Use `Access-Control-Allow-Origin: *` only for public, uncredentialed artifacts. Redirects must preserve working CORS. HTTP range support is recommended for CDN behavior and future resumable downloads, although the current package safely discards an interrupted partial rather than resuming it across reloads.

The package verifies declared lengths and SHA-256 digests before ONNX Runtime session creation. `cache: 'opfs'` persists verified bytes in Origin Private File System; `cache: 'cache-api'` uses verified Cache API entries; and `cache: 'none'` still performs integrity verification. Signed URL query strings are not used as persistent artifact identities.

## Verification

Test the deployed URLs from the real application origin:

```sh
curl -I -H 'Origin: https://app.example.com' \
  https://models.example.com/triposplat-webgpu/0.1.0-fp32.20260715/manifest.json

curl -I -H 'Origin: https://app.example.com' \
  -H 'Range: bytes=0-1023' \
  https://models.example.com/triposplat-webgpu/0.1.0-fp32.20260715/dit_step_webgpu_fp32.onnx.data
```

If range requests are advertised, the second request should return `206 Partial Content` with a valid `Content-Range`. Then load the CDN manifest in the package and verify one cold download, one verified cache hit after refresh, explicit cache clearing, a deliberately corrupted object, an interrupted download, and a manifest-version change.

No generation request should upload an input image. In browser developer tools, the manifest and model artifacts should be the only network traffic initiated by model loading and inference.
