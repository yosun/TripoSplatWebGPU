# Gaussian conventions and export

`@ai3d/gaussian-scene` defines the framework-neutral scene contract independently of SHARP and TripoSplat model orchestration.

## Canonical arrays

For `count = N`:

| Field | Type and length | Canonical meaning |
| --- | --- | --- |
| `positions` | `Float32Array`, `3N` | xyz-interleaved object-space centers |
| `scales` | `Float32Array`, `3N` | anisotropic scale; interpretation declared by `scaleEncoding` |
| `rotations` | `Float32Array`, `4N` | non-zero quaternions; order declared by `rotationOrder` |
| `opacities` | `Float32Array`, `N` | opacity; interpretation declared by `opacityEncoding` |
| `sphericalHarmonics` | optional `Float32Array`, `3N` | degree-zero RGB SH coefficients when declared |
| `colors` | optional `Float32Array`, `3N` | RGB values with explicit linear or sRGB semantics |

Metadata records coordinate system, units, quaternion order, scale and opacity encodings, color/SH semantics, model revision, generation settings, seed, runtime version, and an optional PLY export rotation.

The scene constructor validates and copies arrays. The returned scene owns those copies. `dispose()` is idempotent and releases owned references; later export attempts reject with `GaussianSceneDisposedError`.

## TripoSplat convention

The current TripoSplat host decoder produces:

- object-space positions;
- positive linear scales;
- `wxyz` rotations;
- activated opacities in `[0, 1]`;
- degree-zero SH coefficients.

The official object export orientation is the proper rotation:

```text
[ 1  0  0 ]
[ 0  0 -1 ]
[ 0  1  0 ]
```

This maps a position `(x, y, z)` to `(x, -z, y)`. The export helper applies the same rotation to Gaussian orientation, not only to centers.

SHARP uses different camera-space assumptions. Do not apply SHARP's camera flips or NDC-to-metric conversion to a TripoSplat scene.

## Binary PLY

The package exports the standard binary little-endian 17-float 3DGS vertex layout:

```text
x y z
nx ny nz
f_dc_0 f_dc_1 f_dc_2
opacity
scale_0 scale_1 scale_2
rot_0 rot_1 rot_2 rot_3
```

Normals are zero. Linear scales are converted to log scale, linear opacities to logits, rotations to `wxyz`, and direct colors to degree-zero SH coefficients. Linear RGB is converted through sRGB before the SH transform; sRGB values use the SH transform directly.

## Browser `.splat`

`exportSplat()` emits the de-facto 32-byte browser layout used by common Gaussian viewers:

```text
position:   3 × float32 little-endian
scale:      3 × float32 little-endian, linear
color:      4 × uint8 RGBA
rotation:   4 × uint8 quantized wxyz
```

It applies the same coordinate rotation as PLY, decodes log-scale/logit source arrays, converts degree-zero SH or declared direct color to display RGB, and uses the official opacity-times-volume ordering rule. Unit tests include a 128-byte fixture that is byte-identical to output from the pinned official TripoSplat exporter.

That fixture does not prove byte identity for every float boundary. The browser exporter receives `Float32Array` data and performs some quaternion, score, and color arithmetic as JavaScript numbers, while the official NumPy path has float32 casts and an unstable default sort for equal scores. Quantization boundaries, quaternion sign ambiguity, strict near-ties, and originally-fp16 model values need broader parity fixtures before a universal byte-identity claim.

This describes one named layout only. Other tools use the same extension for incompatible binary records or compression.

## Viewer adapters

A viewer must explicitly document:

- expected coordinate handedness and up axis;
- whether it consumes linear scale or log scale;
- quaternion component order;
- opacity activation;
- RGB versus SH inputs;
- sorting and covariance expectations.

Prefer passing canonical arrays to a small adapter. PLY blob round-tripping is useful for interoperability but adds serialization, parsing, and temporary memory.
