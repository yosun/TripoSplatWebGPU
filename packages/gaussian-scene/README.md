# `@ai3d/gaussian-scene`

Framework-neutral ownership and export helpers for canonical Gaussian scenes.

`createGaussianScene` validates and defensively copies interleaved float32 arrays.
The returned scene owns those copies. Calling `dispose()` is idempotent, drops the
scene's array references, and makes subsequent exports reject.

PLY export uses the standard binary little-endian 3DGS fields. Linear scales and
opacities are encoded as log-scale and logit values. Rotations are written in
`wxyz` order. An optional row-major `plyExportTransform` applies a proper 3D
rotation to positions and Gaussian orientations. Without it, coordinates are
preserved.

`exportSplat()` emits the de-facto 32-byte browser `.splat` layout used by
common Gaussian viewers: float32 position and linear scale, byte RGBA, and a
quantized `wxyz` quaternion. It applies the same coordinate transform as PLY,
decodes log-scale/logit inputs, and converts degree-zero SH or direct color to
display RGB.

This package has no framework, renderer, ONNX Runtime, or Node runtime dependency.
