# Privacy and security

## Browser-local inference boundary

The intended inference path is entirely local to the browser:

1. decode a user-selected `Blob`, `ImageBitmap`, `HTMLImageElement`, or `ImageData`;
2. preprocess pixels in browser memory;
3. run ONNX Runtime WebGPU in a module worker;
4. sample and decode locally;
5. return typed arrays and an optional local export blob.

No inference endpoint is required, and the model layer must not upload input pixels. Model graph requests go to the configured model host; those are not image uploads.

This architecture does not make an entire deployed application private by default. Analytics, crash reporting, CDN logging, authentication, browser extensions, and application code can create separate network paths. Audit the built deployment in browser developer tools with a real image before making a privacy statement.

## Sensitive data handling

- do not log image bytes, generated arrays, bearer tokens, or signed model URLs;
- revoke temporary image, PLY, and viewer object URLs;
- transfer buffers to workers where ownership is clear;
- release graph sessions between stages and terminate workers on disposal or hard cancellation;
- clear UI references to disposed scenes;
- document whether browser HTTP or OPFS model caches persist across sessions;
- never place model credentials in client code unless they are intentionally public or short-lived and scope-limited.

## Content Security Policy

Test a policy suitable for the final bundler and ONNX Runtime build. It may need to permit:

- module workers from the app origin;
- `blob:` worker or image URLs only if the chosen integration actually uses them;
- connections to the model CDN;
- WebAssembly compilation required by the selected fallback;
- local object URLs for downloads and viewers.

Avoid broad `*` and `unsafe-eval` directives. The exact ONNX Runtime/WebAssembly policy must be verified against the packed production build; it is not yet qualified here.

## Integrity and supply chain

- pin the model revision in a manifest;
- record SHA-256 and byte count for every graph and sidecar;
- verify integrity before session creation;
- pin the package manager and lockfile in CI;
- use read-only GitHub Actions permissions by default;
- pack and inspect publishable packages before release;
- generate provenance for exported model artifacts;
- exclude weights, fixtures, benchmarks, and viewer dependencies from the core npm package.

An integrity mismatch is not a recoverable inference error. Delete the bad cache entry, report a typed `ModelIntegrityError`, and require verified bytes before retry.

## Cancellation

An `AbortSignal` can stop orchestration between graph invocations. If ONNX Runtime cannot interrupt a running GPU call, cancellation must still prevent later stages and may terminate the worker to release resources. A retry should begin from a known runtime state, not reuse a possibly detached or partially disposed tensor graph.

## Reporting limitations

Web pages do not receive reliable total/free GPU memory. Do not fingerprint users through unnecessary adapter details or report fabricated capacity. Compatibility diagnostics should contain only information needed to explain support and failures, and telemetry should be opt-in or deployment-controlled.
