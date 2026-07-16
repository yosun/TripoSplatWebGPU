# Direct image-to-splat example

This is the shortest reusable integration for turning one browser image into both a 3DGS PLY and a browser `.splat` file. It is framework-neutral, keeps the input image local, reports progress, supports cancellation, and releases the generated scene after export.

## What is included

| File | Purpose |
| --- | --- |
| [`image-to-splat.ts`](image-to-splat.ts) | reusable image-in/files-out adapter |
| [`browser.ts`](browser.ts) | complete file-picker, progress, cancel, and download wiring |
| [`index.html`](index.html) | minimal Vite-compatible page |
| [`model-config-server.mjs`](model-config-server.mjs) | optional Node endpoint for public model configuration |
| [`tsconfig.json`](tsconfig.json) | repository-local strict typecheck for the example |

Inference runs only in a WebGPU-capable browser. The Node example does not perform inference or receive images.

## Run in an existing Vite app

The package is not on npm yet. First build and pack the two public workspace packages, then install the resulting tarballs in the target app as described in the [package README](../../packages/triposplat-webgpu/README.md). After publication, installation becomes:

```sh
npm install @ai3d/triposplat-webgpu
```

Copy `image-to-splat.ts`, `browser.ts`, and `index.html` into the app. Set the model base URL in `index.html`:

```html
<meta
  name="triposplat-model-base-url"
  content="https://cdn.example.com/triposplat/v1/"
/>
```

Then start the host app normally, open the page in current Chrome or Edge, select an alpha-bearing PNG/WebP, and click **Generate**. The first run downloads and verifies the model artifacts; the current fp32 manifest declares about 6.465 GB.

The example defaults to the 4-step schedule and produces:

- `triposplat.ply` — binary little-endian 3DGS PLY;
- `triposplat.splat` — 32-byte browser splat records;
- count, sizes, and elapsed time in the page status.
