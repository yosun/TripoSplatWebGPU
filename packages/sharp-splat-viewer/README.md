# @bring-shrubbery/sharp-splat-viewer

Drop-in `<sharp-splat>` Web Component for Gaussian splat `.ply` files generated
by [`ml-sharp-web`](https://github.com/bring-shrubbery/ml-sharp-web). Reads any
baked-in default camera and render settings from the file's PLY header so the
splat opens with the creator's intended view.

Wraps [`@mkkellogg/gaussian-splats-3d`](https://www.npmjs.com/package/@mkkellogg/gaussian-splats-3d).

## Quick start (CDN)

```html
<script type="module"
  src="https://cdn.jsdelivr.net/npm/@bring-shrubbery/sharp-splat-viewer/dist/sharp-splat-viewer.iife.js"></script>

<sharp-splat src="my-splat.ply" style="width:600px;height:400px"></sharp-splat>
```

The IIFE bundle ships `three` and the splats viewer inline, so a single
`<script>` is all you need.

## ESM (bundlers)

```bash
npm install @bring-shrubbery/sharp-splat-viewer three @mkkellogg/gaussian-splats-3d
```

```ts
import '@bring-shrubbery/sharp-splat-viewer'
```

The ESM build externalizes `three` and `@mkkellogg/gaussian-splats-3d` (declared
as `peerDependencies`) so your bundler doesn't ship two copies of `three`.

## Attributes

All attributes override values baked into the PLY file. Omit them to use the
file's defaults (or library defaults if the file has none).

| Attribute          | Type       | Example                       | Default    |
| ------------------ | ---------- | ----------------------------- | ---------- |
| `src` (required)   | URL        | `src="splat.ply"`             | —          |
| `camera-position`  | `x y z`    | `camera-position="0 0 5"`     | from file  |
| `camera-target`    | `x y z`    | `camera-target="0 0 0"`       | from file  |
| `camera-up`        | `x y z`    | `camera-up="0 1 0"`           | from file  |
| `fov`              | degrees    | `fov="50"`                    | `60`       |
| `bg-color`         | `#rrggbb`  | `bg-color="#0a0a14"`          | `#101014`  |
| `max-screen-size`  | px         | `max-screen-size="2048"`      | `2048`     |
| `auto-rotate`      | bool       | `auto-rotate` or `auto-rotate="true"` | `false` |

Triple values accept space- or comma-separated numbers.

## Baked-in metadata schema

The file format is plain PLY. Defaults live in standard `comment` lines between
the `format` line and the first `element`:

```
ply
format binary_little_endian 1.0
comment sharp-viewer/version 1
comment sharp-viewer/camera-position 1.20 0.50 -3.00
comment sharp-viewer/camera-target 0 0 0
comment sharp-viewer/camera-up 0 1 0
comment sharp-viewer/fov 60
comment sharp-viewer/bg-color #101014
comment sharp-viewer/max-screen-size 2048
comment sharp-viewer/auto-rotate 0
element vertex …
…
```

PLY parsers ignore unknown comments, so the file remains compatible with
external splat viewers.

## Caveats

- **Three.js double-bundling**: only one copy of `three` should run on a page.
  If the host page already bundles `three` (e.g. in a Vite app), prefer the
  ESM build and add `three` as a real dependency. The IIFE/CDN bundle includes
  its own copy and is intended for plain HTML pages.
- **CORS**: the viewer fetches `src` via `fetch()`, so the splat host must
  allow cross-origin reads (Cloudflare R2, S3 bucket CORS rule, etc.).
- **No SSR**: the component only registers itself in browser environments
  where `customElements` exists.

## License

MIT.
