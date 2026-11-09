# Vortex Web

A web UI for exploring Vortex data files, built with React, TypeScript, and Rust/WASM, and drawn with the Vortex design system.

## Prerequisites

- [Bun](https://bun.com) 1.4.3, the version in `package.json`'s `packageManager` field
- Node.js 24+ (Storybook and Vite run on Node)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/) (for full app development)
- Rust toolchain (for full app development)

## Getting Started

```bash
bun install
```

`bunfig.toml` makes Bun refuse package versions younger than 7 days.

### Full App (requires Rust + wasm-pack)

```bash
# Start dev server (builds WASM in debug mode, then starts Vite)
bun run dev
```

The Explorer is served under `/explorer/` (Vite's `base`), so open http://localhost:5173/explorer/.
The docs' Vercel project deploys it: under `/explorer/` on the docs previews, and at the root of
explore.vortex.dev. See `docs/vercel-output-config.json` and `.github/workflows/web-build.yml`.

### Storybook (no Rust/WASM required)

Storybook lets you develop and preview UI components in isolation:

```bash
bun run storybook
```

This starts a dev server at http://localhost:6006.

## Scripts

| Command                   | Description                                |
| ------------------------- | ------------------------------------------ |
| `bun run dev`             | Build WASM (debug) + start Vite dev server |
| `bun run build`           | Production build (WASM release + Vite)     |
| `bun run storybook`       | Start Storybook dev server on port 6006    |
| `bun run build-storybook` | Build static Storybook site                |
| `bun run lint`            | Lint with Biome                            |
| `bun run lint:fix`        | Lint with Biome and apply safe fixes       |
| `bun run format`          | Format with Biome                          |
| `bun run format:check`    | Check formatting with Biome                |
| `bun run typecheck`       | Run TypeScript type checking               |
| `bun run check`           | Build WASM + Biome check + typecheck       |

## Styles

The Explorer uses the Vortex design system's CSS and no CSS framework.

- `src/styles/vortex-tokens.css`, `vortex-base.css`, and `vortex.css` are copies of the design system's styles. Change the system, then copy them again; do not edit the copies.
- `src/styles/explorer.css` lays out the Explorer on top of the system. Its classes start with `x-`; the system's start with `vx-`.
- Colours come from the system's tokens, so every view follows the theme. A dtype's colour comes from its `vx-dt-*` class (`dtypeClass` in `components/swimlane/utils.ts`). An encoding has no colour; the views name it.
- The production CSP allows no inline script, so `public/theme.js` sets the theme before the first paint.

## Writing Stories

Add story files alongside your components as `*.stories.tsx`:

```tsx
import type { Meta, StoryObj } from '@storybook/react-vite';
import { MyComponent } from './MyComponent';

const meta: Meta<typeof MyComponent> = {
  component: MyComponent,
};
export default meta;

type Story = StoryObj<typeof MyComponent>;

export const Default: Story = {
  args: {},
};
```

## Project Structure

```
vortex-web/
  crate/            # Rust WASM crate (vortex bindings)
  src/              # React/TypeScript frontend
    styles/         # The design system's CSS, and the Explorer's own
    wasm/pkg/       # Generated WASM bindings (not checked in)
  .storybook/       # Storybook configuration
  public/           # Static assets
```
