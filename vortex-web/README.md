# Vortex Web

A web UI for exploring Vortex data files, built with React, TypeScript, Tailwind CSS, and Rust/WASM.

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
    wasm/pkg/       # Generated WASM bindings (not checked in)
  .storybook/       # Storybook configuration
  public/           # Static assets
```
