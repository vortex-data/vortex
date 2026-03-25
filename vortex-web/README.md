# Vortex Web

A web UI for exploring Vortex data files, built with React, TypeScript, Tailwind CSS, and Rust/WASM.

## Prerequisites

- Node.js 22+
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/) (for full app development)
- Rust toolchain (for full app development)

## Getting Started

```bash
npm install
```

### Full App (requires Rust + wasm-pack)

```bash
# Start dev server (builds WASM in debug mode, then starts Vite)
npm run dev
```

### Storybook (no Rust/WASM required)

Storybook lets you develop and preview UI components in isolation:

```bash
npm run storybook
```

This starts a dev server at http://localhost:6006.

## Scripts

| Command | Description |
|---|---|
| `npm run dev` | Build WASM (debug) + start Vite dev server |
| `npm run build` | Production build (WASM release + Vite) |
| `npm run storybook` | Start Storybook dev server on port 6006 |
| `npm run build-storybook` | Build static Storybook site |
| `npm run lint` | Run ESLint |
| `npm run lint:fix` | Run ESLint with auto-fix |
| `npm run typecheck` | Run TypeScript type checking |
| `npm run check` | Build WASM + lint + typecheck |

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
