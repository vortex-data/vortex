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

## Comparing compression output

Open the candidate `.vortex` file, select **Compare…** in the header, and choose the previous
version of the same file. The Compare view shows whole-file, data, and metadata byte deltas and
then aligns the layout and array-encoding trees to explain where those bytes changed.

The tree comparison is semantic rather than a diff of rendered labels. Layout siblings are
matched by field name, chunk row range, or named transparent/auxiliary role. Array children use
their encoding-provided child names, with their stable child position as a fallback for serialized
trees that do not carry names. This makes an inserted field an addition instead of making every
following field look modified.

To open a comparison directly, use the compare hash route with URL-encoded remote file URLs:

```text
https://explorer.example/#/compare?baseline=https%3A%2F%2Fdata.example%2Fbefore.vortex&candidate=https%3A%2F%2Fdata.example%2Fafter.vortex
```

An individual file can also be opened directly:

```text
https://explorer.example/#/file?url=https%3A%2F%2Fdata.example%2Foutput.vortex
```

URLs may be absolute HTTP(S) URLs or paths relative to the Explorer deployment. From the Compare
view, either file can be opened in the regular Details and Swimlane views or replaced with another
local file to recalculate the diff.

The hash route does not require server-side routing, so the Explorer remains a static application:
the browser fetches both files and opens them in the existing Web Workers. Each file host must
permit browser access with CORS. Local files still need to be selected manually because browsers
do not allow a page to read arbitrary local paths.

Use `compress-bench --ingest-jsonl <path>` for repeatable encode/decode timing and file-size
measurements. The Explorer comparison complements those aggregate measurements: it compares the
actual output files and attributes size changes to layout and encoding nodes. Compare files made
from identical logical input; the UI warns when row counts or schemas differ.

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

| Command                   | Description                                |
| ------------------------- | ------------------------------------------ |
| `npm run dev`             | Build WASM (debug) + start Vite dev server |
| `npm run build`           | Production build (WASM release + Vite)     |
| `npm run storybook`       | Start Storybook dev server on port 6006    |
| `npm run build-storybook` | Build static Storybook site                |
| `npm run lint`            | Run ESLint                                 |
| `npm run lint:fix`        | Run ESLint with auto-fix                   |
| `npm run typecheck`       | Run TypeScript type checking               |
| `npm run check`           | Build WASM + lint + typecheck              |

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
