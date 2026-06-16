// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  // tsconfig sets `jsx: "preserve"` (Next.js does its own JSX transform), so
  // Vite (rolldown/oxc) would otherwise leave JSX untransformed and fail to
  // execute the component tests. Transform JSX with the React automatic runtime
  // for tests instead (no `import React` needed; pulls from `react/jsx-runtime`).
  // Vite's transform option is `oxc`, not `esbuild`, under rolldown-vite; the
  // typed form of the automatic runtime is the `{ runtime: 'automatic' }`
  // object (the bare `'react-jsx'` string is runtime-valid but not in the type).
  oxc: { jsx: { runtime: 'automatic' } },
  // Mirror the `@/*` -> project-root path alias from tsconfig.json so tests can
  // import App Router route handlers (which use the `@/lib/...` convention).
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('.', import.meta.url)),
    },
  },
  test: {
    environment: 'node',
    // `lib/**` unit + DB-backed tests, prop-driven component tests under
    // `components/**` that render via `react-dom/server` (no DOM env needed),
    // and `app/**` CSS/stylesheet tests that run in the node environment.
    include: ['lib/**/*.test.ts', 'components/**/*.test.tsx', 'app/**/*.test.ts'],
    // Pulling and starting the Postgres testcontainer can take tens of seconds
    // on a cold image cache, so the hook + test budgets are generous.
    testTimeout: 120_000,
    hookTimeout: 180_000,
    // Restore `vi.spyOn` wrappers between tests so a spy on a module-singleton
    // (e.g. `hydrationQueue.schedule`) does not leak its instrumentation into
    // later tests in the same file.
    restoreMocks: true,
  },
});
