// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  // The docs' Vercel project serves the Explorer under /explorer/, and serves that directory at the
  // root of explore.vortex.dev. Every other address there gets the docs' 404, the one 404 the
  // monorepo builds (docs/vercel-output-config.json).
  base: '/explorer/',
  plugins: [react()],
});
