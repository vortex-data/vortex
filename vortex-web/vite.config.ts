// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  // The docs' Vercel project serves the Explorer under /explorer/, and serves that directory at the
  // root of explore.vortex.dev.
  base: '/explorer/',
  plugins: [react()],
  build: {
    // Two pages: the Explorer, and the 404 the deploy serves for any other address.
    rollupOptions: {
      input: { main: 'index.html', notFound: '404.html' },
    },
  },
});
