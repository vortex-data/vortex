// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    // Two pages: the Explorer, and the 404 the deploy serves for any other address.
    rollupOptions: {
      input: { main: 'index.html', notFound: '404.html' },
    },
  },
});
