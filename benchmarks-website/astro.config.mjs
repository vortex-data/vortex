import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import node from '@astrojs/node';

export default defineConfig({
  integrations: [react()],
  output: 'server',
  adapter: node({
    mode: 'standalone',
  }),
  server: {
    port: 3000,
  },
  vite: {
    ssr: {
      noExternal: ['echarts', 'echarts-for-react', 'zrender'],
    },
    build: {
      chunkSizeWarningLimit: 700,
    },
  },
});
