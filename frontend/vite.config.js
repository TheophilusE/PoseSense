import { defineConfig } from 'vite';

export default defineConfig({
  server: { open: true },
  build: {
    // Three.js bundles are expected to be large for this app.
    chunkSizeWarningLimit: 1200
  }
});
