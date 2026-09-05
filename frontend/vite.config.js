import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The browser talks to the backend through this dev-server proxy, so the app
// only ever uses same-origin relative URLs and needs no API base config.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.BACKEND_ORIGIN || 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
