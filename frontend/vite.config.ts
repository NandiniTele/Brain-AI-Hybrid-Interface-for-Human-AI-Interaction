import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Cleanly handle proxy connection lifecycle and transient socket disconnects
const silenceProxyErrors = (proxy: any) => {
  proxy.on('error', (_err: any, _req: any, res: any) => {
    if (res && typeof res.writeHead === 'function' && !res.headersSent) {
      try {
        res.writeHead(503, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Backend proxy unavailable or reconnecting' }));
      } catch { /* ignore */ }
    }
  });

  proxy.on('proxyReqWs', (_proxyReq: any, req: any, socket: any, _options: any, _head: any) => {
    if (socket && typeof socket.on === 'function') {
      socket.on('error', () => { /* normal client disconnect */ });
    }
    if (req && req.socket && typeof req.socket.on === 'function') {
      req.socket.on('error', () => { /* normal client disconnect */ });
    }
  });

  proxy.on('open', (proxySocket: any) => {
    if (proxySocket && typeof proxySocket.on === 'function') {
      proxySocket.on('error', () => { /* normal upstream disconnect */ });
    }
  });

  proxy.on('close', (_req: any, socket: any) => {
    if (socket && typeof socket.on === 'function') {
      socket.on('error', () => { /* ignore */ });
    }
  });
};

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      // Proxy all REST API calls to the FastAPI backend
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        configure: silenceProxyErrors,
      },
      // Proxy WebSocket connection — timeout:0 disables Vite's ~2-min idle kill
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true,
        timeout: 0,
        configure: silenceProxyErrors,
      },
    },
  },
})

