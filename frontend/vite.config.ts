import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// В режиме разработки запросы /api/* проксируются на FastAPI (по умолчанию localhost:8000),
// поэтому фронтенду не нужен CORS и абсолютный адрес бэкенда. Путь не переписывается
// (префикс /api снимает сам бэкенд), чтобы не испортить закодированные id вида "123%230".
const apiTarget = process.env.VITE_API_PROXY ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  build: {
    // MapLibre (~800 КБ) выносится в отдельный файл: он меняется реже кода приложения и лучше кэшируется
    chunkSizeWarningLimit: 1000,
    rollupOptions: { output: { manualChunks: { maplibre: ["maplibre-gl"], react: ["react", "react-dom"] } } },
  },
  server: {
    port: 5173,
    host: true,
    proxy: {
      "/api": { target: apiTarget, changeOrigin: true },
    },
  },
  preview: {
    port: 4173,
    proxy: {
      "/api": { target: apiTarget, changeOrigin: true },
    },
  },
});
