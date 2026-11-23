import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// VITE_BASE lets a server build serve from "/" while GitHub Pages keeps its sub-path.
export default defineConfig({
  plugins: [react()],
  base: process.env.VITE_BASE || '/Saarthi_-Documentation/',
})
