// =============================================================================
// Aptly — AI Job Search Co-Pilot
// Author: Irfan Mohammed
// License: MIT — see LICENSE file in the project root.
// =============================================================================
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
});
