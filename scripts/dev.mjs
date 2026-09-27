// `npm run dev`: Next's dev server plus the art frame watcher, so saving
// lib/art.json re-renders any painting whose crop changed.
import { spawn } from "node:child_process";

import { findExportDir, watchFrames } from "./frame-art.mjs";

const next = spawn("npx", ["next", "dev", ...process.argv.slice(2)], {
  stdio: "inherit",
});
next.on("exit", (code) => process.exit(code ?? 0));
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => next.kill(signal));
}

const exportDir = findExportDir();
const log = (message) => console.log(`[art] ${message}`);
if (exportDir) {
  watchFrames(exportDir, { log });
} else {
  log("no Instagram export found; set ART_EXPORT_DIR to re-render crops on save");
}
