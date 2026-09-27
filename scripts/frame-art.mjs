// Straightens and crops photographed paintings, then adds a thin white edge,
// for every piece in lib/art.json that has a `frame`:
//
//   "frame": {
//     "source": "media/posts/123.jpg",          // in the export, or art-originals/…
//     "corners": [[x, y], [x, y], [x, y], [x, y]], // canvas TL, TR, BR, BL
//     "edge": false                               // optional: no white edge
//   }
//
// `corners` is optional; without it the whole photo is used. Corners are in
// source pixels, so they can be read off the original photo and nudged by hand.
//
//   node scripts/frame-art.mjs            # render pieces whose frame changed
//   node scripts/frame-art.mjs --watch    # ...and again on every save
//
// `npm run dev` runs the watcher alongside Next, so saving lib/art.json is
// enough; `npm run art:edit` opens a drag-the-corners editor. Each rendered piece gets a `version` (a hash of its frame settings)
// that busts the browser cache. The Instagram export is read from
// $ART_EXPORT_DIR, or else the newest ~/Downloads/instagram-alex_wa_art-*.
import { execFileSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export const ROOT = process.cwd();
export const DATA_FILE = path.join(ROOT, "lib/art.json");
export const FULL_EDGE = 1600;
export const THUMB_WIDTH = 720;
// Trimmed from each side after straightening, to drop canvas-edge slivers.
const INSET = 0.006;
// White edge width, as a fraction of the painting's shorter side. The editor's
// live preview (scripts/art-editor.html) mirrors this value.
const EDGE = 0.025;

export function magick(...args) {
  return execFileSync("magick", args, { encoding: "utf8" }).trim();
}

export function readArtData() {
  return JSON.parse(fs.readFileSync(DATA_FILE, "utf8"));
}

export function writeArtData(data) {
  fs.writeFileSync(DATA_FILE, `${JSON.stringify(data, null, 2)}\n`);
}

// Photos that aren't on Instagram live in art-originals/ (git-ignored, like
// the export); every other source is a path inside the Instagram export.
export const ORIGINALS = "art-originals/";

export function resolveSource(source, exportDir) {
  return source.startsWith(ORIGINALS)
    ? path.join(ROOT, source)
    : path.join(exportDir, source);
}

export function findExportDir() {
  if (process.env.ART_EXPORT_DIR) {
    return process.env.ART_EXPORT_DIR;
  }
  const downloads = path.join(os.homedir(), "Downloads");
  const candidates = fs.existsSync(downloads)
    ? fs
        .readdirSync(downloads)
        .filter((name) => name.startsWith("instagram-alex_wa_art-"))
        .map((name) => path.join(downloads, name))
        .filter((dir) => fs.statSync(dir).isDirectory())
        .sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs)
    : [];
  return candidates[0] ?? null;
}

const distance = ([ax, ay], [bx, by]) => Math.hypot(ax - bx, ay - by);

export function frameVersion(frame) {
  return crypto
    .createHash("sha1")
    .update(JSON.stringify({ ...frame, INSET, EDGE, FULL_EDGE, THUMB_WIDTH }))
    .digest("hex")
    .slice(0, 8);
}

export function renderPiece(piece, exportDir) {
  const source = resolveSource(piece.frame.source, exportDir);
  const [sourceWidth, sourceHeight] = magick("identify", "-format", "%w %h", source)
    .split(" ")
    .map(Number);
  const [tl, tr, br, bl] = piece.frame.corners ?? [
    [0, 0],
    [sourceWidth, 0],
    [sourceWidth, sourceHeight],
    [0, sourceHeight],
  ];
  const width = Math.round((distance(tl, tr) + distance(bl, br)) / 2);
  const height = Math.round((distance(tl, bl) + distance(tr, br)) / 2);
  const edge =
    piece.frame.edge === false ? 0 : Math.round(Math.min(width, height) * EDGE);
  const mapping = [
    [tl, [0, 0]],
    [tr, [width, 0]],
    [br, [width, height]],
    [bl, [0, height]],
  ]
    .map(([from, to]) => `${from.join(",")} ${to.join(",")}`)
    .join("  ");

  const full = path.join(ROOT, "public", piece.src);
  const thumb = path.join(ROOT, "public", piece.thumb);
  magick(
    source,
    "-auto-orient",
    "-virtual-pixel", "edge",
    "-define", `distort:viewport=${width}x${height}+0+0`,
    "-distort", "Perspective", mapping,
    "+repage",
    "-shave", `${Math.round(width * INSET)}x${Math.round(height * INSET)}`,
    "-bordercolor", "#ffffff",
    "-border", `${edge}`,
    "-strip",
    "-resize", `${FULL_EDGE}x${FULL_EDGE}>`,
    "-quality", "86",
    full,
  );
  magick(full, "-resize", `${THUMB_WIDTH}x`, "-quality", "80", "-interlace", "Plane", thumb);

  const [fullWidth, fullHeight] = magick("identify", "-format", "%w %h", full)
    .split(" ")
    .map(Number);
  // Sample the color inside the white edge so the palette dab stays true.
  const color = magick(
    full,
    "-shave", `${Math.round(fullWidth * 0.1)}x${Math.round(fullHeight * 0.1)}`,
    "-resize", "1x1!",
    "-format", "#%[hex:u.p{0,0}]",
    "info:",
  )
    .slice(0, 7)
    .toLowerCase();

  return { width: fullWidth, height: fullHeight, color };
}

// Renders every piece whose frame settings changed since its last render.
// Returns the number of pieces rendered; lib/art.json is only rewritten when
// something changed, so the watcher doesn't retrigger itself.
export function frameChangedPieces(exportDir, { log = console.log } = {}) {
  const data = readArtData();
  let rendered = 0;

  for (const piece of data.pieces) {
    if (!piece.frame) {
      continue;
    }
    const version = frameVersion(piece.frame);
    const current =
      piece.version === version &&
      fs.existsSync(path.join(ROOT, "public", piece.src)) &&
      fs.existsSync(path.join(ROOT, "public", piece.thumb));
    if (current) {
      continue;
    }
    try {
      Object.assign(piece, renderPiece(piece, exportDir), { version });
      rendered += 1;
      log(`framed ${piece.title} (${piece.width}×${piece.height})`);
    } catch (error) {
      log(`could not frame ${piece.title}: ${error.message.split("\n")[0]}`);
    }
  }

  if (rendered) {
    writeArtData(data);
  }
  return rendered;
}

export function watchFrames(exportDir, { log = console.log } = {}) {
  let timer = null;
  const run = () => {
    try {
      frameChangedPieces(exportDir, { log });
    } catch (error) {
      // Usually half-typed JSON; the next save will try again.
      log(`waiting for valid lib/art.json (${error.message.split("\n")[0]})`);
    }
  };

  run();
  fs.watch(path.dirname(DATA_FILE), (_event, filename) => {
    if (filename === path.basename(DATA_FILE)) {
      clearTimeout(timer);
      timer = setTimeout(run, 150);
    }
  });
  log(`watching lib/art.json (photos from ${exportDir})`);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const exportDir = findExportDir();
  if (!exportDir) {
    console.error(
      "No Instagram export found. Set ART_EXPORT_DIR or put the export in ~/Downloads.",
    );
    process.exit(1);
  }
  if (process.argv.includes("--watch")) {
    watchFrames(exportDir);
  } else {
    const count = frameChangedPieces(exportDir);
    console.log(count ? `Rendered ${count} piece(s).` : "All frames are up to date.");
  }
}
