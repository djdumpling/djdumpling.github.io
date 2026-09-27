// A local editor for art crops: drag a painting's four corners on the
// original photo, preview the straightened result live, and save.
//
//   npm run art:edit    # then open http://localhost:4321
//
// Saving writes `frame` to lib/art.json and re-renders that piece right away
// (the same code path as scripts/frame-art.mjs), so a running `npm run dev`
// reloads the art page with the new image.
import fs from "node:fs";
import http from "node:http";
import path from "node:path";

import {
  ORIGINALS,
  ROOT,
  findExportDir,
  frameVersion,
  readArtData,
  renderPiece,
  writeArtData,
} from "./frame-art.mjs";

const PORT = Number(process.env.PORT ?? 4321);
const EDITOR_HTML = path.join(ROOT, "scripts/art-editor.html");
const TYPES = { ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png" };

const exportDir = findExportDir();
if (!exportDir) {
  console.error("No Instagram export found. Set ART_EXPORT_DIR or put the export in ~/Downloads.");
  process.exit(1);
}

// Every photo in every post, so a piece can switch to another carousel image.
function readPostMedia() {
  const file = path.join(exportDir, "your_instagram_activity/media/posts_1.json");
  const posts = JSON.parse(fs.readFileSync(file, "utf8"));
  return new Map(
    posts.map((post) => [
      path.basename(post.media[0].uri, path.extname(post.media[0].uri)),
      post.media.map((media) => media.uri),
    ]),
  );
}
const postMedia = readPostMedia();

function defaultSource(piece) {
  const id = piece.id.split("-")[0];
  return (postMedia.get(piece.post) ?? []).find((uri) => uri.includes(id)) ?? postMedia.get(piece.post)?.[0];
}

// The photos a piece may be cropped from: its Instagram post's images, or
// the art-originals/ photo it was added from.
function photoOptions(piece) {
  return postMedia.get(piece.post) ?? (piece.frame ? [piece.frame.source] : []);
}

function listPieces() {
  return readArtData().pieces.map((piece) => ({
    id: piece.id,
    title: piece.title,
    section: piece.section,
    date: piece.date,
    preview: `${piece.src}${piece.version ? `?v=${piece.version}` : ""}`,
    thumb: `${piece.thumb}${piece.version ? `?v=${piece.version}` : ""}`,
    frame: piece.frame ?? null,
    source: piece.frame?.source ?? defaultSource(piece),
    options: photoOptions(piece),
  }));
}

function saveFrame({ id, source, corners, edge }) {
  const data = readArtData();
  const piece = data.pieces.find((item) => item.id === id);
  if (!piece) {
    throw new Error(`Unknown piece ${id}`);
  }
  if (!photoOptions(piece).includes(source)) {
    throw new Error("That photo isn't part of this post");
  }
  const valid =
    Array.isArray(corners) &&
    corners.length === 4 &&
    corners.every((point) => point.length === 2 && point.every(Number.isFinite));
  if (!valid) {
    throw new Error("Expected four [x, y] corners");
  }

  piece.frame = {
    source,
    corners: corners.map((point) => point.map(Math.round)),
    ...(edge === false ? { edge: false } : {}),
  };
  // Render before writing so a running watcher sees an up-to-date version.
  Object.assign(piece, renderPiece(piece, exportDir), {
    version: frameVersion(piece.frame),
  });
  writeArtData(data);
  return listPieces().find((item) => item.id === id);
}

function sendFile(response, root, relative) {
  const file = path.resolve(root, relative);
  if (!file.startsWith(path.resolve(root) + path.sep) || !fs.existsSync(file)) {
    response.writeHead(404).end("Not found");
    return;
  }
  response.writeHead(200, {
    "Content-Type": TYPES[path.extname(file).toLowerCase()] ?? "application/octet-stream",
    "Cache-Control": "no-cache",
  });
  fs.createReadStream(file).pipe(response);
}

function sendJson(response, status, body) {
  response.writeHead(status, { "Content-Type": "application/json" });
  response.end(JSON.stringify(body));
}

const server = http.createServer((request, response) => {
  const url = new URL(request.url, `http://${request.headers.host}`);
  const pathname = decodeURIComponent(url.pathname);

  if (request.method === "GET" && pathname === "/") {
    response.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    fs.createReadStream(EDITOR_HTML).pipe(response);
  } else if (request.method === "GET" && pathname === "/api/pieces") {
    sendJson(response, 200, listPieces());
  } else if (request.method === "GET" && pathname.startsWith("/source/")) {
    const source = pathname.slice("/source/".length);
    if (source.startsWith(ORIGINALS)) {
      sendFile(response, path.join(ROOT, ORIGINALS), source.slice(ORIGINALS.length));
    } else {
      sendFile(response, exportDir, source);
    }
  } else if (request.method === "GET" && pathname.startsWith("/images/art/")) {
    sendFile(response, path.join(ROOT, "public"), pathname.slice(1));
  } else if (request.method === "POST" && pathname === "/api/frame") {
    let body = "";
    request.on("data", (chunk) => (body += chunk));
    request.on("end", () => {
      try {
        const piece = saveFrame(JSON.parse(body));
        console.log(`saved ${piece.title}`);
        sendJson(response, 200, piece);
      } catch (error) {
        sendJson(response, 400, { error: error.message.split("\n")[0] });
      }
    });
  } else {
    response.writeHead(404).end("Not found");
  }
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`Art editor: http://localhost:${PORT}  (photos from ${exportDir})`);
});
