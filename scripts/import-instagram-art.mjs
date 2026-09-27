// Imports the cover image of every post in an Instagram "Download your
// information" export into public/images/art and records it in lib/art.json.
//
//   node scripts/import-instagram-art.mjs [export-dir]
//
// The export defaults to $ART_EXPORT_DIR or the newest
// ~/Downloads/instagram-alex_wa_art-* folder.
//
// Only posts that lib/art.json doesn't already know about are added, so
// hand-edited pieces (titles, mediums, sections, extra carousel images, crops)
// are left alone. Posts listed in `ignoredPosts` are never re-imported. New
// pieces default to the "mixed" section; move them to the right one by hand.
// Requires ImageMagick (`magick`) for resizing and placeholder colors.
import fs from "node:fs";
import path from "node:path";

import {
  DATA_FILE,
  FULL_EDGE,
  ROOT,
  THUMB_WIDTH,
  findExportDir,
  magick,
  readArtData,
  writeArtData,
} from "./frame-art.mjs";

const exportDir = process.argv[2] ?? findExportDir();
if (!exportDir) {
  console.error("usage: node scripts/import-instagram-art.mjs <export-dir>");
  process.exit(1);
}

const IMAGE_DIR = path.join(ROOT, "public/images/art");
const THUMB_DIR = path.join(IMAGE_DIR, "thumbs");

// Instagram exports UTF-8 text as if it were Latin-1.
function fixEncoding(text) {
  try {
    return Buffer.from(text, "latin1").toString("utf8");
  } catch {
    return text;
  }
}

function localDate(timestamp) {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York",
  }).format(new Date(timestamp * 1000));
}

const postsFile = path.join(
  exportDir,
  "your_instagram_activity/media/posts_1.json",
);
const posts = JSON.parse(fs.readFileSync(postsFile, "utf8"));
const data = fs.existsSync(DATA_FILE)
  ? readArtData()
  : { ignoredPosts: [], pieces: [] };
const known = new Set([
  ...data.ignoredPosts,
  ...data.pieces.map((piece) => piece.post),
]);

fs.mkdirSync(THUMB_DIR, { recursive: true });

const added = posts.flatMap((post) => {
  const cover = post.media[0];
  const id = path.basename(cover.uri, path.extname(cover.uri));
  if (known.has(id)) {
    return [];
  }

  const timestamp = post.creation_timestamp ?? cover.creation_timestamp;
  const date = localDate(timestamp);
  const filename = `${date}-${id}.jpg`;
  const full = path.join(IMAGE_DIR, filename);
  const thumb = path.join(THUMB_DIR, filename);
  const source = path.join(exportDir, cover.uri);

  magick(source, "-auto-orient", "-strip", "-resize", `${FULL_EDGE}x${FULL_EDGE}>`, "-quality", "84", full);
  magick(full, "-resize", `${THUMB_WIDTH}x`, "-quality", "80", "-interlace", "Plane", thumb);

  const [width, height] = magick("identify", "-format", "%w %h", full)
    .split(" ")
    .map(Number);
  const color = magick(full, "-resize", "1x1!", "-format", "#%[hex:u.p{0,0}]", "info:")
    .slice(0, 7)
    .toLowerCase();
  // The Instagram caption's first line is only a starting title to edit.
  const caption = fixEncoding(post.title || cover.title || "").trim();

  return [
    {
      id,
      post: id,
      date,
      src: `/images/art/${filename}`,
      thumb: `/images/art/thumbs/${filename}`,
      width,
      height,
      color,
      title: caption.split("\n")[0].slice(0, 80),
      medium: "",
      section: "mixed",
    },
  ];
});

data.pieces = [...data.pieces, ...added].sort(
  (a, b) => b.date.localeCompare(a.date),
);
writeArtData(data);
console.log(
  `Added ${added.length} new piece(s); ${data.pieces.length} total in ${path.relative(ROOT, DATA_FILE)}`,
);
