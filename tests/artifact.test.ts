import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const ROOT = process.cwd();
const OUT = path.join(ROOT, "out");
const expectedPages = [
  "index.html",
  "archive.html",
  "404.html",
  "feed.xml",
  "sitemap.xml",
  "robots.txt",
  "2025/08/04/rlhf_gpt2.html",
  "2025/11/24/rl_envs.html",
  "2025/12/14/SEA-privacy.html",
  "2026/01/31/frontier_training.html",
  "2026/05/27/modded-nanoGPT-WR.html",
];

// Posts moved to archived_posts/ must never be exported.
const excludedPages = ["2026/12/31/reading_every_day.html"];

function walk(directory: string): string[] {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const entryPath = path.join(directory, entry.name);
    return entry.isDirectory() ? walk(entryPath) : [entryPath];
  });
}

function outputHtmlFiles(): string[] {
  return walk(OUT).filter((filename) => filename.endsWith(".html"));
}

describe("static export", () => {
  it("emits every legacy route exactly once", () => {
    for (const page of expectedPages) {
      expect(fs.existsSync(path.join(OUT, page)), page).toBe(true);
    }
    for (const page of excludedPages) {
      expect(fs.existsSync(path.join(OUT, page)), page).toBe(false);
    }
    expect(walk(OUT).filter((filename) => filename.endsWith(".html.html"))).toEqual(
      [],
    );
  });

  it("retains the Jekyll heading IDs used by the long-form TOC", () => {
    const expected = fs
      .readFileSync(
        path.join(ROOT, "tests/fixtures/frontier-heading-ids.txt"),
        "utf8",
      )
      .trim()
      .split("\n");
    const html = fs.readFileSync(
      path.join(OUT, "2026/01/31/frontier_training.html"),
      "utf8",
    );
    const actual = [...html.matchAll(/<h[1-6] id="([^"]+)"/g)].map(
      (match) => match[1],
    );

    expect(actual).toEqual(expected);
  });

  it("preserves MathJax source and the Fruit Box video", () => {
    const postHtml = outputHtmlFiles()
      .map((filename) => fs.readFileSync(filename, "utf8"))
      .join("\n");
    const fruitBox = fs.readFileSync(
      path.join(OUT, "2025/11/24/rl_envs.html"),
      "utf8",
    );

    expect(postHtml).not.toContain("language-math");
    expect(postHtml).toContain("$\\tanh(\\alpha)$");
    expect(postHtml).toContain("$$");
    expect(fruitBox).toContain(
      'src="https://www.youtube.com/embed/Zja_MsGDKSI"',
    );
    expect(fruitBox).toContain('allowfullscreen');
  });

  it("resolves every local image and internal page link", () => {
    const missing = new Set<string>();

    for (const filename of outputHtmlFiles()) {
      const html = fs.readFileSync(filename, "utf8");
      for (const match of html.matchAll(/(?:href|src)="(\/[^"#?]+)(?:[#?][^"]*)?"/g)) {
        const url = match[1];
        if (url.startsWith("/_next/")) {
          continue;
        }

        const relativePath =
          url === "/"
            ? "index.html"
            : url.endsWith(".html") || path.extname(url)
              ? url.slice(1)
              : `${url.slice(1)}.html`;
        if (!fs.existsSync(path.join(OUT, relativePath))) {
          missing.add(url);
        }
      }
    }

    expect([...missing]).toEqual([]);
  });

  it("resolves every in-page anchor link to an element id", () => {
    const broken: string[] = [];

    for (const filename of outputHtmlFiles()) {
      const html = fs.readFileSync(filename, "utf8");
      const ids = new Set(
        [...html.matchAll(/\sid="([^"]+)"/g)].map((match) => match[1]),
      );
      for (const match of html.matchAll(/href="#([^"]+)"/g)) {
        const target = decodeURIComponent(match[1]);
        if (!ids.has(target)) {
          broken.push(`${path.relative(OUT, filename)} -> #${target}`);
        }
      }
    }

    expect(broken).toEqual([]);
  });

  it("generates complete feed and sitemap documents", () => {
    const feed = fs.readFileSync(path.join(OUT, "feed.xml"), "utf8");
    const sitemap = fs.readFileSync(path.join(OUT, "sitemap.xml"), "utf8");

    expect(feed.match(/<entry>/g)).toHaveLength(5);
    expect(feed).not.toContain("reading_every_day");
    expect(sitemap.match(/<url>/g)).toHaveLength(7);
    expect(sitemap).not.toContain("reading_every_day");
    expect(sitemap).not.toContain("localhost");
  });
});
