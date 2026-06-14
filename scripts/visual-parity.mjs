import fs from "node:fs/promises";
import path from "node:path";
import { chromium } from "@playwright/test";

const legacyBase = process.env.LEGACY_BASE_URL ?? "http://127.0.0.1:4000";
const nextBase = process.env.NEXT_BASE_URL ?? "http://127.0.0.1:4173";
const outputDirectory = path.resolve("test-results/visual-parity");
const cases = [
  { name: "home", path: "/" },
  { name: "archive", path: "/archive.html" },
  { name: "rlhf", path: "/2025/08/04/rlhf_gpt2.html" },
  {
    name: "fruit-box-video",
    path: "/2025/11/24/rl_envs.html#fruit-box-and-intuition",
  },
  {
    name: "frontier-training",
    path: "/2026/01/31/frontier_training.html#post-training",
  },
  {
    name: "nanogpt",
    path: "/2026/05/27/modded-nanoGPT-WR.html",
  },
];
const viewports = [
  { name: "desktop", width: 1440, height: 1000 },
  { name: "mobile", width: 390, height: 844 },
];

async function preparePage(page, url) {
  await page.goto(url, { waitUntil: "networkidle" });
  await page.locator("iframe").evaluateAll((frames) => {
    for (const frame of frames) {
      frame.style.visibility = "hidden";
    }
  });
  await page.waitForTimeout(500);
}

await fs.mkdir(outputDirectory, { recursive: true });
const browser = await chromium.launch(
  process.platform === "darwin" ? { channel: "chrome" } : {},
);

try {
  for (const viewport of viewports) {
    const context = await browser.newContext({
      viewport: { width: viewport.width, height: viewport.height },
    });
    const legacyPage = await context.newPage();
    const nextPage = await context.newPage();

    for (const visualCase of cases) {
      await preparePage(legacyPage, `${legacyBase}${visualCase.path}`);
      await preparePage(nextPage, `${nextBase}${visualCase.path}`);

      await legacyPage.screenshot({
        path: path.join(
          outputDirectory,
          `${visualCase.name}-${viewport.name}-jekyll.png`,
        ),
      });
      await nextPage.screenshot({
        path: path.join(
          outputDirectory,
          `${visualCase.name}-${viewport.name}-next.png`,
        ),
      });
    }

    await context.close();
  }
} finally {
  await browser.close();
}

console.log(`Wrote side-by-side parity captures to ${outputDirectory}`);
