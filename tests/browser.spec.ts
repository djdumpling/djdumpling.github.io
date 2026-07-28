import { expect, test } from "@playwright/test";

test("home and archive preserve their public navigation", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveTitle(/Alex Wa's Blog/);
  await expect(page.locator(".profile-image")).toBeVisible();
  await expect(page.locator(".post-item")).toHaveCount(5);
  await expect(
    page.locator('a[href="/2026/05/27/modded-nanoGPT-WR.html"]'),
  ).toBeVisible();

  await page.goto("/archive.html");
  await expect(page.getByRole("heading", { name: "Blog Archive" })).toBeVisible();
  await expect(page.locator(".post-content li")).toHaveCount(5);
});

test("post media, equations, details, and navigation survive export", async ({
  page,
}) => {
  await page.goto("/2025/11/24/rl_envs.html");
  await expect(
    page.locator('iframe[src*="youtube.com/embed/Zja_MsGDKSI"]'),
  ).toBeAttached();
  await expect(page.locator("details")).toHaveCount(6);
  await expect(page.locator("pre code.language-python").first()).toBeAttached();
  await expect(page.locator(".post_navi-item")).toHaveCount(2);
  await expect(page.locator("body")).toContainText(
    "combinatorial reasoning environments for LLMs and RL",
  );
});

test("long-form table-of-contents anchors resolve", async ({ page }) => {
  await page.goto("/2026/01/31/frontier_training.html");
  await expect(page.locator("#hermes-4")).toBeAttached();
  await expect(page.locator("#hermes-4-1")).toBeAttached();

  const brokenAnchors = await page.locator('.post-content a[href^="#"]').evaluateAll(
    (links) =>
      links
        .map((link) => link.getAttribute("href"))
        .filter(
          (href): href is string =>
            Boolean(href) && !document.querySelector(href as string),
        ),
  );
  expect(brokenAnchors).toEqual([]);
});

test("post images are centered at 80 percent width", async ({ page }) => {
  await page.goto("/2026/07/25/my-draft.html");

  const images = page.locator(".post-content img");
  await expect(images.first()).toBeVisible();
  const imageWidthRatios = await images.evaluateAll((elements) =>
    elements.map(
      (image) =>
        image.getBoundingClientRect().width /
        (image.parentElement?.getBoundingClientRect().width ?? 1),
    ),
  );
  expect(imageWidthRatios.every((ratio) => Math.abs(ratio - 0.8) < 0.01)).toBe(
    true,
  );
});

test("long unlisted posts finish MathJax typesetting", async ({ page }) => {
  await page.goto("/2026/07/25/my-draft.html");

  const criticParagraph = page.locator(
    "#critic-target + p",
  );
  await expect(criticParagraph.locator("mjx-container")).toHaveCount(2, {
    timeout: 15_000,
  });
  await expect
    .poll(() => page.locator("mjx-container").count(), { timeout: 15_000 })
    .toBeGreaterThan(200);
});
