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
