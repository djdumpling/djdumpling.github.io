import { describe, expect, it } from "vitest";

import {
  getPostsChronological,
  getPostsNewestFirst,
  parsePostFile,
} from "@/lib/posts";

describe("post loading", () => {
  it("uses the frontmatter date and preserves filename slug case", async () => {
    const post = await parsePostFile(
      "2025-01-01-Case_Sensitive.md",
      `---
title: Example
date: 2026-02-03
tokens: "~1k"
reading_time: 4
---

First paragraph.
`,
    );

    expect(post.slug).toBe("Case_Sensitive");
    expect(post.route).toBe("/2026/02/03/Case_Sensitive");
    expect(post.legacyUrl).toBe("/2026/02/03/Case_Sensitive.html");
    expect(post.tags).toEqual(["Other"]);
    expect(post.author).toBe("Alex Wa");
  });

  it("loads all posts in chronological and reverse order", async () => {
    const chronological = await getPostsChronological();
    const newestFirst = await getPostsNewestFirst();

    // Only `_posts/` is published; archived_posts/ is local-only.
    expect(chronological).toHaveLength(5);
    expect(chronological[0].slug).toBe("rlhf_gpt2");
    expect(chronological.at(-1)?.slug).toBe("modded-nanoGPT-WR");
    expect(newestFirst.map((post) => post.slug)).toEqual(
      [...chronological].reverse().map((post) => post.slug),
    );
  });

  it("publishes every post in _posts and honors visibility flags", async () => {
    const posts = await getPostsNewestFirst();

    // Every current post is published on both the home page and the archive.
    expect(posts).toHaveLength(5);
    expect(posts.filter((post) => !post.ongoing)).toHaveLength(5);
    expect(posts.filter((post) => post.archive)).toHaveLength(5);

    // The flag semantics still apply when frontmatter sets them.
    const hidden = await parsePostFile(
      "2026-12-31-hidden.md",
      `---
title: Hidden
date: 2026-12-31
ongoing: true
archive: false
---

Body.
`,
    );
    expect(hidden).toMatchObject({ ongoing: true, archive: false });
  });
});
