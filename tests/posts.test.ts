import { describe, expect, it } from "vitest";

import {
  getPostsChronological,
  getPostsNewestFirst,
  getListedPostsChronological,
  getListedPostsNewestFirst,
  isPostListed,
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
    expect(post.unlisted).toBe(false);
  });

  it("loads all posts in chronological and reverse order", async () => {
    const chronological = await getPostsChronological();
    const newestFirst = await getPostsNewestFirst();

    // Only `_posts/` is published; archived_posts/ is local-only.
    expect(chronological).toHaveLength(6);
    expect(chronological[0].slug).toBe("rlhf_gpt2");
    expect(chronological.at(-1)?.slug).toBe("my-draft");
    expect(newestFirst.map((post) => post.slug)).toEqual(
      [...chronological].reverse().map((post) => post.slug),
    );
  });

  it("publishes every post in _posts and honors visibility flags", async () => {
    const posts = await getPostsNewestFirst();

    // Every current post is published on both the home page and the archive.
    expect(posts).toHaveLength(6);
    expect(
      posts.filter((post) => isPostListed(post) && !post.ongoing),
    ).toHaveLength(5);
    expect(
      posts.filter((post) => isPostListed(post) && post.archive),
    ).toHaveLength(5);
    expect(posts.filter((post) => post.unlisted)).toHaveLength(1);

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

  it("keeps unlisted posts addressable but out of public post collections", async () => {
    const unlisted = await parsePostFile(
      "2026-12-31-secret-draft.md",
      `---
title: Secret draft
date: 2026-12-31
unlisted: true
---

Body.
`,
    );

    expect(unlisted).toMatchObject({
      route: "/2026/12/31/secret-draft",
      legacyUrl: "/2026/12/31/secret-draft.html",
      unlisted: true,
    });
    expect(isPostListed(unlisted)).toBe(false);

    const chronological = await getListedPostsChronological();
    const newestFirst = await getListedPostsNewestFirst();
    expect(newestFirst).toEqual([...chronological].reverse());
  });
});
