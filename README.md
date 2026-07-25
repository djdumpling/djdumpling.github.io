# Alex Wa's Blog

This blog is built with Next.js and statically exported for GitHub Pages.
Posts remain Markdown files under `_posts/`.

## Local development

Use Node.js 24, then install dependencies and start the development server:

```bash
npm install
npm run dev
```

Open <http://localhost:3000>. Development links use extensionless routes;
the production export retains the existing `.html` URLs.

## Unlisted drafts

An unlisted draft is still generated and can be opened by its exact URL, but it
is omitted from the homepage, archive, previous/next navigation, feed, and
sitemap. Add `unlisted: true` to the post front matter:

```md
---
title: "My draft"
date: 2026-07-25
unlisted: true
---

Draft content goes here.
```

Save it as `_posts/2026-07-25-my-draft.md`. In production its URL will be
`/2026/07/25/my-draft.html`. Unlisted pages also emit `noindex` metadata, but
they are public rather than access-controlled; do not use this for secrets.

For images that should also render in an editor's Markdown preview, reference
their repository path relative to `_posts/`, such as
`../public/public/example.svg`. The site build converts that to the corresponding
public URL.

## Production preview

```bash
npm run build
npm run preview
```

Open <http://localhost:3000> unless `serve` selects another available port.

## Verification

```bash
npm run verify
```

The GitHub Pages workflow runs the same source and artifact checks before
deploying the `out/` directory.
