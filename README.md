# Alex Wa's Blog

A Next.js blog migrated from Jekyll, featuring MDX support for markdown posts with LaTeX math rendering and syntax highlighting.

## Features

- ✅ MDX blog posts with frontmatter
- ✅ LaTeX math rendering (KaTeX)
- ✅ Code syntax highlighting (highlight.js)
- ✅ Responsive design
- ✅ Google Analytics
- ✅ RSS feed
- ✅ Sitemap generation
- ✅ Static site generation (SSG) for GitHub Pages

## Getting Started

### Install Dependencies

```bash
npm install
```

### Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to see your blog locally.

### Build for Production

```bash
npm run build
```

This will:
1. Build the Next.js static site
2. Generate RSS feed
3. Generate sitemap

The output will be in the `out/` directory, ready for deployment to GitHub Pages.

## Project Structure

```
├── app/                    # Next.js app directory
│   ├── layout.tsx         # Root layout with header/footer
│   ├── page.tsx           # Homepage
│   ├── blog/[slug]/       # Blog post pages
│   └── archive/           # Archive page
├── content/posts/         # MDX blog posts
├── lib/                   # Utility functions
│   └── posts.ts          # Post loading utilities
├── components/            # React components
│   └── ShareLinks.tsx    # Social sharing buttons
├── public/                # Static assets (images, etc.)
└── scripts/               # Build scripts
    └── generate-rss.ts    # RSS feed generator
```

## Adding a New Blog Post

1. Create a new `.mdx` file in `content/posts/`
2. Add frontmatter:

```mdx
---
title: "Your Post Title"
date: 2025-01-15
tags: [Other]
---

Your content here...
```

3. Use standard markdown + MDX features
4. LaTeX math: `$inline$` or `$$display$$`
5. Code blocks: Use triple backticks with language

## Deployment

### GitHub Pages

1. Build the site: `npm run build`
2. The `out/` directory contains the static site
3. Configure GitHub Pages to serve from the `out/` directory (or use GitHub Actions)

### GitHub Actions (Recommended)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '20'
      - run: npm install
      - run: npm run build
      - uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./out
```

## Migration Notes

This blog was migrated from Jekyll. Key differences:

- Posts are now in `content/posts/` instead of `_posts/`
- Frontmatter format is the same (YAML)
- Images still use `/public/` paths
- LaTeX uses KaTeX instead of MathJax (faster)
- Code highlighting uses highlight.js (same as before)

## License

Personal blog - all rights reserved.

