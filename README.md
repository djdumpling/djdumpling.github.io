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
