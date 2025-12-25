# Jekyll to Next.js Migration

## Background and Motivation

The user wants to migrate their Jekyll blog (djdumpling.github.io) to Next.js. The current site includes:
- 3 blog posts with heavy LaTeX math, code blocks, images, tables, YouTube embeds
- Homepage with profile section
- Blog archive page
- Custom CSS styling (code highlighting, responsive layout)
- Google Analytics, RSS feed, sitemap
- Navigation between posts, share links

Goals:
- Preserve all existing content and functionality
- Enable local development with hot reload
- Modern React-based architecture
- Same visual appearance

## Key Challenges and Analysis

1. **MDX with LaTeX**: Need `next-mdx-remote` + `remark-math` + `rehype-katex` for math rendering
2. **Code syntax highlighting**: Use `rehype-highlight` or similar
3. **HTML in Markdown**: MDX supports this natively
4. **Image paths**: Keep `/public/` structure identical
5. **Frontmatter**: Use `gray-matter` for parsing
6. **Dynamic routes**: `app/blog/[slug]/page.tsx` pattern
7. **RSS + Sitemap**: Use `next-sitemap` package
8. **Styling**: Migrate CSS and adapt for Next.js

## High-level Task Breakdown

### Phase 1: Project Setup
- [ ] 1.1 Create Next.js project structure in a new `nextjs-blog/` folder
- [ ] 1.2 Install dependencies (next-mdx-remote, gray-matter, remark-math, rehype-katex, rehype-highlight)
- [ ] 1.3 Configure next.config.js for MDX and static export
- [ ] 1.4 Set up TypeScript types

### Phase 2: Core Components
- [ ] 2.1 Create layout component with header/footer
- [ ] 2.2 Port CSS styles from override.css
- [ ] 2.3 Create MDX components for custom rendering

### Phase 3: Content Migration
- [ ] 3.1 Convert blog posts to MDX format in `content/posts/`
- [ ] 3.2 Create blog post page with dynamic routing
- [ ] 3.3 Create homepage with profile section
- [ ] 3.4 Create archive page

### Phase 4: Features
- [ ] 4.1 Add navigation links between posts
- [ ] 4.2 Add share links component
- [ ] 4.3 Configure Google Analytics
- [ ] 4.4 Generate RSS feed and sitemap

### Phase 5: Testing & Polish
- [ ] 5.1 Test locally with `npm run dev`
- [ ] 5.2 Verify LaTeX rendering
- [ ] 5.3 Verify code highlighting
- [ ] 5.4 Verify images and responsive design
- [ ] 5.5 Final cleanup

## Project Status Board

- [ ] Phase 1: Project Setup
- [ ] Phase 2: Core Components  
- [ ] Phase 3: Content Migration
- [ ] Phase 4: Features
- [ ] Phase 5: Testing & Polish

## Executor's Feedback or Assistance Requests

*Starting Phase 1...*

## Lessons

- (To be filled during execution)

