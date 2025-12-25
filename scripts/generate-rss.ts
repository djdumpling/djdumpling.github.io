import fs from 'fs';
import path from 'path';
import { getAllPosts } from '../lib/posts';

const siteUrl = 'https://djdumpling.github.io';

function escapeXml(unsafe: string): string {
  return unsafe
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function generateRSS() {
  const posts = getAllPosts();

  const rss = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>Alex Wa's Blog</title>
    <description>Alex Wa's blog; a mix of projects, research and life. Currently interested in RL, NLP, and ML systems.</description>
    <link>${siteUrl}</link>
    <atom:link href="${siteUrl}/feed.xml" rel="self" type="application/rss+xml"/>
    <language>en-us</language>
    <lastBuildDate>${new Date().toUTCString()}</lastBuildDate>
    ${posts
      .map(
        (post) => `    <item>
      <title>${escapeXml(post.title)}</title>
      <description>${escapeXml(post.excerpt || post.title)}</description>
      <link>${siteUrl}/blog/${post.slug}/</link>
      <guid isPermaLink="true">${siteUrl}/blog/${post.slug}/</guid>
      <pubDate>${new Date(post.date).toUTCString()}</pubDate>
    </item>`
      )
      .join('\n')}
  </channel>
</rss>`;

  const outputPath = path.join(process.cwd(), 'public', 'feed.xml');
  fs.writeFileSync(outputPath, rss);
  console.log('RSS feed generated at', outputPath);
}

generateRSS();

