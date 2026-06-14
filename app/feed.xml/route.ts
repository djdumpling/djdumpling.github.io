import { getPostsNewestFirst } from "@/lib/posts";
import { absoluteUrl, SITE } from "@/lib/site";
import { cdata, escapeXml } from "@/lib/xml";

export const dynamic = "force-static";

export async function GET() {
  const posts = await getPostsNewestFirst();
  const updated = posts[0]?.publishedIso ?? new Date().toISOString();
  const entries = posts
    .map(
      (post) => `<entry>
  <title type="html">${escapeXml(post.title)}</title>
  <link href="${escapeXml(absoluteUrl(post.legacyUrl))}" rel="alternate" type="text/html" title="${escapeXml(post.title)}" />
  <published>${post.publishedIso}</published>
  <updated>${post.publishedIso}</updated>
  <id>${escapeXml(absoluteUrl(post.route))}</id>
  <content type="html" xml:base="${escapeXml(absoluteUrl(post.legacyUrl))}">${cdata(post.html)}</content>
  <author><name>${escapeXml(post.author)}</name></author>
</entry>`,
    )
    .join("\n");

  const feed = `<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <generator uri="https://nextjs.org/" version="16.2.9">Next.js</generator>
  <link href="${SITE.url}/feed.xml" rel="self" type="application/atom+xml" />
  <link href="${SITE.url}/" rel="alternate" type="text/html" />
  <updated>${updated}</updated>
  <id>${SITE.url}/feed.xml</id>
  <title type="html">${escapeXml(SITE.title)}</title>
  <subtitle>${escapeXml(SITE.description)}</subtitle>
  <author><name>${escapeXml(SITE.author)}</name></author>
${entries}
</feed>`;

  return new Response(feed, {
    headers: { "Content-Type": "application/atom+xml; charset=utf-8" },
  });
}
