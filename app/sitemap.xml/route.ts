import { getPostsChronological } from "@/lib/posts";
import { absoluteUrl } from "@/lib/site";
import { escapeXml } from "@/lib/xml";

export const dynamic = "force-static";

export async function GET() {
  const posts = await getPostsChronological();
  const postEntries = posts
    .map(
      (post) => `<url>
  <loc>${escapeXml(absoluteUrl(post.legacyUrl))}</loc>
  <lastmod>${post.publishedIso}</lastmod>
</url>`,
    )
    .join("\n");
  const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.sitemaps.org/schemas/sitemap/0.9 http://www.sitemaps.org/schemas/sitemap/0.9/sitemap.xsd" xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${postEntries}
<url><loc>${escapeXml(absoluteUrl("/archive.html"))}</loc></url>
<url><loc>${escapeXml(absoluteUrl("/"))}</loc></url>
</urlset>`;

  return new Response(sitemap, {
    headers: { "Content-Type": "application/xml; charset=utf-8" },
  });
}
