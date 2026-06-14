import type { Metadata } from "next";

import {
  formatArchiveDate,
  getPostsNewestFirst,
} from "@/lib/posts";
import { publicPageHref, SITE } from "@/lib/site";

export const metadata: Metadata = {
  title: "Blog Archive",
  alternates: { canonical: "/archive.html" },
  openGraph: {
    title: "Blog Archive",
    description: SITE.description,
    url: "/archive.html",
    type: "website",
  },
  twitter: {
    title: "Blog Archive",
    description: SITE.description,
    card: "summary",
  },
};

export default async function ArchivePage() {
  const posts = (await getPostsNewestFirst()).filter((post) => post.archive);
  const groups = new Map<string, typeof posts>();

  for (const post of posts) {
    for (const tag of post.tags) {
      const taggedPosts = groups.get(tag) ?? [];
      taggedPosts.push(post);
      groups.set(tag, taggedPosts);
    }
  }

  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "WebPage",
    author: { "@type": "Person", name: SITE.author },
    description: SITE.description,
    headline: "Blog Archive",
    url: `${SITE.url}/archive.html`,
  };

  return (
    <article className="post">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(jsonLd).replaceAll("<", "\\u003c"),
        }}
      />
      <header className="post-header">
        <h1 className="post-title">Blog Archive</h1>
      </header>
      <div className="post-content">
        {[...groups.entries()].map(([tag, taggedPosts]) => (
          <section key={tag}>
            <h3>{tag}</h3>
            <ul>
              {taggedPosts.map((post) => (
                <li key={post.legacyUrl}>
                  <a href={publicPageHref(post.route, post.legacyUrl)}>
                    {formatArchiveDate(post.date)} - {post.title}
                  </a>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </article>
  );
}
