import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { PostNavigation } from "@/components/PostNavigation";
import { ShareLinks } from "@/components/ShareLinks";
import {
  formatPostDate,
  getPostBySegments,
  getPostsChronological,
} from "@/lib/posts";
import { absoluteUrl, SITE } from "@/lib/site";

type PostPageParams = {
  year: string;
  month: string;
  day: string;
  slug: string;
};

type PostPageProps = {
  params: Promise<PostPageParams>;
};

export const dynamicParams = false;

export async function generateStaticParams(): Promise<PostPageParams[]> {
  return (await getPostsChronological()).map((post) => {
    const [, year, month, day, slug] = post.route.split("/");
    return { year, month, day, slug };
  });
}

export async function generateMetadata({
  params,
}: PostPageProps): Promise<Metadata> {
  const { year, month, day, slug } = await params;
  const post = await getPostBySegments(year, month, day, slug);

  if (!post) {
    return {};
  }

  return {
    title: post.title,
    description: post.excerptText,
    authors: [{ name: post.author }],
    alternates: { canonical: post.legacyUrl },
    openGraph: {
      title: post.title,
      description: post.excerptText,
      url: post.legacyUrl,
      siteName: SITE.title,
      locale: "en_US",
      type: "article",
      publishedTime: post.publishedIso,
      modifiedTime: post.publishedIso,
      images: post.image ? [post.image] : undefined,
    },
    twitter: {
      title: post.title,
      description: post.excerptText,
      card: post.image ? "summary_large_image" : "summary",
      images: post.image ? [post.image] : undefined,
    },
  };
}

export default async function PostPage({ params }: PostPageProps) {
  const { year, month, day, slug } = await params;
  const post = await getPostBySegments(year, month, day, slug);

  if (!post) {
    notFound();
  }

  const posts = await getPostsChronological();
  const postIndex = posts.findIndex(
    (candidate) => candidate.legacyUrl === post.legacyUrl,
  );
  const previous = postIndex > 0 ? posts[postIndex - 1] : undefined;
  const next = postIndex < posts.length - 1 ? posts[postIndex + 1] : undefined;
  const canonical = absoluteUrl(post.legacyUrl);
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "BlogPosting",
    author: { "@type": "Person", name: post.author },
    dateModified: post.publishedIso,
    datePublished: post.publishedIso,
    description: post.excerptText,
    headline: post.title,
    image: post.image ? absoluteUrl(post.image) : undefined,
    mainEntityOfPage: { "@type": "WebPage", "@id": canonical },
    url: canonical,
  };

  return (
    <article
      className="post h-entry"
      itemScope
      itemType="http://schema.org/BlogPosting"
    >
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(jsonLd).replaceAll("<", "\\u003c"),
        }}
      />
      <header className="post-header">
        <h1 className="post-title p-name" itemProp="name headline">
          {post.title}
        </h1>
        <p className="post-meta">
          <time
            className="dt-published"
            dateTime={post.publishedIso}
            itemProp="datePublished"
          >
            {formatPostDate(post.date)}
          </time>{" "}
          •{" "}
          <span
            itemProp="author"
            itemScope
            itemType="http://schema.org/Person"
          >
            <span className="p-author h-card" itemProp="name">
              {post.author}
            </span>
          </span>
        </p>
      </header>

      <ShareLinks title={post.title} url={post.legacyUrl} />

      <div className="post-content e-content" itemProp="articleBody">
        <div dangerouslySetInnerHTML={{ __html: post.html }} />
        <PostNavigation previous={previous} next={next} />
      </div>

      <a className="u-url" href={post.legacyUrl} hidden>
        {post.title}
      </a>
    </article>
  );
}
