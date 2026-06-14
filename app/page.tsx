import type { Metadata } from "next";

import {
  formatHomeDate,
  getPostsNewestFirst,
} from "@/lib/posts";
import { publicPageHref, SITE } from "@/lib/site";

export const metadata: Metadata = {
  alternates: { canonical: "/" },
  openGraph: {
    title: SITE.title,
    description: SITE.description,
    url: "/",
    type: "website",
  },
  twitter: {
    title: SITE.title,
    description: SITE.description,
    card: "summary",
  },
};

export default async function HomePage() {
  const posts = (await getPostsNewestFirst()).filter((post) => !post.ongoing);
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "WebSite",
    author: { "@type": "Person", name: SITE.author },
    description: SITE.description,
    headline: SITE.title,
    name: SITE.title,
    url: `${SITE.url}/`,
  };

  return (
    <div className="home">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(jsonLd).replaceAll("<", "\\u003c"),
        }}
      />
      <div className="profile-container">
        <img src="/public/pfp.jpg" alt="Alex Wa" className="profile-image" />
        <div className="profile-content">
          <p>
            Hey! I&apos;m Alex Wa, a 2nd year Math and CS major at Yale. My
            research currently span RL and NLP, and I&apos;m also interested in
            LLM architecture and ML systems.
          </p>
          <p>
            This summer, I&apos;m interning with the RL and training team at{" "}
            <a href="https://modal.com/" target="_blank">
              Modal
            </a>
            . Previously, I&apos;ve developed RL environments in{" "}
            <a
              href="https://app.primeintellect.ai/dashboard/environments"
              target="_blank"
            >
              Prime Intellect
            </a>
            &apos;s RL Residency, researched RL4LLMs and rubrics with the{" "}
            <a href="https://nlp.cs.yale.edu/" target="_blank">
              Yale NLP lab
            </a>
            , and researching web agents for human behavior prediction.
            I&apos;ve also done research in rubrics (
            <a href="https://www.judgmentlabs.ai/" target="_blank">
              Judgment Labs
            </a>
            ), geometric algebra (APOLLO Labs), algebraic topology (SUMaC
            &apos;23), abstract algebra (SUMaC &apos;22), and biostatistics
            (Emory). Check out our ICLR NFAM Workshop 2026 submission{" "}
            <a href="https://arxiv.org/abs/2603.03464" target="_blank">
              here
            </a>
            .
          </p>
          <p>
            In my free time, I{" "}
            <a href="https://www.instagram.com/alex_wa_art/" target="_blank">
              draw
            </a>
            , run{" "}
            <a href="https://www.theveritassearch.com/" target="_blank">
              The Veritas Search
            </a>
            , play board games like Catan, and enjoy photography.
          </p>
        </div>
      </div>

      <div className="home-sections">
        <h2 className="post-list-heading">Posts</h2>
        <div className="post-list">
          {posts.map((post) => (
            <div className="post-item" key={post.legacyUrl}>
              <div className="post-date">{formatHomeDate(post.date)}</div>
              <h3 className="post-title">
                <a
                  className="post-link"
                  href={publicPageHref(post.route, post.legacyUrl)}
                >
                  {post.title}
                </a>
              </h3>
              {post.tokens ? (
                <div className="post-meta">{post.tokens} tokens</div>
              ) : null}
              {post.excerptHtml ? (
                <div
                  className="post-summary"
                  dangerouslySetInnerHTML={{ __html: post.excerptHtml }}
                />
              ) : null}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
