import { cache } from "react";
import fs from "node:fs/promises";
import path from "node:path";
import matter from "gray-matter";
import { z } from "zod";

import {
  firstParagraphs,
  htmlToText,
  renderMarkdown,
} from "@/lib/markdown";
import { SITE } from "@/lib/site";

const POSTS_DIRECTORY = path.join(process.cwd(), "_posts");

const frontmatterSchema = z.object({
  title: z.string().min(1),
  date: z.union([z.string(), z.date()]),
  author: z.string().optional(),
  tags: z.union([z.string(), z.array(z.string())]).optional(),
  image: z.string().optional(),
  excerpt_paragraphs: z.number().int().positive().optional(),
  tokens: z.string().optional(),
  reading_time: z.number().int().positive().optional(),
  ongoing: z.boolean().optional(),
  archive: z.boolean().optional(),
  unlisted: z.boolean().optional(),
});

export type PostFrontmatter = {
  title: string;
  date: string;
  author: string;
  tags: string[];
  image?: string;
  excerptParagraphs: number;
  tokens?: string;
  readingTime?: number;
  ongoing: boolean;
  archive: boolean;
  unlisted: boolean;
};

export type Post = PostFrontmatter & {
  filename: string;
  slug: string;
  route: string;
  legacyUrl: string;
  html: string;
  excerptHtml: string;
  excerptText: string;
  source: string;
  publishedIso: string;
};

function normalizedDate(value: string | Date): string {
  if (value instanceof Date) {
    return value.toISOString().slice(0, 10);
  }

  const match = value.match(/^\d{4}-\d{2}-\d{2}/);
  if (!match) {
    throw new Error(`Expected a YYYY-MM-DD date, received "${value}"`);
  }
  return match[0];
}

function normalizeTags(value: string | string[] | undefined): string[] {
  if (!value) {
    return ["Other"];
  }
  return Array.isArray(value) ? value : [value];
}

export function easternIsoDate(date: string): string {
  const midday = new Date(`${date}T12:00:00Z`);
  const offsetName = new Intl.DateTimeFormat("en-US", {
    timeZone: SITE.timeZone,
    timeZoneName: "longOffset",
  })
    .formatToParts(midday)
    .find((part) => part.type === "timeZoneName")?.value;
  const offset = offsetName?.replace("GMT", "") || "-05:00";
  return `${date}T00:00:00${offset}`;
}

export async function parsePostFile(
  filename: string,
  rawFile?: string,
): Promise<Post> {
  const sourceFile =
    rawFile ?? (await fs.readFile(path.join(POSTS_DIRECTORY, filename), "utf8"));
  const parsed = matter(sourceFile);
  const frontmatter = frontmatterSchema.parse(parsed.data);
  const filenameMatch = filename.match(
    /^(\d{4})-(\d{2})-(\d{2})-(.+)\.md$/,
  );

  if (!filenameMatch) {
    throw new Error(`Invalid post filename "${filename}"`);
  }

  const date = normalizedDate(frontmatter.date);
  const [year, month, day] = date.split("-");
  const slug = filenameMatch[4];
  const route = `/${year}/${month}/${day}/${slug}`;
  const legacyUrl = `${route}.html`;
  const html = await renderMarkdown(parsed.content);
  const excerptParagraphs = frontmatter.excerpt_paragraphs ?? 1;
  const excerptHtml = firstParagraphs(html, excerptParagraphs);

  return {
    filename,
    slug,
    route,
    legacyUrl,
    title: frontmatter.title,
    date,
    author: frontmatter.author ?? SITE.author,
    tags: normalizeTags(frontmatter.tags),
    image: frontmatter.image,
    excerptParagraphs,
    tokens: frontmatter.tokens,
    readingTime: frontmatter.reading_time,
    ongoing: frontmatter.ongoing ?? false,
    archive: frontmatter.archive ?? true,
    unlisted: frontmatter.unlisted ?? false,
    html,
    excerptHtml,
    excerptText: htmlToText(excerptHtml),
    source: parsed.content,
    publishedIso: easternIsoDate(date),
  };
}

const loadPosts = cache(async (): Promise<Post[]> => {
  const filenames = (await fs.readdir(POSTS_DIRECTORY))
    .filter((filename) => filename.endsWith(".md"))
    .sort();
  const posts = await Promise.all(filenames.map((name) => parsePostFile(name)));
  return posts.sort((a, b) => a.date.localeCompare(b.date));
});

export async function getPostsChronological(): Promise<Post[]> {
  return loadPosts();
}

export async function getPostsNewestFirst(): Promise<Post[]> {
  return [...(await loadPosts())].reverse();
}

export function isPostListed(
  post: Pick<PostFrontmatter, "unlisted">,
): boolean {
  return !post.unlisted;
}

export async function getListedPostsChronological(): Promise<Post[]> {
  return (await loadPosts()).filter(isPostListed);
}

export async function getListedPostsNewestFirst(): Promise<Post[]> {
  return [...(await getListedPostsChronological())].reverse();
}

export async function getPostBySegments(
  year: string,
  month: string,
  day: string,
  slug: string,
): Promise<Post | undefined> {
  return (await loadPosts()).find(
    (post) => post.route === `/${year}/${month}/${day}/${slug}`,
  );
}

export function formatHomeDate(date: string): string {
  return new Intl.DateTimeFormat("en-US", {
    timeZone: "UTC",
    month: "long",
    day: "2-digit",
    year: "numeric",
  }).format(new Date(`${date}T00:00:00Z`));
}

export function formatPostDate(date: string): string {
  return new Intl.DateTimeFormat("en-US", {
    timeZone: "UTC",
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(new Date(`${date}T00:00:00Z`));
}

export function formatArchiveDate(date: string): string {
  return new Intl.DateTimeFormat("en-US", {
    timeZone: "UTC",
    month: "long",
    year: "numeric",
  }).format(new Date(`${date}T00:00:00Z`));
}
