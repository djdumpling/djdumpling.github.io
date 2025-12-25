import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';

const postsDirectory = path.join(process.cwd(), 'content/posts');

export interface PostMeta {
  slug: string;
  title: string;
  date: string;
  excerpt?: string;
  image?: string;
  tags?: string[];
}

export interface Post extends PostMeta {
  content: string;
}

export function getAllPosts(): PostMeta[] {
  if (!fs.existsSync(postsDirectory)) {
    return [];
  }

  const fileNames = fs.readdirSync(postsDirectory);
  const allPostsData = fileNames
    .filter((fileName) => fileName.endsWith('.mdx'))
    .map((fileName) => {
      const slug = fileName.replace(/\.mdx$/, '');
      const fullPath = path.join(postsDirectory, fileName);
      const fileContents = fs.readFileSync(fullPath, 'utf8');
      const { data, content } = matter(fileContents);

      // Extract first paragraph as excerpt if not provided
      let excerpt = data.excerpt;
      if (!excerpt) {
        // Get the first paragraph, strip markdown formatting
        const firstPara = content.split('\n\n')[0] || '';
        // Remove markdown links, images, bold, italic, etc
        const cleanText = firstPara
          .replace(/!\[.*?\]\(.*?\)/g, '') // images
          .replace(/\[([^\]]+)\]\([^\)]+\)/g, '$1') // links
          .replace(/[*_]{1,2}([^*_]+)[*_]{1,2}/g, '$1') // bold/italic
          .replace(/`([^`]+)`/g, '$1') // inline code
          .replace(/^#+\s+/gm, '') // headers
          .trim();
        
        // Don't add ... if it's the full paragraph
        excerpt = cleanText.length > 300 ? cleanText.slice(0, 300) + '...' : cleanText;
      }

      return {
        slug,
        title: data.title || slug,
        date: data.date ? new Date(data.date).toISOString() : new Date().toISOString(),
        excerpt,
        image: data.image,
        tags: data.tags || ['Other'],
      };
    });

  // Sort posts by date (newest first)
  return allPostsData.sort((a, b) => (a.date < b.date ? 1 : -1));
}

export function getPostBySlug(slug: string): Post | null {
  const fullPath = path.join(postsDirectory, `${slug}.mdx`);
  
  if (!fs.existsSync(fullPath)) {
    return null;
  }

  const fileContents = fs.readFileSync(fullPath, 'utf8');
  const { data, content } = matter(fileContents);

  return {
    slug,
    title: data.title || slug,
    date: data.date ? new Date(data.date).toISOString() : new Date().toISOString(),
    excerpt: data.excerpt,
    image: data.image,
    tags: data.tags || ['Other'],
    content,
  };
}

export function getAdjacentPosts(slug: string): { prev: PostMeta | null; next: PostMeta | null } {
  const posts = getAllPosts();
  const currentIndex = posts.findIndex((post) => post.slug === slug);

  return {
    prev: currentIndex < posts.length - 1 ? posts[currentIndex + 1] : null,
    next: currentIndex > 0 ? posts[currentIndex - 1] : null,
  };
}

export function getPostsByTag(): Record<string, PostMeta[]> {
  const posts = getAllPosts();
  const postsByTag: Record<string, PostMeta[]> = {};

  posts.forEach((post) => {
    const tags = post.tags || ['Other'];
    tags.forEach((tag) => {
      if (!postsByTag[tag]) {
        postsByTag[tag] = [];
      }
      postsByTag[tag].push(post);
    });
  });

  return postsByTag;
}

