import { notFound } from 'next/navigation';
import { MDXRemote } from 'next-mdx-remote/rsc';
import { getAllPosts, getPostBySlug, getAdjacentPosts } from '@/lib/posts';
import { format } from 'date-fns';
import Link from 'next/link';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import ShareLinks from '@/components/ShareLinks';

// Generate static params for all posts
export async function generateStaticParams() {
  const posts = getAllPosts();
  return posts.map((post) => ({
    slug: post.slug,
  }));
}

// Generate metadata for each post
export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const post = getPostBySlug(slug);
  
  if (!post) {
    return {
      title: 'Post Not Found',
    };
  }

  return {
    title: `${post.title} | Alex Wa's Blog`,
    description: post.excerpt || post.title,
    openGraph: {
      title: post.title,
      description: post.excerpt || post.title,
      type: 'article',
      publishedTime: post.date,
      images: post.image ? [{ url: post.image }] : [],
    },
  };
}

// MDX components for custom rendering
const components = {
  // Custom image component with proper paths
  img: (props: React.ImgHTMLAttributes<HTMLImageElement>) => {
    // Handle both /public/... and relative paths
    let src = props.src || '';
    if (src.startsWith('/public/')) {
      src = src; // Keep as-is, Next.js serves from public folder
    }
    return <img {...props} src={src} loading="lazy" />;
  },
  // Custom link component
  a: (props: React.AnchorHTMLAttributes<HTMLAnchorElement>) => {
    const href = props.href || '';
    const isExternal = href.startsWith('http') || href.startsWith('mailto:');
    
    if (isExternal) {
      return <a {...props} target="_blank" rel="noopener noreferrer" />;
    }
    return <a {...props} />;
  },
  // Wrapper for tables to enable horizontal scroll
  table: (props: React.TableHTMLAttributes<HTMLTableElement>) => (
    <div style={{ overflowX: 'auto' }}>
      <table {...props} />
    </div>
  ),
};

export default async function BlogPost({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const post = getPostBySlug(slug);

  if (!post) {
    notFound();
  }

  const { prev, next } = getAdjacentPosts(slug);

  return (
    <article className="post">
      <header className="post-header">
        <h1>{post.title}</h1>
        <p className="post-meta">
          <time dateTime={post.date}>
            {format(new Date(post.date), 'MMMM dd, yyyy')}
          </time>
          {' • '}
          <span>Alex Wa</span>
        </p>
      </header>

      <div className="post-content">
        <MDXRemote
          source={post.content}
          components={components}
          options={{
            mdxOptions: {
              remarkPlugins: [remarkMath, remarkGfm],
              rehypePlugins: [
                [rehypeKatex, { 
                  strict: false,
                  trust: true,
                  macros: {
                    "\\eqref": "\\href{#1}{}",
                  },
                }],
                rehypeHighlight,
              ],
            },
          }}
        />
      </div>

      {/* Post Navigation */}
      <nav className="post-navigation">
        {prev ? (
          <Link href={`/blog/${prev.slug}/`} className="nav-link prev">
            <span className="nav-label">← Previous Post</span>
            <span className="nav-title">{prev.title}</span>
          </Link>
        ) : (
          <Link href="/archive/" className="nav-link prev">
            <span className="nav-label">← Blog Archive</span>
            <span className="nav-title">Archive of all previous blog posts</span>
          </Link>
        )}
        
        {next ? (
          <Link href={`/blog/${next.slug}/`} className="nav-link next">
            <span className="nav-label">Next Post →</span>
            <span className="nav-title">{next.title}</span>
          </Link>
        ) : (
          <Link href="/archive/" className="nav-link next">
            <span className="nav-label">Blog Archive →</span>
            <span className="nav-title">Archive of all previous blog posts</span>
          </Link>
        )}
      </nav>
    </article>
  );
}

