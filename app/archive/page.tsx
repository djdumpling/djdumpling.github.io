import Link from 'next/link';
import { getPostsByTag } from '@/lib/posts';
import { format } from 'date-fns';

export const metadata = {
  title: 'Blog Archive | Alex Wa',
  description: 'Archive of all blog posts',
};

export default function ArchivePage() {
  const postsByTag = getPostsByTag();
  const tags = Object.keys(postsByTag).sort();

  return (
    <div className="archive">
      <h1>Blog Archive</h1>
      
      {tags.map((tag) => (
        <div key={tag} className="archive-section">
          <h3>{tag}</h3>
          <ul>
            {postsByTag[tag].map((post) => (
              <li key={post.slug}>
                <Link href={`/blog/${post.slug}/`}>
                  {format(new Date(post.date), 'MMMM yyyy')} - {post.title}
                </Link>
              </li>
            ))}
          </ul>
        </div>
      ))}

      {tags.length === 0 && (
        <p>No posts yet.</p>
      )}
    </div>
  );
}

