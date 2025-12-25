import Link from 'next/link';
import { getAllPosts } from '@/lib/posts';
import { format } from 'date-fns';

export default function HomePage() {
  const posts = getAllPosts();

  return (
    <div className="home">
      {/* Profile Section */}
      <div className="profile-container">
        <img src="/pfp.jpg" alt="Alex Wa" className="profile-image" />
        <div className="profile-content">
          <p>
            Hey! I&apos;m Alex Wa, a 2nd year Math and CS major at Yale and YES Scholar. My research currently spans RL and NLP, and I&apos;m also interested in ML systems and model architecture.
          </p>
          <p>
            Currently, I&apos;m developing RL environments in{' '}
            <a href="https://app.primeintellect.ai/dashboard/environments" target="_blank" rel="noopener noreferrer">
              Prime Intellect
            </a>
            &apos;s RL Residency and researching RL4LLMs and rubrics as rewards with the{' '}
            <a href="https://nlp.cs.yale.edu/" target="_blank" rel="noopener noreferrer">
              Yale NLP lab
            </a>
            . Previously, I&apos;ve done research in geometric algebra (APOLLO Labs), algebraic topology (SUMaC &apos;23), abstract algebra (SUMaC &apos;22), and biostatistics (Emory).
          </p>
          <p>
            In my free time, I{' '}
            <a href="https://www.instagram.com/alex_wa_art/" target="_blank" rel="noopener noreferrer">
              draw
            </a>
            , help design{' '}
            <a href="https://www.theveritassearch.com/" target="_blank" rel="noopener noreferrer">
              The Veritas Search
            </a>
            , play board games, enjoy photography, and learn pen-spinning.
          </p>
        </div>
      </div>

      {/* Posts Section */}
      {posts.length > 0 && (
        <>
          <h2 className="post-list-heading">Posts</h2>
          <div className="post-list">
            {posts.map((post) => (
              <div key={post.slug} className="post-item">
                <div className="post-date">
                  {format(new Date(post.date), 'MMMM dd, yyyy')}
                </div>
                <h3 className="post-title">
                  <Link href={`/blog/${post.slug}/`}>
                    {post.title}
                  </Link>
                </h3>
                {post.excerpt && (
                  <p className="post-summary">{post.excerpt}</p>
                )}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

