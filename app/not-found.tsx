export default function NotFound() {
  return (
    <article className="post">
      <header className="post-header">
        <h1 className="post-title">Page not found</h1>
      </header>
      <div className="post-content">
        <p>The requested page could not be found.</p>
        <p>
          <a href="/">Return to the blog</a>
        </p>
      </div>
    </article>
  );
}
