import type { Post } from "@/lib/posts";
import { publicPageHref } from "@/lib/site";

type NavigationItemProps = {
  direction: "prev" | "next";
  post?: Post;
};

function NavigationItem({ direction, post }: NavigationItemProps) {
  const isPrevious = direction === "prev";
  const href = post
    ? publicPageHref(post.route, post.legacyUrl)
    : publicPageHref("/archive", "/archive.html");
  const title = post?.title ?? "Blog Archive";
  const label = post
    ? isPrevious
      ? "Previous Post"
      : "Next Post"
    : "Blog Archive";
  const text = post?.title ?? "Archive of all previous blog posts";

  return (
    <a
      className={`post_navi-item nav_${direction}`}
      href={href}
      title={title}
    >
      <div className="post_navi-arrow">{isPrevious ? "<" : ">"}</div>
      <div className="post_navi-label">{label}</div>
      <div>
        <span>{text}</span>
      </div>
    </a>
  );
}

export function PostNavigation({
  previous,
  next,
}: {
  previous?: Post;
  next?: Post;
}) {
  return (
    <>
      <hr />
      <div className="post_navi">
        <NavigationItem direction="prev" post={previous} />
        <NavigationItem direction="next" post={next} />
      </div>
    </>
  );
}
