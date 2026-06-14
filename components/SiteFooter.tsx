import { SITE } from "@/lib/site";

type SocialLinkProps = {
  href: string;
  icon: "github" | "linkedin" | "twitter";
  username: string;
};

function SocialLink({ href, icon, username }: SocialLinkProps) {
  return (
    <li>
      <a href={href}>
        <svg className="svg-icon">
          <use href={`/assets/minima-social-icons.svg#${icon}`} />
        </svg>{" "}
        <span className="username">{username}</span>
      </a>
    </li>
  );
}

export function SiteFooter() {
  return (
    <footer className="site-footer h-card">
      <data className="u-url" value="/" />
      <div className="wrapper">
        <h2 className="footer-heading">{SITE.title}</h2>
        <div className="footer-col-wrapper">
          <div className="footer-col footer-col-1">
            <ul className="contact-list">
              <li className="p-name">{SITE.author}</li>
              <li>
                <a className="u-email" href={`mailto:${SITE.email}`}>
                  {SITE.email}
                </a>
              </li>
            </ul>
          </div>
          <div className="footer-col footer-col-2">
            <ul className="social-media-list">
              <SocialLink
                href={`https://github.com/${SITE.github}`}
                icon="github"
                username={SITE.github}
              />
              <SocialLink
                href={`https://www.linkedin.com/in/${SITE.linkedin}`}
                icon="linkedin"
                username={SITE.linkedin}
              />
              <SocialLink
                href={`https://www.twitter.com/${SITE.twitter}`}
                icon="twitter"
                username={SITE.twitter}
              />
            </ul>
          </div>
          <div className="footer-col footer-col-3">
            <p>{SITE.description}</p>
          </div>
        </div>
      </div>
    </footer>
  );
}
