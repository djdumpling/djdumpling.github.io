export const SITE = {
  name: "Alex Wa's Blog",
  title: "Alex Wa's Blog",
  author: "Alex Wa",
  description:
    "Alex Wa's blog; a mix of projects, research and life. Currently interested in RL, NLP, and ML systems.",
  email: "alex [dot] wa [at] yale [dot] edu",
  url: "https://djdumpling.github.io",
  twitter: "_djdumpling",
  github: "djdumpling",
  linkedin: "alex-wa",
  analyticsId: "G-97R68RHKQG",
  timeZone: "America/New_York",
} as const;

export function publicPageHref(route: string, legacyUrl: string): string {
  return process.env.NODE_ENV === "development" ? route : legacyUrl;
}

export function absoluteUrl(pathname: string): string {
  return new URL(pathname, SITE.url).toString();
}
