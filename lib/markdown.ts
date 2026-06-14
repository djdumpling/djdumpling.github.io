import { decode } from "html-entities";
import rehypeRaw from "rehype-raw";
import rehypeSlug from "rehype-slug";
import rehypeStringify from "rehype-stringify";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkParse from "remark-parse";
import remarkRehype from "remark-rehype";
import remarkSmartypants from "remark-smartypants";
import { unified } from "unified";
import { visit } from "unist-util-visit";

type MathPlaceholder = {
  token: string;
  value: string;
};

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function protectRawMath(markdown: string): {
  markdown: string;
  placeholders: MathPlaceholder[];
} {
  const placeholders: MathPlaceholder[] = [];
  let index = 0;

  const protect = (value: string, block: boolean): string => {
    const id = index++;
    const marker = `NEXT_MATH_${block ? "BLOCK" : "INLINE"}_${id}`;
    const token = block ? `<!--${marker}-->` : marker;
    placeholders.push({ token, value: escapeHtml(value) });
    return token;
  };

  // Protect `$$...$$` as display math before remark-math runs. remark-math
  // classifies single-line `$$...$$` as inline math and re-serializes it with
  // single `$`, which MathJax then renders inline (left-aligned) instead of as
  // a centered display block. Handling it here keeps display math centered
  // regardless of whether the source spans one line or several.
  const withDollarBlocks = markdown.replace(/\$\$[\s\S]*?\$\$/g, (value) =>
    protect(value, true),
  );
  const withBlocks = withDollarBlocks.replace(/\\\[[\s\S]*?\\\]/g, (value) =>
    protect(value, true),
  );
  const withInline = withBlocks.replace(/\\\([\s\S]*?\\\)/g, (value) =>
    protect(value, false),
  );

  return { markdown: withInline, placeholders };
}

function remarkPreserveMath() {
  return (tree: unknown) => {
    const mdast = tree as Parameters<typeof visit>[0];

    visit(mdast, "inlineMath", (node) => {
      const mathNode = node as {
        type: string;
        value: string;
        data?: unknown;
      };
      mathNode.type = "text";
      mathNode.value = `$${mathNode.value}$`;
      delete mathNode.data;
    });
    visit(mdast, "math", (node) => {
      const mathNode = node as {
        type: string;
        value: string;
        data?: unknown;
      };
      mathNode.type = "html";
      mathNode.value = `$$\n${mathNode.value}\n$$`;
      delete mathNode.data;
    });
  };
}

function remarkKramdownCodeClasses() {
  return (tree: unknown) => {
    const mdast = tree as Parameters<typeof visit>[0];
    visit(mdast, "inlineCode", (node) => {
      const codeNode = node as {
        data?: { hProperties?: { className?: string[] } };
      };
      codeNode.data ??= {};
      codeNode.data.hProperties ??= {};
      codeNode.data.hProperties.className = [
        "language-plaintext",
        "highlighter-rouge",
      ];
    });
    visit(mdast, "code", (node) => {
      const codeNode = node as {
        lang?: string | null;
        data?: { hProperties?: { className?: string[] } };
      };
      codeNode.data ??= {};
      codeNode.data.hProperties ??= {};
      const language = codeNode.lang?.toLowerCase() || "python";
      codeNode.lang = language;
      codeNode.data.hProperties.className = [`language-${language}`];
    });
  };
}

export async function renderMarkdown(markdown: string): Promise<string> {
  const protectedMath = protectRawMath(markdown);

  const result = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath, { singleDollarTextMath: true })
    .use(remarkSmartypants)
    .use(remarkPreserveMath)
    .use(remarkKramdownCodeClasses)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeRaw)
    .use(rehypeSlug)
    .use(rehypeStringify, { allowDangerousHtml: true })
    .process(protectedMath.markdown);

  let html = String(result);
  // Replace highest ids first: inline tokens have no terminator, so
  // "NEXT_MATH_INLINE_1" is a substring of "NEXT_MATH_INLINE_10". Descending
  // order guarantees the longer token is consumed before its prefix.
  // Use a function replacement so `$$`/`$&` in math are inserted literally
  // rather than interpreted as replaceAll special patterns.
  for (const placeholder of [...protectedMath.placeholders].reverse()) {
    html = html.replaceAll(placeholder.token, () => placeholder.value);
  }

  return html;
}

export function firstParagraph(html: string): string {
  return html.match(/<p>[\s\S]*?<\/p>/)?.[0] ?? "";
}

export function htmlToText(html: string): string {
  return decode(
    html
      .replace(/<[^>]*>/g, "")
      .replace(/\s+/g, " ")
      .trim(),
  );
}
