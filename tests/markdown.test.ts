import { describe, expect, it } from "vitest";

import { renderMarkdown } from "@/lib/markdown";

describe("renderMarkdown", () => {
  it("preserves every MathJax delimiter without code wrappers", async () => {
    const html = await renderMarkdown(String.raw`
Inline $x_1 + y$ and \(\alpha + \beta\).

$$
\begin{align}
a &= b \\
c &= d
\end{align}
$$

\[
\sum_i x_i
\]
`);

    expect(html).toContain("$x_1 + y$");
    expect(html).toContain(String.raw`\(\alpha + \beta\)`);
    expect(html).toContain(String.raw`$$
\begin{align}
a &amp;= b \\
c &amp;= d
\end{align}
$$`);
    expect(html).toContain(String.raw`\[
\sum_i x_i
\]`);
    expect(html).not.toContain("language-math");
  });

  it("keeps single-line $$...$$ as centered display math", async () => {
    const html = await renderMarkdown(
      String.raw`$$P(Y \mid X, M_\theta) > \tau$$`,
    );

    // Display delimiters must survive as `$$` (not collapse to inline `$`),
    // otherwise MathJax renders the equation inline and left-aligned.
    expect(html).toContain(String.raw`$$P(Y \mid X, M_\theta) &gt; \tau$$`);
    expect(html).not.toMatch(/[^$]\$P\(Y/);
  });

  it("preserves trusted raw HTML used by posts", async () => {
    const html = await renderMarkdown(`
<details><summary>Show code</summary><pre><code class="language-python">print("ok")</code></pre></details>

<think>private reasoning marker</think>

<div style="position: relative;">
  <iframe src="https://www.youtube.com/embed/example" frameborder="0" allowfullscreen></iframe>
</div>
`);

    expect(html).toContain("<details>");
    expect(html).toContain("<summary>Show code</summary>");
    expect(html).toContain("<think>private reasoning marker</think>");
    expect(html).toContain('style="position: relative;"');
    expect(html).toContain(
      '<iframe src="https://www.youtube.com/embed/example" frameborder="0" allowfullscreen></iframe>',
    );
  });

  it("matches Kramdown duplicate heading suffixes", async () => {
    const html = await renderMarkdown(`
# Hermes 4

# Hermes 4
`);

    expect(html).toContain('<h1 id="hermes-4">');
    expect(html).toContain('<h1 id="hermes-4-1">');
  });

  it("applies legacy code classes and defaults fenced code to Python", async () => {
    const html = await renderMarkdown(`
Use \`value\`.

\`\`\`
print("hello")
\`\`\`
`);

    expect(html).toContain(
      '<code class="language-plaintext highlighter-rouge">value</code>',
    );
    expect(html).toContain('<code class="language-python">');
  });
});
