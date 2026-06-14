"use client";

import Script from "next/script";
import { useEffect } from "react";

declare global {
  interface Window {
    MathJax?: {
      typesetPromise?: () => Promise<void>;
    };
  }
}

function typesetMath() {
  void window.MathJax?.typesetPromise?.();
}

export function ClientEnhancements() {
  useEffect(() => {
    let active = true;

    void import("highlight.js").then(({ default: hljs }) => {
      if (!active) {
        return;
      }
      document.querySelectorAll<HTMLElement>("pre code").forEach((block) => {
        if (block.classList.length === 0) {
          block.classList.add("language-python");
        }
        if (!block.dataset.highlighted) {
          hljs.highlightElement(block);
        }
      });
    });

    typesetMath();
    return () => {
      active = false;
    };
  }, []);

  return (
    <Script
      id="MathJax-script"
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"
      strategy="afterInteractive"
      onLoad={typesetMath}
    />
  );
}
