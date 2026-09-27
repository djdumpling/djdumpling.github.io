import type { Metadata } from "next";
import { Fraunces, Source_Serif_4 } from "next/font/google";

import { ArtBoard } from "@/components/ArtBoard";
import { getArtPieces } from "@/lib/art";

const display = Fraunces({
  subsets: ["latin"],
  style: ["normal", "italic"],
  axes: ["SOFT", "opsz"],
  variable: "--font-art-display",
});

const label = Source_Serif_4({
  subsets: ["latin"],
  style: ["normal", "italic"],
  variable: "--font-art-label",
});

const description =
  "Paintings, drawings, and studies.";

export const metadata: Metadata = {
  title: "Art",
  description,
  alternates: { canonical: "/art.html" },
  openGraph: {
    title: "Art",
    description,
    url: "/art.html",
    type: "website",
  },
  twitter: {
    title: "Art",
    description,
    card: "summary_large_image",
  },
};

export default function ArtPage() {
  const pieces = getArtPieces();

  return (
    <article className={`art-page ${display.variable} ${label.variable}`}>
      <header className="art-hero">
        <p className="art-intro">
          A variety of mediums throughout the past few years, spanning oils, watercolors, charcoal, graphite, and pen. 
          Some are studies of architectural forms, oils from painters I admire, and of anatomy; 
          others come from works completed in watercolor and painting classes at school and art studio before college. 
          Parts of the drawing/painting process have been documented online on {" "}
          <a href="https://www.instagram.com/alex_wa_art/">
            instagram
          </a>
          .
        </p>
      </header>
      <ArtBoard pieces={pieces} />
    </article>
  );
}
