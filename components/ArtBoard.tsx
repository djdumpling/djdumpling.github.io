"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  ART_SECTIONS,
  compareRainbow,
  sortBySection,
  type ArtPiece,
} from "@/lib/art";

// Space reserved for each card's wall label, as a fraction of column width.
const LABEL_ALLOWANCE = 0.22;

type SortMode = "date" | "medium";

// Dates are YYYY-MM-DD, or YYYY-MM when only the month is known.
function formatDate(date: string, withDay: boolean): string {
  const monthOnly = date.length === 7;
  return new Date(`${monthOnly ? `${date}-01` : date}T12:00:00Z`).toLocaleDateString("en-US", {
    timeZone: "UTC",
    year: "numeric",
    month: "long",
    day: withDay && !monthOnly ? "numeric" : undefined,
  });
}

// Deterministic, gentle tilt so the board feels pinned rather than printed.
function tilt(seed: string): string {
  let hash = 0;
  for (const char of seed) {
    hash = (hash * 31 + char.charCodeAt(0)) | 0;
  }
  return `${((Math.abs(hash) % 7) - 3) * 0.3}deg`;
}

function describe(piece: ArtPiece): string {
  return piece.medium ? `${piece.title}, ${piece.medium}` : piece.title;
}

function useColumnCount(): number {
  const [columns, setColumns] = useState(3);

  useEffect(() => {
    const query = window.matchMedia("(max-width: 700px)");
    const update = () => setColumns(query.matches ? 2 : 3);
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);

  return columns;
}

// Places each piece in the currently shortest column, which keeps the
// order reading left to right while balancing column heights.
function layoutColumns(pieces: ArtPiece[], count: number): ArtPiece[][] {
  const columns: ArtPiece[][] = Array.from({ length: count }, () => []);
  const heights = new Array<number>(count).fill(0);

  for (const piece of pieces) {
    const shortest = heights.indexOf(Math.min(...heights));
    columns[shortest].push(piece);
    heights[shortest] += piece.height / piece.width + LABEL_ALLOWANCE;
  }

  return columns;
}

function Masonry({
  pieces,
  columnCount,
  onOpen,
}: {
  pieces: ArtPiece[];
  columnCount: number;
  onOpen: (piece: ArtPiece) => void;
}) {
  const columns = useMemo(
    () => layoutColumns(pieces, columnCount),
    [pieces, columnCount],
  );

  return (
    <div
      className="art-board"
      style={{ gridTemplateColumns: `repeat(${columnCount}, minmax(0, 1fr))` }}
    >
      {columns.map((column, columnIndex) => (
        <div className="art-column" key={columnIndex}>
          {column.map((piece) => (
            <figure
              className="art-card"
              key={piece.id}
              style={
                {
                  "--tilt": tilt(piece.id),
                  "--swatch": piece.color,
                } as React.CSSProperties
              }
            >
              <button
                type="button"
                className="art-mat"
                onClick={() => onOpen(piece)}
                aria-label={`View ${piece.title} larger`}
              >
                <img
                  src={piece.thumb}
                  alt={describe(piece)}
                  width={piece.width}
                  height={piece.height}
                  loading="lazy"
                  decoding="async"
                />
              </button>
              <figcaption className="art-label">
                <span className="art-title">{piece.title}</span>
                <span className="art-meta">
                  {[piece.medium, formatDate(piece.date, false)]
                    .filter(Boolean)
                    .join(" · ")}
                </span>
              </figcaption>
            </figure>
          ))}
        </div>
      ))}
    </div>
  );
}

export function ArtBoard({ pieces }: { pieces: ArtPiece[] }) {
  const columnCount = useColumnCount();
  const [sort, setSort] = useState<SortMode>("date");
  const ordered = useMemo(
    () => (sort === "date" ? pieces : sortBySection(pieces)),
    [pieces, sort],
  );
  const palette = useMemo(
    () => [...pieces].sort((a, b) => compareRainbow(a.color, b.color)),
    [pieces],
  );
  const dialogRef = useRef<HTMLDialogElement>(null);
  const touchStart = useRef<number | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const active = activeId === null ? -1 : ordered.findIndex((p) => p.id === activeId);
  const piece = active === -1 ? null : ordered[active];

  const open = useCallback((target: ArtPiece) => {
    setActiveId(target.id);
    dialogRef.current?.showModal();
  }, []);

  const step = useCallback(
    (delta: number) => {
      if (active === -1) return;
      setActiveId(ordered[(active + delta + ordered.length) % ordered.length].id);
    },
    [active, ordered],
  );

  // Warm the cache for the neighbours so paging through feels instant.
  useEffect(() => {
    if (active === -1) {
      return;
    }
    for (const delta of [-1, 1]) {
      new Image().src = ordered[(active + delta + ordered.length) % ordered.length].src;
    }
  }, [active, ordered]);

  return (
    <>
      <ol
        className="art-palette"
        aria-label="Every piece by its average color"
        style={
          {
            "--count": palette.length,
            "--half": Math.ceil(palette.length / 2),
          } as React.CSSProperties
        }
      >
        {palette.map((item) => (
          <li key={item.id}>
            <button
              type="button"
              style={{ background: item.color }}
              title={item.title}
              aria-label={`${item.title}: open`}
              onClick={() => open(item)}
            />
          </li>
        ))}
      </ol>

      <div className="art-controls" role="group" aria-label="Sort pieces">
        <span className="art-controls-label">Sort by</span>
        {(["date", "medium"] as const).map((mode) => (
          <button
            type="button"
            key={mode}
            aria-pressed={sort === mode}
            onClick={() => setSort(mode)}
          >
            {mode === "date" ? "Date" : "Medium"}
          </button>
        ))}
      </div>

      {sort === "date" ? (
        <Masonry pieces={ordered} columnCount={columnCount} onOpen={open} />
      ) : (
        ART_SECTIONS.map(({ key, label }) => {
          const sectionPieces = ordered.filter((item) => item.section === key);
          return sectionPieces.length ? (
            <section className="art-section" key={key} aria-labelledby={`art-${key}`}>
              <h2 className="art-section-heading" id={`art-${key}`}>
                {label}
                <span className="art-section-count">{sectionPieces.length}</span>
              </h2>
              <Masonry pieces={sectionPieces} columnCount={columnCount} onOpen={open} />
            </section>
          ) : null;
        })
      )}

      <dialog
        ref={dialogRef}
        className="art-lightbox"
        aria-label={piece ? piece.title : "Artwork"}
        onClose={() => setActiveId(null)}
        onClick={(event) => {
          if (event.target === event.currentTarget) {
            dialogRef.current?.close();
          }
        }}
        onKeyDown={(event) => {
          if (event.key === "ArrowRight") step(1);
          if (event.key === "ArrowLeft") step(-1);
        }}
        onTouchStart={(event) => {
          touchStart.current = event.touches[0].clientX;
        }}
        onTouchEnd={(event) => {
          if (touchStart.current === null) return;
          const distance = event.changedTouches[0].clientX - touchStart.current;
          touchStart.current = null;
          if (Math.abs(distance) > 50) step(distance < 0 ? 1 : -1);
        }}
      >
        {piece ? (
          <div className="art-lightbox-inner">
            <div className="art-lightbox-stage">
              <img
                key={piece.id}
                src={piece.src}
                alt={describe(piece)}
                width={piece.width}
                height={piece.height}
                style={{
                  backgroundColor: piece.color,
                  backgroundImage: `url(${piece.thumb})`,
                }}
              />
            </div>
            <aside className="art-lightbox-label">
              <p className="art-lightbox-count">
                {active + 1} / {ordered.length}
              </p>
              <h2 className="art-title">{piece.title}</h2>
              <p className="art-meta">
                {[piece.medium, formatDate(piece.date, true)]
                  .filter(Boolean)
                  .join(" · ")}
              </p>
              <div className="art-lightbox-nav">
                <button type="button" onClick={() => step(-1)} aria-label="Previous piece">
                  ←
                </button>
                <button type="button" onClick={() => step(1)} aria-label="Next piece">
                  →
                </button>
              </div>
            </aside>
            <button
              type="button"
              className="art-lightbox-close"
              onClick={() => dialogRef.current?.close()}
              aria-label="Close"
            >
              ×
            </button>
          </div>
        ) : null}
      </dialog>
    </>
  );
}
