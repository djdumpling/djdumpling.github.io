#!/usr/bin/env python3
"""Generate the deterministic SVG figure set used by docs/actual_blog.md.

The script uses only the Python standard library. Quantitative inputs live in
data/*.csv; their run-level lineage is documented in PROVENANCE.md.

Run from any directory:
    python3 docs/images/actual_blog/make_blog_figures.py
"""

from __future__ import annotations

import csv
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

BG = "#181818"
CARD = "#242424"
CARD_2 = "#2f2f2f"
GRID = "#464646"
WHITE = "#ffffff"
TEXT = "#e8e8e8"
MUTED = "#a3a3a3"
DIM = "#737373"
GREEN = "#7fee64"
BLUE = "#91c8ef"
ORANGE = "#ffab5e"
RED = "#f87171"
YELLOW = "#ffea71"
PURPLE = "#c4a7e7"


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(value: str) -> float:
    return float(value)


def txt(x: float, y: float, value: str, *, size: int = 24, fill: str = TEXT,
        weight: int = 400, anchor: str = "start", family: str = "Inter, Arial, sans-serif",
        opacity: float = 1.0) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-family="{family}" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'opacity="{opacity}">{escape(str(value))}</text>'
    )


def lines(x: float, y: float, values: list[str], *, size: int = 22,
          fill: str = TEXT, weight: int = 400, leading: float = 1.3,
          anchor: str = "start") -> str:
    out = [
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" '
        f'font-family="Inter, Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}">'
    ]
    for i, value in enumerate(values):
        dy = 0 if i == 0 else size * leading
        out.append(f'<tspan x="{x:.1f}" dy="{dy:.1f}">{escape(value)}</tspan>')
    out.append("</text>")
    return "".join(out)


def rect(x: float, y: float, w: float, h: float, *, fill: str = CARD,
         stroke: str = "none", radius: float = 12, sw: float = 1.5,
         opacity: float = 1.0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{radius:.1f}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{sw:.1f}" opacity="{opacity}"/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, *, stroke: str = GRID,
         sw: float = 2, dash: str | None = None, marker: bool = False,
         opacity: float = 1.0) -> str:
    attrs = f' stroke-dasharray="{dash}"' if dash else ""
    end = ' marker-end="url(#arrow)"' if marker else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{sw:.1f}" opacity="{opacity}"{attrs}{end}/>'
    )


def polyline(points: list[tuple[float, float]], *, stroke: str = BLUE,
             sw: float = 2, dash: str | None = None,
             opacity: float = 1.0) -> str:
    attrs = f' stroke-dasharray="{dash}"' if dash else ""
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{coords}" fill="none" stroke="{stroke}" '
        f'stroke-width="{sw:.1f}" stroke-linejoin="round" '
        f'stroke-linecap="round" opacity="{opacity}"{attrs}/>'
    )


def circle(cx: float, cy: float, r: float, *, fill: str, stroke: str = "none",
           sw: float = 1.5, opacity: float = 1.0) -> str:
    return (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw:.1f}" opacity="{opacity}"/>')


def base_svg(width: int, height: int, title: str, desc: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(desc)}</desc>',
        "<defs>",
        f'<marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" '
        f'orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="{MUTED}"/></marker>',
        "</defs>",
        f'<rect width="{width}" height="{height}" fill="{BG}"/>'
    ]


def heading(parts: list[str], title: str, subtitle: str, width: int) -> None:
    parts.append(txt(58, 64, title, size=38, fill=WHITE, weight=600))
    parts.append(txt(width - 58, 64, subtitle, size=18, fill=MUTED, anchor="end"))


def write(name: str, parts: list[str]) -> None:
    parts.append("</svg>\n")
    (HERE / name).write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {HERE / name}")


def game_loop() -> None:
    width, height = 1400, 760
    p = base_svg(width, height, "How a MegaGem round works",
                 "Three players make sealed bids; the winner takes the lot, reveals a private gem, and changes the public values used to score all collections.")
    heading(p, "One auction changes the whole table", "schematic · three-player game", width)

    # Private hands.
    p.append(rect(55, 120, 310, 385, stroke=GRID))
    p.append(txt(85, 158, "PRIVATE INFORMATION", size=16, fill=GREEN, weight=600))
    gem_colors = [RED, BLUE, GREEN, PURPLE, YELLOW]
    for seat in range(3):
        y = 205 + seat * 95
        p.append(txt(85, y, f"Player {seat + 1}", size=23, weight=600))
        p.append(txt(85, y + 29, "35 coins", size=17, fill=MUTED))
        for j, color in enumerate(gem_colors):
            p.append(circle(215 + j * 27, y - 8, 10, fill=color, stroke=BG))
    p.append(txt(85, 478, "5 secret gems each", size=18, fill=MUTED))

    # Auction and bid.
    p.append(line(380, 305, 455, 305, stroke=MUTED, sw=3, marker=True))
    p.append(rect(470, 160, 270, 290, fill=CARD_2, stroke=ORANGE))
    p.append(txt(605, 205, "AUCTION", size=18, fill=ORANGE, weight=600, anchor="middle"))
    p.append(lines(605, 254, ["Treasure", "Loan", "Investment"], size=27,
                   anchor="middle", weight=500, leading=1.55))
    p.append(txt(605, 410, "sealed, simultaneous bids", size=17, fill=MUTED, anchor="middle"))

    # Winner and reveal.
    p.append(line(755, 305, 825, 305, stroke=MUTED, sw=3, marker=True))
    p.append(rect(840, 160, 220, 290, stroke=GREEN))
    p.append(txt(950, 205, "WINNER", size=18, fill=GREEN, weight=600, anchor="middle"))
    p.append(txt(950, 260, "takes lot", size=28, anchor="middle", weight=600))
    p.append(txt(950, 315, "+", size=32, fill=MUTED, anchor="middle"))
    p.append(txt(950, 365, "reveals 1 gem", size=25, anchor="middle"))
    p.append(circle(950, 408, 18, fill=RED, stroke=WHITE, sw=2))

    # Public display and repricing.
    p.append(line(1075, 305, 1135, 305, stroke=MUTED, sw=3, marker=True))
    p.append(rect(1150, 120, 195, 385, stroke=BLUE))
    p.append(txt(1247, 158, "PUBLIC", size=16, fill=BLUE, weight=600, anchor="middle"))
    p.append(lines(1247, 208, ["Value", "Display"], size=31, anchor="middle", weight=600))
    for row in range(4):
        for col in range(3):
            p.append(circle(1205 + col * 42, 305 + row * 38, 12,
                            fill=gem_colors[(row + col) % len(gem_colors)]))
    p.append(txt(1247, 476, "reprices every collection", size=16, fill=MUTED, anchor="middle"))

    # Long-horizon feedback and scoring.
    p.append(line(1245, 528, 1245, 575, stroke=MUTED, sw=2, marker=True))
    p.append(rect(120, 590, 1160, 112, fill=CARD_2, stroke=GRID))
    p.append(txt(160, 632, "REPEAT", size=18, fill=ORANGE, weight=600))
    p.append(txt(160, 670, "Future bids react to values, price history, and remaining liquidity.", size=23))
    p.append(txt(1245, 632, "FINAL SCORE", size=18, fill=GREEN, weight=600, anchor="end"))
    p.append(txt(1245, 670, "coins + gems + missions − loans + investments", size=20, anchor="end"))
    write("01_game_loop.svg", p)


def research_pipeline() -> None:
    width, height = 1500, 700
    p = base_svg(width, height, "MegaGem research and training pipeline",
                 "The project moves from a benchmark to SFT, a flat GRPO phase, diagnosis, an analytic distributional best response, and distillation into the final model.")
    heading(p, "The six-stage research loop", "the failed stage is the useful one", width)
    stages = [
        ("01", "Benchmark", ["13 models", "624 games", "paired evals"], BLUE),
        ("02", "SFT", ["6,153 examples", "2.8% → 64.7%", "legal + concise"], BLUE),
        ("03", "Self-play GRPO", ["7 mainline runs", "optimizer active", "anchors flat"], RED),
        ("04", "Diagnose", ["opponent-specific Q", "noisy critic labels", "point-price floor"], ORANGE),
        ("05", "Analytic expert", ["bid distribution", "pacing + calibration", "+12.16 live"], GREEN),
        ("06", "Distill + test", ["weights only", "1275 PL-Elo", "top tier"], GREEN),
    ]
    left, top, gap, box_w, box_h = 55, 150, 24, 215, 365
    for i, (num, title, body, color) in enumerate(stages):
        x = left + i * (box_w + gap)
        p.append(rect(x, top, box_w, box_h, fill=CARD, stroke=color, sw=2))
        p.append(txt(x + 24, top + 42, num, size=18, fill=color, weight=600))
        p.append(txt(x + 24, top + 92, title, size=25, fill=WHITE, weight=600))
        p.append(line(x + 24, top + 116, x + box_w - 24, top + 116, stroke=GRID, sw=1))
        p.append(lines(x + 24, top + 163, body, size=19, fill=MUTED, leading=1.65))
        if i == 2:
            p.append(rect(x + 20, top + box_h - 69, box_w - 40, 42, fill="#3a2020", stroke=RED, radius=8))
            p.append(txt(x + box_w / 2, top + box_h - 40, "NO TRANSFER", size=15,
                         fill=RED, weight=700, anchor="middle"))
        elif i >= 4:
            p.append(rect(x + 20, top + box_h - 69, box_w - 40, 42, fill="#20351f", stroke=GREEN, radius=8))
            p.append(txt(x + box_w / 2, top + box_h - 40, "CONVERTS LIVE", size=15,
                         fill=GREEN, weight=700, anchor="middle"))
        if i < len(stages) - 1:
            p.append(line(x + box_w + 3, top + box_h / 2, x + box_w + gap - 5,
                          top + box_h / 2, stroke=MUTED, sw=2, marker=True))
    p.append(rect(260, 565, 980, 78, fill=CARD_2, stroke=GRID))
    p.append(txt(750, 600, "policy-game logs  →  fitted price law  →  analytic best response  →  supervised distillation",
                 size=23, anchor="middle", weight=500))
    p.append(txt(750, 630, "The decision-level data survived. The policy-gradient objective did not.",
                 size=18, fill=MUTED, anchor="middle"))
    write("02_research_pipeline.svg", p)


def selfplay_diagnosis() -> None:
    training = [d for d in rows("selfplay_training.csv") if d["mainline"] == "1"]
    fixed = rows("selfplay_fixed_eval.csv")
    anchors = rows("selfplay_anchor_series.csv")
    width, height = 1600, 1110
    p = base_svg(
        width,
        height,
        "The optimizer moves while frozen-anchor strength stays flat",
        "Token entropy remains active across seven mainline GRPO runs. Several checkpoints gain own terminal score relative to SFT against a scripted heuristic, including a health-failed long run, while the two mainline runs with direct frozen-SFT anchor measurements show no positive trend. A later weak-KL ablation is also flat.",
    )
    heading(
        p,
        "The optimizer moves; measured transfer does not",
        "seven reconstructed mainline runs · canonical Modal sidecars",
        width,
    )
    panels = [(50, 120, 735, 405), (815, 120, 735, 405), (50, 555, 1500, 485)]
    for x, y, w, h in panels:
        p.append(rect(x, y, w, h, stroke=GRID))

    colors = {
        "repl_03": BLUE,
        "repl_04": GREEN,
        "repl_07": RED,
        "rotate_m3_confirm": ORANGE,
        "apimix_x2_01": PURPLE,
        "flash_repl_05": YELLOW,
        "psro_hetero_ev": "#4dd0e1",
        "leash_kl001": MUTED,
    }

    # Panel A: optimizer activity. Entropy is telemetry, not a skill metric.
    x, y, w, h = panels[0]
    p.append(txt(x + 28, y + 43, "Token entropy stays active", size=23, weight=600))
    p.append(txt(x + 28, y + 70,
                 "token entropy vs normalized optimizer progress · not a strength metric",
                 size=17, fill=MUTED))
    px0, px1, py0, py1 = x + 70, x + w - 30, y + 112, y + 310
    for tick in (0.60, 0.70, 0.80):
        yy = py1 - (tick - 0.56) / 0.30 * (py1 - py0)
        p.append(line(px0, yy, px1, yy, stroke=GRID, sw=1, dash="4 6"))
        p.append(txt(px0 - 12, yy + 5, f"{tick:.2f}", size=17, fill=DIM, anchor="end"))
    for tick in (0, 50, 100):
        xx = px0 + tick / 100 * (px1 - px0)
        p.append(txt(xx, py1 + 25, f"{tick}%", size=17, fill=DIM, anchor="middle"))
    by_run: dict[str, list[dict[str, str]]] = {}
    for datum in training:
        by_run.setdefault(datum["run"], []).append(datum)
    for run, values in by_run.items():
        raw = [(f(d["progress"]), f(d["entropy"])) for d in values if d["entropy"]]
        smooth: list[tuple[float, float]] = []
        for i in range(0, len(raw), 3):
            lo, hi = max(0, i - 5), min(len(raw), i + 6)
            smooth.append((raw[i][0], sum(v for _, v in raw[lo:hi]) / (hi - lo)))
        coords = [
            (
                px0 + progress * (px1 - px0),
                py1 - (value - 0.56) / 0.30 * (py1 - py0),
            )
            for progress, value in smooth
        ]
        p.append(
            polyline(
                coords,
                stroke=colors[run],
                sw=3 if run == "repl_07" else 2,
                dash="8 5" if run == "repl_07" else None,
                opacity=0.95,
            )
        )
    legend = [
        ("repl03", "repl_03"),
        ("repl04", "repl_04"),
        ("long DAPO · KL fail", "repl_07"),
        ("seat rotate", "rotate_m3_confirm"),
        ("API mix", "apimix_x2_01"),
        ("Flash mix", "flash_repl_05"),
        ("heterogeneous", "psro_hetero_ev"),
    ]
    for i, (label, run) in enumerate(legend):
        lx = x + 34 + (i % 3) * 220
        ly = y + 344 + (i // 3) * 24
        p.append(line(lx, ly, lx + 25, ly, stroke=colors[run], sw=3,
                      dash="8 5" if run == "repl_07" else None))
        p.append(txt(lx + 34, ly + 5, label, size=17, fill=MUTED))

    # Panel B: same comparator, but only a scripted heuristic.
    x, y, w, h = panels[1]
    p.append(txt(x + 28, y + 43, "A fixed heuristic is exploitable", size=23, weight=600))
    p.append(txt(x + 28, y + 70,
                 "RL checkpoint − SFT own terminal-score points/game · paired 95% CI",
                 size=17, fill=MUTED))
    fx0, fx1 = x + 220, x + w - 38
    flo, fhi = -5.0, 20.0
    for tick in (-5, 0, 5, 10, 15, 20):
        xx = fx0 + (tick - flo) / (fhi - flo) * (fx1 - fx0)
        p.append(line(xx, y + 102, xx, y + 355, stroke=GRID if tick else MUTED,
                      sw=1.5 if tick == 0 else 1, dash=None if tick == 0 else "4 6"))
        p.append(txt(xx, y + 382, f"{tick:+d}", size=17, fill=DIM, anchor="middle"))
    for i, datum in enumerate(fixed):
        yy = y + 120 + i * 34
        estimate = f(datum["mean_own_score_delta"])
        ci_low, ci_high = f(datum["ci_low"]), f(datum["ci_high"])
        x_est = fx0 + (estimate - flo) / (fhi - flo) * (fx1 - fx0)
        x_lo = fx0 + (ci_low - flo) / (fhi - flo) * (fx1 - fx0)
        x_hi = fx0 + (ci_high - flo) / (fhi - flo) * (fx1 - fx0)
        is_health_fail = datum["health_pass"] == "0"
        is_small = datum["run"] == "rotate_m3_confirm"
        color = RED if is_health_fail else GREEN if datum["heuristic_gate_pass"] == "1" else ORANGE
        label = datum["label"] + (" · n=24×1" if is_small else "")
        p.append(txt(fx0 - 16, yy + 5, label, size=17, fill=TEXT, anchor="end"))
        p.append(line(x_lo, yy, x_hi, yy, stroke=color, sw=3,
                      dash="5 4" if is_small else None))
        p.append(circle(x_est, yy, 7, fill=BG if is_small else color, stroke=color, sw=3))
        p.append(txt(min(fx1 - 2, x_hi + 9), yy + 5, f"{estimate:+.2f}",
                     size=17, fill=color, weight=600))
    p.append(txt(x + w - 28, y + 394, "long DAPO gains +6.25 → +13.22 → +16.48, then fails KL health",
                 size=17, fill=RED, anchor="end"))

    # Panel C: raw frozen-SFT anchor series, only where directly measured.
    x, y, w, h = panels[2]
    p.append(txt(x + 28, y + 43, "Direct frozen-SFT anchors do not climb", size=23, weight=600))
    p.append(txt(x + 28, y + 70,
                 "frozen-SFT win rate vs optimizer step · point size follows games/checkpoint",
                 size=17, fill=MUTED))
    ax0, ax1, ay0, ay1 = x + 75, x + w - 300, y + 110, y + h - 60
    for tick in (0.0, 0.2, 1 / 3, 0.4, 0.6):
        yy = ay1 - tick / 0.65 * (ay1 - ay0)
        p.append(line(ax0, yy, ax1, yy, stroke=GREEN if abs(tick - 1 / 3) < 0.01 else GRID,
                      sw=2 if abs(tick - 1 / 3) < 0.01 else 1,
                      dash="7 5" if abs(tick - 1 / 3) < 0.01 else "4 6"))
        label = "chance 1/3" if abs(tick - 1 / 3) < 0.01 else f"{tick:.1f}"
        p.append(txt(ax0 - 12, yy + 5, label, size=17,
                     fill=GREEN if "chance" in label else DIM, anchor="end"))
    for tick in (0, 50, 100, 150):
        xx = ax0 + tick / 150 * (ax1 - ax0)
        p.append(txt(xx, ay1 + 28, str(tick), size=17, fill=DIM, anchor="middle"))
    p.append(txt((ax0 + ax1) / 2, ay1 + 52, "optimizer step", size=17,
                 fill=MUTED, anchor="middle"))
    by_anchor: dict[str, list[dict[str, str]]] = {}
    for datum in anchors:
        by_anchor.setdefault(datum["run"], []).append(datum)
    for run, values in by_anchor.items():
        coords = [
            (
                ax0 + f(d["step"]) / 150 * (ax1 - ax0),
                ay1 - f(d["win_rate"]) / 0.65 * (ay1 - ay0),
            )
            for d in values
        ]
        p.append(polyline(coords, stroke=colors[run], sw=3,
                          dash="8 5" if run == "leash_kl001" else None,
                          opacity=0.9))
        for (xx, yy), datum in zip(coords, values):
            radius = 4 + min(5, (f(datum["n_games"]) / 192) * 4)
            p.append(circle(xx, yy, radius, fill=colors[run], stroke=BG, sw=1.5))
    notes = [
        ("seat rotate · mainline", "256 games · t = −0.57", "rotate_m3_confirm"),
        ("heterogeneous · mainline", "2,016 games · t = −0.11", "psro_hetero_ev"),
        ("weak-KL · later ablation", "2,600 games · t = +0.58", "leash_kl001"),
    ]
    for i, (label, stat, run) in enumerate(notes):
        ny = y + 135 + i * 92
        p.append(rect(x + w - 265, ny, 225, 72, fill=CARD_2, stroke=colors[run], radius=8))
        p.append(txt(x + w - 248, ny + 28, label, size=17, fill=colors[run], weight=600))
        p.append(txt(x + w - 248, ny + 53, stat, size=17, fill=MUTED))
    p.append(txt(x + w - 40, y + h - 70,
                 "Five other mainline runs did not record this anchor; absence is not zero.",
                 size=17, fill=MUTED, anchor="end"))
    write("03_selfplay_diagnosis.svg", p)


def offline_regret() -> None:
    data = rows("offline_regret.csv")
    width, height = 1300, 760
    p = base_svg(width, height, "Offline regret ladder for bid selectors",
                 "With realized terminal gem value supplied to isolate price modeling, distributional expected value reduces regret more than a point-price selector and approaches the best legal bid at realized prices on logged Flash auctions.")
    heading(p, "Distributional prices reduce regret",
            "coins per treasure decision · lower is better", width)
    x0, x1, top, row_h = 410, 1190, 145, 98
    max_v = 4.5
    for tick in range(5):
        xx = x0 + tick / max_v * (x1 - x0)
        p.append(line(xx, top - 15, xx, top + row_h * len(data) - 22, stroke=GRID, sw=1))
        p.append(txt(xx, top - 28, str(tick), size=15, fill=DIM, anchor="middle"))
    color_for = {
        "baseline": RED, "ablation": ORANGE, "point": YELLOW,
        "distributional": GREEN, "oracle": BLUE,
    }
    for i, d in enumerate(data):
        y = top + i * row_h
        value = f(d["regret_coins_per_decision"])
        color = color_for[d["role"]]
        p.append(txt(x0 - 28, y + 42, d["bidder"], size=22, anchor="end",
                     weight=600 if d["role"] == "distributional" else 400,
                     fill=WHITE if d["role"] == "distributional" else TEXT))
        p.append(rect(x0, y + 12, value / max_v * (x1 - x0), 46, fill=color, radius=7))
        p.append(txt(x0 + value / max_v * (x1 - x0) + 16, y + 45, f"{value:.3f}",
                     size=20, fill=color, weight=700))
    p.append(rect(285, 630, 930, 105, fill=CARD_2, stroke=GRID))
    p.append(txt(750, 660,
                 "reduces regret by 1.304 vs SFT · by 0.456 vs point-EV",
                 size=19, fill=GREEN, weight=600, anchor="middle"))
    p.append(txt(750, 690,
                 "offline Flash logs · realized gem value supplied · no mission bonus",
                 size=15, fill=MUTED, anchor="middle"))
    p.append(txt(750, 716,
                 "future state/opponents fixed · 1,821 decisions · 82 seed clusters",
                 size=15, fill=MUTED, anchor="middle"))
    write("04_offline_regret.svg", p)


def liquidity_and_distillation() -> None:
    prog = rows("winning_progression.csv")
    width, height = 1450, 700
    p = base_svg(
        width,
        height,
        "Held-out effects across the analytic and distilled interventions",
        "Five separate 150-pair experiments show that myopic distributional EV is unresolved and negative, pacing turns the result positive, an SFT-fitted price law retains most of the gain, and style-matched distillation carries the gain into model weights.",
    )
    heading(
        p,
        "What survives each handoff",
        "CV-adjusted paired margin/game · separate experiments · n=150 each",
        width,
    )
    p.append(rect(55, 120, 1340, 485, stroke=GRID))
    p.append(txt(95, 165, "intervention", size=17, fill=MUTED, weight=600))
    p.append(txt(1380, 165, "estimate [95% CI]", size=17, fill=MUTED, weight=600, anchor="end"))
    chart_x0, chart_x1, chart_y0 = 520, 1190, 225
    lo, hi = -8.0, 18.0
    for tick in (-5, 0, 5, 10, 15):
        xx = chart_x0 + (tick - lo) / (hi - lo) * (chart_x1 - chart_x0)
        p.append(line(xx, chart_y0 - 40, xx, chart_y0 + 315,
                      stroke=MUTED if tick == 0 else GRID,
                      sw=1.6 if tick == 0 else 1,
                      dash=None if tick == 0 else "4 6"))
        p.append(txt(xx, chart_y0 - 54, f"{tick:+d}", size=14, fill=DIM, anchor="middle"))
    for i, d in enumerate(prog):
        y = chart_y0 + i * 70
        est = f(d["estimate"])
        p.append(txt(95, y + 7, d["method"], size=19, fill=TEXT,
                     weight=600 if i in (1, 3) else 400))
        xx = chart_x0 + (est - lo) / (hi - lo) * (chart_x1 - chart_x0)
        xlo = chart_x0 + (f(d["ci_low"]) - lo) / (hi - lo) * (chart_x1 - chart_x0)
        xhi = chart_x0 + (f(d["ci_high"]) - lo) / (hi - lo) * (chart_x1 - chart_x0)
        p.append(line(xlo, y, xhi, y, stroke=MUTED, sw=5))
        p.append(line(xlo, y - 10, xlo, y + 10, stroke=MUTED, sw=2))
        p.append(line(xhi, y - 10, xhi, y + 10, stroke=MUTED, sw=2))
        color = RED if est < 0 else GREEN
        p.append(circle(xx, y, 9, fill=color, stroke=WHITE, sw=2))
        p.append(txt(1380, y + 7,
                     f'{est:+.2f}  [{f(d["ci_low"]):+.2f}, {f(d["ci_high"]):+.2f}]',
                     size=17, fill=color, weight=700, anchor="end"))
        if i == 2:
            p.append(line(80, y + 35, 1370, y + 35, stroke=GRID, sw=1))
    p.append(rect(95, 618, 1265, 64, fill=CARD_2, stroke=GRID, radius=8))
    p.append(txt(727, 643,
                 "Each row uses its run-specific SFT control; cross-row differences are descriptive, not paired.",
                 size=14, fill=MUTED, anchor="middle"))
    p.append(txt(727, 667,
                 "CV adjustment is model-assisted; raw paired estimates are primary in the text.",
                 size=14, fill=MUTED, anchor="middle"))
    write("05_liquidity_and_distillation.svg", p)


def dynamics_sweep() -> None:
    data = rows("dynamics_sweep.csv")
    width, height = 1500, 820
    p = base_svg(
        width,
        height,
        "Dynamics simulator sweep over liquidity pacing and value calibration",
        "Eighteen simulator arms compare paired margin and deviations per game. The archived live setting sits near the failed stage-1 myopic selector's behavioral footprint. Lambda values at least one are muted as proxy-rescue artifacts. Absolute calibration failed, so live evaluation supplies the effect estimate.",
    )
    heading(
        p,
        "Read the sweep as a screen, not a forecast",
        "approximate simulator · 200 paired seeds/arm · normal 95% intervals",
        width,
    )
    panels = [(50, 120, 820, 600), (900, 120, 550, 600)]
    for x, y, w, h in panels:
        p.append(rect(x, y, w, h, stroke=GRID))
    series_colors = {"0.0": BLUE, "2.0": GREEN, "5.0": ORANGE}

    # Panel A: requested lambda-by-margin view.
    x, y, w, h = panels[0]
    p.append(txt(x + 28, y + 44, "Paired margin/game across pacing settings", size=21, weight=600))
    p.append(txt(x + 28, y + 72, "series = value de-bias δ · gate 1 unless marked", size=16, fill=MUTED))
    px0, px1, py0, py1 = x + 85, x + w - 35, y + 125, y + h - 105
    ylo, yhi = -15.0, 35.0
    for tick in (-10, 0, 10, 20, 30):
        yy = py1 - (tick - ylo) / (yhi - ylo) * (py1 - py0)
        p.append(line(px0, yy, px1, yy, stroke=MUTED if tick == 0 else GRID,
                      sw=1.6 if tick == 0 else 1,
                      dash=None if tick == 0 else "4 6"))
        p.append(txt(px0 - 12, yy + 5, f"{tick:+d}", size=13, fill=DIM, anchor="end"))
    lambda_values = [0.0, 0.5, 1.0, 1.6, 2.5]
    for lam in lambda_values:
        xx = px0 + lam / 2.5 * (px1 - px0)
        p.append(txt(xx, py1 + 29, f"{lam:g}", size=13, fill=DIM, anchor="middle"))
    p.append(txt((px0 + px1) / 2, py1 + 57, "liquidity premium λcoin", size=15,
                 fill=MUTED, anchor="middle"))

    gate_one = [d for d in data if d["gate_min"] == "1.0"]
    for delta in ("0.0", "2.0", "5.0"):
        values = sorted(
            [d for d in gate_one if d["value_debias"] == delta],
            key=lambda d: f(d["lambda_coin"]),
        )
        coords = []
        for datum in values:
            xx = px0 + f(datum["lambda_coin"]) / 2.5 * (px1 - px0)
            yy = py1 - (f(datum["paired_margin"]) - ylo) / (yhi - ylo) * (py1 - py0)
            y_low = py1 - (f(datum["ci_low_normal"]) - ylo) / (yhi - ylo) * (py1 - py0)
            y_high = py1 - (f(datum["ci_high_normal"]) - ylo) / (yhi - ylo) * (py1 - py0)
            p.append(line(xx, y_high, xx, y_low, stroke=series_colors[delta], sw=1.5, opacity=0.65))
            coords.append((xx, yy))
        p.append(polyline(coords, stroke=series_colors[delta], sw=3))
        for (xx, yy), datum in zip(coords, values):
            chosen = datum["chosen"] == "1"
            p.append(circle(xx, yy, 10 if chosen else 6,
                            fill=WHITE if chosen else series_colors[delta],
                            stroke=series_colors[delta], sw=3 if chosen else 1.5))
    artifact_x = px0 + 1.0 / 2.5 * (px1 - px0)
    p.append(rect(artifact_x, py0, px1 - artifact_x, py1 - py0,
                  fill=BG, radius=0, opacity=0.62))
    p.append(txt((artifact_x + px1) / 2, py0 + 25,
                 "λ ≥ 1: proxy-rescue artifact", size=14, fill=RED,
                 weight=600, anchor="middle"))
    for i, delta in enumerate(("0.0", "2.0", "5.0")):
        lx = x + 485 + i * 100
        p.append(line(lx, y + 44, lx + 25, y + 44, stroke=series_colors[delta], sw=3))
        p.append(txt(lx + 33, y + 49, f"δ = {delta[:-2]}", size=14, fill=MUTED))

    gate_only = [d for d in data if d["gate_min"] != "1.0"]
    for i, datum in enumerate(gate_only):
        xx = px0 + 8 + i * 24
        yy = py1 - (f(datum["paired_margin"]) - ylo) / (yhi - ylo) * (py1 - py0)
        p.append(circle(xx, yy, 6, fill=BG, stroke=PURPLE, sw=2))
    p.append(txt(px0 + 94, py1 - 16, "gate-only arms at λ=0", size=13,
                 fill=PURPLE, anchor="middle"))
    chosen = next(d for d in data if d["chosen"] == "1")
    chosen_x = px0 + f(chosen["lambda_coin"]) / 2.5 * (px1 - px0)
    chosen_y = py1 - (f(chosen["paired_margin"]) - ylo) / (yhi - ylo) * (py1 - py0)
    p.append(txt(chosen_x + 15, chosen_y - 14, "archived: λ=.5, δ=2", size=14,
                 fill=GREEN, weight=700))

    # Panel B: reconstructed behavioral-footprint screen.
    x, y, w, h = panels[1]
    p.append(txt(x + 28, y + 44, "Behavioral footprint", size=23, weight=600))
    p.append(txt(x + 28, y + 72, "simulated margin vs interventions/game", size=16, fill=MUTED))
    sx0, sx1, sy0, sy1 = x + 70, x + w - 35, y + 125, y + h - 105
    xlo, xhi = 1.5, 11.0
    myopic_dev = f(data[0]["stage1_myopic_deviations_per_game"])
    band_lo, band_hi = myopic_dev - 0.75, myopic_dev + 0.75
    bx0 = sx0 + (band_lo - xlo) / (xhi - xlo) * (sx1 - sx0)
    bx1 = sx0 + (band_hi - xlo) / (xhi - xlo) * (sx1 - sx0)
    p.append(rect(bx0, sy0, bx1 - bx0, sy1 - sy0, fill="#20351f", radius=0, opacity=0.5))
    myopic_x = sx0 + (myopic_dev - xlo) / (xhi - xlo) * (sx1 - sx0)
    p.append(line(myopic_x, sy0, myopic_x, sy1, stroke=GREEN, sw=2, dash="7 5"))
    for tick in (2, 4, 6, 8, 10):
        xx = sx0 + (tick - xlo) / (xhi - xlo) * (sx1 - sx0)
        p.append(txt(xx, sy1 + 27, str(tick), size=13, fill=DIM, anchor="middle"))
    for tick in (-10, 0, 10, 20, 30):
        yy = sy1 - (tick - ylo) / (yhi - ylo) * (sy1 - sy0)
        p.append(line(sx0, yy, sx1, yy, stroke=MUTED if tick == 0 else GRID,
                      sw=1.5 if tick == 0 else 1,
                      dash=None if tick == 0 else "4 6"))
        p.append(txt(sx0 - 12, yy + 5, f"{tick:+d}", size=13, fill=DIM, anchor="end"))
    for datum in data:
        xx = sx0 + (f(datum["deviations_per_game"]) - xlo) / (xhi - xlo) * (sx1 - sx0)
        yy = sy1 - (f(datum["paired_margin"]) - ylo) / (yhi - ylo) * (sy1 - sy0)
        delta = datum["value_debias"]
        color = series_colors.get(delta, PURPLE)
        chosen_point = datum["chosen"] == "1"
        artifact = datum["proxy_rescue_artifact"] == "1"
        p.append(circle(xx, yy, 10 if chosen_point else 6,
                        fill=WHITE if chosen_point else color,
                        stroke=color, sw=3 if chosen_point else 1.5,
                        opacity=0.3 if artifact else 1.0))
    p.append(txt((sx0 + sx1) / 2, sy1 + 55, "deviations per game", size=15,
                 fill=MUTED, anchor="middle"))
    p.append(txt(myopic_x, sy0 - 12, "stage-1 myopic: 4.27", size=13,
                 fill=GREEN, anchor="middle"))
    p.append(txt(x + w - 30, y + h - 35,
                 "Green band is a reconstructed ±0.75-deviation screen.",
                 size=14, fill=MUTED, anchor="end"))

    p.append(rect(150, 748, 1200, 42, fill="#3a2c1c", stroke=ORANGE, radius=8))
    p.append(txt(
        750,
        775,
        "λ ≥ 1 is muted as a proxy-rescue artifact · calibration failed; a separate live evaluation supplies the effect estimate.",
        size=16,
        fill=ORANGE,
        weight=600,
        anchor="middle",
    ))
    write("07_dynamics_sweep.svg", p)


def frontier_leaderboard() -> None:
    data = rows("frontier_leaderboard.csv")
    width, height = 1550, 1050
    p = base_svg(width, height, "Final 13-model MegaGem leaderboard",
                 "A point-and-whisker plot of deck-clustered Plackett-Luce Elo for a balanced 624-game tournament, with the trained distilled 4B nominally first and its difference from Gemini 3.1 Pro unresolved.")
    heading(
        p,
        "A trained 4B reaches the frontier",
        "Plackett–Luce Elo · 624 games · 144 appearances/model · 95% deck-bootstrap CI",
        width,
    )
    min_v, max_v = 450, 1375
    x0, x1, top, row_h = 475, 1450, 120, 65
    for tick in (500, 750, 1000, 1250):
        xx = x0 + (tick - min_v) / (max_v - min_v) * (x1 - x0)
        p.append(line(xx, top, xx, top + row_h * len(data), stroke=GRID, sw=1, dash="4 7"))
        p.append(txt(xx, top - 14, str(tick), size=15, fill=DIM, anchor="middle"))
    for i, d in enumerate(data):
        y = top + i * row_h
        elo, ci_lo, ci_hi = f(d["elo"]), f(d["ci_low"]), f(d["ci_high"])
        role = d["tier"]
        color = GREEN if d["model"] == "MegaGem Distilled 4B" else BLUE if role == "trained" else DIM if role == "base" else "#6f8ca0"
        if role == "trained":
            p.append(rect(40, y + 3, 1410, 55, fill="#20351f",
                          radius=6, opacity=0.28))
        p.append(txt(x0 - 30, y + 39, d["model"], size=20, anchor="end",
                     fill=WHITE if role == "trained" else TEXT,
                     weight=700 if role == "trained" else 400))
        xe = x0 + (elo - min_v) / (max_v - min_v) * (x1 - x0)
        p.append(line(x0, y + 31, x1, y + 31, stroke=GRID, sw=1,
                      opacity=0.35))
        xlo = x0 + (ci_lo - min_v) / (max_v - min_v) * (x1 - x0)
        xhi = x0 + (ci_hi - min_v) / (max_v - min_v) * (x1 - x0)
        p.append(line(xlo, y + 31, xhi, y + 31, stroke=color, sw=4,
                      opacity=0.9))
        p.append(line(xlo, y + 22, xlo, y + 40, stroke=color, sw=2,
                      opacity=0.9))
        p.append(line(xhi, y + 22, xhi, y + 40, stroke=color, sw=2,
                      opacity=0.9))
        p.append(circle(xe, y + 31, 8, fill=color, stroke=WHITE, sw=2))
        p.append(txt(xhi + 12, y + 39, f"{elo:.0f}", size=18,
                     fill=color if role == "trained" else TEXT,
                     weight=700))
    p.append(rect(825, 950, 620, 78, fill="#20351f", stroke=GREEN, radius=8))
    p.append(txt(1135, 980, "Separate top-three test · 450 games · 150 deal clusters",
                 size=16, fill=GREEN, weight=600, anchor="middle"))
    p.append(txt(1135, 1010, "Distilled–Pro unresolved; both > Flash", size=19,
                 fill=GREEN, weight=600, anchor="middle"))
    p.append(txt(55, 1003, "Frontier models zero-shot · SFT and Distilled 4B task-trained",
                 size=18, fill=MUTED))
    write("06_frontier_leaderboard.svg", p)


def main() -> None:
    game_loop()
    research_pipeline()
    selfplay_diagnosis()
    offline_regret()
    liquidity_and_distillation()
    dynamics_sweep()
    frontier_leaderboard()


if __name__ == "__main__":
    main()
