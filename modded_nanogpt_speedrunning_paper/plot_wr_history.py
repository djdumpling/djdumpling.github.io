"""
Reconstructs the modded-nanoGPT WR history plot.

Data sourced from https://github.com/KellerJordan/modded-nanogpt/blob/master/README.md
as of WR #82 (2026-04-29). WR #82 time uses our measured 81.2 s.

Output: ../public/modded_nanogpt/wr_history.png
"""

from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


RECORDS = [
    # (wr, date "MM/DD/YY", wall_seconds)
    (1,  "05/28/24", 2700),
    (2,  "06/06/24", 1884),
    (3,  "10/04/24", 1494),
    (4,  "10/11/24", 1338),
    (5,  "10/14/24",  912),
    (6,  "10/18/24",  786),
    (7,  "10/18/24",  720),
    (8,  "11/03/24",  648),
    (9,  "11/06/24",  492),
    (10, "11/08/24",  468),
    (11, "11/10/24",  432),
    (12, "11/19/24",  303),
    (13, "11/24/24",  280),
    (14, "12/04/24",  265),
    (15, "12/08/24",  237),
    (16, "12/10/24",  228),
    (17, "12/17/24",  214),
    (18, "01/04/25",  204),
    (19, "01/13/25",  189),
    (20, "01/16/25",  180),
    (21, "01/26/25",  176),
    (22, "05/24/25",  179),
    (23, "05/25/25",  179),
    (24, "05/30/25",  178),
    (25, "07/13/25",  174),
    (26, "07/13/25",  172),
    (27, "07/18/25",  169),
    (28, "08/23/25",  169),
    (29, "09/03/25",  164),
    (30, "09/05/25",  163),
    (31, "09/10/25",  160),
    (32, "09/11/25",  157),
    (33, "09/15/25",  154),
    (34, "09/18/25",  152),
    (35, "09/21/25",  151),
    (36, "09/23/25",  150),
    (37, "09/27/25",  149),
    (38, "09/29/25",  149),
    (39, "09/30/25",  147),
    (40, "10/04/25",  142),
    (41, "10/24/25",  141),
    (42, "10/27/25",  139),
    (43, "11/10/25",  137),
    (44, "11/16/25",  136),
    (45, "11/18/25",  135),
    (46, "11/29/25",  132),
    (47, "12/10/25",  132),
    (48, "12/11/25",  130),
    (49, "12/14/25",  129),
    (50, "12/18/25",  128),
    (51, "12/19/25",  125),
    (52, "12/21/25",  122),
    (53, "12/22/25",  119),
    (54, "12/26/25",  116),
    (55, "12/29/25",  115),
    (56, "12/31/25",  114),
    (57, "01/04/26",  113),
    (58, "01/07/26",  109),
    (59, "01/10/26",  107),
    (60, "01/16/26",  106),
    (61, "01/18/26",  105),
    (62, "01/19/26",   99),
    (63, "01/26/26",   99),
    (64, "01/30/26",   98),
    (65, "01/30/26",   97),
    (66, "01/31/26",   96),
    (67, "01/31/26",   92),
    (68, "01/31/26",   92),
    (69, "02/02/26",   92),
    (70, "02/03/26",   91),
    (71, "02/06/26",   91),
    (72, "02/10/26",   90),
    (73, "02/12/26",   89),
    (74, "02/16/26",   88),
    (75, "02/23/26",   87),
    (76, "02/28/26",   87),
    (77, "03/06/26",   86),
    (78, "03/22/26",   86),
    (79, "04/04/26",   85),
    (80, "04/08/26",   84),
    (81, "04/22/26",   82),
    (82, "04/29/26",   81.2),  # ours
]


def main() -> None:
    dates = [datetime.strptime(d, "%m/%d/%y") for _, d, _ in RECORDS]
    times = [t for _, _, t in RECORDS]

    blue = "#1f77b4"
    red = "#d62728"

    fig, ax = plt.subplots(figsize=(12, 7))

    # Faint connecting line through the record progression
    ax.plot(dates, times, color="#888888", alpha=0.35, linewidth=1.0, zorder=1)

    # Records #1 - #81 in blue
    ax.scatter(
        dates[:-1], times[:-1],
        c=blue, s=42,
        edgecolors="white", linewidths=0.6,
        zorder=3, label="WR #1 – #81",
    )

    # WR #82 (ours) in red, same size as the others
    ax.scatter(
        [dates[-1]], [times[-1]],
        c=red, s=42,
        edgecolors="white", linewidths=0.6,
        zorder=4, label="WR #82 (ours)",
    )

    # Annotate ours
    ax.annotate(
        "WR #82\n(learnable XSA)",
        xy=(dates[-1], times[-1]),
        xytext=(-95, 30), textcoords="offset points",
        fontsize=10, color=red,
        arrowprops=dict(arrowstyle="->", color=red, lw=1.0),
    )

    ax.set_yscale("log")
    ax.set_xlabel("date")
    ax.set_ylabel("wall-clock time to val loss 3.28 (s, log scale)")
    ax.set_title("modded-nanoGPT speedrun world record progression")
    ax.grid(True, which="both", alpha=0.3)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()

    out_path = Path(__file__).resolve().parent.parent / "public" / "modded_nanogpt" / "wr_history.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
