"""Draw the learning curve of a recorded run: test accuracy against training steps.

Reads a run directory's ``eval/step-*.json`` (what ``run.py`` scores every
few steps) and writes the figure and an ``accuracy.csv`` next to it, in the
figure style the repository's results share with the docs site. The paper's
Table 5 numbers for the same split are drawn as reference lines.

    uv run --no-project --python 3.12 --with matplotlib plot.py --run-dir work --out results/<run>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figure style shared with the docs site (paper surface, warm ink, hairline
# grid, and the site's brand red / blue / green as the fixed series order).
PAPER = "#fbfaf8"
INK = "#302c28"
MUTED = "#716b65"
HAIRLINE = "#ddd8d0"
BRAND, BLUE, GREEN = "#a03729", "#2e5fa6", "#3f7d44"
STYLE = {
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "text.color": INK,
    "axes.titlecolor": INK,
    "axes.titleweight": "normal",
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": HAIRLINE,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": HAIRLINE,
    "grid.linewidth": 1.0,
    "axes.axisbelow": True,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "legend.frameon": False,
    "legend.labelcolor": INK,
    "font.size": 11,
}
#: Table 5 of arXiv:2601.19897, Science Q&A, Qwen2.5-7B-Instruct: exact match on the test split.
PAPER_ACCURACY = {"base": 32.1, "SFT": 66.2, "SDFT": 70.2}


def read_accuracies(run_dir: Path) -> list[tuple[int, float, int, int]]:
    """``(step, accuracy, correct, total)`` for every scored step, in step order."""
    rows = []
    for path in run_dir.glob("eval/step-*.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append((int(path.stem.split("-")[1]), result["accuracy"], result["num_correct"], result["num_total"]))
    if not rows:
        raise FileNotFoundError(f"no eval/step-*.json under {run_dir}")
    return sorted(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("work"))
    parser.add_argument("--out", type=Path, required=True, help="results directory for the figure and accuracy.csv")
    arguments = parser.parse_args()
    rows = read_accuracies(arguments.run_dir)
    arguments.out.mkdir(parents=True, exist_ok=True)
    with (arguments.out / "accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "accuracy", "correct", "total"])
        writer.writerows(rows)

    steps = [row[0] for row in rows]
    accuracies = [100 * row[1] for row in rows]
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        for label, value in PAPER_ACCURACY.items():
            axis.axhline(value, color=HAIRLINE, linewidth=1.0, linestyle=(0, (4, 3)))
            axis.annotate(
                f"paper {label} {value:.1f}",
                xy=(steps[-1], value),
                xytext=(4, 0),
                textcoords="offset points",
                color=MUTED,
                fontsize=9,
                va="center",
            )
        axis.plot(steps, accuracies, color=BRAND, marker="o", markersize=5)
        axis.annotate(
            f"{accuracies[-1]:.1f}",
            xy=(steps[-1], accuracies[-1]),
            xytext=(0, 8),
            textcoords="offset points",
            color=INK,
            ha="center",
        )
        axis.annotate(
            f"{accuracies[0]:.1f}",
            xy=(steps[0], accuracies[0]),
            xytext=(0, 8),
            textcoords="offset points",
            color=INK,
            ha="center",
        )
        axis.set_xlabel("training step (32 prompts each)")
        axis.set_ylabel("test accuracy (%)")
        axis.set_title("SDFT on Science Q&A through Reef: Qwen2.5-7B-Instruct, exact match on 507 test prompts")
        axis.set_xlim(left=0)
        axis.set_ylim(0, 100)
        axis.margins(x=0.02)
        figure.savefig(arguments.out / "learning_curve.png", dpi=160)
    print(f"{len(rows)} scored steps -> {arguments.out / 'learning_curve.png'}")


if __name__ == "__main__":
    main()
