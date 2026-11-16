"""F8 -- generational-turnover lifecycle reversal (drafts/paper_sketch_v1.md 4.3).

Two panels from the read-only `data/turnover_v1/` archive:

  Panel A  mean Corr(prestige, ideal_merit) over t in [1, 400] for the three
           exit modes, with a 95% CI ribbon across the 18 cells per mode, a
           zero line, and the Discovery (t <= 80) and Depletion (t >= 150)
           phases shaded.
  Panel B  the gap-inversion tradeoff: gap_recent shrinkage, terminal
           c_ideal deepening and top-10% concentration, each indexed to the
           immortal-founder baseline (= 100) so three units share one axis.

Provenance rules (publication-figure-production): every plotted number comes
from a file in `data/turnover_v1/`, asserted before drawing; the draft's
quoted top-10% levels (0.2735 / 0.2546 / 0.4133) are NOT reproducible from
that archive -- the artifact values are 0.2605 / 0.2327 / 0.4086 (also the
turnover report's own table) and are what the figure draws. Both are written
to the data CSV so the discrepancy is visible rather than silently resolved.

Run:  MPLBACKEND=Agg python3 scripts/report_notebook/turnover_lifecycle_figure.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from diagram_templates import HOUSE_STYLE, PALETTE          # noqa: E402

DATA = REPO / "data" / "turnover_v1"
OUT = REPO / "output" / "report_notebook_v1"
OUT.mkdir(parents=True, exist_ok=True)

PNG = OUT / "F8_turnover_lifecycle_reversal.png"
CSV = OUT / "F8_turnover_lifecycle_reversal_data.csv"

MODES = ["none", "fixed_span", "age_hazard"]
MODE_COLOR = {"none": PALETTE["purple"], "fixed_span": PALETTE["orange"],
              "age_hazard": PALETTE["green"]}
METRIC_COLOR = {"gap_recent": PALETTE["purple"],
                "|c_ideal| terminal": PALETTE["orange"],
                "top10": PALETTE["green"]}
DISCOVERY_END = 80          # t <= 80: peaks still rich
DEPLETION_START = 150       # t >= 150: correlation inverted
# the draft's quoted top-10% levels, kept for the record (see module docstring)
DRAFT_TOP10 = {"none": 0.2735, "fixed_span": 0.2546, "age_hazard": 0.4133}


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def assert_text_fits(fig, canvas_w: float, canvas_h: float) -> int:
    """Every non-tick text artist must sit inside the canvas.

    Closing a blind spot the vision pass caught: ``ax.title`` is only the
    CENTER title, so a left-aligned ``set_title(..., loc="left")`` was never
    audited -- Panel B's title overflowed the canvas by 8 px and lost its
    final glyph.  Tick labels are skipped (matplotlib keeps off-view ticks
    as invisible artists).
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    offenders = []
    checked = [0]

    def check(artist, where):
        txt = artist.get_text()
        if not txt.strip():
            return
        checked[0] += 1
        bb = artist.get_window_extent(renderer=r)
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > canvas_w + 1 or bb.y1 > canvas_h + 1:
            offenders.append((where, txt[:50].replace("\n", " "),
                              round(bb.x1, 1), round(bb.x0, 1)))

    for t in fig.texts:
        check(t, "fig")
    for i, ax in enumerate(fig.axes):
        for loc, artist in (("center", ax.title),
                            ("left", getattr(ax, "_left_title", None)),
                            ("right", getattr(ax, "_right_title", None))):
            if artist is not None:
                check(artist, f"ax{i}.title({loc})")
        check(ax.xaxis.label, f"ax{i}.xlabel")
        check(ax.yaxis.label, f"ax{i}.ylabel")
        if ax.get_legend():
            for t in ax.get_legend().get_texts():
                check(t, f"ax{i}.legend")
        for t in ax.texts:
            check(t, f"ax{i}.text")
    _require(not offenders,
             "text outside the canvas: "
             + "; ".join(f"{w}: {s!r} x1={x1}" for w, s, x1, _ in offenders))
    return checked[0]


def load_trajectories() -> pd.DataFrame:
    """Per-mode mean and 95% CI across the 18 cell trajectories of each mode."""
    files = sorted((DATA / "traj").glob("cell_*.csv"))
    _require(len(files) == 54, f"traj cells=54, found {len(files)}")
    frames = []
    for f in files:
        m = re.match(r"cell_([a-z_]+?)_g\d+_cur\d+_(?:mgstd5|noisypr5)\.csv",
                     f.name)
        _require(m is not None, f"unparsable traj name: {f.name}")
        d = pd.read_csv(f)
        _require(len(d) == 400 and d["step"].min() == 1
                 and d["step"].max() == 400,
                 f"{f.name}: expected steps 1..400, got "
                 f"{d['step'].min()}..{d['step'].max()}")
        d["exit_mode"] = m.group(1)
        frames.append(d)
    t = pd.concat(frames, ignore_index=True)
    _require(t["cell_tag"].nunique() == 54, "cell tags not unique")
    per_mode = t.groupby("exit_mode")["cell_tag"].nunique().to_dict()
    _require(per_mode == {m: 18 for m in MODES}, f"cells per mode: {per_mode}")

    g = (t.groupby(["exit_mode", "step"], as_index=False)
           .agg(mean=("c_ideal", "mean"), sd=("c_ideal", "std"),
                n=("c_ideal", "size")))
    _require((g["n"] == 18).all(), f"every step needs 18 cells: "
                                   f"{g['n'].unique()}")
    g["ci"] = 1.96 * g["sd"] / np.sqrt(g["n"])
    g["lo"] = g["mean"] - g["ci"]
    g["hi"] = g["mean"] + g["ci"]
    return g


def load_panel_b() -> pd.DataFrame:
    """Indexed gap / inversion / concentration, plus the artifact levels."""
    summary = pd.read_csv(DATA / "tables" / "mode_summary.csv")
    contrasts = pd.read_csv(DATA / "tables" / "paired_contrasts.csv")
    _require(len(summary) == 3, f"mode_summary rows=3, got {len(summary)}")
    summary = summary.set_index("exit_mode").loc[MODES]

    # the trajectory endpoint must equal the summary's terminal value exactly
    traj = load_trajectories()
    end = traj[traj["step"] == 400].set_index("exit_mode")["mean"]
    for m in MODES:
        _require(abs(end[m] - summary.loc[m, "c_ideal_fin"]) < 1e-12,
                 f"{m}: traj endpoint {end[m]} != summary "
                 f"{summary.loc[m, 'c_ideal_fin']}")

    gap = summary["gap_recent_last100"].abs()
    cin = summary["c_ideal_fin"].abs()
    top = summary["top10_fin"]
    base_gap, base_cin, base_top = gap["none"], cin["none"], top["none"]

    rows = []
    for m in MODES:
        rows.append(dict(exit_mode=m, series="gap_recent",
                         raw_value=-gap[m], indexed=100 * gap[m] / base_gap,
                         annotation="" if m == "none" else
                         f"{-100 * (1 - gap[m] / base_gap):.1f}%"))
        rows.append(dict(exit_mode=m, series="|c_ideal| terminal",
                         raw_value=-cin[m], indexed=100 * cin[m] / base_cin,
                         annotation="" if m == "none" else
                         f"{cin[m] / base_cin:.2f}x"))
        rows.append(dict(exit_mode=m, series="top10",
                         raw_value=top[m], indexed=100 * top[m] / base_top,
                         annotation=f"{top[m]:.3f}"))
    out = pd.DataFrame(rows)

    # headline numbers from paired_contrasts.csv must match what we annotate
    pc = contrasts.set_index(["mode", "metric"])
    for mode in ("fixed_span", "age_hazard"):
        shrink = 100 * (1 - summary.loc[mode, "gap_recent_last100"]
                        / summary.loc["none", "gap_recent_last100"])
        _require(abs(shrink - (55.0 if mode == "fixed_span" else 79.3)) < 0.1,
                 f"{mode} gap shrinkage {shrink:.1f}% != brief")
        deep = cin[mode] / base_cin
        _require(abs(deep - (2.79 if mode == "fixed_span" else 6.61)) < 0.01,
                 f"{mode} c_ideal multiple {deep:.2f}x != brief")
        _require(abs(pc.loc[(mode, "gap_recent_mean_last100"),
                            "mean_diff"]) > 0, "contrast missing")
    # the artifact levels differ from the draft's quoted ones: record both
    out["draft_quoted_top10"] = [
        DRAFT_TOP10.get(r.exit_mode, np.nan) if r.series == "top10" else np.nan
        for r in out.itertuples()]
    return out, summary


def draw(png: Path = PNG, csv_path: Path = CSV) -> tuple[Path, Path, dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traj = load_trajectories()
    panel_b, summary = load_panel_b()

    s = HOUSE_STYLE
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(7.5, 3.9), dpi=s["dpi"],
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.28})

    # ---------------- Panel A: lifecycle sign reversal
    axA.axvspan(1, DISCOVERY_END, color=PALETTE["blue_light"], alpha=0.35,
                linewidth=0, zorder=0)
    axA.axvspan(DEPLETION_START, 400, color=PALETTE["orange_light"],
                alpha=0.35, linewidth=0, zorder=0)
    axA.axhline(0.0, color="black", linewidth=0.9, linestyle="--", zorder=3)
    for m in MODES:
        d = traj[traj["exit_mode"] == m].sort_values("step")
        axA.fill_between(d["step"], d["lo"], d["hi"], color=MODE_COLOR[m],
                         alpha=0.18, linewidth=0, zorder=2)
        axA.plot(d["step"], d["mean"], color=MODE_COLOR[m], linewidth=1.7,
                 label=f"{m}  ({summary.loc[m, 'c_ideal_fin']:+.3f})",
                 zorder=4)
    axA.set_xlim(1, 400)
    axA.set_xlabel("step t")
    axA.set_ylabel("Corr(prestige, ideal merit)")
    axA.set_title("(a) Lifecycle sign reversal", loc="left",
                  fontsize=s["title_size"])
    axA.legend(fontsize=s["tick_size"] - 1, frameon=False, loc="upper right")
    yband = axA.get_ylim()[1] * 0.55
    axA.text(DISCOVERY_END / 2, yband, "Discovery", ha="center", va="center",
             fontsize=s["note_size"], color=PALETTE["gray"])
    axA.text((DEPLETION_START + 400) / 2, yband, "Depletion", ha="center",
             va="center", fontsize=s["note_size"], color=PALETTE["gray"])
    axA.text(0.985, 0.045, "zero", transform=axA.transAxes, ha="right",
             fontsize=s["note_size"], color=PALETTE["gray"])
    axA.tick_params(labelsize=s["tick_size"])

    # ---------------- Panel B: gap-inversion tradeoff (indexed bars)
    order = ["gap_recent", "|c_ideal| terminal", "top10"]
    x = np.arange(len(MODES))
    width = 0.26
    for j, series in enumerate(order):
        sub = panel_b[panel_b["series"] == series].set_index("exit_mode").loc[
            MODES]
        off = (j - 1) * width
        bars = axB.bar(x + off, sub["indexed"], width=width,
                       color=METRIC_COLOR[series], label=series, zorder=3)
        labels = []
        for k, r in enumerate(sub.itertuples()):
            raw = f"{r.raw_value:.3f}"
            if not r.annotation or r.annotation == raw:
                labels.append(raw)
            else:
                labels.append(f"{raw}\n{r.annotation}")
        axB.bar_label(bars, labels=labels, fontsize=6.0, padding=2,
                      rotation=90, color="black")
    axB.axhline(100.0, color=PALETTE["gray"], linewidth=0.9,
                linestyle="--", zorder=4)
    axB.set_xticks(x, ["none\n(immortal)", "fixed_span", "age_hazard"],
                   fontsize=s["tick_size"] - 1)
    axB.set_ylim(0, max(760, panel_b["indexed"].max() * 1.18))
    axB.set_ylabel("relative to immortal-founder baseline (= 100)")
    axB.set_title("(b) Gap-inversion tradeoff", loc="left",
                  fontsize=s["title_size"])
    axB.legend(fontsize=s["tick_size"] - 1, frameon=False, loc="upper left")
    axB.tick_params(axis="y", labelsize=s["tick_size"])

    fig.text(0.012, 0.070,
             "source: data/turnover_v1/traj/cell_*.csv (54 cells, 18 per "
             "mode, steps 1-400)",
             fontsize=s["note_size"], color=PALETTE["gray"])
    fig.text(0.012, 0.038,
             "tables: mode_summary.csv, paired_contrasts.csv (360 runs/mode)",
             fontsize=s["note_size"], color=PALETTE["gray"])
    fig.text(0.012, 0.006,
             "ribbon = 95% CI across cells; bars indexed to baseline (= 100)",
             fontsize=s["note_size"], color=PALETTE["gray"])
    fig.subplots_adjust(left=0.115, right=0.985, top=0.875, bottom=0.215,
                        wspace=0.34)
    # every text must fit the canvas (catches a left-aligned title, which
    # ax.title does not expose); raises instead of shipping a clipped glyph
    fig.canvas.draw()
    n_text = assert_text_fits(fig, *fig.canvas.get_width_height())
    fig.savefig(png, dpi=s["dpi"])
    plt.close(fig)

    # ---------------- plotted numbers to CSV
    rows = []
    for r in traj.itertuples():
        rows.append(dict(panel="A", exit_mode=r.exit_mode,
                         series="c_ideal", step=int(r.step),
                         mean=r.mean, ci_lo=r.lo, ci_hi=r.hi,
                         raw_value=np.nan, indexed=np.nan, annotation="",
                         draft_quoted_top10=np.nan,
                         source="data/turnover_v1/traj/cell_*.csv"))
    for r in panel_b.itertuples():
        rows.append(dict(panel="B", exit_mode=r.exit_mode,
                         series=r.series, step=np.nan, mean=np.nan,
                         ci_lo=np.nan, ci_hi=np.nan,
                         raw_value=r.raw_value, indexed=r.indexed,
                         annotation=r.annotation,
                         draft_quoted_top10=r.draft_quoted_top10,
                         source=("data/turnover_v1/tables/mode_summary.csv"
                                 + ("; draft value from "
                                    "drafts/paper_sketch_v1.md 4.3"
                                    if r.series == "top10" else ""))))
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    stats = {
        "png": str(png.relative_to(REPO)),
        "csv": str(csv_path.relative_to(REPO)),
        "panel_a_rows": int(len(traj)),
        "panel_b_rows": int(len(panel_b)),
        "text_artists_checked": int(n_text),
        "cells": 54, "runs_per_mode": 360,
        "terminal": {m: round(float(summary.loc[m, "c_ideal_fin"]), 6)
                     for m in MODES},
        "top10_artifact": {m: round(float(summary.loc[m, "top10_fin"]), 4)
                           for m in MODES},
        "top10_draft_quoted": DRAFT_TOP10,
    }
    return png, csv_path, stats


def main() -> int:
    png, csv_path, stats = draw()
    for k, v in stats.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
