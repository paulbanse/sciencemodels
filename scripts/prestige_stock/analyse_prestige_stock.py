"""Analysis for the fifth-axis prestige-stock sweep.

Question (docs/way_forward.md Path A): does the choice of prestige stock that
weights the supervisor roulette change WHERE the wedge opens, or only HOW FAST?

Design that makes the question answerable:

* The axis has a channel only when `agent_generation_time > 0` -- the stock is
  read exclusively by the supervisor roulette. The 18 entry-off cells are
  therefore required to be BIT-IDENTICAL across the three stocks; that is
  checked first (`identity_entry_off`), and a failure invalidates the run
  rather than the interpretation.
* "Where" = which of the 12 (entry x curiosity x landscape) cells show an open
  wedge, judged under two pre-declared rules:
      D1  wedge = c_local - c_ideal >= 0.10
      D2  c_ideal < 0 (prestige decoupled from quality-side merit)
  Location agreement is the share of cells whose classification is the same
  for all three stocks.
* "How fast" = the paired, within-cell difference in the wedge (and in the
  other reported outcomes) between stocks, seed-matched, with bootstrap 95%
  intervals over the 20 seeds.

NOT CALIBRATED: model-internal only (owner ruling 2026-11-13).

Run: MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/prestige_stock/analyse_prestige_stock.py
"""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data" / "prestige_stock_v1"
FIG_DIR = OUT_DIR / "figures"

STOCKS = ("prestige", "prestige_vanishing", "prestige_visibility")
STOCK_SHORT = {"prestige": "pre", "prestige_vanishing": "pvan",
               "prestige_visibility": "pvis"}
ENTRY_RATES = (0.0, 0.1)
CURIOSITIES = (0.3, 0.5, 0.7)
LANDSCAPES = ("mgstd5", "noisypr5")

TAG_RE = re.compile(
    r"^cell_(?P<stock>pre|pvan|pvis)_gen(?P<gen>\d+)_cur(?P<cur>\d+)_(?P<land>.+)$"
)
STOCK_FROM_SHORT = {v: k for k, v in STOCK_SHORT.items()}

# pre-declared rules for an "open wedge"
WEDGE_OPEN_D1 = 0.10
BOOTSTRAP = 5000

METRICS = {
    "wedge_fin": ("c_local_fin", "c_ideal_fin"),
    "wedge_tm": ("c_local_tm", "c_ideal_tm"),
    "c_ideal_fin": None,
    "c_local_fin": None,
    "c_ideal_tm": None,
    "c_local_tm": None,
    "top10": None,
    "prestige_gap": None,
    "gap_active": None,
    "ideal_loss_sum": None,
    "ideal_loss_mean": None,
}


def load_rows():
    rows, files = [], sorted(OUT_DIR.glob("cell_*.csv"))
    for path in files:
        m = TAG_RE.match(path.stem)
        if not m:
            continue
        stock = STOCK_FROM_SHORT[m.group("stock")]
        with open(path, newline="") as fh:
            for rec in csv.DictReader(fh):
                rec["_stock"] = stock
                rec["_entry"] = int(m.group("gen")) / 10.0
                rec["_curiosity"] = int(m.group("cur")) / 10.0
                rec["_landscape"] = m.group("land")
                rec["_file"] = path.name
                rows.append(rec)
    return rows, files


def f(rec, key):
    return float(rec[key])


COLUMNS = {
    "c_ideal_fin": "corr_prestige_ideal_merit_final",
    "c_local_fin": "corr_prestige_local_merit_final",
    "c_ideal_tm": "Mean_of_Corr_prestige_ideal_merit",
    "c_local_tm": "Mean_of_Corr_prestige_local_merit",
    "top10": "top10pct_prestige_share",
    "prestige_gap": "newcomer_incumbent_prestige_gap",
    "gap_active": "newcomer_incumbent_gap_active_stock",
    "ideal_loss_sum": "ideal_merit_sum",
    "ideal_loss_mean": "ideal_merit_mean",
}


def value(rec, name):
    """Metric value for one run; wedge metrics differ the two correlations."""
    if name in ("wedge_fin", "wedge_tm"):
        local, ideal = METRICS[name]  # metric names
        return value(rec, local) - value(rec, ideal)
    v = float(rec[COLUMNS[name]])
    if name.startswith("ideal_loss"):
        return -v  # model records actual-minus-ideal (<=0); report the loss
    return v


def cell_key(rec):
    return (rec["_entry"], rec["_curiosity"], rec["_landscape"])


def summarise(rows):
    groups = defaultdict(list)
    for rec in rows:
        groups[(rec["_stock"],) + cell_key(rec)].append(rec)
    out = []
    for key in sorted(groups, key=lambda k: (k[1], k[2], k[3], k[0])):
        stock, entry, curiosity, landscape = key
        recs = groups[key]
        row = {
            "stock": stock,
            "agent_generation_time": entry,
            "initial_curiosity": curiosity,
            "landscape": landscape,
            "tag": recs[0]["_file"][:-4],
            "n_seeds": len(recs),
        }
        for name in METRICS:
            vals = np.array([value(r, name) for r in recs], dtype=float)
            finite = vals[np.isfinite(vals)]
            if len(finite) == 0:
                # e.g. the newcomer gap in entry-off cells: undefined, not zero
                row[f"{name}_mean"] = float("nan")
                row[f"{name}_sd"] = float("nan")
                row[f"{name}_se"] = float("nan")
                continue
            row[f"{name}_mean"] = float(finite.mean())
            row[f"{name}_sd"] = (
                float(finite.std(ddof=1)) if len(finite) > 1 else float("nan")
            )
            row[f"{name}_se"] = (
                row[f"{name}_sd"] / np.sqrt(len(finite))
                if np.isfinite(row[f"{name}_sd"]) else float("nan")
            )
        out.append(row)
    return out


def identity_check(rows, seeds: int = 20):
    """entry=0 cells must be bit-identical across stocks (axis has no channel).

    Only cells where every stock has the full `seeds` rows are judged; a
    partial cell (read mid-run) or a missing stock is reported as incomplete,
    never as a difference.
    """
    by_cell = defaultdict(dict)
    for rec in rows:
        if rec["_entry"] != 0.0:
            continue
        key = cell_key(rec)
        stock = rec["_stock"]
        # seed id = row order within the cell file; compare full row content.
        # Exclusions: the active-stock gap (axis-bearing by construction) and
        # `entry_prestige_stock` itself -- since the axis moved from the
        # driver's monkeypatch into the constructor (post-CDF-v5), the report
        # carries the treatment LABEL per row; at g = 0 it must differ while
        # every dynamics column stays bit-identical.
        by_cell[key].setdefault(stock, []).append(
            tuple(sorted((k, repr(v)) for k, v in rec.items()
                         if not k.startswith("_")
                         and k not in ("newcomer_incumbent_gap_active_stock",
                                       "entry_prestige_stock"))))
    mismatches, incomplete = [], []
    for key, per_stock in sorted(by_cell.items()):
        lists = {s: sorted(v) for s, v in per_stock.items()}
        if set(lists) != set(STOCKS):
            incomplete.append((key, "missing stock", sorted(lists)))
            continue
        if any(len(lists[s]) != seeds for s in STOCKS):
            incomplete.append(
                (key, "partial", {s: len(lists[s]) for s in STOCKS}))
            continue
        ref = lists["prestige"]
        for s in STOCKS[1:]:
            if lists[s] != ref:
                mismatches.append((key, s, "differs"))
    return mismatches, incomplete, len(by_cell)


def paired_contrasts(rows, rng):
    """Seed-matched stock differences inside each entry-on cell."""
    out = []
    for entry in (0.1,):
        for curiosity in CURIOSITIES:
            for landscape in LANDSCAPES:
                per_stock = {}
                for stock in STOCKS:
                    recs = [r for r in rows
                            if r["_stock"] == stock
                            and r["_entry"] == entry
                            and r["_curiosity"] == curiosity
                            and r["_landscape"] == landscape]
                    recs.sort(key=lambda r: r["_file"])  # stable; seed = order
                    per_stock[stock] = recs
                n = {s: len(v) for s, v in per_stock.items()}
                if len(set(n.values())) != 1 or next(iter(n.values())) == 0:
                    out.append({"agent_generation_time": entry,
                                "initial_curiosity": curiosity,
                                "landscape": landscape,
                                "error": f"unbalanced panel {n}"})
                    continue
                for name in METRICS:
                    base = np.array([value(r, name) for r in per_stock["prestige"]])
                    for stock in STOCKS[1:]:
                        other = np.array([value(r, name) for r in per_stock[stock]])
                        diff = other - base
                        if np.all(np.isnan(diff)):
                            lo = hi = float("nan")
                        else:
                            boot = [
                                float(np.nanmean(
                                    diff[rng.integers(0, len(diff), len(diff))]))
                                for _ in range(BOOTSTRAP)
                            ]
                            lo, hi = np.percentile(boot, [2.5, 97.5])
                        out.append({
                            "agent_generation_time": entry,
                            "initial_curiosity": curiosity,
                            "landscape": landscape,
                            "metric": name,
                            "contrast": f"{STOCK_SHORT[stock]}-pre",
                            "n_pairs": len(diff),
                            "mean_prestige": float(np.nanmean(base)),
                            "mean_other": float(np.nanmean(other)),
                            "mean_diff": float(np.nanmean(diff)),
                            "boot_lo": lo,
                            "boot_hi": hi,
                            "excludes_zero": bool(np.isfinite(lo) and
                                                  (lo > 0 or hi < 0)),
                        })
    return out


def location_table(summary):
    """Open/closed classification per (entry, curiosity, landscape, stock)."""
    rows = []
    for r in summary:
        rows.append({
            "stock": r["stock"],
            "agent_generation_time": r["agent_generation_time"],
            "initial_curiosity": r["initial_curiosity"],
            "landscape": r["landscape"],
            "wedge_fin": r["wedge_fin_mean"],
            "wedge_tm": r["wedge_tm_mean"],
            "c_ideal_fin": r["c_ideal_fin_mean"],
            "D1_open": r["wedge_fin_mean"] >= WEDGE_OPEN_D1,
            "D2_open": r["c_ideal_fin_mean"] < 0,
        })
    return rows


def agreement(location):
    """Open/closed agreement across the three stocks, per grid cell.

    A cell only counts if all three stocks are present; an incomplete cell is
    reported as such rather than silently counted as agreement.
    """
    cells = sorted({(r["agent_generation_time"], r["initial_curiosity"],
                     r["landscape"]) for r in location})
    report = {}
    for rule in ("D1_open", "D2_open"):
        agree = disagree = incomplete = 0
        changed = []
        for c in cells:
            flags = {r["stock"]: r[rule] for r in location
                     if (r["agent_generation_time"], r["initial_curiosity"],
                         r["landscape"]) == c}
            if len(flags) != len(STOCKS):
                incomplete += 1
                continue
            if len(set(flags.values())) == 1:
                agree += 1
            else:
                disagree += 1
                changed.append((c, flags))
        report[rule] = {"cells": len(cells), "agree": agree,
                        "disagree": disagree, "incomplete": incomplete,
                        "changed": changed}
    return report


def rank_analysis(summary):
    """Do the stocks preserve the ORDER of cells by wedge? (where vs how fast)

    High rank correlation with shifted levels = the same cells open the wedge,
    only the magnitude changes. Scrambled ranks = the stock moves WHERE the
    wedge opens.
    """
    cells = [(e, c, l) for e in ENTRY_RATES for c in CURIOSITIES
             for l in LANDSCAPES]
    out = []
    for metric in ("wedge_fin", "wedge_tm"):
        table = {}
        for stock in STOCKS:
            col = []
            for c in cells:
                hit = [r for r in summary
                       if r["stock"] == stock
                       and (r["agent_generation_time"], r["initial_curiosity"],
                            r["landscape"]) == c]
                col.append(hit[0][f"{metric}_mean"] if hit else np.nan)
            table[stock] = np.array(col, dtype=float)
        for a, b in (("prestige", "prestige_vanishing"),
                     ("prestige", "prestige_visibility"),
                     ("prestige_vanishing", "prestige_visibility")):
            x, y = table[a], table[b]
            ok = np.isfinite(x) & np.isfinite(y)
            rho = pval = float("nan")
            if ok.sum() >= 3:
                rho, pval = stats.spearmanr(x[ok], y[ok])
            both = np.isfinite(x) & np.isfinite(y)
            out.append({
                "metric": metric,
                "pair": f"{STOCK_SHORT[b]} vs {STOCK_SHORT[a]}",
                "cells": int(ok.sum()),
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "mean_level_a": float(np.nanmean(x)) if ok.any() else float("nan"),
                "mean_level_b": float(np.nanmean(y)) if ok.any() else float("nan"),
                "mean_abs_level_diff": (
                    float(np.nanmean(np.abs(y - x))) if both.any() else float("nan")
                ),
            })
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_wedge_matrix(summary, location):
    cells = [(e, c, l) for e in ENTRY_RATES for c in CURIOSITIES
             for l in LANDSCAPES]
    labels = [f"entry {e:g}, cur {c:g}, {l}" for e, c, l in cells]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    for ax, metric, title in (
        (axes[0], "wedge_fin", "wedge at the final step (local - ideal)"),
        (axes[1], "wedge_tm", "wedge time-mean (local - ideal)"),
    ):
        mat = np.full((len(cells), len(STOCKS)), np.nan)
        for i, c in enumerate(cells):
            for j, s in enumerate(STOCKS):
                hit = [r for r in summary
                       if r["stock"] == s
                       and (r["agent_generation_time"], r["initial_curiosity"],
                            r["landscape"]) == c]
                if hit:
                    mat[i, j] = hit[0][f"{metric}_mean"]
        finite = mat[np.isfinite(mat)]
        lim = np.nanmax(np.abs(finite))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center", va="center",
                            fontsize=7, color="black")
        ax.set_xticks(range(len(STOCKS)))
        ax.set_xticklabels([STOCK_SHORT[s] for s in STOCKS])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(-0.5, len(STOCKS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle(
        "Fifth-axis sweep: wedge by prestige stock - NOT CALIBRATED, "
        "model-internal (mean over 20 seeds)",
        fontsize=12, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = FIG_DIR / "wedge_matrix.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_contrasts(contrasts):
    keep = [c for c in contrasts if c.get("metric") in
            ("wedge_fin", "wedge_tm", "c_ideal_fin", "c_local_fin",
             "top10", "prestige_gap", "ideal_loss_mean")]
    # row order is identical in both panels (same sort key), so labels are
    # built once from the first panel's rows and shared
    order = sorted([c for c in keep
                    if c["contrast"] == f"{STOCK_SHORT[STOCKS[1]]}-pre"],
                   key=lambda c: (c["metric"], c["initial_curiosity"],
                                  c["landscape"]))
    ylabels = [f"{c['metric']} | cur {c['initial_curiosity']:g} | {c['landscape']}"
               for c in order]
    fig, axes = plt.subplots(1, len(STOCKS[1:]), figsize=(13, 9.5),
                             sharey=True)
    if len(STOCKS[1:]) == 1:
        axes = [axes]
    for ax, stock in zip(axes, STOCKS[1:]):
        rows = [c for c in keep if c["contrast"] == f"{STOCK_SHORT[stock]}-pre"]
        rows.sort(key=lambda c: (c["metric"],
                                 c["initial_curiosity"],
                                 c["landscape"]))
        for i, c in enumerate(rows):
            col = "#2ca02c" if c["excludes_zero"] else "#999999"
            ax.plot([c["boot_lo"], c["boot_hi"]], [i, i], color=col, lw=2.5)
            ax.plot([c["mean_diff"]], [i], "o", color=col, ms=5)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{STOCK_SHORT[stock]} - pre", fontsize=10)
        ax.set_xlabel("paired difference (95% bootstrap CI)")
        ax.grid(alpha=0.3, axis="x")
    axes[0].set_yticks(range(len(ylabels)))
    axes[0].set_yticklabels(ylabels, fontsize=6.5)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Fifth-axis sweep: within-cell stock effects, entry = 0.1 cells - "
        "NOT CALIBRATED (green = CI excludes zero)",
        fontsize=12, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = FIG_DIR / "stock_effects.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="analyse prestige-stock sweep")
    parser.add_argument("--out", default=None,
                        help="archive dir to analyse (default: data/prestige_stock_v1)")
    args = parser.parse_args()

    global OUT_DIR, FIG_DIR
    if args.out:
        OUT_DIR = Path(args.out)
        FIG_DIR = OUT_DIR / "figures"

    rows, files = load_rows()
    if not rows:
        print("no cell CSVs under", OUT_DIR)
        return 1
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cells = sorted({cell_key(r) for r in rows})
    n_grid_cells = len(ENTRY_RATES) * len(CURIOSITIES) * len(LANDSCAPES)
    n_expected = len(STOCKS) * n_grid_cells
    print(f"GROSS grid cells (entry x curiosity x landscape)={n_grid_cells} "
          f"x {len(STOCKS)} stocks = {n_expected} stock-cells "
          f"x 20 seeds = {n_expected * 20} runs")
    print(f"NET  stock-cells={len(files)} | runs={len(rows)} | "
          f"distinct grid cells present={len(cells)}")
    counts = defaultdict(int)
    for r in rows:
        counts[(r["_stock"],) + cell_key(r)] += 1
    off = {k: v for k, v in counts.items() if v != 20}
    print(f"stock-cells with != 20 rows: {off if off else 'none'}")

    mismatches, incomplete, n_entry_off = identity_check(rows)
    print(f"IDENTITY check (entry=0 must be identical across stocks): "
          f"{n_entry_off} cells present, {len(mismatches)} real mismatches, "
          f"{len(incomplete)} incomplete")
    for m in mismatches[:5]:
        print("   MISMATCH", m)
    for m in incomplete[:3]:
        print("   incomplete:", m)

    summary = summarise(rows)
    write_csv(OUT_DIR / "summary_cell.csv", summary)

    rng = np.random.default_rng(20261114)
    contrasts = paired_contrasts(rows, rng)
    write_csv(OUT_DIR / "paired_contrasts.csv", contrasts)

    location = location_table(summary)
    write_csv(OUT_DIR / "location_open_wedge.csv", location)

    ranks = rank_analysis(summary)
    write_csv(OUT_DIR / "wedge_rank_agreement.csv", ranks)
    print("\nRANK agreement of cells by wedge (how similar is the ORDERING):")
    for r in ranks:
        print(f"  {r['metric']:10s} {r['pair']:16s} cells={r['cells']:2d} "
              f"rho={r['spearman_rho']:+.3f} p={r['p_value']:.3g} "
              f"levels {r['mean_level_a']:+.4f} vs {r['mean_level_b']:+.4f} "
              f"mean|diff|={r['mean_abs_level_diff']:.4f}")
    agr = agreement(location)
    print("\nLOCATION agreement across the three stocks "
          f"({len(location) // 3} grid cells):")
    for rule, rep in agr.items():
        print(f"  {rule}: agree {rep['agree']}/{rep['cells']}, "
              f"disagree {rep['disagree']}, incomplete {rep['incomplete']}")
        for c, flags in rep["changed"]:
            print(f"      changed at entry {c[0]:g}, cur {c[1]:g}, {c[2]}: "
                  + ", ".join(f"{STOCK_SHORT[s]}={'open' if v else 'closed'}"
                              for s, v in flags.items()))

    sig = [c for c in contrasts if c.get("excludes_zero")]
    tot = len([c for c in contrasts if "metric" in c])
    print(f"\nSPEED: {len(sig)} of {tot} paired contrasts have a 95% CI "
          "excluding zero")
    for c in sorted(sig, key=lambda c: (-abs(c["mean_diff"])))[:12]:
        print(f"  {c['metric']:16s} cur {c['initial_curiosity']:g} "
              f"{c['landscape']:8s} {c['contrast']:9s} "
              f"diff={c['mean_diff']:+.4f} "
              f"[{c['boot_lo']:+.4f}, {c['boot_hi']:+.4f}] "
              f"(base {c['mean_prestige']:+.4f}, other {c['mean_other']:+.4f})")

    figs = [plot_wedge_matrix(summary, location),
            plot_contrasts([c for c in contrasts if "metric" in c])]
    print("\nfigures:")
    for p in figs:
        print("  ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
