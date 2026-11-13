"""Welfare probe v1 - analysis: welfare curves, collapse threshold, phase diagram.

Reads the run-tagged CSVs produced by run_welfare_sweep.py, writes
    data/welfare_v1/summary_cell.csv       per-cell means / sd / se across seeds
    data/welfare_v1/threshold_estimate.csv hinge-breakpoint fits + bootstrap CIs
    data/welfare_v1/figures/*.png          welfare curves, phase diagram, threshold

NOT CALIBRATED: model-internal exploration only (owner ruling 2026-11-13).

Collapse is defined operationally: within the sampled visibility-weight range
[0.1, 0.9] a series "collapses" when a continuous two-piece (hinge) fit in
    y = seed_FE + b1*x + b2*max(0, x - xi)
beats the single-slope fit at the 5% level (F test, 1 restriction) AND the
bootstrap-over-seeds breakpoint distribution is interior to the range. Every
fit is resampled over the 20 seeds for its uncertainty.

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/welfare/analyse_welfare.py
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
OUT_DIR = REPO / "data" / "welfare_v1"
FIG_DIR = OUT_DIR / "figures"

TAG_RE = re.compile(
    r"^cell_cur(?P<cur>\d+)_vis(?P<vis>[01])_gen(?P<gen>\d+)_mg(?P<leg>\d+)$"
)

OUTCOMES = {
    "welfare_loss_sum": "ideal_merit_sum",
    "welfare_loss_mean": "ideal_merit_mean",
    "ideal_merit_mean": "ideal_merit_mean",
    "pct_knowledge_harvested": "percentage_knowledge_harvested",
    "explored_pct": "explored_percentage",
    "top10_prestige_share": "top10pct_prestige_share",
    "corr_ideal_final": "corr_prestige_ideal_merit_final",
    "corr_local_final": "corr_prestige_local_merit_final",
    "corr_ideal_timemean": "Mean_of_Corr_prestige_ideal_merit",
    "corr_local_timemean": "Mean_of_Corr_prestige_local_merit",
}

SERIES_KEYS = ("number_gaussians", "use_visibility_reward", "agent_generation_time")
BOOTSTRAP = 500
XIS = (0.2, 0.4, 0.6, 0.8)


def load_rows():
    rows, files = [], sorted(OUT_DIR.glob("cell_*.csv"))
    for path in files:
        m = TAG_RE.match(path.stem)
        if not m:
            continue
        with open(path, newline="") as fh:
            for rec in csv.DictReader(fh):
                rec["_curiosity"] = int(m.group("cur")) / 10.0
                rec["_visibility_weight"] = 1.0 - int(m.group("cur")) / 10.0
                rec["_use_visibility_reward"] = m.group("vis") == "1"
                rec["_agent_generation_time"] = int(m.group("gen")) / 10.0
                rec["_number_gaussians"] = int(m.group("leg"))
                rec["_file"] = path.name
                rows.append(rec)
    return rows, files


def as_float(rec, key):
    return float(rec[key])


def outcome_value(rec, name):
    if name == "welfare_loss_sum":
        return -as_float(rec, "ideal_merit_sum")
    if name == "welfare_loss_mean":
        return -as_float(rec, "ideal_merit_mean")
    return as_float(rec, OUTCOMES[name])


def cell_key(rec):
    return (
        rec["_number_gaussians"],
        rec["_use_visibility_reward"],
        rec["_agent_generation_time"],
        rec["_curiosity"],
    )


def summarise(rows):
    """Per-cell mean/sd/se for every outcome -> list of dict rows."""
    groups = defaultdict(list)
    for rec in rows:
        groups[cell_key(rec)].append(rec)
    out = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], k[3])):
        leg, vis, entry, cur = key
        recs = groups[key]
        row = {
            "tag": f"cell_cur{int(round(cur * 10)):02d}_vis{1 if vis else 0}"
                   f"_gen{int(round(entry * 10)):02d}_mg{leg}",
            "number_gaussians": leg,
            "use_visibility_reward": vis,
            "agent_generation_time": entry,
            "curiosity": cur,
            "visibility_weight": 1.0 - cur,
            "n_seeds": len(recs),
        }
        for name in OUTCOMES:
            vals = np.array([outcome_value(r, name) for r in recs], dtype=float)
            row[f"{name}_mean"] = float(vals.mean())
            row[f"{name}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            row[f"{name}_se"] = (
                row[f"{name}_sd"] / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
            )
        out.append(row)
    return out


def panel_for(rows, outcome, leg, vis, entry):
    """(x, y, seed) arrays for one series/outcome, sorted by visibility weight."""
    sel = [
        r
        for r in rows
        if r["_number_gaussians"] == leg
        and r["_use_visibility_reward"] == vis
        and r["_agent_generation_time"] == entry
    ]
    sel.sort(key=lambda r: (r["_visibility_weight"], r["_curiosity"]))
    x = np.array([r["_visibility_weight"] for r in sel], dtype=float)
    y = np.array([outcome_value(r, outcome) for r in sel], dtype=float)
    # no seed column exists in the report: generate_data_parametric_exploration
    # writes repeats 0..19 in order per cell, and the stable sort above keeps
    # that order inside each visibility-weight block
    seed = np.arange(len(sel)) % 20
    return x, y, seed


def fit_sse(y, x, seed_ids, xi=None):
    uniq = np.unique(seed_ids)
    # (n_seeds - 1) dummies PLUS an intercept: n dummies without an intercept
    # is the equivalent saturated coding, but n-1 alone omits the constant and
    # mis-specifies the model (SSE then exceeds the total sum of squares).
    cols = [np.ones(len(y))]
    cols += [(seed_ids == u).astype(float) for u in uniq[1:]]
    cols.append(x)
    if xi is not None:
        cols.append(np.maximum(0.0, x - xi))
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(resid @ resid), X.shape[1]


def hinge_stats(y, x, seed_ids):
    """Best hinge over the candidate grid, with the F test against the linear fit."""
    sse0, k0 = fit_sse(y, x, seed_ids, xi=None)
    # invariant: an OLS fit that spans the constant cannot exceed the total
    # sum of squares. This is the check that caught the missing-intercept
    # coding, so it stays in the path.
    sst = float(((y - y.mean()) ** 2).sum())
    if sse0 > sst + 1e-9 * max(sst, 1.0):
        raise AssertionError(
            f"M0 SSE {sse0:.6e} exceeds SST {sst:.6e}: the seed fixed-effect "
            "design is mis-specified (missing intercept)"
        )
    n = len(y)
    best = None
    for xi in XIS:
        sse1, k1 = fit_sse(y, x, seed_ids, xi=xi)
        if best is None or sse1 < best[1]:
            best = (xi, sse1, k1)
    xi, sse1, k1 = best
    df1, df0 = n - k1, n - k0
    if not (df1 > 0 and df0 > 0) or sse1 <= 0:
        # degenerate panel (too few observations for the fixed effects):
        # no F test can be formed
        uniq = np.unique(seed_ids)
        X0 = np.column_stack(
            [np.ones(len(y))]
            + [(seed_ids == u).astype(float) for u in uniq[1:]]
            + [x]
        )
        beta0, *_ = np.linalg.lstsq(X0, y, rcond=None)
        return {"xi": xi, "sse0": sse0, "sse1": sse1, "f": np.nan,
                "p": np.nan, "slope": float(beta0[-1]), "n": n, "k1": k1}
    fstat = ((sse0 - sse1) / 1) / (sse1 / df1)
    pvalue = 1.0 - stats.f.cdf(fstat, 1, df0) if np.isfinite(fstat) else np.nan
    # linear slope (price of visibility pressure) from the no-hinge fit
    uniq = np.unique(seed_ids)
    X0 = np.column_stack(
        [np.ones(len(y))]
        + [(seed_ids == u).astype(float) for u in uniq[1:]]
        + [x]
    )
    beta0, *_ = np.linalg.lstsq(X0, y, rcond=None)
    slope = float(beta0[-1])
    return {"xi": xi, "sse0": sse0, "sse1": sse1, "f": fstat, "p": pvalue,
            "slope": slope, "n": n, "k1": k1}


def bootstrap_series(y, x, seed_ids, rng):
    uniq = np.unique(seed_ids)
    idx = {u: np.flatnonzero(seed_ids == u) for u in uniq}
    slopes, xis, sig = [], [], 0
    for _ in range(BOOTSTRAP):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        take = np.concatenate([idx[u] for u in draw])
        # seed labels must stay unique per drawn panel member
        lab = np.repeat(np.arange(len(draw)), [len(idx[u]) for u in draw])
        st = hinge_stats(y[take], x[take], lab)
        slopes.append(st["slope"])
        xis.append(st["xi"])
        if np.isfinite(st["p"]) and st["p"] < 0.05:
            sig += 1
    return np.array(slopes), np.array(xis), sig / BOOTSTRAP


def threshold_table(rows, rng):
    results = []
    for outcome in ("welfare_loss_sum", "explored_pct", "pct_knowledge_harvested"):
        for leg in (2, 6):
            for vis in (True, False):
                for entry in (0.0, 0.1):
                    x, y, seed = panel_for(rows, outcome, leg, vis, entry)
                    # complete panel only: 5 visibility weights x 20 seeds
                    if len(y) != 100 or len(np.unique(x)) != 5:
                        continue
                    st = hinge_stats(y, x, seed)
                    slopes, xis, sigshare = bootstrap_series(y, x, seed, rng)
                    lo, hi = np.percentile(xis, [2.5, 97.5])
                    results.append(
                        {
                            "outcome": outcome,
                            "number_gaussians": leg,
                            "use_visibility_reward": vis,
                            "agent_generation_time": entry,
                            "n_obs": st["n"],
                            "slope_per_unit_visibility": st["slope"],
                            "slope_boot_lo": float(np.percentile(slopes, 2.5)),
                            "slope_boot_hi": float(np.percentile(slopes, 97.5)),
                            "xi_star": st["xi"],
                            "f_stat": st["f"],
                            "p_value": st["p"],
                            "xi_boot_lo": float(lo),
                            "xi_boot_hi": float(hi),
                            "boot_sig_share": sigshare,
                            "collapse_detected": bool(
                                np.isfinite(st["p"])
                                and st["p"] < 0.05
                                and lo > 0.1
                                and hi < 0.9
                                and sigshare >= 0.5
                            ),
                        }
                    )
    return results


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def style_for(leg, vis):
    color = "#1f77b4" if vis else "#d62728"
    linestyle = "-" if leg == 2 else "--"
    label = f"mg{leg}, vis {'on' if vis else 'off'}"
    return color, linestyle, label


def plot_welfare_curves(summary):
    panels = [
        ("welfare_loss_sum", "total ideal-merit loss (sum over agents)"),
        ("ideal_merit_mean", "mean ideal_Merit (raw, always <= 0)"),
        ("pct_knowledge_harvested", "share of knowledge harvested"),
        ("explored_pct", "explored share of landscape"),
        ("top10_prestige_share", "top-10% prestige share"),
        ("corr_ideal_final", "wedge: Corr(prestige, ideal merit), final"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    for i, (ax, (name, title)) in enumerate(zip(axes.ravel(), panels)):
        for leg in (2, 6):
            for vis in (True, False):
                for entry in (0.0, 0.1):
                    pts = [
                        r for r in summary
                        if r["number_gaussians"] == leg
                        and r["use_visibility_reward"] == vis
                        and r["agent_generation_time"] == entry
                    ]
                    if not pts:
                        continue
                    pts.sort(key=lambda r: r["visibility_weight"])
                    xs = [p["visibility_weight"] for p in pts]
                    ys = [p[f"{name}_mean"] for p in pts]
                    ses = [p[f"{name}_se"] for p in pts]
                    color, ls, label = style_for(leg, vis)
                    ls = ls if entry == 0.0 else ":"
                    lbl = label + (", entry" if entry == 0.1 else "")
                    ax.plot(xs, ys, marker="o", ms=3.5, color=color,
                            linestyle=ls, label=lbl, lw=1.4)
                    ax.fill_between(
                        xs,
                        np.array(ys) - 1.96 * np.array(ses),
                        np.array(ys) + 1.96 * np.array(ses),
                        color=color,
                        alpha=0.10,
                        linewidth=0,
                    )
        ax.set_title(title, fontsize=10)
        if i >= 3:  # bottom row only; sharex hides the rest
            ax.set_xlabel("visibility weight (1-c)")
        ax.grid(alpha=0.3)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               frameon=False)
    fig.suptitle(
        "Welfare probe v1 - NOT CALIBRATED, model-internal only "
        "(mean +/- 1.96 SE over 20 seeds)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    path = FIG_DIR / "welfare_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_phase_diagram(summary, loss_name="welfare_loss_mean"):
    """Heatmap grid: rows = reward setting x entry, cols = visibility weight."""
    legs, vis_opts, entries = (2, 6), (True, False), (0.0, 0.1)
    xs = sorted({r["visibility_weight"] for r in summary})
    row_keys = [(v, e) for v in vis_opts for e in entries]
    row_labels = [
        f"vis {'on' if v else 'off'}, entry {e:g}" for v, e in row_keys
    ]
    all_vals = [r[f"{loss_name}_mean"] for r in summary]
    vmin, vmax = min(all_vals), max(all_vals)
    span = (vmax - vmin) or 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    im = None
    for ax, leg in zip(axes, legs):
        arr = np.full((len(row_keys), len(xs)), np.nan)
        for ri, (vis, entry) in enumerate(row_keys):
            for j, xw in enumerate(xs):
                hit = [
                    r for r in summary
                    if r["number_gaussians"] == leg
                    and r["use_visibility_reward"] == vis
                    and r["agent_generation_time"] == entry
                    and abs(r["visibility_weight"] - xw) < 1e-9
                ]
                if hit:
                    arr[ri, j] = hit[0][f"{loss_name}_mean"]
        im = ax.imshow(
            np.ma.masked_invalid(arr), cmap="magma",
            vmin=vmin, vmax=vmax, aspect="auto",
        )
        for ri in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if not np.isfinite(arr[ri, j]):
                    continue
                frac = (arr[ri, j] - vmin) / span
                ax.text(
                    j, ri, f"{arr[ri, j]:.2e}", ha="center", va="center",
                    fontsize=7.5, color="black" if frac > 0.55 else "white",
                )
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([f"{v:.1f}" for v in xs])
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8.5)
        ax.set_xlabel("visibility weight (1-c)")
        ax.set_title(f"landscape: {leg} bumps (std 5)", fontsize=10)
        ax.set_xticks(np.arange(-0.5, len(xs), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(
        im, ax=axes, fraction=0.03, pad=0.02,
        label="mean ideal-merit loss per agent (terminal step)",
    )
    fig.suptitle(
        "Phase diagram v1 - NOT CALIBRATED, model-internal only "
        "(cell = mean over 20 seeds; blank = not yet run)",
        fontsize=12,
        y=0.98,
    )
    # no subplots_adjust here: fig.colorbar() has already positioned the
    # axes, and a later adjust re-lays them out under the colorbar, which
    # paints over the rightmost column's annotations.
    path = FIG_DIR / "phase_diagram.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_threshold(rows, table):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    ax = axes[0]
    for leg in (2, 6):
        for vis in (True, False):
            pts = [
                r for r in table
                if r["outcome"] == "welfare_loss_sum"
                and r["number_gaussians"] == leg
                and r["use_visibility_reward"] == vis
                and r["agent_generation_time"] == 0.0
            ]
            if not pts:
                continue
            p = pts[0]
            color, ls, _ = style_for(leg, vis)
            # seed-level scatter, entry = 0
            mask = (
                (np.array([r["_number_gaussians"] for r in rows]) == leg)
                & (np.array([r["_use_visibility_reward"] for r in rows]) == vis)
                & (np.array([r["_agent_generation_time"] for r in rows]) == 0.0)
            )
            xs = np.array(
                [r["_visibility_weight"] for r, m in zip(rows, mask) if m],
                dtype=float,
            )
            ys = np.array(
                [outcome_value(r, "welfare_loss_sum")
                 for r, m in zip(rows, mask) if m],
                dtype=float,
            )
            order = np.argsort(xs)
            ax.scatter(xs[order], ys[order], s=14, color=color, alpha=0.45)
            grid = np.linspace(0.1, 0.9, 50)
            b_disp, a_disp = np.polyfit(xs[order], ys[order], 1)
            ax.plot(grid, a_disp + b_disp * grid, color=color, ls=ls,
                    label=f"mg{leg}, vis {'on' if vis else 'off'} "
                          f"(within-seed slope {p['slope_per_unit_visibility']:.2e}, "
                          f"p={p['p_value']:.3g})")
    ax.set_xlabel("visibility weight $(1-c)$")
    ax.set_ylabel("total ideal-merit loss (terminal step)")
    ax.set_title("Seed-level fit, entry = 0", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    detected = [r for r in table if r["collapse_detected"]]
    if detected:
        for i, r in enumerate(detected):
            ax.barh(
                i,
                r["xi_boot_hi"] - r["xi_boot_lo"],
                left=r["xi_boot_lo"],
                height=0.6,
                color="#2ca02c",
                alpha=0.7,
            )
            ax.plot(r["xi_star"], i, "k|", ms=18, mew=2)
        ax.set_yticks(range(len(detected)))
        ax.set_yticklabels(
            [f"{r['outcome']}, mg{r['number_gaussians']}, "
             f"vis {'on' if r['use_visibility_reward'] else 'off'}, "
             f"entry {r['agent_generation_time']}" for r in detected],
            fontsize=8,
        )
        ax.set_title("Bootstrap 95% CI of the breakpoint", fontsize=10)
    else:
        ax.text(
            0.5, 0.5,
            "no series shows a significant interior break\n"
            "(F test p < 0.05 with a bootstrap CI inside [0.1, 0.9])\n"
            "=> no collapse detected in range",
            ha="center", va="center", fontsize=11,
            transform=ax.transAxes,
        )
        ax.set_title("Breakpoint search", fontsize=10)
    ax.set_xlim(0.1, 0.9)
    ax.set_xlabel("visibility weight $(1-c)$")
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle(
        "Collapse-threshold probe v1 - NOT CALIBRATED, model-internal only "
        f"({BOOTSTRAP} bootstrap resamples over 20 seeds)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = FIG_DIR / "collapse_threshold.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    rows, files = load_rows()
    if not rows:
        print("no cell CSVs found under", OUT_DIR)
        return 1
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary = summarise(rows)
    write_csv(OUT_DIR / "summary_cell.csv", summary)

    rng = np.random.default_rng(20261113)
    table = threshold_table(rows, rng)
    write_csv(OUT_DIR / "threshold_estimate.csv", table)

    figs = [
        plot_welfare_curves(summary),
        plot_phase_diagram(summary),
        plot_threshold(rows, table),
    ]

    n_expected = 40
    n_cells = len(summary)
    n_runs = len(rows)
    print(f"cells found: {n_cells}/{n_expected} | runs: {n_runs} "
          f"(expected 800) | csv files: {len(files)}")
    print("\n--- per-cell means (welfare loss, explored, wedge) ---")
    for r in summary:
        print(
            f"  {r['tag']:34s} n={r['n_seeds']:2d} "
            f"loss={r['welfare_loss_sum_mean']:.4e} "
            f"expl={r['explored_pct_mean']:.3f} "
            f"harv={r['pct_knowledge_harvested_mean']:.3f} "
            f"top10={r['top10_prestige_share_mean']:.3f} "
            f"corI={r['corr_ideal_final_mean']:.3f} "
            f"corL={r['corr_local_final_mean']:.3f}"
        )
    print("\n--- threshold table ---")
    for r in table:
        print(
            f"  {r['outcome']:26s} mg{r['number_gaussians']} "
            f"vis {'on ' if r['use_visibility_reward'] else 'off'} "
            f"entry {r['agent_generation_time']} | "
            f"slope={r['slope_per_unit_visibility']:+.3e} "
            f"[{r['slope_boot_lo']:+.3e}, {r['slope_boot_hi']:+.3e}] | "
            f"xi*={r['xi_star']} p={r['p_value']:.3g} "
            f"CI=[{r['xi_boot_lo']:.2f},{r['xi_boot_hi']:.2f}] "
            f"sig={r['boot_sig_share']:.2f} "
            f"COLLAPSE={r['collapse_detected']}"
        )
    print("\nfigures:")
    for p in figs:
        print("  ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
