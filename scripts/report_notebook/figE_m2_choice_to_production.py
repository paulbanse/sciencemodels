"""F10a - M2 choice-to-production dashboard (Section 6, Part 1).

Fig 15 of the paper (``F10a_m2_choice_to_production``), re-based on the
ADR-0003 clock.

Inputs (read-only), ``data_root`` defaults to the ADR-0003 Path B arm::

    data/path_b_adr0003/cells/pb_g-*_lam-*_pull-*.csv
        18 cells = dyadic_lambda {0, 0.5, 1} x dyadic_move_pull {0, 1}
        x g {0.125, 0.25, 0.5} x 20 seeds = 360 runs, mgstd5, logit +
        equal_split ONLY, H = 200, fixed_span, tau_max = 80.
    data/path_b_v2/cells/cell_lam*_pull*.csv   (pre-ADR, selectable)

Four panels (every number computed here from the raw CSVs):

  (a) choice layer      : active declared links ``links_tm`` vs partner-choice
                          weight lambda, one line per movement pull;
  (b) production layer   : co-harvest team size ``team_tm`` vs lambda, one
                          line per pull, with the M1 reference and the
                          unfitted Klann benchmark;
  (c) signal alignment   : time-mean ``c_ideal_tm`` vs movement pull, one line
                          per lambda -- the pull reduces alignment at every
                          lambda, least at lambda = 1;
  (d) lambda x pull cells: team size for the six (lambda, pull) cells with
                          95% CIs (bars) plus the three g pins as markers.

Two panels of the pre-ADR figure could not survive the schema change; both are
replaced and disclosed on the canvas:

  * panel (d) was a 2 x 2 formulation x credit_sharing robustness cross.  The
    ADR-0003 Path B archive is the flagship configuration ONLY (logit +
    equal_split, one formulation x one credit level), so there are no cells to
    draw.  The substitute is the lambda x pull cell panel with CIs, with the g
    pins overlaid: it answers a robustness question the archive CAN answer
    (does the pull effect hold at every entry rate and every lambda?) and it
    keeps the per-cell CIs the old panel showed;
  * the ``rep_tm`` strip is dropped: the archive carries ``repeated_tm``,
    whose cell means span a narrow band (0.81-0.92) laid out mainly along the
    g axis rather than along lambda or pull, so the old strip would have
    rendered twelve overlapping markers with no lambda/pull ordering.  Its
    cell-mean range is reported in the footnote instead of plotted.

M1 (dyadic = off) reference: the ADR-0003 Path B arm has NO dyadic-off
baseline of its own, so the reference is derived from the welfare arm at the
matching pins -- ``data/welfare_adr0003`` at curiosity 0.5, mgstd5, visibility
reward on, fixed_span, H = 200, leg 2, dyadic lambda = 0 / pull = 0 -- pooled
over the same g grid (60 runs).  It is a M1 reference, NOT an M2 cell: the
welfare arm has no dyads at all (0.00 declared links), while the Path B
lambda = 0 arm still forms them (2.44 links at pull = 0).  Lambda = 1 is ALWAYS
labelled ``closeness x local merit``.

Provenance: ZERO hardcoded model numbers -- every plotted value, CI and
annotation is computed here from the raw CSVs and written into
``F10a_m2_choice_to_production_data.csv``.  The commission anchors (team size
lambda = 1, pull 1 minus pull 0 = -4.33; links 2.44 -> 15.34 with lambda at
pull 0) are recomputed and reported against the computed value.

Run:  MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
          scripts/report_notebook/figE_m2_choice_to_production.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from diagram_templates import PALETTE, HOUSE_STYLE, _cosmetics, write_figure_data
import visual_qa  # noqa: E402
from fig_F6_v2 import _place_text, audit_text_over_data  # noqa: E402  (read-only)

REPO = _HERE.parent.parent
FNAME = "F10a_m2_choice_to_production"
DEFAULT_ROOT = REPO / "data" / "path_b_adr0003"
PRE_ROOT = REPO / "data" / "path_b_v2"

#: unfitted Klann empirical benchmarks (diagnostic reference lines ONLY)
KLANN_TEAM_SIZE = 2.65
KLANN_REPEATED_SHARE = 0.34
BENCH_LABEL = "Empirical benchmark (unfitted)"
Z95 = 1.96

PULL_COLOR = {0.0: PALETTE["purple"], 0.5: PALETTE["orange"], 1.0: PALETTE["pink"]}
LAM_COLOR = {0.0: PALETTE["blue"], 0.5: PALETTE["orange"], 1.0: PALETTE["green"]}
M1_COLOR = PALETTE["gray"]
LAM_LABEL = {0.0: "λ = 0 (prestige matching)", 0.5: "λ = 0.5",
             1.0: "λ = 1 (closeness × local merit)"}
LAM_SHORT = {0.0: "λ = 0", 0.5: "λ = 0.5", 1.0: "λ = 1"}
M_REF = "reference (M1 dyadic-off baseline, Klann benchmark, archive pins)"


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _mean_ci(vals) -> tuple[float, float, float, int]:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    _require(n > 1, "CI needs at least two observations")
    m = float(v.mean())
    half = Z95 * float(v.std(ddof=1)) / np.sqrt(n)
    return m, m - half, m + half, n


def data_root_path(data_root=None) -> Path:
    root = DEFAULT_ROOT if data_root is None else Path(data_root)
    if not root.is_absolute():
        root = REPO / root
    if not list(root.glob("cells/pb_*.csv")) and list(
            (root / "cells").glob("cell_lam*.csv")):
        return root / "cells"
    return root


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def load_runs(data_root=None) -> tuple[pd.DataFrame, dict]:
    """Canonical Path B run table (one row per run) plus the archive pins."""
    root = data_root_path(data_root)
    if root.name == "cells" and not list(root.glob("pb_*.csv")):
        files = sorted(root.glob("cell_lam*.csv"))          # pre-ADR
        frames = []
        for f in files:
            d = pd.read_csv(f)
            _require(len(d) == 20, f"{f.name}: 20 seed rows expected")
            frames.append(d)
        raw = pd.concat(frames, ignore_index=True)
        layout = "pre-ADR cells"
    else:
        files = sorted((root / "cells").glob("pb_*.csv"))
        _require(bool(files), f"no pb_*.csv cells under {root}")
        frames = []
        for f in files:
            d = pd.read_csv(f)
            _require(len(d) == 20, f"{f.name}: 20 seed rows expected")
            frames.append(d)
        raw = pd.concat(frames, ignore_index=True)
        layout = "ADR-0003 cells"

    out = pd.DataFrame({
        "lam": raw["dyadic_lambda"].astype(float),
        "pull": raw["dyadic_move_pull"].astype(float),
        "g": raw["g"].astype(float) if "g" in raw.columns
        else raw["agent_generation_time"].astype(float),
        "links_tm": raw["links_tm"].astype(float),
        "team_tm": raw["team_tm"].astype(float),
        "c_ideal_tm": raw["c_ideal_tm"].astype(float),
        "seed": raw["seed"].astype(int) if "seed" in raw.columns
        else raw["agent_seed"].astype(int),
        "version": layout,
    })
    if "repeated_tm" in raw.columns:
        out["repeated_tm"] = raw["repeated_tm"].astype(float)
    elif "rep_tm" in raw.columns:
        out["repeated_tm"] = raw["rep_tm"].astype(float)

    _require(out[["links_tm", "team_tm", "c_ideal_tm"]].notna().all().all(),
             "links_tm / team_tm / c_ideal_tm must be complete")
    _require(out.groupby(["lam", "pull", "g"]).size().max() >= 1,
             "empty cell in the Path B grid")
    _sizes = out.groupby(["lam", "pull", "g"]).size()
    _require(bool((_sizes % 20 == 0).all()) and _sizes.nunique() == 1,
             "every (lambda, pull, g) cell must hold whole 20-seed blocks "
             f"of equal size (got {sorted(_sizes.unique())[:4]})")
    pins = {
        "root": str(root.relative_to(REPO)) if root.is_relative_to(REPO) else str(root),
        "layout": layout,
        "cells": int(out.groupby(["lam", "pull", "g"]).ngroups),
        "runs": int(len(out)),
        "lams": sorted(out["lam"].unique().tolist()),
        "pulls": sorted(out["pull"].unique().tolist()),
        "gs": sorted(out["g"].unique().tolist()),
        "n_per_cell": int(out.groupby(["lam", "pull", "g"]).size().max()),
        "formulations": sorted(raw["formulation"].unique().tolist())
        if "formulation" in raw.columns else [],
        "credit": sorted(raw["credit_sharing_axis"].unique().tolist())
        if "credit_sharing_axis" in raw.columns else [],
        "landscapes": sorted(raw["landscape"].unique().tolist())
        if "landscape" in raw.columns else [],
    }
    return out, pins


def m1_reference(data_root=None) -> dict | None:
    """M1 (no dyads) reference at the Path B pins, from the archived arms.

    ADR-0003: the welfare arm (dyadic lambda = 0, pull = 0, visibility on,
    curiosity 0.5, leg 2) pooled over the same g grid.  Pre-ADR: the archived
    ``path_b_v1/baseline_off.csv`` table.  Returns ``None`` if neither exists.
    """
    root = data_root_path(data_root)
    # archives are siblings under data/: a <arm>/cells path lives two levels down
    base = root.parent.parent if root.name == "cells" else root.parent
    welfare = base / "welfare_adr0003" / "cells"
    if welfare.exists():
        frames = [pd.read_csv(f) for f in sorted(welfare.glob("*.csv"))]
        w = pd.concat(frames, ignore_index=True)
        w = w[(w["curiosity"] == 0.5) & (w["use_visibility_reward"] == True)  # noqa: E712
              & (w["landscape"] == "mgstd5") & (w["leg"] == 2)]
        if len(w):
            _require((w["dyadic_lambda"] == 0.0).all(),
                     "welfare M1 reference must be dyadic lambda = 0")
            _require((w["dyadic_move_pull"] == 0.0).all(),
                     "welfare M1 reference must be pull = 0")
            return {"source": str(welfare.relative_to(REPO)) + "/*.csv",
                    "n": int(len(w)),
                    "links_tm": float(w["links_tm"].mean()),
                    "team_tm": float(w["team_tm"].mean()),
                    "c_ideal_tm": float(w["c_ideal_tm"].mean()),
                    "gs": sorted(w["g"].unique().tolist()),
                    "kind": "welfare arm (dyadic off) at c = 0.5, visibility on"}
    off = base / "path_b_v1" / "baseline_off.csv"
    if off.exists():
        d = pd.read_csv(off)
        return {"source": str(off.relative_to(REPO)), "n": int(len(d)),
                "links_tm": float(d["links_tm"].mean()),
                "team_tm": float(d["team_tm"].mean()),
                "c_ideal_tm": float(d["c_ideal_tm"].mean()),
                "gs": sorted(d["agent_generation_time"].unique().tolist()),
                "kind": "archived dyadic-off baseline"}
    return None


# --------------------------------------------------------------- sidecar ---
def compute_table(data_root=None) -> tuple[pd.DataFrame, dict, dict]:
    runs, pins = load_runs(data_root)
    m1 = m1_reference(data_root)
    src = f"{pins['root']}/cells/*.csv"
    rows: list[dict] = []

    def add(panel, role, series, metric, x, *, vals=None, value=None,
            note="", extra=None):
        if vals is not None:
            v, lo, hi, n = _mean_ci(vals)
        else:
            v, lo, hi, n = float(value), float("nan"), float("nan"), 0
        rows.append(dict(panel=panel, role=role, series=series, metric=metric,
                         x=x, value=v, ci_lo=lo, ci_hi=hi, n=n, note=note,
                         **(extra or {})))

    lams, pulls = pins["lams"], pins["pulls"]
    #: extra strata inside a (lambda, pull) group (landscapes, formulation x
    #: credit in the pre-ADR archive): every (lambda, pull, g) cell still holds
    #: exactly one 20-seed block
    n_strata = pins["runs"] // (20 * len(lams) * len(pulls) * len(pins["gs"]))
    _require(n_strata >= 1, "run count is not a multiple of the 20-seed blocks")

    # ---- (a)/(b): one line per pull over lambda --------------------------
    for pull in pulls:
        for lam in lams:
            sel = runs[(runs["pull"] == pull) & (runs["lam"] == lam)]
            _require(len(sel) == 20 * len(pins["gs"]) * n_strata,
                     f"pull={pull} lam={lam}: {len(sel)} runs")
            series = f"pull = {pull:g}"
            add("A", "point", series, "links_tm", lam, vals=sel["links_tm"],
                note=f"pooled over {len(pins['gs'])} g pins x 20 seeds = "
                     f"{len(sel)} runs")
            add("B", "point", series, "team_tm", lam, vals=sel["team_tm"],
                note=f"pooled over {len(pins['gs'])} g pins x 20 seeds")

    # ---- (c): one line per lambda over pull ------------------------------
    for lam in lams:
        for pull in pulls:
            sel = runs[(runs["lam"] == lam) & (runs["pull"] == pull)]
            add("C", "point", LAM_LABEL[lam], "c_ideal_tm", pull,
                vals=sel["c_ideal_tm"],
                note=f"time-mean alignment, {len(sel)} runs")

    # ---- (d): lambda x pull cells with CIs + the g pins ------------------
    for lam in lams:
        for pull in pulls:
            sel = runs[(runs["lam"] == lam) & (runs["pull"] == pull)]
            add("D", "point", f"λ = {lam:g}, pull = {pull:g}", "team_tm",
                lam + (0.18 if pull == max(pulls) else -0.18) * 0.9,
                vals=sel["team_tm"],
                note="lambda x pull cell, pooled over the g pins")
            for g in pins["gs"]:
                sub = sel[sel["g"] == g]
                add("D", "point", f"λ = {lam:g}, pull = {pull:g}, g = {g:g}",
                    "team_tm", lam + (0.18 if pull == max(pulls) else -0.18) * 0.9,
                    vals=sub["team_tm"], note="per-g pin of the same cell",
                    extra={"g": float(g)})

    # ---- derived contrasts (annotations) --------------------------------
    def cell(lam, pull, col):
        return runs[(runs["lam"] == lam) & (runs["pull"] == pull)][col]

    lo_pull, hi_pull = min(pulls), max(pulls)
    ann = {}
    for lam in lams:
        d = cell(lam, hi_pull, "team_tm").to_numpy() - cell(lam, lo_pull,
                                                            "team_tm").to_numpy()
        # pairing by g and seed: the archive repeats the same draws per cell
        piv = runs[runs["lam"] == lam].pivot_table(index=["g", "seed"],
                                                   columns="pull",
                                                   values="team_tm")
        d = (piv[hi_pull] - piv[lo_pull]).to_numpy()
        v, lo, hi, n = _mean_ci(d)
        ann[f"team_pull_effect_lam{lam:g}"] = v
        add("D", "annotation", f"λ = {lam:g}", "team_tm_pull_effect",
            lam, value=v, note=f"team_tm at pull {hi_pull:g} minus pull "
                               f"{lo_pull:g}, paired by (g, seed); 95% CI "
                               f"[{lo:+.3f}, {hi:+.3f}] over {n} pairs")
    add("A", "annotation", "λ = 0", "links_lambda_step",
        float(lams[0]), value=float(cell(lams[0], lo_pull, "links_tm").mean()),
        note="links_tm at the lowest lambda, lowest pull")
    add("A", "annotation", "λ = 1", "links_lambda_step",
        float(lams[-1]), value=float(cell(lams[-1], lo_pull, "links_tm").mean()),
        note="links_tm at the highest lambda, lowest pull")
    for lam in lams:
        add("C", "annotation", LAM_LABEL[lam], "c_tm_at_high_pull",
            hi_pull, value=float(cell(lam, hi_pull, "c_ideal_tm").mean()),
            note="time-mean alignment at the highest pull")
    if "repeated_tm" in runs.columns:
        cell_means = runs.groupby(["lam", "pull", "g"])["repeated_tm"].mean()
        for lab, val in (("min", float(cell_means.min())),
                         ("max", float(cell_means.max()))):
            add("ref", "annotation", f"repeated_tm cell mean {lab}",
                "repeated_tm_cell_range", np.nan, value=val,
                note=f"repeated_tm cell means span {cell_means.min():.3f}-"
                     f"{cell_means.max():.3f} over the "
                     f"{len(cell_means)} cells (per-run range "
                     f"{runs['repeated_tm'].min():.3f}-"
                     f"{runs['repeated_tm'].max():.3f})")
    for pull in pulls:
        v = float(runs[runs["pull"] == pull]["team_tm"].mean())
        add("B", "annotation", f"pull = {pull:g}", "team_tm_pull_mean", pull,
            value=v, note="team_tm pooled over lambda and g")

    # ---- references ------------------------------------------------------
    if m1 is not None:
        for metric in ("links_tm", "team_tm", "c_ideal_tm"):
            rows.append(dict(panel="ref", role="reference",
                             series=f"M1 dyadic off ({metric})", metric=metric,
                             x=np.nan, value=float(m1[metric]), ci_lo=np.nan,
                             ci_hi=np.nan, n=int(m1["n"]), note=m1["kind"],
                             source_file=m1["source"]))
    rows.append(dict(panel="ref", role="reference", series=BENCH_LABEL,
                     metric="team_tm", x=np.nan, value=KLANN_TEAM_SIZE,
                     ci_lo=np.nan, ci_hi=np.nan, n=0, note=BENCH_LABEL,
                     source_file="Klann et al. team-size benchmark (unfitted)"))
    for cat, val, note in (
            ("cells", pins["cells"], "distinct (lambda, pull, g)"),
            ("runs", pins["runs"], "one row per run"),
            ("n_per_cell", pins["n_per_cell"], "runs per cell"),
            ("lambda_min", lams[0], "lowest partner-choice weight"),
            ("lambda_max", lams[-1], "highest partner-choice weight"),
            ("pull_min", lo_pull, "lowest movement pull"),
            ("pull_max", hi_pull, "highest movement pull"),
            ("g_min", pins["gs"][0], "entry-rate grid, low pin"),
            ("g_max", pins["gs"][-1], "entry-rate grid, high pin")):
        rows.append(dict(panel="ref", role="definitional constant", series=cat,
                         metric=M_REF, x=np.nan, value=float(val), ci_lo=np.nan,
                         ci_hi=np.nan, n=np.nan, note=note,
                         source_file=src))

    table = pd.DataFrame(rows)
    # the per-g pins are drawn as markers WITHOUT error bars: collapse their CI
    # columns so the rendered-artist contract compares like with like
    marker_rows = table["series"].astype(str).str.contains(", g = ")
    table.loc[marker_rows, "ci_lo"] = table.loc[marker_rows, "value"]
    table.loc[marker_rows, "ci_hi"] = table.loc[marker_rows, "value"]
    if "source_file" not in table.columns:
        table["source_file"] = src
    table["source_file"] = table["source_file"].fillna(src)
    return table, pins, (m1 or {})


# ---------------------------------------------------------------- figure ---
def _errband(ax, g: pd.DataFrame, *, color, ls, marker, label=None) -> None:
    x = g["x"].to_numpy(float)
    y = g["value"].to_numpy(float)
    lo = g["ci_lo"].to_numpy(float)
    hi = g["ci_hi"].to_numpy(float)
    ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), color=color,
                linestyle=ls, marker=marker, markersize=5, linewidth=1.7,
                capsize=2.5, elinewidth=0.9, label=label, zorder=4)


def build(outdir: str = "output/report_notebook_v1", data_root=None) -> dict:
    out = Path(outdir)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{FNAME}.png"

    table, pins, m1 = compute_table(data_root)
    lams, pulls = pins["lams"], pins["pulls"]

    def t(panel, role, metric, series=None, x=None):
        g = table[(table["panel"] == panel) & (table["role"] == role)
                  & (table["metric"] == metric)]
        if series is not None:
            g = g[g["series"] == series]
        if x is not None:
            g = g[g["x"] == x]
        return g

    fig = plt.figure(figsize=(11.8, 8.2))
    gsp = fig.add_gridspec(2, 2, left=0.095, right=0.985, top=0.885,
                           bottom=0.255, hspace=0.50, wspace=0.26)
    axA = fig.add_subplot(gsp[0, 0])
    axB = fig.add_subplot(gsp[0, 1])
    axC = fig.add_subplot(gsp[1, 0])
    axD = fig.add_subplot(gsp[1, 1])

    # ---- A: choice layer -------------------------------------------------
    for pull in pulls:
        _errband(axA, t("A", "point", "links_tm", f"pull = {pull:g}"),
                 color=PULL_COLOR.get(pull, PALETTE["gray"]), ls="-",
                 marker="o", label=f"pull = {pull:g}")
    if m1:
        axA.axhline(m1["links_tm"], color=M1_COLOR, linestyle="--", linewidth=1.2,
                    zorder=2)
    lan = {float(r["x"]): float(r["value"]) for _, r in
           t("A", "annotation", "links_lambda_step").iterrows()}
    _cosmetics(axA, HOUSE_STYLE,
               "A · Choice layer: active declared links (links_tm)",
               "partner choice  λ = dyadic_lambda",
               "links_tm (time mean · run mean)")
    lo, hi = axA.get_ylim()
    axA.set_ylim(0.0, max(hi, (lan.get(lams[-1], 0.0))) * 1.28)
    _place_text(axA, (
        f"λ {lams[0]:g} → {lams[-1]:g} at pull {pulls[0]:g}:\n"
        f"{lan.get(lams[0], float('nan')):.2f} → {lan.get(lams[-1], float('nan')):.2f}"
        f" links\nM1 (dyadic off) has "
        f"{m1.get('links_tm', float('nan')):.2f}"),
        ((0.98, 0.97, "right", "top"), (0.02, 0.97, "left", "top")))

    # ---- B: production layer --------------------------------------------
    for pull in pulls:
        _errband(axB, t("B", "point", "team_tm", f"pull = {pull:g}"),
                 color=PULL_COLOR.get(pull, PALETTE["gray"]), ls="-",
                 marker="o", label=f"pull = {pull:g}")
    axB.axhline(KLANN_TEAM_SIZE, color=M1_COLOR, linestyle=":", linewidth=1.3,
                zorder=2)
    if m1:
        axB.axhline(m1["team_tm"], color=M1_COLOR, linestyle="--", linewidth=1.2,
                    zorder=2)
    _cosmetics(axB, HOUSE_STYLE,
               "B · Production layer: co-harvest team size (team_tm)",
               "partner choice  λ = dyadic_lambda",
               "team_tm (time mean · run mean)")
    lo, hi = axB.get_ylim()
    axB.set_ylim(min(lo, 0.0), hi * 1.32)
    kf = (KLANN_TEAM_SIZE - axB.get_ylim()[0]) / (axB.get_ylim()[1] - axB.get_ylim()[0])
    axB.annotate(f"{KLANN_TEAM_SIZE:.2f} Klann team size ({BENCH_LABEL})",
                 xy=(0.0, kf), xycoords="axes fraction", xytext=(2, 3),
                 textcoords="offset points", fontsize=6.6,
                 color=PALETTE["gray"], va="bottom", zorder=6)
    if m1:
        mf = (m1["team_tm"] - axB.get_ylim()[0]) / (axB.get_ylim()[1] - axB.get_ylim()[0])
        axB.annotate(f"M1 (dyadic off) {m1['team_tm']:.2f}",
                     xy=(0.0, mf), xycoords="axes fraction", xytext=(2, -3),
                     textcoords="offset points", fontsize=6.6,
                     color=PALETTE["gray"], va="top", zorder=6)
    pm = {float(r["x"]): float(r["value"]) for _, r in
          t("B", "annotation", "team_tm_pull_mean").iterrows()}
    _place_text(axB, (
        "team size by pull (pooled):\n"
        + "  ".join(f"{p:g}: {pm[p]:.2f}" for p in pulls)),
        ((0.98, 0.02, "right", "bottom"), (0.98, 0.14, "right", "bottom")))
    axB.legend(loc="upper center", ncol=len(pulls), frameon=False,
               fontsize=HOUSE_STYLE["tick_size"] - 1.5, handlelength=2.0,
               labelspacing=0.3, borderaxespad=0.1)

    # ---- C: signal alignment vs pull -------------------------------------
    for lam in lams:
        _errband(axC, t("C", "point", "c_ideal_tm", LAM_LABEL[lam]),
                 color=LAM_COLOR.get(lam, PALETTE["gray"]), ls="-", marker="o",
                 label=LAM_SHORT.get(lam, f"λ = {lam:g}"))
    if m1:
        axC.axhline(m1["c_ideal_tm"], color=M1_COLOR, linestyle="--",
                    linewidth=1.2, zorder=2)
    axC.axhline(0.0, color=PALETTE["gray_light"], linewidth=0.9, zorder=1)
    _cosmetics(axC, HOUSE_STYLE,
               "C · Signal alignment vs movement coupling",
               "movement coupling  dyadic_move_pull",
               "Corr(prestige, ideal merit) (time mean · run mean)")
    hi_pull = max(pulls)
    ch = {str(r["series"]): float(r["value"]) for _, r in
          t("C", "annotation", "c_tm_at_high_pull").iterrows()}
    _place_text(axC, (
        f"at pull {hi_pull:g}: "
        + "; ".join(f"{LAM_SHORT.get(l, f'λ={l:g}')} "
                    f"{ch.get(LAM_LABEL[l], float('nan')):+.3f}"
                    for l in lams)
        + (f"\nM1 (dyadic off) {m1['c_ideal_tm']:+.3f}" if m1 else "")),
        ((0.02, 0.02, "left", "bottom"), (0.98, 0.02, "right", "bottom"),
         (0.02, 0.14, "left", "bottom")))
    axC.set_xticks(pulls, [f"{p:g}" for p in pulls])
    axC.set_xlim(min(pulls) - 0.12, max(pulls) + 0.12)
    vv = table[(table["panel"] == "C") & (table["role"] == "point")]
    c_lo = float(min(vv["ci_lo"].min(), 0.0))
    c_hi = float(vv["ci_hi"].max())
    span_c = c_hi - c_lo
    axC.set_ylim(c_lo - 0.10 * span_c, c_hi + 0.62 * span_c)
    axC.legend(loc="upper left", ncol=len(lams), frameon=False,
               fontsize=HOUSE_STYLE["tick_size"] - 2, handlelength=1.6,
               labelspacing=0.3, borderaxespad=0.2)

    # ---- D: lambda x pull cells -----------------------------------------
    xs = np.arange(len(lams), dtype=float)
    width = 0.34
    for k, pull in enumerate(pulls):
        pos = xs + (k - 0.5) * width
        vals, los, his = [], [], []
        for lam in lams:
            r = t("D", "point", "team_tm", f"λ = {lam:g}, pull = {pull:g}").iloc[0]
            vals.append(float(r["value"]))
            los.append(float(r["ci_lo"]))
            his.append(float(r["ci_hi"]))
        axD.bar(pos, vals, width=width * 0.92,
                color=PULL_COLOR.get(pull, PALETTE["gray"]),
                edgecolor="white", zorder=3,
                label=f"pull = {pull:g} (pooled over g)")
        axD.errorbar(pos, vals,
                     yerr=[np.array(vals) - np.array(los),
                           np.array(his) - np.array(vals)],
                     fmt="none", ecolor="black", elinewidth=0.9, capsize=2.6,
                     zorder=4)
        # per-g markers: three hollow dots per bar, same hue
        for gi, g in enumerate(pins["gs"]):
            ys = [float(t("D", "point", "team_tm",
                          f"λ = {lam:g}, pull = {pull:g}, g = {g:g}").iloc[0]["value"])
                  for lam in lams]
            axD.plot(pos + (gi - 1) * 0.055, ys, linestyle="none", marker="o",
                     markersize=3.4, markerfacecolor="white",
                     markeredgecolor=PULL_COLOR.get(pull, PALETTE["gray"]),
                     markeredgewidth=0.9, zorder=5)
    axD.set_xticks(xs, [LAM_SHORT.get(l, f"λ = {l:g}") for l in lams])
    axD.set_xlim(-0.6, len(lams) - 0.4)
    vmax = float(t("D", "point", "team_tm")["ci_hi"].max())
    axD.set_ylim(0.0, vmax * 1.45)
    _cosmetics(axD, HOUSE_STYLE,
               "D · λ × pull cells: team size, with the g pins",
               "partner choice λ", "team_tm (time mean · run mean)")
    lpk = {float(r["x"]): float(r["value"]) for _, r in
           t("D", "annotation", "team_tm_pull_effect").iterrows()}
    _place_text(axD, (
        f"pull {max(pulls):g} − pull {min(pulls):g}: "
        + ", ".join(f"{LAM_SHORT.get(l, f'λ={l:g}')} {lpk.get(l, float('nan')):+.2f}"
                    for l in lams)
        + "\nhollow dots: the three g pins"),
        ((0.98, 0.97, "right", "top"), (0.02, 0.97, "left", "top")))
    axD.legend(handles=[
        Patch(facecolor=PULL_COLOR.get(p, PALETTE["gray"]), label=f"pull = {p:g}")
        for p in pulls
    ] + [Line2D([], [], linestyle="none", marker="o", markersize=3.4,
                markerfacecolor="white", markeredgecolor=PALETTE["gray"],
                label="g pin (per g)")],
        loc="upper left", fontsize=HOUSE_STYLE["tick_size"] - 2, frameon=False,
        handlelength=1.4, labelspacing=0.25, borderaxespad=0.2)

    # ---- frame + footnote ------------------------------------------------
    fig.text(0.5, 0.985,
             "F10a · How M2 works end-to-end: partner choice (λ) → movement "
             "coupling → teams, alignment, repetition\n"
             f"(ADR-0003 Path B: {len(lams)} λ × {len(pulls)} pull × "
             f"{len(pins['gs'])} g cells, mgstd5, logit + equal_split)",
             fontsize=HOUSE_STYLE["title_size"] + 1, ha="center", va="top")
    rng = table[table["metric"] == "repeated_tm_cell_range"]
    rt = (f" repeated_tm cell means span {rng['value'].min():.3f}-"
          f"{rng['value'].max():.3f} (a g gradient, no lambda/pull ordering)."
          if len(rng) else "")
    m1_txt = (f" M1 (dyadic = off) reference: {m1['source']} — {m1['kind']}, "
              f"{m1['n']} runs pooled over the same g grid; it is a reference, "
              "not an M2 cell (M1 forms no dyads, the Path B λ = 0 arm does)."
              if m1 else " No dyadic-off baseline available in this archive.")
    note_lines = [
        f"Sources (read-only): {pins['root']}/cells/*.csv — {pins['cells']} cells "
        f"× {pins['n_per_cell']} seeds = {pins['runs']} runs; λ in "
        f"{{{', '.join(f'{l:g}' for l in lams)}}}, pull in "
        f"{{{', '.join(f'{p:g}' for p in pulls)}}}, g in "
        f"{{{', '.join(f'{g:g}' for g in pins['gs'])}}}, "
        f"formulation/credit: {'/'.join(pins['formulations']) or 'n/a'} + "
        f"{'/'.join(pins['credit']) or 'n/a'}, landscape "
        f"{'/'.join(pins['landscapes']) or 'n/a'}, H = 200. Panels (a)-(c) pool "
        "over the g pins and 20 seeds; panel (d) shows the six λ × pull cells "
        "with the g pins as hollow dots. Error bars 95% CI = mean ± 1.96 SE.",
        "Schema substitutions (disclosed): the ADR-0003 Path B arm is the "
        "flagship configuration only, so the pre-ADR formulation × credit "
        "robustness panel has no cells and panel (d) replaces it with the "
        "λ × pull cells (with CIs) plus the g pins; the pre-ADR rep_tm strip is "
        "not plotted because repeated_tm has no lambda/pull ordering" + rt
        + m1_txt,
        "ADR-0003 clock: 1 step = 0.5 yr, entry age 25, tau_max = 80 (40-yr "
        "careers), gamma = 0.5**(1/10) = 0.933 (5-yr retention half-life). "
        "λ = 1 (closeness × local merit); λ = 0 (prestige matching). Klann team "
        f"size {KLANN_TEAM_SIZE:.2f} and repeated share {KLANN_REPEATED_SHARE:.2f}: "
        "Empirical benchmark (unfitted) — model-internal, not calibrated. Every "
        "plotted value, CI and annotation is computed from the raw CSVs and "
        f"stored in {FNAME}_data.csv.",
    ]
    note = "\n".join(textwrap.fill(t_, width=176) for t_ in note_lines)
    fig.text(0.01, 0.008, note, fontsize=HOUSE_STYLE["note_size"] - 1.0,
             color=PALETTE["gray"], va="bottom")

    # ---- gates -----------------------------------------------------------
    for ax, xl in ((axA, (-0.06, 1.06)), (axB, (-0.06, 1.06))):
        ax.set_xlim(*xl)
    for ax in (axA, axB, axC, axD):
        lo_, hi_ = ax.get_ylim()
        ax.set_yticks([t_ for t_ in ax.get_yticks()
                       if lo_ - 1e-12 <= t_ <= hi_ + 1e-12])
        xlo, xhi = ax.get_xlim()
        ax.set_xticks([t_ for t_ in ax.get_xticks()
                       if xlo - 1e-12 <= t_ <= xhi + 1e-12])
    n_g = len(pins["gs"])
    visual_qa.assert_figure_clean(fig, [
        {"ax": axA, "metric": "links_tm", "expect_lines": len(pulls),
         "points_per_line": len(lams), "sidecar": table,
         "filters": {"panel": "A", "role": "point"}},
        {"ax": axB, "metric": "team_tm", "expect_lines": len(pulls),
         "points_per_line": len(lams), "sidecar": table,
         "filters": {"panel": "B", "role": "point"}},
        {"ax": axC, "metric": "c_ideal_tm", "expect_lines": len(lams),
         "points_per_line": len(pulls), "sidecar": table,
         "filters": {"panel": "C", "role": "point"}},
        {"ax": axD, "metric": "team_tm", "expect_bars": len(lams) * len(pulls),
         "expect_lines": len(pulls), "points_per_line": len(lams),
         "sidecar": table, "filters": {"panel": "D", "role": "point"}},
    ])
    audit_text_over_data(fig, (axA, axB, axC, axD))

    fig.savefig(png, dpi=HOUSE_STYLE["dpi"])
    fig.savefig(png.with_suffix(".pdf"))
    plt.close(fig)

    csv = write_figure_data(table, png)
    return {"png": str(png), "pdf": str(png.with_suffix(".pdf")),
            "data_csv": str(csv), "sidecar_rows": len(table)}


if __name__ == "__main__":
    print(build())
