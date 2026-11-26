"""Independent verifier for F6_v2 (fig_F6_v2.py) on the ADR-0003 clock.

``check(outdir="output/report_notebook_v1", data_root=None, tol=1e-9)``
returns a list of failures -- empty means everything reconciles.

Independence: this module re-reads the RAW per-cell CSVs of
``data/fig1_xi_adr0003/cells`` with its own selectors and recomputes every
plotted statistic (mean + 95% CI) and every annotation number itself; it
never imports the builder's loaders.  It

  * asserts the input shape and the PINNED tier definitions (55 cells,
    20 seeds per cell, 11-point xi grid, exit_mode / g / horizon /
    dyadic lambda / pull per tier, mgstd5, visibility reward on,
    tau_max = 80),
  * cross-checks the plotted per-(tier, xi) means against a second,
    per-file re-read of the same cells (the ADR-0003 replacement for the
    pre-ADR ``tables/four_tier_xi_summary.csv`` identity),
  * reproduces the Figure-1 anchors quoted in
    ``output/reports/adr0003_remeasure_v1.md`` to 3 decimals,
  * reproduces the ideal-channel disclosure (run-level
    Corr(xi, c_ideal_tm) = +0.304 for M2 lambda=0, negative elsewhere),
  * rebuilds the figure in memory and re-runs
    ``visual_qa.assert_figure_clean`` on the RENDERED canvas, and
  * asserts the PNG / PDF / sidecar triplets are on disk and that the
    footnote names the new archive, the ADR-0003 pins and the
    "model-internal, not calibrated" status.

The pre-ADR archive is still verifiable: point ``data_root`` at a root that
holds ``bug0006_remeasure_v2/`` and the shape pins switch to that archive
(5-point xi grid, 500 runs, 25 cells).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import visual_qa  # noqa: E402

FNAME = "F6_v2_two_stage_proxy_failure"
DEFAULT_DATA_ROOT = REPO / "data" / "fig1_xi_adr0003"
LANDSCAPE = "mgstd5"
Z95 = 1.96
SEEDS_PER_CELL = 20

TIERS = ("M0", "M1_fixed", "M1_staggered", "M2_l0", "M2_l1")
M_HARVEST = "harvested knowledge share"
M_CORR = "Corr(prestige, ideal merit) (time mean)"
PANELS = {"a": M_HARVEST, "b": M_CORR}
COL = {"a": "harvested_fin", "b": "c_ideal_tm"}
ENTRY_ARMS = ("M1_fixed", "M2_l0", "M2_l1")
REPORT_HI_KNOT = 0.9

#: pinned tier definitions of the ADR-0003 ``fig1_xi`` arm
#: (scripts/adr0003_remeasure/run_adr0003_wave.py) -- asserted, not assumed
TIER_PIN = {
    "M0": dict(exit_mode="none", g=0.0, horizon=200, lam=0.0, pull=0.0),
    "M1_fixed": dict(exit_mode="fixed_span", g=0.25, horizon=400, lam=0.0,
                     pull=0.0),
    "M1_staggered": dict(exit_mode="age_hazard_staggered", g=0.25, horizon=400,
                         lam=0.0, pull=0.0),
    "M2_l0": dict(exit_mode="fixed_span", g=0.25, horizon=200, lam=0.0,
                  pull=1.0),
    "M2_l1": dict(exit_mode="fixed_span", g=0.25, horizon=200, lam=1.0,
                  pull=1.0),
}
TAU_MAX_PIN = 80

#: Figure-1 anchors quoted in output/reports/adr0003_remeasure_v1.md
#: (section "Figure 1 dense xi"), harvested share at xi = 0.1 / 0.5 / 0.9
REPORT_ANCHORS = {
    "M1_fixed": (0.829, 0.825, 0.493),
    "M1_staggered": (0.980, 0.979, 0.868),
    "M0": (0.665, 0.657, 0.327),
    "M2_l1": (0.661, 0.648, 0.296),
    "M2_l0": (0.663, 0.651, 0.329),
}
REPORT_M2L0_CORR = 0.304


def _mean_ci(v) -> tuple[float, float, float, int]:
    a = np.asarray(v, dtype=float)
    n = int(a.size)
    m = float(a.mean())
    h = Z95 * float(a.std(ddof=1)) / np.sqrt(n) if n > 1 else float("nan")
    return m, m - h, m + h, n


def _root(data_root) -> Path:
    return Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT


def _layout(root: Path) -> str:
    if (root / "cells").is_dir():
        return "adr0003"
    if (root / "bug0006_remeasure_v2").is_dir():
        return "pre_adr"
    raise AssertionError(f"{root}: no cells/ and no bug0006_remeasure_v2/")


def _knot(xis, target: float) -> float:
    return float(min(xis, key=lambda x: abs(float(x) - target)))


def _load_adr0003(root: Path):
    """Own selector over the per-cell CSVs -> (exp, runs, xis, fails)."""
    fails: list[str] = []
    files = sorted((root / "cells").glob("*.csv"))
    if len(files) != 55:
        fails.append(f"{root}/cells: {len(files)} files, expected 55")
    cols = None
    rows = []
    for f in files:
        d = pd.read_csv(f)
        cols = list(d.columns) if cols is None else cols
        if len(d) != SEEDS_PER_CELL:
            fails.append(f"{f.name}: {len(d)} rows != {SEEDS_PER_CELL}")
        for c in ("tier", "xi", "g", "exit_mode", "horizon", "dyadic_lambda",
                  "dyadic_move_pull", "landscape", "use_visibility_reward",
                  "tau_max", "curiosity", "seed", "harvested_fin",
                  "c_ideal_tm", "formulation", "credit_sharing"):
            if c not in d.columns:
                fails.append(f"{f.name}: missing column {c}")
        if d["tier"].nunique() != 1 or d["xi"].nunique() != 1:
            fails.append(f"{f.name}: not a single (tier, xi) cell")
        rows.append(d)
    if fails:
        return {}, pd.DataFrame(), [], fails
    df = pd.concat(rows, ignore_index=True)

    if set(df["landscape"]) != {LANDSCAPE}:
        fails.append(f"landscape grid {sorted(set(df['landscape']))}")
    if not bool(df["use_visibility_reward"].all()):
        fails.append("visibility reward must be on in every Figure-1 tier")
    if set(df["formulation"]) != {"logit"} or set(df["credit_sharing"]) != {
            "equal_split"}:
        fails.append("expected logit + equal_split throughout")
    if set(int(t) for t in df["tau_max"]) != {TAU_MAX_PIN}:
        fails.append(f"tau_max {sorted(set(df['tau_max']))} != {TAU_MAX_PIN}")
    if set(df["tier"]) != set(TIERS):
        fails.append(f"tiers {sorted(set(df['tier']))} != {list(TIERS)}")
    if not np.allclose(df["xi"], np.round(1.0 - df["curiosity"], 2), atol=1e-9):
        fails.append("xi != 1 - curiosity")
    df["xi"] = np.round(df["xi"], 2)
    xis = sorted(float(x) for x in df["xi"].unique())
    if len(xis) != 11:
        fails.append(f"{len(xis)} xi values, expected the 11-point dense grid")
    if len(df) != 55 * SEEDS_PER_CELL:
        fails.append(f"{len(df)} runs, expected {55 * SEEDS_PER_CELL}")
    if df.duplicated(["tier", "xi", "seed"]).any():
        fails.append("duplicate (tier, xi, seed) rows")

    for tier in TIERS:
        d = df[df["tier"] == tier]
        pin = TIER_PIN[tier]
        if sorted(float(x) for x in d["xi"].unique()) != xis:
            fails.append(f"{tier}: incomplete xi grid")
        for col, key in (("exit_mode", "exit_mode"), ("g", "g"),
                         ("horizon", "horizon"),
                         ("dyadic_lambda", "lam"),
                         ("dyadic_move_pull", "pull")):
            vals = d[col].unique()
            if len(vals) != 1:
                fails.append(f"{tier}: {col} not constant {vals}")
                continue
            got, want = vals[0], pin[key]
            if isinstance(want, str):
                if str(got) != want:
                    fails.append(f"{tier}: {col} {got!r} != {want!r}")
            elif abs(float(got) - float(want)) > 1e-9:
                fails.append(f"{tier}: {col} {got} != {want}")
        counts = d.groupby("xi").size()
        if not (counts == SEEDS_PER_CELL).all():
            fails.append(f"{tier}: seed counts per xi {dict(counts)}")

    exp: dict[tuple, tuple] = {}
    for tier in TIERS:
        for xi in xis:
            sel = df[(df["tier"] == tier) & (df["xi"] == xi)]
            if len(sel) != SEEDS_PER_CELL:
                fails.append(f"{tier} xi={xi}: {len(sel)} runs")
            for panel, c in COL.items():
                exp[(panel, tier, xi)] = _mean_ci(sel[c])
    runs = df[["tier", "xi", "harvested_fin", "c_ideal_tm"]].copy()
    return exp, runs, xis, fails


def _load_pre_adr(root: Path):
    """Same contract on the pre-ADR archives (5-point grid, 500 runs)."""
    fails: list[str] = []
    w = pd.read_csv(root / "bug0006_remeasure_v2" / "w_v05_runs.csv")
    wv = w[(w["landscape"] == LANDSCAPE)
           & (w["use_visibility_reward"] == True)].copy()  # noqa: E712
    if len(wv) != 200:
        fails.append(f"w_v05 mgstd5+visibility: {len(wv)} runs, expected 200")
    wv["xi"] = np.round(1.0 - wv["curiosity"], 2)
    gs = sorted({float(np.round(x, 3)) for x in wv["agent_generation_time"]})
    if gs != [0.0, 0.1]:
        fails.append(f"w_v05: entry grid {gs}")
    wv["tier"] = np.where(np.round(wv["agent_generation_time"], 3) == 0.0,
                          "M0", "M1_fixed")
    m = pd.read_csv(root / "bug0006_remeasure_v2" / "m2_xi_runs.csv")
    if len(m) != 200:
        fails.append(f"m2_xi: {len(m)} runs, expected 200")
    m["xi"] = np.round(m["xi"], 2)
    m["tier"] = np.where(m["dyadic_lambda"] == 0.0, "M2_l0", "M2_l1")
    t = pd.read_csv(root / "turnover_v2" / "runs.csv")
    if len(t) != 1520:
        fails.append(f"turnover_v2: {len(t)} runs, expected 1,520")
    ts = t[(t["exit_mode"] == "age_hazard_staggered")
           & (t["landscape"] == LANDSCAPE)
           & (np.round(t["entry"], 3) == gs[-1])].copy()
    if len(ts) != 100:
        fails.append(f"turnover staggered: {len(ts)} runs, expected 100")
    ts["xi"] = np.round(ts["xi"], 2)
    ts["tier"] = "M1_staggered"
    df = pd.concat([wv[["tier", "xi", "harvested_fin", "c_ideal_tm", "seed"]],
                    m[["tier", "xi", "harvested_fin", "c_ideal_tm", "seed"]],
                    ts[["tier", "xi", "harvested_fin", "c_ideal_tm", "seed"]]],
                   ignore_index=True)
    xis = sorted(float(x) for x in df["xi"].unique())
    if len(xis) != 5:
        fails.append(f"pre-ADR: {len(xis)} xi values, expected 5")
    if len(df) != 500:
        fails.append(f"pre-ADR: {len(df)} runs, expected 500")
    exp: dict[tuple, tuple] = {}
    for tier in TIERS:
        for xi in xis:
            sel = df[(df["tier"] == tier) & (df["xi"] == xi)]
            if len(sel) != SEEDS_PER_CELL:
                fails.append(f"{tier} xi={xi}: {len(sel)} runs")
            for panel, c in COL.items():
                exp[(panel, tier, xi)] = _mean_ci(sel[c])
    return exp, df[["tier", "xi", "harvested_fin", "c_ideal_tm"]], xis, fails


def _expected_annotations(exp, runs, xis):
    """Every annotation number the builder writes, recomputed here."""
    v = {k: val[0] for k, val in exp.items()}
    out: dict[tuple, float] = {}
    lo_knot = _knot(xis, min(xis))
    hi_knot = _knot(xis, REPORT_HI_KNOT)
    plateau = [x for x in xis if x <= 0.5]

    out[("a", "entry_plateau_harvested_min")] = min(
        v[("a", t, x)] for t in ENTRY_ARMS for x in plateau)
    out[("a", "entry_plateau_harvested_max")] = max(
        v[("a", t, x)] for t in ENTRY_ARMS for x in plateau)
    out[("a", "stage2_harvest_min")] = min(v[("a", t, hi_knot)]
                                           for t in ENTRY_ARMS)
    out[("a", "stage2_harvest_max")] = max(v[("a", t, hi_knot)]
                                           for t in ENTRY_ARMS)
    out[("a", "m0_xi09_harvested")] = v[("a", "M0", hi_knot)]
    out[("a", "m1_fixed_harvested_min")] = min(v[("a", "M1_fixed", x)]
                                               for x in plateau)
    out[("a", "m1_fixed_harvested_max")] = max(v[("a", "M1_fixed", x)]
                                               for x in plateau)
    out[("a", "staggered_xi09_harvested")] = v[("a", "M1_staggered", hi_knot)]
    out[("a", "staggered_rescue_pp")] = 100.0 * (
        v[("a", "M1_staggered", hi_knot)] - v[("a", "M1_fixed", hi_knot)])

    for tier in TIERS:
        out[("b", f"c_tm_xi01::{tier}")] = v[("b", tier, lo_knot)]
        out[("b", f"c_tm_xi09::{tier}")] = v[("b", tier, hi_knot)]
    out[("b", "alignment_spread_xi01")] = (
        max(v[("b", t, lo_knot)] for t in TIERS)
        - min(v[("b", t, lo_knot)] for t in TIERS))
    out[("b", "alignment_spread_xi09")] = (
        max(v[("b", t, hi_knot)] for t in TIERS)
        - min(v[("b", t, hi_knot)] for t in TIERS))
    out[("b", "m2_l0_alignment_rise")] = (v[("b", "M2_l0", hi_knot)]
                                          - v[("b", "M2_l0", lo_knot)])
    out[("b", "m2_l0_alignment_min")] = min(v[("b", "M2_l0", x)] for x in xis)

    for tier in TIERS:
        d = runs[runs["tier"] == tier]
        out[("b", f"corr_xi_ctm_run::{tier}")] = float(
            np.corrcoef(d["xi"], d["c_ideal_tm"])[0, 1])
        cell = d.groupby("xi")["c_ideal_tm"].mean()
        out[("b", f"corr_xi_ctm_cell::{tier}")] = float(
            np.corrcoef(cell.index, cell.values)[0, 1])
    return out


def _check_cell_identity(exp, data_root) -> list[str]:
    """Second route to every plotted point: one file at a time, mean of raw col."""
    fails: list[str] = []
    root = _root(data_root)
    if _layout(root) != "adr0003":
        return fails
    got = set()
    for f in sorted((root / "cells").glob("*.csv")):
        d = pd.read_csv(f)
        tier = str(d["tier"].iloc[0])
        xi = float(np.round(d["xi"].iloc[0], 2))
        for panel, c in COL.items():
            key = (panel, tier, xi)
            if key not in exp:
                fails.append(f"{f.name}: no plotted point for {key}")
                continue
            ref = float(d[c].mean())
            if abs(ref - exp[key][0]) > 1e-12:
                fails.append(f"cell-mean drift {tier} xi={xi} panel={panel}: "
                             f"{exp[key][0]:.12g} vs {ref:.12g}")
            got.add(key)
    if got != set(exp):
        fails.append(f"cell identity covered {len(got)} of {len(exp)} points")
    return fails


def _check_report_anchors(exp, xis) -> list[str]:
    fails: list[str] = []
    for tier, anchor in REPORT_ANCHORS.items():
        for xi, want in zip((0.1, 0.5, 0.9), anchor):
            k = _knot(xis, xi)
            if abs(k - xi) > 1e-9:
                fails.append(f"xi grid has no knot at {xi} (nearest {k})")
                continue
            got = exp[("a", tier, k)][0]
            if abs(round(got, 3) - want) > 1e-9:
                fails.append(f"report anchor drift {tier} xi={xi}: "
                             f"computed {got:.4f} vs report {want:.3f}")
    return fails


def _check_corr_disclosure(exp, runs, tol) -> list[str]:
    fails: list[str] = []
    corr = {t: float(np.corrcoef(runs[runs["tier"] == t]["xi"],
                                 runs[runs["tier"] == t]["c_ideal_tm"])[0, 1])
            for t in TIERS}
    if abs(round(corr["M2_l0"], 3) - REPORT_M2L0_CORR) > 1e-9:
        fails.append(f"M2 lambda=0 run-level corr {corr['M2_l0']:.4f} != "
                     f"report {REPORT_M2L0_CORR:.3f}")
    for t in TIERS:
        if t == "M2_l0":
            if corr[t] <= 0:
                fails.append(f"{t}: expected the only POSITIVE run-level corr")
        elif corr[t] >= 0:
            fails.append(f"{t}: run-level corr {corr[t]:+.3f} must be negative")
    xis_sorted = sorted({float(x) for x in runs["xi"].unique()})
    d = exp[("b", "M2_l0", _knot(xis_sorted, 0.9))][0] \
        - exp[("b", "M2_l0", _knot(xis_sorted, 0.1))][0]
    if d <= 0:
        fails.append("M2 lambda=0 alignment must rise from xi=0.1 to 0.9 "
                     f"(got {d:+.4f})")
    return fails


def _recheck_canvas(data_root, tol, outdir):
    """Rebuild in memory and re-run the shared visual QA contract."""
    fails: list[str] = []
    spec = importlib.util.spec_from_file_location("fig_F6_v2", HERE / "fig_F6_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fig, sidecar, _runs, _pins, meta = mod.build_figure(data_root)
    try:
        axes = fig.axes
        if len(axes) != 2:
            return [f"figure has {len(axes)} axes, expected 2"]
        n_xi = len(meta["xis"])
        visual_qa.assert_figure_clean(fig, [
            {"ax": axes[0], "metric": M_HARVEST, "expect_lines": len(TIERS),
             "points_per_line": n_xi, "sidecar": sidecar,
             "filters": {"statistic": "mean"}, "tol": tol},
            {"ax": axes[1], "metric": M_CORR, "expect_lines": len(TIERS),
             "points_per_line": n_xi, "sidecar": sidecar,
             "filters": {"statistic": "mean"}, "tol": tol},
        ])
        # provenance: the footnote must name the archive, the ADR-0003 pins
        # and the non-calibration status (pre-ADR build: its own archives)
        note = " ".join(t.get_text() for t in fig.texts
                        if "Sources" in t.get_text())
        if _layout(_root(data_root)) == "adr0003":
            tokens = ("fig1_xi_adr0003", "ADR-0003", "not calibrated",
                      f"tau_max = {TAU_MAX_PIN}", "0.25")
        else:
            tokens = ("bug0006_remeasure_v2", "turnover_v2", "not calibrated")
        for token in tokens:
            if token not in note:
                fails.append(f"footnote is missing {token!r}")
        if _layout(_root(data_root)) == "adr0003":
            # the disclosure callout must carry the computed correlation
            canvas = " ".join(t.get_text() for ax in axes for t in ax.texts)
            if "0.304" not in canvas:
                fails.append("panel canvas does not carry the +0.304 disclosure")
        # outputs must be the freshly built ones (stem check)
        for ext in (".png", ".pdf"):
            if not (Path(outdir) / f"{FNAME}{ext}").exists():
                fails.append(f"missing {FNAME}{ext}")
        del outdir
    except AssertionError as exc:
        fails.append(f"visual_qa: {exc}")
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return fails


def check(outdir="output/report_notebook_v1", data_root=None,
          tol: float = 1e-9) -> list[str]:
    out = Path(outdir)
    if not out.is_absolute():
        out = REPO / out
    root = _root(data_root)
    exp, runs, xis, fails = (_load_adr0003(root) if _layout(root) == "adr0003"
                             else _load_pre_adr(root))
    if fails:
        return fails

    side_path = out / f"{FNAME}_data.csv"
    for p in (side_path, out / f"{FNAME}.png", out / f"{FNAME}.pdf"):
        if not p.exists():
            fails.append(f"missing output {p}")
    if fails:
        return fails

    got = pd.read_csv(side_path)
    need = {"panel", "metric", "statistic", "tier", "xi", "value",
            "ci_lo", "ci_hi", "n"}
    if not need <= set(got.columns):
        fails.append(f"sidecar missing columns {sorted(need - set(got.columns))}")
        return fails

    # ---- plotted statistics ----------------------------------------------
    means = got[got["statistic"] == "mean"]
    if len(means) != 2 * len(TIERS) * len(xis):
        fails.append(f"sidecar has {len(means)} mean rows, expected "
                     f"{2 * len(TIERS) * len(xis)}")
    seen = set()
    for _, r in means.iterrows():
        key = (r["panel"], r["tier"], float(r["xi"]))
        if key in seen:
            fails.append(f"duplicate sidecar cell {key}")
            continue
        seen.add(key)
        if r["metric"] != PANELS[r["panel"]]:
            fails.append(f"{key}: metric {r['metric']!r} does not belong to "
                         f"panel {r['panel']!r}")
        if key not in exp:
            fails.append(f"unexpected sidecar cell {key}")
            continue
        mean, lo, hi, n = exp[key]
        for name, a, b in (("value", r["value"], mean),
                           ("ci_lo", r["ci_lo"], lo),
                           ("ci_hi", r["ci_hi"], hi)):
            if abs(float(a) - float(b)) > tol:
                fails.append(f"{key} {name}: sidecar {float(a)!r} vs "
                             f"recomputed {float(b)!r}")
        if int(r["n"]) != n:
            fails.append(f"{key} n: sidecar {int(r['n'])} vs {n}")
    for key in exp:
        if key not in seen:
            fails.append(f"missing sidecar cell {key}")

    # ---- annotations -----------------------------------------------------
    anns = got[got["statistic"] == "annotation"]
    have = {(r["panel"], r["metric"]): float(r["value"])
            for _, r in anns.iterrows()}
    want = _expected_annotations(exp, runs, xis)
    for key, val in want.items():
        if key not in have:
            fails.append(f"missing annotation {key}")
        elif abs(have[key] - val) > tol:
            fails.append(f"annotation {key}: sidecar {have[key]!r} vs "
                         f"recomputed {val!r}")
    for key in have:
        if key not in want:
            fails.append(f"unexpected annotation {key}")

    # ---- independent gates ----------------------------------------------
    fails.extend(_check_cell_identity(exp, data_root))
    if _layout(root) == "adr0003":
        fails.extend(_check_report_anchors(exp, xis))
        fails.extend(_check_corr_disclosure(exp, runs, tol))

    # ---- rendered canvas + provenance ------------------------------------
    fails.extend(_recheck_canvas(data_root, tol, out))
    return fails


if __name__ == "__main__":
    problems = check()
    print("PASS" if not problems else "FAIL")
    for p in problems:
        print(" -", p)
    sys.exit(1 if problems else 0)
