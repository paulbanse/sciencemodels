"""Independent verifier for F6_v2 (fig_F6_v2.py).

``check(outdir="output/report_notebook_v1", data_root=None, tol=1e-9)``
returns a list of failures -- empty means everything reconciles.

Independence: this module re-reads the three RAW archives with its own
selectors and recomputes every plotted statistic (mean + 95% CI) and every
annotation number itself; it never imports the builder's loaders.  It also

  * asserts the input shape (w_v05 200 mgstd5+visibility runs, m2_xi 200
    runs, turnover staggered 100 runs; 20 seeds per cell; xi grids),
  * cross-checks the per-cell means against
    ``tables/four_tier_xi_summary.csv`` at ``tol``,
  * rebuilds the figure in memory and re-runs
    ``visual_qa.assert_figure_clean`` on the RENDERED canvas (rendered-artist
    data contract + collision-free layout), and
  * asserts the PNG / PDF / sidecar triplets are on disk.
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
DEFAULT_DATA_ROOT = REPO / "data"
LANDSCAPE = "mgstd5"
XIS = (0.1, 0.3, 0.5, 0.7, 0.9)
Z95 = 1.96

TIERS = ("M0 (g = 0)", "M1 (g = 0.1)", "M1 + turnover_staggered",
         "M2 λ = 0 (prestige matching)", "M2 λ = 1 (closeness × local merit)")
M0, M1, TV = TIERS[0], TIERS[1], TIERS[2]
M2_0, M2_1 = TIERS[3], TIERS[4]
ENTRY_ARMS = (M1, M2_0, M2_1)
M_HARVEST = "harvested knowledge share"
M_CORR = "Corr(prestige, ideal merit) (time mean)"
PANELS = {"a": M_HARVEST, "b": M_CORR}
TABLE_TIER = {M0: "M0_closed_Wv05", M1: "M1_lineage_Wv05",
              TV: "M1_turnover_staggered", M2_0: "M2_lambda0",
              M2_1: "M2_lambda1"}


def _mean_ci(v) -> tuple[float, float, float, int]:
    a = np.asarray(v, dtype=float)
    n = int(a.size)
    m = float(a.mean())
    h = Z95 * float(a.std(ddof=1)) / np.sqrt(n) if n > 1 else float("nan")
    return m, m - h, m + h, n


def _root(data_root) -> Path:
    return Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT


def _load_cells(data_root):
    """(panel, tier, xi) -> (mean, lo, hi, n) recomputed from the raw CSVs."""
    root = _root(data_root)
    fails: list[str] = []
    exp: dict[tuple, tuple] = {}

    w = pd.read_csv(root / "bug0006_remeasure_v2" / "w_v05_runs.csv")
    wv = w[(w["landscape"] == LANDSCAPE)
           & (w["use_visibility_reward"] == True)].copy()  # noqa: E712
    if len(wv) != 200:
        fails.append(f"w_v05 mgstd5+visibility: {len(wv)} runs, expected 200")
    if set(np.round(wv["vanishing"], 3)) != {0.5}:
        fails.append("w_v05: vanishing pin is not the single value 0.5")
    gs = sorted(set(np.round(wv["agent_generation_time"], 2)))
    if gs != [0.0, 0.1]:
        fails.append(f"w_v05: g grid {gs}, expected [0.0, 0.1]")
    wv["xi"] = np.round(1.0 - wv["curiosity"], 1)
    if sorted(set(wv["xi"])) != list(XIS):
        fails.append(f"w_v05: xi grid {sorted(set(wv['xi']))}")
    for g, tier in ((0.0, M0), (0.1, M1)):
        for xi in XIS:
            sel = wv[(np.round(wv["agent_generation_time"], 2) == g)
                     & (wv["xi"] == xi)]
            if len(sel) != 20:
                fails.append(f"w_v05 {tier} xi={xi}: {len(sel)} runs, expected 20")
            for panel, col in (("a", "harvested_fin"), ("b", "c_ideal_tm")):
                exp[(panel, tier, xi)] = _mean_ci(sel[col])

    m = pd.read_csv(root / "bug0006_remeasure_v2" / "m2_xi_runs.csv")
    if len(m) != 200:
        fails.append(f"m2_xi: {len(m)} runs, expected 200")
    if set(np.round(m["vanishing"], 3)) != {0.9}:
        fails.append("m2_xi: vanishing pin is not the single value 0.9")
    m["xi"] = np.round(m["xi"], 1)
    if sorted(set(m["xi"])) != list(XIS):
        fails.append(f"m2_xi: xi grid {sorted(set(m['xi']))}")
    if set(np.round(m["dyadic_lambda"], 1)) != {0.0, 1.0}:
        fails.append("m2_xi: lambda grid is not {0, 1}")
    for lam, tier in ((0.0, M2_0), (1.0, M2_1)):
        for xi in XIS:
            sel = m[(np.round(m["dyadic_lambda"], 1) == lam) & (m["xi"] == xi)]
            if len(sel) != 20:
                fails.append(f"m2 {tier} xi={xi}: {len(sel)} runs, expected 20")
            for panel, col in (("a", "harvested_fin"), ("b", "c_ideal_tm")):
                exp[(panel, tier, xi)] = _mean_ci(sel[col])

    t = pd.read_csv(root / "turnover_v2" / "runs.csv")
    if len(t) != 1520:
        fails.append(f"turnover_v2: {len(t)} runs, expected 1,520")
    g_pin = max(gs)
    entries = sorted(set(np.round(t["entry"], 2)))
    if g_pin not in entries:
        fails.append(f"turnover_v2: M1 g pin {g_pin} not in entry grid {entries}")
    ts = t[(t["exit_mode"] == "age_hazard_staggered")
           & (t["landscape"] == LANDSCAPE)
           & (np.round(t["entry"], 2) == g_pin)].copy()
    if len(ts) != 100:
        fails.append(f"turnover staggered (entry={g_pin}, mgstd5): {len(ts)} "
                     "runs, expected 100 (5 xi x 20 seeds)")
    ts["xi"] = np.round(ts["xi"], 1)
    if sorted(set(ts["xi"])) != list(XIS):
        fails.append(f"turnover staggered: xi grid {sorted(set(ts['xi']))}")
    for xi in XIS:
        sel = ts[ts["xi"] == xi]
        if len(sel) != 20:
            fails.append(f"turnover staggered xi={xi}: {len(sel)} runs")
        for panel, col in (("a", "harvested_fin"), ("b", "c_ideal_tm")):
            exp[(panel, TV, xi)] = _mean_ci(sel[col])

    if len(exp) != 2 * len(TIERS) * len(XIS):
        fails.append(f"{len(exp)} expected cells, "
                     f"expected {2 * len(TIERS) * len(XIS)}")
    return exp, fails


def _expected_annotations(exp, data_root):
    """Every annotation number, recomputed from the cell aggregates."""
    v = {k: val[0] for k, val in exp.items()}
    out: dict[tuple, float] = {}

    plateau = [v[("a", t, xi)] for t in ENTRY_ARMS for xi in (0.1, 0.3, 0.5)]
    collapse = [v[("a", t, 0.9)] for t in ENTRY_ARMS]
    out[("a", "entry_plateau_harvested_min")] = min(plateau)
    out[("a", "entry_plateau_harvested_max")] = max(plateau)
    out[("a", "stage2_harvest_min")] = min(collapse)
    out[("a", "stage2_harvest_max")] = max(collapse)
    out[("a", "m0_xi09_harvested")] = v[("a", M0, 0.9)]
    out[("a", "staggered_xi01_harvested")] = v[("a", TV, 0.1)]
    out[("a", "staggered_xi09_harvested")] = v[("a", TV, 0.9)]
    out[("a", "staggered_rescue_pp")] = 100.0 * (v[("a", TV, 0.9)]
                                                 - v[("a", M1, 0.9)])
    for tier in TIERS:
        out[("b", f"c_tm_xi01::{tier}")] = v[("b", tier, 0.1)]
        out[("b", f"c_tm_xi09::{tier}")] = v[("b", tier, 0.9)]
    lam0 = [v[("b", M2_0, xi)] for xi in XIS]
    out[("b", "m2_lambda0_flat_min")] = min(lam0)
    out[("b", "m2_lambda0_flat_max")] = max(lam0)
    out[("b", "m0_alignment_drop_pct")] = 100.0 * (
        v[("b", M0, 0.9)] / v[("b", M0, 0.1)] - 1.0)
    four = pd.read_csv(_root(data_root) / "bug0006_remeasure_v2" / "tables"
                       / "four_tier_xi_summary.csv")
    pin = four[four["tier"] == TABLE_TIER[TV]]["vanishing"].unique()
    if len(pin) != 1:
        raise AssertionError("four_tier_xi_summary: turnover vanishing pin")
    out[("b", "turnover_vanishing_pin")] = float(pin[0])

    tabs = _root(data_root) / "bug0006_remeasure_v2" / "tables"
    dw = pd.read_csv(tabs / "w_v05_bug0006_delta.csv")
    dm = pd.read_csv(tabs / "m2_xi_bug0006_delta.csv")

    def _w(scope, metric, col):
        q = dw[(dw["scope"] == scope) & (dw["metric"] == metric)]
        if len(q) != 1:
            raise AssertionError(f"w_v05 delta row {scope}/{metric}")
        return float(q[col].iloc[0])

    out[("b", "bug0006_m0_c_tm_max_abs_diff")] = _w(
        "g=0.0 (control)", "c_ideal_tm", "max_abs_diff")
    out[("b", "bug0006_m1_mgstd5_c_tm_mean_delta")] = _w(
        "g=0.1 (mgstd5)", "c_ideal_tm", "mean_delta")
    q = dm[(dm["arm"] == "M2_xi") & (dm["scope"] == "all")
           & (dm["metric"] == "c_ideal_tm")]
    if len(q) != 1:
        raise AssertionError("m2_xi_bug0006_delta: c_ideal_tm/all row")
    out[("b", "bug0006_m2_c_tm_mean_delta")] = float(q["mean_delta"].iloc[0])
    return out


def _check_tier_table(exp, data_root, tol):
    fails = []
    four = pd.read_csv(_root(data_root) / "bug0006_remeasure_v2" / "tables"
                       / "four_tier_xi_summary.csv")
    col = {"a": "harvested_fin", "b": "c_ideal_tm"}
    for (panel, tier, xi), (mean, _lo, _hi, _n) in exp.items():
        q = four[(four["tier"] == TABLE_TIER[tier])
                 & (np.round(four["xi"], 1) == xi)]
        if len(q) != 1:
            fails.append(f"four_tier_xi_summary: no row {tier} xi={xi}")
            continue
        ref = float(q[col[panel]].iloc[0])
        if abs(ref - mean) > tol:
            fails.append(f"tier-table drift {tier} xi={xi} panel={panel}: "
                         f"{mean:.12g} vs {ref:.12g}")
    return fails


def _recheck_canvas(data_root, tol):
    """Rebuild in memory and re-run the shared visual QA contract."""
    fails: list[str] = []
    spec = importlib.util.spec_from_file_location("fig_F6_v2", HERE / "fig_F6_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fig, sidecar, _runs, _pins = mod.build_figure(data_root)
    try:
        axes = fig.axes
        if len(axes) != 2:
            return [f"figure has {len(axes)} axes, expected 2"]
        report = visual_qa.assert_figure_clean(fig, [
            {"ax": axes[0], "metric": M_HARVEST, "expect_lines": len(TIERS),
             "points_per_line": len(XIS), "sidecar": sidecar,
             "filters": {"statistic": "mean"}, "tol": tol},
            {"ax": axes[1], "metric": M_CORR, "expect_lines": len(TIERS),
             "points_per_line": len(XIS), "sidecar": sidecar,
             "filters": {"statistic": "mean"}, "tol": tol},
        ])
    except AssertionError as exc:
        fails.append(f"visual_qa: {exc}")
        report = {}
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
    if report.get("_total") not in (0, None):
        fails.append(f"visual_qa reported {report['_total']} failures")
    return fails


def check(outdir="output/report_notebook_v1", data_root=None,
          tol: float = 1e-9) -> list[str]:
    out = Path(outdir)
    if not out.is_absolute():
        out = REPO / out
    exp, fails = _load_cells(data_root)
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

    # ---- plotted statistics ------------------------------------------------
    means = got[got["statistic"] == "mean"]
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
        for name, a, b in (("value", r["value"], mean), ("ci_lo", r["ci_lo"], lo),
                           ("ci_hi", r["ci_hi"], hi)):
            if abs(float(a) - float(b)) > tol:
                fails.append(f"{key} {name}: sidecar {float(a)!r} vs "
                             f"recomputed {float(b)!r}")
        if int(r["n"]) != n:
            fails.append(f"{key} n: sidecar {int(r['n'])} vs {n}")
    for key in exp:
        if key not in seen:
            fails.append(f"missing sidecar cell {key}")

    # ---- annotations -------------------------------------------------------
    anns = got[got["statistic"] == "annotation"]
    have = {(r["panel"], r["metric"]): float(r["value"]) for _, r in anns.iterrows()}
    want = _expected_annotations(exp, data_root)
    for key, val in want.items():
        if key not in have:
            fails.append(f"missing annotation {key}")
        elif abs(have[key] - val) > tol:
            fails.append(f"annotation {key}: sidecar {have[key]!r} vs "
                         f"recomputed {val!r}")
    for key in have:
        if key not in want:
            fails.append(f"unexpected annotation {key}")

    # ---- independent aggregation identity ---------------------------------
    fails.extend(_check_tier_table(exp, data_root, tol))

    # ---- rendered canvas ---------------------------------------------------
    fails.extend(_recheck_canvas(data_root, tol))
    return fails


if __name__ == "__main__":
    problems = check()
    print("PASS" if not problems else "FAIL")
    for p in problems:
        print(" -", p)
    sys.exit(1 if problems else 0)
