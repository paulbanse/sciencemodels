"""Independent verifier for F10b_v2 (figF_v2_m2_visibility_postbug.py).

``check(outdir="output/report_notebook_v1", data_root=None, tol=1e-9)``
returns a list of failures -- empty means everything reconciles.

Independence: M0/M1 are re-read from ``w_v05_runs.csv`` and M2 from
``m2_xi_runs.csv`` with this module's own selectors; every plotted mean,
CI, annotation and the Klann reference row is recomputed here and compared
against ``F10b_v2_m2_visibility_like_for_like_data.csv``.  The pre-vs-post
BUG-0006 numbers quoted on the canvas are re-read from the archived delta
tables (``tables/m2_xi_bug0006_delta.csv``, ``tables/w_v05_bug0006_delta.csv``).

Two regression guards for the defects this figure was commissioned to fix:
  * no text artist in any panel reads as a bare "0" (the pre-fix stray
    label just above panel B's zero line);
  * no vertical band-boundary rule survives in any panel (the pre-fix
    dashed axvlines that crossed every series and panel B's annotation).

Finally the figure is rebuilt in memory and the shared
``visual_qa.assert_figure_clean`` contract is re-run on the rendered canvas.
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

FNAME = "F10b_v2_m2_visibility_like_for_like"
DEFAULT_DATA_ROOT = REPO / "data" / "bug0006_remeasure_v2"
LANDSCAPE = "mgstd5"
XIS = (0.1, 0.3, 0.5, 0.7, 0.9)
Z95 = 1.96
KLANN_TEAM_SIZE = 2.65

M0 = "M0 (g = 0)"
M1 = "M1 (g = 0.1)"
M2_TIERS = {0.0: "M2 λ = 0 (prestige matching)",
            1.0: "M2 λ = 1 (closeness × local merit)"}
TIERS = (M0, M1, M2_TIERS[0.0], M2_TIERS[1.0])
ENTRY_ARMS = (M1, M2_TIERS[0.0], M2_TIERS[1.0])
M_HARVEST = "harvested knowledge share"
M_CORR = "Corr(prestige, ideal merit) (time mean)"
M_TEAM = "co-harvest team size (time mean)"
PANELS = {"a": M_HARVEST, "b": M_CORR, "c": M_TEAM}


def _mean_ci(v) -> tuple[float, float, float, int]:
    a = np.asarray(v, dtype=float)
    n = int(a.size)
    m = float(a.mean())
    h = Z95 * float(a.std(ddof=1)) / np.sqrt(n) if n > 1 else float("nan")
    return m, m - h, m + h, n


def _root(data_root) -> Path:
    return Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT


def _load_cells(data_root):
    root = _root(data_root)
    fails: list[str] = []
    exp: dict[tuple, tuple] = {}

    w = pd.read_csv(root / "w_v05_runs.csv")
    wv = w[(w["landscape"] == LANDSCAPE)
           & (w["use_visibility_reward"] == True)].copy()  # noqa: E712
    if len(wv) != 200:
        fails.append(f"w_v05 mgstd5+visibility: {len(wv)} runs, expected 200")
    if set(np.round(wv["vanishing"], 3)) != {0.5}:
        fails.append("w_v05: vanishing pin is not the single value 0.5")
    wv["xi"] = np.round(1.0 - wv["curiosity"], 1)
    if sorted(set(wv["xi"])) != list(XIS):
        fails.append(f"w_v05: xi grid {sorted(set(wv['xi']))}")
    for g, tier in ((0.0, M0), (0.1, M1)):
        for xi in XIS:
            sel = wv[(np.round(wv["agent_generation_time"], 2) == g)
                     & (wv["xi"] == xi)]
            if len(sel) != 20:
                fails.append(f"w_v05 {tier} xi={xi}: {len(sel)} runs")
            for panel, col in (("a", "harvested_fin"), ("b", "c_ideal_tm")):
                exp[(panel, tier, xi)] = _mean_ci(sel[col])

    m = pd.read_csv(root / "m2_xi_runs.csv")
    if len(m) != 200:
        fails.append(f"m2_xi: {len(m)} runs, expected 200")
    if set(np.round(m["vanishing"], 3)) != {0.9}:
        fails.append("m2_xi: vanishing pin is not the single value 0.9")
    if not (m["landscape"] == LANDSCAPE).all():
        fails.append("m2_xi: landscape is not uniformly mgstd5")
    if not (m["use_visibility_reward"] == True).all():  # noqa: E712
        fails.append("m2_xi: visibility reward is not uniformly on")
    if set(np.round(m["dyadic_move_pull"], 3)) != {1.0}:
        fails.append("m2_xi: pull is not uniformly 1.0")
    m["xi"] = np.round(m["xi"], 1)
    if sorted(set(m["xi"])) != list(XIS):
        fails.append(f"m2_xi: xi grid {sorted(set(m['xi']))}")
    for lam, tier in M2_TIERS.items():
        for xi in XIS:
            sel = m[(np.round(m["dyadic_lambda"], 1) == lam) & (m["xi"] == xi)]
            if len(sel) != 20:
                fails.append(f"m2 {tier} xi={xi}: {len(sel)} runs")
            for panel, col in (("a", "harvested_fin"), ("b", "c_ideal_tm"),
                               ("c", "team_tm")):
                exp[(panel, tier, xi)] = _mean_ci(sel[col])

    want = 2 * len(TIERS) * len(XIS) + len(M2_TIERS) * len(XIS)
    if len(exp) != want:
        fails.append(f"{len(exp)} expected cells, expected {want}")
    return exp, fails


def _expected_annotations(exp, data_root):
    v = {k: val[0] for k, val in exp.items()}
    out: dict[tuple, float] = {}

    plateau = [v[("a", t, xi)] for t in ENTRY_ARMS for xi in (0.1, 0.3, 0.5)]
    collapse = [v[("a", t, 0.9)] for t in ENTRY_ARMS]
    out[("a", "entry_plateau_harvested_min")] = min(plateau)
    out[("a", "entry_plateau_harvested_max")] = max(plateau)
    out[("a", "stage2_collapse_min")] = min(collapse)
    out[("a", "stage2_collapse_max")] = max(collapse)
    out[("a", "m0_xi09_harvested")] = v[("a", M0, 0.9)]
    for tier in TIERS:
        out[("b", f"c_tm_xi01::{tier}")] = v[("b", tier, 0.1)]
        out[("b", f"c_tm_xi09::{tier}")] = v[("b", tier, 0.9)]
    lam0 = [v[("b", M2_TIERS[0.0], xi)] for xi in XIS]
    out[("b", "m2_lambda0_flat_min")] = min(lam0)
    out[("b", "m2_lambda0_flat_max")] = max(lam0)
    small = [v[("c", t, xi)] for t in M2_TIERS.values() for xi in (0.1, 0.3, 0.5)]
    out[("c", "team_small_min")] = min(small)
    out[("c", "team_small_max")] = max(small)
    for xi, key in ((0.7, "team_xi07"), (0.9, "team_xi09")):
        vals = [v[("c", t, xi)] for t in M2_TIERS.values()]
        out[("c", f"{key}_min")] = min(vals)
        out[("c", f"{key}_max")] = max(vals)

    root = _root(data_root)
    d = pd.read_csv(root / "tables" / "m2_xi_bug0006_delta.csv")
    q = d[(d["arm"] == "M2_xi") & (d["scope"] == "all")
          & (d["metric"] == "c_ideal_tm")]
    if len(q) != 1:
        raise AssertionError("m2_xi_bug0006_delta: c_ideal_tm/all row")
    out[("b", "bug0006_m2_c_tm_delta")] = float(q["mean_delta"].iloc[0])
    dw = pd.read_csv(root / "tables" / "w_v05_bug0006_delta.csv")
    out[("a", "bug0006_w_v05_harvest_max_abs_diff")] = float(
        dw[dw["metric"] == "harvested_fin"]["max_abs_diff"].abs().max())

    def _w(scope, metric, col):
        q = dw[(dw["scope"] == scope) & (dw["metric"] == metric)]
        if len(q) != 1:
            raise AssertionError(f"w_v05 delta row {scope}/{metric}")
        return float(q[col].iloc[0])

    out[("b", "bug0006_w_v05_m0_c_tm_max_abs_diff")] = _w(
        "g=0.0 (control)", "c_ideal_tm", "max_abs_diff")
    out[("b", "bug0006_w_v05_m1_mgstd5_c_tm_mean_delta")] = _w(
        "g=0.1 (mgstd5)", "c_ideal_tm", "mean_delta")
    out[("b", "bug0006_w_v05_m1_mgstd5_c_tm_max_abs_diff")] = _w(
        "g=0.1 (mgstd5)", "c_ideal_tm", "max_abs_diff")
    return out


def _vertical_rules(ax) -> list:
    out = []
    for ln in ax.lines:
        try:
            x = np.asarray(ln.get_xdata(orig=False), dtype=float)
            y = np.asarray(ln.get_ydata(orig=False), dtype=float)
        except (TypeError, ValueError):
            continue
        if len(x) == 2 and len(y) == 2 and np.ptp(x) == 0.0 and np.ptp(y) > 0.0:
            out.append(ln)
    return out


def _recheck_canvas(data_root, tol):
    fails: list[str] = []
    spec = importlib.util.spec_from_file_location(
        "figF_v2_m2_visibility_postbug",
        HERE / "figF_v2_m2_visibility_postbug.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fig, sidecar, _runs, _pins = mod.build_figure(data_root)
    try:
        axes = fig.axes
        if len(axes) != 3:
            return [f"figure has {len(axes)} axes, expected 3"]
        # regression guards for the two pre-fix defects in panel B refit
        for i, ax in enumerate(axes):
            bad = [t.get_text() for t in ax.texts
                   if t.get_text().strip() in {"0", "0.0"}]
            if bad:
                fails.append(f"panel {i}: stray zero text label(s) {bad}")
            vr = _vertical_rules(ax)
            if vr:
                fails.append(f"panel {i}: {len(vr)} vertical boundary rule(s) "
                             "still drawn")
        visual_qa.assert_figure_clean(fig, [
            {"ax": axes[0], "metric": M_HARVEST, "expect_lines": len(TIERS),
             "points_per_line": len(XIS), "sidecar": sidecar,
             "filters": {"statistic": "mean"}, "tol": tol},
            {"ax": axes[1], "metric": M_CORR, "expect_lines": len(TIERS),
             "points_per_line": len(XIS), "sidecar": sidecar,
             "filters": {"statistic": "mean"}, "tol": tol},
            {"ax": axes[2], "metric": M_TEAM, "expect_lines": len(M2_TIERS),
             "points_per_line": len(XIS), "sidecar": sidecar,
             "filters": {"statistic": "mean"}, "tol": tol},
        ])
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

    # Klann benchmark reference row (panel c)
    ref = got[(got["statistic"] == "reference") & (got["metric"] == M_TEAM)]
    if len(ref) != 1:
        fails.append(f"sidecar: {len(ref)} Klann reference rows, expected 1")
    elif abs(float(ref["value"].iloc[0]) - KLANN_TEAM_SIZE) > tol:
        fails.append(f"Klann reference row {float(ref['value'].iloc[0])!r} != "
                     f"{KLANN_TEAM_SIZE}")

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

    fails.extend(_recheck_canvas(data_root, tol))
    return fails


if __name__ == "__main__":
    problems = check()
    print("PASS" if not problems else "FAIL")
    for p in problems:
        print(" -", p)
    sys.exit(1 if problems else 0)
