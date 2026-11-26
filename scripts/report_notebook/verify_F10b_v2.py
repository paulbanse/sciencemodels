"""Independent verifier for ``F10b_v2_m2_visibility_like_for_like`` (Fig 16).

``check(outdir="output/report_notebook_v1", data_root=None, tol=1e-9)``
returns a list of failures; empty means everything reconciles.

Independence: ``data/fig1_xi_adr0003/cells/fig1_tier-*_xi-*.csv`` is re-read
here with this module's own filename parser and tier map (nothing is imported
from ``figF_v2_m2_visibility_postbug``), and every plotted mean, 95% CI and
annotation is recomputed and compared against the ``*_data.csv`` sidecar.

Guards this verifier enforces:
  * the archive shape the figure claims: 55 cells x 20 seeds = 1,100 runs,
    5 tiers, 11 xi points, mgstd5, one H pin per tier;
  * panel (c) draws ONLY tiers whose population matches the modal one -- the
    staggered M1 tier (N = 121) must not appear as a team-size series, because
    team size is a level and a 121-agent pool is not comparable to a 20-agent
    one.  Its population is asserted here from the cells;
  * two rendering regressions: no text artist reads as a bare "0" (the pre-fix
    stray label above panel B's zero line) and no vertical band-boundary rule
    survives in any panel (the bands are shaded regions only).

Finally the figure is rebuilt in memory and ``visual_qa.assert_figure_clean``
re-runs on the rendered canvas.

Run:  MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/report_notebook/verify_F10b_v2.py
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
import visual_qa  # noqa: E402

FNAME = "F10b_v2_m2_visibility_like_for_like"
DEFAULT_ROOT = REPO / "data" / "fig1_xi_adr0003"
LANDSCAPE = "mgstd5"
Z95 = 1.96
KLANN_TEAM_SIZE = 2.65
BENCH_LABEL = "Empirical benchmark (unfitted)"

M_HARVEST = "harvested knowledge share"
M_CORR = "Corr(prestige, ideal merit) (time mean)"
M_TEAM = "co-harvest team size (time mean)"

T0 = "M0 (g = 0)"
T1F = "M1 (g = 0.25), fixed span"
T1S = "M1 (g = 0.25), staggered"
T20 = "M2 λ = 0 (prestige matching)"
T21 = "M2 λ = 1 (closeness × local merit)"
TIERS = (T0, T1F, T1S, T20, T21)
ENTRY_ARMS = (T1F, T1S, T20, T21)
TIER_CODE = {"M0": T0, "M1_fixed": T1F, "M1_staggered": T1S,
             "M2_l0": T20, "M2_l1": T21}

_FILE_RE = re.compile(r"fig1_tier-(?P<tier>[A-Za-z0-9_]+)_xi-(?P<xi>\d+p\d+)\.csv")


def _mean_ci(v):
    a = pd.Series(v, dtype=float).dropna()
    n = int(a.size)
    m = float(a.mean())
    h = Z95 * float(a.std(ddof=1)) / np.sqrt(n) if n > 1 else float("nan")
    return m, m - h, m + h, n


def load(data_root=None) -> tuple[pd.DataFrame, dict]:
    root = Path(data_root) if data_root is not None else DEFAULT_ROOT
    if not root.is_absolute():
        root = REPO / root
    files = sorted((root / "cells").glob("fig1_tier-*.csv"))
    if not files:
        raise AssertionError(f"no fig1 cells under {root}/cells")
    frames = []
    for f in files:
        m = _FILE_RE.match(f.name)
        if m is None:
            raise AssertionError(f"unparsable cell filename {f.name}")
        d = pd.read_csv(f)
        if len(d) != 20:
            raise AssertionError(f"{f.name}: {len(d)} seed rows, expected 20")
        d["tier_label"] = TIER_CODE.get(m.group("tier"))
        d["xi_fn"] = float(m.group("xi").replace("p", "."))
        frames.append(d)
    runs = pd.concat(frames, ignore_index=True)
    if runs["tier_label"].isna().any():
        raise AssertionError(f"unmapped tier codes in filenames")
    if not np.allclose(runs["xi"], runs["xi_fn"]):
        raise AssertionError("xi column disagrees with the filename")
    pins = dict(files=len(files),
                cells=runs.groupby(["tier_label", "xi"]).ngroups,
                runs=len(runs), seeds=int(runs["seed"].nunique()),
                xs=sorted(runs["xi"].unique().tolist()),
                landscapes=sorted(runs["landscape"].unique().tolist()),
                horizon=runs.groupby("tier_label")["horizon"].first().to_dict())
    return runs, pins


def team_tiers(runs: pd.DataFrame) -> tuple:
    """Tiers whose population matches the modal one (team size is a level)."""
    pop = runs.groupby("tier_label")["n_final"].mean()
    modal = float(pop.round().mode().iloc[0])
    return tuple(t for t in TIERS
                 if t in pop.index and abs(pop[t] - modal) <= 0.75 * modal)


def expected(runs: pd.DataFrame, pins: dict) -> pd.DataFrame:
    rows = []
    keep_c = team_tiers(runs)
    for panel, metric, col in (("a", M_HARVEST, "harvested_fin"),
                               ("b", M_CORR, "c_ideal_tm"),
                               ("c", M_TEAM, "team_tm")):
        for tier in (TIERS if panel != "c" else keep_c):
            for xi in pins["xs"]:
                sel = runs[(runs["tier_label"] == tier) & (runs["xi"] == xi)][col]
                if sel.notna().sum() == 0:
                    continue
                m, lo, hi, n = _mean_ci(sel)
                rows.append(dict(panel=panel, metric=metric, statistic="mean",
                                 tier=tier, xi=xi, value=m, ci_lo=lo, ci_hi=hi,
                                 n=n))
    rows.append(dict(panel="c", metric=M_TEAM, statistic="reference",
                     tier=BENCH_LABEL, xi=np.nan, value=KLANN_TEAM_SIZE,
                     ci_lo=np.nan, ci_hi=np.nan, n=0))
    V = {(r["panel"], r["tier"], float(r["xi"])): r["value"] for r in rows
         if r["statistic"] == "mean"}

    def ann(panel, metric, value):
        rows.append(dict(panel=panel, metric=metric, statistic="annotation",
                         tier="", xi=np.nan, value=float(value), ci_lo=np.nan,
                         ci_hi=np.nan, n=100))

    plateau = [V[("a", t, xi)] for t in ENTRY_ARMS for xi in pins["xs"]
               if xi <= 0.5]
    tail = [V[("a", t, xi)] for t in ENTRY_ARMS for xi in pins["xs"] if xi >= 0.9]
    ann("a", "entry_plateau_harvested_min", min(plateau))
    ann("a", "entry_plateau_harvested_max", max(plateau))
    ann("a", "stage2_collapse_min", min(tail))
    ann("a", "stage2_collapse_max", max(tail))
    ann("a", "m0_min_harvested", min(V[("a", T0, xi)] for xi in pins["xs"]))
    for tier in TIERS:
        ann("b", f"c_tm_first::{tier}", V[("b", tier, pins["xs"][0])])
        ann("b", f"c_tm_last::{tier}", V[("b", tier, pins["xs"][-1])])
    small = [v for (p, t, x), v in V.items() if p == "c" and x <= 0.5]
    ann("c", "team_small_min", min(small))
    ann("c", "team_small_max", max(small))
    for lab, thr in (("xi07", 0.7), ("xi09", 0.9)):
        vals = [v for (p, t, x), v in V.items() if p == "c" and x == thr]
        ann("c", f"team_{lab}_min", min(vals))
        ann("c", f"team_{lab}_max", max(vals))
    return pd.DataFrame(rows)


def check(outdir="output/report_notebook_v1", data_root=None,
          tol: float = 1e-9) -> list[str]:
    fails: list[str] = []
    runs, pins = load(data_root)

    if pins["files"] != 55:
        fails.append(f"{pins['files']} cell files, expected 55")
    if pins["runs"] != 1100:
        fails.append(f"{pins['runs']} runs, expected 1,100")
    if pins["cells"] != 55:
        fails.append(f"{pins['cells']} (tier, xi) cells, expected 55")
    if len(pins["xs"]) != 11:
        fails.append(f"{len(pins['xs'])} xi points, expected 11")
    if pins["landscapes"] != [LANDSCAPE]:
        fails.append(f"landscape grid {pins['landscapes']}")
    if set(runs["tier_label"].unique()) != set(TIERS):
        fails.append(f"tier set {sorted(runs['tier_label'].unique())}")

    # the population asymmetry that justifies excluding the staggered tier
    pop = runs.groupby("tier_label")["n_final"].mean()
    if abs(pop[T1S] - pop[T0]) <= 1.5 * pop[T0] * 0.5:
        fails.append("staggered M1 tier is no longer a different population: "
                     f"N={pop[T1S]:.0f} vs {pop[T0]:.0f}")

    exp = expected(runs, pins)
    out = Path(outdir)
    if not out.is_absolute():
        out = REPO / out
    side_path = out / f"{FNAME}_data.csv"
    if not side_path.exists():
        return fails + [f"missing sidecar {side_path}"]
    side = pd.read_csv(side_path)
    if len(side) != len(exp):
        fails.append(f"sidecar rows {len(side)} != expected {len(exp)}")

    key = ["panel", "metric", "statistic", "tier"]
    e, s = exp.copy(), side.copy()
    e["xi"] = pd.to_numeric(e["xi"], errors="coerce").round(6)
    s["xi"] = pd.to_numeric(s["xi"], errors="coerce").round(6)
    for d in (e, s):
        d["xi"] = d["xi"].fillna(-999.0)
        d["tier"] = d["tier"].fillna("").astype(str)
    m = s.merge(e, on=key + ["xi"], how="outer", suffixes=("_side", "_exp"),
                indicator=True)
    for _, r in m[m["_merge"] != "both"].iterrows():
        fails.append(f"sidecar/expected key mismatch: "
                     f"{[r[k] for k in key + ['xi']]} ({r['_merge']})")
    both = m[m["_merge"] == "both"]
    for col in ("value", "ci_lo", "ci_hi"):
        a = both[f"{col}_side"].astype(float).to_numpy()
        b = both[f"{col}_exp"].astype(float).to_numpy()
        ok = (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= tol)
        for i in np.where(~ok)[0][:5]:
            r = both.iloc[i]
            fails.append(f"{col} drift at {[r[k] for k in key + ['xi']]}: "
                         f"{a[i]!r} vs {b[i]!r}")

    # panel (c) must not carry the staggered tier
    pc = side[(side["panel"] == "c") & (side["statistic"] == "mean")]
    if T1S in set(pc["tier"]):
        fails.append("panel (c) plots the staggered M1 tier, whose population "
                     "is not comparable")

    # ---- rendering regressions + rebuild --------------------------------
    # ``build_figure`` runs ``visual_qa.assert_figure_clean`` and the local
    # display-space audit itself, so a render-time regression raises here.
    import matplotlib.pyplot as plt
    import figF_v2_m2_visibility_postbug as B
    fig, _side, _runs, _pins = B.build_figure(data_root)
    texts = [t.get_text() for t in fig.texts]
    for ax in fig.axes:
        texts += [t.get_text() for t in ax.texts]
        for line in ax.get_lines():
            x = line.get_xdata(orig=False)
            if len(x) == 2 and np.ptp(np.asarray(x, float)) == 0.0:
                fails.append("vertical band rule still drawn in "
                             f"{ax.get_title()[:24]!r}")
    plt.close(fig)
    if any(t.strip() == "0" for t in texts):
        fails.append('a text artist still reads as a bare "0"')
    return fails


if __name__ == "__main__":
    problems = check()
    if problems:
        print(f"FAIL ({len(problems)}):")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print("verify_F10b_v2: OK — every plotted value, CI and annotation "
          "reconciles; visual_qa clean on rebuild")
