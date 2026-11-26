"""Independent verifier for ``F2b_v2_career_flows_noisy_vs_clean`` (ADR-0003).

``check()`` re-derives every plotted quantity from the RAW career cell CSVs
``data/career_flows_adr0003/cells/career_*.csv`` and compares the result
against ``F2b_v2_career_flows_noisy_vs_clean_data.csv``.  It asserts:

  * 18 cells (3 curiosity x 3 g x 2 exit modes) at 20 seeds each = 360 runs,
    mgstd5 only (the no-noisy-landscape fact the figure note discloses);
  * per-capita harvest = harvested_fin / career_pop_final and
    career_pop_final == n_final on every run;
  * the ADR-0003 qualification (ii): the fixed_span population is pinned at 20
    and its per-capita harvest is flat across g, while age_hazard_staggered
    grows the population and its per-capita harvest falls monotonically.

Run:  MPLBACKEND=Agg python3 scripts/report_notebook/verify_F2b_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

BASENAME = "F2b_v2_career_flows_noisy_vs_clean"
ROOT = REPO / "data" / "career_flows_adr0003"
SIDECAR = REPO / "output" / "report_notebook_v1" / f"{BASENAME}_data.csv"

EXIT_MODES = ("fixed_span", "age_hazard_staggered")
G_LEVELS = (0.125, 0.25, 0.5)
Z95 = 1.96
TOL = 1e-9


def _mean_ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = v.size
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return m, m - Z95 * se, m + Z95 * se, n


def check(verbose: bool = True) -> int:
    fails: list[str] = []

    files = sorted((ROOT / "cells").glob("career_*.csv"))
    if len(files) != 18:
        fails.append(f"career cells: {len(files)} != 18")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if len(d) != 18 * 20:
        fails.append(f"career rows: {len(d)} != 360")
    if sorted(d.exit_mode.unique()) != sorted(EXIT_MODES):
        fails.append(f"exit grid {sorted(d.exit_mode.unique())}")
    if sorted(d.g.unique()) != list(G_LEVELS):
        fails.append(f"g grid {sorted(d.g.unique())}")
    if sorted(d.landscape.unique()) != ["mgstd5"]:
        fails.append(f"landscape grid {sorted(d.landscape.unique())} "
                     "(the career arm must be mgstd5 only)")

    rel = float((d.harvested_fin / d.career_pop_final
                 - d.career_percap_harvest_final).abs().max())
    if rel > 1e-9:
        fails.append(f"per-capita identity violated (max {rel:g})")
    if not bool((d.career_pop_final == d.n_final).all()):
        fails.append("career_pop_final != n_final")

    side = pd.read_csv(SIDECAR)
    metrics = {"percap": "f2b_percap", "align": "f2b_align", "npop": "f2b_npop"}
    tables = {}
    for mode in EXIT_MODES:
        sub = d[d.exit_mode == mode]
        for g in G_LEVELS:
            s = sub[sub.g == g]
            if len(s) != 60:
                fails.append(f"{mode} g={g}: {len(s)} runs != 60")
            rec = {}
            for key, col in (("percap", "career_percap_harvest_final"),
                             ("align", "c_ideal_fin"), ("npop", "career_pop_final")):
                m, lo, hi, n = _mean_ci(s[col])
                rec[key] = (m, lo, hi, n)
                row = side[(side.metric == metrics[key]) & (side.group == mode)
                           & (np.isclose(side.g, g))]
                if len(row) != 1:
                    fails.append(f"sidecar row missing {key} {mode} g={g}")
                    continue
                r = row.iloc[0]
                for name, got, want in (("value", r.value, m),
                                        ("ci_lo", r.ci_lo, lo),
                                        ("ci_hi", r.ci_hi, hi)):
                    if abs(got - want) > TOL:
                        fails.append(f"{key} {mode} g={g} {name}: "
                                     f"{got:.9g} != {want:.9g}")
            tables[(mode, g)] = rec

    # qualification (ii)
    fixed_n = [tables[("fixed_span", g)]["npop"][0] for g in G_LEVELS]
    stag_n = [tables[("age_hazard_staggered", g)]["npop"][0] for g in G_LEVELS]
    if len(set(fixed_n)) != 1:
        fails.append(f"fixed_span population not pinned: {fixed_n}")
    if not all(stag_n[i] < stag_n[i + 1] for i in range(len(stag_n) - 1)):
        fails.append(f"staggered population not growing: {stag_n}")
    fixed_pc = np.array([tables[("fixed_span", g)]["percap"][0] for g in G_LEVELS])
    stag_pc = np.array([tables[("age_hazard_staggered", g)]["percap"][0]
                        for g in G_LEVELS])
    if np.ptp(fixed_pc) > 0.001:
        fails.append(f"fixed_span per-capita harvest not flat: {fixed_pc}")
    if np.ptp(fixed_pc) >= abs(stag_pc[0] - stag_pc[-1]):
        fails.append(f"fixed_span flatness ({np.ptp(fixed_pc):.4f}) not smaller "
                     f"than the staggered fall "
                     f"({abs(stag_pc[0] - stag_pc[-1]):.4f})")
    if not all(stag_pc[i] > stag_pc[i + 1] for i in range(len(stag_pc) - 1)):
        fails.append(f"staggered per-capita harvest not monotone: {stag_pc}")

    if verbose:
        print(f"cells {len(files)} | runs {len(d)} | landscapes "
              f"{sorted(d.landscape.unique())}")
        print(f"fixed_span  per-capita {fixed_pc[0]:.4f} -> {fixed_pc[-1]:.4f} "
              f"(ptp {np.ptp(fixed_pc):.4f}) | N {fixed_n}")
        print(f"staggered   per-capita {stag_pc[0]:.4f} -> {stag_pc[-1]:.4f} "
              f"| N {stag_n}")
        print(f"sidecar rows {len(side)} | checks failed {len(fails)}")
        for f in fails:
            print("  -", f)
    return len(fails)


if __name__ == "__main__":
    raise SystemExit(1 if check() else 0)
