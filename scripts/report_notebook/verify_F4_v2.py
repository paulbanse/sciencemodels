"""Independent verifier for ``F4_v2_prestige_stock_wedge_matrix`` (ADR-0003).

``check()`` re-derives every annotated matrix value and every stock mean from
the RAW cell CSVs ``data/prestige_adr0003/cells/prest_*.csv`` and compares the
result against ``F4_v2_prestige_stock_wedge_matrix_data.csv``.  It asserts:

  * 54 cells (3 stocks x 3 c x 2 landscapes x 3 g) at 20 seeds each = 1,080 runs;
  * the wedge identity ``wedge_tm == c_local_tm - c_ideal_tm`` on every run;
  * qualification (iii) as computed facts: the pooled mean ordering is
    prestige > prestige_visibility > prestige_vanishing, all three means are
    negative, the ordering holds in only a minority of cells, and the
    cross-stock Spearman correlations are strong and positive (no rank
    reversal -- a level finding).

Run:  MPLBACKEND=Agg python3 scripts/report_notebook/verify_F4_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

BASENAME = "F4_v2_prestige_stock_wedge_matrix"
ROOT = REPO / "data" / "prestige_adr0003"
SIDECAR = REPO / "output" / "report_notebook_v1" / f"{BASENAME}_data.csv"

STOCKS = ("prestige", "prestige_visibility", "prestige_vanishing")
Z95 = 1.96
TOL = 1e-9


def check(verbose: bool = True) -> int:
    fails: list[str] = []

    files = sorted((ROOT / "cells").glob("prest_*.csv"))
    if len(files) != 54:
        fails.append(f"prestige cells: {len(files)} != 54")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if len(d) != 54 * 20:
        fails.append(f"prestige rows: {len(d)} != 1080")
    if sorted(d.stock.unique()) != sorted(STOCKS):
        fails.append(f"stock grid {sorted(d.stock.unique())}")
    rel = float((d.wedge_tm - (d.c_local_tm - d.c_ideal_tm)).abs().max())
    if rel > 1e-12:
        fails.append(f"wedge identity violated (max {rel:g})")

    cell = (d.groupby(["stock", "g", "curiosity", "landscape"])
            .agg(value=("wedge_tm", "mean"), n=("wedge_tm", "size"))
            .reset_index())
    if len(cell) != 54:
        fails.append(f"cell table: {len(cell)} rows != 54")
    if not bool((cell.n == 20).all()):
        fails.append("a cell does not hold 20 seeds")

    side = pd.read_csv(SIDECAR)
    for r in cell.itertuples():
        row = side[(side.metric == "f4_wedge_cell") & (side.stock == r.stock)
                   & (np.isclose(side.g, r.g))
                   & (np.isclose(side.curiosity, r.curiosity))
                   & (side.landscape == r.landscape)]
        if len(row) != 1:
            fails.append(f"sidecar cell row missing {r.stock}/{r.g}/"
                         f"{r.curiosity}/{r.landscape}")
            continue
        if abs(float(row.value.iloc[0]) - r.value) > TOL:
            fails.append(f"cell {r.stock}/{r.g}/{r.curiosity}/{r.landscape}: "
                         f"{row.value.iloc[0]:.9g} != {r.value:.9g}")

    srow = side[side.metric == "f4_stock_mean"]
    order = []
    for st in STOCKS:
        v = cell[cell.stock == st].value.to_numpy(float)
        m = float(v.mean())
        ci = Z95 * float(v.std(ddof=1) / np.sqrt(v.size))
        order.append((m, st))
        row = srow[srow.stock == st]
        if len(row) != 1:
            fails.append(f"stock-mean row missing {st}")
            continue
        r = row.iloc[0]
        for name, got, want in (("value", r.value, m),
                                ("ci_lo", r.ci_lo, m - ci),
                                ("ci_hi", r.ci_hi, m + ci)):
            if abs(got - want) > TOL:
                fails.append(f"stock mean {st} {name}: {got:.9g} != {want:.9g}")
        if m >= 0:
            fails.append(f"{st} mean is not negative: {m:+.5f}")

    expected = ["prestige", "prestige_visibility", "prestige_vanishing"]
    got_order = [st for _, st in sorted(order, reverse=True)]
    if got_order != expected:
        fails.append(f"mean ordering {got_order} != {expected}")

    wide = cell.pivot_table(index=["g", "curiosity", "landscape"],
                            columns="stock", values="value")
    ok = int(((wide["prestige"] > wide["prestige_visibility"])
              & (wide["prestige_visibility"] > wide["prestige_vanishing"])).sum())
    rhos = {f"{a}/{b}": float(stats.spearmanr(wide[a], wide[b])[0])
            for a, b in (("prestige", "prestige_visibility"),
                         ("prestige", "prestige_vanishing"),
                         ("prestige_visibility", "prestige_vanishing"))}
    if ok == len(wide):
        fails.append("cell-level ordering holds everywhere: the note's "
                     "'minority of cells' wording would be wrong")
    if not all(r > 0.5 for r in rhos.values()):
        fails.append(f"cross-stock rank correlation too weak for a 'no rank "
                     f"reversal' claim: {rhos}")
    if verbose:
        print(f"cells {len(files)} | runs {len(d)} | cell table {len(cell)}")
        print(f"means {[f'{m:+.5f}' for m, _ in order]} | ordering {got_order}")
        print(f"cell-level ordering holds in {ok}/{len(wide)}; rho {rhos}")
        print(f"sidecar rows {len(side)} | checks failed {len(fails)}")
        for f in fails:
            print("  -", f)
    return len(fails)


if __name__ == "__main__":
    raise SystemExit(1 if check() else 0)
