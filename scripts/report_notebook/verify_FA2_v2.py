"""Independent verifier for ``FA2_v2_drop_decomposition`` (ADR-0003 clock).

``check()`` re-derives every plotted quantity from the RAW cell CSVs --
``data/interventions_adr0003/cells/intA_*.csv`` (spawn sub-arm) and the
``M0`` tier ``data/fig1_xi_adr0003/cells/fig1_tier-M0_*.csv`` -- and compares
the result against ``FA2_v2_drop_decomposition_data.csv``.  It re-implements
the loader and the seed bootstrap inside the verifier (no import of the
builder for the numbers) and asserts:

  * 27 intA cells, 18 in the exact/random sub-arm at 20 seeds each;
  * every cell's ``state/<cell_tag>.json`` agrees on g / spawn / exit / tau_max;
  * the M0 reference exists at c in {0.3, 0.5, 0.7} with 20 seeds each;
  * the decomposition identity (shares sum to 100%) and the plotted levels;
  * the figure's qualitative claim: on the time-mean measure the random-spawn
    arm beats exact spawn at every (c, g) -- starvation share > 0 -- which is
    the reverse of the pre-ADR terminal-step reading.

Run:  MPLBACKEND=Agg python3 scripts/report_notebook/verify_FA2_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

BASENAME = "FA2_v2_drop_decomposition"
SPAWN_ROOT = REPO / "data" / "interventions_adr0003"
M0_ROOT = REPO / "data" / "fig1_xi_adr0003"
SIDECAR = REPO / "output" / "report_notebook_v1" / f"{BASENAME}_data.csv"

CURIOSITIES = (0.3, 0.5, 0.7)
G_LEVELS = (0.125, 0.25, 0.5)
SPAWNS = ("exact", "random")
PRIMARY = "c_ideal_tm"
Z95 = 1.96
N_BOOT = 4000
BOOT_SEED = 20261126
TOL = 1e-9


def _check(cond: bool, msg: str, fails: list[str]) -> None:
    if not cond:
        fails.append(msg)


def _mean_ci(v):
    v = np.asarray(v, float)
    n = v.size
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(n))
    return m, m - Z95 * se, m + Z95 * se, n


def _raw():
    files = sorted((SPAWN_ROOT / "cells").glob("intA_*.csv"))
    cells = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cells = cells[(cells.refill_mode == "off") & (cells.landscape == "mgstd5")
                  & (cells.spawn.isin(SPAWNS))]
    m0 = pd.concat([pd.read_csv(f) for f in
                    sorted((M0_ROOT / "cells").glob("fig1_tier-M0_*.csv"))],
                   ignore_index=True)
    m0 = m0[m0.curiosity.isin(CURIOSITIES)]
    return len(files), cells, m0


def _shares(cells, m0, meas=PRIMARY):
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for c in CURIOSITIES:
        a_arr = m0[(m0.curiosity == c)][meas].to_numpy(float)
        a = float(a_arr.mean())
        for g in G_LEVELS:
            e_arr = cells[(cells.curiosity == c) & (cells.g == g)
                          & (cells.spawn == "exact")][meas].to_numpy(float)
            r_arr = cells[(cells.curiosity == c) & (cells.g == g)
                          & (cells.spawn == "random")][meas].to_numpy(float)
            e, r = float(e_arr.mean()), float(r_arr.mean())
            drop = a - e
            bs, bh = [], []
            for _ in range(N_BOOT):
                aa = rng.choice(a_arr, e_arr.size)
                ee = rng.choice(e_arr, e_arr.size)
                rr = rng.choice(r_arr, e_arr.size)
                dd = aa.mean() - ee.mean()
                if abs(dd) > 1e-9:
                    bs.append(100.0 * (rr.mean() - ee.mean()) / dd)
                    bh.append(100.0 * (aa.mean() - rr.mean()) / dd)
            rows.append({"curiosity": c, "g": g, "m0": a, "exact": e,
                         "random": r, "drop": drop,
                         "starvation": 100.0 * (r - e) / drop,
                         "herding": 100.0 * (a - r) / drop,
                         "starv_lo": float(np.percentile(bs, 2.5)),
                         "starv_hi": float(np.percentile(bs, 97.5))})
    return pd.DataFrame(rows)


def check(verbose: bool = True) -> int:
    fails: list[str] = []
    n_files, cells, m0 = _raw()
    _check(n_files == 27, f"intA cells: {n_files} != 27", fails)
    _check(len(cells) == 18 * 20, f"spawn rows: {len(cells)} != 360", fails)
    _check(sorted(cells.curiosity.unique()) == list(CURIOSITIES),
           "curiosity grid", fails)
    _check(sorted(cells.g.unique()) == list(G_LEVELS), "g grid", fails)
    _check(sorted(cells.exit_mode.unique()) == ["fixed_span"], "exit grid", fails)
    _check(len(m0) == 3 * 20, f"M0 rows: {len(m0)} != 60", fails)

    # design records
    for tag in sorted(cells.cell_tag.unique()):
        p = SPAWN_ROOT / "state" / f"{tag}.json"
        if not p.is_file():
            fails.append(f"missing design record {tag}")
            continue
        spec = json.loads(p.read_text())["spec"]
        sub = cells[cells.cell_tag == tag]
        for key, val in (("g", sub.g.iloc[0]), ("spawn", sub.spawn.iloc[0]),
                         ("exit", sub.exit_mode.iloc[0]),
                         ("tau_max", int(sub.tau_max.iloc[0]))):
            if spec[key] != val:
                fails.append(f"{tag}: design {key}={spec[key]} != cell {val}")

    side = pd.read_csv(SIDECAR)
    sh = _shares(cells, m0)

    # levels
    for c in CURIOSITIES:
        for sp in SPAWNS:
            for g in G_LEVELS:
                v = cells[(cells.curiosity == c) & (cells.g == g)
                          & (cells.spawn == sp)][PRIMARY].to_numpy(float)
                m, lo, hi, n = _mean_ci(v)
                row = side[(side.metric == f"f8_levels_c{c:.1f}")
                           & (side.spawn == sp) & (np.isclose(side.g, g))]
                if len(row) != 1:
                    fails.append(f"sidecar levels row missing c={c} {sp} g={g}")
                    continue
                r = row.iloc[0]
                for name, got, want in (("value", r.value, m), ("ci_lo", r.ci_lo, lo),
                                        ("ci_hi", r.ci_hi, hi)):
                    if abs(got - want) > TOL:
                        fails.append(f"levels c={c} {sp} g={g} {name}: "
                                     f"{got:.9g} != {want:.9g}")
                if int(r.n) != n:
                    fails.append(f"levels c={c} {sp} g={g}: n {r.n} != {n}")

    # M0 reference
    for c in CURIOSITIES:
        v = m0[m0.curiosity == c][PRIMARY].to_numpy(float)
        m, lo, hi, n = _mean_ci(v)
        row = side[(side.metric == "f8_m0_reference")
                   & (np.isclose(side.curiosity, c))]
        if len(row) != 1:
            fails.append(f"M0 reference row missing c={c}")
            continue
        if abs(row.value.iloc[0] - m) > TOL:
            fails.append(f"M0 c={c}: {row.value.iloc[0]:.9g} != {m:.9g}")

    # shares + identity + bootstrap CIs
    for r in sh.itertuples():
        row = side[(side.metric == "f8_shares")
                   & (np.isclose(side.curiosity, r.curiosity))
                   & (np.isclose(side.g, r.g))]
        if len(row) != 2:
            fails.append(f"share rows missing c={r.curiosity} g={r.g}")
            continue
        for comp in ("starvation", "herding"):
            got = float(row[row.series == comp].value.iloc[0])
            if abs(got - getattr(r, comp)) > TOL:
                fails.append(f"share c={r.curiosity} g={r.g} {comp}: "
                             f"{got:.9g} != {getattr(r, comp):.9g}")
        note = str(row[row.series == "starvation"].note.iloc[0])
        if f"{r.starv_lo:+.0f}" not in note or f"{r.starv_hi:+.0f}" not in note:
            fails.append(f"bootstrap CI for c={r.curiosity} g={r.g} not "
                         f"reproduced in the sidecar note: {note}")
    _check(np.allclose(sh.starvation + sh.herding, 100.0, atol=1e-9),
           "shares do not sum to 100%", fails)

    # the figure's claim: random beats exact at every (c, g)
    if not (sh.starvation > 0).all():
        fails.append("random does not beat exact at every (c, g): "
                     f"{sh[['curiosity', 'g', 'starvation']].to_dict('records')}")

    # pre-ADR reference for the drift statement (both measures, same code path)
    legacy_sp = REPO / "data/interventions_v2/tables/arm_a_spawn.csv"
    legacy_runs = REPO / "data/interventions_v2/runs.csv"
    legacy_txt = ""
    if legacy_sp.is_file() and legacy_runs.is_file():
        tbl = pd.read_csv(legacy_sp)
        tbl = tbl[(tbl.landscape == "mgstd5") & (tbl.r_spawn.isin(SPAWNS))]
        runs = pd.read_csv(legacy_runs)
        runs = runs[(runs.arm == "C") & (runs.agent_generation_time == 0)
                    & (runs.landscape == "mgstd5") & (runs.refill_mode == "off")]
        old = []
        for c in CURIOSITIES:
            a = float(runs[runs.initial_curiosity == c]
                      .corr_prestige_ideal_merit_final.mean())
            e = float(tbl[(tbl.curiosity == c) & (tbl.r_spawn == "exact")]
                      .c_ideal_fin_post.iloc[0])
            r = float(tbl[(tbl.curiosity == c) & (tbl.r_spawn == "random")]
                      .c_ideal_fin_post.iloc[0])
            old.append(100.0 * (r - e) / (a - e))
        legacy_txt = (" | pre-ADR (g=0.1, terminal) starvation "
                      + "/".join(f"{v:+.1f}%" for v in old))

    if verbose:
        print(f"intA cells {n_files} | spawn rows {len(cells)} | M0 rows {len(m0)}")
        print(sh.round(3).to_string(index=False))
        print("previous-archive reference:" + legacy_txt)
        print(f"sidecar rows {len(side)} | checks failed {len(fails)}")
        for f in fails:
            print("  -", f)
    return len(fails)


if __name__ == "__main__":
    raise SystemExit(1 if check() else 0)
