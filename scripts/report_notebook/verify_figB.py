"""Independent verifier for Figure B (F12_sabbaticals_vs_lotteries).

``check()`` recomputes every plotted quantity and every annotation value in
``F12_sabbaticals_vs_lotteries_data.csv`` straight from the RAW cell CSVs
(``data/interventions_v1/cells/cellB_*.csv`` -- never from the sidecar),
asserts the expected input counts (36 cells x 20 seeds = 720 runs) and
compares at ``tol``.  Empty return list = all checks pass.

Run:  MPLBACKEND=Agg python3 -c "import verify_figB; print(verify_figB.check())"
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CELLS = REPO / "data" / "interventions_v1" / "cells"
FNAME = "F12_sabbaticals_vs_lotteries"

METRIC_COL = {
    "harvested knowledge share": "percentage_knowledge_harvested",
    "explored grid share": "explored_final",
    "terminal 1-step loss (-ideal_merit_sum_final)": "ideal_merit_sum_final",
    "Corr(prestige, ideal merit) (time mean)": "Mean_of_Corr_prestige_ideal_merit",
    "top-10% prestige share": "top10_final",
}
XIS = (0.5, 0.7, 0.9)
ALPHAS = (0.0, 0.5, 1.0)
PRELOCS = (0.00, 0.02)


def _load() -> pd.DataFrame:
    files = sorted(CELLS.glob("cellB_*.csv"))
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["source_file"] = f.name
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def _sel(b: pd.DataFrame, xi: float, alpha: float, preloc: float,
         land: str) -> pd.DataFrame:
    s = (b["visibility_weight"] == xi) & (b["alpha"] == alpha) \
        & (b["p_reloc"] == preloc)
    if land != "pooled":
        s &= b["landscape"] == land
    return b[s]


def _mean(b: pd.DataFrame, col: str, xi: float, alpha: float, preloc: float,
          land: str = "pooled") -> float:
    d = _sel(b, xi, alpha, preloc, land)
    v = -d[col] if col == "ideal_merit_sum_final" else d[col]
    return float(v.mean())


def check(outdir: str = "output/report_notebook_v1", tol: float = 1e-9) -> list[str]:
    fails: list[str] = []
    out = (REPO / outdir) if not Path(outdir).is_absolute() else Path(outdir)
    sidecar_path = out / f"{FNAME}_data.csv"
    if not sidecar_path.exists():
        return [f"missing sidecar: {sidecar_path}"]
    for ext in ("png", "pdf"):
        p = out / f"{FNAME}.{ext}"
        if not p.exists():
            fails.append(f"missing figure file: {p}")
    side = pd.read_csv(sidecar_path)
    b = _load()

    # ---- input counts -------------------------------------------------
    files = sorted(CELLS.glob("cellB_*.csv"))
    if len(files) != 36:
        fails.append(f"cellB files: expected 36, got {len(files)}")
    if len(b) != 720:
        fails.append(f"runs: expected 720, got {len(b)}")
    for f in files:
        n = int((b["source_file"] == f.name).sum())
        if n != 20:
            fails.append(f"{f.name}: expected 20 seeds, got {n}")
    if sorted(b["alpha"].unique().tolist()) != list(ALPHAS):
        fails.append(f"alpha grid changed: {sorted(b['alpha'].unique())}")
    if sorted(b["p_reloc"].unique().tolist()) != list(PRELOCS):
        fails.append(f"p_reloc grid changed: {sorted(b['p_reloc'].unique())}")
    if sorted(b["visibility_weight"].unique().tolist()) != list(XIS):
        fails.append(f"xi grid changed: {sorted(b['visibility_weight'].unique())}")

    # ---- series rows: recompute mean + 95% CI from raw runs -----------
    srows = side[side["statistic"] == "mean"]
    for _, r in srows.iterrows():
        col = METRIC_COL.get(r["metric"])
        if col is None:
            fails.append(f"unknown metric in sidecar: {r['metric']}")
            continue
        d = _sel(b, float(r["xi"]), float(r["alpha"]), float(r["p_reloc"]),
                 str(r["landscape"]))
        v = -d[col] if col == "ideal_merit_sum_final" else d[col]
        exp_n = 40 if r["landscape"] == "pooled" else 20
        if len(v) != exp_n or int(r["n"]) != exp_n:
            fails.append(f"n mismatch {r['metric']} {r['xi']} {r['alpha']} "
                         f"{r['p_reloc']} {r['landscape']}: "
                         f"raw {len(v)} vs sidecar {r['n']}")
            continue
        mean = float(v.mean())
        ci = 1.96 * float(v.std(ddof=1)) / np.sqrt(len(v))
        for label, got, want in (("value", r["value"], mean),
                                 ("ci_lo", r["ci_lo"], mean - ci),
                                 ("ci_hi", r["ci_hi"], mean + ci)):
            if abs(float(got) - want) > tol:
                fails.append(f"{r['metric']} {r['xi']} {r['alpha']} "
                             f"{r['p_reloc']} {r['landscape']} {label}: "
                             f"sidecar {got} vs recomputed {want}")

    # ---- annotation rows: recompute the formulas ----------------------
    ann = side[side["statistic"] == "annotation"].set_index("metric")
    h = "percentage_knowledge_harvested"
    L = "ideal_merit_sum_final"
    expected: dict[str, float] = {}
    expected["mgstd5_stage2_harvest_rescue_pp"] = 100.0 * (
        _mean(b, h, 0.9, 1.0, 0.02, "mgstd5")
        - _mean(b, h, 0.9, 1.0, 0.00, "mgstd5"))
    for land, key in (("mgstd5", "mgstd5_stage2_harvest_rescue_pp_all_alpha"),
                      ("pooled", "pooled_stage2_harvest_rescue_pp_all_alpha")):
        s = b[b["visibility_weight"] == 0.9]
        if land != "pooled":
            s = s[s["landscape"] == land]
        expected[key] = 100.0 * (s[s["p_reloc"] == 0.02][h].mean()
                                 - s[s["p_reloc"] == 0.00][h].mean())
    expected["pooled_stage2_harvest_rescue_pp"] = 100.0 * (
        _mean(b, h, 0.9, 1.0, 0.02) - _mean(b, h, 0.9, 1.0, 0.00))
    expected["lottery_harvest_effect_pp_max"] = max(
        abs(100.0 * (_mean(b, h, 0.9, 0.0, p) - _mean(b, h, 0.9, 1.0, p)))
        for p in PRELOCS)
    expected["explored_xi09_baseline"] = _mean(b, "explored_final", 0.9, 1.0, 0.00)
    expected["explored_xi09_sabbatical"] = _mean(b, "explored_final", 0.9, 1.0, 0.02)
    expected["explored_xi09_fold"] = (expected["explored_xi09_sabbatical"]
                                      / expected["explored_xi09_baseline"])
    pen_b = (_mean(b, L, 0.9, 1.0, 0.00) - _mean(b, L, 0.5, 1.0, 0.00))
    pen_s = (_mean(b, L, 0.9, 1.0, 0.02) - _mean(b, L, 0.5, 1.0, 0.02))
    pen_l = (_mean(b, L, 0.9, 0.0, 0.00) - _mean(b, L, 0.5, 0.0, 0.00))
    expected["high_visibility_penalty_baseline"] = pen_b
    expected["high_visibility_penalty_sabbatical"] = pen_s
    expected["penalty_reduction_pct_sabbatical"] = 100.0 * (pen_s / pen_b - 1.0)
    expected["loss_xi09_reduction_pct_sabbatical"] = 100.0 * (
        _mean(b, L, 0.9, 1.0, 0.02) / _mean(b, L, 0.9, 1.0, 0.00) - 1.0)
    expected["penalty_reduction_pct_lottery"] = 100.0 * (pen_l / pen_b - 1.0)
    expected["corr_tm_xi09_baseline"] = _mean(
        b, "Mean_of_Corr_prestige_ideal_merit", 0.9, 1.0, 0.00)
    expected["corr_tm_xi09_sabbatical"] = _mean(
        b, "Mean_of_Corr_prestige_ideal_merit", 0.9, 1.0, 0.02)
    expected["top10_xi09_baseline"] = _mean(b, "top10_final", 0.9, 1.0, 0.00)
    expected["top10_xi09_lottery"] = _mean(b, "top10_final", 0.9, 0.0, 0.00)
    expected["top10_xi09_sabbatical"] = _mean(b, "top10_final", 0.9, 1.0, 0.02)

    for key, want in expected.items():
        if key not in ann.index:
            fails.append(f"missing annotation row: {key}")
            continue
        got = float(ann.loc[key, "value"])
        if abs(got - want) > tol:
            fails.append(f"annotation {key}: sidecar {got} vs recomputed {want}")
    extra = set(ann.index) - set(expected)
    if extra:
        fails.append(f"unexpected annotation rows: {sorted(extra)}")
    return fails


if __name__ == "__main__":
    for f in check():
        print("FAIL:", f)
    print("all checks pass" if not check() else "FAILURES PRESENT")
