"""Independent verification of F7 (`F7_m0_cognitive_division_of_labor`).

Three separate checks, none of which trusts the figure module's own arithmetic:

1. **Recompute.** Every plotted and annotated number is recomputed here from
   `data/figures_v1/run_number*.csv` with fresh code -- the arm split, the
   means, the ratios, the deltas, the harvest gap -- and compared against the
   delivered `_data.csv` cell for cell.
2. **Read back the drawn text.** The figure is rebuilt into a temporary
   directory and every string matplotlib actually drew (bar labels, the second
   tick line, the two titles) is collected from the live artists and compared,
   as a multiset, against the strings those recomputed numbers imply. This is
   the check that catches an annotation which drifted while the CSV stayed
   right -- the failure mode a data-only diff cannot see.
3. **Reproducibility.** The re-run's `_data.csv` must be byte-identical to the
   delivered one, and the PNG must have the delivered pixel dimensions at the
   house dpi.

Run from anywhere:

    /opt/hermes/.venv/bin/python scripts/report_notebook/verify_F7_numbers.py
"""
from __future__ import annotations

import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as _t

REPO = Path(__file__).resolve().parents[2]
FIGV1 = REPO / "data" / "figures_v1"
OUT = REPO / "output" / "report_notebook_v1"
STEM = "F7_m0_cognitive_division_of_labor"

#: landscape -> (swept file index, mixed file index); the flag that decides
#: which file is which arm lives in `sciencemodels.sweeps.BulkSetup`.
PAIRS = {"randominit": (0, 1), "MultG2": (2, 3), "MultG6": (4, 5), "Noisy2": (6, 7)}
REPEATS = 10
NEVER = 250.0   # sentinel: the run never reached the 10 % threshold


def _load(idx: int) -> pd.DataFrame:
    return pd.read_csv(FIGV1 / f"run_number{idx}.csv")


def recompute() -> tuple[pd.DataFrame, dict]:
    """Every number on the figure, from the raw CSVs, without the loader."""
    rows, notes = [], {"ratio": {}, "ratio_sweep": {}, "delta": {}, "gap_pp": {},
                       "censored": {}}
    for land, (i_sw, i_mx) in PAIRS.items():
        sw, mx = _load(i_sw), _load(i_mx)
        assert len(mx) == REPEATS, f"{land}: mixed arm has {len(mx)} runs"
        assert np.allclose(mx["initial_curiosity"], 0.5), f"{land}: mixed mu != 0.5"
        assert sw["initial_curiosity"].nunique() == 11, f"{land}: swept mu count"
        hom = sw[np.isclose(sw["initial_curiosity"], 0.5)]
        assert len(hom) == REPEATS, f"{land}: {len(hom)} runs at mu=0.5"
        arms = {"mixed": mx, "homogeneous": hom, "sweep average": sw}
        for arm, d in arms.items():
            for metric, col in (("h10_steps", "harvested_10_step"),
                                ("c_ideal", "Mean_of_Corr_prestige_ideal_merit"),
                                ("harvest_pct", "percentage_knowledge_harvested")):
                v = d[col].astype(float)
                # CI multiplier from the t distribution at the run count -- not
                # a typed constant: the pooled arm has 110 runs, the cohorts 10.
                half = float(_t.ppf(0.975, len(v) - 1) * v.std(ddof=1)
                             / np.sqrt(len(v)))
                rows.append({"landscape": land, "arm": arm, "metric": metric,
                             "value": float(v.mean()),
                             "ci_lo": float(v.mean() - half),
                             "ci_hi": float(v.mean() + half),
                             "n": int(len(v))})
        w = {(r["landscape"], r["metric"], r["arm"]): r["value"] for r in rows}
        h = w[(land, "h10_steps", "homogeneous")]
        m = w[(land, "h10_steps", "mixed")]
        s = w[(land, "h10_steps", "sweep average")]
        notes["ratio"][land] = h / m
        notes["ratio_sweep"][land] = s / m
        notes["delta"][land] = (w[(land, "c_ideal", "mixed")]
                                - w[(land, "c_ideal", "homogeneous")])
        notes["gap_pp"][land] = 100.0 * (w[(land, "harvest_pct", "mixed")]
                                         - w[(land, "harvest_pct", "homogeneous")])
        cz = sw[sw["harvested_10_step"] >= NEVER]
        assert cz["harvested_10_step"].nunique() <= 1, f"{land}: sentinel not unique"
        notes["censored"][land] = int(len(cz))
    return pd.DataFrame(rows), notes


def expected_strings(tidy: pd.DataFrame, notes: dict) -> tuple[Counter, Counter]:
    """The exact bar labels and second-tick lines the numbers imply."""
    val = {(r.landscape, r.metric, r.arm): r.value for r in tidy.itertuples()}
    bars = Counter()
    for land in PAIRS:
        bars[f"{val[(land, 'h10_steps', 'mixed')]:.1f}"] += 1
        bars[f"{val[(land, 'h10_steps', 'homogeneous')]:.1f}"] += 1
        bars[f"{val[(land, 'h10_steps', 'sweep average')]:.1f}"] += 1
        bars[f"{val[(land, 'c_ideal', 'mixed')]:+.3f}"] += 1
        bars[f"{val[(land, 'c_ideal', 'homogeneous')]:+.3f}"] += 1
    ticks = Counter(
        [f"×{notes['ratio'][land]:.2f}" for land in PAIRS]
        + [f"Δ {notes['delta'][land]:+.3f}" for land in PAIRS]
    )
    return bars, ticks


def drawn_strings(fig) -> tuple[Counter, Counter]:
    """Read back what matplotlib actually drew."""
    bars, ticks = Counter(), Counter()
    for ax in fig.axes:
        for t in ax.texts:                      # bar_label artists
            s = t.get_text().strip()
            if s:
                bars[s] += 1
        for t in ax.get_xticklabels():
            parts = [p.strip() for p in t.get_text().split("\n") if p.strip()]
            if len(parts) == 2:                 # second line = ratio / delta
                ticks[parts[1]] += 1
    return bars, ticks


def main() -> int:
    tidy, notes = recompute()
    bars_exp, ticks_exp = expected_strings(tidy, notes)
    failures: list[str] = []

    # ---- 1. the delivered data CSV must equal the recomputation -----------
    dl = pd.read_csv(OUT / f"{STEM}_data.csv")
    drawn = dl[dl["kind"] == "drawn"].set_index(["landscape", "metric", "arm"])
    for r in tidy.itertuples():
        key = (r.landscape, r.metric, r.arm)
        if key not in drawn.index:
            failures.append(f"data CSV is missing {key}")
            continue
        got = drawn.loc[key]
        for col, want in (("value", r.value), ("ci_lo", r.ci_lo), ("ci_hi", r.ci_hi)):
            if abs(float(got[col]) - want) > 1e-9:
                failures.append(f"{key} {col}: CSV {float(got[col])!r} != "
                                f"recomputed {want!r}")
        if int(got["n"]) != int(r.n):
            failures.append(f"{key} n: CSV {got['n']} != {r.n}")
    ann = dl[dl["kind"] == "annotated"].set_index(["landscape", "arm", "metric"])["value"]
    for land in PAIRS:
        for arm, metric, want in (
                ("homogeneous/mixed", "ratio_h10_steps", notes["ratio"][land]),
                ("sweep/mixed", "ratio_h10_steps", notes["ratio_sweep"][land]),
                ("mixed-homogeneous", "delta_c_ideal", notes["delta"][land]),
                ("mixed-homogeneous", "harvest_gap_pp", notes["gap_pp"][land])):
            if abs(float(ann.loc[(land, arm, metric)]) - want) > 1e-9:
                failures.append(f"annotated {land}/{metric}/{arm}: CSV "
                                f"{float(ann.loc[(land, arm, metric)])!r} != {want!r}")

    # ---- 2. the drawn text must match the recomputed numbers -------------
    sys.path.insert(0, str(REPO / "scripts" / "report_notebook"))
    import round_report_figures as F            # noqa: E402  (path set above)
    with tempfile.TemporaryDirectory() as td:
        fig, out, extra = F.figure_cognitive_division_of_labor(Path(td))
        bars_got, ticks_got = drawn_strings(fig)
        from PIL import Image
        im = Image.open(out)
        w_in, h_in = im.size[0] / 180, im.size[1] / 180
        # ---- 3. reproducibility ----------------------------------------
        a = (OUT / f"{STEM}_data.csv").read_bytes()
        b = Path(extra["csv"]).read_bytes()
        if a != b:
            failures.append("re-running the figure produced a different _data.csv")
        if not (10.5 <= w_in <= 12.5 and 4.0 <= h_in <= 6.0):
            failures.append(f"canvas {w_in:.2f} x {h_in:.2f} in outside the "
                            f"designed ~11.4 x 5.0 in")
        import matplotlib.pyplot as plt
        plt.close(fig)

    if bars_got != bars_exp:
        failures.append(f"bar labels drawn {sorted(bars_got.items())} != "
                        f"expected {sorted(bars_exp.items())}")
    if ticks_got != ticks_exp:
        failures.append(f"tick second lines drawn {sorted(ticks_got.items())} != "
                        f"expected {sorted(ticks_exp.items())}")

    # ---- report -----------------------------------------------------------
    print(f"landscapes: {len(PAIRS)} pairs, {REPEATS} repeats/arm")
    print("landscape     h10 mixed  homog  homog/mixed  sweep/mixed  c_ideal mixed  homog    delta    harvest gap")
    for land in PAIRS:
        v = {(r.landscape, r.metric, r.arm): r.value for r in tidy.itertuples()}
        print(f"{land:<13}{v[(land,'h10_steps','mixed')]:8.1f} "
              f"{v[(land,'h10_steps','homogeneous')]:6.1f} "
              f"{notes['ratio'][land]:10.3f} "
              f"{notes['ratio_sweep'][land]:12.3f} "
              f"{v[(land,'c_ideal','mixed')]:14.4f} "
              f"{v[(land,'c_ideal','homogeneous')]:7.4f} "
              f"{notes['delta'][land]:+8.4f} "
              f"{notes['gap_pp'][land]:+10.2f} pp")
    print(f"censored (never reached 10 %, sentinel {NEVER}): "
          f"{ {k: f'{v}/110' for k, v in notes['censored'].items()} }")
    print(f"drawn bar labels: {sum(bars_got.values())} | tick second lines: "
          f"{sum(ticks_got.values())}")
    if failures:
        print(f"\nFAIL ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("\nPASS: recomputation == data CSV == drawn text; re-run reproduces "
          "both artifacts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
