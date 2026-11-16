"""Independent verification of F6 (`F6_two_stage_proxy_failure`).

Re-computes, from the run-level cell files and with code that does not touch the
figure module's loaders, every point of the four extended series and checks it
against the delivered `F6_two_stage_proxy_failure_data.csv`; then re-derives the
footnote's verdict numbers and asserts the drawn text carries them.

Checked:
  * M1 + turnover (age_hazard, entry 0.1, mgstd5): 5 xi points x 2 moments
  * M2 (pull 1.0, logit, equal_split, mgstd5):      5 xi points x 2 lambda x 2

    /opt/hermes/.venv/bin/python output/report_notebook_v1/verify_F6_numbers.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CSV = HERE / "F6_two_stage_proxy_failure_data.csv"

XI = (0.1, 0.3, 0.5, 0.7, 0.9)
TURN_PIN = dict(exit_mode="age_hazard", entry=0.1, landscape="mgstd5", horizon=400)
M2_PIN = dict(dyadic_move_pull=1.0, formulation="logit",
              credit_sharing_axis="equal_split", landscape="mgstd5")


def turnover_runs() -> pd.DataFrame:
    """The five turnover cells, read straight from the run-level files."""
    frames = []
    for pat in ("data/turnover_v1/cells/cell_*.csv",
                "data/fig1_extension_v1/cells/m1_*.csv"):
        for f in sorted(glob.glob(str(REPO / pat))):
            d = pd.read_csv(f)
            if not len(d):
                continue
            d = d[(d.exit_mode == TURN_PIN["exit_mode"])
                  & np.isclose(d.entry, TURN_PIN["entry"])
                  & (d.landscape == TURN_PIN["landscape"])
                  & (d.horizon == TURN_PIN["horizon"])]
            if len(d):
                frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out["xi"] = 1.0 - out.curiosity.round(2)
    return out


def m2_runs() -> pd.DataFrame:
    """The ten M2 cells (five xi x two lambda), read from the run-level files."""
    frames = []
    for pat in ("data/path_b_v2/cells/cell_*.csv",
                "data/fig1_extension_v1/cells/m2_*.csv"):
        for f in sorted(glob.glob(str(REPO / pat))):
            d = pd.read_csv(f)
            d = d[(d.landscape == M2_PIN["landscape"])
                  & np.isclose(d.dyadic_move_pull, M2_PIN["dyadic_move_pull"])
                  & (d.formulation == M2_PIN["formulation"])
                  & (d.credit_sharing_axis == M2_PIN["credit_sharing_axis"])
                  & (d.dyadic_lambda.isin([0.0, 1.0]))]
            if len(d):
                if "xi" not in d.columns:          # archived c = 0.5 cells
                    d = d.assign(xi=0.5)
                frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out["xi"] = out.xi.round(2)
    return out


def main() -> int:
    fails: list[str] = []
    turn, m2 = turnover_runs(), m2_runs()

    # ---- shape guards ---------------------------------------------------
    for name, d, expect in (("turnover", turn, 100), ("M2", m2, 200)):
        if len(d) != expect:
            fails.append(f"{name}: {len(d)} runs, expected {expect} (5 or 10 cells x 20)")
    for xi, d in turn.groupby("xi"):
        if len(d) != 20:
            fails.append(f"turnover xi={xi}: {len(d)} runs, expected 20")
    for (xi, lam), d in m2.groupby(["xi", "dyadic_lambda"]):
        if len(d) != 20:
            fails.append(f"M2 xi={xi} lam={lam}: {len(d)} runs, expected 20")

    # ---- provenance: the new ends must come from this round's folder -----
    ext = pd.concat([turn, m2])
    new_xi = ext[(ext.xi == 0.1) | (ext.xi == 0.9)]
    if "driver_version" in new_xi.columns:
        bad = new_xi[new_xi.driver_version != "fig1_extension_v1 / driver 1"]
        if len(bad):
            fails.append(f"{len(bad)} runs at xi in (0.1, 0.9) are not this "
                         f"round's (driver_version mismatch)")

    # ---- expected curve points ------------------------------------------
    exp = {}
    for xi, d in turn.groupby("xi"):
        exp[("M1 + Turnover (age_hazard)", "L", xi)] = (
            float(d.harvested_fin.mean()), len(d))
        exp[("M1 + Turnover (age_hazard)", "R", xi)] = (
            float(d.c_ideal_fin.mean()), len(d))
    tags = {0.0: "M2 (lambda=0)", 1.0: "M2 (lambda=1)"}
    for (xi, lam), d in m2.groupby(["xi", "dyadic_lambda"]):
        exp[(tags[lam], "L", xi)] = (
            float(d.percentage_knowledge_harvested.mean()), len(d))
        exp[(tags[lam], "R", xi)] = (
            float(d.Mean_of_Corr_prestige_ideal_merit.mean()), len(d))

    # ---- compare against the delivered CSV -------------------------------
    got = pd.read_csv(CSV)
    got["xi"] = got.xi.round(2)
    for (series, panel, xi), (want, n) in exp.items():
        sel = got[(got.series == series) & (got.panel == panel)
                  & np.isclose(got.xi, xi)]
        if len(sel) != 1:
            fails.append(f"{series}/{panel}/xi={xi}: {len(sel)} CSV rows")
            continue
        r = sel.iloc[0]
        if abs(float(r.value) - want) > 1e-9:
            fails.append(f"{series}/{panel}/xi={xi}: CSV {float(r.value)!r} != "
                         f"recomputed {want!r}")
        if int(r.n) != n:
            fails.append(f"{series}/{panel}/xi={xi}: n {int(r.n)} != {n}")
        for folder in ("data/turnover_v1", "data/fig1_extension_v1",
                       "data/path_b_v2"):
            pass  # provenance text is asserted below, per series
    # every extended series must reach both ends of the axis
    for series in ("M1 + Turnover (age_hazard)", "M2 (lambda=0)", "M2 (lambda=1)"):
        have = set(got[got.series == series].xi.round(2))
        if have != set(XI):
            fails.append(f"{series}: xi coverage {sorted(have)} != {sorted(XI)}")
    # provenance strings must name the extension folder for the extended legs
    for series in ("M1 + Turnover (age_hazard)", "M2 (lambda=0)", "M2 (lambda=1)"):
        src = " ".join(got[got.series == series].source_file.unique())
        if "fig1_extension_v1" not in src:
            fails.append(f"{series}: provenance does not name fig1_extension_v1")
        if series.startswith("M1") and "turnover_v1" not in src:
            fails.append(f"{series}: provenance does not name turnover_v1")
        if series.startswith("M2") and "path_b_v2" not in src:
            fails.append(f"{series}: provenance does not name path_b_v2")

    # ---- the footnote's numbers must come from these same frames --------
    sys.path.insert(0, str(HERE))
    import f6_two_stage_proxy_failure as F            # noqa: E402
    macro, micro = got[got.panel == "L"].copy(), got[got.panel == "R"].copy()
    note = F._footnote(macro, micro)
    g2 = macro[macro.series == "MultG2"].sort_values("xi")
    g6 = macro[macro.series == "MultG6"].sort_values("xi")
    l2 = 100 * (g2[g2.xi <= 0.5].value.max() - g2[(g2.xi >= 0.3) & (g2.xi <= 0.6)].value.min())
    l6 = 100 * (g6[g6.xi <= 0.5].value.max() - g6[(g6.xi >= 0.3) & (g6.xi <= 0.6)].value.min())
    turn_micro = micro[micro.series == "M1 + Turnover (age_hazard)"]
    claims = [
        f"{l2:.1f} pp", f"{l6:.1f} pp",
        f"{int((turn_micro.value < 0).sum())}/{len(turn_micro)}",
        f"[{turn_micro.value.min():+.3f}, {turn_micro.value.max():+.3f}]",
        f"{micro[micro.series == 'M2 (lambda=0)'].value.min():+.3f}",
        f"{micro[micro.series == 'M2 (lambda=1)'].value.max():+.3f}",
    ]
    for c in claims:
        if c not in note:
            fails.append(f"footnote does not carry the measured claim {c!r}")

    # ---- report ---------------------------------------------------------
    print(f"turnover runs {len(turn)} | M2 runs {len(m2)} | CSV rows {len(got)}")
    tbl = got[got.series.isin(["M1 + Turnover (age_hazard)", "M2 (lambda=0)",
                               "M2 (lambda=1)"])]
    print(tbl.pivot_table(index="xi", columns=["panel", "series"], values="value")
          .round(3).to_string())
    print(f"\nfootnote claims checked: {len(claims)} | xi coverage: 5 points x 3 series")
    if fails:
        print(f"\nFAIL ({len(fails)}):")
        for f in fails:
            print("  -", f)
        return 1
    print("\nPASS: recomputed cell means == F6 data CSV; footnote numbers are "
          "derived from the same frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
