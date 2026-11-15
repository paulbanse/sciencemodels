"""Path D typology analysis (2026-11-15, P-0005): the 4-quadrant typology
table and the control-arm checks, computed from data/typology_v1 artefacts.

Measures (definitions pinned in the driver's design.json):
* Corr_prestige_ideal_merit / Corr_prestige_local_merit: the model's
  per-step prestige-merit correlations, time-mean over the run (the
  Mean_of_ convention of sciencemodels/io.py build_report);
* wedge_fin = c_ideal_fin - c_local_fin on FINAL-step correlations
  (remeasure-round convention; positive = prestige tracks ideal merit
  better than local merit);
* ideal_loss_cum = -sum(a.ideal_merit_cum) over final scientists
  (remeasure-round convention; cumulative ideal harvest shortfall).

Quadrants (commission naming):
  (gamma=0, kappa=0) Pure Distraction
  (gamma=1, kappa=0) Distraction + Destruction
  (gamma=0, kappa=1) Distraction + Counterforce
  (gamma=1, kappa=1) Full Interaction

Control arm (built into the grid by construction): at g_entry=0.0 no
newcomers exist, so kappa has no object - the (gamma, kappa=0, e=0) and
(gamma, kappa=1, e=0) cells must be BIT-IDENTICAL per seed across EVERY
column.  Verified here over all 120 paired runs (2 gamma x 3 curiosity x
20 seeds); any difference is a wiring bug, not a finding.

Gross vs net: the cell count is recounted from disk (glob + line counts),
never taken from the manifest's memory.

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/typology/analyse_typology_v1.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data" / "typology_v1"
CELLS_DIR = OUT_DIR / "cells"

GAMMAS = (0.0, 1.0)
CURIOSITIES = (0.3, 0.5, 0.7)
SEEDS = 20

QUADRANT_NAMES = {
    (0.0, 0.0): "Pure Distraction",
    (1.0, 0.0): "Distraction + Destruction",
    (0.0, 1.0): "Distraction + Counterforce",
    (1.0, 1.0): "Full Interaction",
}

MEASURES = {
    "Mean_of_Corr_prestige_ideal_merit": "Corr_prestige_ideal_merit",
    "Mean_of_Corr_prestige_local_merit": "Corr_prestige_local_merit",
    "wedge_fin": "wedge_fin",
    "ideal_loss_cum": "ideal_loss_cum",
}


def quad_label(g, k):
    return QUADRANT_NAMES[(float(g), float(k))]


def _quad_row(g, k, sub):
    row = {
        "gamma": float(g),
        "kappa": float(k),
        "quadrant": quad_label(g, k),
        "n_runs": int(len(sub)),
    }
    for src, dst in MEASURES.items():
        row[dst + "_mean"] = float(sub[src].mean())
        row[dst + "_sd"] = float(sub[src].std(ddof=1))
    row["agents_final_mean"] = float(sub["agents_final"].mean())
    row["newcomers_final_mean"] = float(sub["newcomers_final"].mean())
    return row


def main():
    # ---- gross vs net, recounted from disk -------------------------------
    cell_files = sorted(CELLS_DIR.glob("cell_*.csv"))
    agent_files = sorted((CELLS_DIR / "agents").glob("cell_*.csv"))
    rows_net = sum(sum(1 for _ in open(p)) - 1 for p in cell_files)
    agent_rows_net = sum(sum(1 for _ in open(p)) - 1 for p in agent_files)
    runs = pd.read_csv(OUT_DIR / "runs.csv")
    gross_cells = 2 * 2 * 3 * 2
    gross_runs = gross_cells * SEEDS
    assert len(cell_files) == gross_cells, "cell file count %d != %d" % (
        len(cell_files), gross_cells)
    assert len(agent_files) == gross_cells, "agent-file count %d != %d" % (
        len(agent_files), gross_cells)
    assert len(runs) == rows_net == gross_runs, (
        "panel shape broken: runs.csv=%d cells recount=%d gross=%d"
        % (len(runs), rows_net, gross_runs))
    # per-cell row-count guard: a probe or interrupted unit must not pass
    for p in cell_files:
        n = sum(1 for _ in open(p)) - 1
        assert n == SEEDS, "%s carries %d rows" % (p.name, n)

    # ---- the 4-quadrant typology table (collapse c and g) -----------------
    rows = [_quad_row(g, k, sub) for (g, k), sub in runs.groupby(["gamma", "kappa"], sort=True)]
    quad = pd.DataFrame(rows)
    quad.to_csv(OUT_DIR / "typology_table.csv", index=False)

    # quadrant x curiosity and quadrant x entry gradients
    rows = []
    for (g, k, c), sub in runs.groupby(["gamma", "kappa", "curiosity"], sort=True):
        r = _quad_row(g, k, sub)
        r["curiosity"] = float(c)
        rows.append(r)
    by_c = pd.DataFrame(rows)
    by_c.to_csv(OUT_DIR / "typology_by_curiosity.csv", index=False)
    rows = []
    for (g, k, e), sub in runs.groupby(["gamma", "kappa", "g_entry"], sort=True):
        r = _quad_row(g, k, sub)
        r["g_entry"] = float(e)
        rows.append(r)
    by_e = pd.DataFrame(rows)
    by_e.to_csv(OUT_DIR / "typology_by_entry.csv", index=False)

    # full 24-cell summary
    rows = []
    for (g, k, c, e), sub in runs.groupby(["gamma", "kappa", "curiosity", "g_entry"], sort=True):
        r = _quad_row(g, k, sub)
        r.update(curiosity=float(c), g_entry=float(e))
        rows.append(r)
    cells = pd.DataFrame(rows)
    cells.to_csv(OUT_DIR / "cell_summary.csv", index=False)

    # ---- control arm: kappa bit-identical at g=0 --------------------------
    e0 = runs[runs["g_entry"] == 0.0]
    pairs_checked, worst_diff, worst_col, mismatches = 0, 0.0, None, []

    def _num(v):
        return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)

    for g in GAMMAS:
        for c in CURIOSITIES:
            a = e0[(e0["gamma"] == g) & (e0["kappa"] == 0.0) & (e0["curiosity"] == c)].set_index("seed")
            b = e0[(e0["gamma"] == g) & (e0["kappa"] == 1.0) & (e0["curiosity"] == c)].set_index("seed")
            assert set(a.index) == set(b.index), "kappa control pair incomplete at g=%s c=%s" % (g, c)
            for s in a.index:
                ra, rb = a.loc[s], b.loc[s]
                pairs_checked += 1
                # kappa is the treatment label, not a dynamics output: the
                # two arms differ there BY DESIGN.  Compare everything else.
                if ra.drop(labels=["kappa"]).equals(rb.drop(labels=["kappa"])):
                    continue
                for col in ra.index:  # ra.index = value columns (seed is the pair key)
                    if col == "kappa":
                        continue
                    if _num(va) and _num(vb):
                        if pd.isna(va) and pd.isna(vb):
                            continue
                        d = abs(float(va) - float(vb))
                        if d > worst_diff:
                            worst_diff, worst_col = d, col
                    elif str(va) != str(vb):
                        mismatches.append((g, c, s, col))
    control = {
        "kappa_inert_at_g0_pairs_checked": pairs_checked,
        "kappa_inert_at_g0_max_abs_diff": float(worst_diff),
        "kappa_inert_at_g0_worst_column": worst_col,
        "kappa_inert_at_g0_non_numeric_mismatches": mismatches[:10],
        "kappa_inert_at_g0_pass": bool(worst_diff == 0.0 and not mismatches),
    }
    assert control["kappa_inert_at_g0_pass"], "CONTROL ARM BROKEN: %s" % control

    # gamma binding descriptive at g=0 (destruction acts on founders too);
    # matched within kappa levels (both arms carry both kappas at g=0)
    a = e0[e0["gamma"] == 0.0].set_index(["curiosity", "g_entry", "seed", "kappa"])
    b = e0[e0["gamma"] == 1.0].set_index(["curiosity", "g_entry", "seed", "kappa"])
    common = a.index.intersection(b.index)
    gamma_binds = {
        "g0_pairs": int(len(common)),
        "wedge_fin_differs_frac_gamma1_vs_0": float(
            np.mean([float(a.loc[i, "wedge_fin"]) != float(b.loc[i, "wedge_fin"]) for i in common])
        ),
        "wedge_fin_mean_delta": float(
            np.mean([float(b.loc[i, "wedge_fin"]) - float(a.loc[i, "wedge_fin"]) for i in common])
        ),
    }

    # kappa binding descriptive at g=0.1 (the channel's live regime)
    e1 = runs[runs["g_entry"] == 0.1]
    a = e1[e1["kappa"] == 0.0].set_index(["gamma", "curiosity", "seed"])
    b = e1[e1["kappa"] == 1.0].set_index(["gamma", "curiosity", "seed"])
    common = a.index.intersection(b.index)
    kappa_binds = {
        "g01_runs_compared": int(len(common)),
        "wedge_fin_differs_frac": float(
            np.mean([float(a.loc[i, "wedge_fin"]) != float(b.loc[i, "wedge_fin"]) for i in common])
        ),
        "wedge_fin_mean_delta": float(
            np.mean([float(b.loc[i, "wedge_fin"]) - float(a.loc[i, "wedge_fin"]) for i in common])
        ),
    }

    # ---- paired factorial contrasts (seed-paired bootstrap CIs) -----------
    # Pairs are matched on (curiosity, g_entry, seed) across quadrants; the
    # destruction and counterforce mains additionally match on the other
    # factor.  Bootstrap: 20000 percentile resamples of the paired deltas.
    rng_ci = np.random.default_rng(20261115)

    def boot_ci(deltas, n_boot=20000):
        d = np.asarray(deltas, dtype=float)
        idx = rng_ci.integers(0, len(d), size=(n_boot, len(d)))
        means = d[idx].mean(axis=1)
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    contrast_rows = []
    for src, meas in MEASURES.items():
        a = runs[runs["gamma"] == 0.0].set_index(["curiosity", "g_entry", "seed", "kappa"])
        b = runs[runs["gamma"] == 1.0].set_index(["curiosity", "g_entry", "seed", "kappa"])
        common = a.index.intersection(b.index)
        d = (b.loc[common, src] - a.loc[common, src]).to_numpy(float)
        lo, hi = boot_ci(d)
        contrast_rows.append(dict(
            measure=meas, contrast="destruction_main (gamma 0->1, kappa matched)",
            n_pairs=len(d), mean_delta=float(d.mean()), ci_lo=lo, ci_hi=hi,
            frac_positive=float((d > 0).mean())))
        a = runs[runs["kappa"] == 0.0].set_index(["curiosity", "g_entry", "seed", "gamma"])
        b = runs[runs["kappa"] == 1.0].set_index(["curiosity", "g_entry", "seed", "gamma"])
        common = a.index.intersection(b.index)
        d = (b.loc[common, src] - a.loc[common, src]).to_numpy(float)
        lo, hi = boot_ci(d)
        contrast_rows.append(dict(
            measure=meas, contrast="counterforce_main (kappa 0->1, gamma matched)",
            n_pairs=len(d), mean_delta=float(d.mean()), ci_lo=lo, ci_hi=hi,
            frac_positive=float((d > 0).mean())))
        q = {(float(g2), float(k2)): sub for (g2, k2), sub in runs.groupby(["gamma", "kappa"])}
        base = q[(0.0, 0.0)].set_index(["curiosity", "g_entry", "seed"])
        t1 = q[(1.0, 0.0)].set_index(["curiosity", "g_entry", "seed"])
        t2 = q[(0.0, 1.0)].set_index(["curiosity", "g_entry", "seed"])
        t3 = q[(1.0, 1.0)].set_index(["curiosity", "g_entry", "seed"])
        common = base.index.intersection(t1.index).intersection(t2.index).intersection(t3.index)
        d = (
            (t3.loc[common, src] - t1.loc[common, src])
            - (t2.loc[common, src] - base.loc[common, src])
        ).to_numpy(float)
        lo, hi = boot_ci(d)
        contrast_rows.append(dict(
            measure=meas,
            contrast="interaction ((g1k1 - g1k0) - (g0k1 - g0k0), quartets matched)",
            n_pairs=len(d), mean_delta=float(d.mean()), ci_lo=lo, ci_hi=hi,
            frac_positive=float((d > 0).mean())))
    contrast_df = pd.DataFrame(contrast_rows)
    contrast_df.to_csv(OUT_DIR / "contrast_table.csv", index=False)

    checks = {
        "gross_cells": gross_cells,
        "gross_runs": gross_runs,
        "net_cell_files": len(cell_files),
        "net_rows": rows_net,
        "net_agent_rows": agent_rows_net,
        "panel_shape_guard": "24 cells x 20 seeds asserted from disk",
        "kappa_control": control,
        "gamma_binding_descriptive_g0": gamma_binds,
        "kappa_binding_descriptive_g01": kappa_binds,
    }
    with open(OUT_DIR / "control_checks.json", "w") as f:
        json.dump(checks, f, indent=1)

    # ---- console summary ---------------------------------------------------
    print("GROSS cells=%d runs=%d | NET cell_files=%d rows=%d agent_rows=%d"
          % (gross_cells, gross_runs, len(cell_files), rows_net, agent_rows_net))
    print(quad.to_string(index=False))
    print("kappa control at g=0: %d pairs, max_abs_diff=%r, PASS=%s"
          % (pairs_checked, worst_diff, control["kappa_inert_at_g0_pass"]))
    print("gamma binding at g=0 (wedge differs frac):", gamma_binds)
    print("kappa binding at g=0.1:", kappa_binds)
    print(contrast_df.to_string(index=False))
    print("ANALYSIS OK -> typology_table.csv, typology_by_curiosity.csv, typology_by_entry.csv, cell_summary.csv, contrast_table.csv, control_checks.json")


if __name__ == "__main__":
    main()
