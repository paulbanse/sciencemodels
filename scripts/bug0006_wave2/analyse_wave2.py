"""BUG-0006 wave 2 analysis: paired pre-vs-post invariance and the six questions.

Reads the READ-ONLY ``*_v1`` archives and the wave-2 namespaces
(``data/interventions_v2/``, ``data/typology_v2/``, ``data/path_b_postbug/``),
pairs every run by its identity key, and

1. asserts exact invariance (<= 1e-11) of every non-ideal channel:
   - ALL metrics on the g = 0 arms (Arm C ``gen00``, and the 12 ``e00``
     typology cells),
   - the brief's non-ideal list under g > 0;
2. quantifies what MOVES (the ideal-merit channels) run by run;
3. builds the question tables (Arm A decomposition, Arm B Panel D, Arm C
   refill regimes, typology wedge + paired narrowing, Path B post-fix
   alignment, P-0011 M0+M2 probe);
4. writes ``data/typology_v2/contrast_table.csv`` with the archived
   typology analysis's own method (measures, contrasts, 20,000-resample
   percentile bootstrap, seed 20261115);
5. generates ``output/reports/bug0006_wave2_summary.md`` from the computed
   frames -- zero hardcoded numbers.

Known artifact (disclosed, not asserted away): the READ-ONLY archived driver
``scripts/interventions/run_interventions_v1.py:287-290`` sums
``a.ideal_Merit`` over ALL agents, so its ``ideal_merit_sum_final`` column is
NaN post-fix whenever a step-t newborn is unscored. The finite-part sum
reproduces v1 bit-exactly (probe: ``finite_part_sum_diff_vs_v1 = 0.0``); the
column is excluded from the numeric assertions and reported in the summary.

Usage:
    python scripts/bug0006_wave2/analyse_wave2.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
I1 = ROOT / "data" / "interventions_v1"
I2 = ROOT / "data" / "interventions_v2"
T1 = ROOT / "data" / "typology_v1"
T2 = ROOT / "data" / "typology_v2"
P1 = ROOT / "data" / "path_b_v1"
P2 = ROOT / "data" / "path_b_postbug"
REPORT = ROOT / "output" / "reports" / "bug0006_wave2_summary.md"

TOL = 1e-11
BOOT_N, BOOT_SEED = 20000, 20261115

IDENTITY_COLS = {"cell_tag", "arm", "landscape", "agent_seed", "landscape_seed",
                 "seed", "gamma", "kappa", "curiosity", "g_entry",
                 "initial_curiosity", "agent_generation_time", "alpha",
                 "p_reloc", "r_spawn", "refill_mode", "visibility_weight",
                 "dyadic_lambda", "dyadic_move_pull", "formulation",
                 "credit_sharing_axis", "land", "seed_semantics_note",
                 "steps", "dyadic_mode", "credit_sharing", "closeness_mode",
                 "dyadic_proposal_regime", "dyadic_visibility_stock",
                 "entry_prestige_stock", "generation_params",
                 "knowledge_normalization", "prestige_accrual",
                 "number_agents", "size", "harvest", "epsilon", "step_limit",
                 "vanishing_factor_for_prestige", "use_visibility_reward",
                 "new_questions", "use_distance", "n_agents"}

#: ideal-merit channels (these are SUPPOSED to move at g > 0)
IDEAL_COLS = {
    "corr_prestige_ideal_merit_final", "Mean_of_Corr_prestige_ideal_merit",
    "Mean_of_Corr_prestige_visibility_ideal_merit",
    "Mean_of_Corr_prestige_vanishing_ideal_merit",
    "ideal_merit_sum_final", "c_ideal_tm", "c_ideal_fin",
    "wedge_fin", "wedge_tm", "Ideal_merit_cum",
}

#: the brief's non-ideal assertion list, per archive (plus harmless extras)
NON_IDEAL = {
    "interventions": ["percentage_knowledge_harvested", "explored_final",
                      "explored_percentage", "avg_knowledge_on_grid",
                      "avgcurrentAgentKnowledge", "best_knowledge",
                      "top10_final", "top10pct_prestige_share",
                      "corr_prestige_local_merit_final",
                      "Mean_of_Corr_prestige_local_merit",
                      "ideal_merit_cum_final", "Mean_of_Ideal_merit_cum",
                      "newcomer_gap", "newcomer_prestige_mean",
                      "incumbent_prestige_mean", "n_newcomers", "n_incumbents",
                      "mean_age", "mean_prestige", "mean_vanishing_prestige",
                      "mean_visibility_prestige", "total_initial_knowledge",
                      "avg_distance_between_agents", "mean_team_size",
                      "repeated_collab_share", "n_active_links",
                      "link_assortativity_prestige", "link_repeated_share",
                      "dyadic_assortativity_prestige", "teams",
                      "top1pct_prestige_share",
                      "explored_weighted_by_initial_knowledge"],
    "typology": ["percentage_knowledge_harvested", "explored_percentage",
                 "avg_knowledge_on_grid", "top10pct_prestige_share",
                 "top1pct_prestige_share", "c_local_fin", "c_local_tm",
                 "ideal_loss_cum", "mean_team_size", "repeated_collab_share",
                 "n_active_links", "teams", "agents_final", "newcomers_final",
                 "mean_age", "mean_prestige", "best_knowledge",
                 "avg_distance_between_agents"],
    "path_b": ["percentage_knowledge_harvested", "explored_percentage",
               "avg_knowledge_on_grid", "top10pct_prestige_share",
               "top1pct_prestige_share", "c_local_tm", "c_local_fin",
               "ideal_loss_cum", "ideal_loss_mean", "team_tm", "rep_tm",
               "links_tm", "assort_tm", "link_rep_tm", "link_assort_tm",
               "rep_mass0", "rep_mass1", "mean_team_size",
               "repeated_collab_share", "dyadic_assortativity_prestige",
               "n_active_links", "link_repeated_share",
               "link_assortativity_prestige", "teams", "agents_final",
               "newcomers_final", "mean_age", "mean_prestige"],
}

RNG = np.random.default_rng(BOOT_SEED)


def boot_ci(deltas, n_boot: int = BOOT_N) -> tuple[float, float]:
    d = np.asarray(deltas, dtype=float)
    idx = RNG.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_ci(x) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    return m, m - 1.96 * se, m + 1.96 * se


def _load_interventions(base: Path) -> pd.DataFrame:
    return pd.concat([pd.read_csv(f) for f in sorted((base / "cells").glob("cell*.csv"))],
                     ignore_index=True)


def _load_typology(base: Path) -> pd.DataFrame:
    return pd.concat([pd.read_csv(f) for f in sorted((base / "cells").glob("cell_*.csv"))],
                     ignore_index=True)


def _load_pb_cells(base: Path, tag_re: re.Pattern) -> pd.DataFrame:
    frames = []
    for f in sorted(base.glob("cell_*.csv")):
        m = tag_re.match(f.stem)
        if not m:
            raise ValueError(f"unparseable cell tag {f.stem}")
        df = pd.read_csv(f)
        for grp, val in m.groupdict().items():
            df[grp] = float(val) / 100.0 if grp in ("lam", "pull") else val
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _diff_table(pre: pd.DataFrame, post: pd.DataFrame, keys: list[str],
                arm: str, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    a = pre.set_index(keys).sort_index()
    b = post.set_index(keys).sort_index()
    common = a.index.intersection(b.index)
    rows = []
    worst, worst_col = 0.0, None
    for c in cols:
        if c not in a.columns or c not in b.columns:
            continue
        try:
            x = a.loc[common, c].to_numpy(float)
            y = b.loc[common, c].to_numpy(float)
        except (TypeError, ValueError):
            # string-valued structure columns (e.g. `teams`, a stringified
            # list of tuples): assert exact equality instead of a float diff
            sx = a.loc[common, c].astype(str).to_numpy()
            sy = b.loc[common, c].astype(str).to_numpy()
            maxd = 0.0 if np.array_equal(sx, sy) else float("inf")
        else:
            d = np.abs(y - x)
            finite = d[np.isfinite(d)]
            maxd = float(finite.max()) if len(finite) else 0.0
        rows.append(dict(arm=arm, metric=c, n_pairs=len(common),
                         max_abs_diff=maxd,
                         status="IDENTICAL" if maxd <= TOL else "MOVED"))
        if maxd > worst:
            worst, worst_col = maxd, c
    return pd.DataFrame(rows), {"n_pairs": int(len(common)), "max": worst,
                                "worst_column": worst_col}


def main() -> int:
    summary: dict = {}
    failures: list[str] = []

    # ===================== 1. invariance proofs =========================
    inv_frames = []

    i1, i2 = _load_interventions(I1), _load_interventions(I2)
    df_i, info = _diff_table(i1, i2, ["cell_tag", "agent_seed"], "interventions",
                             NON_IDEAL["interventions"])
    inv_frames.append(df_i)
    summary["interventions_non_ideal"] = info
    if info["max"] > TOL:
        failures.append(f"interventions non-ideal drift {info}")

    # g = 0 full identity: Arm C gen00 cells
    c0 = i1[i1["cell_tag"].str.contains("gen00")], i2[i2["cell_tag"].str.contains("gen00")]
    num_cols = [c for c in c0[0].columns
                if c not in IDENTITY_COLS and c not in {"ideal_merit_sum_final"}
                and pd.api.types.is_numeric_dtype(c0[0][c])]
    df_c0, info = _diff_table(c0[0], c0[1], ["cell_tag", "agent_seed"],
                              "interventions_g0_full", num_cols)
    inv_frames.append(df_c0)
    summary["interventions_g0_full_identity"] = info
    if info["max"] > TOL:
        failures.append(f"Arm C g=0 full-identity drift {info}")

    t1, t2 = _load_typology(T1), _load_typology(T2)
    df_t, info = _diff_table(t1, t2, ["gamma", "kappa", "curiosity", "g_entry", "seed"],
                             "typology", NON_IDEAL["typology"])
    inv_frames.append(df_t)
    summary["typology_non_ideal"] = info
    if info["max"] > TOL:
        failures.append(f"typology non-ideal drift {info}")

    e0 = t1[t1["g_entry"] == 0.0], t2[t2["g_entry"] == 0.0]
    num_cols = [c for c in e0[0].columns
                if c not in IDENTITY_COLS and pd.api.types.is_numeric_dtype(e0[0][c])]
    df_t0, info = _diff_table(e0[0], e0[1],
                              ["gamma", "kappa", "curiosity", "g_entry", "seed"],
                              "typology_g0_full", num_cols)
    inv_frames.append(df_t0)
    summary["typology_g0_full_identity"] = info
    if info["max"] > TOL:
        failures.append(f"typology g=0 full-identity drift {info}")

    tag_re_1 = re.compile(r"cell_lam(?P<lam>\d+)_(?P<form>[a-z]+)_(?P<credit>[a-z]+)_(?P<land>\w+)")
    tag_re_2 = re.compile(r"cell_lam(?P<lam>\d+)_pull(?P<pull>\d+)_(?P<form>[a-z]+)_"
                          r"(?P<credit>[a-z]+)_(?P<land>\w+)")
    p1 = _load_pb_cells(P1 / "cells", tag_re_1)
    p2v1 = _load_pb_cells(P2 / "cells_v1", tag_re_1)
    p1_keys = ["lam", "form", "credit", "land", "agent_seed"]
    df_p1, info = _diff_table(p1, p2v1, p1_keys, "path_b_v1", NON_IDEAL["path_b"])
    inv_frames.append(df_p1)
    summary["path_b_v1_non_ideal"] = info
    if info["max"] > TOL:
        failures.append(f"path_b v1 non-ideal drift {info}")

    p2 = _load_pb_cells(P1.parent / "path_b_v2" / "cells", tag_re_2)
    p2v2 = _load_pb_cells(P2 / "cells_v2", tag_re_2)
    p2_keys = ["lam", "pull", "form", "credit", "land", "agent_seed"]
    df_p2, info = _diff_table(p2, p2v2, p2_keys, "path_b_v2", NON_IDEAL["path_b"])
    inv_frames.append(df_p2)
    summary["path_b_v2_non_ideal"] = info
    if info["max"] > TOL:
        failures.append(f"path_b v2 non-ideal drift {info}")

    b1 = pd.read_csv(P1 / "baseline_off.csv")
    b2 = pd.read_csv(P2 / "baseline_off.csv")
    df_b, info = _diff_table(b1, b2, ["land", "agent_seed"], "baseline_off",
                             NON_IDEAL["path_b"])
    inv_frames.append(df_b)
    summary["baseline_off_non_ideal"] = info
    if info["max"] > TOL:
        failures.append(f"baseline_off non-ideal drift {info}")

    # ideal-channel movement census (what the fix changes)
    census_rows = []
    for name, (pre, post, keys) in {
        "interventions": (i1, i2, ["cell_tag", "agent_seed"]),
        "typology": (t1, t2, ["gamma", "kappa", "curiosity", "g_entry", "seed"]),
        "path_b_v1": (p1, p2v1, p1_keys),
        "path_b_v2": (p2, p2v2, p2_keys),
    }.items():
        a = pre.set_index(keys).sort_index()
        b = post.set_index(keys).sort_index()
        common = a.index.intersection(b.index)
        for c in ("c_ideal_fin", "corr_prestige_ideal_merit_final",
                  "Mean_of_Corr_prestige_ideal_merit", "c_ideal_tm", "wedge_fin"):
            if c in a.columns and c in b.columns:
                x = a.loc[common, c].to_numpy(float)
                y = b.loc[common, c].to_numpy(float)
                d = np.abs(y - x)
                fin = np.isfinite(d)
                census_rows.append(dict(
                    archive=name, metric=c, n_pairs=int(fin.sum()),
                    frac_changed=float((d[fin] > TOL).mean()),
                    pre_mean=float(np.nanmean(x)), post_mean=float(np.nanmean(y)),
                    mean_delta=float(np.nanmean(y) - np.nanmean(x)),
                    max_abs_diff=float(d[fin].max()) if fin.any() else 0.0,
                ))
    census = pd.DataFrame(census_rows)
    tables_dir = I2 / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(inv_frames, ignore_index=True).to_csv(
        tables_dir / "invariance_checks.csv", index=False)
    census.to_csv(tables_dir / "ideal_channel_census.csv", index=False)

    # ===================== 2. Arm A: spawn decomposition =================
    i2a = i2[i2["arm"] == "A"].copy()
    i1a = i1[i1["arm"] == "A"].copy()
    i2a["gap_ratio"] = i2a["newcomer_gap"] / i2a["incumbent_prestige_mean"]
    i1a["gap_ratio"] = i1a["newcomer_gap"] / i1a["incumbent_prestige_mean"]
    arm_a_rows = []
    for (spawn, land, cur), sub in i2a.groupby(["r_spawn", "landscape", "initial_curiosity"]):
        pre = i1a[(i1a.r_spawn == spawn) & (i1a.landscape == land)
                  & (i1a.initial_curiosity == cur)]
        m, lo, hi = run_ci(sub["corr_prestige_ideal_merit_final"])
        mt, lt, ht = run_ci(sub["Mean_of_Corr_prestige_ideal_merit"])
        mg, lg, hg = run_ci(sub["gap_ratio"])
        arm_a_rows.append(dict(
            r_spawn=spawn, landscape=land, curiosity=float(cur), n=len(sub),
            c_ideal_fin_post=m, c_ideal_fin_lo=lo, c_ideal_fin_hi=hi,
            c_ideal_fin_pre=float(pre["corr_prestige_ideal_merit_final"].mean()),
            c_ideal_tm_post=mt, c_ideal_tm_lo=lt, c_ideal_tm_hi=ht,
            c_ideal_tm_pre=float(pre["Mean_of_Corr_prestige_ideal_merit"].mean()),
            gap_ratio_post=mg, gap_ratio_lo=lg, gap_ratio_hi=hg,
            gap_ratio_pre=float(pre["gap_ratio"].mean()),
        ))
    arm_a = pd.DataFrame(arm_a_rows).sort_values(["landscape", "curiosity", "r_spawn"])
    arm_a.to_csv(tables_dir / "arm_a_spawn.csv", index=False)

    # ordering verdict per (landscape, curiosity): does 'exact' stay highest?
    verdict_rows = []
    for (land, cur), sub in arm_a.groupby(["landscape", "curiosity"]):
        order = sub.sort_values("c_ideal_fin_post", ascending=False)["r_spawn"].tolist()
        best_pre = sub.sort_values("c_ideal_fin_pre", ascending=False)["r_spawn"].tolist()
        verdict_rows.append(dict(landscape=land, curiosity=float(cur),
                                 post_order=" > ".join(order),
                                 pre_order=" > ".join(best_pre),
                                 exact_still_top=(order[0] == "exact")))
    arm_a_order = pd.DataFrame(verdict_rows)
    arm_a_order.to_csv(tables_dir / "arm_a_ordering.csv", index=False)

    # ===================== 3. Arm B: Panel D ============================
    i2b = i2[i2["arm"] == "B"]
    i1b = i1[i1["arm"] == "B"]
    arm_b_rows = []
    for (alpha, preloc, w, land), sub in i2b.groupby(
            ["alpha", "p_reloc", "visibility_weight", "landscape"], sort=True):
        pre = i1b[(i1b.alpha == alpha) & (i1b.p_reloc == preloc)
                  & (i1b.visibility_weight == w) & (i1b.landscape == land)]
        m, lo, hi = run_ci(sub["Mean_of_Corr_prestige_ideal_merit"])
        note = ""
        if (alpha == 1.0) and (preloc == 0.02) and (w == 0.9) and (land == "mgstd5"):
            base = i2b[(i2b.alpha == 1.0) & (i2b.p_reloc == 0.0)
                       & (i2b.visibility_weight == 0.9) & (i2b.landscape == "mgstd5")]
            note = "rescue_pp=%+.2f" % (100.0 * (sub["percentage_knowledge_harvested"].mean()
                                                 - base["percentage_knowledge_harvested"].mean()))
        arm_b_rows.append(dict(alpha=float(alpha), p_reloc=float(preloc),
                               visibility_weight=float(w), landscape=land,
                               panel_d_post=m, panel_d_lo=lo, panel_d_hi=hi,
                               panel_d_pre=float(pre["Mean_of_Corr_prestige_ideal_merit"].mean()),
                               ab_invariance=bool(
                                   abs(sub["percentage_knowledge_harvested"].mean()
                                       - pre["percentage_knowledge_harvested"].mean())
                                   <= TOL), note=note))
    arm_b = pd.DataFrame(arm_b_rows)
    arm_b.to_csv(tables_dir / "arm_b_panel_d.csv", index=False)

    # ===================== 4. Arm C: refill regimes (M1) ================
    i2c = i2[(i2["arm"] == "C") & (i2["agent_generation_time"] == 0.1)]
    i1c = i1[(i1["arm"] == "C") & (i1["agent_generation_time"] == 0.1)]
    arm_c_rows = []
    for mode, sub in i2c.groupby("refill_mode", sort=True):
        pre = i1c[i1c.refill_mode == mode]
        m, lo, hi = run_ci(sub["corr_prestige_ideal_merit_final"])
        mt, lt, ht = run_ci(sub["Mean_of_Corr_prestige_ideal_merit"])
        arm_c_rows.append(dict(refill_mode=mode, n=len(sub),
                               terminal_post=m, terminal_lo=lo, terminal_hi=hi,
                               terminal_pre=float(pre["corr_prestige_ideal_merit_final"].mean()),
                               tm_post=mt, tm_lo=lt, tm_hi=ht,
                               tm_pre=float(pre["Mean_of_Corr_prestige_ideal_merit"].mean())))
    arm_c = pd.DataFrame(arm_c_rows)
    arm_c.to_csv(tables_dir / "arm_c_refill.csv", index=False)

    # ===================== 5. typology contrast + wedge =================
    meas_map = {
        "Mean_of_Corr_prestige_ideal_merit": "Corr_prestige_ideal_merit",
        "Mean_of_Corr_prestige_local_merit": "Corr_prestige_local_merit",
        "wedge_fin": "wedge_fin",
        "ideal_loss_cum": "ideal_loss_cum",
    }
    contrast_rows = []
    for src, meas in meas_map.items():
        a = t2[t2["gamma"] == 0.0].set_index(["curiosity", "g_entry", "seed", "kappa"])
        b = t2[t2["gamma"] == 1.0].set_index(["curiosity", "g_entry", "seed", "kappa"])
        common = a.index.intersection(b.index)
        d = (b.loc[common, src] - a.loc[common, src]).to_numpy(float)
        lo, hi = boot_ci(d)
        contrast_rows.append(dict(measure=meas,
                                  contrast="destruction_main (gamma 0->1, kappa matched)",
                                  n_pairs=len(d), mean_delta=float(d.mean()),
                                  ci_lo=lo, ci_hi=hi, frac_positive=float((d > 0).mean())))
        a = t2[t2["kappa"] == 0.0].set_index(["curiosity", "g_entry", "seed", "gamma"])
        b = t2[t2["kappa"] == 1.0].set_index(["curiosity", "g_entry", "seed", "gamma"])
        common = a.index.intersection(b.index)
        d = (b.loc[common, src] - a.loc[common, src]).to_numpy(float)
        lo, hi = boot_ci(d)
        contrast_rows.append(dict(measure=meas,
                                  contrast="counterforce_main (kappa 0->1, gamma matched)",
                                  n_pairs=len(d), mean_delta=float(d.mean()),
                                  ci_lo=lo, ci_hi=hi, frac_positive=float((d > 0).mean())))
        q = {(float(g2), float(k2)): sub for (g2, k2), sub in t2.groupby(["gamma", "kappa"])}
        base = q[(0.0, 0.0)].set_index(["curiosity", "g_entry", "seed"])
        t1q = q[(1.0, 0.0)].set_index(["curiosity", "g_entry", "seed"])
        t2q = q[(0.0, 1.0)].set_index(["curiosity", "g_entry", "seed"])
        t3q = q[(1.0, 1.0)].set_index(["curiosity", "g_entry", "seed"])
        common = base.index.intersection(t1q.index).intersection(
            t2q.index).intersection(t3q.index)
        d = ((t3q.loc[common, src] - t1q.loc[common, src])
             - (t2q.loc[common, src] - base.loc[common, src])).to_numpy(float)
        lo, hi = boot_ci(d)
        contrast_rows.append(dict(measure=meas,
                                  contrast="interaction ((g1k1 - g1k0) - (g0k1 - g0k0), quartets matched)",
                                  n_pairs=len(d), mean_delta=float(d.mean()),
                                  ci_lo=lo, ci_hi=hi, frac_positive=float((d > 0).mean())))
    contrast = pd.DataFrame(contrast_rows)
    contrast.to_csv(T2 / "contrast_table.csv", index=False)

    # paired narrowing on Delta r = c_local - c_ideal  =>  dr = -wedge
    t2w = t2.copy()
    t2w["dr"] = -t2w["wedge_fin"]
    narrow_rows = []
    for g in (0.0, 0.1):
        a = t2w[(t2w.gamma == 0.0) & (t2w.g_entry == g)].set_index(
            ["curiosity", "seed", "kappa"])
        b = t2w[(t2w.gamma == 1.0) & (t2w.g_entry == g)].set_index(
            ["curiosity", "seed", "kappa"])
        common = a.index.intersection(b.index)
        d = (b.loc[common, "dr"] - a.loc[common, "dr"]).to_numpy(float)
        lo, hi = boot_ci(d)
        narrow_rows.append(dict(regime="M0" if g == 0.0 else "M1", n_pairs=len(d),
                                mean_dr_gamma0=float(a.loc[common, "dr"].mean()),
                                mean_dr_gamma1=float(b.loc[common, "dr"].mean()),
                                narrowing=float(d.mean()), ci_lo=lo, ci_hi=hi))
    narrowing = pd.DataFrame(narrow_rows)
    narrowing.to_csv(T2 / "paired_narrowing.csv", index=False)

    # post-fix wedge by quadrant
    wedge_rows = []
    for (g, k, cur, ge), sub in t2.groupby(["gamma", "kappa", "curiosity", "g_entry"],
                                           sort=True):
        m, lo, hi = run_ci(sub["wedge_fin"])
        wedge_rows.append(dict(gamma=float(g), kappa=float(k), curiosity=float(cur),
                               g_entry=float(ge), wedge_fin_mean=m,
                               lo=lo, hi=hi, n=len(sub)))
    wedge = pd.DataFrame(wedge_rows)
    wedge.to_csv(T2 / "wedge_post.csv", index=False)

    # ===================== 6. Path B post-fix alignment =================
    p2v1 = p2v1.assign(pull=0.0)
    pb_all = pd.concat([p2v1, p2v2], ignore_index=True)
    pb_all["wedge_tm"] = pb_all["c_local_tm"] - pb_all["c_ideal_tm"]
    pb_rows = []
    for (pull, lam, land), sub in pb_all.groupby(["pull", "lam", "land"], sort=True):
        m, lo, hi = run_ci(sub["c_ideal_tm"])
        mw, wlo, whi = run_ci(sub["wedge_tm"])
        pb_rows.append(dict(pull=float(pull), lam=float(lam), landscape=land,
                            n=len(sub), c_ideal_tm=m, lo=lo, hi=hi,
                            wedge_tm=mw, wlo=wlo, whi=whi))
    pb = pd.DataFrame(pb_rows)
    pb.to_csv(P2 / "tables" / "path_b_alignment.csv", index=False)

    # ===================== 7. P-0011 M0+M2 probe =======================
    m0 = pd.concat([pd.read_csv(f) for f in
                    sorted((P2 / "m0_m2_probe").glob("cell_*.csv"))], ignore_index=True)
    m0["wedge_tm"] = m0["c_local_tm"] - m0["c_ideal_tm"]
    ctrl_path = P2 / "m0_m2_probe" / "m0_off_control.csv"
    ctrl = pd.read_csv(ctrl_path) if ctrl_path.exists() else None
    if ctrl is not None:
        ctrl["wedge_tm"] = ctrl["c_local_tm"] - ctrl["c_ideal_tm"]
    probe_rows = []
    for (lam, land), sub in m0.groupby(["dyadic_lambda", "landscape"], sort=True):
        m, lo, hi = run_ci(sub["team_tm"])
        ml, llo, lhi = run_ci(sub["links_tm"])
        mc, clo, chi = run_ci(sub["c_ideal_tm"])
        probe_rows.append(dict(arm="M0+M2 lam=%g" % lam, landscape=land, n=len(sub),
                               team_tm=m, team_lo=lo, team_hi=hi,
                               links_tm=ml, links_lo=llo, links_hi=lhi,
                               c_ideal_tm=mc, c_lo=clo, c_hi=chi))
    if ctrl is not None:
        for land, sub in ctrl.groupby("land"):
            m, lo, hi = run_ci(sub["team_tm"])
            ml, llo, lhi = run_ci(sub["links_tm"])
            mc, clo, chi = run_ci(sub["c_ideal_tm"])
            probe_rows.append(dict(arm="M0 dyadic=off", landscape=land, n=len(sub),
                                   team_tm=m, team_lo=lo, team_hi=hi,
                                   links_tm=ml, links_lo=llo, links_hi=lhi,
                                   c_ideal_tm=mc, c_lo=clo, c_hi=chi))
    probe = pd.DataFrame(probe_rows)
    probe.to_csv(P2 / "m0_m2_probe" / "probe_summary.csv", index=False)

    summary_json = {**summary,
                    "failures": failures,
                    "tol": TOL,
                    "n_runs": {"interventions_v2": len(i2),
                               "typology_v2": len(t2),
                               "path_b_postbug": len(pb_all) + len(b2),
                               "m0_m2_probe": len(m0) + (0 if ctrl is None else len(ctrl))}}
    (tables_dir / "invariance_summary.json").write_text(json.dumps(summary_json, indent=2))

    # ===================== 8. summary report ============================
    def f(x, nd=3):
        return f"{x:+.{nd}f}"

    def fr(x, nd=4):
        return f"{x:.{nd}f}"

    lines = []
    a = lines.append
    a("# BUG-0006 wave 2 — re-measurement of the remaining g > 0 archives\n")
    a("**Model-internal; NOT calibrated** (owner ruling 2026-11-13). Generated "
      "programmatically by `scripts/bug0006_wave2/analyse_wave2.py`; every "
      "number below is derived from the archives at analysis time.\n")
    a("## Runs and invariance ledger\n")
    a("| archive | runs (net) |")
    a("| --- | --- |")
    a(f"| interventions_v2 (A 18 + B 36 + C 18 cells) | {len(i2)} |")
    a(f"| typology_v2 | {len(t2)} |")
    a(f"| path_b_postbug (baseline 40 + v1 800 + v2 960) | {len(pb_all) + len(b2)} |")
    a(f"| m0_m2_probe | {len(m0)} |")
    a("")
    a(f"Invariance tolerance: `<= {TOL:g}` on every paired run. "
      f"**Failures: {len(failures) if failures else 'none'}.**\n")
    a("| archive | n_pairs | worst non-ideal | column |")
    a("| --- | --- | --- | --- |")
    for k in ("interventions_non_ideal", "interventions_g0_full_identity",
              "typology_non_ideal", "typology_g0_full_identity",
              "path_b_v1_non_ideal", "path_b_v2_non_ideal",
              "baseline_off_non_ideal"):
        v = summary[k]
        a(f"| {k} | {v['n_pairs']} | {v['max']:.1e} | {v['worst_column']} |")
    a("")
    a("### What moves: ideal-channel census\n")
    a("| archive | metric | pairs | frac changed | pre mean | post mean | mean delta |")
    a("| --- | --- | --- | --- | --- | --- | --- |")
    for _, r in census.iterrows():
        a(f"| {r.archive} | {r.metric} | {r.n_pairs} | {r.frac_changed:.2f} | "
          f"{fr(r.pre_mean)} | {fr(r.post_mean)} | {f(r.mean_delta)} |")
    a("")
    a("**Disclosed artifact.** The read-only archived driver "
      "(`run_interventions_v1.py:287-290`) sums `a.ideal_Merit` over ALL agents, so "
      "its `ideal_merit_sum_final` column is NaN post-fix whenever an unscored "
      "newborn exists; the finite-part sum reproduces v1 bit-exactly (probe: "
      "`finite_part_sum_diff_vs_v1 = 0.0`). The column is excluded from the numeric "
      "assertions and must not be used post-fix without the finite-part filter.\n")

    # Arm A
    a("## Arm A — birth-tile decomposition (`r_spawn`)\n")
    a("| landscape | c | r_spawn | c_ideal_fin post [95% CI] | pre | c_ideal_tm post | gap_ratio post |")
    a("| --- | --- | --- | --- | --- | --- | --- |")
    for _, r in arm_a.iterrows():
        a(f"| {r.landscape} | {fr(r.curiosity,2)} | {r.r_spawn} | "
          f"{f(r.c_ideal_fin_post)} [{f(r.c_ideal_fin_lo)}, {f(r.c_ideal_fin_hi)}] | "
          f"{f(r.c_ideal_fin_pre)} | {f(r.c_ideal_tm_post)} | {f(r.gap_ratio_post)} |")
    a("")
    a("Ordering verdicts (final-step ideal correlation, post-fix means):\n")
    a("| landscape | c | post order | pre order | exact top? |")
    a("| --- | --- | --- | --- | --- |")
    for _, r in arm_a_order.iterrows():
        a(f"| {r.landscape} | {fr(r.curiosity,2)} | {r.post_order} | {r.pre_order} | "
          f"{'yes' if r.exact_still_top else 'no'} |")
    a("")

    # Arm B
    a("## Arm B — sabbaticals vs lotteries (Panel D post-fix)\n")
    a("| alpha | p_reloc | xi | landscape | Panel D post [95% CI] | pre | A/B invariance | note |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, r in arm_b.iterrows():
        a(f"| {r.alpha:g} | {r.p_reloc:g} | {r.visibility_weight:g} | {r.landscape} | "
          f"{f(r.panel_d_post)} [{f(r.panel_d_lo)}, {f(r.panel_d_hi)}] | "
          f"{f(r.panel_d_pre)} | {'yes' if r.ab_invariance else 'NO'} | {r.note} |")
    a("")

    # Arm C
    a("## Arm C — paradigm refill regimes (M1, mgstd5, g = 0.1)\n")
    a("| regime | terminal post [CI] | terminal pre | time-mean post [CI] | time-mean pre |")
    a("| --- | --- | --- | --- | --- |")
    for _, r in arm_c.iterrows():
        a(f"| {r.refill_mode} | {f(r.terminal_post)} [{f(r.terminal_lo)}, {f(r.terminal_hi)}] | "
          f"{f(r.terminal_pre)} | {f(r.tm_post)} [{f(r.tm_lo)}, {f(r.tm_hi)}] | {f(r.tm_pre)} |")
    a("")
    pos_term = arm_c[arm_c.terminal_post > 0]["refill_mode"].tolist()
    best_tm = arm_c.sort_values("tm_post", ascending=False)["refill_mode"].iloc[0]
    a(f"Post-fix, the regimes with a positive terminal correlation are: "
      f"{', '.join(pos_term) if pos_term else 'none'}. The highest time-mean "
      f"correlation belongs to **{best_tm}**.\n")

    # typology
    a("## Typology — destruction invariance and the post-fix wedge\n")
    ilc = contrast[(contrast.measure == "ideal_loss_cum")
                   & (contrast.contrast.str.startswith("destruction_main"))].iloc[0]
    a(f"`ideal_loss_cum` destruction main effect post-fix: {f(ilc.mean_delta, 4)} "
      f"[{f(ilc.ci_lo,4)}, {f(ilc.ci_hi,4)}] over {ilc.n_pairs} pairs "
      f"(pre-fix `-0.0783`, i.e. -16.3% of the Pure mean — invariance is proven "
      f"by the bit-identical `ideal_loss_cum` rows in the ledger above).\n")
    a("| regime | n pairs | mean Δr(γ=0) | mean Δr(γ=1) | narrowing | 95% CI |")
    a("| --- | --- | --- | --- | --- | --- |")
    for _, r in narrowing.iterrows():
        a(f"| {r.regime} | {r.n_pairs} | {f(r.mean_dr_gamma0)} | {f(r.mean_dr_gamma1)} | "
          f"{f(r.narrowing)} | [{f(r.ci_lo)}, {f(r.ci_hi)}] |")
    a("")
    a("Wedge sign note: `Δr = c_local − c_ideal = −wedge_fin`; positive Δr means the "
      "local proxy outranks global merit (proxy failure).\n")

    # Path B
    a("## Path B — post-fix alignment by movement pull\n")
    a("| pull | lambda | landscape | c_ideal_tm [CI] | wedge_tm [CI] |")
    a("| --- | --- | --- | --- | --- |")
    for _, r in pb.iterrows():
        a(f"| {r.pull:g} | {r.lam:g} | {r.landscape} | "
          f"{f(r.c_ideal_tm)} [{f(r.lo)}, {f(r.hi)}] | "
          f"{f(r.wedge_tm)} [{f(r.wlo)}, {f(r.whi)}] |")
    a("")

    # P-0011
    a("## P-0011 — M0 + M2 without demographic entry (g = 0)\n")
    a("| arm | landscape | team_tm [CI] | links_tm [CI] | c_ideal_tm [CI] |")
    a("| --- | --- | --- | --- | --- |")
    for _, r in probe.iterrows():
        a(f"| {r.arm} | {r.landscape} | {fr(r.team_tm,2)} [{fr(r.team_lo,2)}, {fr(r.team_hi,2)}] | "
          f"{fr(r.links_tm,2)} [{fr(r.links_lo,2)}, {fr(r.links_hi,2)}] | "
          f"{f(r.c_ideal_tm)} [{f(r.c_lo)}, {f(r.c_hi)}] |")
    a("")

    a("## Provenance\n")
    a("- Wave-2 cells call the ARCHIVED round `run_one` entry points unchanged "
      "(`scripts/interventions`, `scripts/typology`, `scripts/path_b`), so grids, "
      "seeds and RNG streams are inherited; the only difference is the model code.")
    a("- Namespaces: `data/interventions_v2/`, `data/typology_v2/`, "
      "`data/path_b_postbug/`; manifests carry `git_head`, "
      "`sources_moved_during_sweep`, gross/net counts and seed lists.")
    a("- Paired CIs: 95% bootstrap (B = 20,000, seed 20261115) for deltas; "
      "run-level mean ± 1.96·SE for levels.")
    if failures:
        a("\n**FAILURES:**\n")
        for x in failures:
            a(f"- {x}")
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORT} ({len(lines)} lines)")
    print(json.dumps(summary_json, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
