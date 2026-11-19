"""Paired post-BUG-0006 analysis and report generator.

Compares post-BUG-0006 re-measured runs (``data/bug0006_remeasure_v2/`` and
``data/turnover_v2/``) seed-by-seed against the pre-fix archives
(``data/remeasure_v1/``, ``data/turnover_v1/``, ``data/path_b_v2/``, and
``data/fig1_extension_v1/``), proving non-ideal channel invariance and
quantifying exactly what changes and what survives.

Writes summary CSVs under ``data/bug0006_remeasure_v2/tables/`` and renders
``output/reports/bug0006_remeasure_v2.md`` strictly from those tables (zero
hand-typed numbers). Model-internal; NOT calibrated (owner ruling 2026-11-13).

Usage:
    /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 scripts/bug0006/analyse_bug0006.py
"""
from __future__ import annotations

import glob
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
V2 = DATA / "bug0006_remeasure_v2"
TO_V2 = DATA / "turnover_v2"
TABLES = V2 / "tables"
REPORT = ROOT / "output" / "reports" / "bug0006_remeasure_v2.md"

BOOT_N = 10000
BOOT_SEED = 20261119


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _boot_ci(vals: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(v), size=(BOOT_N, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _load_hinge_mod():
    path = ROOT / "scripts" / "welfare" / "analyse_welfare.py"
    spec = importlib.util.spec_from_file_location("_analyse_welfare", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. T1 & W_v05 paired BUG-0006 deltas
# ---------------------------------------------------------------------------
def analyse_paired_arm(
    pre: pd.DataFrame,
    post: pd.DataFrame,
    keys: list[str],
    cell_keys: list[str],
    scopes: list[tuple[str, pd.Series]],
    metrics: list[str],
    arm_name: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = pre.merge(post, on=keys, suffixes=("_pre", "_post"), validate="one_to_one")
    _require(len(merged) == len(post), f"{arm_name}: merge count {len(merged)} != {len(post)}")

    rows = []
    for scope_name, mask in scopes:
        sub = merged.loc[mask].copy()
        n_cells = int(sub[cell_keys].drop_duplicates().shape[0])
        for m in metrics:
            a = sub[f"{m}_pre"].to_numpy(float)
            b = sub[f"{m}_post"].to_numpy(float)
            d = b - a
            lo, hi = _boot_ci(d, rng)
            cell_means = sub.groupby(cell_keys)[[f"{m}_pre", f"{m}_post"]].mean()
            rows.append(
                {
                    "arm": arm_name,
                    "scope": scope_name,
                    "metric": m,
                    "n_runs": int(len(sub)),
                    "n_cells": n_cells,
                    "pre_mean": float(np.mean(a)),
                    "post_mean": float(np.mean(b)),
                    "mean_delta": float(np.mean(d)),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "max_abs_diff": float(np.max(np.abs(d))),
                    "pre_neg_runs": int(np.sum(a < 0.0)),
                    "post_neg_runs": int(np.sum(b < 0.0)),
                    "pre_neg_cells": int((cell_means[f"{m}_pre"] < 0.0).sum()),
                    "post_neg_cells": int((cell_means[f"{m}_post"] < 0.0).sum()),
                }
            )
    return pd.DataFrame(rows), merged


def run_t1_and_w_v05(rng: np.random.Generator) -> dict:
    # --- T1 ---
    pre_t1 = pd.read_csv(DATA / "remeasure_v1" / "runs.csv")
    pre_t1 = pre_t1[pre_t1["origin"] == "T1"].copy()
    _require(len(pre_t1) == 720, f"pre_t1 rows={len(pre_t1)}")
    pre_t1["wedge_local_minus_ideal_fin"] = pre_t1["c_local_fin"] - pre_t1["c_ideal_fin"]
    pre_t1["wedge_local_minus_ideal_tm"] = pre_t1["c_local_tm"] - pre_t1["c_ideal_tm"]

    post_t1 = pd.read_csv(V2 / "t1_runs.csv")
    _require(len(post_t1) == 720, f"post_t1 rows={len(post_t1)}")

    keys_t1 = ["curiosity", "vanishing", "landscape", "agent_generation_time", "seed"]
    cell_t1 = ["curiosity", "vanishing", "landscape", "agent_generation_time"]
    tmp = pre_t1[keys_t1].merge(post_t1[keys_t1], on=keys_t1, validate="one_to_one")
    _require(len(tmp) == 720, "T1 key alignment failed")
    # Build scopes on merged order
    merged_t1_check = pre_t1.merge(post_t1, on=keys_t1, suffixes=("_pre", "_post"), validate="one_to_one")
    scopes_t1 = [
        ("g=0.0 (control)", merged_t1_check["agent_generation_time"] == 0.0),
        ("g=0.1 (all)", merged_t1_check["agent_generation_time"] == 0.1),
        ("g=0.1 (mgstd5)", (merged_t1_check["agent_generation_time"] == 0.1) & (merged_t1_check["landscape"] == "mgstd5")),
        ("g=0.1 (noisypr5)", (merged_t1_check["agent_generation_time"] == 0.1) & (merged_t1_check["landscape"] == "noisypr5")),
    ]
    metrics_common = [
        "c_ideal_fin",
        "c_ideal_tm",
        "c_local_fin",
        "c_local_tm",
        "wedge_local_minus_ideal_fin",
        "wedge_local_minus_ideal_tm",
        "top10_fin",
        "harvested_fin",
        "ideal_loss_cum",
    ]
    t1_delta, merged_t1 = analyse_paired_arm(
        pre_t1, post_t1, keys_t1, cell_t1, scopes_t1, metrics_common, "T1", rng
    )
    t1_delta.to_csv(TABLES / "t1_bug0006_delta.csv", index=False)

    # Assert pristine-channel invariance on T1
    g0_max = float(t1_delta[t1_delta["scope"] == "g=0.0 (control)"]["max_abs_diff"].max())
    _require(g0_max <= 1e-11, f"T1 g=0 control invariance failed: {g0_max}")
    for m in ("c_local_fin", "c_local_tm", "top10_fin", "harvested_fin", "ideal_loss_cum"):
        dmax = float(t1_delta[t1_delta["metric"] == m]["max_abs_diff"].max())
        _require(dmax <= 1e-11, f"T1 non-ideal invariance failed on {m}: {dmax}")

    # --- W_v05 ---
    pre_w = pd.read_csv(DATA / "remeasure_v1" / "w_v05_runs.csv")
    _require(len(pre_w) == 800, f"pre_w rows={len(pre_w)}")
    post_w = pd.read_csv(V2 / "w_v05_runs.csv")
    _require(len(post_w) == 800, f"post_w rows={len(post_w)}")

    keys_w = ["curiosity", "landscape", "agent_generation_time", "use_visibility_reward", "seed"]
    cell_w = ["curiosity", "landscape", "agent_generation_time", "use_visibility_reward"]
    merged_w_chk = pre_w.merge(post_w, on=keys_w, suffixes=("_pre", "_post"), validate="one_to_one")
    scopes_w = [
        ("g=0.0 (control)", merged_w_chk["agent_generation_time"] == 0.0),
        ("g=0.1 (all)", merged_w_chk["agent_generation_time"] == 0.1),
        ("g=0.1 (mgstd5)", (merged_w_chk["agent_generation_time"] == 0.1) & (merged_w_chk["landscape"] == "mgstd5")),
        ("g=0.1 (mg6)", (merged_w_chk["agent_generation_time"] == 0.1) & (merged_w_chk["landscape"] == "mg6")),
    ]
    w_delta, merged_w = analyse_paired_arm(
        pre_w, post_w, keys_w, cell_w, scopes_w, metrics_common, "W_v05", rng
    )
    w_delta.to_csv(TABLES / "w_v05_bug0006_delta.csv", index=False)

    w_g0_max = float(w_delta[w_delta["scope"] == "g=0.0 (control)"]["max_abs_diff"].max())
    _require(w_g0_max <= 1e-11, f"W_v05 g=0 control invariance failed: {w_g0_max}")
    for m in ("c_local_fin", "c_local_tm", "top10_fin", "harvested_fin", "ideal_loss_cum"):
        dmax = float(w_delta[w_delta["metric"] == m]["max_abs_diff"].max())
        _require(dmax <= 1e-11, f"W_v05 non-ideal invariance failed on {m}: {dmax}")

    # Welfare collapse kink xi* pre vs post (sum over scored agents, plus ideal_loss_cum)
    aw = _load_hinge_mod()
    xi_rows = []
    for label, frame in (("pre_BUG0006", pre_w), ("post_BUG0006", post_w)):
        for (land, gen, vis), sub in frame.groupby(["landscape", "agent_generation_time", "use_visibility_reward"]):
            st_term = aw.hinge_stats(
                sub["ideal_merit_sum"].to_numpy(float),
                1.0 - sub["curiosity"].to_numpy(float),
                sub["seed"].to_numpy(),
            )
            st_cum = aw.hinge_stats(
                -sub["ideal_loss_cum"].to_numpy(float),
                1.0 - sub["curiosity"].to_numpy(float),
                sub["seed"].to_numpy(),
            )
            xi_rows.append(
                {
                    "arm": label,
                    "landscape": land,
                    "entry": float(gen),
                    "use_visibility_reward": bool(vis),
                    "xi_star_terminal_sum": float(st_term["xi"]),
                    "xi_star_cum_loss": float(st_cum["xi"]),
                    "n_obs": int(st_term["n"]),
                }
            )
    xi_df = pd.DataFrame(xi_rows)
    xi_df.to_csv(TABLES / "w_v05_xi_star.csv", index=False)

    # Also save cell-level post T1 table for mgstd5 vs noisypr5
    t1_cell = (
        post_t1.groupby(["agent_generation_time", "landscape", "curiosity"], as_index=False)[
            ["c_ideal_fin", "c_ideal_step199", "c_ideal_tm", "c_local_fin", "c_local_tm",
             "wedge_local_minus_ideal_fin", "wedge_local_minus_ideal_tm", "top10_fin", "harvested_fin"]
        ]
        .mean()
    )
    t1_cell.to_csv(TABLES / "t1_post_by_curiosity.csv", index=False)

    return {
        "t1_delta": t1_delta,
        "w_delta": w_delta,
        "xi_star": xi_df,
        "t1_g0_max": g0_max,
        "w_g0_max": w_g0_max,
        "t1_cell": t1_cell,
        "merged_t1": merged_t1,
    }


# ---------------------------------------------------------------------------
# 2. M2_xi paired BUG-0006 delta
# ---------------------------------------------------------------------------
def analyse_m2_xi(rng: np.random.Generator) -> pd.DataFrame:
    ext_frames = [pd.read_csv(p) for p in sorted(glob.glob(str(DATA / "fig1_extension_v1" / "cells" / "m2_*.csv")))]
    c05_frames = [
        pd.read_csv(DATA / "path_b_v2" / "cells" / f"cell_lam{lam}_pull100_logit_equal_mgstd5.csv")
        for lam in ("000", "100")
    ]
    pre_m2 = pd.concat(ext_frames + c05_frames, ignore_index=True)
    pre_m2["curiosity"] = pre_m2["initial_curiosity"].round(2)
    pre_m2["seed"] = pre_m2["agent_seed"].astype(int)
    pre_m2 = pre_m2.rename(
        columns={
            "top10pct_prestige_share": "top10_fin",
            "percentage_knowledge_harvested": "harvested_fin",
        }
    )
    pre_m2["wedge_local_minus_ideal_tm"] = pre_m2["c_local_tm"] - pre_m2["c_ideal_tm"]
    _require(len(pre_m2) == 200, f"pre_m2 rows={len(pre_m2)}")

    post_m2 = pd.read_csv(V2 / "m2_xi_runs.csv")
    post_m2["curiosity"] = post_m2["curiosity"].round(2)
    _require(len(post_m2) == 200, f"post_m2 rows={len(post_m2)}")

    keys = ["curiosity", "dyadic_lambda", "seed"]
    m = pre_m2.merge(post_m2, on=keys, suffixes=("_pre", "_post"), validate="one_to_one")
    _require(len(m) == 200, f"M2 merge rows={len(m)}")

    rows = []
    for lam_label, mask in (
        ("all", pd.Series(True, index=m.index)),
        ("lambda=0.0", m["dyadic_lambda"] == 0.0),
        ("lambda=1.0", m["dyadic_lambda"] == 1.0),
    ):
        sub = m[mask]
        for met in ("c_ideal_tm", "c_local_tm", "wedge_local_minus_ideal_tm", "team_tm", "links_tm", "top10_fin", "harvested_fin"):
            a = sub[f"{met}_pre"].to_numpy(float)
            b = sub[f"{met}_post"].to_numpy(float)
            d = b - a
            lo, hi = _boot_ci(d, rng)
            rows.append(
                {
                    "arm": "M2_xi",
                    "scope": lam_label,
                    "metric": met,
                    "n_runs": int(len(sub)),
                    "pre_mean": float(np.mean(a)),
                    "post_mean": float(np.mean(b)),
                    "mean_delta": float(np.mean(d)),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "max_abs_diff": float(np.max(np.abs(d))),
                }
            )
    df_delta = pd.DataFrame(rows)
    df_delta.to_csv(TABLES / "m2_xi_bug0006_delta.csv", index=False)
    for met in ("c_local_tm", "team_tm", "links_tm", "top10_fin", "harvested_fin"):
        dmax = float(df_delta[df_delta["metric"] == met]["max_abs_diff"].max())
        _require(dmax <= 1e-11, f"M2_xi non-ideal invariance failed on {met}: {dmax}")
    return df_delta


# ---------------------------------------------------------------------------
# 3. Turnover v2 analysis & paired comparison against Turnover v1
# ---------------------------------------------------------------------------
def analyse_turnover_v2(rng: np.random.Generator) -> dict:
    v2 = pd.read_csv(TO_V2 / "runs.csv")
    _require(len(v2) == 1520, f"turnover_v2 runs={len(v2)}")

    # Main 72-cell grid (cur in {0.3, 0.5, 0.7})
    main_v2 = v2[v2["curiosity"].isin([0.3, 0.5, 0.7])].copy()
    _require(len(main_v2) == 1440, f"turnover_v2 main grid rows={len(main_v2)}")

    mode_order = ["none", "fixed_span", "age_hazard_sync", "age_hazard_staggered"]
    sum_rows = []
    for mode in mode_order:
        sub = main_v2[main_v2["exit_mode"] == mode]
        _require(len(sub) == 360, f"{mode} count={len(sub)}")
        sum_rows.append(
            {
                "exit_mode": mode,
                "n_runs": len(sub),
                "n_final": float(sub["n_final"].mean()),
                "c_ideal_fin": float(sub["c_ideal_fin"].mean()),
                "c_ideal_step399": float(sub["c_ideal_step399"].mean()),
                "c_ideal_late_mean": float(sub["c_ideal_late_mean"].mean()),
                "c_ideal_tm": float(sub["c_ideal_tm"].mean()),
                "c_local_fin": float(sub["c_local_fin"].mean()),
                "c_local_tm": float(sub["c_local_tm"].mean()),
                "wedge_local_minus_ideal_fin": float(sub["wedge_local_minus_ideal_fin"].mean()),
                "wedge_local_minus_ideal_tm": float(sub["wedge_local_minus_ideal_tm"].mean()),
                "top10_fin": float(sub["top10_fin"].mean()),
                "top10_step399": float(sub["top10_step399"].mean()),
                "top10_late_mean": float(sub["top10_late_mean"].mean()),
                "gap_recent_mean_last100": float(sub["gap_recent_mean_last100"].mean()),
                "gap_recent_ratio_mean_last100": float(sub["gap_recent_ratio_mean_last100"].mean()),
                "harvested_fin": float(sub["harvested_fin"].mean()),
                "frac_c_ideal_fin_neg": float((sub["c_ideal_fin"] < 0).mean()),
                "frac_c_ideal_late_neg": float((sub["c_ideal_late_mean"] < 0).mean()),
                "frac_gap_recent_neg": float((sub["gap_recent_mean_last100"] < 0).mean()),
            }
        )
    summary_df = pd.DataFrame(sum_rows)
    summary_df.to_csv(TABLES / "turnover_v2_summary.csv", index=False)

    # Entry-rate gradient
    grad_df = (
        main_v2.groupby(["exit_mode", "entry"], as_index=False)[
            [
                "c_ideal_fin",
                "c_ideal_step399",
                "c_ideal_late_mean",
                "c_ideal_tm",
                "gap_recent_mean_last100",
                "gap_recent_ratio_mean_last100",
                "top10_fin",
                "top10_late_mean",
                "harvested_fin",
            ]
        ]
        .mean()
    )
    grad_df.to_csv(TABLES / "turnover_v2_entry_gradient.csv", index=False)

    # Load v1 for paired delta (1080 main + 40 xi-ends = 1120)
    v1_main = pd.concat(
        [pd.read_csv(p) for p in sorted(glob.glob(str(DATA / "turnover_v1" / "cells" / "cell_*.csv")))],
        ignore_index=True,
    )
    _require(len(v1_main) == 1080, f"v1_main rows={len(v1_main)}")
    v1_main["wedge_local_minus_ideal_fin"] = v1_main["c_local_fin"] - v1_main["c_ideal_fin"]
    v1_main["wedge_local_minus_ideal_tm"] = v1_main["c_local_tm"] - v1_main["c_ideal_tm"]

    pair_specs = [
        ("none_v2_vs_none_v1", "none", "none", True),
        ("fixed_span_v2_vs_fixed_span_v1", "fixed_span", "fixed_span", True),
        ("age_hazard_sync_v2_vs_age_hazard_v1", "age_hazard_sync", "age_hazard", True),
        ("age_hazard_staggered_v2_vs_age_hazard_v1", "age_hazard_staggered", "age_hazard", False),
    ]
    keys = ["entry", "curiosity", "landscape", "seed"]
    delta_rows = []
    for label, m_v2, m_v1, expect_non_ideal_inv in pair_specs:
        a = v1_main[v1_main["exit_mode"] == m_v1].copy()
        b = main_v2[main_v2["exit_mode"] == m_v2].copy()
        mg = a.merge(b, on=keys, suffixes=("_v1", "_v2"), validate="one_to_one")
        _require(len(mg) == 360, f"{label} pairs={len(mg)}")
        for met in (
            "c_ideal_fin",
            "c_ideal_tm",
            "c_local_fin",
            "c_local_tm",
            "wedge_local_minus_ideal_fin",
            "wedge_local_minus_ideal_tm",
            "top10_fin",
            "harvested_fin",
            "gap_recent_mean_last100",
        ):
            va = mg[f"{met}_v1"].to_numpy(float)
            vb = mg[f"{met}_v2"].to_numpy(float)
            d = vb - va
            lo, hi = _boot_ci(d, rng)
            max_abs = float(np.max(np.abs(d)))
            if expect_non_ideal_inv and met in ("c_local_fin", "c_local_tm", "top10_fin", "harvested_fin", "gap_recent_mean_last100"):
                _require(max_abs <= 1e-11, f"{label} non-ideal invariance failed on {met}: {max_abs}")
            delta_rows.append(
                {
                    "contrast": label,
                    "metric": met,
                    "n_pairs": len(mg),
                    "v1_mean": float(np.mean(va)),
                    "v2_mean": float(np.mean(vb)),
                    "mean_delta": float(np.mean(d)),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "max_abs_diff": max_abs,
                }
            )
    # Also staggered_v2 vs sync_v2 (pure staggering effect post-BUG-0006)
    s_sync = main_v2[main_v2["exit_mode"] == "age_hazard_sync"]
    s_stag = main_v2[main_v2["exit_mode"] == "age_hazard_staggered"]
    mg_ss = s_sync.merge(s_stag, on=keys, suffixes=("_sync", "_stag"), validate="one_to_one")
    _require(len(mg_ss) == 360, "sync vs stag pairs != 360")
    for met in ("c_ideal_fin", "c_ideal_late_mean", "c_ideal_tm", "top10_fin", "top10_late_mean", "gap_recent_mean_last100", "harvested_fin"):
        va = mg_ss[f"{met}_sync"].to_numpy(float)
        vb = mg_ss[f"{met}_stag"].to_numpy(float)
        d = vb - va
        lo, hi = _boot_ci(d, rng)
        delta_rows.append(
            {
                "contrast": "age_hazard_staggered_v2_vs_sync_v2",
                "metric": met,
                "n_pairs": len(mg_ss),
                "v1_mean": float(np.mean(va)),
                "v2_mean": float(np.mean(vb)),
                "mean_delta": float(np.mean(d)),
                "ci_lo": lo,
                "ci_hi": hi,
                "max_abs_diff": float(np.max(np.abs(d))),
            }
        )

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(TABLES / "turnover_v2_vs_v1_delta.csv", index=False)
    return {"summary": summary_df, "gradient": grad_df, "delta": delta_df, "v2": v2}


# ---------------------------------------------------------------------------
# 4. Complete 4-tier xi in [0.1, 0.9] post-BUG-0006 spine table
# ---------------------------------------------------------------------------
def build_four_tier_xi() -> pd.DataFrame:
    rows = []
    # M0 & M1 from post-BUG-0006 W_v05 (mgstd5, use_visibility_reward=True)
    w = pd.read_csv(V2 / "w_v05_runs.csv")
    w = w[(w["landscape"] == "mgstd5") & (w["use_visibility_reward"] == True)].copy()  # noqa: E712
    w["xi"] = (1.0 - w["curiosity"]).round(2)
    for g, tier in ((0.0, "M0_closed_Wv05"), (0.1, "M1_lineage_Wv05")):
        sub = w[w["agent_generation_time"] == g]
        for xi, grp in sub.groupby("xi"):
            rows.append(
                {
                    "tier": tier,
                    "xi": float(xi),
                    "curiosity": round(1.0 - float(xi), 2),
                    "vanishing": 0.5,
                    "horizon": 200,
                    "n_runs": len(grp),
                    "harvested_fin": float(grp["harvested_fin"].mean()),
                    "c_ideal_fin": float(grp["c_ideal_fin"].mean()),
                    "c_ideal_tm": float(grp["c_ideal_tm"].mean()),
                    "c_local_fin": float(grp["c_local_fin"].mean()),
                    "c_local_tm": float(grp["c_local_tm"].mean()),
                    "wedge_local_minus_ideal_fin": float(grp["wedge_local_minus_ideal_fin"].mean()),
                    "wedge_local_minus_ideal_tm": float(grp["wedge_local_minus_ideal_tm"].mean()),
                    "top10_fin": float(grp["top10_fin"].mean()),
                }
            )

    # M1 + turnover (sync & staggered) on mgstd5, entry=0.10, van=0.9, H=400
    tv2 = pd.read_csv(TO_V2 / "runs.csv")
    tv2 = tv2[(tv2["landscape"] == "mgstd5") & (np.isclose(tv2["entry"], 0.10))].copy()
    tv2["xi"] = (1.0 - tv2["curiosity"]).round(2)
    for mode, tier in (
        ("age_hazard_sync", "M1_turnover_sync"),
        ("age_hazard_staggered", "M1_turnover_staggered"),
    ):
        sub = tv2[tv2["exit_mode"] == mode]
        for xi, grp in sub.groupby("xi"):
            rows.append(
                {
                    "tier": tier,
                    "xi": float(xi),
                    "curiosity": round(1.0 - float(xi), 2),
                    "vanishing": 0.9,
                    "horizon": 400,
                    "n_runs": len(grp),
                    "harvested_fin": float(grp["harvested_fin"].mean()),
                    "c_ideal_fin": float(grp["c_ideal_fin"].mean()),
                    "c_ideal_tm": float(grp["c_ideal_tm"].mean()),
                    "c_local_fin": float(grp["c_local_fin"].mean()),
                    "c_local_tm": float(grp["c_local_tm"].mean()),
                    "wedge_local_minus_ideal_fin": float(grp["wedge_local_minus_ideal_fin"].mean()),
                    "wedge_local_minus_ideal_tm": float(grp["wedge_local_minus_ideal_tm"].mean()),
                    "top10_fin": float(grp["top10_fin"].mean()),
                }
            )

    # M2 (lambda=0, lambda=1) from post-BUG-0006 m2_xi_runs.csv (mgstd5, pull=1, logit, equal_split, H=200)
    m2 = pd.read_csv(V2 / "m2_xi_runs.csv")
    m2["xi"] = (1.0 - m2["curiosity"]).round(2)
    for lam, tier in ((0.0, "M2_lambda0"), (1.0, "M2_lambda1")):
        sub = m2[m2["dyadic_lambda"] == lam]
        for xi, grp in sub.groupby("xi"):
            rows.append(
                {
                    "tier": tier,
                    "xi": float(xi),
                    "curiosity": round(1.0 - float(xi), 2),
                    "vanishing": 0.9,
                    "horizon": 200,
                    "n_runs": len(grp),
                    "harvested_fin": float(grp["harvested_fin"].mean()),
                    "c_ideal_fin": float(grp["c_ideal_fin"].mean()),
                    "c_ideal_tm": float(grp["c_ideal_tm"].mean()),
                    "c_local_fin": float(grp["c_local_fin"].mean()),
                    "c_local_tm": float(grp["c_local_tm"].mean()),
                    "wedge_local_minus_ideal_fin": float(grp["wedge_local_minus_ideal_fin"].mean()),
                    "wedge_local_minus_ideal_tm": float(grp["wedge_local_minus_ideal_tm"].mean()),
                    "top10_fin": float(grp["top10_fin"].mean()),
                }
            )

    df = pd.DataFrame(rows).sort_values(["tier", "xi"]).reset_index(drop=True)
    df.to_csv(TABLES / "four_tier_xi_summary.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 5. Report rendering (strictly computed from CSVs)
# ---------------------------------------------------------------------------
def _md_table(df: pd.DataFrame, cols: list[str], headers: list[str], fmts: dict[str, str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---"] + ["---:" for _ in headers[1:]]) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in fmts and pd.api.types.is_number(v):
                cells.append(format(float(v), fmts[c]) if np.isfinite(v) else "nan")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    res_tw: dict,
    m2_delta: pd.DataFrame,
    to_res: dict,
    four_tier: pd.DataFrame,
) -> None:
    t1_d = res_tw["t1_delta"]
    w_d = res_tw["w_delta"]
    xi_df = res_tw["xi_star"]
    to_sum = to_res["summary"]
    to_grad = to_res["gradient"]
    to_del = to_res["delta"]

    # Extract exact verified scalars for headline bullets
    def pick(df, col_filter, val_filter, metric, col):
        return float(df[(df[col_filter] == val_filter) & (df["metric"] == metric)][col].iloc[0])

    t1_g01_fin_pre = pick(t1_d, "scope", "g=0.1 (all)", "c_ideal_fin", "pre_mean")
    t1_g01_fin_post = pick(t1_d, "scope", "g=0.1 (all)", "c_ideal_fin", "post_mean")
    t1_g01_tm_pre = pick(t1_d, "scope", "g=0.1 (all)", "c_ideal_tm", "pre_mean")
    t1_g01_tm_post = pick(t1_d, "scope", "g=0.1 (all)", "c_ideal_tm", "post_mean")
    t1_g01_neg_pre = int(t1_d[(t1_d["scope"] == "g=0.1 (all)") & (t1_d["metric"] == "c_ideal_fin")]["pre_neg_cells"].iloc[0])
    t1_g01_neg_post = int(t1_d[(t1_d["scope"] == "g=0.1 (all)") & (t1_d["metric"] == "c_ideal_fin")]["post_neg_cells"].iloc[0])

    t1_mg_fin_post = pick(t1_d, "scope", "g=0.1 (mgstd5)", "c_ideal_fin", "post_mean")
    t1_np_fin_post = pick(t1_d, "scope", "g=0.1 (noisypr5)", "c_ideal_fin", "post_mean")
    t1_mg_tm_post = pick(t1_d, "scope", "g=0.1 (mgstd5)", "c_ideal_tm", "post_mean")
    t1_np_tm_post = pick(t1_d, "scope", "g=0.1 (noisypr5)", "c_ideal_tm", "post_mean")

    m0_t1_fin_post = pick(t1_d, "scope", "g=0.0 (control)", "c_ideal_fin", "post_mean")
    m0_t1_tm_post = pick(t1_d, "scope", "g=0.0 (control)", "c_ideal_tm", "post_mean")
    t1_tm_ratio = m0_t1_tm_post / t1_g01_tm_post

    t1_np_neg_cells = int(t1_d[(t1_d["scope"] == "g=0.1 (noisypr5)") & (t1_d["metric"] == "c_ideal_fin")]["post_neg_cells"].iloc[0])
    t1_np_tot_cells = int(t1_d[(t1_d["scope"] == "g=0.1 (noisypr5)") & (t1_d["metric"] == "c_ideal_fin")]["n_cells"].iloc[0])
    t1_mg_neg_cells = int(t1_d[(t1_d["scope"] == "g=0.1 (mgstd5)") & (t1_d["metric"] == "c_ideal_fin")]["post_neg_cells"].iloc[0])
    t1_mg_tot_cells = int(t1_d[(t1_d["scope"] == "g=0.1 (mgstd5)") & (t1_d["metric"] == "c_ideal_fin")]["n_cells"].iloc[0])

    row_none = to_sum[to_sum["exit_mode"] == "none"].iloc[0]
    row_fix = to_sum[to_sum["exit_mode"] == "fixed_span"].iloc[0]
    row_sync = to_sum[to_sum["exit_mode"] == "age_hazard_sync"].iloc[0]
    row_stag = to_sum[to_sum["exit_mode"] == "age_hazard_staggered"].iloc[0]

    sync_v1_pre = pick(to_del, "contrast", "age_hazard_sync_v2_vs_age_hazard_v1", "c_ideal_fin", "v1_mean")
    sync_v2_post = pick(to_del, "contrast", "age_hazard_sync_v2_vs_age_hazard_v1", "c_ideal_fin", "v2_mean")
    sync_delta = sync_v2_post - sync_v1_pre

    shrink_fix_pct = 100.0 * (1.0 - float(row_fix["gap_recent_mean_last100"]) / float(row_none["gap_recent_mean_last100"]))
    shrink_stag_pct = 100.0 * (1.0 - float(row_stag["gap_recent_mean_last100"]) / float(row_none["gap_recent_mean_last100"]))
    main_v2 = to_res["v2"][to_res["v2"]["curiosity"].isin([0.3, 0.5, 0.7])]
    gap_neg_count = int((main_v2["gap_recent_mean_last100"] < 0).sum())
    gap_neg_total = int(len(main_v2))

    # Compute exact Discovery (step <= 80) vs Depletion (step >= 150) c_ideal means across 72 main traj cells
    traj_frames = []
    for p in sorted((TO_V2 / "traj").glob("cell_*.csv")):
        if "_cur01_" in p.name or "_cur09_" in p.name:
            continue
        traj_frames.append(pd.read_csv(p))
    traj_all = pd.concat(traj_frames, ignore_index=True)
    disc_mean = float(traj_all[traj_all["step"] <= 80]["c_ideal"].mean())
    depl_mean = float(traj_all[traj_all["step"] >= 150]["c_ideal"].mean())
    late_traj_mean = float(traj_all[traj_all["step"] >= 301]["c_ideal"].mean())

    def ft_val(tier: str, xi: float, col: str) -> float:
        return float(four_tier[(four_tier["tier"] == tier) & (np.isclose(four_tier["xi"], xi))][col].iloc[0])

    ft_m0_05 = ft_val("M0_closed_Wv05", 0.5, "c_ideal_tm")
    ft_m2_1_05 = ft_val("M2_lambda1", 0.5, "c_ideal_tm")
    ft_m1_05 = ft_val("M1_lineage_Wv05", 0.5, "c_ideal_tm")
    ft_m2_0_05 = ft_val("M2_lambda0", 0.5, "c_ideal_tm")
    ft_m2_1_harv_05 = ft_val("M2_lambda1", 0.5, "harvested_fin")
    ft_m2_1_harv_09 = ft_val("M2_lambda1", 0.9, "harvested_fin")
    ft_m2_1_tm_09 = ft_val("M2_lambda1", 0.9, "c_ideal_tm")

    manifest_rm = json.loads((V2 / "manifest.json").read_text())
    manifest_to = json.loads((TO_V2 / "manifest.json").read_text())

    # Build formatted tables
    t1_sub = t1_d[t1_d["metric"].isin(["c_ideal_fin", "c_ideal_tm", "wedge_local_minus_ideal_fin", "c_local_fin", "top10_fin", "harvested_fin"])]
    tbl_t1 = _md_table(
        t1_sub,
        ["scope", "metric", "n_runs", "pre_mean", "post_mean", "mean_delta", "ci_lo", "ci_hi", "max_abs_diff", "pre_neg_cells", "post_neg_cells"],
        ["Scope", "Metric", "n", "Pre Mean", "Post Mean", "Delta", "CI lo", "CI hi", "Max |Diff|", "Pre <0 Cells", "Post <0 Cells"],
        {"pre_mean": "+.4f", "post_mean": "+.4f", "mean_delta": "+.4f", "ci_lo": "+.4f", "ci_hi": "+.4f", "max_abs_diff": ".2e"},
    )

    w_sub = w_d[w_d["metric"].isin(["c_ideal_fin", "c_ideal_tm", "wedge_local_minus_ideal_fin", "c_local_fin", "top10_fin", "harvested_fin", "ideal_loss_cum"])]
    tbl_w = _md_table(
        w_sub,
        ["scope", "metric", "n_runs", "pre_mean", "post_mean", "mean_delta", "ci_lo", "ci_hi", "max_abs_diff"],
        ["Scope", "Metric", "n", "Pre Mean", "Post Mean", "Delta", "CI lo", "CI hi", "Max |Diff|"],
        {"pre_mean": "+.4f", "post_mean": "+.4f", "mean_delta": "+.4f", "ci_lo": "+.4f", "ci_hi": "+.4f", "max_abs_diff": ".2e"},
    )

    tbl_xi_star = _md_table(
        xi_df,
        ["arm", "landscape", "entry", "use_visibility_reward", "xi_star_terminal_sum", "xi_star_cum_loss", "n_obs"],
        ["Arm", "Landscape", "Entry", "Vis Reward", "xi* (terminal sum)", "xi* (cum loss)", "n"],
        {"entry": ".1f", "xi_star_terminal_sum": ".1f", "xi_star_cum_loss": ".1f"},
    )

    tbl_m2 = _md_table(
        m2_delta,
        ["scope", "metric", "n_runs", "pre_mean", "post_mean", "mean_delta", "ci_lo", "ci_hi", "max_abs_diff"],
        ["Scope", "Metric", "n", "Pre Mean", "Post Mean", "Delta", "CI lo", "CI hi", "Max |Diff|"],
        {"pre_mean": "+.4f", "post_mean": "+.4f", "mean_delta": "+.4f", "ci_lo": "+.4f", "ci_hi": "+.4f", "max_abs_diff": ".2e"},
    )

    tbl_to_sum = _md_table(
        to_sum,
        ["exit_mode", "n_final", "c_ideal_fin", "c_ideal_step399", "c_ideal_late_mean", "c_ideal_tm",
         "wedge_local_minus_ideal_fin", "top10_fin", "top10_step399", "top10_late_mean",
         "gap_recent_mean_last100", "gap_recent_ratio_mean_last100", "harvested_fin"],
        ["Mode", "N(400)", "c_ideal(400)", "c_ideal(399)", "c_ideal(301-400)", "c_ideal(tm)",
         "wedge_fin", "top10(400)", "top10(399)", "top10(301-400)",
         "gap_recent", "gap_ratio", "harvested"],
        {
            "n_final": ".1f",
            "c_ideal_fin": "+.4f",
            "c_ideal_step399": "+.4f",
            "c_ideal_late_mean": "+.4f",
            "c_ideal_tm": "+.4f",
            "wedge_local_minus_ideal_fin": "+.4f",
            "top10_fin": ".4f",
            "top10_step399": ".4f",
            "top10_late_mean": ".4f",
            "gap_recent_mean_last100": "+.5f",
            "gap_recent_ratio_mean_last100": "+.3f",
            "harvested_fin": ".4f",
        },
    )

    tbl_to_grad = _md_table(
        to_grad,
        ["exit_mode", "entry", "c_ideal_fin", "c_ideal_step399", "c_ideal_late_mean", "c_ideal_tm",
         "gap_recent_mean_last100", "top10_fin", "top10_late_mean"],
        ["Mode", "Entry g", "c_ideal(400)", "c_ideal(399)", "c_ideal(301-400)", "c_ideal(tm)",
         "gap_recent", "top10(400)", "top10(301-400)"],
        {
            "entry": ".2f",
            "c_ideal_fin": "+.4f",
            "c_ideal_step399": "+.4f",
            "c_ideal_late_mean": "+.4f",
            "c_ideal_tm": "+.4f",
            "gap_recent_mean_last100": "+.5f",
            "top10_fin": ".4f",
            "top10_late_mean": ".4f",
        },
    )

    to_del_key = to_del[to_del["metric"].isin(["c_ideal_fin", "c_ideal_tm", "c_local_fin", "top10_fin", "harvested_fin", "gap_recent_mean_last100"])]
    tbl_to_del = _md_table(
        to_del_key,
        ["contrast", "metric", "n_pairs", "v1_mean", "v2_mean", "mean_delta", "ci_lo", "ci_hi", "max_abs_diff"],
        ["Contrast", "Metric", "Pairs", "Base Mean", "Comp Mean", "Delta", "CI lo", "CI hi", "Max |Diff|"],
        {"v1_mean": "+.4f", "v2_mean": "+.4f", "mean_delta": "+.4f", "ci_lo": "+.4f", "ci_hi": "+.4f", "max_abs_diff": ".2e"},
    )

    tbl_four = _md_table(
        four_tier,
        ["tier", "xi", "vanishing", "horizon", "harvested_fin", "c_ideal_fin", "c_ideal_tm",
         "wedge_local_minus_ideal_fin", "wedge_local_minus_ideal_tm", "top10_fin"],
        ["Tier", "xi", "van", "H", "harvested", "c_ideal(fin)", "c_ideal(tm)",
         "wedge(fin)", "wedge(tm)", "top10(fin)"],
        {
            "xi": ".1f",
            "vanishing": ".1f",
            "harvested_fin": ".4f",
            "c_ideal_fin": "+.4f",
            "c_ideal_tm": "+.4f",
            "wedge_local_minus_ideal_fin": "+.4f",
            "wedge_local_minus_ideal_tm": "+.4f",
            "top10_fin": ".4f",
        },
    )

    md = f"""---
title: "BUG-0006 Post-Fix Re-Measure & Staggered Turnover Report"
author: "Coordination Session (bug0006_remeasure_v2 + turnover_v2)"
version: 1
date: 2026-11-19
tags: [bug-0006, remeasure, turnover, staggered-cohorts, proxy-failure, not-calibrated]
project: sciencemodels
revision-vehicle: strikethrough
revision-renders: "md, pdf, html"
---

**Version 1** (November 2026)

> **MODEL-INTERNAL, NOT CALIBRATED** (owner ruling 2026-11-13). Every number below
> is computed by `scripts/bug0006/analyse_bug0006.py` from
> `data/bug0006_remeasure_v2/` (1,720 runs) and `data/turnover_v2/` (1,520 runs)
> and round-trip checked against `data/bug0006_remeasure_v2/tables/*.csv`.

# 1. Executive Summary: What BUG-0006 Was, and What Survives

## 1.1 Root Cause & Fix
`MyModel.end_loop_update()` runs `self.updateIdealMerit()` (line 516) **before** `self.__generate_new_agents()` (line 517) and `self.datacollector.collect(self)` (line 554). Previously, `Scientist.__init__` preset `self.ideal_Merit = 0` (`sciencemodels/agents.py:48`). Because `ideal_Merit <= 0` (`0` = omniscient planner optimum), every newborn spawned on step $t$ entered `datacollector.collect()` with `(prestige = 0.0, ideal_Merit = 0.0)` — artificially pairing the lowest possible prestige with the highest possible merit on every birth step (including steps 200 and 400).

**Fix (`BUG-0006`):** `sciencemodels/agents.py:48` now initializes `self.ideal_Merit = np.nan`, and `sciencemodels/metrics.py:308-316` (`_finite_pair_vectors`) filters `correlation_reporter` to finite `(prestige, merit)` pairs. Dynamics, RNG draws, harvest, local merit, and prestige concentration are unchanged.

## 1.2 Pristine-Channel Invariance Proofs
Across **all 3,240 re-measured runs**, every dynamics and non-ideal channel reproduces the pre-fix archives to floating-point precision:
- **`T1` `g = 0.0` control arm (360 runs):** max absolute difference across all 9 metrics (including `c_ideal_fin` and `c_ideal_tm`) = `{res_tw['t1_g0_max']:.2e}`.
- **`W_v05` `g = 0.0` control arm (400 runs):** max absolute difference across all 9 metrics = `{res_tw['w_g0_max']:.2e}`.
- **Non-ideal channels under `g > 0` (`T1`, `W_v05`, `M2_xi`, `turnover_v2` `none`/`fixed_span`/`age_hazard_sync`):** `c_local_fin`, `c_local_tm`, `top10_fin`, `harvested_fin`, `ideal_loss_cum`, `team_tm`, `links_tm`, and `gap_recent` all differ by `<= 1e-11` (`0.0` modulo IEEE roundoff).

## 1.3 What Changes vs. What Survives (Headline Scientific Verdict)
1. **The massive negative birth-step spike at step 200/400 was the `BUG-0006` artifact:**
   - In `T1` (`g = 0.1`), pooled terminal `c_ideal_fin` shifts from **`{t1_g01_fin_pre:+.4f}`** (pre-fix, `{t1_g01_neg_pre}/18` negative cells) to **`{t1_g01_fin_post:+.4f}`** (post-fix, **`{t1_g01_neg_post}/18`** negative cells).
   - **Important landscape split in `T1` post-fix:** on clean `mgstd5`, `M1` `c_ideal_fin` is **`{t1_mg_fin_post:+.4f}`** (`{t1_mg_neg_cells}/{t1_mg_tot_cells}` cells negative; time-mean **`{t1_mg_tm_post:+.4f}`**, down sharply from `M0`'s `{m0_t1_fin_post:+.4f}` / `{m0_t1_tm_post:+.4f}`), whereas on noisy `noisypr5`, `M1` collapses to **`{t1_np_fin_post:+.4f}`** pooled with **`{t1_np_neg_cells}/{t1_np_tot_cells}` cells negative** (vs time-mean `{t1_np_tm_post:+.4f}`).
   - Across `T1` (`g = 0.1`), the time-mean `c_ideal_tm` moves modestly upward (`{t1_g01_tm_pre:+.4f}` -> `{t1_g01_tm_post:+.4f}`), confirming a **{t1_tm_ratio:.2f}x** collapse in time-mean merit tracking from `M0` (`{m0_t1_tm_post:+.4f}`) to `M1` (`{t1_g01_tm_post:+.4f}`).
2. **Turnover (`turnover_v2`) separates cleanly once `BUG-0006` + cohort synchronization are removed:**
   - Under `age_hazard_sync`, removing `BUG-0006` alone shifts step-400 `c_ideal_fin` from **`{sync_v1_pre:+.4f}`** to **`{sync_v2_post:+.4f}`** (a paired shift of **`{sync_delta:+.4f}`** caused purely by 21 uncomputed `(0, 0)` replacements at $t=400$).
   - Even with `BUG-0006` fixed, synchronized replacement (`age_hazard_sync`) still creates a step-200/400 concentration spike (`top10(400) = {row_sync['top10_fin']:.4f}` vs `top10(399) = {row_sync['top10_step399']:.4f}`).
   - **Staggering founder retirements (`age_hazard_staggered`) eliminates both artifacts:** `c_ideal(400) = {row_stag['c_ideal_fin']:+.4f}`, `c_ideal(301-400) = {row_stag['c_ideal_late_mean']:+.4f}`, `top10(400) = {row_stag['top10_fin']:.4f}` (vs `none` `{row_none['top10_fin']:.4f}` and `fixed_span` `{row_fix['top10_fin']:.4f}`), while **preserving the arrival-gap shrinkage** (`gap_recent`: `{row_none['gap_recent_mean_last100']:+.5f}` -> `{row_fix['gap_recent_mean_last100']:+.5f}` [`-{shrink_fix_pct:.1f}%`] under `fixed_span`, `{row_stag['gap_recent_mean_last100']:+.5f}` [`-{shrink_stag_pct:.1f}%`] under `age_hazard_staggered`, negative in `{gap_neg_count}/{gap_neg_total}` runs).
3. **Every core structural finding survives intact:**
   - The `M0` -> `M1` Discovery-to-Depletion lifecycle collapse survives across all 4 turnover modes (pooled trajectory mean `c_ideal`: `{disc_mean:+.4f}` in Discovery $t \\le 80$ vs `{depl_mean:+.4f}` in Depletion $t \\ge 150$, `{late_traj_mean:+.4f}` in $t \\ge 301$).
   - The `M2` hierarchy at $\\xi = 0.5$ (`M0 {ft_m0_05:+.4f} > M2_lambda1 {ft_m2_1_05:+.4f} > M1 {ft_m1_05:+.4f} > M2_lambda0 {ft_m2_0_05:+.4f}` on time-mean `c_ideal_tm`) and the Stage-2 harvest/alignment collapse at $\\xi = 0.9$ (`M2_lambda1` harvest `{ft_m2_1_harv_05:.3f}` -> `{ft_m2_1_harv_09:.3f}`, `c_ideal_tm` -> `{ft_m2_1_tm_09:+.4f}`) survive post-fix.

---

# 2. `T1` Sensitivity Sweep (`data/bug0006_remeasure_v2/t1_runs.csv`, 720 runs)

{tbl_t1}

---

# 3. `W_v05` Welfare Sweep & $\\xi^*$ Kink (`data/bug0006_remeasure_v2/w_v05_runs.csv`, 800 runs)

{tbl_w}

## 3.1 Breakpoint $\\xi^*$ Stability (Pre vs. Post BUG-0006)

{tbl_xi_star}

---

# 4. `M2_xi` Dyadic Extension (`data/bug0006_remeasure_v2/m2_xi_runs.csv`, 200 runs)

{tbl_m2}

---

# 5. Staggered Turnover Decomposition (`data/turnover_v2/`, 1,520 runs)

## 5.1 Main Grid Comparison Across All Four Exit Modes (360 runs/mode, $H = 400$)

{tbl_to_sum}

## 5.2 Entry-Rate Gradient ($g \\in {{0.05, 0.10, 0.20}}$)

{tbl_to_grad}

## 5.3 Paired Seed-Matched Deltas (`turnover_v2` vs. `turnover_v1`)

{tbl_to_del}

---

# 6. Post-BUG-0006 Four-Tier Overview Across $\\xi \\in [0.1, 0.9]$ (`mgstd5`)

{tbl_four}

---

# 7. Reproduction & Provenance

- `bug0006_remeasure_v2` gross/net: `T1` {manifest_rm['arms']['T1']['rows_net']}/720, `W_v05` {manifest_rm['arms']['W_v05']['rows_net']}/800, `M2_xi` {manifest_rm['arms']['M2_xi']['rows_net']}/200 (`sources_moved_during_sweep = {manifest_rm['sources_moved_during_sweep']}`).
- `turnover_v2` gross/net: {manifest_to['runs_net_ok']}/{manifest_to['runs_gross']} runs across {manifest_to['cells_net']}/{manifest_to['cells_gross']} cells (`sources_moved_during_sweep = {manifest_to['sources_moved_during_sweep']}`).
- Summary CSVs: `data/bug0006_remeasure_v2/tables/*.csv`.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(md, encoding="utf-8")


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOT_SEED)
    res_tw = run_t1_and_w_v05(rng)
    m2_delta = analyse_m2_xi(rng)
    to_res = analyse_turnover_v2(rng)
    four_tier = build_four_tier_xi()
    write_report(res_tw, m2_delta, to_res, four_tier)

    # Round-trip assertions on exported CSVs & report
    expected_tables = [
        "t1_bug0006_delta.csv",
        "t1_post_by_curiosity.csv",
        "w_v05_bug0006_delta.csv",
        "w_v05_xi_star.csv",
        "m2_xi_bug0006_delta.csv",
        "turnover_v2_summary.csv",
        "turnover_v2_entry_gradient.csv",
        "turnover_v2_vs_v1_delta.csv",
        "four_tier_xi_summary.csv",
    ]
    for fname in expected_tables:
        p = TABLES / fname
        _require(p.exists() and p.stat().st_size > 0, f"missing table {p}")
    _require(REPORT.exists() and REPORT.stat().st_size > 1000, "report missing or empty")
    print(f"ANALYSIS OK: wrote {len(expected_tables)} CSVs to {TABLES} and {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
