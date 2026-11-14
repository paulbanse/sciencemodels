"""Loaders for the round-results report notebook.

ADR-0002: the notebook is a thin presentation wrapper; every number it shows
is computed here, from the round archives on disk, with a completeness
assertion before anything is plotted. Nothing is remembered -- each loader
asserts the gross shape it expects and reports the net rows it got.

Rounds covered (evidence ledger, AdmiralsLog/HANDOVER-2026-11-14.md section
1.3):

- T1 sensitivity      data/sensitivity_v1/          36 cells x 20 seeds
- T3 career flows     data/career_flows_v1/         9 cells  x 20 seeds
- T4 welfare probe    data/welfare_v1/              40 cells x 20 seeds
- W3-A prestige stock data/prestige_stock_v1/       36 cells x 20 seeds
- N1/N2 re-measure    data/remeasure_v1/            76 cells, 1,520 runs

All model-internal: nothing here is calibrated to empirical data (owner
ruling 2026-11-13).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"

WELFARE_LEG = {"2": "mg2", "6": "mg6"}
# the re-measure driver tagged the welfare 2-gaussian leg as "mgstd5"
REMEASURE_LEG = {"mg2": "mgstd5", "mg6": "mg6"}


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _gen_params_seed(series: pd.Series) -> pd.Series:
    """Seed identity lives inside ``generation_params`` for welfare cells."""
    return series.map(lambda s: ast.literal_eval(s)["seed"])


# --------------------------------------------------------------------------
# T1 sensitivity: 36 cells, step-level panel
# --------------------------------------------------------------------------
def sensitivity_cells() -> pd.DataFrame:
    """Cell-level means from the raw step-level panels (20 seeds each)."""
    files = sorted(
        p for p in (DATA / "sensitivity_v1").glob("cell_*.csv")
        if p.name != "cell_summary.csv"      # analysis artifact, not a cell
    )
    _require(len(files) == 36,
             f"T1 gross cells=36, found {len(files)}")
    rows = []
    for f in files:
        d = pd.read_csv(f)
        _require(d["seed"].nunique() == 20,
                 f"{f.name}: gross seeds=20, got {d['seed'].nunique()}")
        last = d["Step"].max()
        fin = d[d["Step"] == last]
        _require(len(fin) == 20, f"{f.name}: {len(fin)} final-step rows")
        rows.append({
            "curiosity": d["curiosity"].iloc[0],
            "vanishing": d["vanishing_factor_for_prestige"].iloc[0],
            "landscape": d["landscape"].iloc[0],
            "entry": d["agent_generation_time"].iloc[0],
            "cell_tag": d["cell_tag"].iloc[0],
            # per-seed final values
            "c_ideal_fin": fin["Corr_prestige_ideal_merit"].to_numpy(),
            "c_local_fin": fin["Corr_prestige_local_merit"].to_numpy(),
            "top10_fin": fin["top10pct_prestige_share"].to_numpy(),
            "seed": fin["seed"].to_numpy(),
            # per-seed time means (all steps of that seed)
            "c_ideal_tm": d.groupby("seed")["Corr_prestige_ideal_merit"]
                            .mean().reindex(fin["seed"]).to_numpy(),
            "c_local_tm": d.groupby("seed")["Corr_prestige_local_merit"]
                           .mean().reindex(fin["seed"]).to_numpy(),
            "top10_tm": d.groupby("seed")["top10pct_prestige_share"]
                         .mean().reindex(fin["seed"]).to_numpy(),
        })
    out = pd.DataFrame({
        k: np.concatenate([r[k] for r in rows])
        if isinstance(rows[0][k], np.ndarray) else
        np.repeat([r[k] for r in rows], 20)
        for k in rows[0]
    })
    out["wedge_fin"] = out["c_local_fin"] - out["c_ideal_fin"]
    out["wedge_tm"] = out["c_local_tm"] - out["c_ideal_tm"]
    _require(len(out) == 720, f"T1 net runs=720, got {len(out)}")
    return out


def sensitivity_summary() -> pd.DataFrame:
    """F1's input: the POST-AUDIT T1 arm, not the pre-fix sweep.

    The pairing audit (commit d3503ac) moved ideal-merit levels, so F1 reads
    the re-measured arm from ``data/remeasure_v1/runs.csv`` (720 T1 runs,
    36 cells) rather than the pre-fix step-level panels in
    ``data/sensitivity_v1/``. The pre-fix panels stay in
    ``sensitivity_cells()`` because they are the pre arm of the paired delta.
    """
    post = pd.read_csv(DATA / "remeasure_v1" / "runs.csv")
    t1 = post[post["origin"] == "T1"].copy()
    _require(len(t1) == 720, f"F1 post-audit gross runs=720, got {len(t1)}")
    t1 = t1.rename(columns={"agent_generation_time": "entry"})
    n_cells = t1.groupby(["curiosity", "vanishing", "landscape",
                          "entry"]).ngroups
    _require(n_cells == 36, f"F1 post-audit cells=36, got {n_cells}")
    g = (t1.groupby(["entry", "curiosity"], as_index=False)
           .agg(c_ideal_fin=("c_ideal_fin", "mean"),
                c_local_fin=("c_local_fin", "mean"),
                wedge_fin=("c_local_fin", "mean"),   # placeholder, fixed below
                top10_fin=("top10_fin", "mean"),
                n=("seed", "size")))
    # wedge is a difference of means here (means commute), so compute it from
    # the two columns rather than trusting any stored wedge column
    wedge = (t1.assign(w=t1["c_local_fin"] - t1["c_ideal_fin"])
                .groupby(["entry", "curiosity"], as_index=False)["w"].mean()
                .rename(columns={"w": "wedge_fin"}))
    g = g.drop(columns="wedge_fin").merge(wedge, on=["entry", "curiosity"])
    _require(len(g) == 6, f"F1 grid 2 entry x 3 curiosity = 6 rows, got {len(g)}")
    _require((g["n"] == 120).all(),
             f"F1 rows must carry 3 van x 2 leg x 20 seeds, got {g['n'].tolist()}")
    return g


def sensitivity_caption() -> dict:
    """The numbers F1's caption must state, computed from the same arm."""
    post = pd.read_csv(DATA / "remeasure_v1" / "runs.csv")
    t1 = post[post["origin"] == "T1"].copy()
    _require(len(t1) == 720, f"F1 caption gross runs=720, got {len(t1)}")
    on = t1[t1["agent_generation_time"] == 0.1]
    off = t1[t1["agent_generation_time"] == 0.0]
    cells_on = on.groupby(["curiosity", "vanishing", "landscape"]
                          )["c_ideal_fin"].mean()
    cells_off = off.groupby(["curiosity", "vanishing", "landscape"]
                            )["c_ideal_fin"].mean()
    cap = {
        "runs": int(len(t1)),
        "cells": int(len(cells_on) + len(cells_off)),
        "entry0_mean": float(off["c_ideal_fin"].mean()),
        "entry01_mean": float(on["c_ideal_fin"].mean()),
        "entry01_cells": int(len(cells_on)),
        "entry01_negative": int((cells_on < 0).sum()),
        "top10_entry0": float(off["top10_fin"].mean()),
        "top10_entry01": float(on["top10_fin"].mean()),
        "local_entry01": float(on["c_local_fin"].mean()),
    }
    # sign flips against the pre-fix arm (CONT3 claims 0/36)
    pre = pd.read_csv(DATA / "sensitivity_v1" / "cell_summary.csv")
    pre_cells = pre.rename(columns={
        "vanishing_factor_for_prestige": "vanishing",
        "agent_generation_time": "entry",
        "Corr_prestige_ideal_merit_final": "pre",
    })[["curiosity", "vanishing", "landscape", "entry", "pre"]]
    post_cells = (t1.groupby(["curiosity", "vanishing", "landscape",
                              "agent_generation_time"], as_index=False)
                    ["c_ideal_fin"].mean()
                    .rename(columns={"agent_generation_time": "entry",
                                     "c_ideal_fin": "post"}))
    mg = pre_cells.merge(post_cells,
                         on=["curiosity", "vanishing", "landscape", "entry"],
                         validate="one_to_one")
    _require(len(mg) == 36, f"F1 pre/post cell match=36, got {len(mg)}")
    flips = ((np.sign(mg["pre"]) != np.sign(mg["post"]))
             & (mg["pre"].abs() > 1e-9) & (mg["post"].abs() > 1e-9))
    cap["sign_flips"] = int(flips.sum())
    cap["matched_cells"] = int(len(mg))
    return cap


# --------------------------------------------------------------------------
# T3 career flows: clean AND noisy landscape analysis tables
# --------------------------------------------------------------------------
def career_flows() -> pd.DataFrame:
    d = pd.read_csv(DATA / "career_flows_v1" / "careerB_analysis.csv")
    _require(len(d) == 9, f"T3 gross cells=9, got {len(d)}")
    _require((d["status"] == "ok").all(),
             f"T3 non-ok cells: {d.loc[d.status != 'ok', 'cell_tag'].tolist()}")
    _require((d["seeds_ok"] == 20).all(), "T3 every cell must carry 20 seeds")
    _require(not d.duplicated(["agent_generation_time", "curiosity"]).any(),
             "T3 duplicate (entry, curiosity) rows")
    return d


def career_flows_noisy() -> pd.DataFrame:
    d = pd.read_csv(DATA / "career_flows_noisy_v1" / "noisyB_analysis.csv")
    _require(len(d) == 9, f"T3-noisy gross cells=9, got {len(d)}")
    _require((d["status"] == "ok").all(),
             f"T3-noisy non-ok: {d.loc[d.status != 'ok', 'cell_tag'].tolist()}")
    _require((d["seeds_ok"] == 20).all(),
             "T3-noisy every cell must carry 20 seeds")
    _require(not d.duplicated(["agent_generation_time", "curiosity"]).any(),
             "T3-noisy duplicate (entry, curiosity) rows")
    _require(set(d["landscape"].unique()) == {"noisypr5"},
             f"T3-noisy landscape={sorted(d['landscape'].unique())}")
    return d


def career_flows_comparison() -> pd.DataFrame:
    """F2's input: gap ratio by entry, landscape x statistic (4 series).

    Both headline levels live in this frame with their statistic named:
    the clean FINAL level (-0.71) and the noisy LATE level (-0.51), so the
    figure never compares one statistic on one line with another on the next.
    """
    clean, noisy = career_flows(), career_flows_noisy()
    rows = []
    for landscape, d in (("clean", clean), ("noisy", noisy)):
        for stat, col in (("final", "gap_ratio_final"),
                          ("late", "gap_ratio_mean_late")):
            g = d.groupby("agent_generation_time", as_index=False)[col].mean()
            for r in g.itertuples():
                rows.append({"entry": r.agent_generation_time,
                             "landscape": landscape,
                             "series": f"{landscape}, {stat}",
                             "gap_ratio": getattr(r, col)})
    out = pd.DataFrame(rows)
    _require(len(out) == 12,
             f"F2 grid 3 entry x 2 landscape x 2 stat = 12 rows, got {len(out)}")
    _require(not out.duplicated(["entry", "series"]).any(),
             "F2 duplicate (entry, series) rows")
    return out


def career_flows_ideal_corr() -> pd.DataFrame:
    """F2b's input: final prestige-ideal correlation by entry, per landscape.

    This is where the slow-entry decoupling gradient is visible: the most
    negative correlation sits at the LOWEST entry rate on both terrains.
    """
    rows = []
    for landscape, d in (("clean", career_flows()),
                         ("noisy", career_flows_noisy())):
        g = d.groupby("agent_generation_time", as_index=False)[
            "ideal_corr_final"].mean()
        for r in g.itertuples():
            rows.append({"entry": r.agent_generation_time,
                         "landscape": landscape,
                         "ideal_corr": r.ideal_corr_final})
    out = pd.DataFrame(rows)
    _require(len(out) == 6,
             f"F2b grid 3 entry x 2 landscape = 6 rows, got {len(out)}")
    _require(not out.duplicated(["entry", "landscape"]).any(),
             "F2b duplicate (entry, landscape) rows")
    return out


# --------------------------------------------------------------------------
# T4 welfare probe: 40 cells, one row per seed
# --------------------------------------------------------------------------
def welfare_cells() -> pd.DataFrame:
    files = sorted((DATA / "welfare_v1").glob("cell_*.csv"))
    _require(len(files) == 40, f"T4 gross cells=40, found {len(files)}")
    frames = []
    for f in files:
        m = re.match(r"cell_cur(\d+)_vis(\d)_gen(\d+)_mg(\d+)\.csv", f.name)
        _require(m is not None, f"unparsed welfare cell name: {f.name}")
        cur, vis, gen, gauss = m.groups()
        d = pd.read_csv(f)
        _require(len(d) == 20, f"{f.name}: gross seeds=20, got {len(d)}")
        d = d.copy()
        d["visibility_weight"] = 1.0 - int(cur) / 10
        d["use_visibility_reward"] = vis == "1"
        d["entry"] = int(gen) / 10          # gen00 -> 0.0, gen01 -> 0.1
        d["legibility"] = WELFARE_LEG[gauss]
        d["seed"] = _gen_params_seed(d["generation_params"])
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    _require(len(out) == 800, f"T4 net runs=800, got {len(out)}")
    _require(out["seed"].nunique() == 20, "T4 seeds must be 0..19")
    return out


def welfare_summary() -> pd.DataFrame:
    """F3's input: cell means over seeds, visibility toggle averaged away.

    The toggle's own spread is carried in ``toggle_gap`` so the figure's
    title can state it instead of hiding it in the average.
    """
    d = welfare_cells()
    g = (d.groupby(["visibility_weight", "entry", "legibility"], as_index=False)
           .agg(loss=("ideal_merit_sum", "mean"),
                n=("seed", "size")))
    tog = (d.groupby(["visibility_weight", "entry", "legibility",
                      "use_visibility_reward"], as_index=False)
             ["ideal_merit_sum"].mean())
    spread = (tog.groupby(["visibility_weight", "entry", "legibility"])
                 ["ideal_merit_sum"].agg(lambda s: float(s.max() - s.min())))
    g["toggle_gap"] = [
        float(spread.loc[(r.visibility_weight, r.entry, r.legibility)])
        for r in g.itertuples()
    ]
    g["series"] = np.where(g["entry"] == 0.0, "no entry", "entry 0.1") \
        + " / " + g["legibility"]
    _require(len(g) == 20,
             f"F3 grid 5 weight x 2 entry x 2 leg = 20 rows, got {len(g)}")
    _require((g["n"] == 40).all(),
             f"F3 rows must carry 2 vis toggles x 20 seeds = 40, got "
             f"{g['n'].tolist()}")
    return g


# --------------------------------------------------------------------------
# W3-A prestige stock: the round's own cell summary
# --------------------------------------------------------------------------
def prestige_stock_summary() -> pd.DataFrame:
    d = pd.read_csv(DATA / "prestige_stock_v1" / "summary_cell.csv")
    _require(len(d) == 36, f"W3-A gross cells=36, got {len(d)}")
    _require((d["n_seeds"] == 20).all(), "W3-A every cell must carry 20 seeds")
    _require(not d.duplicated(["stock", "agent_generation_time",
                               "initial_curiosity", "landscape"]).any(),
             "W3-A duplicate cell keys")
    return d


# --------------------------------------------------------------------------
# N1/N2 re-measure: paired pre-fix vs post-fix runs
# --------------------------------------------------------------------------
PAIR_REPORTERS = ["c_ideal_fin", "c_ideal_tm", "c_local_fin", "c_local_tm",
                  "top10_fin", "wedge_fin", "wedge_tm"]


def remeasure_pairs() -> tuple[pd.DataFrame, dict]:
    """Match pre-fix runs to post-fix runs on every grid key + seed.

    Pre-fix arm: T1 from data/sensitivity_v1 (produced BEFORE the
    ideal-merit pairing audit, commit d3503ac). Post-fix arm:
    data/remeasure_v1/runs.csv, origin T1.

    The welfare (W) arm of the re-measure is EXCLUDED from the paired delta
    by design, not by oversight: the original welfare sweep ran at
    ``vanishing_factor_for_prestige = 0.5`` and the re-measure re-ran it at
    ``0.9``, so that arm carries two changes at once (pairing fix + vanishing
    unification) and its delta cannot be attributed to the pairing fix
    alone. The mismatch is measured below and reported in ``counts``.
    """
    post = pd.read_csv(DATA / "remeasure_v1" / "runs.csv")
    _require(len(post) == 1520, f"re-measure gross runs=1520, got {len(post)}")

    # ---- T1 pre arm
    t1_pre = sensitivity_cells().rename(columns={"vanishing": "vanishing"})
    t1_post = post[post["origin"] == "T1"].copy()
    # runs.csv stores wedge = c_ideal - c_local, the OPPOSITE of the pre-arm
    # convention (c_local - c_ideal). Assert the archive's convention, then
    # derive the wedge from the components on BOTH sides so the paired delta
    # compares the same quantity rather than two sign conventions.
    _require(np.allclose(t1_post["wedge_fin"],
                         t1_post["c_ideal_fin"] - t1_post["c_local_fin"],
                         atol=1e-12),
             "runs.csv wedge_fin no longer equals c_ideal - c_local")
    t1_post["wedge_fin"] = (t1_post["c_local_fin"] - t1_post["c_ideal_fin"])
    t1_post["wedge_tm"] = (t1_post["c_local_tm"] - t1_post["c_ideal_tm"])
    keys_t1 = ["curiosity", "vanishing", "landscape", "entry", "seed"]
    t1_pre = t1_pre[keys_t1 + PAIR_REPORTERS]
    t1_post = t1_post.rename(columns={"agent_generation_time": "entry"})
    t1_post = t1_post[keys_t1 + PAIR_REPORTERS]
    t1 = t1_pre.merge(t1_post, on=keys_t1, suffixes=("_pre", "_post"),
                      validate="one_to_one")
    _require(len(t1) == 720, f"T1 paired runs=720, got {len(t1)}")

    # ---- W arm: diagnose the mismatch instead of pairing it
    w_pre = welfare_cells()
    w_post = post[post["origin"] == "W"]
    w_pre_van = sorted(w_pre["vanishing_factor_for_prestige"].unique())
    w_post_van = sorted(w_post["vanishing"].unique())
    _require(w_pre_van != w_post_van,
             "expected the W-arm vanishing mismatch to be detected")
    pairs = t1.copy()
    pairs["origin"] = "T1"

    rows = []
    for rep in PAIR_REPORTERS:
        sub = pairs[[f"{rep}_pre", f"{rep}_post"]].dropna()
        delta = sub[f"{rep}_post"] - sub[f"{rep}_pre"]
        rows.append({
            "reporter": rep,
            "family": ("ideal side" if "ideal" in rep else
                       "local side" if "local" in rep else
                       "concentration" if "top10" in rep else "wedge"),
            "delta": delta.mean(),
            "se": delta.std(ddof=1) / np.sqrt(len(delta)),
            "n": len(delta),
            "max_abs": float(np.abs(delta).max()),
        })
    deltas = pd.DataFrame(rows)
    _require((deltas["n"] == 720).all(),
             f"every reporter must pair all 720 T1 runs, got "
             f"{deltas['n'].tolist()}")
    deltas["arm"] = "T1"

    # ---- optional W_v05 arm: only paired if the re-measure round banked it.
    # The original welfare sweep ran at vanishing 0.5, so it IS the v05 arm
    # and pairs against w_v05_runs.csv on the same keys as the T1 arm.
    w_v05_path = DATA / "remeasure_v1" / "w_v05_runs.csv"
    counts = {
        "post_gross": int(len(post)),
        "t1_paired_net": int(len(t1)),
        "w_paired_net": 0,
        "w_excluded": (f"W arm re-run at vanishing {w_post_van} against "
                       f"{w_pre_van} in the original sweep -- two changes "
                       f"at once, delta not attributable to the pairing fix"),
        "w_post_rows": int(len(w_post)),
        "w_pre_rows": int(len(w_pre)),
        "wedge_sign": ("runs.csv stores c_ideal - c_local; both arms "
                       "recomputed as c_local - c_ideal before differencing"),
        "w_v05_file": str(w_v05_path.relative_to(REPO)),
        "w_v05_present": bool(w_v05_path.exists()),
    }
    if w_v05_path.exists():
        wv = pd.read_csv(w_v05_path)
        need = ({"curiosity", "vanishing", "landscape",
                 "agent_generation_time", "use_visibility_reward", "seed"}
                | {"c_ideal_fin", "c_ideal_tm", "c_local_fin", "c_local_tm",
                   "top10_fin"})
        missing = sorted(need - set(wv.columns))
        _require(not missing,
                 f"{w_v05_path.name} missing columns: {missing}")
        # The cell files carry no wedge column at all, and the one aggregate
        # that does (runs.csv) stores it with the OPPOSITE sign -- so derive
        # it from the components on both sides instead of trusting a column.
        if {"wedge_fin", "wedge_tm"} <= set(wv.columns):
            _require(np.allclose(wv["wedge_fin"],
                                 wv["c_ideal_fin"] - wv["c_local_fin"],
                                 atol=1e-12),
                     "w_v05 wedge_fin not in the c_ideal - c_local convention")
        wv["wedge_fin"] = wv["c_local_fin"] - wv["c_ideal_fin"]
        wv["wedge_tm"] = wv["c_local_tm"] - wv["c_ideal_tm"]
        wv = wv.rename(columns={"agent_generation_time": "entry"})
        keys_w = ["curiosity", "vanishing", "landscape", "entry",
                  "use_visibility_reward", "seed"]
        wv = wv[keys_w + PAIR_REPORTERS]
        w_pre = welfare_cells().rename(columns={
            "initial_curiosity": "curiosity",
            "vanishing_factor_for_prestige": "vanishing",
            "legibility": "landscape",
        })
        w_pre["landscape"] = w_pre["landscape"].map(REMEASURE_LEG)
        w_pre = w_pre.rename(columns={
            "corr_prestige_ideal_merit_final": "c_ideal_fin",
            "Mean_of_Corr_prestige_ideal_merit": "c_ideal_tm",
            "corr_prestige_local_merit_final": "c_local_fin",
            "Mean_of_Corr_prestige_local_merit": "c_local_tm",
            "top10pct_prestige_share": "top10_fin",
        })
        w_pre["wedge_fin"] = w_pre["c_local_fin"] - w_pre["c_ideal_fin"]
        w_pre["wedge_tm"] = w_pre["c_local_tm"] - w_pre["c_ideal_tm"]
        w_pre = w_pre[keys_w + PAIR_REPORTERS]
        wp = w_pre.merge(wv, on=keys_w, suffixes=("_pre", "_post"),
                         validate="one_to_one")
        _require(len(wp) == 800,
                 f"W_v05 paired runs=800, got {len(wp)}")
        rows_w = []
        for rep in PAIR_REPORTERS:
            delta = wp[f"{rep}_post"] - wp[f"{rep}_pre"]
            rows_w.append({
                "reporter": rep,
                "family": ("ideal side" if "ideal" in rep else
                           "local side" if "local" in rep else
                           "concentration" if "top10" in rep else "wedge"),
                "delta": delta.mean(),
                "se": delta.std(ddof=1) / np.sqrt(len(delta)),
                "n": len(delta),
                "max_abs": float(np.abs(delta).max()),
                "arm": "W_v05",
            })
        w_deltas = pd.DataFrame(rows_w)
        _require((w_deltas["n"] == 800).all(),
                 f"W_v05 every reporter must pair 800 runs, got "
                 f"{w_deltas['n'].tolist()}")
        deltas = pd.concat([deltas, w_deltas], ignore_index=True)
        counts["w_v05_paired_net"] = int(len(wp))
    else:
        counts["w_v05_paired_net"] = 0
        counts["w_v05_note"] = ("not banked by the re-measure round at the "
                                "time of this build; F5 stays T1-only")
    return deltas, counts


def manifest() -> pd.DataFrame:
    """Gross vs net for every round the notebook draws on (computed)."""
    cap = sensitivity_caption()
    rows = [
        {"round": "T1 sensitivity (post-audit arm)",
         "gross": "36 cells x 20 seeds = 720",
         "net": f"{cap['runs']} runs in data/remeasure_v1/runs.csv (F1); "
                f"{len(sensitivity_cells())} pre-fix runs (F5 pre arm)"},
        {"round": "T3 career flows (clean + noisy)",
         "gross": "9 + 9 cells x 20 seeds = 360",
         "net": f"{len(career_flows())} + {len(career_flows_noisy())} cells, "
                f"{int(career_flows()['seeds_ok'].sum())} + "
                f"{int(career_flows_noisy()['seeds_ok'].sum())} seeds"},
        {"round": "T4 welfare probe", "gross": "40 cells x 20 seeds = 800",
         "net": f"{len(welfare_cells())} runs"},
        {"round": "W3-A prestige stock", "gross": "36 cells x 20 seeds = 720",
         "net": f"{len(prestige_stock_summary())} cells x "
                f"{int(prestige_stock_summary()['n_seeds'].sum())} seeds"},
        {"round": "N1/N2 re-measure", "gross": "1,520 runs (76 cells)",
         "net": f"{remeasure_pairs()[1]['t1_paired_net']} T1 pairs "
                f"+ {remeasure_pairs()[1]['w_v05_paired_net']} W_v05 pairs "
                f"(W arm excluded: {remeasure_pairs()[1]['w_paired_net']})"},
    ]
    return pd.DataFrame(rows)
