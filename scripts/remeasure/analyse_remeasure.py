"""Analysis for the post-audit re-measure pass (N1/N2 arms + the W_v05 bridge).

Paired, seed-matched comparison of the PRE-fix archives against the POST-fix
re-measure. The only code change between the two is the ideal-merit pairing fix
(commit ``d3503ac``), so the expected signature is: every reporter that does not
read the ideal channel is bit-identical, and only ``Corr_prestige_ideal_merit``
(and everything derived from it, i.e. the wedge) moves.

Pre-fix archives
----------------
``data/sensitivity_v1/cell_*.csv``  36 cells x 20 seeds, step-level panels
``data/welfare_v1/cell_*.csv``      40 cells x 20 seeds, one row per seed

Post-fix re-measure
-------------------
``data/remeasure_v1/runs.csv``      legacy arms: T1 (van 0.5/0.9/1.0) + W (van 0.9)
``data/remeasure_v1/w_v05_runs.csv`` W_v05 arm: welfare grid at van 0.5, i.e. the
                                    configuration the archive actually used

Wedge convention
----------------
House convention throughout: ``wedge = c_local - c_ideal`` (positive when
prestige tracks the visible proxy better than it tracks quality). This matches
``output/reports/sensitivity_map_v1.md`` and ``prestige_stock_v1.md`` and the
in-memory flip in ``scripts/report_notebook/round_report_data.py``.

``runs.csv`` stores the OPPOSITE sign (``c_ideal - c_local``) because that is its
published contract with the reporting notebook. This script does not touch that
file: it asserts the stored sign on the raw CSV (so the discrepancy is proven,
not assumed) and recomputes the wedge from the components on BOTH sides before
differencing.

Everything here is model-internal. Nothing is calibrated to empirical data
(owner ruling 2026-11-13).

Run (from the repo root):

    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/remeasure/analyse_remeasure.py
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "remeasure_v1")
TABLES = os.path.join(OUT, "tables")

# pre-fix welfare legibility tag -> the re-measure driver's landscape tag.
# both legs are multipleGaussians with std 5; mg2 is the 2-Gaussian landscape
# that the re-measure driver happens to call "mgstd5".
WELFARE_LEG_TO_LAND = {"2": "mgstd5", "6": "mg6"}

# reporters compared on both sides. `harvested` exists only as a terminal value
# in the re-measure, so it is compared final-only.
REPORTERS = [
    "c_ideal_fin",
    "c_ideal_tm",
    "c_local_fin",
    "c_local_tm",
    "top10_fin",
    "harvested_fin",
]


def _require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _gen_params_seed(series):
    """Seed identity lives inside ``generation_params`` for the welfare cells."""
    return series.map(lambda s: ast.literal_eval(s)["seed"])


def _family(rep):
    if "ideal" in rep:
        return "ideal side"
    if "local" in rep:
        return "local side"
    if "top10" in rep:
        return "concentration"
    return "harvest"


# ---------------------------------------------------------------------------
# T1 sensitivity arm: pre-fix step-level panels
# ---------------------------------------------------------------------------
def load_t1_pre():
    """Per-seed final value and per-seed time mean, recomputed from the panels.

    The archive stores step-level rows (Step 0..200 per seed), so both the
    terminal value and the all-step mean are re-derived here rather than read
    from the round's own summary -- that is what makes the comparison paired at
    the seed level.
    """
    cells = sorted(
        p for p in os.listdir(os.path.join(DATA, "sensitivity_v1"))
        if p.startswith("cell_") and p.endswith(".csv")
        and p != "cell_summary.csv"
    )
    _require(len(cells) == 36, "T1 gross cells=36, found %d" % len(cells))
    rows = []
    for name in cells:
        d = pd.read_csv(os.path.join(DATA, "sensitivity_v1", name))
        _require(d["seed"].nunique() == 20, "%s: seeds=%d" % (name, d["seed"].nunique()))
        last = int(d["Step"].max())
        fin = d[d["Step"] == last]
        _require(len(fin) == 20, "%s: %d final rows" % (name, len(fin)))
        agg = {
            "curiosity": d["curiosity"].iloc[0],
            "vanishing": d["vanishing_factor_for_prestige"].iloc[0],
            "landscape": d["landscape"].iloc[0],
            "agent_generation_time": d["agent_generation_time"].iloc[0],
        }
        for s in range(20):
            sub = d[d["seed"] == s]
            rec = sub[sub["Step"] == last].iloc[0]
            # Time means come from ``Series.mean()`` on the seed's own series,
            # NOT from ``groupby(...)[col].mean()``. The two paths accumulate in
            # different orders and disagree by 1 ulp (2.2e-16) on ~36% of seeds,
            # which would masquerade as a model change in a bit-identity test.
            # ``Series.mean()`` is the path the model's own datacollector uses,
            # so this compares like with like and the deltas can be asserted at
            # exactly zero rather than against a tolerance.
            rows.append(
                dict(
                    agg,
                    seed=s,
                    c_ideal_fin=float(rec["Corr_prestige_ideal_merit"]),
                    c_local_fin=float(rec["Corr_prestige_local_merit"]),
                    top10_fin=float(rec["top10pct_prestige_share"]),
                    harvested_fin=float(rec["percentage_knowledge_harvested"]),
                    c_ideal_tm=float(sub["Corr_prestige_ideal_merit"].mean()),
                    c_local_tm=float(sub["Corr_prestige_local_merit"].mean()),
                )
            )
    out = pd.DataFrame(rows)
    _require(len(out) == 720, "T1 pre runs=720, got %d" % len(out))
    return out


def load_t1_post():
    post = pd.read_csv(os.path.join(OUT, "runs.csv"))
    # published contract of the legacy file: len == 1520 and the opposite sign.
    _require(len(post) == 1520, "runs.csv gross runs=1520, got %d" % len(post))
    _require(
        np.allclose(
            post["wedge_fin"],
            post["c_ideal_fin"] - post["c_local_fin"],
            atol=1e-12,
            equal_nan=True,
        ),
        "runs.csv wedge_fin is no longer c_ideal - c_local (contract broken)",
    )
    t1 = post[post["origin"] == "T1"].copy()
    _require(len(t1) == 720, "runs.csv T1 rows=720, got %d" % len(t1))
    t1 = t1.rename(columns={"agent_generation_time": "entry"})
    return t1


def pair_t1():
    pre = load_t1_pre()
    post = load_t1_post()
    keys = ["curiosity", "vanishing", "landscape", "entry", "seed"]
    pre = pre.rename(columns={"agent_generation_time": "entry"})
    keep = keys + REPORTERS
    m = pre[keep].merge(
        post[keys + REPORTERS], on=keys, suffixes=("_pre", "_post"),
        validate="one_to_one",
    )
    _require(len(m) == 720, "T1 paired runs=720, got %d" % len(m))
    return m


# ---------------------------------------------------------------------------
# Welfare arm: pre-fix archive vs the W_v05 re-measure
# ---------------------------------------------------------------------------
def load_w_pre():
    d_dir = os.path.join(DATA, "welfare_v1")
    files = sorted(f for f in os.listdir(d_dir) if f.startswith("cell_") and f.endswith(".csv"))
    _require(len(files) == 40, "T4 gross cells=40, found %d" % len(files))
    frames = []
    for name in files:
        # cell_cur05_vis1_gen00_mg2.csv
        stem = name[len("cell_"):-len(".csv")].split("_")
        cur = int(stem[0][3:]) / 10.0
        vis = stem[1][3:] == "1"
        gen = int(stem[2][3:]) / 10.0
        leg = stem[3][2:]
        d = pd.read_csv(os.path.join(d_dir, name))
        _require(len(d) == 20, "%s: gross seeds=20, got %d" % (name, len(d)))
        d = d.copy()
        d["curiosity"] = cur
        d["vanishing"] = 0.5
        d["landscape"] = WELFARE_LEG_TO_LAND[leg]
        d["entry"] = gen
        d["use_visibility_reward"] = vis
        d["seed"] = _gen_params_seed(d["generation_params"])
        frames.append(
            d[[
                "curiosity", "vanishing", "landscape", "entry",
                "use_visibility_reward", "seed",
                "corr_prestige_ideal_merit_final", "corr_prestige_local_merit_final",
                "Mean_of_Corr_prestige_ideal_merit", "Mean_of_Corr_prestige_local_merit",
                "top10pct_prestige_share", "percentage_knowledge_harvested",
                "ideal_merit_sum", "ideal_merit_mean",
            ]]
        )
    out = pd.concat(frames, ignore_index=True)
    _require(len(out) == 800, "T4 net runs=800, got %d" % len(out))
    _require(out["seed"].nunique() == 20, "T4 seeds must be 0..19")
    return out.rename(
        columns={
            "corr_prestige_ideal_merit_final": "c_ideal_fin",
            "corr_prestige_local_merit_final": "c_local_fin",
            "Mean_of_Corr_prestige_ideal_merit": "c_ideal_tm",
            "Mean_of_Corr_prestige_local_merit": "c_local_tm",
            "top10pct_prestige_share": "top10_fin",
            "percentage_knowledge_harvested": "harvested_fin",
        }
    )


def load_w_post():
    path = os.path.join(OUT, "w_v05_runs.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    _require(d["origin"].unique().tolist() == ["Wv05"], "unexpected W_v05 origins")
    _require(len(d) == 800, "W_v05 runs=800, got %d" % len(d))
    _require(d["vanishing"].unique().tolist() == [0.5], "W_v05 must be vanishing 0.5")
    d = d.rename(columns={"agent_generation_time": "entry"})
    # stored wedge columns are house-convention; assert them before relying on them
    _require(
        np.allclose(d["wedge_local_minus_ideal_fin"], d["c_local_fin"] - d["c_ideal_fin"], atol=1e-12),
        "w_v05_runs.csv wedge_local_minus_ideal_fin is not c_local - c_ideal",
    )
    return d


def pair_w(pre, post):
    keys = ["curiosity", "vanishing", "landscape", "entry", "use_visibility_reward", "seed"]
    cols = keys + REPORTERS + ["ideal_merit_sum", "ideal_merit_mean"]
    m = pre[cols].merge(
        post[cols], on=keys, suffixes=("_pre", "_post"), validate="one_to_one"
    )
    _require(len(m) == 800, "W paired runs=800, got %d" % len(m))
    return m


# ---------------------------------------------------------------------------
# Delta tables
# ---------------------------------------------------------------------------
def channel_summary(pairs, label):
    rows = []
    for rep in REPORTERS:
        a = pairs[rep + "_pre"].to_numpy()
        b = pairs[rep + "_post"].to_numpy()
        ok = ~(np.isnan(a) | np.isnan(b))
        d = b[ok] - a[ok]
        rows.append(
            {
                "arm": label,
                "reporter": rep,
                "family": _family(rep),
                "n_pairs": int(ok.sum()),
                "pre_mean": float(np.mean(a[ok])),
                "post_mean": float(np.mean(b[ok])),
                "mean_delta": float(np.mean(d)),
                "max_abs_delta": float(np.max(np.abs(d))),
                "n_exactly_zero": int(np.sum(d == 0.0)),
                "share_exactly_zero": float(np.mean(d == 0.0)),
            }
        )
    return pd.DataFrame(rows)


def cell_deltas(pairs, label):
    keys = ["curiosity", "vanishing", "landscape", "entry"] if label == "T1" else [
        "curiosity", "vanishing", "landscape", "entry", "use_visibility_reward"
    ]
    out = pairs[keys].copy()
    for rep in REPORTERS:
        out[rep] = pairs[rep + "_post"] - pairs[rep + "_pre"]
    g = out.groupby(keys, as_index=False).mean(numeric_only=True)
    g.insert(0, "arm", label)
    g["n_seeds"] = out.groupby(keys).size().to_numpy()
    return g


# ---------------------------------------------------------------------------
# Collapse breakpoint xi* (hinge fit with seed fixed effects)
# ---------------------------------------------------------------------------
def _load_hinge_mod():
    path = os.path.join(ROOT, "scripts", "welfare", "analyse_welfare.py")
    spec = importlib.util.spec_from_file_location("_analyse_welfare", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def xi_star(frames, label):
    """Best hinge over the candidate grid, per (legibility, entry) panel.

    ``x`` is the visibility weight, which the welfare design sets to
    ``1 - initial_curiosity``; ``y`` is terminal-step ideal-merit sum (the
    welfare-loss measure the probe used); seeds enter as fixed effects. The
    estimator is imported, not reimplemented, so pre and post use one home.
    """
    aw = _load_hinge_mod()
    rows = []
    for (leg, entry), sub in frames.groupby(["landscape", "entry"]):
        sub = sub.dropna(subset=["ideal_merit_sum"])
        if sub["curiosity"].nunique() < 3:
            continue
        y = sub["ideal_merit_sum"].to_numpy(dtype=float)
        x = 1.0 - sub["curiosity"].to_numpy(dtype=float)
        ids = sub["seed"].to_numpy()
        st = aw.hinge_stats(y, x, ids)
        rows.append(
            {
                "arm": label,
                "landscape": leg,
                "entry": float(entry),
                "xi_star": float(st["xi"]),
                "f": float(st["f"]) if np.isfinite(st["f"]) else np.nan,
                "p": float(st["p"]) if np.isfinite(st["p"]) else np.nan,
                "slope_per_unit_visibility": float(st["slope"]),
                "n_obs": int(st["n"]),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The companion audit run (scripts/career_flows_noisy/audit_code_delta.py)
# ---------------------------------------------------------------------------
def audit_code_delta():
    """Read the noisy-landscape code-delta audit that was already on disk.

    That script re-ran a few seeds of one cell through the current tree and
    diffed them against the archived run. It raised on a row-count check
    (603 vs 4020 rows: the archive stores step-level rows, the re-run stores
    one row per seed), but its per-reporter deltas were computed and written
    before the raise, and they are the interesting part -- so they are quoted
    here rather than discarded.

    File shape: ``{"checked_at": ..., "cases": [ {cell, seeds_rerun, rows,
    model_max_abs_delta_<reporter>, model_mean_old_<reporter>,
    model_mean_new_<reporter>, ...}, ... ]}``
    """
    path = os.path.join(DATA, "career_flows_noisy_v1", "code_delta.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        raw = json.load(fh)
    cases = raw.get("cases", [])
    rows = []
    for rec in cases:
        cell = rec.get("cell")
        for k, v in rec.items():
            if not k.startswith("model_max_abs_delta_"):
                continue
            rep = k[len("model_max_abs_delta_"):]
            rows.append(
                {
                    "cell": cell,
                    "reporter": rep,
                    "seeds_rerun": rec.get("seeds_rerun"),
                    "rows_compared": rec.get("rows"),
                    "max_abs_delta": float(v),
                    "mean_old": rec.get("model_mean_old_" + rep),
                    "mean_new": rec.get("model_mean_new_" + rep),
                }
            )
    return pd.DataFrame(rows)


def ordering_table(pairs):
    """Does the fix move WHERE the wedge is, or only HOW BIG it is?

    Cell-level comparison of the pre- and post-fix wedge (house convention,
    ``c_local - c_ideal``): sign flips, rank agreement, and the entry-cell
    count that the Phase-1 dossier quotes. Preserved ordering with shifted
    levels means "how fast"; scrambled ordering would mean "where".
    """
    from scipy import stats as sstats

    d = pairs.assign(
        w_pre=pairs["c_local_fin_pre"] - pairs["c_ideal_fin_pre"],
        w_post=pairs["c_local_fin_post"] - pairs["c_ideal_fin_post"],
    )
    keys = ["curiosity", "vanishing", "landscape", "entry"]
    g = d.groupby(keys, as_index=False).agg(
        w_pre=("w_pre", "mean"), w_post=("w_post", "mean")
    )
    ent = g[g["entry"] > 0]
    ent_seeds = d[d["entry"] > 0]
    return pd.DataFrame(
        [
            {"metric": "cells (T1 grid)", "value": int(len(g))},
            {"metric": "sign flips at cell level", "value": int((np.sign(g["w_pre"]) != np.sign(g["w_post"])).sum())},
            {"metric": "Spearman rank agreement (cell wedge, pre vs post)", "value": float(sstats.spearmanr(g["w_pre"], g["w_post"]).statistic)},
            {"metric": "Pearson agreement (cell wedge, pre vs post)", "value": float(np.corrcoef(g["w_pre"], g["w_post"])[0, 1])},
            {"metric": "cells with negative wedge, pre", "value": int((g["w_pre"] < 0).sum())},
            {"metric": "cells with negative wedge, post", "value": int((g["w_post"] < 0).sum())},
            {"metric": "entry cells (entry > 0)", "value": int(len(ent))},
            {"metric": "entry cells with negative wedge, pre", "value": int((ent["w_pre"] < 0).sum())},
            {"metric": "entry cells with negative wedge, post", "value": int((ent["w_post"] < 0).sum())},
            {"metric": "entry seeds with negative wedge, pre", "value": int((ent_seeds["w_pre"] < 0).sum())},
            {"metric": "entry seeds with negative wedge, post", "value": int((ent_seeds["w_post"] < 0).sum())},
            {"metric": "entry seeds total", "value": int(len(ent_seeds))},
            {"metric": "mean wedge level, pre", "value": float(g["w_pre"].mean())},
            {"metric": "mean wedge level, post", "value": float(g["w_post"].mean())},
        ]
    )


def main():
    os.makedirs(TABLES, exist_ok=True)
    summary = {}

    # ---- T1 arm --------------------------------------------------------
    t1 = pair_t1()
    t1_sum = channel_summary(t1, "T1")
    t1_sum.to_csv(os.path.join(TABLES, "t1_channel_summary.csv"), index=False)
    ordering = ordering_table(t1)
    ordering.to_csv(os.path.join(TABLES, "t1_ordering.csv"), index=False)
    summary["t1_ordering"] = {r.metric: r.value for r in ordering.itertuples()}
    cell_deltas(t1, "T1").to_csv(os.path.join(TABLES, "t1_cell_deltas.csv"), index=False)
    t1.assign(
        wedge_fin_delta=(t1["c_local_fin_post"] - t1["c_ideal_fin_post"])
        - (t1["c_local_fin_pre"] - t1["c_ideal_fin_pre"]),
        wedge_tm_delta=(t1["c_local_tm_post"] - t1["c_ideal_tm_post"])
        - (t1["c_local_tm_pre"] - t1["c_ideal_tm_pre"]),
    ).to_csv(os.path.join(TABLES, "t1_paired_runs.csv"), index=False)

    # the load-bearing claim of the whole round: the non-ideal channels are
    # untouched by the fix. Exact zeros, not "small".
    for rep in ("c_local_fin", "c_local_tm", "top10_fin", "harvested_fin"):
        row = t1_sum[t1_sum["reporter"] == rep].iloc[0]
        _require(
            row["max_abs_delta"] == 0.0 and row["n_exactly_zero"] == row["n_pairs"],
            "%s is not bit-identical across the fix: max_abs_delta=%r, "
            "exact=%d/%d" % (
                rep, row["max_abs_delta"], row["n_exactly_zero"], row["n_pairs"]
            ),
        )
    summary["t1"] = {
        "pairs": int(len(t1)),
        "exact_zero_channels": ["c_local_fin", "c_local_tm", "top10_fin", "harvested_fin"],
        "channels": t1_sum.set_index("reporter")[
            ["pre_mean", "post_mean", "mean_delta", "max_abs_delta", "n_exactly_zero"]
        ].to_dict(orient="index"),
    }

    # ---- W_v05 arm -----------------------------------------------------
    w_pre = load_w_pre()
    w_post = load_w_post()
    if w_post is not None:
        w = pair_w(w_pre, w_post)
        w_sum = channel_summary(w, "W_v05")
        w_sum.to_csv(os.path.join(TABLES, "w_v05_channel_summary.csv"), index=False)
        cell_deltas(w, "W_v05").to_csv(os.path.join(TABLES, "w_v05_cell_deltas.csv"), index=False)
        w.assign(
            wedge_fin_delta=(w["c_local_fin_post"] - w["c_ideal_fin_post"])
            - (w["c_local_fin_pre"] - w["c_ideal_fin_pre"]),
        ).to_csv(os.path.join(TABLES, "w_v05_paired_runs.csv"), index=False)
        for rep in ("c_local_fin", "c_local_tm", "top10_fin", "harvested_fin"):
            row = w_sum[w_sum["reporter"] == rep].iloc[0]
            _require(
                row["max_abs_delta"] == 0.0,
                "W_v05 %s not bit-identical: max_abs_delta=%r"
                % (rep, row["max_abs_delta"]),
            )
        summary["w_v05"] = {
            "pairs": int(len(w)),
            "channels": w_sum.set_index("reporter")[
                ["pre_mean", "post_mean", "mean_delta", "max_abs_delta", "n_exactly_zero"]
            ].to_dict(orient="index"),
            "ideal_merit_sum": {
                "pre_mean": float(w["ideal_merit_sum_pre"].mean()),
                "post_mean": float(w["ideal_merit_sum_post"].mean()),
                "mean_delta": float((w["ideal_merit_sum_post"] - w["ideal_merit_sum_pre"]).mean()),
                "max_abs_delta": float((w["ideal_merit_sum_post"] - w["ideal_merit_sum_pre"]).abs().max()),
            },
        }

        # ---- breakpoint, pre vs post -----------------------------------
        pre_panel = w_pre.rename(columns={})
        post_panel = w_post.rename(columns={"agent_generation_time": "entry"})
        xi = pd.concat(
            [xi_star(pre_panel, "pre-fix archive"), xi_star(post_panel, "W_v05 re-measure")],
            ignore_index=True,
        )
        xi.to_csv(os.path.join(TABLES, "xi_star.csv"), index=False)
        summary["xi_star"] = xi.to_dict(orient="records")
    else:
        print("W_v05 arm not present yet -- skipping the welfare comparison and xi*")

    # ---- companion audit ----------------------------------------------
    aud = audit_code_delta()
    if aud is not None and len(aud):
        aud.to_csv(os.path.join(TABLES, "audit_code_delta.csv"), index=False)
        zero = aud[aud["max_abs_delta"] == 0.0]["reporter"].nunique()
        moved = aud[aud["max_abs_delta"] > 0.0]
        summary["audit_code_delta"] = {
            "reporters_at_exactly_zero": int(zero),
            "reporters_moved": sorted(moved["reporter"].unique().tolist()),
            "max_move": float(aud["max_abs_delta"].max()),
            "note": "max_abs_delta per reporter over the audit's re-run seeds",
        }

    with open(os.path.join(TABLES, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)

    # ---- console ------------------------------------------------------
    for label, frame in (("T1", t1_sum),):
        print("\n== %s paired deltas (n pairs per reporter) ==" % label)
        print(frame[["reporter", "family", "n_pairs", "pre_mean", "post_mean",
                     "mean_delta", "max_abs_delta", "n_exactly_zero"]].to_string(index=False))
    if w_post is not None:
        print("\n== W_v05 paired deltas ==")
        print(w_sum[["reporter", "n_pairs", "pre_mean", "post_mean",
                     "mean_delta", "max_abs_delta", "n_exactly_zero"]].to_string(index=False))
        print("\n== xi* (hinge with seed fixed effects) ==")
        print(xi.to_string(index=False))
    print("\ntables written to %s" % TABLES)


if __name__ == "__main__":
    main()
