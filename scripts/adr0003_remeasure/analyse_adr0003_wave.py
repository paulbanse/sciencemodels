"""Directional robustness ledger for the ADR-0003 re-run wave.

Reads the new ``data/*_adr0003/cells/`` archives and the pre-ADR archives; it
does not run the model.  The ledger compares the *sign or ordering* of each
archive's headline contrast, never the raw level: the clock deliberately moves
gamma, g and the exit rule, so level equality is neither expected nor tested.

Outputs (under ``data/adr0003_remeasure/``):
    summary_by_cell.csv   new-run means by arm/cell
    summary_by_arm.csv    new-run means by arm and clock state
    ledger.csv            pre vs post directional contrasts
    analysis.json         compact headline values
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT = REPO / "data" / "adr0003_remeasure"
ARMS = (
    "sensitivity", "welfare", "career_flows", "prestige",
    "turnover", "interventions", "path_b", "fig1_xi",
)
G_VALUES = (0.125, 0.25, 0.5)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def load_new() -> pd.DataFrame:
    frames = []
    for arm in ARMS:
        files = sorted(glob.glob(str(REPO / "data" / f"{arm}_adr0003" / "cells" / "*.csv")))
        _require(files, f"no new cells for {arm}")
        d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        _require(d["agent_seed"].nunique() == 20, f"{arm}: seed grid")
        _require(len(d) == len(files) * 20, f"{arm}: row count")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    _require(len(df) == 6380, f"expected 6380 new rows, got {len(df)}")
    df["subarm"] = df["cell_tag"].str.split("_").str[0]
    return df


def summary_tables(df: pd.DataFrame):
    keep = [
        "arm", "cell_tag", "subarm", "tier", "g", "exit_mode", "curiosity", "xi",
        "landscape", "horizon", "stock", "use_visibility_reward", "leg",
        "spawn", "p_reloc", "refill_mode", "dyadic_move_pull",
        "dyadic_lambda", "formulation", "credit_sharing",
    ]
    metrics = [
        "c_ideal_fin", "c_ideal_tm", "c_ideal_late", "c_local_fin",
        "c_local_tm", "c_local_late", "wedge_fin", "wedge_tm",
        "top10_fin", "top10_tm", "harvested_fin", "explored_fin",
        "explored_tm", "ideal_loss_cum", "ideal_merit_sum", "n_final",
        "mean_age_final", "mean_tenure_final", "gap_recent_final",
        "gap_recent_mean_late", "team_tm", "links_tm", "repeated_tm",
        "career_percap_harvest_final", "career_harvest_flow_mean_last50",
        "career_gap_ratio_mean_late",
    ]
    cols = [c for c in keep + metrics if c in df.columns]
    cell = (
        df[cols]
        .groupby(keep, dropna=False)
        .agg({m: ["mean", "std", "size"] for m in metrics if m in df.columns})
        .reset_index()
    )
    cell.columns = [
        "_".join([str(x) for x in c if x != ""]).rstrip("_")
        for c in cell.columns
    ]
    arm = (
        df.groupby("arm")
        .agg(
            n_cells=("cell_tag", "nunique"),
            n_runs=("agent_seed", "size"),
            c_ideal_tm=("c_ideal_tm", "mean"),
            c_ideal_fin=("c_ideal_fin", "mean"),
            harvested_fin=("harvested_fin", "mean"),
            explored_fin=("explored_fin", "mean"),
            ideal_loss_cum=("ideal_loss_cum", "mean"),
            top10_fin=("top10_fin", "mean"),
            n_final=("n_final", "mean"),
        )
        .reset_index()
    )
    return cell, arm


def _old_t1() -> dict:
    d = pd.read_csv(REPO / "data/remeasure_v1/runs.csv")
    d = d[d.origin == "T1"]
    m0 = float(d[d.agent_generation_time == 0.0].c_ideal_tm.mean())
    m1 = float(d[d.agent_generation_time == 0.1].c_ideal_tm.mean())
    return {"g0": m0, "g1": m1, "delta": m1 - m0,
            "source": "data/remeasure_v1/runs.csv (origin=T1)"}


def _old_welfare() -> dict:
    d = pd.read_csv(REPO / "data/remeasure_v1/w_v05_runs.csv")
    d["xi"] = 1.0 - d.curiosity
    return {
        "xi_loss_corr": float(d[["xi", "ideal_loss_cum"]].corr().iloc[0, 1]),
        "xi_harvest_corr": float(d[["xi", "harvested_fin"]].corr().iloc[0, 1]),
        "g0_loss": float(d[d.agent_generation_time == 0.0].ideal_loss_cum.mean()),
        "g1_loss": float(d[d.agent_generation_time == 0.1].ideal_loss_cum.mean()),
        "source": "data/remeasure_v1/w_v05_runs.csv",
    }


def _old_career() -> dict:
    d = pd.read_csv(REPO / "data/career_flows_v1/careerB_analysis.csv")
    lo = d[d.agent_generation_time == 0.05]
    hi = d[d.agent_generation_time == 0.20]
    return {
        "percap_lo": float(lo.percap_harvest_final.mean()),
        "percap_hi": float(hi.percap_harvest_final.mean()),
        "percap_delta": float(hi.percap_harvest_final.mean()
                              - lo.percap_harvest_final.mean()),
        "top10_delta": float(hi.top10_final.mean() - lo.top10_final.mean()),
        "source": "data/career_flows_v1/careerB_analysis.csv",
    }


def _old_prestige() -> dict:
    d = pd.read_csv(REPO / "data/prestige_stock_v2/summary_cell.csv")
    return {
        "wedge_tm": {k: float(v) for k, v in
                     d.groupby("stock").wedge_tm_mean.mean().items()},
        "wedge_fin": {k: float(v) for k, v in
                      d.groupby("stock").wedge_fin_mean.mean().items()},
        "source": "data/prestige_stock_v2/summary_cell.csv",
    }


def _old_turnover() -> dict:
    d = pd.read_csv(REPO / "data/turnover_v2/runs.csv")
    d = d[d.entry == 0.10]
    g = d.groupby("exit_mode").c_ideal_late_mean.mean()
    return {
        "fixed": float(g["fixed_span"]),
        "staggered": float(g["age_hazard_staggered"]),
        "delta": float(g["fixed_span"] - g["age_hazard_staggered"]),
        "source": "data/turnover_v2/runs.csv (entry=0.10)",
    }


def _old_interventions() -> dict:
    def read(arm: str) -> pd.DataFrame:
        fs = sorted(glob.glob(str(REPO / "data/interventions_v2/cells"
                                / f"cell{arm}_*.csv")))
        _require(fs, f"missing interventions_v2 arm {arm}")
        return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)

    c = read("C")
    c = c[(c.initial_curiosity == 0.5) & (c.agent_generation_time == 0.1)]
    ref = {k: float(v) for k, v in
           c.groupby("refill_mode").explored_final.mean().items()}
    a = read("A")
    a = a[(a.initial_curiosity == 0.5) & (a.agent_generation_time == 0.1)]
    spawn = {k: float(v) for k, v in a.groupby("r_spawn").explored_final.mean().items()}
    b = read("B")
    b = b[(b.initial_curiosity == 0.5) & (b.agent_generation_time == 0.1)]
    reloc = {float(k): float(v) for k, v in
             b.groupby("p_reloc").explored_final.mean().items()}
    return {
        "refill": ref,
        "spawn": spawn,
        "reloc": reloc,
        "refill_delta": ref["occupied"] - ref["off"],
        "spawn_delta": spawn["random"] - spawn["exact"],
        "reloc_delta": reloc[0.02] - reloc[0.0],
        "source": "data/interventions_v2/cells/cell{A,B,C}_*.csv",
    }


def _old_path_b() -> dict:
    fs = sorted(glob.glob(str(REPO / "data/path_b_v2/cells/*.csv")))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d[(d.dyadic_lambda == 1.0)]
    p05 = float(d[d.dyadic_move_pull == 0.5].team_tm.mean())
    p10 = float(d[d.dyadic_move_pull == 1.0].team_tm.mean())
    return {
        "team_p05": p05, "team_p10": p10, "team_delta": p10 - p05,
        "source": "data/path_b_v2/cells/*.csv (lambda=1)",
    }


def _old_fig1() -> dict:
    d = pd.read_csv(REPO / "data/remeasure_v1/w_v05_runs.csv")
    d = d[(d.use_visibility_reward) & (d.agent_generation_time == 0.1)]
    d["xi"] = 1.0 - d.curiosity
    return {
        "xi_harvest_corr": float(d[["xi", "harvested_fin"]].corr().iloc[0, 1]),
        "source": "data/remeasure_v1/w_v05_runs.csv (visibility=True, g=0.1)",
    }


def _sign(x: float) -> int:
    return int(np.sign(x))


def _verdict(pre: float, post: float) -> str:
    if not np.isfinite(pre) or not np.isfinite(post):
        return "not comparable"
    return "robust" if _sign(pre) == _sign(post) else "flip"


def ledger(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(archive, contrast, pre_value, post_value, verdict,
            pre_source, post_source, note=""):
        rows.append(dict(
            archive=archive, contrast=contrast,
            pre_value=float(pre_value), post_value=float(post_value),
            pre_direction=_sign(pre_value), post_direction=_sign(post_value),
            verdict=verdict, pre_source=pre_source, post_source=post_source,
            note=note,
        ))

    # 1 T1: entry gradient of time-mean ideal alignment
    old = _old_t1()
    s = df[df.arm == "sensitivity"]
    g = s.groupby("g").c_ideal_tm.mean()
    post = float(g.loc[0.5] - g.loc[0.125])
    add("T1 / sensitivity", "c_ideal_tm: high g minus low g",
        old["delta"], post, _verdict(old["delta"], post),
        old["source"], "data/sensitivity_adr0003/cells/*.csv",
        f"pre g=0.1 minus g=0; post g=0.5 minus g=0.125; pre {old['g0']:.4f}->{old['g1']:.4f}")

    # 2 W_v05: visibility gradient of loss and harvest
    old = _old_welfare()
    w = df[df.arm == "welfare"].copy()
    w["xi"] = 1.0 - w.curiosity
    add("W_v05 / welfare", "corr(xi, ideal_loss_cum)",
        old["xi_loss_corr"], float(w[["xi", "ideal_loss_cum"]].corr().iloc[0, 1]),
        _verdict(old["xi_loss_corr"],
                 float(w[["xi", "ideal_loss_cum"]].corr().iloc[0, 1])),
        old["source"], "data/welfare_adr0003/cells/*.csv")
    add("W_v05 / welfare", "corr(xi, harvested_fin)",
        old["xi_harvest_corr"], float(w[["xi", "harvested_fin"]].corr().iloc[0, 1]),
        _verdict(old["xi_harvest_corr"],
                 float(w[["xi", "harvested_fin"]].corr().iloc[0, 1])),
        old["source"], "data/welfare_adr0003/cells/*.csv")
    old_gap = old["g1_loss"] - old["g0_loss"]
    wg = w.groupby("g").ideal_loss_cum.mean()
    post_gap = float(wg.loc[0.5] - wg.loc[0.125])
    add("W_v05 / welfare", "ideal_loss_cum: high g minus low g",
        old_gap, post_gap, _verdict(old_gap, post_gap),
        old["source"], "data/welfare_adr0003/cells/*.csv",
        "pre g=0.1 minus g=0; post g=0.5 minus g=0.125")

    # 3 career flows: entry gradient of per-capita harvest, split by exit mode
    old = _old_career()
    c = df[df.arm == "career_flows"]
    for mode in ("fixed_span", "age_hazard_staggered"):
        sub = c[c.exit_mode == mode].groupby("g").career_percap_harvest_final.mean()
        post = float(sub.loc[0.5] - sub.loc[0.125])
        add("career_flows", f"per-capita harvest: high g minus low g ({mode})",
            old["percap_delta"], post, _verdict(old["percap_delta"], post),
            old["source"], "data/career_flows_adr0003/cells/*.csv",
            "pre is g=0.2 minus g=0.05; post is g=0.5 minus g=0.125")

    # 4 prestige stock: time-mean wedge ordering and level sign
    old = _old_prestige()
    p = df[df.arm == "prestige"].groupby("stock").wedge_tm.mean()
    pre_order = sorted(old["wedge_tm"], key=old["wedge_tm"].get, reverse=True)
    post_order = sorted(p.to_dict(), key=p.to_dict().get, reverse=True)
    add("prestige_stock", "time-mean wedge ordering",
        float(old["wedge_tm"][pre_order[0]]),
        float(p.loc[post_order[0]]),
        "ordering robust" if pre_order == post_order else "ordering flip",
        old["source"], "data/prestige_adr0003/cells/*.csv",
        f"pre order {pre_order}; post order {post_order}; "
        f"pre signs {[round(old['wedge_tm'][k], 4) for k in pre_order]}; "
        f"post signs {[round(float(p.loc[k]), 4) for k in post_order]}")
    add("prestige_stock", "time-mean wedge level: prestige minus visibility",
        float(old["wedge_tm"]["prestige"] - old["wedge_tm"]["prestige_visibility"]),
        float(p.loc["prestige"] - p.loc["prestige_visibility"]),
        _verdict(float(old["wedge_tm"]["prestige"] - old["wedge_tm"]["prestige_visibility"]),
                 float(p.loc["prestige"] - p.loc["prestige_visibility"])),
        old["source"], "data/prestige_adr0003/cells/*.csv")

    # 5 turnover: fixed_span minus staggered late alignment
    old = _old_turnover()
    t = df[df.arm == "turnover"].groupby("exit_mode").c_ideal_late.mean()
    post = float(t.loc["fixed_span"] - t.loc["age_hazard_staggered"])
    add("turnover", "c_ideal_late: fixed_span minus staggered",
        old["delta"], post, _verdict(old["delta"], post),
        old["source"], "data/turnover_adr0003/cells/*.csv",
        f"pre fixed {old['fixed']:.4f}, staggered {old['staggered']:.4f}")

    # 6 interventions: refill / spawn / sabbatical signs.  Each sub-arm has
    # its own baseline row, so pooling intA/intB/intC would compare different
    # interventions; filter by the cell-tag prefix.
    old = _old_interventions()
    i = df[df.arm == "interventions"]
    ic = i[i.subarm == "intC"]
    ref = ic[(ic.g == 0.25) & (ic.curiosity == 0.5)]
    ref = ref.groupby("refill_mode").explored_fin.mean()
    off = float(ref.loc["off"])
    post_refill = float(ref.loc["occupied"] - off)
    add("interventions", "explored_fin: occupied minus off",
        old["refill_delta"], post_refill, _verdict(old["refill_delta"], post_refill),
        old["source"], "data/interventions_adr0003/cells/*.csv")
    old_front = old["refill"]["frontier"] - old["refill"]["off"]
    post_front = float(ref.loc["frontier"] - off)
    add("interventions", "explored_fin: frontier minus off",
        old_front, post_front, _verdict(old_front, post_front),
        old["source"], "data/interventions_adr0003/cells/*.csv")
    ia = i[i.subarm == "intA"]
    sp = ia[(ia.g == 0.25) & (ia.curiosity == 0.5)].groupby("spawn").explored_fin.mean()
    post_spawn = float(sp.loc["random"] - sp.loc["exact"])
    add("interventions", "explored_fin: random spawn minus exact",
        old["spawn_delta"], post_spawn, _verdict(old["spawn_delta"], post_spawn),
        old["source"], "data/interventions_adr0003/cells/*.csv")
    ib = i[i.subarm == "intB"]
    rel = ib[(ib.g == 0.25) & (ib.curiosity == 0.5)].groupby("p_reloc").explored_fin.mean()
    post_rel = float(rel.loc[0.02] - rel.loc[0.0])
    add("interventions", "explored_fin: relocation 0.02 minus 0",
        old["reloc_delta"], post_rel, _verdict(old["reloc_delta"], post_rel),
        old["source"], "data/interventions_adr0003/cells/*.csv")

    # 7 Path B: pull effect on team size at lambda=1
    old = _old_path_b()
    pb = df[(df.arm == "path_b") & (df.dyadic_lambda == 1.0)]
    p0 = float(pb[pb.dyadic_move_pull == 0.0].team_tm.mean())
    p1 = float(pb[pb.dyadic_move_pull == 1.0].team_tm.mean())
    add("path_b", "team size: pull=1 minus pull=0 (lambda=1)",
        old["team_delta"], p1 - p0, _verdict(old["team_delta"], p1 - p0),
        old["source"], "data/path_b_adr0003/cells/*.csv",
        "pre uses pull=1 minus pull=0.5 (the archived grid has no pull=0); "
        f"pre {old['team_p05']:.3f}->{old['team_p10']:.3f}")

    # 8 Figure 1 dense xi: harvest gradient by tier
    old = _old_fig1()
    f = df[df.arm == "fig1_xi"].copy()
    for tier in ("M0", "M1_fixed", "M1_staggered", "M2_l1", "M2_l0"):
        sub = f[f.tier == tier]
        post = float(sub[["xi", "harvested_fin"]].corr().iloc[0, 1])
        add("fig1_xi", f"corr(xi, harvested_fin) [{tier}]",
            old["xi_harvest_corr"], post, _verdict(old["xi_harvest_corr"], post),
            old["source"], "data/fig1_xi_adr0003/cells/*.csv",
            "pre reference is the visibility-on W_v05 g=0.1 leg")

    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_new()
    cell, arm = summary_tables(df)
    cell.to_csv(OUT / "summary_by_cell.csv", index=False)
    arm.to_csv(OUT / "summary_by_arm.csv", index=False)
    led = ledger(df)
    led.to_csv(OUT / "ledger.csv", index=False)

    payload = {
        "new_rows": int(len(df)),
        "new_cells": int(df.cell_tag.nunique()),
        "arms": {a: int((df.arm == a).sum()) for a in ARMS},
        "ledger_verdicts": {k: int(v) for k, v in
                            led.verdict.value_counts().items()},
        "old": {
            "T1": _old_t1(),
            "W_v05": _old_welfare(),
            "career_flows": _old_career(),
            "prestige_stock": _old_prestige(),
            "turnover": _old_turnover(),
            "interventions": _old_interventions(),
            "path_b": _old_path_b(),
            "fig1_xi": _old_fig1(),
        },
    }
    (OUT / "analysis.json").write_text(json.dumps(payload, indent=1))
    print(led.to_string(index=False))
    print(json.dumps({"new_rows": payload["new_rows"],
                      "verdicts": payload["ledger_verdicts"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
