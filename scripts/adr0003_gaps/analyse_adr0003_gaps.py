"""Analysis for the ADR-0003 gap arms (Prompt F2): lottery, career noisy
leg, and the n = 40 M2 lift check.

Reads the three archives written by ``run_adr0003_gaps.py`` plus the D6
reference archives (bit-identity canaries and the old-clock n = 20
reference) and writes the canonical tables for
``output/reports/adr0003_gaps_v1.md``:

    summary_lottery.csv        alpha x p_reloc cell means / SEs
    contrasts_lottery.csv      paired (alpha vs 1.0) differences
    summary_career_noisy.csv   g x exit cell means / SEs
    gradients_career_noisy.csv high-g minus low-g paired gradients
    summary_m2_n40.csv         cell means / SEs for the M2 panel
    lift_tests.csv             the lambda = 1 lift over M1, paired + Welch
    chain_m2_n40.csv           D3-style chain shares at n = 40
    crosscheck.csv             bit-identity canaries against D6 archives
    analysis.json              headline verdicts

Statistics: paired contrasts use the common agent_seed grid (t interval on
the paired differences, 95%); the Welch interval is the unpaired
robustness twin. The chain-share intervals are seed-paired bootstrap
percentile CIs (10,000 draws, RNG seed 20261126), the D3 convention.

NOT CALIBRATED: model-internal only (owner ruling 2026-11-13).

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
        scripts/adr0003_gaps/analyse_adr0003_gaps.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
OUT = DATA / "adr0003_gaps"
BOOT_N = 10_000
BOOT_SEED = 20261126

LOT_DIR = DATA / "interventions_lottery_adr0003" / "cells"
CAR_DIR = DATA / "career_flows_noisy_adr0003" / "cells"
M2_DIR = DATA / "path_b_m2_n40_adr0003" / "cells"
PB_D6 = DATA / "path_b_adr0003" / "cells"
INT_D6 = DATA / "interventions_adr0003" / "cells"
PB_OLD = DATA / "path_b_postbug"

Z95 = stats.t.ppf(0.975, 19)


def load_dir(path: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(path / "*.csv")))
    assert files, f"no CSVs under {path}"
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df


def mean_se(x: pd.Series) -> tuple[float, float]:
    v = x.dropna().values.astype(float)
    if len(v) < 2:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def paired(a: pd.Series, b: pd.Series) -> dict:
    """Paired difference a - b on the common seed grid; t interval."""
    d = (a.values - b.values).astype(float)
    d = d[np.isfinite(d)]
    n = len(d)
    m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))
    tcrit = stats.t.ppf(0.975, n - 1)
    tt = stats.ttest_rel(a.values, b.values)
    return dict(n=int(n), diff=m, se=se,
                ci_lo=m - tcrit * se, ci_hi=m + tcrit * se,
                t=float(tt.statistic), p=float(tt.pvalue))


def welch(a: pd.Series, b: pd.Series) -> dict:
    """Unpaired Welch twin of ``paired``."""
    xa, xb = a.values.astype(float), b.values.astype(float)
    xa, xb = xa[np.isfinite(xa)], xb[np.isfinite(xb)]
    va, vb = xa.var(ddof=1), xb.var(ddof=1)
    se = float(np.sqrt(va / len(xa) + vb / len(xb)))
    df = (va / len(xa) + vb / len(xb)) ** 2 / (
        (va / len(xa)) ** 2 / (len(xa) - 1) + (vb / len(xb)) ** 2 / (len(xb) - 1)
    )
    m = float(xa.mean() - xb.mean())
    tcrit = stats.t.ppf(0.975, df)
    tt = stats.ttest_ind(xa, xb, equal_var=False)
    return dict(n=int(len(xa) + len(xb)), diff=m, se=se, df=float(df),
                ci_lo=m - tcrit * se, ci_hi=m + tcrit * se,
                t=float(tt.statistic), p=float(tt.pvalue))


def cell_series(df: pd.DataFrame, **filters) -> pd.Series:
    """One cell's outcome series ordered by seed (one value per seed)."""
    s = df
    for k, v in filters.items():
        s = s[s[k] == v]
    return s.sort_values("seed").reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lot = load_dir(LOT_DIR)
    car = load_dir(CAR_DIR)
    m2 = load_dir(M2_DIR)
    m2d = m2[m2.cell_tag != "m2n40_kind-m1base"].reset_index(drop=True)

    # ------------------------------------------------------------------
    # (a) Lottery arm
    # ------------------------------------------------------------------
    lot_rows = []
    for (alpha, preloc), g in lot.groupby(["alpha", "p_reloc"]):
        r = dict(alpha=float(alpha), p_reloc=float(preloc), n=len(g))
        for col in ("top10_fin", "top10_tm", "ideal_loss_cum", "c_ideal_tm",
                    "c_ideal_fin", "harvested_fin", "explored_fin"):
            m, se = mean_se(g[col])
            r[col], r[col + "_se"] = m, se
        lot_rows.append(r)
    summary_lot = pd.DataFrame(lot_rows).sort_values(["p_reloc", "alpha"],
                                                     ascending=[True, False])
    summary_lot.to_csv(OUT / "summary_lottery.csv", index=False)

    con_rows = []
    for preloc in (0.0, 0.02):
        base = cell_series(lot, alpha=1.0, p_reloc=preloc)
        for alpha in (0.5, 0.0):
            arm = cell_series(lot, alpha=alpha, p_reloc=preloc)
            assert len(arm) == len(base) == 20
            for col in ("top10_fin", "ideal_loss_cum", "c_ideal_tm",
                        "harvested_fin"):
                d = paired(arm[col], base[col])
                con_rows.append(dict(scope=f"p_reloc={preloc:g}",
                                     alpha=alpha, outcome=col, **d))
    # pooled over the two relocation regimes (40 paired units per alpha)
    for alpha in (0.5, 0.0):
        for col in ("top10_fin", "ideal_loss_cum", "c_ideal_tm", "harvested_fin"):
            a = pd.concat([cell_series(lot, alpha=alpha, p_reloc=p)[col]
                           for p in (0.0, 0.02)], ignore_index=True)
            b = pd.concat([cell_series(lot, alpha=1.0, p_reloc=p)[col]
                           for p in (0.0, 0.02)], ignore_index=True)
            d = paired(a, b)
            con_rows.append(dict(scope="pooled", alpha=alpha, outcome=col, **d))
    contrasts_lot = pd.DataFrame(con_rows)
    contrasts_lot.to_csv(OUT / "contrasts_lottery.csv", index=False)

    # ------------------------------------------------------------------
    # (b) career_flows noisypr5 leg
    # ------------------------------------------------------------------
    car_rows = []
    for (g_val, exit_mode), g in car.groupby(["g", "exit_mode"]):
        r = dict(g=float(g_val), exit_mode=exit_mode, n=len(g))
        for col in ("career_percap_harvest_final", "career_gap_final",
                    "career_gap_ratio_mean_late", "career_harvest_flow_mean_last50",
                    "c_ideal_tm", "harvested_fin", "top10_fin"):
            m, se = mean_se(g[col])
            r[col], r[col + "_se"] = m, se
        car_rows.append(r)
    summary_car = pd.DataFrame(car_rows).sort_values(["exit_mode", "g"])
    summary_car.to_csv(OUT / "summary_career_noisy.csv", index=False)

    grad_rows = []
    for exit_mode in ("fixed_span", "age_hazard_staggered"):
        for col in ("career_percap_harvest_final", "career_gap_ratio_mean_late",
                    "c_ideal_tm", "harvested_fin"):
            hi = cell_series(car, g=0.5, exit_mode=exit_mode)[col]
            lo = cell_series(car, g=0.125, exit_mode=exit_mode)[col]
            d = paired(hi, lo)
            grad_rows.append(dict(exit_mode=exit_mode, contrast="high g minus low g",
                                  outcome=col, **d))
    gradients_car = pd.DataFrame(grad_rows)
    gradients_car.to_csv(OUT / "gradients_career_noisy.csv", index=False)

    # ------------------------------------------------------------------
    # (c) M2 n = 40 panel
    # ------------------------------------------------------------------
    base = cell_series(m2, cell_tag="m2n40_kind-m1base")
    assert len(base) == 40

    m2_rows = []
    for (lam, pull, credit), g in m2[m2.cell_tag != "m2n40_kind-m1base"].groupby(
            ["dyadic_lambda", "dyadic_move_pull", "credit_sharing"]):
        r = dict(lam=float(lam), pull=float(pull), credit=credit, n=len(g))
        for col in ("c_ideal_tm", "c_ideal_fin", "links_tm", "team_tm",
                    "top10_late", "harvested_fin"):
            m, se = mean_se(g[col])
            r[col], r[col + "_se"] = m, se
        m2_rows.append(r)
    m1row = dict(lam=float("nan"), pull=float("nan"), credit="M1 (dyadic off)",
                 n=len(base))
    for col in ("c_ideal_tm", "c_ideal_fin", "links_tm", "team_tm",
                "top10_late", "harvested_fin"):
        m, se = mean_se(base[col])
        m1row[col], m1row[col + "_se"] = m, se
    summary_m2 = pd.DataFrame(m2_rows + [m1row]).sort_values(
        ["credit", "lam", "pull"])
    summary_m2.to_csv(OUT / "summary_m2_n40.csv", index=False)

    lift_rows = []
    for lam in (0.0, 1.0):
        for pull in (0.0, 1.0):
            for credit in ("structural", "equal_split"):
                arm = cell_series(m2d, dyadic_lambda=lam, dyadic_move_pull=pull,
                                  credit_sharing=credit)
                assert len(arm) == 40
                for col in ("c_ideal_tm", "c_ideal_fin"):
                    d = paired(arm[col], base[col])
                    w = welch(arm[col], base[col])
                    d20 = paired(arm[col].iloc[:20].reset_index(drop=True),
                                 base[col].iloc[:20].reset_index(drop=True))
                    lift_rows.append(dict(
                        lam=lam, pull=pull, credit=credit, outcome=col,
                        n_paired=d["n"], lift=d["diff"], se=d["se"],
                        ci_lo=d["ci_lo"], ci_hi=d["ci_hi"], t=d["t"], p=d["p"],
                        lift20=d20["diff"], ci20_lo=d20["ci_lo"],
                        ci20_hi=d20["ci_hi"], p20=d20["p"],
                        welch_df=w["df"], w_ci_lo=w["ci_lo"], w_ci_hi=w["ci_hi"],
                        w_p=w["p"],
                        clears=(d["ci_lo"] > 0) or (d["ci_hi"] < 0)))
    lift = pd.DataFrame(lift_rows)
    lift.to_csv(OUT / "lift_tests.csv", index=False)

    # ------------------------------------------------------------------
    # (c') D3-style chain at n = 40 (same definitions as the D3 report)
    # ------------------------------------------------------------------
    def node(lam: float, pull: float, credit: str) -> pd.Series:
        return cell_series(m2d, dyadic_lambda=lam, dyadic_move_pull=pull,
                           credit_sharing=credit)["c_ideal_tm"].values

    chain_rows = []
    rng = np.random.default_rng(BOOT_SEED)
    for lam in (0.0, 1.0):
        a0 = base["c_ideal_tm"].values
        a1 = node(lam, 0.0, "structural")
        a2 = node(lam, 1.0, "structural")
        a3 = node(lam, 1.0, "equal_split")
        idx = rng.integers(0, 40, size=(BOOT_N, 40))
        b0, b1, b2, b3 = (x[idx].mean(axis=1) for x in (a0, a1, a2, a3))
        gap = a0.mean() - a3.mean()
        bt = b0 - b3
        shares = {"link": 100 * (a0 - a1).mean() / gap,
                  "move": 100 * (a1 - a2).mean() / gap,
                  "credit": 100 * (a2 - a3).mean() / gap}
        bshares = {"link": 100 * (b0 - b1) / bt,
                   "move": 100 * (b1 - b2) / bt,
                   "credit": 100 * (b2 - b3) / bt}
        row = dict(lam=lam, n=40, L0=a0.mean(), L1=a1.mean(), L2=a2.mean(),
                   L3=a3.mean(), gap=gap,
                   gap_lo=float(np.percentile(bt, 2.5)),
                   gap_hi=float(np.percentile(bt, 97.5)))
        for k in ("link", "move", "credit"):
            row[k] = float(shares[k])
            row[k + "_lo"], row[k + "_hi"] = np.percentile(bshares[k], [2.5, 97.5])
        row["sum_check"] = sum(shares.values())
        chain_rows.append(row)
    chain = pd.DataFrame(chain_rows)
    chain.to_csv(OUT / "chain_m2_n40.csv", index=False)

    # ------------------------------------------------------------------
    # Cross-archive canaries (bit identity on shared seeds)
    # ------------------------------------------------------------------
    checks = []

    def numdiff(a: pd.DataFrame, b: pd.DataFrame) -> float:
        worst = 0.0
        for c in [c for c in a.columns if c in b.columns]:
            if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
                d = np.abs(a[c].values.astype(float) - b[c].values.astype(float))
                d = d[np.isfinite(d)]
                if len(d):
                    worst = max(worst, float(d.max()))
        return worst

    for lam in (0.0, 1.0):
        for pull in (0.0, 1.0):
            mine = pd.read_csv(M2_DIR / f"m2n40_credit-equal_lam-{lam:g}_pull-{pull:g}.csv")
            d6 = pd.read_csv(PB_D6 / f"pb_g-0p25_lam-{lam:g}_pull-{pull:g}.csv")
            mine = mine.sort_values("seed").head(20).reset_index(drop=True)
            d6 = d6.sort_values("seed").reset_index(drop=True)
            checks.append(dict(
                check=f"m2_n40 seeds 0-19 vs D6 path_b_adr0003 lam={lam:g} pull={pull:g}",
                n_rows=20, max_abs_diff=numdiff(mine, d6)))
    for preloc_tag, preloc_val in (("0", 0.0), ("0p02", 0.02)):
        mine = pd.read_csv(LOT_DIR / f"lot_alpha-1_g-0p25_reloc-{preloc_tag}.csv")
        d6 = pd.read_csv(INT_D6 / f"intB_c-0p5_g-0p25_reloc-{preloc_tag}.csv")
        checks.append(dict(
            check=f"lottery alpha=1 vs D6 interventions_adr0003 intB reloc={preloc_val:g}",
            n_rows=20,
            max_abs_diff=numdiff(mine.sort_values("seed").reset_index(drop=True),
                                 d6.sort_values("seed").reset_index(drop=True))))
    cross = pd.DataFrame(checks)
    cross["status"] = np.where(cross.max_abs_diff == 0.0, "PASS", "FAIL")
    cross.to_csv(OUT / "crosscheck.csv", index=False)
    assert (cross.max_abs_diff == 0.0).all(), cross

    # ------------------------------------------------------------------
    # Old-clock n = 20 reference (D3 archive, mgstd5, logit + equal_split)
    # ------------------------------------------------------------------
    old_arm = pd.read_csv(PB_OLD / "cells_v2" / "cell_lam100_pull100_logit_equal_mgstd5.csv")
    old_arm = old_arm.sort_values("agent_seed")
    old_base = pd.read_csv(PB_OLD / "baseline_off.csv")
    old_base = old_base[old_base.land == "mgstd5"].sort_values("agent_seed")
    old_lift = paired(old_arm["c_ideal_tm"].reset_index(drop=True),
                      old_base["c_ideal_tm"].reset_index(drop=True))

    # ------------------------------------------------------------------
    # Headline verdicts
    # ------------------------------------------------------------------
    head = lift[(lift.outcome == "c_ideal_tm") & (lift.lam == 1.0)
                & (lift.pull == 1.0)].set_index("credit")
    analysis = {
        "schema": "adr0003_gaps_analysis_v1",
        "m2_lift_n40": {
            "note": "lift = c_ideal_tm(dyadic) - c_ideal_tm(M1); paired on 40 seeds",
            **{f"credit={c}": {k: float(head.loc[c, k]) for k in
                               ("lift", "se", "ci_lo", "ci_hi", "p",
                                "lift20", "ci20_lo", "ci20_hi",
                                "w_ci_lo", "w_ci_hi")}
               for c in head.index},
        },
        "m2_lift_n20_oldclock_logit_equal": old_lift,
        "lottery_headline": {
            f"alpha={a}": {k: float(contrasts_lot[(contrasts_lot.scope == "pooled")
                                                 & (contrasts_lot.alpha == a)
                                                 & (contrasts_lot.outcome == o)]
                                    [k].iloc[0]) for k in ("diff", "ci_lo", "ci_hi")}
            for a in (0.5, 0.0) for o in ("top10_fin", "ideal_loss_cum")},
        "career_gradient": {
            exit_mode: {str(grad["outcome"].iloc[i]): {k: float(grad[k].iloc[i])
                        for k in ("diff", "ci_lo", "ci_hi")}
                        for i in range(len(grad))}
            for exit_mode, grad in gradients_car.groupby("exit_mode")},
        "counts": {
            "lottery": int(len(lot)), "career_noisy": int(len(car)),
            "m2_n40": int(len(m2)),
        },
    }
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=1))

    # ------------------------------------------------------------------
    # Console tables
    # ------------------------------------------------------------------
    pd.set_option("display.width", 200, "display.max_columns", 40)

    def fmt_df(df: pd.DataFrame) -> str:
        return df.to_string(index=False, float_format=lambda v: f"{v:+.4f}")

    print("== lottery summary ==")
    print(fmt_df(summary_lot[["alpha", "p_reloc", "n", "top10_fin",
                              "ideal_loss_cum", "c_ideal_tm", "harvested_fin"]]))
    print("\n== lottery paired contrasts (alpha minus alpha=1) ==")
    print(fmt_df(contrasts_lot[["scope", "alpha", "outcome", "diff", "ci_lo",
                                "ci_hi", "p"]]))
    print("\n== career noisy summary ==")
    print(fmt_df(summary_car[["exit_mode", "g", "career_percap_harvest_final",
                              "career_gap_final", "career_gap_ratio_mean_late",
                              "c_ideal_tm"]]))
    print("\n== career gradients (high g minus low g) ==")
    print(fmt_df(gradients_car))
    print("\n== M2 n=40 cells ==")
    print(fmt_df(summary_m2))
    print("\n== lift tests (c_ideal_tm) ==")
    print(fmt_df(lift[lift.outcome == "c_ideal_tm"]))
    print("\n== chain at n=40 ==")
    print(fmt_df(chain))
    print("\n== crosscheck ==")
    print(fmt_df(cross))
    print("\n== old-clock n=20 reference (logit+equal, mgstd5) ==")
    print({k: round(v, 4) for k, v in old_lift.items()})


if __name__ == "__main__":
    main()