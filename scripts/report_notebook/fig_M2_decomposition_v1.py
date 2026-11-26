"""M2_decomposition_v1 - Tier M2 counterfactual ablation-share decomposition.

Extends the Figure 9 method (FA2_v2_drop_decomposition, builder
``fig_FA2_v2_arm_a_decomposition.py``; paper section 4.2 and the Section 8
figure-to-artifact map, row "Birth-tile decomposition") to the contested Tier
M2 lever ``dyadic_move_pull``. The M1 -> full-M2 change in time-mean
``Corr(prestige, ideal merit)`` is decomposed into additive mechanism shares
along the counterfactual chain

    M1  (dyadic layer off)               baseline_off.csv
    + link formation only (pull = 0)     cells_v1,    credit = structural
    + spatial co-movement (pull = 1)     cells_v2,    credit = structural
    + equal-split credit sharing         cells_v2,    credit = equal_split

per ``dyadic_lambda`` in {0, 1} and landscape in {mgstd5, noisypr5}, pooled
over formulation (logit/threshold). The shares telescope, so they sum to
exactly 100% of the gap, as in Figure 9:

    gap                = full_M2 - M1
    link-formation     = (links - M1)      / gap
    co-movement        = (pull - links)    / gap
    equal-split credit = (credit - pull)   / gap

Sources (read-only; NO new runs):
  data/path_b_postbug/baseline_off.csv                      (M1, n = 20/landscape)
  data/path_b_postbug/cells_v1/cell_lam{000,100}_{logit,threshold}_struct_{land}.csv
  data/path_b_postbug/cells_v1/cell_lam{000,100}_{logit,threshold}_equal_{land}.csv
  data/path_b_postbug/cells_v2/cell_lam{000,100}_pull100_{logit,threshold}_struct_{land}.csv
  data/path_b_postbug/cells_v2/cell_lam{000,100}_pull100_{logit,threshold}_equal_{land}.csv

The auxiliary ``pull = 0`` equal-split arm (``links_eq``) supports the
order-robustness chains (credit step before the pull step, and the two-share
Figure-9 sub-case with ``dyadic_move_pull`` as the single ablated lever).
Uncertainty: paired bootstrap over the 20 agent_seeds (level values matched by
seed across all levels), 10,000 resamples. Model-internal, NOT calibrated.

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
        scripts/report_notebook/fig_M2_decomposition_v1.py
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import diagram_templates as T
import visual_qa

ARCH = REPO / "data/path_b_postbug"
OUT = ARCH / "figures/m2_decomposition_v1_shares.png"

LAMS = (0.0, 1.0)
LANDS = ("mgstd5", "noisypr5")
FORMS = ("logit", "threshold")
SEEDS = tuple(range(20))
N_BOOT = 10_000
BOOT_SEED = 20261126

#: primary chain (the prompt's enumeration order) and its share labels
CHAIN = ("M1", "links", "pull", "credit")
CHAIN_LABELS = ("link formation (pull = 0)", "spatial co-movement (pull = 1)",
                "equal-split credit sharing")
#: order-B chain: credit step taken before the pull step
CHAIN_B = ("M1", "links", "links_eq", "credit")
#: two-share Figure-9 sub-case: dyadic_move_pull as the single ablated lever
CHAIN_2 = ("M1", "links_eq", "credit")
CHAIN_2_LABELS = ("residual (links + credit at pull = 0)",
                  "dyadic_move_pull lever")

ARM_ORDER = ("links", "pull", "credit")
ARM_COLORS = {"links": T.PALETTE["purple"], "pull": T.PALETTE["orange"],
              "credit": T.PALETTE["green"]}


# ---------------------------------------------------------------------------
# data loading (one tidy long frame, seed-matched across levels)
# ---------------------------------------------------------------------------

def _read_cell(rel: str) -> pd.DataFrame:
    df = pd.read_csv(ARCH / rel)
    assert len(df) == 20, f"{rel}: expected 20 runs, got {len(df)}"
    assert sorted(int(s) for s in df.agent_seed) == list(SEEDS), \
        f"{rel}: agent_seed coverage != 0..19"
    return df


def _add(rows: list, df: pd.DataFrame, lam, land: str, level: str,
         formulation: str) -> None:
    for r in df.itertuples():
        rows.append({
            "lam": lam, "land": land, "level": level,
            "formulation": formulation, "seed": int(r.agent_seed),
            "c_ideal": float(r.c_ideal_tm),
            "team": float(r.team_tm),
            "wedge": float(r.c_local_tm) - float(r.c_ideal_tm),
        })


def _load() -> pd.DataFrame:
    rows: list[dict] = []
    b = pd.read_csv(ARCH / "baseline_off.csv")
    base_seeds: dict[str, list[int]] = {}
    for land in LANDS:
        sub = b[b.land == land]
        assert len(sub) == 20, f"baseline_off {land}: n={len(sub)}"
        assert sorted(int(s) for s in sub.agent_seed) == list(SEEDS), \
            f"baseline_off {land}: seed coverage"
        base_seeds[land] = (sub.sort_values("agent_seed")
                            .landscape_seed.astype(int).tolist())
        _add(rows, sub, None, land, "M1", "off")
    checked = set()
    for lam in LAMS:
        tag = f"{int(round(lam * 100)):03d}"
        for form in FORMS:
            for land in LANDS:
                for rel, level in (
                        (f"cells_v1/cell_lam{tag}_{form}_struct_{land}.csv",
                         "links"),
                        (f"cells_v1/cell_lam{tag}_{form}_equal_{land}.csv",
                         "links_eq"),
                        (f"cells_v2/cell_lam{tag}_pull100_{form}_struct_"
                         f"{land}.csv", "pull"),
                        (f"cells_v2/cell_lam{tag}_pull100_{form}_equal_"
                         f"{land}.csv", "credit")):
                    cell = _read_cell(rel)
                    if (level, land) not in checked:
                        # seed-matched design: same landscape per agent_seed
                        got = (cell.sort_values("agent_seed")
                               .landscape_seed.astype(int).tolist())
                        assert got == base_seeds[land], \
                            f"{rel}: landscape_seed != baseline pairing"
                        checked.add((level, land))
                    _add(rows, cell, lam, land, level, form)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# chain decomposition with paired-seed bootstrap
# ---------------------------------------------------------------------------

def _pivot(df: pd.DataFrame, lam, land: str, forms, metric: str,
           levels: tuple[str, ...]) -> pd.DataFrame:
    """Per-seed level values (mean over formulations), one column per level."""
    sub = df[df.land == land]
    sub = sub[sub.level.isin(levels)]
    if lam is None:
        sub = sub[sub.level == "M1"]
    else:
        sub = sub[(sub.lam == lam) | (sub.level == "M1")]
    if forms != "pooled":
        keep = sub.formulation.isin(forms) | (sub.level == "M1")
        sub = sub[keep]
    g = sub.groupby(["seed", "level"])[metric].mean().reset_index()
    wide = g.pivot(index="seed", columns="level", values=metric).sort_index()
    assert list(wide.index) == list(SEEDS), f"pivot seeds off: {land}/{lam}"
    assert not wide.isna().any().any(), f"pivot NaN: {land}/{lam}/{forms}"
    for lv in levels:
        assert lv in wide.columns, f"missing level {lv}"
    return wide[list(levels)]


def _chain(wide: pd.DataFrame, chain: tuple[str, ...],
          paired: bool = True) -> dict:
    """Point + bootstrap shares of the chain (steps telescope to the gap).

    ``paired=True`` resamples seed indices jointly across levels (the
    seed-matched design); ``paired=False`` resamples each level
    independently, the robustness check against cross-level RNG-stream
    divergence (link declaration shifts the stream, so same-seed runs of
    different levels are not bit-paired).
    """
    rng = np.random.default_rng(BOOT_SEED)
    n = len(wide)
    if paired:
        idx = rng.integers(0, n, size=(N_BOOT, n))
        boot = {lv: wide[lv].to_numpy()[idx].mean(axis=1) for lv in chain}
    else:
        boot = {lv: wide[lv].to_numpy()[
            rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
            for lv in chain}
    means = {lv: float(wide[lv].mean()) for lv in chain}

    def shares(vals: dict) -> tuple[np.ndarray | float, list]:
        gap = vals[chain[-1]] - vals[chain[0]]
        steps = [vals[chain[i + 1]] - vals[chain[i]]
                 for i in range(len(chain) - 1)]
        if isinstance(gap, np.ndarray):
            with np.errstate(divide="ignore", invalid="ignore"):
                out = [100.0 * s / gap for s in steps]
            return gap, out
        return gap, [100.0 * s / gap for s in steps]

    gap_p, sh_p = shares(means)
    gap_b, sh_b = shares(boot)
    assert abs(gap_p) > 1e-9, "degenerate M1 -> full-M2 gap"
    # decomposition identity: the chain shares sum to exactly 100%
    assert np.allclose(sum(sh_p), 100.0, atol=1e-9), "point shares != 100%"
    assert np.allclose(np.sum(np.vstack(sh_b), axis=0), 100.0, atol=1e-6), \
        "bootstrap shares != 100%"
    near = int(np.sum(np.abs(gap_b) < 1e-9))
    return {
        "levels": means,
        "gap": gap_p,
        "gap_ci": tuple(float(v) for v in np.percentile(gap_b, [2.5, 97.5])),
        "shares": [float(v) for v in sh_p],
        "share_ci": [tuple(float(v) for v in np.percentile(s, [2.5, 97.5]))
                     for s in sh_b],
        "n_boot_degenerate": near,
    }


def _fmt_ci(lo: float, hi: float, digits: int = 1) -> str:
    return f"[{lo:+.{digits}f}, {hi:+.{digits}f}]"


def _fmt_level(v: float, ci: tuple[float, float]) -> str:
    return f"{v:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"


def analyse(df: pd.DataFrame) -> dict:
    """All decomposition variants; prints the verification tables."""
    out = {"primary": {}, "orderB": {}, "twoshare": {}, "unpaired": {},
           "logit": {}, "threshold": {}, "team": {}, "wedge": {}}

    def level_ci(wide: pd.DataFrame, lv: str) -> tuple[float, float]:
        v = wide[lv].to_numpy()
        rng = np.random.default_rng(BOOT_SEED)
        idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
        m = v[idx].mean(axis=1)
        return tuple(float(x) for x in np.percentile(m, [2.5, 97.5]))

    print("\n== primary chain (pooled over formulation), metric c_ideal_tm ==")
    hdr = ("lambda", "landscape", "M1 ref", "+links (pull=0)",
           "+co-move (pull=1)", "+equal-split", "gap M2-M1",
           "links %", "co-move %", "credit %", "sum %")
    print(" | ".join(hdr))
    for lam in LAMS:
        for land in LANDS:
            w = _pivot(df, lam, land, "pooled", "c_ideal", CHAIN)
            r = _chain(w, CHAIN)
            cis = {lv: level_ci(w, lv) for lv in CHAIN}
            r["level_ci"] = cis
            out["primary"][(lam, land)] = r
            print(f"{lam:.0f} | {land} | "
                  + " | ".join(_fmt_level(r["levels"][lv], cis[lv])
                               for lv in CHAIN)
                  + f" | {r['gap']:+.4f} {_fmt_ci(*r['gap_ci'], 4)}"
                  + " | " + " | ".join(
                      f"{s:+.1f} {_fmt_ci(*ci)}" for s, ci
                      in zip(r["shares"], r["share_ci"]))
                  + f" | {sum(r['shares']):.6f}")

    print("\n== level n (runs) ==")
    for lv in ("M1", "links", "links_eq", "pull", "credit"):
        n = len(df[df.level == lv])
        print(f"{lv:9s}: {n:4d} runs ({n // len(SEEDS)} cells x "
              f"{len(SEEDS)} seeds)")

    print("\n== order-B chain (credit step before pull step) ==")
    print("lambda | landscape | links % | credit % | co-move % | sum %")
    for lam in LAMS:
        for land in LANDS:
            w = _pivot(df, lam, land, "pooled", "c_ideal", CHAIN_B)
            r = _chain(w, CHAIN_B)
            out["orderB"][(lam, land)] = r
            print(f"{lam:.0f} | {land} | "
                  + " | ".join(f"{s:+.1f} {_fmt_ci(*ci)}" for s, ci
                               in zip(r["shares"], r["share_ci"]))
                  + f" | {sum(r['shares']):.6f}")

    print("\n== two-share sub-case (dyadic_move_pull alone ablated, "
          "credit held at equal_split) ==")
    print("lambda | landscape | residual % | pull lever % | sum %")
    for lam in LAMS:
        for land in LANDS:
            w = _pivot(df, lam, land, "pooled", "c_ideal", CHAIN_2)
            r = _chain(w, CHAIN_2)
            out["twoshare"][(lam, land)] = r
            print(f"{lam:.0f} | {land} | "
                  + " | ".join(f"{s:+.1f} {_fmt_ci(*ci)}" for s, ci
                               in zip(r["shares"], r["share_ci"]))
                  + f" | {sum(r['shares']):.6f}")

    print("\n== unpaired-bootstrap robustness (independent resampling per "
          "level) ==")
    print("lambda | landscape | gap | links % | co-move % | credit % | sum %")
    for lam in LAMS:
        for land in LANDS:
            w = _pivot(df, lam, land, "pooled", "c_ideal", CHAIN)
            r = _chain(w, CHAIN, paired=False)
            out["unpaired"][(lam, land)] = r
            print(f"{lam:.0f} | {land} | {r['gap']:+.4f} "
                  f"{_fmt_ci(*r['gap_ci'], 4)} | "
                  + " | ".join(f"{s:+.1f} {_fmt_ci(*ci)}" for s, ci
                               in zip(r["shares"], r["share_ci"]))
                  + f" | {sum(r['shares']):.6f}")

    for form in FORMS:
        print(f"\n== primary chain, {form} only ==")
        print("lambda | landscape | links % | co-move % | credit % | sum %")
        for lam in LAMS:
            for land in LANDS:
                w = _pivot(df, lam, land, (form,), "c_ideal", CHAIN)
                r = _chain(w, CHAIN)
                out[form][("c_ideal", lam, land)] = r
                print(f"{lam:.0f} | {land} | "
                      + " | ".join(f"{s:+.1f}" for s in r["shares"])
                      + f" | {sum(r['shares']):.6f}")

    for metric, label in (("team", "mean_team_size (team_tm)"),
                          ("wedge", "prestige-ideal wedge "
                           "(c_local_tm - c_ideal_tm)")):
        print(f"\n== secondary metric: {label}, primary chain ==")
        print("lambda | landscape | gap | links % | co-move % | credit % | sum %")
        for lam in LAMS:
            for land in LANDS:
                w = _pivot(df, lam, land, "pooled", metric, CHAIN)
                r = _chain(w, CHAIN)
                out[metric][(lam, land)] = r
                print(f"{lam:.0f} | {land} | {r['gap']:+.4f} | "
                      + " | ".join(f"{s:+.1f}" for s in r["shares"])
                      + f" | {sum(r['shares']):.6f}")

    deg = sum(r["n_boot_degenerate"] for r in out["primary"].values())
    print(f"\nnear-degenerate bootstrap gaps in the primary chain: {deg} "
          f"of {N_BOOT * len(out['primary'])} replicates")
    return out


# ---------------------------------------------------------------------------
# draft figure (house style + visual_qa)
# ---------------------------------------------------------------------------

def build_figure(df: pd.DataFrame, res: dict) -> dict:
    s = {**T.HOUSE_STYLE, "figsize": (10.2, 5.2)}
    fig, axes = plt.subplots(1, 2, figsize=s["figsize"], sharey=True)
    x = np.arange(len(LAMS))
    w = 0.26
    offsets = (-w - 0.015, 0.0, w + 0.015)
    title = ("Spatial co-movement carries the M2 alignment change; "
             "nominal links alone are inert")

    for ax, land in zip(axes, LANDS):
        handles = []
        for off, arm, label in zip(offsets, ARM_ORDER, CHAIN_LABELS):
            pts = [res["primary"][(lam, land)] for lam in LAMS]
            vals = [p["shares"][ARM_ORDER.index(arm)] for p in pts]
            bars = ax.bar(x + off, vals, width=w,
                          color=ARM_COLORS[arm], label=label,
                          edgecolor="none")
            ax.bar_label(bars,
                         labels=[f"{v:+.1f}%" for v in vals],
                         padding=3, fontsize=s["tick_size"] - 0.5)
            handles.append(bars)
        ax.axhline(0, color=T.PALETTE["gray"], linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(["\u03bb = 0\n(prestige matching)",
                            "\u03bb = 1\n(closeness \u00d7 local-merit)"])
        T._cosmetics(ax, s,
                     f"{land} ({'clean terrain' if land == 'mgstd5' else 'noisy terrain'})",
                     "partner-choice rule \u03bb",
                     "share of the M1 \u2192 full-M2 gap (%)")

    # one joint scale across panels (shares are comparable), with explicit
    # in-range ticks so no out-of-view locator ticks reach the text inventory
    ends = [res["primary"][(lam, land)]["shares"][i]
            for land in LANDS for lam in LAMS for i in range(len(ARM_ORDER))]
    span = max(ends) - min(ends)
    joint_lo = min(ends) - 0.30 * span
    joint_hi = max(ends) + 0.26 * span
    ticks = [t for t in np.arange(-100.0, 251.0, 50.0)
             if joint_lo <= t <= joint_hi]
    for ax in axes:
        ax.set_ylim(joint_lo, joint_hi)
        ax.set_yticks(ticks)

    fig.suptitle(title, fontsize=s["title_size"], y=0.985)
    handles, labels = axes[0].get_legend_handles_labels()
    assert len(handles) == 3, f"expected 3 bar containers, got {len(handles)}"
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925),
               ncol=3, fontsize=s["tick_size"], frameon=False)

    note = textwrap.fill(
        "source: data/path_b_postbug/ (archive: baseline_off 40 runs + "
        "cells_v1 800 + cells_v2 960; this figure reads 24 cell files "
        "= 3 levels x 2 lambda x 2 formulation x 2 landscapes x 20 seeds "
        "+ the 40 baseline runs; NO new runs). Chain per lambda x landscape, "
        "pooled over formulation: "
        "M1 (dyadic off) -> +link formation (pull=0, structural credit) -> "
        "+spatial co-movement (pull=1, structural) -> +equal-split credit "
        "(pull=1, equal_split). Shares telescope to 100% of "
        "gap = full-M2 - M1 (for lambda=0 the gap is negative: a positive "
        "share moves alignment along that drop). Bars are chain-share point "
        "estimates; whiskers are omitted (the ratio CI is unidentified where "
        "the gap CI grazes zero, lambda=1 mgstd5) - paired-seed bootstrap "
        "95% CIs (20 agent_seeds matched across levels, 10,000 resamples) "
        "are in the report table and the sidecar ci95_* columns. "
        "Level means, order-B and "
        "per-formulation chains: output/reports/m2_decomposition_v1.md. "
        "Metric: time-mean Corr(prestige, ideal merit), c_ideal_tm. "
        "Model-internal, NOT calibrated.",
        width=155)
    fig.text(0.006, 0.012, note, fontsize=s["note_size"],
             color=T.PALETTE["gray"], va="bottom", linespacing=1.45)
    fig.tight_layout(rect=(0, 0.30, 1, 0.86))

    rows = []
    for land in LANDS:
        for lam in LAMS:
            r = res["primary"][(lam, land)]
            for i, arm in enumerate(ARM_ORDER):
                rows.append({
                    "metric": "m2share", "landscape": land,
                    "lam": lam, "arm": arm,
                    "value": r["shares"][i],
                    # FA2 convention: bars carry no whiskers (ratio CIs are
                    # unidentified where the gap CI grazes zero), so the QA
                    # ci columns are degenerate at the plotted value; the
                    # paired-seed bootstrap CIs ride along as ci95_*.
                    "ci_lo": r["shares"][i],
                    "ci_hi": r["shares"][i],
                    "ci95_lo": r["share_ci"][i][0],
                    "ci95_hi": r["share_ci"][i][1],
                    "gap": r["gap"], "gap_ci_lo": r["gap_ci"][0],
                    "gap_ci_hi": r["gap_ci"][1],
                    "m1_level": r["levels"]["M1"],
                    "links_level": r["levels"]["links"],
                    "pull_level": r["levels"]["pull"],
                    "credit_level": r["levels"]["credit"],
                    "n_per_level": 40, "n_m1": 20,
                })
    sidecar = pd.DataFrame(rows)

    panels = []
    for ax, land in zip(axes, LANDS):
        panels.append({"ax": ax, "metric": "m2share",
                       "filters": {"landscape": land},
                       "expect_bars": 6, "sidecar": sidecar,
                       "ci_cols": ("ci_lo", "ci_hi")})
    report = visual_qa.assert_figure_clean(fig, panels)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=s["dpi"])
    pdf = OUT.with_suffix(".pdf")
    fig.savefig(pdf)
    plt.close(fig)
    csv = T.write_figure_data(sidecar, OUT)
    print(f"\nQA: {report.get('_total', 0)} failure(s)")
    for item in report.get("failures", []):
        print("  FAIL:", item)
    print(f"Wrote {OUT} / {pdf} / {csv}")
    return {"png": str(OUT), "pdf": str(pdf), "data_csv": str(csv)}


if __name__ == "__main__":
    data = _load()
    results = analyse(data)
    build_figure(data, results)
