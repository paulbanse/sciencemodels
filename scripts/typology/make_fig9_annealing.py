"""F9 for drafts/paper_sketch_v1.md Section 5.2: crowding perception noise
as simulated annealing, not epistemic destruction.

The sketch's claim (5.2, `typology_v1` / P-0005): when candidate-tile
perception noise scales with local crowding (eps_eff = eps * m_t**gamma,
gamma=1), the endpoint proxy wedge NARROWS (+0.0316 paired, 95% CI
[+0.0101, +0.0532] in wedge_fin = c_ideal - c_local terms) and the
cumulative 200-step ideal-merit loss FALLS 16.3% (-0.0783 paired, 95% CI
[-0.0977, -0.0598]) - because the current depleted tile is read
noise-free (stay bias), wider noise on crowded candidates occasionally
produces a high upward draw that dislodges trapped agents onto fresher
tiles (endogenous Zollman transient diversity / annealing).

Two-panel contrast, gamma=0 (noise off) vs gamma=1 (crowding noise):
* A - endpoint proxy excess dr = c_local_fin - c_ideal_fin per run
  (240 runs per arm, collapsed over kappa/curiosity/entry), arm means with
  run-level bootstrap 95% CIs, and the paired narrowing annotated;
* B - the paired ideal_loss_cum delta (gamma=1 - gamma=0) per
  curiosity x entry condition (40 seed/kappa-matched pairs each) with
  bootstrap 95% CIs, plus the collapsed overall main as a reference line.

House style: PALETTE / HOUSE_STYLE / _cosmetics / write_figure_data from
scripts/report_notebook/diagram_templates.py (read-only import; the F7
builder in round_report_figures.py is the structural precedent).  dpi 180
per commission.  Every number is recomputed from data/typology_v1/runs.csv;
the two paired statistics the sketch cites are annotated from the round's
canonical contrast_table.csv (so the figure cannot disagree with the paper
text in the last decimal), and a fresh bootstrap must reproduce them to
within Monte-Carlo noise before anything is drawn.

Outputs (commissioned):
  output/report_notebook_v1/F9_simulated_annealing_crowding_noise.png
  output/report_notebook_v1/F9_simulated_annealing_crowding_noise_data.csv

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/typology/make_fig9_annealing.py
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "report_notebook"))

import diagram_templates as T  # noqa: E402
from diagram_templates import HOUSE_STYLE, PALETTE, write_figure_data  # noqa: E402

DATA = REPO / "data" / "typology_v1"
OUT_DIR = REPO / "output" / "report_notebook_v1"
STEM = "F9_simulated_annealing_crowding_noise"
PNG = OUT_DIR / (STEM + ".png")

BOOT_N = 20000
RNG_SEED = 20261115  # the typology round's bootstrap seed (MC cross-check)
C_LEVELS = (0.3, 0.5, 0.7)
G_LEVELS = (0.0, 0.1)


def _boot_ci(vals, rng, n=BOOT_N):
    v = np.asarray(vals, dtype=float)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _paired(r):
    """gamma=1 and gamma=0 frames, matched on (curiosity, g_entry, seed, kappa)."""
    a = r[r["gamma"] == 0].set_index(["curiosity", "g_entry", "seed", "kappa"])
    b = r[r["gamma"] == 1].set_index(["curiosity", "g_entry", "seed", "kappa"])
    common = a.index.intersection(b.index)
    assert len(common) == 240, "pairing broken: %d" % len(common)
    return a.loc[common], b.loc[common]


def build_data():
    r = pd.read_csv(DATA / "runs.csv")
    r["dr_fin"] = r["c_local_fin"] - r["c_ideal_fin"]  # = -wedge_fin

    # canonical paired statistics (the numbers the sketch's 5.2 cites)
    ct = pd.read_csv(DATA / "contrast_table.csv")

    def ct_row(meas):
        sub = ct[(ct["measure"] == meas) &
                 (ct["contrast"].str.startswith("destruction_main"))]
        assert len(sub) == 1
        return sub.iloc[0]

    w, l = ct_row("wedge_fin"), ct_row("ideal_loss_cum")

    a0, a1 = _paired(r)
    narrow = a1["dr_fin"] - a0["dr_fin"]
    assert abs(narrow.mean() + w["mean_delta"]) < 1e-12, "dr narrowing != -wedge main"
    loss_a0, loss_a1 = _paired(r)
    loss_d = loss_a1["ideal_loss_cum"] - loss_a0["ideal_loss_cum"]
    assert abs(loss_d.mean() - l["mean_delta"]) < 1e-12, "loss delta != contrast main"

    # fresh bootstrap must reproduce the canonical CIs to MC noise
    rng = np.random.default_rng(RNG_SEED)
    nlo_chk, nhi_chk = _boot_ci(narrow, rng)
    lo_chk, hi_chk = _boot_ci(loss_d, rng)
    assert abs(nlo_chk + w["ci_hi"]) < 2e-3 and abs(nhi_chk + w["ci_lo"]) < 2e-3, (
        "fresh narrowing CI diverges from contrast_table.csv beyond MC noise")
    assert abs(lo_chk - l["ci_lo"]) < 2e-3 and abs(hi_chk - l["ci_hi"]) < 2e-3, (
        "fresh loss CI diverges from contrast_table.csv beyond MC noise")

    loss_cond = []
    for c in C_LEVELS:
        for g in G_LEVELS:
            mask = (r["curiosity"] == c) & (r["g_entry"] == g)
            d = (r[mask & (r["gamma"] == 1)].set_index(["kappa", "seed"])["ideal_loss_cum"]
                 - r[mask & (r["gamma"] == 0)].set_index(["kappa", "seed"])["ideal_loss_cum"])
            assert len(d) == 40, "condition pair count %d != 40" % len(d)
            clo, chi = _boot_ci(d, rng)
            loss_cond.append(dict(curiosity=c, g_entry=g, delta=float(d.mean()),
                                  lo=clo, hi=chi, n=len(d)))
    return {
        "runs": r,
        "dr": {g: r[r["gamma"] == g]["dr_fin"].to_numpy(float) for g in (0.0, 1.0)},
        "dr_means": {g: float(r[r["gamma"] == g]["dr_fin"].mean()) for g in (0.0, 1.0)},
        "narrow_mean": float(narrow.mean()),
        "narrow_ci": (-float(w["ci_hi"]), -float(w["ci_lo"])),  # canonical, sign-flipped
        "loss_cond": loss_cond,
        "loss_overall": (float(loss_d.mean()), float(l["ci_lo"]), float(l["ci_hi"]),
                         int(len(loss_d))),
        "loss_base": float(r[r["gamma"] == 0]["ideal_loss_cum"].mean()),
        "loss_d": loss_d,
    }


def figure_annealing(outdir=None):
    out = Path(outdir) if outdir else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    d = build_data()
    s = {**HOUSE_STYLE, "figsize": (9.2, 4.8)}
    ov, ovlo, ovhi, ovn = d["loss_overall"]
    pct = 100.0 * ov / d["loss_base"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=s["figsize"])

    # ---------------------------------------------------------------- A
    T._cosmetics(
        axA, s,
        "A · Proxy excess narrows under crowding noise",
        "candidate-tile perception noise",
        "$\\Delta r = c_{local} - c_{ideal}$ (final step)")
    jitter = np.random.default_rng(7)
    for x, g, color in ((0.0, 0.0, PALETTE["gray"]), (1.0, 1.0, PALETTE["purple"])):
        vals = d["dr"][g]
        axA.scatter(x + jitter.uniform(-0.13, 0.13, size=len(vals)), vals,
                    s=6, alpha=0.22, color=color, edgecolors="none", rasterized=True)
        rng_arm = np.random.default_rng(RNG_SEED + 10 + int(g))
        lo, hi = _boot_ci(vals, rng_arm)
        mean = d["dr_means"][g]
        axA.errorbar([x], [mean], yerr=[[mean - lo], [hi - mean]], fmt="none",
                     ecolor=color, elinewidth=1.2, capsize=3.5)
        axA.plot([x], [mean], marker="D", ms=6.5, color=color,
                 markeredgecolor="white", markeredgewidth=0.6, zorder=5)
    axA.axhline(0.0, color=PALETTE["gray"], lw=0.8, ls=(0, (4, 3)))
    axA.set_xlim(-0.55, 1.55)
    axA.set_xticks([0.0, 1.0])
    axA.set_xticklabels(["$\\gamma = 0$\n(noise off)",
                         "$\\gamma = 1$\n($\\epsilon \\cdot m$ scaling)"],
                        fontsize=s["tick_size"])
    nm, (nlo, nhi) = d["narrow_mean"], d["narrow_ci"]
    axA.text(0.03, 0.97,
             "paired narrowing %+.4f  95%% CI [%+.4f, %+.4f]\n"
             "(wedge_fin %+.4f [%+.4f, %+.4f]; 240 pairs)"
             % (nm, nlo, nhi, -nm, -nhi, -nlo),
             transform=axA.transAxes, fontsize=s["note_size"] + 0.5, va="top",
             color=PALETTE["purple"],
             bbox=dict(boxstyle="round,pad=0.32", fc="white",
                       ec=PALETTE["purple_light"], lw=0.8))

    # ---------------------------------------------------------------- B
    T._cosmetics(
        axB, s,
        "B · Loss delta negative in all six conditions",
        "curiosity c (purple: g=0.0, orange: g=0.1)",
        "paired $\\Delta$ideal_loss_cum ($\\gamma=1 - \\gamma=0$)")
    for g, color in zip(G_LEVELS, (PALETTE["purple"], PALETTE["orange"])):
        for i, c in enumerate(C_LEVELS):
            cd = next(x for x in d["loss_cond"]
                      if x["curiosity"] == c and x["g_entry"] == g)
            off = -0.18 if g == 0.0 else 0.18
            axB.errorbar([i + off], [cd["delta"]],
                         yerr=[[cd["delta"] - cd["lo"]], [cd["hi"] - cd["delta"]]],
                         fmt="none", ecolor=color, elinewidth=1.1, capsize=3.0)
            axB.plot([i + off], [cd["delta"]], marker="o", ms=6.0, color=color,
                     markeredgecolor="white", markeredgewidth=0.6, zorder=5,
                     label="entry g=%s" % ("0.0" if g == 0.0 else "0.1") if i == 0 else None)
    axB.axhline(0.0, color=PALETTE["gray"], lw=0.8, ls=(0, (4, 3)))
    axB.axhline(ov, color=PALETTE["gray"], lw=1.1, ls=(0, (6, 3)),
                label="overall (240 pairs)")
    axB.text(0.02, 0.06,
             "overall paired Δ %+.4f  95%% CI [%+.4f, %+.4f]\n"
             "= %.1f%% of the γ=0 baseline loss (%.3f)"
             % (ov, ovlo, ovhi, pct, d["loss_base"]),
             transform=axB.transAxes, fontsize=s["note_size"] + 0.5, va="bottom",
             color=PALETTE["blue"],
             bbox=dict(boxstyle="round,pad=0.32", fc="white",
                       ec=PALETTE["blue_light"], lw=0.8))
    axB.set_xticks(np.arange(len(C_LEVELS)), ["c=0.3", "c=0.5", "c=0.7"],
                   fontsize=s["tick_size"])
    axB.legend(fontsize=s["tick_size"], frameon=False, loc="center right",
               bbox_to_anchor=(1.0, 0.42))
    lo_all = min([x["lo"] for x in d["loss_cond"]] + [ovlo])
    axB.set_ylim(1.25 * lo_all - 0.01, 0.035)

    # ---------------------------------------------------------------- save
    note = textwrap.fill(
        "sources: data/typology_v1/runs.csv (480 runs; per arm 240 = 2 kappa x 3 c x "
        "2 g x 20 seeds, collapsed) + contrast_table.csv | pairing: seed/kappa/c/g-"
        "matched; CIs: percentile bootstrap, 20,000 resamples (the two paired stats "
        "are the typology round's canonical contrast_table.csv CIs; a fresh bootstrap "
        "reproduces them to MC noise) | dr = c_local_fin - c_ideal_fin = -wedge_fin; "
        "ideal_loss_cum = -sum(ideal_merit_cum), positive = shortfall | mechanism "
        "(sketch 5.2): the current depleted tile is read noise-free (stay bias), so "
        "wider noise on crowded candidates occasionally draws an agent upward onto "
        "fresher tiles - annealing, not destruction | model-internal typology_v1 "
        "(P-0005), NOT calibrated | palette: diagram_templates.PALETTE, dpi 180",
        width=148)
    fig.text(0.006, 0.012, note, fontsize=s["note_size"], color=PALETTE["gray"],
             va="bottom", linespacing=1.45)
    fig.tight_layout(rect=(0, 0.20, 1, 1), w_pad=2.2)
    fig.savefig(PNG, dpi=s["dpi"])
    pdf = PNG.with_suffix(".pdf")
    fig.savefig(pdf)
    plt.close(fig)

    # ------------------------------------------------------- companion CSV
    r = d["runs"]
    rows = []
    for g in (0.0, 1.0):
        vals = d["dr"][g]
        lo, hi = _boot_ci(vals, np.random.default_rng(RNG_SEED + 10 + int(g)))
        rows.append(dict(panel="A", kind="arm_mean_ci", gamma=g, curiosity="",
                         g_entry="", kappa="", seed="", estimate=d["dr_means"][g],
                         n=len(vals), lo=lo, hi=hi, pct_reduction="",
                         note="dr_fin per-run mean over 240 runs; run-level bootstrap CI"))
    rows.append(dict(panel="A", kind="paired_narrowing", gamma="", curiosity="",
                     g_entry="", kappa="", seed="", estimate=d["narrow_mean"], n=240,
                     lo=d["narrow_ci"][0], hi=d["narrow_ci"][1], pct_reduction="",
                     note="gamma1-gamma0 on dr; wedge_fin equivalent %+.4f [%+.4f, %+.4f]"
                          % (-d["narrow_mean"], -d["narrow_ci"][1], -d["narrow_ci"][0])))
    sub = r[["gamma", "kappa", "curiosity", "g_entry", "seed", "dr_fin"]].copy()
    sub = sub.rename(columns={"dr_fin": "estimate"})
    sub.insert(0, "panel", "A")
    sub.insert(1, "kind", "run_dot")
    for col in ("n", "lo", "hi", "pct_reduction", "note"):
        sub[col] = ""
    rows_df = pd.DataFrame(rows)
    for x in d["loss_cond"]:
        rows_df = pd.concat([rows_df, pd.DataFrame([dict(
            panel="B", kind="condition_delta_ci", gamma="", curiosity=x["curiosity"],
            g_entry=x["g_entry"], kappa="", seed="", estimate=x["delta"], n=x["n"],
            lo=x["lo"], hi=x["hi"], pct_reduction="",
            note="paired loss delta per condition; fresh bootstrap CI")])],
            ignore_index=True)
    rows_df = pd.concat([rows_df, pd.DataFrame([dict(
        panel="B", kind="overall_delta_ci", gamma="", curiosity="", g_entry="",
        kappa="", seed="", estimate=d["loss_overall"][0], n=d["loss_overall"][3],
        lo=d["loss_overall"][1], hi=d["loss_overall"][2], pct_reduction=pct,
        note="paired loss delta, all 240 pairs; canonical CI; reference line in B")])],
        ignore_index=True)
    pairs = pd.DataFrame({
        "panel": "B", "kind": "bootstrap_substrate", "gamma": "",
        "curiosity": [i[0] for i in d["loss_d"].index],
        "g_entry": [i[1] for i in d["loss_d"].index],
        "kappa": [i[3] for i in d["loss_d"].index],
        "seed": [i[2] for i in d["loss_d"].index],
        "estimate": d["loss_d"].to_numpy(float), "n": "", "lo": "", "hi": "",
        "pct_reduction": "", "note": ""})
    export = pd.concat([rows_df, sub, pairs], ignore_index=True)
    csv_path = write_figure_data(export, PNG)
    return {"png": PNG, "pdf": pdf, "csv": csv_path,
            "narrow": (d["narrow_mean"], *d["narrow_ci"]),
            "loss_overall": (ov, ovlo, ovhi, pct),
            "dr_means": d["dr_means"]}


def main():
    res = figure_annealing()
    T.ok("F9 written: %s (%.0f kB)" % (res["png"].name, res["png"].stat().st_size / 1024))
    T.ok("companion CSV: %s" % res["csv"].name)
    T.info("panel A: dr means %+.4f (gamma=0) -> %+.4f (gamma=1); paired narrowing %+.4f "
           "[%+.4f, %+.4f]" % (res["dr_means"][0.0], res["dr_means"][1.0],
                               res["narrow"][0], res["narrow"][1], res["narrow"][2]))
    T.info("panel B: overall paired loss delta %+.4f [%+.4f, %+.4f] = %.1f%% of baseline"
           % (res["loss_overall"][0], res["loss_overall"][1], res["loss_overall"][2],
              res["loss_overall"][3]))
    print("F9 OK ->", res["png"])


if __name__ == "__main__":
    main()
