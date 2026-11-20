"""F4 - Closed-cohort equality (top10) and the 400-step lifecycle trajectory.

Section 3.4 (`Findings L4` & `L5`) of the paper sketch. Two panels:

  (a) top-10% prestige share vs. curiosity for the four swept homogeneous
      M0 landscapes (data/figures_v1/run_number{0,2,4,6}.csv, 10 repeats/c,
      H = 200): inequality hugs the 0.100 equality floor for c >= 0.1 on the
      low-noise landscapes and spikes only at c = 0 (frozen initial
      placement). Noisy2 drifts higher with curiosity -- drawn, not hidden.
  (b) within-run alignment corr(prestige, localMerit) over t in [0, 400]
      for the three legacy longitudinal runs
      (data/legacy_2025_2026/model_long_{MGauss1,NGauss,random}.csv): the
      Discovery Boom opens at +0.85 / +0.97 (step 10, MGauss1 / NGauss)
      and the two peaked landscapes drift negative by step 400 as their
      Gaussian peaks are mined out; random terrain oscillates instead
      (terminal +0.32). Model-internal, not calibrated.

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
        scripts/report_notebook/fig_3_4_m0_equality_and_lifecycle.py
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

OUT = REPO / "output/report_notebook_v1/F4_m0_equality_and_lifecycle.png"

SWEPT = [("MultG2", 2, T.PALETTE["purple"]),
         ("MultG6", 4, T.PALETTE["orange"]),
         ("Noisy2", 6, T.PALETTE["green"]),
         ("randominit", 0, T.PALETTE["blue"])]
LONG = [("MGauss1", T.PALETTE["purple"]),
        ("NGauss", T.PALETTE["orange"]),
        ("random", T.PALETTE["blue"])]
EQUALITY = 0.100
C_GRID = [round(0.1 * i, 10) for i in range(11)]


def load_top10() -> pd.DataFrame:
    rows = []
    for land, idx, _hue in SWEPT:
        df = pd.read_csv(REPO / f"data/figures_v1/run_number{idx}.csv")
        counts = df.groupby("initial_curiosity").size()
        assert (counts == 10).all(), f"{land}: repeats {counts.to_dict()}"
        for c in C_GRID:
            sub = df[df["initial_curiosity"] == c]["top10pct_prestige_share"]
            m = float(sub.mean())
            se = float(sub.std(ddof=1) / np.sqrt(len(sub)))
            rows.append(dict(landscape=land, curiosity=c, metric="top10",
                             value=m, ci_lo=m - 1.96 * se,
                             ci_hi=m + 1.96 * se, n=len(sub)))
    out = pd.DataFrame(rows)
    assert len(out) == 44, f"top10 rows {len(out)} != 44"
    return out


def load_long() -> pd.DataFrame:
    rows = []
    for name, _hue in LONG:
        df = pd.read_csv(REPO / f"data/legacy_2025_2026/model_long_{name}.csv")
        assert df["Step"].min() == 0 and df["Step"].max() == 400, (
            f"{name}: Step range {df['Step'].min()}-{df['Step'].max()}")
        for _, r in df.iterrows():
            rows.append(dict(landscape=name, step=int(r["Step"]),
                             metric="corr_local",
                             value=float(r["corr_prestige_localMerit"]),
                             ci_lo=np.nan, ci_hi=np.nan, n=1))
    out = pd.DataFrame(rows)
    assert len(out) == 3 * 401, f"long rows {len(out)} != 1203"
    return out


def build(outdir: Path | str | None = None) -> dict:
    top = load_top10()
    lng = load_long()
    s = {**T.HOUSE_STYLE, "figsize": (11.4, 4.9)}
    outdir = Path(outdir) if outdir else OUT.parent
    out = outdir / OUT.name

    fig, (axA, axB) = plt.subplots(1, 2, figsize=s["figsize"])

    # ---- (a) equality floor ---------------------------------------------
    n_a = 0
    for land, _i, hue in SWEPT:
        sel = top[top.landscape == land].sort_values("curiosity")
        axA.plot(sel.curiosity, sel.value, color=hue, lw=1.9, marker="o",
                 ms=3.6, label=land)
        n_a += 1
    axA.axhline(EQUALITY, color=T.PALETTE["gray"], ls=":", lw=1.4)
    axA.text(1.01, EQUALITY, "0.100 equality", fontsize=s["tick_size"] - 1,
             color=T.PALETTE["gray"], va="bottom", ha="right",
             transform=axA.get_yaxis_transform())
    # the c = 0 spike and the c >= 0.1 band, annotated from drawn values
    band = top[top.curiosity >= 0.1]
    lo, hi = float(band.value.min()), float(band.value.max())
    nois = top[(top.landscape == "Noisy2") & (top.curiosity >= 0.1)]
    land_mean = band.groupby("curiosity").value.mean()
    mean_lo, mean_hi = float(land_mean.min()), float(land_mean.max())
    axA.annotate(
        f"c = 0 spike: frozen initial placement\n"
        f"c $\\geq$ 0.1: {lo:.3f}–{hi:.3f} pooled\n"
        f"(Noisy2 alone reaches {float(nois.value.max()):.3f})",
        xy=(0.0, float(top[top.curiosity == 0.0].value.max())),
        xytext=(0.10, 0.185), fontsize=s["note_size"],
        color=T.PALETTE["gray"],
        arrowprops=dict(arrowstyle="->", color=T.PALETTE["gray"], lw=0.9))
    T._cosmetics(axA, s, "A · Closed cohorts stay (nearly) equal",
                 "curiosity c", "top-10% prestige share")
    axA.set_xlim(-0.03, 1.03)
    axA.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])  # pin: AutoLocator also
    # emitted out-of-view ticks (-0.2, 1.2), which the canvas QA rightly flags
    axA.set_ylim(0.075, 0.24)
    axA.legend(fontsize=s["tick_size"], frameon=False, loc="upper right",
               ncol=1)

    # ---- (b) 400-step lifecycle -----------------------------------------
    axB.axvspan(10, 50, color=T.PALETTE["gray_light"], alpha=0.25, zorder=0)
    axB.text(30, 0.985, "Discovery\nBoom", fontsize=s["tick_size"] - 1,
             color=T.PALETTE["gray"], ha="center", va="top")
    n_b = 0
    for name, hue in LONG:
        sel = lng[lng.landscape == name].sort_values("step")
        axB.plot(sel.step, sel.value, color=hue, lw=1.7, label=name)
        n_b += 1
        term = float(sel[sel.step == 400].value.iloc[0])
        # direct end labels instead of a legend box (which sat on the data)
        axB.text(410, term, f"{name} {term:+.2f}", fontsize=s["tick_size"] - 1,
                 color=hue, va="center")
    axB.axhline(0.0, color=T.PALETTE["gray"], ls="--", lw=1.0)
    # boom opening values, from the drawn series
    for name, hue in (("MGauss1", T.PALETTE["purple"]),
                      ("NGauss", T.PALETTE["orange"])):
        v10 = float(lng[(lng.landscape == name)
                        & (lng.step == 10)].value.iloc[0])
        axB.plot([10], [v10], marker="o", ms=5, color=hue, zorder=5)
    T._cosmetics(axB, s,
                 "B · Alignment booms, then exhausts on peaked terrain",
                 "step t", "Corr(prestige, local merit)")
    axB.set_xlim(0, 545)
    axB.set_xticks([0, 100, 200, 300, 400])
    axB.set_ylim(-0.66, 1.10)

    # lifecycle numbers for the note, computed from the drawn series
    term = {n: float(lng[(lng.landscape == n)
                         & (lng.step == 400)].value.iloc[0]) for n, _ in LONG}
    boom10 = {n: float(lng[(lng.landscape == n)
                           & (lng.step == 10)].value.iloc[0])
              for n in ("MGauss1", "NGauss")}
    rand_late_min = float(lng[(lng.landscape == "random")
                              & (lng.step >= 200)].value.min())

    note = textwrap.fill(
        "source: (a) data/figures_v1/run_number0/2/4/6.csv (swept "
        "homogeneous M0 cohorts, 10 repeats/cell, H = 200); (b) "
        "data/legacy_2025_2026/model_long_{MGauss1,NGauss,random}.csv "
        "(single 400-step runs). Panel (a): the per-landscape mean curves "
        f"span {lo:.3f}-{hi:.3f} for c >= 0.1 (randominit/MultG2 hug "
        f"{float(top[(top.landscape.isin(['randominit', 'MultG2'])) & (top.curiosity >= 0.1)].value.min()):.3f}-"
        f"{float(top[(top.landscape.isin(['randominit', 'MultG2'])) & (top.curiosity >= 0.1)].value.max()):.3f}, "
        f"MultG6 drifts to {float(top[(top.landscape == 'MultG6') & (top.curiosity >= 0.1)].value.max()):.3f}, "
        f"Noisy2 to {float(nois.value.max()):.3f}); the four-landscape mean at each c "
        f"stays in {mean_lo:.3f}-{mean_hi:.3f}. The 0.100 line is exact "
        "equality for N = 20. Panel (b): the two Gaussian-peak landscapes "
        f"open the boom at +{boom10['MGauss1']:.2f}/+{boom10['NGauss']:.2f} (step 10) and end "
        f"at {term['MGauss1']:.2f}/{term['NGauss']:.2f} at step 400; random terrain ends "
        f"positive (+{term['random']:.2f}) and oscillates (late minimum "
        f"{rand_late_min:.2f}). Model-internal, NOT calibrated.", width=176)
    fig.text(0.006, 0.012, note, fontsize=s["note_size"],
             color=T.PALETTE["gray"], va="bottom", linespacing=1.45)
    fig.tight_layout(rect=(0, 0.13, 1, 1))

    # ---- sidecar + QA ----------------------------------------------------
    export = pd.concat([top.assign(panel="a", step=np.nan, landscape_col=top.landscape),
                        lng.assign(panel="b", curiosity=np.nan,
                                   landscape_col=lng.landscape)],
                       ignore_index=True, sort=False)
    export = export.drop(columns=["landscape_col"])
    qa_a = top.copy()
    qa_a["metric"] = "panel_a"
    qa_b = lng.copy()
    qa_b["metric"] = "panel_b"
    report = visual_qa.assert_figure_clean(fig, [
        {"ax": axA, "metric": "panel_a", "expect_lines": n_a,
         "points_per_line": 11, "sidecar": qa_a,
         "ci_cols": ("value", "value")},
        {"ax": axB, "metric": "panel_b", "expect_lines": n_b,
         "points_per_line": 401, "sidecar": qa_b,
         "ci_cols": ("value", "value")},
    ])
    assert n_a == 4 and n_b == 3, f"series counts a={n_a} b={n_b}"

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=s["dpi"])
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf)
    plt.close(fig)
    csv = T.write_figure_data(export, out)
    print(f"QA: {report.get('_total', 0)} failure(s)")
    print(f"Wrote {out} / {pdf} / {csv}")
    return {"png": str(out), "pdf": str(pdf), "data_csv": str(csv)}


if __name__ == "__main__":
    build()
