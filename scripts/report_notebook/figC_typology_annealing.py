"""F9_v2 -- typology: crowding-noise destruction as simulated annealing (5.2 Part B).

Figure C of the section-figures-v2 commission, replacing
``F9_simulated_annealing_crowding_noise`` (whose Panel B title falsely read
"negative in all six conditions" -- the c=0.7 condition is positive).  From
``data/typology_v1/runs.csv`` (480 runs, mgstd5; 2 gamma x 2 kappa x 3
curiosity x 2 g_entry x 20 seeds) and ``data/typology_v1/contrast_table.csv``:

  (a) Four-quadrant paired effects on ideal_loss_cum vs Pure Distraction (0,0)
      (20,000-resample bootstrap 95% CIs, seed 20261115).
  (b) Relational wedge dr = c_local - c_ideal (final step) = -wedge_fin,
      M0 (g=0) vs M1 (g=0.1) x curiosity, gamma=0 vs gamma=1 paired means.
  (c) Paired delta ideal_loss_cum (gamma=1 minus gamma=0) across c x g,
      40 matched pairs per condition, with the accurate panel title.

Sign convention (commission sec. 1 + coordinator contract): runs.csv stores
``wedge_fin = c_ideal_fin - c_local_fin`` (asserted here on load), so the
plotted relational wedge is ``dr = -wedge_fin`` (positive = proxy failure).

Run:  MPLBACKEND=Agg python3 scripts/report_notebook/figC_typology_annealing.py
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from diagram_templates import (  # noqa: E402
    HOUSE_STYLE,
    PALETTE,
    _cosmetics,
    write_figure_data,
)
from turnover_lifecycle_figure import assert_text_fits  # noqa: E402

DATA = REPO / "data" / "typology_v1"
FNAME = "F9_v2_typology_annealing"

BOOT_SEED = 20261115
BOOT_N = 20000
CUR = (0.3, 0.5, 0.7)
GS = (0.0, 0.1)
GAMMAS = (0.0, 1.0)
KAPPAS = (0.0, 1.0)

GAMMA_COLOR = {0.0: PALETTE["purple"], 1.0: PALETTE["green"]}
G_COLOR = {0.0: PALETTE["purple"], 0.1: PALETTE["gray"]}


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _boot_ci(x: np.ndarray) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean: 20k resamples, seed 20261115.

    A fresh generator per call, so every interval is reproducible
    independently of computation order (verifier reimplements this).
    """
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(x), size=(BOOT_N, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


# ----------------------------------------------------------------- data ----
def load_runs() -> pd.DataFrame:
    t = pd.read_csv(DATA / "runs.csv")
    _require(len(t) == 480, f"480 runs expected, got {len(t)}")
    counts = t.groupby(["gamma", "kappa", "curiosity", "g_entry"]).size()
    _require(len(counts) == 24 and (counts == 20).all(),
             f"24 conditions x 20 seeds expected: {counts.to_dict()}")
    ident = (t["wedge_fin"] - (t["c_ideal_fin"] - t["c_local_fin"])).abs().max()
    _require(float(ident) < 1e-12,
             f"wedge_fin identity broken, max|err|={ident}")
    return t


def _paired(t: pd.DataFrame, col: str, a: tuple, z: tuple,
            match: list[str]) -> pd.Series:
    """Matched-pair delta (arm `a` minus arm `z`) on `col`, keyed by `match`
    (the design axes held fixed in the contrast).  A ``None`` in an arm's
    ``(gamma, kappa)`` slot means the axis is free and carried by `match`."""
    ga, gz = ("gamma", "kappa")

    def _sel(arm: tuple) -> np.ndarray:
        s = np.ones(len(t), dtype=bool)
        if arm[0] is not None:
            s &= (t[ga] == arm[0]).to_numpy()
        if arm[1] is not None:
            s &= (t[gz] == arm[1]).to_numpy()
        return s

    pa = t.loc[_sel(a)].set_index(match)[col]
    pz = t.loc[_sel(z)].set_index(match)[col]
    _require(len(pa) == len(pz) and len(pa) > 0,
             f"pairing size mismatch: {len(pa)} vs {len(pz)}")
    _require(set(pa.index) == set(pz.index), "pairing index sets differ")
    d = pa - pz
    _require(len(d) == len(pa), "pairing lost rows in alignment")
    return d


def series_rows(t: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(panel: str, metric: str, statistic: str, value: float,
            ci: tuple[float, float] | None, n: int, note: str, **keys) -> None:
        r = dict(figure=FNAME, panel=panel, metric=metric,
                 statistic=statistic, gamma=np.nan, kappa=np.nan,
                 curiosity=np.nan, g_entry=np.nan, arm="",
                 value=float(value),
                 ci_lo=np.nan if ci is None else ci[0],
                 ci_hi=np.nan if ci is None else ci[1],
                 n=int(n), method="", note=note,
                 source_file="data/typology_v1/runs.csv")
        r.update(keys)
        rows.append(r)

    ilc = "ideal_loss_cum"

    # ---- panel (a): quadrant paired deltas vs Pure (0, 0) -----------------
    arms = [("Pure (0,0)", (0.0, 0.0)), ("+Destruction (g=1)", (1.0, 0.0)),
            ("+Counterforce (k=1)", (0.0, 1.0)), ("Full (g=1,k=1)", (1.0, 1.0))]
    for label, (g, k) in arms:
        if label == "Pure (0,0)":
            add("a", "ideal_loss_cum paired delta vs Pure", "arm_delta",
                0.0, None, 120, "baseline arm", arm=label,
                method="reference")
            continue
        d = _paired(t, ilc, (g, k), (0.0, 0.0),
                    match=["curiosity", "g_entry", "seed"])
        ci = _boot_ci(d.to_numpy())
        add("a", "ideal_loss_cum paired delta vs Pure", "arm_delta",
            float(d.mean()), ci, len(d),
            "paired by (curiosity, g_entry, seed)",
            arm=label, gamma=g, kappa=k,
            method=f"percentile bootstrap, B={BOOT_N}, seed={BOOT_SEED}")
    # main effects (kappa/gamma matched) -- the -0.0783 / -0.0006 pins
    d_dest_full = _paired(t, ilc, (1.0, None), (0.0, None),
                          match=["kappa", "curiosity", "g_entry", "seed"])
    d_cf_full = _paired(t, ilc, (None, 1.0), (None, 0.0),
                        match=["gamma", "curiosity", "g_entry", "seed"])
    add("a", "ideal_loss_cum destruction_main", "main_effect",
        float(d_dest_full.mean()), _boot_ci(d_dest_full.to_numpy()),
        len(d_dest_full), "gamma 0->1, kappa matched, paired",
        method=f"percentile bootstrap, B={BOOT_N}, seed={BOOT_SEED}")
    add("a", "ideal_loss_cum counterforce_main", "main_effect",
        float(d_cf_full.mean()), _boot_ci(d_cf_full.to_numpy()),
        len(d_cf_full), "kappa 0->1, gamma matched, paired",
        method=f"percentile bootstrap, B={BOOT_N}, seed={BOOT_SEED}")
    pure_mean = float(t[(t["gamma"] == 0) & (t["kappa"] == 0)][ilc].mean())
    add("a", "pure_ideal_loss_cum_mean", "level", pure_mean, None, 120,
        "mean ideal_loss_cum of Pure (0,0)", arm="Pure (0,0)")
    add("a", "destruction_pct_of_pure", "annotation",
        100.0 * float(d_dest_full.mean()) / pure_mean, None,
        len(d_dest_full), "destruction_main in % of Pure mean "
        "(commission: -16.3%)")
    add("a", "mean_prestige_all_runs", "level",
        float(t["mean_prestige"].mean()), None, 480,
        "scale trap P-bar (commission: ~0.025)")

    # ---- panel (b): relational wedge dr = -wedge_fin (final step) --------
    t = t.assign(dr=-t["wedge_fin"])
    for g in GS:
        for c in CUR:
            for gamma in GAMMAS:
                sub = t[(t["g_entry"] == g) & (t["curiosity"] == c)
                        & (t["gamma"] == gamma)]["dr"]
                _require(len(sub) == 40, f"panel b n=40, got {len(sub)}")
                add("b", "relational wedge dr (c_local - c_ideal, final)",
                    "mean", float(sub.mean()), _boot_ci(sub.to_numpy()),
                    len(sub), f"gamma={gamma}, pooled over kappa and seeds",
                    g_entry=g, curiosity=c, gamma=gamma,
                    method=f"percentile bootstrap, B={BOOT_N}, seed={BOOT_SEED}")
    d_dr = _paired(t, "dr", (1.0, None), (0.0, None),
                   match=["kappa", "curiosity", "g_entry", "seed"])
    add("b", "paired wedge narrowing (gamma 1 - 0)", "paired_delta",
        float(d_dr.mean()), _boot_ci(d_dr.to_numpy()), len(d_dr),
        "dr(gamma=1) - dr(gamma=0), matched on (kappa, curiosity, "
        "g_entry, seed); commission quotes -0.0316, CI [-0.0532, -0.0101] "
        "(contrast_table method)",
        method=f"percentile bootstrap, B={BOOT_N}, seed={BOOT_SEED}")

    # ---- panel (c): paired delta ideal_loss_cum by condition -------------
    d_ilc = _paired(t, ilc, (1.0, None), (0.0, None),
                    match=["kappa", "curiosity", "g_entry", "seed"])
    for (c, g), sub in d_ilc.groupby(["curiosity", "g_entry"]):
        add("c", "paired delta ideal_loss_cum (gamma 1 - 0)", "paired_delta",
            float(sub.mean()), _boot_ci(sub.to_numpy()), len(sub),
            "matched on (kappa, seed)", curiosity=float(c), g_entry=float(g),
            method=f"percentile bootstrap, B={BOOT_N}, seed={BOOT_SEED}")
    add("c", "paired delta ideal_loss_cum (gamma 1 - 0)", "paired_delta",
        float(d_ilc.mean()), _boot_ci(d_ilc.to_numpy()), len(d_ilc),
        "overall reference (commission: -0.0783)",
        method=f"percentile bootstrap, B={BOOT_N}, seed={BOOT_SEED}")

    # ---- cross-check rows against the archive's contrast_table ----------
    ct = pd.read_csv(DATA / "contrast_table.csv")
    for measure, contrast, key in (
            ("ideal_loss_cum", "destruction_main (gamma 0->1, kappa matched)",
             "ideal_loss_cum destruction_main"),
            ("ideal_loss_cum", "counterforce_main (kappa 0->1, gamma matched)",
             "ideal_loss_cum counterforce_main"),
            ("wedge_fin", "destruction_main (gamma 0->1, kappa matched)",
             "paired wedge narrowing (gamma 1 - 0)")):
        r = ct[(ct["measure"] == measure) & (ct["contrast"] == contrast)]
        _require(len(r) == 1, f"contrast_table row missing: {measure} {contrast}")
        r = r.iloc[0]
        val = float(r["mean_delta"])
        if key.startswith("paired wedge"):
            val = -val  # wedge_fin is c_ideal - c_local; dr flips the sign
        add("footnote", key + " (contrast_table cross-check)", "cross_check",
            val, (float(r["ci_lo"]), float(r["ci_hi"])), int(r["n_pairs"]),
            "archive contrast_table.csv mean_delta"
            + (" (sign-flipped to dr)" if key.startswith("paired wedge") else "")
            + "; its CI uses the archive's own method, not the mandated "
            "bootstrap", method="data/typology_v1/contrast_table.csv",
            source_file="data/typology_v1/contrast_table.csv")

    return pd.DataFrame(rows)


# --------------------------------------------------------------- drawing ---
def _no_text_overlap(fig) -> None:
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []
    for t in fig.texts:
        if t.get_text().strip():
            boxes.append((t.get_window_extent(renderer=r), t.get_text()))
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_text().strip():
                boxes.append((t.get_window_extent(renderer=r), t.get_text()))
    for leg in fig.legends:
        for t in leg.get_texts():
            if t.get_text().strip():
                boxes.append((t.get_window_extent(renderer=r), t.get_text()))
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                if t.get_text().strip():
                    boxes.append((t.get_window_extent(renderer=r), t.get_text()))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (b1, s1), (b2, s2) = boxes[i], boxes[j]
            ov = (min(b1.x1, b2.x1) - max(b1.x0, b2.x0),
                  min(b1.y1, b2.y1) - max(b1.y0, b2.y0))
            if ov[0] > 4 and ov[1] > 4:
                raise AssertionError(
                    f"overlapping text artists: {s1[:30]!r} vs {s2[:30]!r}")


def build(outdir: str = "output/report_notebook_v1") -> dict:
    t = load_runs()
    sidecar = series_rows(t)
    S = sidecar

    def get(metric: str, statistic: str, **keys) -> pd.Series:
        m = (S["metric"] == metric) & (S["statistic"] == statistic)
        for k, v in keys.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                m &= S[k].isna()
            else:
                m &= S[k] == v
        sub = S[m]
        _require(len(sub) == 1, f"lookup {metric} {keys}: {len(sub)} rows")
        return sub.iloc[0]

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(13.2, 4.6))

    # ---- Panel A: four-quadrant paired effects --------------------------
    arm_order = ["Pure (0,0)", "+Destruction (g=1)", "+Counterforce (k=1)",
                 "Full (g=1,k=1)"]
    xs = np.arange(len(arm_order))
    for x, label in zip(xs, arm_order):
        r = get("ideal_loss_cum paired delta vs Pure", "arm_delta", arm=label)
        if label == "Pure (0,0)":
            ax_a.plot([x], [0.0], marker="o", markersize=7, color=PALETTE["gray"],
                      markerfacecolor="white", markeredgewidth=1.2, zorder=5)
            continue
        col = GAMMA_COLOR[float(r["gamma"])]
        err = [[r["value"] - r["ci_lo"]], [r["ci_hi"] - r["value"]]]
        ax_a.bar(x, r["value"], width=0.62, color=col, alpha=0.85,
                 edgecolor=col, linewidth=1.0, zorder=3)
        ax_a.errorbar(x, r["value"], yerr=err, fmt="none", ecolor="black",
                      elinewidth=1.0, capsize=3.5, zorder=4)
        ax_a.text(x, min(r["value"], r["ci_lo"]) - 0.004, f"{r['value']:+.4f}",
                  ha="center", va="top", fontsize=HOUSE_STYLE["tick_size"],
                  color=PALETTE["gray"], zorder=5)
    d_main = get("ideal_loss_cum destruction_main", "main_effect")
    k_main = get("ideal_loss_cum counterforce_main", "main_effect")
    pure_mean = get("pure_ideal_loss_cum_mean", "level")
    pct = get("destruction_pct_of_pure", "annotation")
    pbar = get("mean_prestige_all_runs", "level")
    ax_a.axhline(0.0, color=PALETTE["gray"], ls="--", lw=1.0, zorder=2)
    ax_a.axhline(d_main["value"], color=PALETTE["green"], ls=":", lw=1.2, zorder=2)
    ax_a.text(0.98, d_main["value"], f"γ main effect {d_main['value']:+.4f}",
              transform=ax_a.get_yaxis_transform(), ha="right", va="bottom",
              fontsize=HOUSE_STYLE["tick_size"], color=PALETTE["green"])
    ax_a.set_xticks(xs)
    # Two-line display labels (visual QA: single-line arm labels collide at
    # 3-up panel width); lookup keys above stay the ASCII arm strings.
    disp = {"Pure (0,0)": "Pure\n(γ=0, κ=0)",
            "+Destruction (g=1)": "+Destruction\n(γ=1, κ=0)",
            "+Counterforce (k=1)": "+Counterforce\n(γ=0, κ=1)",
            "Full (g=1,k=1)": "Full\n(γ=1, κ=1)"}
    ax_a.set_xticklabels([disp[a] for a in arm_order],
                         fontsize=HOUSE_STYLE["tick_size"] - 1)
    _cosmetics(ax_a, HOUSE_STYLE,
               "A · Four-quadrant paired effects",
               "arm (vs Pure Distraction (0,0))",
               "paired Δ ideal_loss_cum")
    ax_a.text(0.03, 0.985,
              f"γ: {d_main['value']:+.4f} ({pct['value']:+.1f}% of Pure)\n"
              f"κ: {k_main['value']:+.4f} (inert)\n"
              f"scale trap $\\bar{{P}}$ ≈ {pbar['value']:.3f}",
              transform=ax_a.transAxes, va="top", ha="left",
              fontsize=HOUSE_STYLE["note_size"], color=PALETTE["gray"])

    # ---- Panel B: relational wedge by M0/M1, curiosity, gamma -----------
    width = 0.17
    xmap = {c: i for i, c in enumerate(CUR)}
    for g in GS:
        for gamma in GAMMAS:
            base = np.array([xmap[c] for c in CUR], dtype=float)
            shift = (-0.27 if g == 0.0 else 0.09) + (0.0 if gamma == 0.0 else 0.18)
            col = GAMMA_COLOR[gamma]
            vals, los, his = [], [], []
            for c in CUR:
                r = get("relational wedge dr (c_local - c_ideal, final)",
                        "mean", g_entry=g, curiosity=c, gamma=gamma)
                vals.append(r["value"])
                los.append(r["value"] - r["ci_lo"])
                his.append(r["ci_hi"] - r["value"])
            ax_b.bar(base + shift, vals, width=width, color=col,
                     alpha=1.0 if gamma == 1.0 else 0.55,
                     edgecolor=col, linewidth=1.0, zorder=3,
                     hatch=None if gamma == 1.0 else "///",
                     label=f"γ={gamma:.0f}, " +
                           ("M1 (g=0.1)" if g == 0.1 else "M0 (g=0)"))
            ax_b.errorbar(base + shift, vals, yerr=[los, his], fmt="none",
                          ecolor="black", elinewidth=0.9, capsize=2.8, zorder=4)
    for c in CUR:
        r0 = get("relational wedge dr (c_local - c_ideal, final)", "mean",
                 g_entry=0.1, curiosity=c, gamma=0.0)
        r1 = get("relational wedge dr (c_local - c_ideal, final)", "mean",
                 g_entry=0.1, curiosity=c, gamma=1.0)
        ax_b.text(xmap[c] + 0.19, max(r0["value"], r1["value"]) + 0.022,
                  f"+{r0['value']:.3f}→+{r1['value']:.3f}", ha="center",
                  va="bottom", fontsize=HOUSE_STYLE["tick_size"] - 1,
                  color=PALETTE["gray"])
    nar = get("paired wedge narrowing (gamma 1 - 0)", "paired_delta")
    ax_b.axhline(0.0, color=PALETTE["gray"], ls="--", lw=1.0, zorder=2)
    ax_b.set_xticks([xmap[c] for c in CUR])
    ax_b.set_xticklabels([f"c = {c}" for c in CUR])
    ax_b.set_ylim(-0.06, 0.32)
    _cosmetics(ax_b, HOUSE_STYLE,
               "B · Relational wedge Δr (final step)\n"
               f"paired narrowing {nar['value']:+.4f} "
               f"[{nar['ci_lo']:+.4f}, {nar['ci_hi']:+.4f}]",
               "initial curiosity c   (bar pair: M0 left, M1 right)",
               "Δr = c_local − c_ideal (final step)")
    ax_b.legend(fontsize=7, frameon=False,
                loc="upper right", ncol=2)

    # ---- Panel C: paired delta ideal_loss_cum by condition ---------------
    conds = [(c, g) for c in CUR for g in GS]
    xs = np.arange(len(conds))
    overall = get("paired delta ideal_loss_cum (gamma 1 - 0)", "paired_delta",
                  curiosity=np.nan)
    for x, (c, g) in zip(xs, conds):
        r = get("paired delta ideal_loss_cum (gamma 1 - 0)", "paired_delta",
                curiosity=c, g_entry=g)
        col = G_COLOR[g]
        ax_c.bar(x, r["value"], width=0.62, color=col, alpha=0.85,
                 edgecolor=col, linewidth=1.0, zorder=3)
        ax_c.errorbar(x, r["value"],
                      yerr=[[r["value"] - r["ci_lo"]],
                            [r["ci_hi"] - r["value"]]],
                      fmt="none", ecolor="black", elinewidth=1.0, capsize=3.5,
                      zorder=4)
        if r["value"] < 0:
            y_lab, va = min(r["value"], r["ci_lo"]) - 0.004, "top"
        else:
            y_lab, va = max(r["value"], r["ci_hi"]) + 0.004, "bottom"
        ax_c.text(x, y_lab, f"{r['value']:+.3f}", ha="center", va=va,
                  fontsize=HOUSE_STYLE["tick_size"], color=PALETTE["gray"],
                  zorder=5)
    ax_c.axhline(0.0, color=PALETTE["gray"], ls="--", lw=1.0, zorder=2)
    ax_c.axhline(overall["value"], color=PALETTE["green"], ls=":", lw=1.2,
                 zorder=2)
    ax_c.text(0.98, overall["value"], f"overall {overall['value']:+.4f}",
              transform=ax_c.get_yaxis_transform(), ha="right", va="top",
              fontsize=HOUSE_STYLE["tick_size"], color=PALETTE["green"])
    ax_c.set_xticks(xs)
    ax_c.set_xticklabels([f"c = {c:.1f}\n{'M0' if g == 0.0 else 'M1'}"
                          for c, g in conds])
    _cosmetics(ax_c, HOUSE_STYLE,
               "C · Annealing cuts loss at c ∈ {0.3, 0.5};\nslight friction at c = 0.7",
               "curiosity c × entry regime",
               "paired Δ ideal_loss_cum (γ=1 − γ=0)")

    footnote = (
        "Sources: data/typology_v1/runs.csv (480 runs, mgstd5; 24 conditions x "
        "20 seeds) + contrast_table.csv. Paired deltas match runs on the "
        "remaining design axes; 95% CIs are 20,000-resample percentile "
        "bootstraps of the paired mean (seed 20261115). Sign: Δr = c_local − "
        "c_ideal = −wedge_fin (wedge_fin = c_ideal_fin − c_local_fin asserted "
        "identical on load). "
        f"Measured: γ main effect {d_main['value']:+.4f} "
        f"({pct['value']:+.1f}% of Pure mean), κ {k_main['value']:+.4f}; "
        f"paired wedge narrowing {nar['value']:+.4f} "
        f"[{nar['ci_lo']:+.4f}, {nar['ci_hi']:+.4f}] (contrast_table quotes "
        "[-0.0532, -0.0101] from its own CI method). Panel C title states the "
        "computed signs. Model-internal, not calibrated."
    )
    fig.text(0.01, 0.005, "\n".join(textwrap.fill(x, width=175)
                                    for x in footnote.split("\n")),
             fontsize=HOUSE_STYLE["note_size"], color=PALETTE["gray"],
             va="bottom")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.88, bottom=0.235,
                        wspace=0.32)

    w, h = fig.get_size_inches() * fig.dpi
    assert_text_fits(fig, w, h)
    _no_text_overlap(fig)

    out = (REPO / outdir) if not Path(outdir).is_absolute() else Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{FNAME}.png"
    pdf = out / f"{FNAME}.pdf"
    fig.savefig(png, dpi=HOUSE_STYLE["dpi"])
    fig.savefig(pdf)
    plt.close(fig)
    csv_path = write_figure_data(sidecar, png)
    return {"png": str(png), "pdf": str(pdf), "data_csv": str(csv_path)}


if __name__ == "__main__":
    print(build())
