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

v3 repairs (Team 2, visual QA).  Every ylim is now derived from the plotted
sidecar rows instead of matplotlib autoscaling on the bars, so error-bar
endpoints never touch a spine:

  * Panel A: bottom padding below the -0.0784/-0.0788 caps, top padding so
    the gamma/kappa/scale-trap callout sits in a boxed white-space block
    clear of the Pure marker, of the bars and of the zero line; arm value
    labels lifted above their top caps.
  * Panel B: ylim auto-padded from the c=0.7, gamma=1, M0 bar (-0.1308 cap,
    previously clipped); value labels lifted above the group caps; M0
    swatches hollow+hatched vs M1 solid (one legend key per cell).
  * Panel C: ylim padded so the -0.191 bar and its -0.2643 cap sit clear of
    the bottom spine.

The Q/A contract (``visual_qa.assert_figure_clean``) runs on the rendered
canvas before every ``savefig``; ``verify_figC.check`` re-runs it on a figure
rebuilt in memory from the raw archive.

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
import matplotlib.ticker as mticker  # noqa: E402

from diagram_templates import (  # noqa: E402
    HOUSE_STYLE,
    PALETTE,
    _cosmetics,
    write_figure_data,
)
from turnover_lifecycle_figure import assert_text_fits  # noqa: E402

import visual_qa  # noqa: E402

#: default archive; switch to ``data/typology_v2`` in one line when it lands
DATA_ROOT = REPO / "data" / "typology_v1"
FNAME = "F9_v2_typology_annealing"

BOOT_SEED = 20261115
BOOT_N = 20000
CUR = (0.3, 0.5, 0.7)
GS = (0.0, 0.1)
GAMMAS = (0.0, 1.0)
KAPPAS = (0.0, 1.0)

GAMMA_COLOR = {0.0: PALETTE["purple"], 1.0: PALETTE["green"]}
G_COLOR = {0.0: PALETTE["purple"], 0.1: PALETTE["gray"]}

M_DELTA = "ideal_loss_cum paired delta vs Pure"
M_WEDGE = "relational wedge dr (c_local - c_ideal, final)"
M_COND = "paired delta ideal_loss_cum (gamma 1 - 0)"


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def data_root_path(data_root=None) -> Path:
    """Archive directory holding ``runs.csv`` + ``contrast_table.csv``.

    ``data_root`` may name the archive itself (default ``data/typology_v1``)
    or its ``cells`` subdirectory (accepted so callers can pass a uniform
    ``<archive>/cells`` root across families).
    """
    root = Path(data_root) if data_root is not None else DATA_ROOT
    if not (root / "runs.csv").exists() and (root.parent / "runs.csv").exists():
        root = root.parent
    return root


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
def load_runs(data_root=None) -> pd.DataFrame:
    t = pd.read_csv(data_root_path(data_root) / "runs.csv")
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


def series_rows(t: pd.DataFrame, data_root=None) -> pd.DataFrame:
    rows = []
    root = data_root_path(data_root)
    try:
        src_runs = str((root / "runs.csv").relative_to(REPO))
        src_ct = str((root / "contrast_table.csv").relative_to(REPO))
    except ValueError:  # pragma: no cover - root outside the repo
        src_runs, src_ct = str(root / "runs.csv"), str(root / "contrast_table.csv")

    def add(panel: str, metric: str, statistic: str, value: float,
            ci: tuple[float, float] | None, n: int, note: str, **keys) -> None:
        r = dict(figure=FNAME, panel=panel, metric=metric,
                 statistic=statistic, gamma=np.nan, kappa=np.nan,
                 curiosity=np.nan, g_entry=np.nan, arm="",
                 value=float(value),
                 ci_lo=np.nan if ci is None else ci[0],
                 ci_hi=np.nan if ci is None else ci[1],
                 n=int(n), method="", note=note,
                 source_file=src_runs)
        r.update(keys)
        rows.append(r)

    ilc = "ideal_loss_cum"

    # ---- panel (a): quadrant paired deltas vs Pure (0, 0) -----------------
    arms = [("Pure (0,0)", (0.0, 0.0)), ("+Destruction (g=1)", (1.0, 0.0)),
            ("+Counterforce (k=1)", (0.0, 1.0)), ("Full (g=1,k=1)", (1.0, 1.0))]
    for label, (g, k) in arms:
        if label == "Pure (0,0)":
            add("a", M_DELTA, "arm_delta",
                0.0, None, 120, "baseline arm", arm=label,
                method="reference")
            continue
        d = _paired(t, ilc, (g, k), (0.0, 0.0),
                    match=["curiosity", "g_entry", "seed"])
        ci = _boot_ci(d.to_numpy())
        add("a", M_DELTA, "arm_delta",
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
                add("b", M_WEDGE,
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
        add("c", M_COND, "paired_delta",
            float(sub.mean()), _boot_ci(sub.to_numpy()), len(sub),
            "matched on (kappa, seed)", curiosity=float(c), g_entry=float(g),
            method=f"percentile bootstrap, B={BOOT_N}, seed={BOOT_SEED}")
    add("c", M_COND, "paired_delta",
        float(d_ilc.mean()), _boot_ci(d_ilc.to_numpy()), len(d_ilc),
        "overall reference (commission: -0.0783)",
        method=f"percentile bootstrap, B={BOOT_N}, seed={BOOT_SEED}")

    # ---- cross-check rows against the archive's contrast_table ----------
    ct = pd.read_csv(root / "contrast_table.csv")
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
            "bootstrap", method=src_ct, source_file=src_ct)

    return pd.DataFrame(rows)


# --------------------------------------------------------------- drawing ---
def _no_text_overlap(fig) -> None:
    """Text-vs-text guard (kept alongside visual_qa: fails fast, local)."""
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


def _clean_yticks(ax, nbins: int = 6) -> None:
    """Re-seat the y ticks inside the (re)padded view.

    matplotlib keeps the ticks of the pre-padding view alive: their labels
    stay non-empty, they can sit outside the canvas, and when the old view was
    narrow they are too dense to read.  A fresh MaxNLocator tick set fixes
    both.
    """
    lo, hi = ax.get_ylim()
    ticks = mticker.MaxNLocator(nbins=nbins).tick_values(lo, hi)
    ax.set_yticks([t for t in ticks if lo - 1e-12 <= t <= hi + 1e-12])


def _pad_ylim(ax, spans, *, pad_bottom: float = 0.12, pad_top: float = 0.12,
              extra=()) -> tuple[float, float]:
    """ylim from the plotted (lo, hi) pairs, padded by a fraction of span.

    Values are read from the sidecar rows, never from matplotlib's bar-only
    autoscale -- that is what clipped the F9_v2 panel-B error bar and pinned
    the panel-A/C caps to their spines.
    """
    los = [float(s[0]) for s in spans] + [float(v) for v in extra]
    his = [float(s[1]) for s in spans] + [float(v) for v in extra]
    lo, hi = min(los), max(his)
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0)
    ax.set_ylim(lo - pad_bottom * span, hi + pad_top * span)
    _clean_yticks(ax)
    return ax.get_ylim()


def _cap_label(ax, x: float, y: float, text: str, *, color=None,
               below: bool = False, fontsize=None, ha: str = "center",
               extra_pt: float = 4.0) -> None:
    """Value label anchored `extra_pt` points clear of an error-bar cap."""
    ax.annotate(text, xy=(x, y), xycoords="data",
                xytext=(0, -extra_pt if below else extra_pt),
                textcoords="offset points", ha=ha,
                va="top" if below else "bottom",
                fontsize=fontsize or HOUSE_STYLE["tick_size"],
                color=color or PALETTE["gray"], zorder=6)


def _legend_free(ax, fig) -> bool:
    """True when the axes legend hits neither plotted data nor other text."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    leg = ax.get_legend()
    if leg is None:
        return False
    segs = []
    art = visual_qa.collect_panel_artists(ax)
    for a in art["lines"] + art["bars"] + art["errorbars"] + art["refs"]:
        segs.extend(visual_qa._segments_from_artist(a))
    leg_texts = list(leg.get_texts())
    for t in leg_texts:
        b = visual_qa._bbox(t, r)
        if b is None:
            continue
        if any(visual_qa._seg_hits_box(s, b) for s in segs):
            return False
    others = [t for t in visual_qa._text_artists(fig, ax) if t not in leg_texts]
    for t in others:
        b1 = visual_qa._bbox(t, r)
        if b1 is None:
            continue
        for t2 in leg_texts:
            b2 = visual_qa._bbox(t2, r)
            if b2 is not None and visual_qa._boxes_overlap(b1, b2):
                return False
    return True


def _place_legend(ax, **kw):
    """First candidate legend location whose text hits no data and no text.

    A fixed ``loc`` is not safe across archives: the typology_v2 refresh moves
    the M1 wedge bars to near zero, so the upper-right corner stops being
    empty.  Candidates are tried in house order, the first clean one wins.
    """
    handles, labels = ax.get_legend_handles_labels()
    for loc in ("upper right", "upper left", "lower right", "lower left",
                "center right", "center left", "upper center", "lower center"):
        leg = ax.legend(handles, labels, loc=loc, frameon=False, **kw)
        if _legend_free(ax, ax.figure):
            return leg
        leg.remove()
    raise AssertionError("no collision-free legend placement found")


def _figure(sidecar: pd.DataFrame):
    """Draw F9_v2 from its sidecar; return (fig, visual_qa panel specs)."""
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

    fig = plt.figure(figsize=(13.2, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.18, 1.0, 1.0))
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.885, bottom=0.28,
                        wspace=0.32)

    # ---- Panel A: four-quadrant paired effects --------------------------
    arm_order = ["Pure (0,0)", "+Destruction (g=1)", "+Counterforce (k=1)",
                 "Full (g=1,k=1)"]
    xs = np.arange(len(arm_order))
    spans_a = []
    for x, label in zip(xs, arm_order):
        r = get(M_DELTA, "arm_delta", arm=label)
        if label == "Pure (0,0)":
            ax_a.plot([x], [0.0], marker="o", markersize=7,
                      color=PALETTE["gray"], markerfacecolor="white",
                      markeredgewidth=1.2, zorder=5, label="_nolegend_")
            spans_a.append((0.0, 0.0))
            continue
        col = GAMMA_COLOR[float(r["gamma"])]
        err = [[r["value"] - r["ci_lo"]], [r["ci_hi"] - r["value"]]]
        ax_a.bar(x, r["value"], width=0.62, color=col, alpha=0.85,
                 edgecolor=col, linewidth=1.0, zorder=3)
        ax_a.errorbar(x, r["value"], yerr=err, fmt="none", ecolor="black",
                      elinewidth=1.0, capsize=3.5, zorder=4)
        # label on the far side of the cap: below for negative bars (above
        # the baseline would sit inside the bar rectangle), above otherwise
        if r["value"] < 0:
            _cap_label(ax_a, x, r["ci_lo"], f"{r['value']:+.4f}", below=True,
                       extra_pt=3.0)
        else:
            _cap_label(ax_a, x, r["ci_hi"], f"{r['value']:+.4f}", extra_pt=3.0)
        spans_a.append((r["ci_lo"], r["ci_hi"]))
    d_main = get("ideal_loss_cum destruction_main", "main_effect")
    k_main = get("ideal_loss_cum counterforce_main", "main_effect")
    pure_mean = get("pure_ideal_loss_cum_mean", "level")
    pct = get("destruction_pct_of_pure", "annotation")
    pbar = get("mean_prestige_all_runs", "level")
    ax_a.axhline(0.0, color=PALETTE["gray"], ls="--", lw=1.0, zorder=2)
    ax_a.axhline(d_main["value"], color=PALETTE["green"], ls=":", lw=1.2,
                 zorder=2)
    # left of the first bar column: the only x-band with no bar rectangle
    _cap_label(ax_a, -0.55, d_main["value"],
               f"γ main {d_main['value']:+.4f}", color=PALETTE["green"],
               ha="left", extra_pt=2.5, fontsize=HOUSE_STYLE["note_size"])
    ax_a.set_xticks(xs)
    ax_a.set_xlim(-0.62, len(xs) - 0.5)
    # Two-line display labels (visual QA: single-line arm labels collide at
    # 3-up panel width); lookup keys above stay the ASCII arm strings.
    disp = {"Pure (0,0)": "Pure\n(γ=0, κ=0)",
            "+Destruction (g=1)": "+Destruction\n(γ=1, κ=0)",
            "+Counterforce (k=1)": "+Counterforce\n(γ=0, κ=1)",
            "Full (g=1,k=1)": "Full\n(γ=1, κ=1)"}
    ax_a.set_xticklabels([disp[a] for a in arm_order],
                         fontsize=HOUSE_STYLE["note_size"])
    _cosmetics(ax_a, HOUSE_STYLE,
               "A · Four-quadrant paired effects",
               "arm (vs Pure Distraction (0,0))",
               "paired Δ ideal_loss_cum")
    # top padding leaves this callout a clean box above every artist
    _pad_ylim(ax_a, spans_a, pad_bottom=0.22, pad_top=0.34,
              extra=(0.0, float(d_main["value"])))
    ax_a.text(0.03, 0.975,
              f"γ: {d_main['value']:+.4f} ({pct['value']:+.1f}% of Pure)\n"
              f"κ: {k_main['value']:+.4f} (inert)\n"
              f"scale trap $\\bar{{P}}$ ≈ {pbar['value']:.3f}",
              transform=ax_a.transAxes, va="top", ha="left",
              fontsize=HOUSE_STYLE["note_size"], color=PALETTE["gray"],
              bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                        edgecolor=PALETTE["gray_light"], linewidth=0.6,
                        alpha=0.95))

    # ---- Panel B: relational wedge by M0/M1, curiosity, gamma -----------
    width = 0.17
    xmap = {c: i for i, c in enumerate(CUR)}
    shift = {(0.0, 0.0): -0.27, (0.0, 1.0): -0.09,
             (0.1, 0.0): 0.09, (0.1, 1.0): 0.27}
    spans_b = []
    group_top: dict[float, list[float]] = {}
    for g in GS:
        for gamma in GAMMAS:
            base = np.array([xmap[c] for c in CUR], dtype=float)
            col = GAMMA_COLOR[gamma]
            hollow = g == 0.0
            vals, los, his = [], [], []
            for c in CUR:
                r = get(M_WEDGE, "mean", g_entry=g, curiosity=c, gamma=gamma)
                vals.append(r["value"])
                los.append(r["value"] - r["ci_lo"])
                his.append(r["ci_hi"] - r["value"])
                spans_b.append((r["ci_lo"], r["ci_hi"]))
                group_top.setdefault(c, []).append(r["ci_hi"])
            ax_b.bar(base + shift[(g, gamma)], vals, width=width,
                     color="white" if hollow else col,
                     edgecolor=col, linewidth=1.1, zorder=3,
                     hatch="///" if hollow else None,
                     label=f"M{'0' if hollow else '1'} (g={g:g}) · γ={gamma:.0f}")
            ax_b.errorbar(base + shift[(g, gamma)], vals, yerr=[los, his],
                          fmt="none", ecolor="black", elinewidth=0.9,
                          capsize=2.8, zorder=4)
    for c in CUR:
        r0 = get(M_WEDGE, "mean", g_entry=0.1, curiosity=c, gamma=0.0)
        r1 = get(M_WEDGE, "mean", g_entry=0.1, curiosity=c, gamma=1.0)
        # above the tallest cap of the whole group (any arm), so the tag never
        # lands on a bar or cap whichever arm happens to be tallest
        _cap_label(ax_b, xmap[c], max(group_top[c]),
                   f"{r0['value']:+.3f}→{r1['value']:+.3f}",
                   fontsize=HOUSE_STYLE["tick_size"] - 1, extra_pt=5.0)
    nar = get("paired wedge narrowing (gamma 1 - 0)", "paired_delta")
    ax_b.axhline(0.0, color=PALETTE["gray"], ls="--", lw=1.0, zorder=2)
    ax_b.set_xticks([xmap[c] for c in CUR])
    ax_b.set_xticklabels([f"c = {c}" for c in CUR])
    _cosmetics(ax_b, HOUSE_STYLE,
               "B · Relational wedge Δr (final step)\n"
               f"paired narrowing {nar['value']:+.4f} "
               f"[{nar['ci_lo']:+.4f}, {nar['ci_hi']:+.4f}]",
               "initial curiosity c   (bar pair: M0 left, M1 right)",
               "Δr = c_local − c_ideal (final step)")
    # auto-pad: the c=0.7, gamma=1, M0 cap (-0.1308) used to fall outside
    _pad_ylim(ax_b, spans_b, pad_bottom=0.10, pad_top=0.20, extra=(0.0,))
    _place_legend(ax_b, fontsize=7, ncol=2, handlelength=1.5, handleheight=1.1,
                  columnspacing=1.0, labelspacing=0.35, borderaxespad=0.2)

    # ---- Panel C: paired delta ideal_loss_cum by condition --------------
    conds = [(c, g) for c in CUR for g in GS]
    xs = np.arange(len(conds))
    overall = get(M_COND, "paired_delta", curiosity=np.nan)
    spans_c = []
    for x, (c, g) in zip(xs, conds):
        r = get(M_COND, "paired_delta", curiosity=c, g_entry=g)
        col = G_COLOR[g]
        ax_c.bar(x, r["value"], width=0.62, color=col, alpha=0.85,
                 edgecolor=col, linewidth=1.0, zorder=3)
        ax_c.errorbar(x, r["value"],
                      yerr=[[r["value"] - r["ci_lo"]],
                            [r["ci_hi"] - r["value"]]],
                      fmt="none", ecolor="black", elinewidth=1.0, capsize=3.5,
                      zorder=4)
        if r["value"] < 0:
            _cap_label(ax_c, x, r["ci_lo"], f"{r['value']:+.3f}", below=True,
                       extra_pt=3.0)
        else:
            _cap_label(ax_c, x, r["ci_hi"], f"{r['value']:+.3f}", extra_pt=3.0)
        spans_c.append((r["ci_lo"], r["ci_hi"]))
    ax_c.axhline(0.0, color=PALETTE["gray"], ls="--", lw=1.0, zorder=2)
    ax_c.axhline(overall["value"], color=PALETTE["green"], ls=":", lw=1.2,
                 zorder=2)
    _cap_label(ax_c, len(conds) - 0.55, overall["value"],
               f"overall {overall['value']:+.4f}", color=PALETTE["green"],
               ha="right", below=True, extra_pt=3.0)
    ax_c.set_xticks(xs)
    # short tick labels: the axis label already names c, and six "c = 0.x"
    # labels collide at this panel width
    ax_c.set_xticklabels([f"{c:.1f}\n{'M0' if g == 0.0 else 'M1'}"
                          for c, g in conds])
    _cosmetics(ax_c, HOUSE_STYLE,
               "C · Annealing cuts loss at c ∈ {0.3, 0.5};\nslight friction at c = 0.7",
               "curiosity c × entry regime",
               "paired Δ ideal_loss_cum (γ=1 − γ=0)")
    # bottom padding keeps the -0.191 bar and its -0.2643 cap off the spine
    _pad_ylim(ax_c, spans_c, pad_bottom=0.13, pad_top=0.20,
              extra=(0.0, float(overall["value"])))

    footnote = (
        f"Sources: {S['source_file'].iloc[0]} (480 runs, mgstd5; 24 conditions "
        "x 20 seeds) + contrast_table.csv. Paired deltas match runs on the "
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
    # layout first: legend/callout placement below is measured on the final
    # axes geometry, and a later subplots_adjust would invalidate it

    panels = [
        {"ax": ax_a, "metric": M_DELTA, "expect_lines": 3,
         "points_per_line": 1, "expect_bars": 3, "sidecar": S,
         "headroom": 0.03},
        {"ax": ax_b, "metric": M_WEDGE, "expect_lines": 4,
         "points_per_line": 3, "expect_bars": 12, "sidecar": S,
         "headroom": 0.03},
        {"ax": ax_c, "metric": M_COND, "expect_lines": 6,
         "points_per_line": 1, "expect_bars": 6, "sidecar": S,
         "headroom": 0.03},
    ]
    return fig, panels


def build(outdir: str = "output/report_notebook_v1",
          data_root=None) -> dict:
    t = load_runs(data_root)
    sidecar = series_rows(t, data_root=data_root)

    fig, panels = _figure(sidecar)

    w, h = fig.get_size_inches() * fig.dpi
    assert_text_fits(fig, w, h)
    _no_text_overlap(fig)
    visual_qa.assert_figure_clean(fig, panels)  # v3 hardened contract

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
