"""FA2_v2 - Birth-tile decomposition of the entry-induced alignment drop (ADR-0003 clock).

Section 4.2 (birth-tile falsification) of the paper sketch.  Two rows:

  row 1  alignment level against the entry rate g, one panel per curiosity c,
         for the exact-spawn and random-spawn arms plus the no-entry M0 level;
  row 2  the decomposition shares of the M0 -> exact gap, one bar pair per
         (c, g) cell:

             drop             = M0 - exact
             starvation share = (random - exact) / drop
             herding share    = (M0 - random)    / drop
             identity: starvation + herding == 100% (asserted)

PRIMARY MEASURE.  ``c_ideal_tm`` (time-mean Corr(prestige, ideal merit)).
The pre-ADR figure used the single terminal step; on the ADR-0003 clock that
step is far too dispersed at n = 20 seeds per cell (seed bootstrap 95% CIs on
the shares span hundreds of percentage points), so the figure reports the
time-mean level and carries the terminal (``c_ideal_fin``) and settled
(``c_ideal_late``, mean of the last 100 steps) variants in the sidecar.

ARCHIVE NAMING.  On the ADR-0003 clock the exit rule is ``fixed_span``
(pre-ADR name ``none``) and the entry grid is g in {0.125, 0.25, 0.5}
(pre-ADR: the single rate g = 0.1); one step is 0.5 yr, tau_max = 80,
gamma = 0.933.

Sources (read-only):
  data/interventions_adr0003/cells/intA_*.csv  spawn sub-arm: curiosity x g x
      spawn in {exact, random, ring1}, 20 seeds/cell, mgstd5, refill off,
      fixed_span.  ring1 belongs to a later refill round and is excluded, as in
      the pre-ADR figure.  Spawn levels are rebuilt from the cells' ``spawn``
      column; each cell's design is cross-checked against
      ``state/<cell_tag>.json`` (there is no ``tables/arm_a_spawn.csv``).
  data/fig1_xi_adr0003/cells/fig1_tier-M0_*.csv  the M0 tier (g = 0, exit
      none, spawn exact, refill off) at the same curiosity values.  The
      interventions archive has no g = 0 control row, so M0 is reconstructed
      from this tier; the model configuration is identical (the ADR wave
      runner passes only ``initial_curiosity``: xi = 1 - c is derived inside
      the agent, so the ``xi`` column is bookkeeping and the curiosity match
      is the experiment match).

``build(outdir, data_root)``: ``data_root`` defaults to the ADR-0003 archive;
pass ``data_root="data/interventions_v2"`` to regenerate the pre-ADR figure
from ``tables/arm_a_spawn.csv`` + ``runs.csv`` (that archive has one entry
rate, g = 0.1, and only the terminal measure, so row 2 collapses to one
cluster group and the settled variant is absent).

Model-internal, NOT calibrated.

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
        scripts/report_notebook/fig_FA2_v2_arm_a_decomposition.py
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

BASENAME = "FA2_v2_drop_decomposition"
DATA_ROOT = REPO / "data" / "interventions_adr0003"
M0_ROOT = REPO / "data" / "fig1_xi_adr0003"

CURIOSITIES = (0.3, 0.5, 0.7)
G_LEVELS = (0.125, 0.25, 0.5)
SPAWNS = ("exact", "random")
MEASURES = ("c_ideal_tm", "c_ideal_late", "c_ideal_fin")
PRIMARY = "c_ideal_tm"
MEASURE_LABEL = {"c_ideal_tm": "time-mean",
                 "c_ideal_late": "settled (last 100 steps)",
                 "c_ideal_fin": "terminal step"}
Z95 = 1.96
N_BOOT = 4000
BOOT_SEED = 20261126


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def data_root_path(data_root=None) -> Path:
    """Spawn-arm archive root (default: ``data/interventions_adr0003``)."""
    root = Path(data_root) if data_root is not None else DATA_ROOT
    return root if root.is_absolute() else (REPO / root)


def m0_root_path(m0_root=None) -> Path:
    """Figure-1 archive root holding the M0 (no-entry) tier."""
    root = Path(m0_root) if m0_root is not None else M0_ROOT
    return root if root.is_absolute() else (REPO / root)


def _mean_ci(v) -> tuple[float, float, float, int]:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return m, m - Z95 * se, m + Z95 * se, n


# ---------------------------------------------------------------------------
# ADR-0003 cell layout
# ---------------------------------------------------------------------------
def _read_new_spawn(root: Path) -> pd.DataFrame:
    files = sorted((root / "cells").glob("intA_*.csv"))
    _require(len(files) == 27,
             f"{root}/cells: 27 intA cells expected (3 c x 3 g x 3 spawn), "
             f"found {len(files)}")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d[d.refill_mode == "off"]
    d = d[d.landscape == "mgstd5"]
    d = d[d.spawn.isin(SPAWNS)].copy()
    _require(len(d) == len(CURIOSITIES) * len(G_LEVELS) * len(SPAWNS) * 20,
             f"spawn sub-arm: {len(d)} seed rows, expected 18 cells x 20")
    _require(sorted(d.curiosity.unique().tolist()) == list(CURIOSITIES),
             f"curiosity grid {sorted(d.curiosity.unique())}")
    _require(sorted(d.g.unique().tolist()) == list(G_LEVELS),
             f"entry grid {sorted(d.g.unique())}")
    _require(sorted(d.exit_mode.unique().tolist()) == ["fixed_span"],
             f"exit mode {sorted(d.exit_mode.unique())}")
    return d


def _check_design(root: Path, cells: pd.DataFrame) -> int:
    """Cross-check every cell file against its ``state/<cell_tag>.json``."""
    import json
    seen = 0
    for tag in sorted(cells.cell_tag.unique()):
        p = root / "state" / f"{tag}.json"
        _require(p.is_file(), f"missing design record {p}")
        spec = json.loads(p.read_text())["spec"]
        sub = cells[cells.cell_tag == tag]
        _require(float(spec["g"]) == float(sub.g.iloc[0]), f"{tag}: g mismatch")
        _require(spec["spawn"] == sub.spawn.iloc[0], f"{tag}: spawn mismatch")
        _require(spec["exit"] == sub.exit_mode.iloc[0], f"{tag}: exit mismatch")
        _require(spec["land"] == sub.landscape.iloc[0], f"{tag}: land mismatch")
        _require(int(spec["tau_max"]) == int(sub.tau_max.iloc[0]),
                 f"{tag}: tau_max mismatch")
        seen += 1
    return seen


def _read_new_m0(root: Path) -> pd.DataFrame:
    files = sorted((root / "cells").glob("fig1_tier-M0_*.csv"))
    _require(bool(files), f"{root}/cells: no fig1_tier-M0_*.csv")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d[d.curiosity.isin(CURIOSITIES)].copy()
    _require(len(d) == len(CURIOSITIES) * 20,
             f"M0 tier: {len(d)} rows at c in {CURIOSITIES}, expected 60")
    _require((d.g == 0.0).all(), "M0 tier is not at g = 0")
    _require((d.exit_mode == "none").all(), "M0 tier is not exit-free")
    return d


def _levels_from_seeds(d: pd.DataFrame, keys) -> pd.DataFrame:
    rows = []
    for key, sub in d.groupby(list(keys), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(keys, key))
        for meas in MEASURES:
            m, lo, hi, n = _mean_ci(sub[meas])
            rows.append({**rec, "measure": meas, "value": m,
                         "ci_lo": lo, "ci_hi": hi, "n": n,
                         "ci_flag": "seed", "seed_values": None,
                         "seed_key": (tuple(sorted(rec.items())), meas)})
    return pd.DataFrame(rows)


def _shares_from_seeds(levels: pd.DataFrame, m0: pd.DataFrame) -> pd.DataFrame:
    """Shares per (curiosity, g) with a seed bootstrap CI on each share."""
    rng = np.random.default_rng(BOOT_SEED)
    present = [m for m in MEASURES
               if m in set(levels.measure) and m in set(m0.measure)]
    _require(PRIMARY in present, f"primary measure {PRIMARY} absent from archive")
    rows = []
    for meas in present:
        lv = levels[levels.measure == meas]
        ref = m0[m0.measure == meas]
        for c in CURIOSITIES:
            m0c = ref[ref.curiosity == c]
            _require(len(m0c) == 1, f"M0 reference missing for c={c}, {meas}")
            a = float(m0c.value.iloc[0])
            m0_seeds = m0c.seed_values.iloc[0]
            for g in sorted(lv.g.unique()):
                sub = lv[(lv.curiosity == c) & (lv.g == g)]
                _require(len(sub) == len(SPAWNS), f"c={c} g={g}: spawn levels")
                e = float(sub[sub.spawn == "exact"].value.iloc[0])
                r = float(sub[sub.spawn == "random"].value.iloc[0])
                drop = a - e
                _require(abs(drop) > 1e-9,
                         f"c={c} g={g}: degenerate drop ({meas})")
                bs, bh = [], []
                lo_, hi_ = float("nan"), float("nan")
                ea = sub[sub.spawn == "exact"].seed_values.iloc[0]
                ra = sub[sub.spawn == "random"].seed_values.iloc[0]
                if ea is not None and ra is not None and m0_seeds is not None:
                    for _ in range(N_BOOT):
                        aa = rng.choice(m0_seeds, ea.size)
                        ee = rng.choice(ea, ea.size)
                        rr = rng.choice(ra, ea.size)
                        dd = aa.mean() - ee.mean()
                        if abs(dd) > 1e-9:
                            bs.append(100.0 * (rr.mean() - ee.mean()) / dd)
                            bh.append(100.0 * (aa.mean() - rr.mean()) / dd)
                if bs:
                    lo_, hi_ = (float(np.percentile(bs, 2.5)),
                                float(np.percentile(bs, 97.5)))
                rows.append({
                    "curiosity": c, "g": g, "measure": meas, "m0": a,
                    "exact": e, "random": r, "drop": drop,
                    "starvation": 100.0 * (r - e) / drop,
                    "herding": 100.0 * (a - r) / drop,
                    "starv_lo": lo_, "starv_hi": hi_,
                    "herd_lo": 100.0 - hi_ if np.isfinite(hi_) else hi_,
                    "herd_hi": 100.0 - lo_ if np.isfinite(lo_) else lo_,
                })
    out = pd.DataFrame(rows)
    _require(np.allclose(out.starvation + out.herding, 100.0, atol=1e-9),
             "decomposition identity violated (shares do not sum to 100)")
    return out


# ---------------------------------------------------------------------------
# pre-ADR layout (tables/arm_a_spawn.csv + runs.csv), selectable via data_root
# ---------------------------------------------------------------------------
def _read_legacy(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tbl = root / "tables" / "arm_a_spawn.csv"
    runs = root / "runs.csv"
    _require(tbl.is_file() and runs.is_file(),
             f"{root}: neither an ADR-0003 cells/ archive nor a pre-ADR "
             f"tables/arm_a_spawn.csv + runs.csv")
    sp = pd.read_csv(tbl)
    sp = sp[(sp.landscape == "mgstd5") & (sp.r_spawn.isin(SPAWNS))]
    _require(len(sp) == len(CURIOSITIES) * len(SPAWNS),
             f"{tbl}: {len(sp)} mgstd5 spawn rows, expected 6")
    g = float(sp.g.iloc[0]) if "g" in sp.columns else 0.1
    rows = []
    for r in sp.itertuples():
        for meas, col in (("c_ideal_tm", "c_ideal_tm_post"),
                          ("c_ideal_fin", "c_ideal_fin_post")):
            if col not in sp.columns:
                continue
            val = float(getattr(r, col))
            rows.append({"curiosity": float(r.curiosity), "g": g,
                         "spawn": r.r_spawn, "measure": meas, "value": val,
                         "ci_lo": val, "ci_hi": val, "n": int(r.n),
                         "ci_flag": "table (no per-seed CI)",
                         "seed_values": None, "seed_key": None})
    levels = pd.DataFrame(rows)
    m0r = pd.read_csv(runs)
    m0r = m0r[(m0r.arm == "C") & (m0r.agent_generation_time == 0)
              & (m0r.landscape == "mgstd5") & (m0r.refill_mode == "off")]
    _require(len(m0r) == len(CURIOSITIES) * 20,
             f"M0 rows: {len(m0r)}, expected 60")
    mrows = []
    for c, sub in m0r.groupby("initial_curiosity"):
        for meas, col in (("c_ideal_fin", "corr_prestige_ideal_merit_final"),
                          ("c_ideal_tm", "Mean_of_Corr_prestige_ideal_merit")):
            if col not in sub.columns:
                continue
            m, lo, hi, n = _mean_ci(sub[col])
            mrows.append({"curiosity": float(c), "measure": meas, "value": m,
                          "ci_lo": lo, "ci_hi": hi, "n": n,
                          "ci_flag": "seed", "seed_values": None,
                          "seed_key": None})
    return levels, pd.DataFrame(mrows)


def compute(data_root=None, m0_root=None) -> dict:
    root = data_root_path(data_root)
    if sorted((root / "cells").glob("intA_*.csv")):
        cells = _read_new_spawn(root)
        n_design = _check_design(root, cells)
        m0cells = _read_new_m0(m0_root_path(m0_root))
        levels = _levels_from_seeds(cells, ("curiosity", "g", "spawn"))
        m0 = _levels_from_seeds(m0cells, ("curiosity",))
        for k, r in levels.iterrows():
            sel = cells[(cells.curiosity == r.curiosity) & (cells.g == r.g)
                        & (cells.spawn == r.spawn)]
            levels.at[k, "seed_values"] = sel[r.measure].to_numpy(float)
        for k, r in m0.iterrows():
            sel = m0cells[m0cells.curiosity == r.curiosity]
            m0.at[k, "seed_values"] = sel[r.measure].to_numpy(float)
        layout = "adr0003"
        design_note = (f"{n_design} cell design records checked against "
                       "state/<cell_tag>.json")
        src_spawn = f"{root.relative_to(REPO)}/cells/intA_*.csv"
        src_m0 = (f"{m0_root_path(m0_root).relative_to(REPO)}"
                  "/cells/fig1_tier-M0_*.csv")
        horizon = sorted(cells.horizon.unique().tolist())
        tau_max = sorted(cells.tau_max.unique().tolist())
    else:
        levels, m0 = _read_legacy(root)
        layout = "pre-adr"
        design_note = "pre-ADR spawn table (no state/ records)"
        src_spawn = f"{root.relative_to(REPO)}/tables/arm_a_spawn.csv"
        src_m0 = f"{root.relative_to(REPO)}/runs.csv (arm C, g = 0, refill off)"
        horizon = [int(pd.read_csv(root / "runs.csv").steps.max())]
        tau_max = [80]
    shares = _shares_from_seeds(levels, m0)
    return {"levels": levels, "m0": m0, "shares": shares, "layout": layout,
            "design_note": design_note, "src_spawn": src_spawn,
            "src_m0": src_m0, "root": root,
            "horizon": horizon, "tau_max": tau_max}


# ---------------------------------------------------------------------------
# sidecar
# ---------------------------------------------------------------------------
def _sidecar(fr: dict) -> pd.DataFrame:
    rows = []

    def add(panel, metric, statistic, series, **kw):
        base = {"figure": BASENAME, "panel": panel, "metric": metric,
                "statistic": statistic, "series": series,
                "curiosity": None, "g": None, "spawn": None, "value": None,
                "ci_lo": None, "ci_hi": None, "n": None, "source_file": None,
                "note": None}
        base.update(kw)
        rows.append(base)

    lv = fr["levels"][fr["levels"].measure == PRIMARY]
    for r in lv.itertuples():
        add("levels", f"f8_levels_c{r.curiosity:.1f}", "mean", r.spawn,
            curiosity=r.curiosity, g=r.g, spawn=r.spawn, value=r.value,
            ci_lo=r.ci_lo, ci_hi=r.ci_hi, n=r.n, source_file=fr["src_spawn"],
            note=f"{MEASURE_LABEL[PRIMARY]} alignment, {r.n} seed rows")
    for r in fr["m0"][fr["m0"].measure == PRIMARY].itertuples():
        add("levels_ref", "f8_m0_reference", "mean", "M0", curiosity=r.curiosity,
            g=0.0, value=r.value, ci_lo=r.ci_lo, ci_hi=r.ci_hi, n=r.n,
            source_file=fr["src_m0"],
            note="no-entry baseline, drawn as a horizontal reference line")
    bounds = {"starvation": ("starv_lo", "starv_hi"),
              "herding": ("herd_lo", "herd_hi")}
    sh = fr["shares"][fr["shares"].measure == PRIMARY]
    for r in sh.itertuples():
        for comp in ("starvation", "herding"):
            blo, bhi = bounds[comp]
            add("shares", "f8_shares", "share_of_drop_pct", comp,
                curiosity=r.curiosity, g=r.g, value=float(getattr(r, comp)),
                ci_lo=float(getattr(r, comp)), ci_hi=float(getattr(r, comp)),
                n=20, source_file=fr["src_spawn"],
                note=f"drop = M0 - exact = {r.drop:+.4f}; seed-bootstrap 95% "
                     f"CI [{getattr(r, blo):+.0f}, {getattr(r, bhi):+.0f}] pp "
                     f"(in sidecar only, not drawn)")
    for r in sh.itertuples():
        add("shares", "f8_drop", "m0_minus_exact", "drop", curiosity=r.curiosity,
            g=r.g, value=float(r.drop), ci_lo=float(r.drop),
            ci_hi=float(r.drop), n=20, source_file=fr["src_spawn"],
            note="denominator of the two shares (M0 - exact)")
        for comp, blo, bhi in (("starvation", r.starv_lo, r.starv_hi),
                               ("herding", r.herd_lo, r.herd_hi)):
            add("shares", "f8_share_boot", "seed_bootstrap_95ci", comp,
                curiosity=r.curiosity, g=r.g,
                value=float(getattr(r, comp)), ci_lo=float(blo),
                ci_hi=float(bhi), n=20, source_file=fr["src_spawn"],
                note="4000-draw seed bootstrap, independent draws per arm, "
                     "not drawn (ratio unstable where the drop is small)")
    for meas in MEASURES:
        if meas == PRIMARY or meas not in set(fr["shares"].measure):
            continue
        for r in fr["shares"][fr["shares"].measure == meas].itertuples():
            for comp in ("starvation", "herding"):
                add("robustness", f"f8_shares_{meas}", "share_of_drop_pct", comp,
                    curiosity=r.curiosity, g=r.g,
                    value=float(getattr(r, comp)),
                    ci_lo=float(getattr(r, comp)), ci_hi=float(getattr(r, comp)),
                    n=20, source_file=fr["src_spawn"],
                    note=f"{MEASURE_LABEL[meas]} variant (not plotted); "
                         f"drop = {r.drop:+.4f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def build(outdir="output/report_notebook_v1", data_root=None,
          m0_root=None) -> dict:
    fr = compute(data_root=data_root, m0_root=m0_root)
    lv = fr["levels"][fr["levels"].measure == PRIMARY]
    m0 = fr["m0"][fr["m0"].measure == PRIMARY].set_index("curiosity")
    sh = fr["shares"][fr["shares"].measure == PRIMARY]
    g_levels = sorted(lv.g.unique())
    layout = fr["layout"]
    s = {**T.HOUSE_STYLE, "figsize": (9.6, 7.2)}
    fig = plt.figure(figsize=s["figsize"])
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0],
                             left=0.078, right=0.988, top=0.865, bottom=0.295,
                             hspace=0.60)
    top = outer[0].subgridspec(1, len(CURIOSITIES), wspace=0.20)

    # ---- row 1: levels ----------------------------------------------------
    ax_lv = []
    pos = np.arange(len(g_levels))
    lo_all = np.nanmin([lv.value.min(), lv.ci_lo.min(), m0.value.min()])
    hi_all = np.nanmax([lv.value.max(), lv.ci_hi.max(), m0.value.max()])
    span = hi_all - lo_all
    ylim1 = (lo_all - 0.10 * span, hi_all + 0.10 * span)
    for i, c in enumerate(CURIOSITIES):
        ax = fig.add_subplot(top[0, i])
        ax_lv.append(ax)
        v_end = {sp: float(lv[(lv.curiosity == c) & (lv.spawn == sp)]
                           .sort_values("g").value.iloc[-1])
                 for sp in SPAWNS}
        room = {sp: ((ylim1[1] - v_end[sp]) / (ylim1[1] - ylim1[0]),
                     (v_end[sp] - ylim1[0]) / (ylim1[1] - ylim1[0]))
                for sp in SPAWNS}
        dy_lbl = {"random": 11.0 if room["random"][0] >= 0.22 else -11.0,
                  "exact": -11.0 if room["exact"][1] >= 0.22 else 11.0}
        if (dy_lbl["exact"] == dy_lbl["random"]
                and abs(v_end["exact"] - v_end["random"]) < 0.06 * span):
            dy_lbl["exact"] = -dy_lbl["exact"]
        for spawn, colr, mk in (("exact", T.PALETTE["blue"], "o"),
                                ("random", T.PALETTE["orange"], "s")):
            sub = (lv[(lv.curiosity == c) & (lv.spawn == spawn)]
                   .sort_values("g").reset_index(drop=True))
            _require(len(sub) == len(g_levels), f"c={c} {spawn}: g grid")
            if (sub.ci_flag == "seed").all():
                ax.errorbar(pos, sub.value,
                            yerr=[sub.value - sub.ci_lo, sub.ci_hi - sub.value],
                            fmt=mk + "-", color=colr, markersize=5,
                            linewidth=1.8, capsize=3, elinewidth=1.2)
            else:
                ax.plot(pos, sub.value, mk + "-", color=colr, markersize=5,
                        linewidth=1.8, label=f"{spawn} spawn")
            # end label: anchored away from the frame and, for a close pair
            # (e.g. a one-entry-rate archive), pushed apart vertically
            dy = dy_lbl[spawn]
            ax.annotate(f"{spawn}\n{sub.value.iloc[-1]:+.3f}",
                        xy=(pos[-1], sub.value.iloc[-1]), xytext=(9, dy),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=s["tick_size"] - 2, color=colr,
                        linespacing=1.15)
        ax.axhline(m0.loc[c, "value"], color=T.PALETTE["gray"], linestyle="--",
                   linewidth=1.3)
        ax.text(-0.40, m0.loc[c, "value"] + 0.012 * span, "M0 (g = 0)",
                fontsize=s["tick_size"] - 2, color=T.PALETTE["gray"],
                va="bottom", ha="left")
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{g:g}" for g in g_levels])
        ax.set_xlim(-0.45, len(g_levels) - 1 + 0.95)
        ax.set_ylim(*ylim1)
        if i:
            ax.set_yticklabels([])
        T._cosmetics(ax, s, f"c = {c:.1f} | M0 = {m0.loc[c, 'value']:+.3f}",
                     "entry rate g (newcomers per step)", "")
        if i == 0:
            ax.set_ylabel("time-mean Corr(prestige, ideal merit)",
                          fontsize=s["label_size"])

    # ---- row 2: shares (one cluster per (c, g) cell) -----------------------
    ax_sh = fig.add_subplot(outer[1])
    cells = [(c, g) for g in g_levels for c in CURIOSITIES]
    x = np.arange(len(cells), dtype=float)
    w = 0.38
    star = [float(sh[(sh.curiosity == c) & (sh.g == g)].starvation.iloc[0])
            for c, g in cells]
    herd = [float(sh[(sh.curiosity == c) & (sh.g == g)].herding.iloc[0])
            for c, g in cells]
    b1 = ax_sh.bar(x - w / 2, star, width=w, color=T.PALETTE["pink"],
                   label="starvation share")
    b2 = ax_sh.bar(x + w / 2, herd, width=w, color=T.PALETTE["purple"],
                   label="herding share")
    for bars, vals in ((b1, star), (b2, herd)):
        for rect, v in zip(bars, vals):
            ax_sh.text(rect.get_x() + rect.get_width() / 2,
                       v + (7 if v >= 0 else -7), f"{v:+.0f}",
                       ha="center", va="bottom" if v >= 0 else "top",
                       fontsize=s["tick_size"] - 1.5)
    ax_sh.axhline(0, color=T.PALETTE["gray"], linewidth=1.0)
    ax_sh.axhline(100, color=T.PALETTE["gray"], linewidth=1.0, linestyle=":")
    lo2, hi2 = min(star + herd), max(star + herd)
    span2 = hi2 - lo2
    ax_sh.set_ylim(lo2 - 0.16 * span2, hi2 + 0.20 * span2)
    ax_sh.set_xticks(x)
    ax_sh.set_xticklabels([f"c = {c:.1f}\n g = {g:g}" for c, g in cells],
                          fontsize=s["tick_size"] - 2)
    for k in range(len(g_levels) - 1):
        ax_sh.axvline(k * len(CURIOSITIES) + len(CURIOSITIES) - 0.5,
                      color=T.PALETTE["gray_light"], linewidth=0.8)
    ax_sh.set_xlim(-0.7, len(cells) - 0.3)
    T._cosmetics(ax_sh, s, "", "curiosity c and entry rate g",
                 "share of the M0 \u2192 exact drop (%)")
    fig.legend([b1, b2], ["starvation share", "herding share"],
               loc="center", bbox_to_anchor=(0.5, 0.515), ncol=2,
               fontsize=s["tick_size"] - 1, frameon=False, handlelength=1.6)

    starv_lo, starv_hi = float(min(star)), float(max(star))
    fig.suptitle(
        "Dispersal now helps: starvation share rises from "
        f"{starv_lo:+.0f}% to {starv_hi:+.0f}% as entry speeds up",
        fontsize=s["title_size"] + 0.5, y=0.982)

    drops = ", ".join(f"c={r.curiosity:.1f}/g={r.g:g}: {r.drop:+.3f}"
                      for r in sh.itertuples())
    if layout == "adr0003":
        recon = ("M0 reconstructed from the Figure-1 M0 tier (the "
                 "interventions archive has no g = 0 control); spawn levels "
                 "rebuilt from the cells' spawn column, " + fr["design_note"])
    else:
        recon = ("pre-ADR spawn table plus arm-C g = 0 runs; that archive "
                 "carries only the terminal measure")
    note = textwrap.fill(
        f"source: {fr['src_spawn']} (exact vs random spawn, mgstd5, refill "
        "off, 20 seeds/cell) + " + fr["src_m0"] + ". " + recon +
        ". ADR-0003 pins: 1 step = 0.5 yr, entry age 25, tau_max = "
        f"{fr['tau_max'][0]}, gamma = 0.933, g in "
        "{" + ",".join(f"{g:g}" for g in g_levels) + "}, "
        f"H = {fr['horizon'][0]}. Levels = "
        f"{MEASURE_LABEL[PRIMARY]} Corr(prestige, ideal merit), 95% CI over "
        "seeds; shares = (random-exact)/drop, (M0-random)/drop, identity "
        f"100% asserted; drop per (c, g): {drops}. Terminal and settled "
        "variants plus seed-bootstrap share CIs in the sidecar (the ratio is "
        "unstable where the drop is small). Model-internal, NOT calibrated.",
        width=150)
    fig.text(0.006, 0.012, note, fontsize=s["note_size"],
             color=T.PALETTE["gray"], va="bottom", linespacing=1.45)

    sidecar = _sidecar(fr)
    panels = [{"ax": a, "metric": f"f8_levels_c{c:.1f}", "expect_lines": 2,
               "points_per_line": len(g_levels), "sidecar": sidecar}
              for a, c in zip(ax_lv, CURIOSITIES)]
    panels.append({"ax": ax_sh, "metric": "f8_shares",
                   "expect_bars": 2 * len(cells), "sidecar": sidecar,
                   "ci_cols": ("value", "value")})
    report = visual_qa.assert_figure_clean(fig, panels)

    out = Path(outdir)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{BASENAME}.png"
    pdf = out / f"{BASENAME}.pdf"
    fig.savefig(png, dpi=s["dpi"])
    fig.savefig(pdf)
    plt.close(fig)
    csv_path = T.write_figure_data(sidecar, png)
    print(f"QA: {report.get('_total', 0)} failure(s)")
    print(f"Wrote {png} / {pdf} / {csv_path}")
    print(sh.round(3).to_string(index=False))
    return {"png": str(png), "pdf": str(pdf), "data_csv": str(csv_path),
            "qa": report, "layout": layout, "sidecar_rows": len(sidecar)}


if __name__ == "__main__":
    build()
