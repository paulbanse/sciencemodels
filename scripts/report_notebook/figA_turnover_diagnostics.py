"""Figure A / ``F8_v2`` -- turnover diagnostics (paper sketch 4.3).

Separates what is SUBSTANTIVE (Discovery -> Depletion decay of
Corr(prestige, ideal merit), 55%-79% arrival-gap shrinkage) from the
BIRTH-STEP MEASUREMENT ARTIFACT (``updateIdealMerit()`` runs before
``__generate_new_agents()``, so step-t newborns carry ``prestige=0,
ideal_Merit=0``; amplified by the synchronized ``tau_max = 200`` cohort
replacement at t = 200, 400).  Replaces the comb-like ``F8``.

Four 2x2 panels from the read-only ``data/turnover_v1/`` archive:

  (a) lifecycle trajectory: non-birth-step mean c_ideal per exit mode with
      95% CI ribbons across the 18 cells per mode, Discovery (t <= 80) and
      Depletion (t >= 150) shading, birth-step means as hollow markers and
      dotted drop-lines at the synchronized-replacement steps t = 200, 400.
  (b) entry-rate gradient (g in {0.05, 0.10, 0.20}): settled non-birth mean
      (steps 301-399, solid) vs. terminal step-400 birth snapshot (dashed,
      faint) -- two EXPLICITLY labeled statistics on one axis (commission
      design), exposing how the step-400 gradient reverses the settled one.
  (c) arrival-gap shrinkage: gap_recent_mean_last100 in native units,
      pooled bars per mode with 95% CIs over runs and per-landscape dots
      (mgstd5 diamond / noisypr5 triangle).
  (d) output and concentration in native shares: harvested knowledge share
      alongside top-10% prestige share at step 399 (settled) and step 400
      (cohort shock), with the 0.10 equality reference line.

Provenance rules (commission rule 3): ZERO hardcoded numbers -- every
plotted value, CI and text annotation is computed here from the raw CSVs
and written into ``F8_v2_turnover_diagnostics_data.csv``.  Commission-quoted
values are recomputed; where a quoted rounding does not reproduce (harvested
share of age_hazard, gap_recent of age_hazard) the COMPUTED value is plotted
and the drift is reported to the coordinator.

Run:  MPLBACKEND=Agg python3 scripts/report_notebook/figA_turnover_diagnostics.py
"""
from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from diagram_templates import (  # noqa: E402
    HOUSE_STYLE, PALETTE, _cosmetics, write_figure_data,
)

DATA = REPO / "data" / "turnover_v1"
BASENAME = "F8_v2_turnover_diagnostics"

MODES = ["none", "fixed_span", "age_hazard"]
#: exit-mode hues = the F8 convention (tiers M0/M1/M2 do not appear here)
MODE_COLOR = {"none": PALETTE["purple"], "fixed_span": PALETTE["orange"],
              "age_hazard": PALETTE["green"]}
LANDSCAPES = ["mgstd5", "noisypr5"]
G_LEVELS = [0.05, 0.10, 0.20]
D_SERIES = ["harvested_fin", "top10_step399", "top10_step400"]
#: panel (d) metric hues (distinct strong hues per co-plotted category)
D_COLOR = {"harvested_fin": PALETTE["blue"],
           "top10_step399": PALETTE["gray"],
           "top10_step400": PALETTE["pink"]}

# --- definitional constants (commission); recorded in the sidecar ---------
DISCOVERY_END = 80          # Discovery: t <= 80
DEPLETION_START = 150       # Depletion: t >= 150
BIRTH_SHOCK_STEPS = (200, 400)   # tau_max = 200 synchronized cohort replacement
LATE_START, LATE_END = 301, 399  # settled late window for panel (b)
EQUALITY_TOP10 = 0.10       # top-10% share under perfect equality
Z95 = 1.96                  # house 95% CI multiplier (_CRIT in diagram_templates)

_TRAJ_RE = re.compile(r"cell_([a-z_]+?)_g(\d+)_cur(\d+)_(mgstd5|noisypr5)\.csv")
SRC_TRAJ = "data/turnover_v1/traj/cell_*.csv"
SRC_CELLS = "data/turnover_v1/cells/cell_*.csv"

# stat strings (mirrored verbatim by verify_figA.py)
S_NB = ("mean of c_ideal across cells whose step t is a non-birth step "
        "(floor(t*g) = floor((t-1)*g)) and t not in {200, 400}; 95% CI = "
        "mean +/- 1.96*SE across contributing cells")
S_BIRTH = ("mean of c_ideal across cells whose step t is a birth step "
           "(floor(t*g) > floor((t-1)*g))")
S_B_SETTLED = ("mean of c_ideal over non-birth steps 301-399 pooled over "
               "the 6 (curiosity x landscape) cells of the (mode, g) group")
S_B_SNAP = ("mean of c_ideal at step 400 (birth step) over the 6 "
            "(curiosity x landscape) cells of the (mode, g) group")
S_C = ("mean of gap_recent_mean_last100 over runs; 95% CI = mean +/- "
       "1.96*SE over runs")
S_D = {
    "harvested_fin": "mean of harvested_fin over runs; 95% CI = mean +/- 1.96*SE over runs",
    "top10_step400": "mean of top10_fin (step 400) over runs; 95% CI = mean +/- 1.96*SE over runs",
    "top10_step399": ("mean of traj top10 at step 399 over the 18 cell "
                      "trajectories; 95% CI = mean +/- 1.96*SE across cells"),
}


# ---------------------------------------------------------------------------
# annotation rendering (mirrored verbatim by verify_figA.py)
# ---------------------------------------------------------------------------

def _frag_a(mode: str, b400: float, nb399: float) -> str:
    return f"{mode} {b400:+.3f} vs {nb399:+.3f}"


def _line_a(frags: list[str]) -> str:
    return "step 400 (birth) vs step 399 (settled): " + " · ".join(frags)


def _frag_b(mode: str, v05: float, v20: float) -> str:
    return f"{mode} {v05:+.3f}→{v20:+.3f}"


def _line_b_settled(frags: list[str]) -> str:
    return ("settled non-birth mean 301–399 (g 0.05→0.20): "
            + " · ".join(frags))


def _line_b_snap(frags: list[str]) -> str:
    return "step-400 birth snapshot (g 0.05→0.20): " + " · ".join(frags)


def _label_c(v: float, rel: float, mode: str) -> str:
    return f"{v:.5f}" if mode == "none" else f"{v:.5f}\n[{rel:.1f}%]"


def _note_c(neg: int, tot: int) -> str:
    return f"negative in {neg:,}/{tot:,} runs"


def _label_d(series: str, v: float) -> str:
    return f"{100 * v:.1f}%" if series == "harvested_fin" else f"{v:.3f}"


def _label_discovery() -> str:
    return f"Discovery (t ≤ {DISCOVERY_END})"


def _label_depletion() -> str:
    return f"Depletion (t ≥ {DEPLETION_START})"


def _label_equality() -> str:
    return f"{EQUALITY_TOP10:.2f} equality"


# ---------------------------------------------------------------------------
# load + compute (every number here comes from the raw CSVs)
# ---------------------------------------------------------------------------

def _load_traj() -> pd.DataFrame:
    files = sorted((DATA / "traj").glob("cell_*.csv"))
    if len(files) != 54:
        raise AssertionError(f"traj: expected 54 cell files, found {len(files)}")
    frames = []
    for f in files:
        m = _TRAJ_RE.match(f.name)
        if m is None:
            raise AssertionError(f"traj: unparsable filename {f.name}")
        d = pd.read_csv(f)
        if (len(d) != 400 or int(d["step"].min()) != 1
                or int(d["step"].max()) != 400):
            raise AssertionError(f"{f.name}: expected steps 1..400")
        d["exit_mode"] = m.group(1)
        d["g"] = int(m.group(2)) / 100
        d["curiosity"] = int(m.group(3)) / 10
        d["landscape"] = m.group(4)
        frames.append(d)
    traj = pd.concat(frames, ignore_index=True)
    if traj["cell_tag"].nunique() != 54:
        raise AssertionError("traj: cell_tag not unique across 54 files")
    per_mode = traj.groupby("exit_mode")["cell_tag"].nunique().to_dict()
    if per_mode != {m: 18 for m in MODES}:
        raise AssertionError(f"traj: cells per mode {per_mode} != 18 each")
    t = traj["step"].to_numpy(float)
    g = traj["g"].to_numpy(float)
    traj["is_birth"] = np.floor(t * g) > np.floor((t - 1) * g)
    traj["non_birth"] = (~traj["is_birth"]) & (
        ~traj["step"].isin(BIRTH_SHOCK_STEPS))
    return traj


def _load_cells() -> pd.DataFrame:
    files = sorted((DATA / "cells").glob("cell_*.csv"))
    if len(files) != 54:
        raise AssertionError(f"cells: expected 54 cell files, found {len(files)}")
    frames = []
    for f in files:
        m = _TRAJ_RE.match(f.name)
        if m is None:
            raise AssertionError(f"cells: unparsable filename {f.name}")
        d = pd.read_csv(f)
        if len(d) != 20 or d["seed"].nunique() != 20:
            raise AssertionError(f"{f.name}: expected 20 seeds")
        d["file_g"] = int(m.group(2)) / 100
        frames.append(d)
    cells = pd.concat(frames, ignore_index=True)
    if len(cells) != 1080:
        raise AssertionError(f"cells: expected 1,080 runs, found {len(cells)}")
    per_mode = cells.groupby("exit_mode").size().to_dict()
    if per_mode != {m: 360 for m in MODES}:
        raise AssertionError(f"cells: runs per mode {per_mode} != 360 each")
    if not np.allclose(cells["entry"], cells["file_g"]):
        raise AssertionError("cells: entry column disagrees with filename g")
    return cells.drop(columns=["file_g"])


def _mean_ci(vals: pd.Series) -> tuple[float, float, float, int]:
    n = int(vals.size)
    mean = float(vals.mean())
    if n > 1:
        ci = Z95 * float(vals.std()) / np.sqrt(n)
    else:
        ci = float("nan")
    return mean, mean - ci, mean + ci, n


def compute(traj: pd.DataFrame, cells: pd.DataFrame):
    """All plotted quantities + sidecar rows + rendered annotation strings."""
    rows: list[dict] = []

    def row(**kw):
        rows.append(kw)

    # ---------------- panel (a): non-birth trajectory + birth markers -----
    nb = (traj[traj["non_birth"]]
          .groupby(["exit_mode", "step"])["c_ideal"]
          .agg(mean="mean", sd="std", n="size")
          .reset_index())
    nb["ci"] = Z95 * nb["sd"] / np.sqrt(nb["n"])
    nb["ci_lo"] = nb["mean"] - nb["ci"]
    nb["ci_hi"] = nb["mean"] + nb["ci"]
    for r in nb.itertuples():
        row(panel="a", series="c_ideal_nonbirth_mean", exit_mode=r.exit_mode,
            g=np.nan, landscape="pooled", step=int(r.step), category="",
            value=r.mean, ci_lo=r.ci_lo, ci_hi=r.ci_hi, n=int(r.n),
            stat=S_NB, annotation="", source=SRC_TRAJ)

    birth = (traj[traj["is_birth"]]
             .groupby(["exit_mode", "step"])["c_ideal"]
             .agg(mean="mean", n="size")
             .reset_index())
    for r in birth.itertuples():
        row(panel="a", series="c_ideal_birth_mean", exit_mode=r.exit_mode,
            g=np.nan, landscape="pooled", step=int(r.step), category="",
            value=r.mean, ci_lo=np.nan, ci_hi=np.nan, n=int(r.n),
            stat=S_BIRTH, annotation="", source=SRC_TRAJ)

    # callout values: step-400 birth mean vs step-399 settled mean, per mode
    b400 = birth[birth["step"] == 400].set_index("exit_mode")["mean"]
    nb399 = nb[nb["step"] == 399].set_index("exit_mode")["mean"]
    frags_a = [_frag_a(m, b400[m], nb399[m]) for m in MODES]
    line_a = _line_a(frags_a)
    for m, frag in zip(MODES, frags_a):
        row(panel="a", series="annotation", exit_mode=m, g=np.nan,
            landscape="pooled", step=400, category="callout_fragment",
            value=float(b400[m]), ci_lo=np.nan, ci_hi=np.nan, n=np.nan,
            stat="", annotation=frag, source=SRC_TRAJ)
    row(panel="a", series="annotation", exit_mode="", g=np.nan,
        landscape="", step=np.nan, category="callout_line",
        value=np.nan, ci_lo=np.nan, ci_hi=np.nan, n=np.nan,
        stat="", annotation=line_a, source=SRC_TRAJ)

    # ---------------- panel (b): entry-rate gradient ----------------------
    late = (traj[(traj["step"] >= LATE_START) & (traj["step"] <= LATE_END)
                 & traj["non_birth"]]
            .groupby(["exit_mode", "g"])["c_ideal"]
            .agg(settled="mean", n_settled="size").reset_index())
    snap = (traj[traj["step"] == BIRTH_SHOCK_STEPS[-1]]
            .groupby(["exit_mode", "g"])["c_ideal"]
            .agg(snap="mean", n_snap="size").reset_index())
    b = late.merge(snap, on=["exit_mode", "g"], validate="one_to_one")
    for r in b.itertuples():
        row(panel="b", series="settled_late_nonbirth_mean",
            exit_mode=r.exit_mode, g=float(r.g), landscape="pooled",
            step=np.nan, category="", value=r.settled, ci_lo=np.nan,
            ci_hi=np.nan, n=int(r.n_settled), stat=S_B_SETTLED,
            annotation="", source=SRC_TRAJ)
        row(panel="b", series="step400_birth_snapshot",
            exit_mode=r.exit_mode, g=float(r.g), landscape="pooled",
            step=BIRTH_SHOCK_STEPS[-1], category="", value=r.snap,
            ci_lo=np.nan, ci_hi=np.nan, n=int(r.n_snap), stat=S_B_SNAP,
            annotation="", source=SRC_TRAJ)

    def _ends(series: str) -> list[tuple[float, float]]:
        out = []
        for m in MODES:
            sub = b[(b["exit_mode"] == m)].set_index("g")[series]
            out.append((float(sub.loc[G_LEVELS[0]]), float(sub.loc[G_LEVELS[-1]])))
        return out

    settled_ends = _ends("settled")
    snap_ends = _ends("snap")
    frags_bs = [_frag_b(m, *e) for m, e in zip(MODES, settled_ends)]
    frags_bn = [_frag_b(m, *e) for m, e in zip(MODES, snap_ends)]
    line_bs, line_bn = _line_b_settled(frags_bs), _line_b_snap(frags_bn)
    for m, frag, (v05, v20) in zip(MODES, frags_bs, settled_ends):
        row(panel="b", series="annotation", exit_mode=m, g=np.nan,
            landscape="pooled", step=np.nan, category="box_fragment_settled",
            value=float(v20 - v05), ci_lo=np.nan, ci_hi=np.nan, n=np.nan,
            stat="", annotation=frag, source=SRC_TRAJ)
    for m, frag, (v05, v20) in zip(MODES, frags_bn, snap_ends):
        row(panel="b", series="annotation", exit_mode=m, g=np.nan,
            landscape="pooled", step=np.nan, category="box_fragment_step400",
            value=float(v20 - v05), ci_lo=np.nan, ci_hi=np.nan, n=np.nan,
            stat="", annotation=frag, source=SRC_TRAJ)
    row(panel="b", series="annotation", exit_mode="", g=np.nan,
        landscape="", step=np.nan, category="box_line_settled",
        value=np.nan, ci_lo=np.nan, ci_hi=np.nan, n=np.nan,
        stat="", annotation=line_bs, source=SRC_TRAJ)
    row(panel="b", series="annotation", exit_mode="", g=np.nan,
        landscape="", step=np.nan, category="box_line_step400",
        value=np.nan, ci_lo=np.nan, ci_hi=np.nan, n=np.nan,
        stat="", annotation=line_bn, source=SRC_TRAJ)

    # ---------------- panel (c): arrival-gap shrinkage --------------------
    c_rows = []
    for m in MODES:
        for land in ["pooled"] + LANDSCAPES:
            vals = (cells.loc[cells["exit_mode"] == m, "gap_recent_mean_last100"]
                    if land == "pooled" else
                    cells.loc[(cells["exit_mode"] == m)
                              & (cells["landscape"] == land),
                              "gap_recent_mean_last100"])
            v, lo, hi, n = _mean_ci(vals)
            c_rows.append(dict(exit_mode=m, landscape=land, value=v,
                               ci_lo=lo, ci_hi=hi, n=n))
            row(panel="c", series="gap_recent_mean_last100", exit_mode=m,
                g=np.nan, landscape=land, step=np.nan, category="",
                value=v, ci_lo=lo, ci_hi=hi, n=n, stat=S_C,
                annotation="", source=SRC_CELLS)
    cdf = pd.DataFrame(c_rows)
    base = float(cdf[(cdf["exit_mode"] == "none")
                     & (cdf["landscape"] == "pooled")]["value"].iloc[0])
    c_rel, c_labels = {}, {}
    for m in MODES:
        v = float(cdf[(cdf["exit_mode"] == m)
                      & (cdf["landscape"] == "pooled")]["value"].iloc[0])
        rel = 100.0 * (v / base - 1.0)
        c_rel[m] = rel
        c_labels[m] = _label_c(v, rel, m)
        row(panel="c", series="annotation", exit_mode=m, g=np.nan,
            landscape="pooled", step=np.nan, category="shrinkage_pct",
            value=rel, ci_lo=np.nan, ci_hi=np.nan, n=np.nan, stat="",
            annotation=(f"[{rel:.1f}%]" if m != "none" else ""),
            source=SRC_CELLS)
        row(panel="c", series="annotation", exit_mode=m, g=np.nan,
            landscape="pooled", step=np.nan, category="bar_label",
            value=v, ci_lo=np.nan, ci_hi=np.nan, n=np.nan, stat="",
            annotation=c_labels[m], source=SRC_CELLS)
    neg = int((cells["gap_recent_mean_last100"] < 0).sum())
    tot = int(len(cells))
    note_c = _note_c(neg, tot)
    row(panel="c", series="annotation", exit_mode="", g=np.nan,
        landscape="pooled", step=np.nan, category="negative_runs_note",
        value=float(neg), ci_lo=np.nan, ci_hi=np.nan, n=tot, stat="",
        annotation=note_c, source=SRC_CELLS)

    # ---------------- panel (d): output & concentration -------------------
    d_rows = []
    for m in MODES:
        sub = cells[cells["exit_mode"] == m]
        v, lo, hi, n = _mean_ci(sub["harvested_fin"])
        d_rows.append(dict(exit_mode=m, series="harvested_fin", step=np.nan,
                           value=v, ci_lo=lo, ci_hi=hi, n=n))
        row(panel="d", series="harvested_fin", exit_mode=m, g=np.nan,
            landscape="pooled", step=np.nan, category="", value=v,
            ci_lo=lo, ci_hi=hi, n=n, stat=S_D["harvested_fin"],
            annotation="", source=SRC_CELLS)
        v, lo, hi, n = _mean_ci(sub["top10_fin"])
        d_rows.append(dict(exit_mode=m, series="top10_step400",
                           step=BIRTH_SHOCK_STEPS[-1], value=v, ci_lo=lo,
                           ci_hi=hi, n=n))
        row(panel="d", series="top10_step400", exit_mode=m, g=np.nan,
            landscape="pooled", step=BIRTH_SHOCK_STEPS[-1], category="",
            value=v, ci_lo=lo, ci_hi=hi, n=n, stat=S_D["top10_step400"],
            annotation="", source=SRC_CELLS)
        t399 = traj[(traj["exit_mode"] == m) & (traj["step"] == 399)]["top10"]
        v, lo, hi, n = _mean_ci(t399)
        d_rows.append(dict(exit_mode=m, series="top10_step399", step=399,
                           value=v, ci_lo=lo, ci_hi=hi, n=n))
        row(panel="d", series="top10_step399", exit_mode=m, g=np.nan,
            landscape="pooled", step=399, category="", value=v,
            ci_lo=lo, ci_hi=hi, n=n, stat=S_D["top10_step399"],
            annotation="", source=SRC_TRAJ)
    ddf = pd.DataFrame(d_rows)
    d_labels = {}
    for r in ddf.itertuples():
        lab = _label_d(r.series, r.value)
        d_labels[(r.exit_mode, r.series)] = lab
        row(panel="d", series="annotation", exit_mode=r.exit_mode, g=np.nan,
            landscape="pooled", step=r.step, category="bar_label",
            value=r.value, ci_lo=np.nan, ci_hi=np.nan, n=np.nan, stat="",
            annotation=lab,
            source=(SRC_TRAJ if r.series == "top10_step399" else SRC_CELLS))

    # ---------------- reference rows (definitional constants) ------------
    for cat, val, step, ann in [
        ("discovery_end", DISCOVERY_END, DISCOVERY_END, _label_discovery()),
        ("depletion_start", DEPLETION_START, DEPLETION_START, _label_depletion()),
        ("birth_shock_step", BIRTH_SHOCK_STEPS[0], BIRTH_SHOCK_STEPS[0],
         "dotted drop-line; tau_max = 200 synchronized cohort replacement"),
        ("birth_shock_step", BIRTH_SHOCK_STEPS[1], BIRTH_SHOCK_STEPS[1],
         "dotted drop-line; tau_max = 200 synchronized cohort replacement"),
        ("late_window_start", LATE_START, LATE_START, ""),
        ("late_window_end", LATE_END, LATE_END, ""),
        ("equality_top10", EQUALITY_TOP10, np.nan, _label_equality()),
        ("z95", Z95, np.nan, "95% CI multiplier (house convention)"),
    ]:
        row(panel="ref", series="reference", exit_mode="", g=np.nan,
            landscape="", step=step, category=cat, value=float(val),
            ci_lo=np.nan, ci_hi=np.nan, n=np.nan,
            stat="definitional constant (commission)",
            annotation=ann, source="commission_section_figures_v2.md")

    sidecar = pd.DataFrame(rows, columns=[
        "panel", "series", "exit_mode", "g", "landscape", "step", "category",
        "value", "ci_lo", "ci_hi", "n", "stat", "annotation", "source"])

    frames = dict(a_nb=nb, a_birth=birth, b=b, c=cdf, d=ddf)
    texts = dict(a_frags=frags_a, a_line=line_a,
                 b_line_settled=line_bs, b_line_snap=line_bn,
                 c_rel=c_rel, c_labels=c_labels, c_note=note_c,
                 d_labels=d_labels)
    return frames, texts, sidecar


# ---------------------------------------------------------------------------
# text-fit guards (assert_text_fits pattern, scripts/report_notebook/
# turnover_lifecycle_figure.py) + pairwise overlap audit
# ---------------------------------------------------------------------------

def _text_artists(fig):
    """Every auditable text artist: fig texts, titles, axis labels, legend,
    in-axes annotations (tick labels excluded, as in the source pattern)."""
    out = []
    for t in fig.texts:
        out.append((t, "fig"))
    for i, ax in enumerate(fig.axes):
        for loc, artist in ((f"ax{i}.title", ax.title),
                            (f"ax{i}.left_title", getattr(ax, "_left_title", None)),
                            (f"ax{i}.right_title", getattr(ax, "_right_title", None))):
            if artist is not None:
                out.append((artist, loc))
        out.append((ax.xaxis.label, f"ax{i}.xlabel"))
        out.append((ax.yaxis.label, f"ax{i}.ylabel"))
        if ax.get_legend():
            for t in ax.get_legend().get_texts():
                out.append((t, f"ax{i}.legend"))
        for t in ax.texts:
            out.append((t, f"ax{i}.text"))
    return out


def assert_text_fits(fig, canvas_w: float, canvas_h: float) -> int:
    """Every non-tick text artist must sit inside the canvas (pattern from
    turnover_lifecycle_figure.py; catches clipped glyphs)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    offenders = []
    checked = 0
    for artist, where in _text_artists(fig):
        txt = artist.get_text()
        if not txt.strip():
            continue
        checked += 1
        bb = artist.get_window_extent(renderer=r)
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > canvas_w + 1 or bb.y1 > canvas_h + 1:
            offenders.append((where, txt[:50].replace("\n", " "),
                              round(bb.x1, 1), round(bb.x0, 1)))
    if offenders:
        raise AssertionError(
            "text outside the canvas: "
            + "; ".join(f"{w}: {s!r} x1={x1}" for w, s, x1, _ in offenders))
    return checked


def assert_no_text_overlap(fig) -> int:
    """No two auditable text artists may overlap (positive-area bbox
    intersection beyond a 1e-3 px tolerance)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for artist, where in _text_artists(fig):
        txt = artist.get_text()
        if not txt.strip():
            continue
        items.append((where, txt[:40].replace("\n", " "),
                      artist.get_window_extent(renderer=r)))
    offenders = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            w1, s1, b1 = items[i]
            w2, s2, b2 = items[j]
            dx = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            dy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if dx > 1e-3 and dy > 1e-3:
                offenders.append(f"{w1} {s1!r} X {w2} {s2!r} "
                                 f"(overlap {dx:.1f}x{dy:.1f} px)")
    if offenders:
        raise AssertionError("overlapping text artists: " + "; ".join(offenders))
    return len(items)


# ---------------------------------------------------------------------------
# drawing + export
# ---------------------------------------------------------------------------

def _assert_box_clear_of_data(ax, txt, curves: pd.DataFrame, xcol: str,
                              ycols: list[str], what: str) -> None:
    """The annotation box must stay inside the axes and must NOT occlude
    data: its rendered bbox (converted to data coordinates) has to sit
    above every curve point within its x-range.  Guards against the silent
    failure mode where a white callout hides the lines it explains."""
    fig = ax.figure
    fig.canvas.draw()
    bb = txt.get_window_extent(renderer=fig.canvas.get_renderer())
    inv = ax.transData.inverted()
    (cx0, cy0), (cx1, cy1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    xlo, xhi = ax.get_xlim()
    if cx0 < xlo - 1e-6 or cx1 > xhi + 1e-6:
        raise AssertionError(
            f"{what}: box x [{cx0:.3f}, {cx1:.3f}] outside xlim ({xlo}, {xhi})")
    seg = curves[(curves[xcol] >= cx0) & (curves[xcol] <= cx1)]
    if len(seg):
        y_max = float(max(seg[c].max() for c in ycols))
        if cy0 < y_max + 0.02:
            raise AssertionError(
                f"{what}: box bottom {cy0:.3f} occludes data up to {y_max:.3f}")


def _draw(frames, texts, png: Path, pdf: Path) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    s = HOUSE_STYLE
    nb, birth, bdf, cdf, ddf = (frames["a_nb"], frames["a_birth"],
                                frames["b"], frames["c"], frames["d"])
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), dpi=s["dpi"])
    axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ---------------- (a) lifecycle trajectory --------------------------
    axA.axvspan(1, DISCOVERY_END, color=PALETTE["blue_light"], alpha=0.35,
                linewidth=0, zorder=0)
    axA.axvspan(DEPLETION_START, 400, color=PALETTE["orange_light"],
                alpha=0.35, linewidth=0, zorder=0)
    axA.axhline(0.0, color="black", linewidth=0.9, linestyle="--", zorder=3)
    for step in BIRTH_SHOCK_STEPS:
        axA.axvline(step, color=PALETTE["gray"], linewidth=1.0,
                    linestyle=":", alpha=0.75, zorder=1)
    for m in MODES:
        d = nb[nb["exit_mode"] == m].sort_values("step")
        axA.fill_between(d["step"], d["ci_lo"], d["ci_hi"],
                         color=MODE_COLOR[m], alpha=0.18, linewidth=0, zorder=2)
        axA.plot(d["step"], d["mean"], color=MODE_COLOR[m], linewidth=1.7,
                 zorder=4, label=f"{m} (non-birth mean ± 95% CI)")
        b = birth[birth["exit_mode"] == m].sort_values("step")
        axA.plot(b["step"], b["mean"], linestyle="none", marker="o",
                 markersize=3.6, markerfacecolor="white",
                 markeredgecolor=MODE_COLOR[m], alpha=0.65, zorder=5)
    axA.plot([], [], linestyle="none", marker="o", markersize=3.6,
             markerfacecolor="white", markeredgecolor=PALETTE["gray"],
             label="birth-step mean (open markers)")
    _cosmetics(axA, s,
               "(a) Lifecycle: non-birth trajectory vs. birth-step artifact",
               "step t",
               "Corr(prestige, ideal merit)\n"
               "(per-step mean across cells)")
    axA.set_xlim(1, 400)
    axA.set_ylim(-1.02, 1.45)
    axA.text(DISCOVERY_END / 2, 1.38, _label_discovery(), ha="center",
             va="center", fontsize=s["note_size"], color=PALETTE["gray"])
    axA.text((DEPLETION_START + 400) / 2, 1.38, _label_depletion(), ha="center",
             va="center", fontsize=s["note_size"], color=PALETTE["gray"])
    axA.legend(fontsize=s["tick_size"] - 1, frameon=False, loc="lower left")
    callout = textwrap.fill(
        "Birth-step artifact (measurement, not decay): newborns arrive with "
        "prestige = 0, ideal merit = 0 (updateIdealMerit() runs before "
        "__generate_new_agents()); τ_max = 200 cohort replacement at "
        "t = 200, 400 amplifies the dip.\n"
        + texts["a_line"], width=55)
    callout_txt = axA.text(120, 1.26, callout, ha="left", va="top",
                           fontsize=6.8, zorder=6,
                           bbox=dict(boxstyle="round,pad=0.35", fc="white",
                                     ec=PALETTE["gray_light"], alpha=0.92))
    _assert_box_clear_of_data(axA, callout_txt, nb, "step",
                              ["mean", "ci_hi"], "panel (a) callout")

    # ---------------- (b) entry-rate gradient ---------------------------
    for m in MODES:
        d = bdf[bdf["exit_mode"] == m].sort_values("g")
        axB.plot(d["g"], d["settled"], color=MODE_COLOR[m], linewidth=1.9,
                 marker="o", markersize=5, zorder=4,
                 label=f"{m} — settled (301–399)")
        axB.plot(d["g"], d["snap"], color=MODE_COLOR[m], linewidth=1.5,
                 linestyle="--", alpha=0.65, marker="s", markersize=4.5,
                 markerfacecolor="white", zorder=3,
                 label=f"{m} — step-400 snapshot")
    _cosmetics(axB, s,
               "(b) Entry-rate gradient: settled vs. step-400 snapshot",
               "entry rate g (birth every 1/g steps)",
               "Corr(prestige, ideal merit)\n"
               "(settled 301–399 / step-400 snapshot)")
    axB.set_xticks(G_LEVELS, [f"{g:.2f}" for g in G_LEVELS])
    axB.set_xlim(0.035, 0.235)
    axB.set_ylim(-1.05, 0.80)
    axB.legend(fontsize=s["tick_size"] - 1, frameon=False, loc="lower right",
               ncol=2, handlelength=1.6, columnspacing=0.9)
    box_b = textwrap.fill(texts["b_line_settled"], width=42) + "\n" + \
        textwrap.fill(texts["b_line_snap"], width=42)
    box_b_txt = axB.text(0.115, 0.76, box_b, ha="left", va="top", fontsize=6.9,
                         zorder=6,
                         bbox=dict(boxstyle="round,pad=0.35", fc="white",
                                   ec=PALETTE["gray_light"], alpha=0.92))
    _assert_box_clear_of_data(axB, box_b_txt, bdf, "g",
                              ["settled", "snap"], "panel (b) box")

    # ---------------- (c) arrival-gap shrinkage -------------------------
    vmin = float(cdf["ci_lo"].min())
    span = abs(vmin)
    axC.set_ylim(vmin - 0.45 * span, 0.12 * span)
    handles_c = []
    for i, m in enumerate(MODES):
        x = float(i)
        pooled = cdf[(cdf["exit_mode"] == m)
                     & (cdf["landscape"] == "pooled")].iloc[0]
        axC.bar(x, pooled["value"] - 0.0, width=0.42,
                color=MODE_COLOR[m], zorder=3)
        axC.errorbar([x], [pooled["value"]],
                     yerr=[[pooled["value"] - pooled["ci_lo"]],
                           [pooled["ci_hi"] - pooled["value"]]],
                     fmt="none", ecolor=MODE_COLOR[m], elinewidth=1.2,
                     capsize=3, zorder=4)
        axC.text(x, pooled["ci_lo"] - 0.035 * span, texts["c_labels"][m],
                 ha="center", va="top", fontsize=s["note_size"], zorder=5)
        for land, mk, off in (("mgstd5", "D", -0.32), ("noisypr5", "^", 0.32)):
            sub = cdf[(cdf["exit_mode"] == m)
                      & (cdf["landscape"] == land)].iloc[0]
            axC.errorbar([x + off], [sub["value"]],
                         yerr=[[sub["value"] - sub["ci_lo"]],
                               [sub["ci_hi"] - sub["value"]]],
                         fmt=mk, color=MODE_COLOR[m], markerfacecolor="white",
                         markersize=5, elinewidth=1.1, capsize=3, zorder=4)
    handles_c = [
        Patch(facecolor=PALETTE["gray"], label="pooled (360 runs, bar)"),
        Line2D([], [], linestyle="none", marker="D", color=PALETTE["gray"],
               markerfacecolor="white", markersize=5,
               label="mgstd5 (180 runs)"),
        Line2D([], [], linestyle="none", marker="^", color=PALETTE["gray"],
               markerfacecolor="white", markersize=5,
               label="noisypr5 (180 runs)"),
    ]
    _cosmetics(axC, s, "(c) Arrival-gap shrinkage (gap_recent)",
               "exit mode",
               "gap_recent (mean of last 100 steps)\n"
               "(run means, native units)")
    axC.set_xticks(range(len(MODES)), MODES)
    axC.legend(handles=handles_c, fontsize=s["tick_size"] - 1, frameon=False,
               loc="lower right")
    axC.text(0.02, 0.965, texts["c_note"], transform=axC.transAxes,
             ha="left", va="top", fontsize=s["note_size"], zorder=5)

    # ---------------- (d) output & concentration ------------------------
    axD.axhline(EQUALITY_TOP10, color=PALETTE["gray"], linewidth=1.0,
                linestyle=":", zorder=1)
    axD.text(3.05, EQUALITY_TOP10 + 0.015, _label_equality(), ha="right",
             va="bottom", fontsize=s["note_size"], color=PALETTE["gray"],
             zorder=5)
    width = 0.26
    for i, m in enumerate(MODES):
        for j, ser in enumerate(D_SERIES):
            r = ddf[(ddf["exit_mode"] == m) & (ddf["series"] == ser)].iloc[0]
            x = i + (j - 1) * width
            axD.bar(x, r["value"], width=width, color=D_COLOR[ser], zorder=3)
            axD.errorbar([x], [r["value"]],
                         yerr=[[r["value"] - r["ci_lo"]],
                               [r["ci_hi"] - r["value"]]],
                         fmt="none", ecolor=D_COLOR[ser], elinewidth=1.2,
                         capsize=3, zorder=4)
            axD.text(x, r["ci_hi"] + 0.022, texts["d_labels"][(m, ser)],
                     ha="center", va="bottom", fontsize=7.2,
                     zorder=5)
    _cosmetics(axD, s, "(d) Output and concentration (native shares)",
               "exit mode", "share (native units [0, 1])")
    axD.set_xticks(range(len(MODES)), MODES)
    axD.set_xlim(-0.55, 3.15)
    axD.set_ylim(0.0, 1.22)
    axD.legend(
        handles=[Patch(facecolor=D_COLOR["harvested_fin"],
                       label="harvested knowledge share"),
                 Patch(facecolor=D_COLOR["top10_step399"],
                       label="top-10% prestige share (step 399, settled)"),
                 Patch(facecolor=D_COLOR["top10_step400"],
                       label="top-10% prestige share (step 400, cohort shock)")],
        fontsize=s["tick_size"] - 1, frameon=False, loc="upper left")

    # ---------------- source footnotes (wrapped) ------------------------
    foot = [
        textwrap.fill(
            "Source: data/turnover_v1/traj/cell_*.csv (54 cell trajectories x "
            "400 steps; each trajectory is the mean over its 20 seeds) and "
            "data/turnover_v1/cells/cell_*.csv (54 cells x 20 seeds = 1,080 "
            "runs). Every plotted value, CI and text annotation is computed "
            f"from these CSVs and stored in {BASENAME}_data.csv.", width=172),
        textwrap.fill(
            "Birth steps: floor(t*g) > floor((t-1)*g) (newborns arrive with "
            "prestige = 0, ideal merit = 0 because updateIdealMerit() runs "
            "before __generate_new_agents()); t = 200 and 400 also carry the "
            "synchronized τ_max = 200 cohort replacement. The settled series "
            "in (a)-(b) excludes birth steps and t in {200, 400}.", width=172),
        textwrap.fill(
            "95% CIs = mean +/- 1.96*SE: (a) across the contributing cells "
            "per step (18 per mode where all qualify); (c) across runs (360 "
            "pooled / 180 per landscape); (d) across runs except the step-399 "
            "top-10% share, which is across the 18 cell trajectories. Panel "
            "(b) shows two explicitly labeled statistics on one axis "
            "(commission design): settled non-birth mean over steps 301-399 "
            "vs. terminal step-400 birth snapshot.", width=172),
    ]
    y = 0.105
    for block in foot:
        for line in block.split("\n"):
            fig.text(0.012, y, line, fontsize=s["note_size"],
                     color=PALETTE["gray"])
            y -= 0.0165

    fig.subplots_adjust(left=0.105, right=0.986, top=0.915, bottom=0.195,
                        hspace=0.42, wspace=0.24)

    n_text = assert_text_fits(fig, *fig.canvas.get_width_height())
    assert_no_text_overlap(fig)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=s["dpi"])
    fig.savefig(pdf)
    plt.close(fig)
    return n_text


def build(outdir: str = "output/report_notebook_v1") -> dict:
    """Build Figure A (F8_v2): PNG + PDF + _data.csv; returns the paths."""
    out = Path(outdir)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{BASENAME}.png"
    pdf = out / f"{BASENAME}.pdf"

    traj = _load_traj()
    cells = _load_cells()
    frames, texts, sidecar = compute(traj, cells)
    n_text = _draw(frames, texts, png, pdf)
    data_csv = write_figure_data(sidecar, png)
    print(f"figure A: {png} | {pdf} | {data_csv}")
    print(f"text artists checked: {n_text} (fit + overlap)")
    print(f"sidecar rows: {len(sidecar)}")
    return {"png": str(png), "pdf": str(pdf), "data_csv": str(data_csv)}


def main() -> int:
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
