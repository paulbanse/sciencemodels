"""F8_v3 -- turnover diagnostics after ``BUG-0006`` (paper sketch 4.3).

Successor of ``figA_turnover_diagnostics.py`` (``F8_v2``, turnover_v1).  The
old figure separated the SUBSTANTIVE Discovery -> Depletion decay from the
BIRTH-STEP MEASUREMENT ARTIFACT (``updateIdealMerit()`` ran before
``__generate_new_agents()``, so step-*t* newborns entered the data collector
with ``prestige = 0, ideal_Merit = 0``; amplified by the synchronized
``tau_max = 200`` cohort replacement at ``t = 200, 400``).  ``BUG-0006`` is
now fixed (newborn ``ideal_Merit = NaN``, filtered out of
``correlation_reporter``), so the artifact is GONE and the figure is
re-drawn on the post-fix archive ``data/turnover_v2/`` with FOUR exit modes
(``age_hazard`` is split into a synchronized and a staggered arm).

Four panels, every number computed here from the raw CSVs:

  (a) post-fix lifecycle trajectories: per-step mean ``c_ideal`` across the
      18 cells of each exit mode over steps 1-400, Discovery (t <= 80) and
      Depletion (t >= 150) shading, dotted drop-lines at the synchronized
      cohort-replacement steps t = 200 / 400 -- and a compact inset GHOST of
      the pre-fix ``age_hazard`` series (read-only ``data/turnover_v1/``)
      showing the removed -0.568 spike at t = 400.
  (b) settled alignment over steps 301-400 vs entry rate g in
      {0.05, 0.10, 0.20} for all four modes (all steps count: with the fix
      in place birth steps are valid measurements, not artifacts).
  (c) arrival-gap shrinkage: ``gap_recent_mean_last100`` pooled bars per mode
      with per-landscape markers (mgstd5 diamond / noisypr5 triangle) in
      native units, percent change against ``none`` on the bar labels.
  (d) output and concentration in native shares: harvested knowledge share
      alongside the top-10% prestige share at step 399 (settled) and step 400
      (synchronized cohort shock), showing that staggering removes the
      ``age_hazard_sync`` step-400 concentration spike.

Sampling frame (must match ``data/bug0006_remeasure_v2/tables/``):
``data/turnover_v2/runs.csv`` holds 1,520 runs, but the two hazard arms carry
two extra curiosity cells (0.1 / 0.9) that ``none`` and ``fixed_span`` do not
have.  Every panel therefore uses the COMMON SUPPORT curiosity in
{0.3, 0.5, 0.7}: 4 modes x 3 g x 3 curiosity x 2 landscapes x 20 seeds =
1,440 runs and 54 trajectory cells (18 per mode).  The 80 excluded runs are
named in the footnote and in the sidecar notes.

Provenance rules (commission rule 3): ZERO hardcoded numbers -- every plotted
value, CI and text annotation is computed here from the raw CSVs and written
into ``F8_v3_turnover_diagnostics_data.csv``.  Commission-quoted values
(-55.0%, -82.9%, 1,440/1,440 runs, top10(400) 0.409 -> 0.270, pre-fix
-0.568) are recomputed and reported against the computed value.

Run:  MPLBACKEND=Agg python3 scripts/report_notebook/figA_v3_turnover_postbug.py
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from diagram_templates import (  # noqa: E402
    HOUSE_STYLE,
    PALETTE,
    _cosmetics,
    write_figure_data,
)
import visual_qa  # noqa: E402

FNAME = "F8_v3_turnover_diagnostics"
BASENAME = FNAME

POST_ROOT = REPO / "data" / "turnover_v2"
PRE_ROOT = REPO / "data" / "turnover_v1"

MODES = ["none", "fixed_span", "age_hazard_sync", "age_hazard_staggered"]
#: exit-mode hues (F8 convention: modes are hues, tiers do not appear here)
MODE_COLOR = {
    "none": PALETTE["purple"],
    "fixed_span": PALETTE["orange"],
    "age_hazard_sync": PALETTE["green"],
    "age_hazard_staggered": PALETTE["blue"],
}
MODE_LABEL = {
    "none": "none — no retirement",
    "fixed_span": "fixed_span — fixed span",
    "age_hazard_sync": "age_hazard_sync — synchronized",
    "age_hazard_staggered": "age_hazard_staggered — staggered",
}
D_SERIES = ["harvested_fin", "top10_step399", "top10_step400"]
#: panel (d) metric hues (distinct strong hues per co-plotted category)
D_COLOR = {"harvested_fin": PALETTE["blue"],
           "top10_step399": PALETTE["gray"],
           "top10_step400": PALETTE["pink"]}
LANDSCAPES = ["mgstd5", "noisypr5"]
LAND_MARK = {"mgstd5": "D", "noisypr5": "^"}
LAND_OFF = {"mgstd5": -0.30, "noisypr5": 0.30}
G_LEVELS = [0.05, 0.10, 0.20]
#: common-support curiosity cells shared by all four exit modes
COMMON_CURIOSITY = (0.3, 0.5, 0.7)

#: the pre-fix archive calls the synchronized arm simply ``age_hazard``
PRE_MODE = "age_hazard"
GHOST_MODE = "age_hazard_sync"

# --- definitional constants (commission); recorded in the sidecar ---------
DISCOVERY_END = 80           # Discovery: t <= 80
DEPLETION_START = 150        # Depletion: t >= 150
BIRTH_SHOCK_STEPS = (200, 400)   # tau_max = 200 synchronized cohort replacement
LATE_START, LATE_END = 301, 400  # settled window, panel (b) and (d)
GHOST_START = DEPLETION_START - 5    # inset window (removed pre-fix spike)
EQUALITY_TOP10 = 0.10        # top-10% share under perfect equality
Z95 = 1.96                   # house 95% CI multiplier

_TRAJ_RE = re.compile(r"cell_([a-z_]+?)_g(\d+)_cur(\d+)_(mgstd5|noisypr5)\.csv")
POST_TRAJ_GLOB = "data/turnover_v2/traj/cell_*.csv"
POST_RUNS = "data/turnover_v2/runs.csv"
PRE_TRAJ_GLOB = "data/turnover_v1/traj/cell_*.csv"

#: sidecar metric names -- ONE metric per panel (the rendered-artist data
#: contract of visual_qa checks a panel against a single metric name)
M_A = "c_ideal (per-step mean across the 18 cells of the mode)"
M_GHOST = "c_ideal (age_hazard sync: pre-fix ghost vs post-fix)"
M_B = "c_ideal (settled mean over steps 301-400)"
M_C = "gap_recent_mean_last100 (pooled bar and per-landscape markers)"
M_D = "native share (harvested_fin, top10 at step 399, top10_fin at step 400)"
M_REF = "reference (definitional constants)"


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# loaders (raw CSVs only)
# ---------------------------------------------------------------------------

def data_root_path(data_root=None) -> Path:
    """Post-fix archive root (``.../turnover_v2``)."""
    return Path(data_root) if data_root is not None else POST_ROOT


def _read_traj(root: Path) -> pd.DataFrame:
    files = sorted((root / "traj").glob("cell_*.csv"))
    _require(bool(files), f"{root}/traj: no cell_*.csv found")
    frames = []
    for f in files:
        m = _TRAJ_RE.match(f.name)
        _require(m is not None, f"traj: unparsable filename {f.name}")
        d = pd.read_csv(f)
        _require(len(d) == 400 and int(d["step"].min()) == 1
                 and int(d["step"].max()) == 400,
                 f"{f.name}: expected steps 1..400, got {len(d)} rows")
        d["exit_mode"] = m.group(1)
        d["g"] = int(m.group(2)) / 100
        d["curiosity"] = int(m.group(3)) / 10
        d["landscape"] = m.group(4)
        frames.append(d)
    traj = pd.concat(frames, ignore_index=True)
    _require(traj["cell_tag"].nunique() == len(files),
             "traj: one cell_tag per cell file expected")
    return traj


def load_post(data_root=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Post-fix trajectories + runs, restricted to the common support."""
    root = data_root_path(data_root)
    traj_all = _read_traj(root)
    files = sorted((root / "traj").glob("cell_*.csv"))
    _require(len(files) == 76, f"traj: 76 cell files expected, found {len(files)}")

    runs_all = pd.read_csv(root / "runs.csv")
    _require(len(runs_all) == 1520,
             f"runs: 1,520 rows expected, found {len(runs_all)}")
    _require(sorted(runs_all["exit_mode"].unique().tolist()) == sorted(MODES),
             "runs: exit_mode grid")
    _require(sorted(runs_all["entry"].unique().tolist()) == G_LEVELS,
             "runs: entry grid")
    _require(sorted(runs_all["landscape"].unique().tolist()) == sorted(LANDSCAPES),
             "runs: landscape grid")

    traj = traj_all[traj_all["curiosity"].isin(COMMON_CURIOSITY)].copy()
    runs = runs_all[runs_all["curiosity"].isin(COMMON_CURIOSITY)].copy()

    per_mode_cells = traj.groupby("exit_mode")["cell_tag"].nunique().to_dict()
    _require(per_mode_cells == {m: 18 for m in MODES},
             f"traj: common-support cells per mode {per_mode_cells} != 18 each")
    _require(len(traj) == 72 * 400,
             f"traj: {72 * 400} common-support rows expected, found {len(traj)}")

    per_mode_runs = runs.groupby("exit_mode").size().to_dict()
    _require(per_mode_runs == {m: 360 for m in MODES},
             f"runs: common-support runs per mode {per_mode_runs} != 360 each")
    _require(len(runs) == 1440,
             f"runs: 1,440 common-support runs expected, found {len(runs)}")
    _require(bool(np.allclose(runs["xi"], 1.0 - runs["curiosity"])),
             "runs: xi must equal 1 - curiosity")
    return traj, runs


def load_pre() -> pd.DataFrame:
    """Read-only pre-fix trajectories (``data/turnover_v1``)."""
    traj = _read_traj(PRE_ROOT)
    files = sorted((PRE_ROOT / "traj").glob("cell_*.csv"))
    _require(len(files) == 54, f"pre traj: 54 cell files expected, found {len(files)}")
    _require(sorted(traj["exit_mode"].unique().tolist())
             == sorted(["none", "fixed_span", PRE_MODE]),
             "pre traj: exit_mode grid")
    per_mode = traj.groupby("exit_mode")["cell_tag"].nunique().to_dict()
    _require(per_mode == {"none": 18, "fixed_span": 18, PRE_MODE: 18},
             f"pre traj: cells per mode {per_mode} != 18 each")
    return traj


def _mean_ci(vals) -> tuple[float, float, float, int]:
    v = pd.Series(vals, dtype=float).dropna()
    n = int(v.size)
    _require(n > 1, "CI needs at least two observations")
    mean = float(v.mean())
    ci = Z95 * float(v.std(ddof=1)) / np.sqrt(n)
    return mean, mean - ci, mean + ci, n


# ---------------------------------------------------------------------------
# annotation strings (spec; mirrored verbatim by verify_F8_v3.py)
# ---------------------------------------------------------------------------

def fmt_pct(rel: float) -> str:
    return f"{rel:+.1f}%"


def label_gap(v: float, rel: float) -> str:
    return f"{v:.5f}\n[{fmt_pct(rel)}]"


def label_share(series: str, v: float) -> str:
    return f"{v:.3f}"


def note_regimes(disc: float, depl: float, late: float) -> str:
    return (f"pooled c_ideal:Discovery {disc:+.4f} → Depletion {depl:+.4f}"
            f" (301–400 {late:+.4f})")


def note_negative(neg: int, tot: int) -> str:
    return f"negative in {neg:,}/{tot:,} runs"


def note_spike(spike: float, staggered: float) -> str:
    return (f"step-400 top-10%: sync {spike:.3f} → staggered {staggered:.3f}")


def label_ghost(pre400: float, post400: float) -> str:
    return f"t=400: pre-fix {pre400:+.3f} → post-fix {post400:+.3f}"


def label_equality() -> str:
    return f"{EQUALITY_TOP10:.2f} equality"


# ---------------------------------------------------------------------------
# compute (every plotted number comes from the raw CSVs)
# ---------------------------------------------------------------------------

def compute(data_root=None):
    """Return ``(frames, texts, sidecar)`` for the four panels."""
    traj, runs = load_post(data_root)
    pre = load_pre()

    rows: list[dict] = []

    def add_plot(panel, metric, series, *, value, ci_lo=np.nan, ci_hi=np.nan,
                 n, exit_mode="", g=np.nan, landscape="pooled", step=np.nan,
                 statistic="mean", source_file, note=""):
        rows.append(dict(figure=FNAME, panel=panel, metric=metric,
                         statistic=statistic, series=series,
                         exit_mode=exit_mode, g=g, landscape=landscape,
                         step=step, value=float(value), ci_lo=float(ci_lo),
                         ci_hi=float(ci_hi), n=n, source_file=source_file,
                         note=note))

    def add_annotation(panel, metric, *, value, n, exit_mode="", g=np.nan,
                       landscape="pooled", step=np.nan, source_file, note,
                       series="annotation"):
        rows.append(dict(figure=FNAME, panel=panel, metric=metric,
                         statistic="annotation", series=series,
                         exit_mode=exit_mode, g=g, landscape=landscape,
                         step=step, value=float(value), ci_lo=np.nan,
                         ci_hi=np.nan, n=n, source_file=source_file,
                         note=note))

    # ---------------- panel (a): post-fix lifecycle trajectories ----------
    a = (traj.groupby(["exit_mode", "step"])["c_ideal"]
         .agg(mean="mean", sd="std", n="size").reset_index())
    a["ci"] = Z95 * a["sd"] / np.sqrt(a["n"])
    a["ci_lo"] = a["mean"] - a["ci"]
    a["ci_hi"] = a["mean"] + a["ci"]
    _require(len(a) == len(MODES) * 400, f"panel a rows {len(a)} != {len(MODES) * 400}")
    for r in a.itertuples():
        add_plot("a", M_A, "trajectory", value=r.mean, ci_lo=r.ci_lo,
                 ci_hi=r.ci_hi, n=int(r.n), exit_mode=r.exit_mode,
                 step=int(r.step), source_file=POST_TRAJ_GLOB,
                 note="per-step mean of c_ideal across the 18 cells of the mode")

    disc = float(traj.loc[traj["step"] <= DISCOVERY_END, "c_ideal"].mean())
    depl = float(traj.loc[traj["step"] >= DEPLETION_START, "c_ideal"].mean())
    late = float(traj.loc[traj["step"] >= LATE_START, "c_ideal"].mean())
    text_a = note_regimes(disc, depl, late)
    for key, val in (("discovery_mean", disc), ("depletion_mean", depl),
                     ("late_mean", late)):
        add_annotation("a", f"annotation: {key}", value=val, n=int(len(traj)),
                       source_file=POST_TRAJ_GLOB,
                       note=text_a if key == "discovery_mean" else "")

    # (a) inset: pre-fix ghost vs post-fix, same mode
    pre_ghost = (pre[pre["exit_mode"] == PRE_MODE]
                 .groupby("step")["c_ideal"].mean())
    post_ghost = (traj[traj["exit_mode"] == GHOST_MODE]
                  .groupby("step")["c_ideal"].mean())
    _require(len(pre_ghost) == 400 and len(post_ghost) == 400,
             "inset: both ghost series must cover steps 1..400")
    ghost_steps = np.arange(GHOST_START, 401)
    ghost = pd.DataFrame({
        "step": ghost_steps,
        "pre_fix": pre_ghost.reindex(ghost_steps).to_numpy(float),
        "post_fix": post_ghost.reindex(ghost_steps).to_numpy(float),
    })
    for r in ghost.itertuples():
        for series, col, src in (("pre_fix_ghost", "pre_fix", PRE_TRAJ_GLOB),
                                 ("post_fix", "post_fix", POST_TRAJ_GLOB)):
            add_plot("a_inset", M_GHOST, series, value=getattr(r, col),
                     n=18, exit_mode=GHOST_MODE, step=int(r.step),
                     source_file=src,
                     note="mean across the 18 cells; pre-fix = BUG-0006 unfixed")
    pre400 = float(pre_ghost.loc[400])
    post400 = float(post_ghost.loc[400])
    add_annotation("a_inset", "annotation: pre_fix_step400", value=pre400,
                   n=18, exit_mode=GHOST_MODE, step=400, source_file=PRE_TRAJ_GLOB,
                   note=label_ghost(pre400, post400))
    add_annotation("a_inset", "annotation: post_fix_step400", value=post400,
                   n=18, exit_mode=GHOST_MODE, step=400, source_file=POST_TRAJ_GLOB,
                   note=label_ghost(pre400, post400))

    # ---------------- panel (b): settled alignment vs entry rate ----------
    late_traj = traj[(traj["step"] >= LATE_START) & (traj["step"] <= LATE_END)]
    _require(int(late_traj["step"].nunique()) == LATE_END - LATE_START + 1,
             "panel b: settled window must contain every step 301..400")
    cell_means = (late_traj.groupby(["exit_mode", "g", "cell_tag"])["c_ideal"]
                  .mean().reset_index())
    b = (cell_means.groupby(["exit_mode", "g"])["c_ideal"]
         .agg(mean="mean", sd="std", n="size").reset_index())
    b["ci"] = Z95 * b["sd"] / np.sqrt(b["n"])
    b["ci_lo"] = b["mean"] - b["ci"]
    b["ci_hi"] = b["mean"] + b["ci"]
    b["value"] = b["mean"]
    for r in b.itertuples():
        add_plot("b", M_B, "settled", value=r.mean, ci_lo=r.ci_lo,
                 ci_hi=r.ci_hi, n=int(r.n), exit_mode=r.exit_mode,
                 g=float(r.g), source_file=POST_TRAJ_GLOB,
                 note=("mean over the 6 cells of their steps-301-400 means; "
                       "95% CI across the 6 cell means"))
    for m in MODES:
        sub = b[b["exit_mode"] == m].set_index("g")["mean"]
        delta = float(sub.loc[G_LEVELS[-1]] - sub.loc[G_LEVELS[0]])
        add_annotation("b", "annotation: settled_change_g005_to_g020",
                       value=delta, n=6, exit_mode=m, source_file=POST_TRAJ_GLOB,
                       note=f"{m}: settled change g 0.05 -> 0.20")

    # ---------------- panel (c): arrival-gap shrinkage --------------------
    c_frames = []
    for m in MODES:
        sub = runs[runs["exit_mode"] == m]
        v, lo, hi, n = _mean_ci(sub["gap_recent_mean_last100"])
        add_plot("c", M_C, "pooled", value=v, ci_lo=lo, ci_hi=hi, n=n,
                 exit_mode=m, source_file=POST_RUNS,
                 note="mean of gap_recent_mean_last100 over the 360 runs; "
                      "95% CI = mean +/- 1.96*SE over runs")
        c_frames.append(dict(exit_mode=m, series="pooled", landscape="pooled",
                             value=v, ci_lo=lo, ci_hi=hi, n=n))
        for land in LANDSCAPES:
            vv, llo, hhi, nn = _mean_ci(
                sub.loc[sub["landscape"] == land, "gap_recent_mean_last100"])
            add_plot("c", M_C, land, value=vv, ci_lo=llo, ci_hi=hhi, n=nn,
                     exit_mode=m, landscape=land, source_file=POST_RUNS,
                     note=f"mean over the {nn} {land} runs")
            c_frames.append(dict(exit_mode=m, series=land, landscape=land,
                                 value=vv, ci_lo=llo, ci_hi=hhi, n=nn))
    cdf = pd.DataFrame(c_frames)
    base = float(cdf[(cdf["exit_mode"] == "none")
                     & (cdf["series"] == "pooled")]["value"].iloc[0])
    c_rel = {}
    for m in MODES:
        v = float(cdf[(cdf["exit_mode"] == m)
                      & (cdf["series"] == "pooled")]["value"].iloc[0])
        rel = 100.0 * (v / base - 1.0)
        c_rel[m] = rel
        add_annotation("c", "annotation: shrinkage_pct_vs_none", value=rel,
                       n=360, exit_mode=m, source_file=POST_RUNS,
                       note=f"{m}: gap_recent change vs none = {fmt_pct(rel)}")
    neg = int((runs["gap_recent_mean_last100"] < 0).sum())
    tot = int(len(runs))
    text_c = note_negative(neg, tot)
    add_annotation("c", "annotation: negative_runs_note", value=float(neg),
                   n=tot, source_file=POST_RUNS, note=text_c)

    # ---------------- panel (d): output & concentration -------------------
    d_frames = []
    t399 = (traj[traj["step"] == 399]
            .groupby("exit_mode")["top10"].agg(list))
    for m in MODES:
        sub = runs[runs["exit_mode"] == m]
        v, lo, hi, n = _mean_ci(sub["harvested_fin"])
        add_plot("d", M_D, "harvested_fin", value=v, ci_lo=lo, ci_hi=hi, n=n,
                 exit_mode=m, source_file=POST_RUNS,
                 note="mean of harvested_fin over the 360 runs")
        d_frames.append(dict(exit_mode=m, series="harvested_fin", step=np.nan,
                             value=v, ci_lo=lo, ci_hi=hi, n=n))
        v, lo, hi, n = _mean_ci(sub["top10_fin"])
        add_plot("d", M_D, "top10_step400", value=v, ci_lo=lo, ci_hi=hi, n=n,
                 exit_mode=m, step=BIRTH_SHOCK_STEPS[-1], source_file=POST_RUNS,
                 note="mean of top10_fin (terminal step 400) over the 360 runs")
        d_frames.append(dict(exit_mode=m, series="top10_step400",
                             step=BIRTH_SHOCK_STEPS[-1], value=v, ci_lo=lo,
                             ci_hi=hi, n=n))
        v, lo, hi, n = _mean_ci(t399.loc[m])
        add_plot("d", M_D, "top10_step399", value=v, ci_lo=lo, ci_hi=hi, n=n,
                 exit_mode=m, step=399, source_file=POST_TRAJ_GLOB,
                 note="mean of the trajectory top10 share at step 399 over the "
                      "18 cells; 95% CI across cells")
        d_frames.append(dict(exit_mode=m, series="top10_step399", step=399,
                             value=v, ci_lo=lo, ci_hi=hi, n=n))
    ddf = pd.DataFrame(d_frames)
    spike = float(ddf[(ddf["exit_mode"] == "age_hazard_sync")
                      & (ddf["series"] == "top10_step400")]["value"].iloc[0])
    stag = float(ddf[(ddf["exit_mode"] == "age_hazard_staggered")
                     & (ddf["series"] == "top10_step400")]["value"].iloc[0])
    text_d = note_spike(spike, stag)
    for m in MODES:
        sub = ddf[(ddf["exit_mode"] == m) & (ddf["series"] == "top10_step400")]
        add_annotation("d", "annotation: top10_step400", value=float(sub["value"].iloc[0]),
                       n=360, exit_mode=m, step=BIRTH_SHOCK_STEPS[-1],
                       source_file=POST_RUNS, note=text_d if m == "age_hazard_sync" else "")

    # ---------------- reference rows (definitional constants) ------------
    for cat, val, step, ann in [
        ("discovery_end", DISCOVERY_END, DISCOVERY_END, f"Discovery (t ≤ {DISCOVERY_END})"),
        ("depletion_start", DEPLETION_START, DEPLETION_START,
         f"Depletion (t ≥ {DEPLETION_START})"),
        ("birth_shock_step", BIRTH_SHOCK_STEPS[0], BIRTH_SHOCK_STEPS[0],
         "dotted drop-line; tau_max = 200 synchronized cohort replacement"),
        ("birth_shock_step", BIRTH_SHOCK_STEPS[1], BIRTH_SHOCK_STEPS[1],
         "dotted drop-line; tau_max = 200 synchronized cohort replacement"),
        ("late_window_start", LATE_START, LATE_START, ""),
        ("late_window_end", LATE_END, LATE_END, ""),
        ("equality_top10", EQUALITY_TOP10, np.nan, label_equality()),
        ("z95", Z95, np.nan, "95% CI multiplier (house convention)"),
        ("common_support_curiosity_min", COMMON_CURIOSITY[0], np.nan,
         "common-support curiosity floor (all four modes)"),
        ("common_support_curiosity_max", COMMON_CURIOSITY[-1], np.nan,
         "common-support curiosity ceiling (all four modes)"),
    ]:
        rows.append(dict(figure=FNAME, panel="ref", metric=M_REF,
                         statistic="definitional constant", series=cat,
                         exit_mode="", g=np.nan, landscape="", step=step,
                         value=float(val), ci_lo=np.nan, ci_hi=np.nan, n=np.nan,
                         source_file="commission (team2_figure_repair_v3.md)",
                         note=ann))

    sidecar = pd.DataFrame(rows, columns=[
        "figure", "panel", "metric", "statistic", "series", "exit_mode", "g",
        "landscape", "step", "value", "ci_lo", "ci_hi", "n", "source_file", "note"])

    frames = dict(a=a, ghost=ghost, b=b, c=cdf, d=ddf)
    texts = dict(regimes=text_a, negative=text_c, spike=text_d,
                 ghost=label_ghost(pre400, post400), c_rel=c_rel,
                 disc=disc, depl=depl, late=late, pre400=pre400, post400=post400)
    return frames, texts, sidecar


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def _ylim_from(values, ci_lo, ci_hi, pad_lo=0.10, pad_hi=0.10) -> tuple[float, float]:
    lo = float(min(np.min(ci_lo), np.min(values)))
    hi = float(max(np.max(ci_hi), np.max(values)))
    span = hi - lo
    if span <= 0:
        span = abs(hi) if hi else 1.0
    return lo - pad_lo * span, hi + pad_hi * span


def _clip_ticks(ax, *, prune_x: bool = False, eps: float = 1e-12) -> None:
    """Drop locator ticks that fall outside the view limits.

    matplotlib's ``MaxNLocator`` happily returns one "nice" tick beyond the
    axis limits; such a tick is still drawn -- OUTSIDE the axes, i.e. on top
    of the title or the neighbouring panel.  Pruning keeps the rendered
    inventory honest.
    """
    lo, hi = ax.get_ylim()
    ax.set_yticks([t for t in ax.get_yticks() if lo - eps <= t <= hi + eps])
    if prune_x:
        lo, hi = ax.get_xlim()
        ax.set_xticks([t for t in ax.get_xticks() if lo - eps <= t <= hi + eps])


def _draw(frames, texts, sidecar, png: Path, pdf: Path,
          collector: dict | None = None) -> dict:
    s = HOUSE_STYLE
    a, ghost, bdf, cdf, ddf = (frames["a"], frames["ghost"], frames["b"],
                               frames["c"], frames["d"])
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.6), dpi=s["dpi"])
    axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ---------------- (a) lifecycle trajectories -------------------------
    axA.axvspan(1, DISCOVERY_END, color=PALETTE["blue_light"], alpha=0.35,
                linewidth=0, zorder=0)
    axA.axvspan(DEPLETION_START, 400, color=PALETTE["orange_light"],
                alpha=0.30, linewidth=0, zorder=0)
    axA.axhline(0.0, color="black", linewidth=0.9, linestyle="--", zorder=3)
    for step in BIRTH_SHOCK_STEPS:
        axA.axvline(step, color=PALETTE["gray"], linewidth=1.0,
                    linestyle=":", alpha=0.75, zorder=1)
    for m in MODES:
        d = a[a["exit_mode"] == m].sort_values("step")
        axA.fill_between(d["step"], d["ci_lo"], d["ci_hi"],
                         color=MODE_COLOR[m], alpha=0.16, linewidth=0, zorder=2)
        axA.plot(d["step"], d["mean"], color=MODE_COLOR[m], linewidth=1.7,
                 zorder=4, label=MODE_LABEL[m])
    ylo, yhi = _ylim_from(a["mean"], a["ci_lo"], a["ci_hi"],
                          pad_lo=0.10, pad_hi=0.09)
    axA.set_xlim(1, 400)
    axA.set_ylim(ylo, yhi)
    _cosmetics(axA, s,
               "(a) Post-fix lifecycle: Discovery → Depletion, four exit modes",
               "step t",
               "Corr(prestige, ideal merit)\n(per-step mean across 18 cells)")
    span = yhi - ylo
    axA.text(DISCOVERY_END / 2, yhi - 0.015 * span, f"Discovery (t ≤ {DISCOVERY_END})",
             ha="center", va="top", fontsize=s["note_size"] - 0.5,
             color=PALETTE["gray"], zorder=6)
    axA.text((DEPLETION_START + 400) / 2, yhi - 0.015 * span,
             f"Depletion (t ≥ {DEPLETION_START})", ha="center", va="top",
             fontsize=s["note_size"] - 0.5, color=PALETTE["gray"], zorder=6)
    axA.text(0.995, 0.575, texts["regimes"], transform=axA.transAxes,
             ha="right", va="top", fontsize=s["note_size"] - 0.5, zorder=7,
             bbox=dict(boxstyle="round,pad=0.28", fc="white",
                       ec=PALETTE["gray_light"], alpha=0.95))
    axA.legend(fontsize=s["tick_size"] - 1.5, frameon=False, loc="upper right",
               bbox_to_anchor=(0.998, 0.470), handlelength=1.6,
               labelspacing=0.35, borderaxespad=0.0)

    # inset ghost: pre-fix vs post-fix for the same (synchronized) arm
    axin = axA.inset_axes([0.435, 0.615, 0.345, 0.265])
    axin.set_facecolor("white")
    axin.axhline(0.0, color=PALETTE["gray_light"], linewidth=0.8, zorder=1)
    axin.plot(ghost["step"], ghost["pre_fix"], color=PALETTE["gray"],
              linewidth=1.0, linestyle="--", zorder=3,
              label="pre-fix (BUG-0006 unfixed)")
    axin.plot(ghost["step"], ghost["post_fix"], color=MODE_COLOR[GHOST_MODE],
              linewidth=1.1, zorder=4, label="post-fix")
    axin.set_xlim(GHOST_START, 408)
    axin.set_ylim(-0.95, 0.30)
    axin.set_yticks([-0.5, 0.0])
    axin.set_xticks([200, 300, 400])
    axin.set_title("pre-fix vs post-fix (age_hazard_sync)",
                   fontsize=5.8, pad=1.6)
    axin.set_ylabel("c_ideal", fontsize=6.0, labelpad=1.5)
    axin.set_xlabel("step t", fontsize=6.0, labelpad=1.0)
    axin.tick_params(labelsize=5.6, length=2, pad=1.2)
    axin.text(GHOST_START + 8, 0.275, texts["ghost"], ha="left", va="top",
              fontsize=5.0, zorder=6)
    axin.legend(fontsize=5.2, frameon=False, loc="lower center", ncol=2,
                handlelength=1.3, labelspacing=0.2, columnspacing=0.8,
                borderaxespad=0.1)

    # ---------------- (b) settled alignment vs entry rate ----------------
    for m in MODES:
        d = bdf[bdf["exit_mode"] == m].sort_values("g")
        axB.errorbar(d["g"], d["value"], yerr=[d["value"] - d["ci_lo"],
                                              d["ci_hi"] - d["value"]],
                     color=MODE_COLOR[m], marker="o", markersize=5,
                     linewidth=1.7, elinewidth=1.0, capsize=2.5, zorder=4,
                     label=MODE_LABEL[m])
    ylo, yhi = _ylim_from(bdf["value"], bdf["ci_lo"], bdf["ci_hi"],
                          pad_lo=0.12, pad_hi=0.55)
    axB.set_xlim(G_LEVELS[0] - 0.02, G_LEVELS[-1] + 0.02)
    axB.set_ylim(ylo, yhi)
    _cosmetics(axB, s,
               "(b) Settled alignment (steps 301–400) vs entry rate",
               "entry rate g (birth every 1/g steps)",
               "Corr(prestige, ideal merit)\n(settled mean, 6 cells)")
    axB.set_xticks(G_LEVELS, [f"{g:.2f}" for g in G_LEVELS])
    axB.legend(fontsize=s["tick_size"] - 1.5, frameon=False, loc="upper left",
               bbox_to_anchor=(0.0, 1.0), ncol=2, columnspacing=1.0,
               handlelength=1.4, labelspacing=0.3, borderaxespad=0.2)

    # ---------------- (c) arrival-gap shrinkage --------------------------
    vmin = float(cdf["ci_lo"].min())
    span_c = abs(vmin)
    axC.set_ylim(vmin - 0.40 * span_c, 0.55 * span_c)
    axC.axhline(0.0, color="black", linewidth=0.8, zorder=2)
    for i, m in enumerate(MODES):
        x = float(i)
        pooled = cdf[(cdf["exit_mode"] == m) & (cdf["series"] == "pooled")].iloc[0]
        axC.bar(x, pooled["value"], width=0.42, color=MODE_COLOR[m], zorder=3)
        axC.errorbar([x], [pooled["value"]],
                     yerr=[[pooled["value"] - pooled["ci_lo"]],
                           [pooled["ci_hi"] - pooled["value"]]],
                     fmt="none", ecolor=MODE_COLOR[m], elinewidth=1.2,
                     capsize=3, zorder=4)
        axC.text(x, pooled["ci_lo"] - 0.035 * span_c,
                 label_gap(pooled["value"], texts["c_rel"][m]),
                 ha="center", va="top", fontsize=s["note_size"] - 1.0, zorder=5)
        for land in LANDSCAPES:
            sub = cdf[(cdf["exit_mode"] == m) & (cdf["series"] == land)].iloc[0]
            axC.errorbar([x + LAND_OFF[land]], [sub["value"]],
                         yerr=[[sub["value"] - sub["ci_lo"]],
                               [sub["ci_hi"] - sub["value"]]],
                         fmt=LAND_MARK[land], color=MODE_COLOR[m],
                         markerfacecolor="white", markersize=4.6,
                         elinewidth=0.9, capsize=2.0, zorder=5)
    _cosmetics(axC, s, "(c) Arrival-gap shrinkage (gap_recent, native units)",
               "exit mode",
               "gap_recent (mean of last 100 steps)\n(run means)")
    axC.set_xticks(range(len(MODES)), MODES)
    axC.set_xticks(range(len(MODES)))
    axC.set_xticklabels(MODES, fontsize=s["tick_size"] - 1.0)
    axC.legend(handles=[
        Patch(facecolor=PALETTE["gray"], label="pooled (360 runs, bar)"),
        Line2D([], [], linestyle="none", marker="D", color=PALETTE["gray"],
               markerfacecolor="white", markersize=4.6,
               label="mgstd5 (180 runs)"),
        Line2D([], [], linestyle="none", marker="^", color=PALETTE["gray"],
               markerfacecolor="white", markersize=4.6,
               label="noisypr5 (180 runs)"),
    ], fontsize=s["tick_size"] - 1.5, frameon=False, loc="upper right",
        handlelength=1.4, labelspacing=0.25, borderaxespad=0.2)
    axC.text(0.012, 0.965, texts["negative"], transform=axC.transAxes,
             ha="left", va="top", fontsize=s["note_size"] - 1.0, zorder=6)

    # ---------------- (d) output & concentration -------------------------
    width = 0.26
    for i, m in enumerate(MODES):
        for j, ser in enumerate(["harvested_fin", "top10_step399",
                                 "top10_step400"]):
            r = ddf[(ddf["exit_mode"] == m) & (ddf["series"] == ser)].iloc[0]
            x = i + (j - 1) * width
            axD.bar(x, r["value"], width=width, color=D_COLOR[ser], zorder=3)
            axD.errorbar([x], [r["value"]],
                         yerr=[[r["value"] - r["ci_lo"]],
                               [r["ci_hi"] - r["value"]]],
                         fmt="none", ecolor=D_COLOR[ser], elinewidth=1.1,
                         capsize=2.5, zorder=4)
            axD.text(x, r["ci_hi"] + 0.018, label_share(ser, r["value"]),
                     ha="center", va="bottom", fontsize=s["note_size"] - 1.2,
                     zorder=5)
    _cosmetics(axD, s, "(d) Output and concentration (native shares)",
               "exit mode", "share (native units [0, 1])")
    axD.set_xticks(range(len(MODES)))
    axD.set_xticklabels(MODES, fontsize=s["tick_size"] - 1.0)
    axD.set_xlim(-0.55, 3.55)
    axD.set_ylim(0.0, 1.30)
    axD.axhline(EQUALITY_TOP10, color=PALETTE["gray"], linewidth=1.0,
                linestyle=":", zorder=1)
    axD.legend(handles=[
        Patch(facecolor=D_COLOR["harvested_fin"], label="harvested knowledge share"),
        Patch(facecolor=D_COLOR["top10_step399"],
              label="top-10% prestige (step 399, settled)"),
        Patch(facecolor=D_COLOR["top10_step400"],
              label="top-10% prestige (step 400, cohort shock)"),
        Line2D([], [], color=PALETTE["gray"], linestyle=":", linewidth=1.0,
               label=f"{EQUALITY_TOP10:.2f} equality"),
    ], fontsize=s["tick_size"] - 1.5, frameon=False, loc="upper right",
        bbox_to_anchor=(0.998, 0.925), handlelength=1.6, labelspacing=0.3,
        borderaxespad=0.0)
    axD.text(0.012, 0.985, texts["spike"], transform=axD.transAxes,
             ha="left", va="top", fontsize=s["note_size"] - 1.0, zorder=6)

    # ---------------- source footnotes (wrapped) ------------------------
    foot = [
        textwrap.fill(
            "Source: data/turnover_v2/traj/cell_*.csv (76 cell trajectories x 400 "
            "steps, each the mean over 20 seeds) and data/turnover_v2/runs.csv "
            "(1,520 runs), restricted to the COMMON SUPPORT curiosity "
            "{0.3, 0.5, 0.7}: 4 modes x 3 g x 3 curiosity x 2 landscapes x 20 "
            "seeds = 1,440 runs / 72 cells (18 per mode). The 80 excluded runs "
            "are the two extra curiosity cells (0.1, 0.9) only the hazard arms "
            "carry. Inset ghost: data/turnover_v1/traj/cell_*.csv (read-only).",
            width=196),
        textwrap.fill(
            "BUG-0006 (fixed): newborns used to enter the data collector with "
            "prestige = 0, ideal_Merit = 0, so every birth step carried an "
            "artificial 0-vs-optimum pair; the tau_max = 200 synchronized cohort "
            "replacement at t = 200, 400 amplified it. Post-fix newborn ideal "
            "merit is NaN and correlation_reporter drops non-finite pairs, so all "
            "birth steps are valid measurements and (b) uses every step of "
            "301-400 without a birth-step filter.", width=196),
        textwrap.fill(
            "95% CIs = mean +/- 1.96*SE: (a) across the 18 cells per step; (b) "
            "across the 6 cell means of the (mode, g) group; (c) across runs (360 "
            "pooled / 180 per landscape); (d) across 360 runs for harvested and "
            "step-400 top-10%, and across 18 cells for the step-399 top-10%. "
            "Every plotted value, CI and annotation is computed from the raw "
            "CSVs and stored in " + f"{BASENAME}_data.csv. MODEL-INTERNAL, NOT "
            "CALIBRATED: no empirical benchmark is plotted here.", width=196),
    ]
    y = 0.125
    for block in foot:
        for line in block.split("\n"):
            fig.text(0.012, y, line, fontsize=s["note_size"] - 1.5,
                     color=PALETTE["gray"])
            y -= 0.0125

    fig.subplots_adjust(left=0.085, right=0.986, top=0.905, bottom=0.20,
                        hspace=0.40, wspace=0.20)

    # ---------------- hardened visual QA (before savefig) ----------------
    _clip_ticks(axA, prune_x=True)
    _clip_ticks(axin)
    _clip_ticks(axB)
    _clip_ticks(axC)
    _clip_ticks(axD)
    specs = [
        {"ax": axA, "metric": M_A, "expect_lines": 4, "points_per_line": 400,
         "sidecar": sidecar, "filters": {"panel": "a", "series": "trajectory"},
         # the 95% ribbon is a PolyCollection (not an errorbar container), so
         # the data contract compares the plotted means against the values
         "ci_cols": ("value", "value")},
        {"ax": axin, "metric": M_GHOST, "expect_lines": 2,
         "points_per_line": 401 - GHOST_START, "sidecar": sidecar,
         "filters": {"panel": "a_inset"},
         # the ghost inset plots point estimates only (no CI ribbon), so the
         # data contract must compare against the values themselves
         "ci_cols": ("value", "value")},
        {"ax": axB, "metric": M_B, "expect_lines": 4, "points_per_line": 3,
         "sidecar": sidecar, "filters": {"panel": "b", "series": "settled"}},
        {"ax": axC, "metric": M_C, "expect_bars": 4, "expect_lines": 12,
         "sidecar": sidecar, "filters": {"panel": "c"}},
        {"ax": axD, "metric": M_D, "expect_bars": 12, "expect_lines": 12,
         "sidecar": sidecar, "filters": {"panel": "d"}},
    ]
    report = visual_qa.assert_figure_clean(fig, specs)

    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=s["dpi"])
    fig.savefig(pdf)
    if collector is not None:
        # hand the live Figure to the verifier (it re-checks rendered geometry)
        collector["fig"] = fig
    plt.close(fig)
    return report


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(outdir="output/report_notebook_v1", data_root=None,
          collector: dict | None = None) -> dict:
    """Build F8_v3 (PNG + PDF + _data.csv); returns paths and the QA report."""
    out = Path(outdir)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{BASENAME}.png"
    pdf = out / f"{BASENAME}.pdf"

    frames, texts, sidecar = compute(data_root=data_root)
    report = _draw(frames, texts, sidecar, png, pdf, collector=collector)
    data_csv = write_figure_data(sidecar, png)
    print(f"F8_v3: {png} | {pdf} | {data_csv}")
    print(f"sidecar rows: {len(sidecar)} | visual_qa failures: {report['_total']}")
    return {"png": str(png), "pdf": str(pdf), "data_csv": str(data_csv),
            "qa": report, "sidecar_rows": len(sidecar)}


def main() -> int:
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
