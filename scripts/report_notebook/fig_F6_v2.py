"""F6_v2 -- two-stage proxy failure on the ADR-0003 clock (Figure 1, sec. 4.1).

Rebuilt from the ADR-0003 dense-xi archive ``data/fig1_xi_adr0003`` (one CSV
per cell, 20 seed rows each; 11 xi values x five tiers = 55 cells = 1,100
runs), with the pre-ADR archive still selectable through ``data_root``:

  tier (key)       exit_mode              g     lambda / pull    H
  ---------------------------------------------------------------------
  M0               none                   0.00  -               200
  M1_fixed         fixed_span             0.25  -               400
  M1_staggered     age_hazard_staggered   0.25  -               400
  M2_l0            fixed_span             0.25  0.0 / 1.0       200
  M2_l1            fixed_span             0.25  1.0 / 1.0       200

The tier configuration is not assumed: it is read from the cells
``tier/exit_mode/g/dyadic_lambda/dyadic_move_pull/horizon`` columns and
asserted against the ``fig1_xi`` arm definition in
``scripts/adr0003_remeasure/run_adr0003_wave.py`` (M0 = old "none",
M1_fixed/M1_staggered = the two ADR exit modes at the g = 0.25 pin,
M2_l0/M2_l1 = dyadic pull = 1.0 with lambda = 0/1).  H = 200 for the core
tiers and H = 400 for both M1 tiers, exactly as the horizon column records.

Panels:
  (a) macro efficiency      : harvested knowledge share (``harvested_fin``)
  (b) strictly like-for-like: time-mean ``c_ideal_tm`` in every arm (no
      time-mean / final-step mixing -- the distortion F6 originally suffered)

Stage bands are F6's exact grammar: Stage 1 (xi <= 0.6), welfare kink
[0.6, 0.8], Stage 2 (xi >= 0.8).  The bands are shaded REGIONS only: no
boundary rule cuts the data.

Provenance is data-driven: tier pins (exit mode, entry rate, horizon,
dyadic lambda / pull) come from the cells; the clock pins (tau_max in every
cell, canonical vanishing factor) come from the archive / ``sciencemodels.model``.

Gates: the builder asserts its own plotted per-(tier, xi) means against a
fresh re-read of the SAME per-cell CSVs (``check_against_cells``) -- the old
assertion against ``tables/four_tier_xi_summary.csv`` has no counterpart in
the new archive, so the aggregation identity is now cell-level.  Rendering:
``visual_qa.assert_figure_clean`` runs on the rendered canvas, plus
``audit_text_over_data`` -- a local display-space text-vs-data gate, because
``visual_qa`` compares line / error-bar segments in DATA coordinates against
DISPLAY-space text boxes and therefore cannot see a label sitting on a curve.

Vocabulary: hue+marker = tier; every plotted value, error bar and annotation
(callouts, correlations, footnote numbers) is computed here from the cells and
written into the ``*_data.csv`` sidecar.

Run:  MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
          scripts/report_notebook/fig_F6_v2.py
      (pre-ADR: build(data_root="data")) -- no writes into data/.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from diagram_templates import (  # noqa: E402
    HOUSE_STYLE,
    PALETTE,
    _cosmetics,
    write_figure_data,
)
import visual_qa  # noqa: E402
from sciencemodels.model import (  # noqa: E402
    CANONICAL_TAU_MAX,
    CANONICAL_VANISHING_FACTOR,
)

FNAME = "F6_v2_two_stage_proxy_failure"
DEFAULT_DATA_ROOT = REPO / "data" / "fig1_xi_adr0003"
LANDSCAPE = "mgstd5"
Z95 = 1.96                       # normal-approximation 95% CI over seeds
SEEDS_PER_CELL = 20
BAND_STAGE1, BAND_KINK, BAND_STAGE2 = (0.02, 0.6), (0.6, 0.8), (0.8, 1.0)
REPORT_HI_KNOT = 0.9             # the report's high-xi anchor (0.1/0.5/0.9)

#: panel metrics (sidecar vocabulary)
M_HARVEST = "harvested knowledge share"
M_CORR = "Corr(prestige, ideal merit) (time mean)"
PANELS = {"a": M_HARVEST, "b": M_CORR}
VALUE_COL = {"a": "harvested_fin", "b": "c_ideal_tm"}

#: tier keys, in legend order
TIERS = ("M0", "M1_fixed", "M1_staggered", "M2_l0", "M2_l1")
LABEL = {
    "M0": "M0 · no exit",
    "M1_fixed": "M1 · fixed_span",
    "M1_staggered": "M1 · age_hazard_staggered",
    "M2_l0": "M2 · λ = 0 (pull 1)",
    "M2_l1": "M2 · λ = 1 (pull 1)",
}
SHORT = {
    "M0": "M0",
    "M1_fixed": "M1 fixed",
    "M1_staggered": "M1 staggered",
    "M2_l0": "M2 λ = 0",
    "M2_l1": "M2 λ = 1",
}
STYLE = {
    "M0": dict(color=PALETTE["purple"], ls="-", marker="o",
               mfc=PALETTE["purple"]),
    "M1_fixed": dict(color=PALETTE["gray"], ls="--", marker="o", mfc="white"),
    "M1_staggered": dict(color=PALETTE["orange"], ls="-.", marker="D",
                         mfc=PALETTE["orange"]),
    "M2_l0": dict(color=PALETTE["blue"], ls="--", marker="s",
                  mfc=PALETTE["blue"]),
    "M2_l1": dict(color=PALETTE["green"], ls="-", marker="^",
                  mfc=PALETTE["green"]),
}
ENTRY_ARMS = ("M1_fixed", "M2_l0", "M2_l1")

#: the archive's own ``fig1_xi`` arm definition (run_adr0003_wave.py), used
#: ONLY as an assertion target: the pins themselves are read from the cells.
TIER_EXPECT = {
    "M0": dict(exit_mode="none", g=0.0, horizon=200,
               dyadic_lambda=0.0, dyadic_move_pull=0.0),
    "M1_fixed": dict(exit_mode="fixed_span", g=0.25, horizon=400,
                     dyadic_lambda=0.0, dyadic_move_pull=0.0),
    "M1_staggered": dict(exit_mode="age_hazard_staggered", g=0.25, horizon=400,
                         dyadic_lambda=0.0, dyadic_move_pull=0.0),
    "M2_l0": dict(exit_mode="fixed_span", g=0.25, horizon=200,
                  dyadic_lambda=0.0, dyadic_move_pull=1.0),
    "M2_l1": dict(exit_mode="fixed_span", g=0.25, horizon=200,
                  dyadic_lambda=1.0, dyadic_move_pull=1.0),
}
LEGACY_TABLE_TIER = {          # pre-ADR tables/four_tier_xi_summary.csv ids
    "M0": "M0_closed_Wv05",
    "M1_fixed": "M1_lineage_Wv05",
    "M1_staggered": "M1_turnover_staggered",
    "M2_l0": "M2_lambda0",
    "M2_l1": "M2_lambda1",
}


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _mean_ci(vals) -> tuple[float, float, float, int]:
    v = np.asarray(vals, dtype=float)
    n = int(v.size)
    m = float(v.mean())
    half = Z95 * float(v.std(ddof=1)) / np.sqrt(n) if n > 1 else float("nan")
    return m, m - half, m + half, n


def _knot(xis, target: float) -> float:
    """Nearest knot of the plotted xi grid to a named reference value."""
    return float(min(xis, key=lambda x: abs(float(x) - target)))


# ------------------------------------------------------------------ data ---
def data_root_path(data_root=None) -> Path:
    """Resolve the archive root (default: the ADR-0003 Figure-1 archive)."""
    return Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT


def layout_of(data_root=None) -> str:
    root = data_root_path(data_root)
    _require(root.is_dir(), f"missing data root {root}")
    if (root / "cells").is_dir():
        return "adr0003"
    if (root / "bug0006_remeasure_v2").is_dir():
        return "pre_adr"
    raise AssertionError(f"{root}: neither an ADR-0003 cells/ archive nor a "
                         "pre-ADR bug0006_remeasure_v2/ archive")


def _load_cells_archive(root: Path) -> tuple[pd.DataFrame, dict, dict]:
    """Read one CSV per cell from ``<root>/cells``; validate the tier pins."""
    files = sorted((root / "cells").glob("*.csv"))
    _require(len(files) == 55,
             f"{root}/cells: {len(files)} cell files, expected 55 "
             "(11 xi x five tiers)")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        _require(len(d) == SEEDS_PER_CELL,
                 f"{f.name}: {len(d)} seed rows, expected {SEEDS_PER_CELL}")
        frames.append(d.assign(source_file=f"{root.name}/cells/{f.name}"))
    df = pd.concat(frames, ignore_index=True)

    need = {"tier", "cell_tag", "seed", "g", "exit_mode", "tau_max",
            "curiosity", "xi", "landscape", "horizon", "use_visibility_reward",
            "dyadic_lambda", "dyadic_move_pull", "formulation",
            "credit_sharing", "harvested_fin", "c_ideal_tm"}
    _require(need <= set(df.columns), f"cells missing columns {sorted(need - set(df.columns))}")
    _require(set(df["landscape"]) == {LANDSCAPE},
             f"cells: landscape grid {sorted(set(df['landscape']))}")
    _require(bool(df["use_visibility_reward"].all()),
             "cells: visibility reward must be on in every Figure-1 tier")
    _require(set(df["formulation"]) == {"logit"}
             and set(df["credit_sharing"]) == {"equal_split"},
             "cells: expected logit + equal_split")
    _require(set(int(t) for t in df["tau_max"]) == {int(CANONICAL_TAU_MAX)},
             f"cells: tau_max {sorted(set(df['tau_max']))} != canonical "
             f"{CANONICAL_TAU_MAX}")
    _require(set(df["tier"]) == set(TIERS),
             f"cells: tiers {sorted(set(df['tier']))} != {list(TIERS)}")
    _require(np.allclose(df["xi"], np.round(1.0 - df["curiosity"], 2), atol=1e-9),
             "cells: xi must equal 1 - curiosity")
    df["xi"] = np.round(df["xi"], 2)
    xis = sorted(float(x) for x in df["xi"].unique())
    _require(len(xis) == 11, f"cells: {len(xis)} xi values, expected 11")

    pins: dict[str, dict] = {}
    for tier in TIERS:
        d = df[df["tier"] == tier]
        _require(sorted(float(x) for x in d["xi"].unique()) == xis,
                 f"{tier}: xi grid differs from the archive grid")
        for col in ("exit_mode", "g", "horizon", "dyadic_lambda",
                    "dyadic_move_pull"):
            vals = d[col].unique()
            _require(len(vals) == 1, f"{tier}: {col} not constant ({vals})")
        cfg = {col: d[col].iloc[0] for col in
               ("exit_mode", "g", "horizon", "dyadic_lambda",
                "dyadic_move_pull")}
        exp = TIER_EXPECT[tier]
        _require(str(cfg["exit_mode"]) == exp["exit_mode"],
                 f"{tier}: exit_mode {cfg['exit_mode']!r} != arm definition "
                 f"{exp['exit_mode']!r}")
        for col, key in (("g", "g"), ("horizon", "horizon"),
                         ("dyadic_lambda", "dyadic_lambda"),
                         ("dyadic_move_pull", "dyadic_move_pull")):
            _require(abs(float(cfg[col]) - float(exp[key])) <= 1e-9,
                     f"{tier}: {col} {cfg[col]} != arm definition {exp[key]}")
        _require((d.groupby("xi").size() == SEEDS_PER_CELL).all(),
                 f"{tier}: expected {SEEDS_PER_CELL} seeds per xi")
        _require(d["seed"].nunique() == SEEDS_PER_CELL,
                 f"{tier}: seed ids are not {SEEDS_PER_CELL} distinct values")
        pins[tier] = dict(
            exit_mode=str(cfg["exit_mode"]), g=float(cfg["g"]),
            horizon=int(cfg["horizon"]),
            dyadic_lambda=float(cfg["dyadic_lambda"]),
            dyadic_move_pull=float(cfg["dyadic_move_pull"]),
            vanishing=float(CANONICAL_VANISHING_FACTOR),
            cells=len(xis), runs=int(len(d)),
            source=f"{root.name}/cells",
        )

    runs = df[["tier", "xi", "harvested_fin", "c_ideal_tm", "seed",
               "source_file"]].copy()
    _require(len(runs) == len(TIERS) * len(xis) * SEEDS_PER_CELL,
             f"cells: {len(runs)} runs, expected "
             f"{len(TIERS) * len(xis) * SEEDS_PER_CELL}")
    meta = dict(layout="adr0003", root=root, xis=xis,
                cells=len(files), runs=int(len(runs)),
                tau_max=int(CANONICAL_TAU_MAX),
                gamma=float(CANONICAL_VANISHING_FACTOR),
                g_pins=sorted({p["g"] for p in pins.values()}),
                horizons=sorted({p["horizon"] for p in pins.values()}),
                archive=f"{root.name}/cells")
    return runs, pins, meta


def _load_pre_adr_archive(root: Path) -> tuple[pd.DataFrame, dict, dict]:
    """Legacy loader (pre-ADR archives) -- kept so before/after can be rebuilt."""
    f_w = root / "bug0006_remeasure_v2" / "w_v05_runs.csv"
    f_m2 = root / "bug0006_remeasure_v2" / "m2_xi_runs.csv"
    f_t = root / "turnover_v2" / "runs.csv"
    for f in (f_w, f_m2, f_t):
        _require(f.exists(), f"missing archive {f}")

    w = pd.read_csv(f_w)
    _require((w["landscape"] == LANDSCAPE).any(), "w_v05: no mgstd5 rows")
    wv = w[(w["landscape"] == LANDSCAPE)
           & (w["use_visibility_reward"] == True)].copy()  # noqa: E712
    _require(len(wv) == 200, f"w_v05 mgstd5+visibility: {len(wv)} runs")
    _require(set(np.round(wv["vanishing"], 3)) == {0.5},
             "w_v05: vanishing must be pinned at 0.5")
    wv["xi"] = np.round(1.0 - wv["curiosity"], 2)
    gs = sorted({float(np.round(x, 3)) for x in wv["agent_generation_time"]})
    _require(gs == [0.0, 0.1], f"w_v05: entry grid {gs}, expected [0.0, 0.1]")
    g_pin = gs[-1]
    wv["tier"] = np.where(np.round(wv["agent_generation_time"], 3) == 0.0,
                          "M0", "M1_fixed")

    m = pd.read_csv(f_m2)
    _require(len(m) == 200, f"m2_xi: {len(m)} runs")
    _require(set(np.round(m["vanishing"], 3)) == {0.9},
             "m2_xi: vanishing must be pinned at 0.9")
    _require((m["landscape"] == LANDSCAPE).all(), "m2_xi: landscape")
    _require(bool((m["use_visibility_reward"] == True).all()),  # noqa: E712
             "m2_xi: visibility reward must be on")
    _require(set(np.round(m["dyadic_move_pull"], 3)) == {1.0},
             "m2_xi: pull must be 1.0")
    m["xi"] = np.round(m["xi"], 2)
    _require(np.allclose(m["xi"], np.round(1.0 - m["curiosity"], 2)),
             "m2_xi: xi must equal 1 - curiosity")
    m["tier"] = np.where(m["dyadic_lambda"] == 0.0, "M2_l0", "M2_l1")

    t = pd.read_csv(f_t)
    _require(len(t) == 1520, f"turnover_v2: {len(t)} runs")
    _require("age_hazard_staggered" in set(t["exit_mode"]),
             "turnover_v2: age_hazard_staggered mode missing")
    _require(len({float(np.round(x, 3)) for x in m["agent_generation_time"]}) == 1,
             "m2_xi: expected one g pin")
    _require(abs(float(np.round(m["agent_generation_time"].iloc[0], 3)) - g_pin) <= 1e-9,
             "m2_xi g pin != M1 g pin")
    ts = t[(t["exit_mode"] == "age_hazard_staggered")
           & (t["landscape"] == LANDSCAPE)
           & (np.round(t["entry"], 3) == g_pin)].copy()
    _require(len(ts) == 100, f"turnover staggered entry={g_pin}: {len(ts)} runs")
    ts["xi"] = np.round(ts["xi"], 2)
    ts["tier"] = "M1_staggered"

    for d, name in ((wv, f_w), (m, f_m2), (ts, f_t)):
        d["source_file"] = f"{root.name}/{name.parent.name}/{name.name}"
    runs = pd.concat([wv[["tier", "xi", "harvested_fin", "c_ideal_tm", "seed",
                          "source_file"]],
                      m[["tier", "xi", "harvested_fin", "c_ideal_tm", "seed",
                         "source_file"]],
                      ts[["tier", "xi", "harvested_fin", "c_ideal_tm", "seed",
                          "source_file"]]], ignore_index=True)
    xis = sorted(float(x) for x in runs["xi"].unique())
    _require(len(xis) == 5, f"pre-ADR: {len(xis)} xi values, expected 5")
    for tier in TIERS:
        d = runs[runs["tier"] == tier]
        _require((d.groupby("xi").size() == SEEDS_PER_CELL).all(),
                 f"{tier}: expected {SEEDS_PER_CELL} seeds per xi")

    design_w = json.loads((root / "bug0006_remeasure_v2"
                           / "design.json").read_text())
    design_t = json.loads((root / "turnover_v2" / "design.json").read_text())
    four = pd.read_csv(root / "bug0006_remeasure_v2" / "tables"
                       / "four_tier_xi_summary.csv")
    tv = four[four["tier"] == LEGACY_TABLE_TIER["M1_staggered"]]["vanishing"].unique()
    _require(len(tv) == 1, "four_tier_xi_summary: staggered vanishing pin")
    vanish = {"M0": 0.5, "M1_fixed": 0.5, "M1_staggered": float(tv[0]),
              "M2_l0": 0.9, "M2_l1": 0.9}
    pins = {
        "M0": dict(exit_mode="none", g=0.0,
                   horizon=int(design_w["horizon"]), dyadic_lambda=0.0,
                   dyadic_move_pull=0.0, vanishing=vanish["M0"], cells=len(xis),
                   runs=100, source=f"{root.name}/bug0006_remeasure_v2/w_v05_runs.csv"),
        "M1_fixed": dict(exit_mode="none", g=g_pin,
                         horizon=int(design_w["horizon"]), dyadic_lambda=0.0,
                         dyadic_move_pull=0.0, vanishing=vanish["M1_fixed"],
                         cells=len(xis), runs=100,
                         source=f"{root.name}/bug0006_remeasure_v2/w_v05_runs.csv"),
        "M1_staggered": dict(exit_mode="age_hazard_staggered", g=g_pin,
                             horizon=int(design_t["horizon"]), dyadic_lambda=0.0,
                             dyadic_move_pull=0.0,
                             vanishing=vanish["M1_staggered"], cells=len(xis),
                             runs=100, source=f"{root.name}/turnover_v2/runs.csv"),
        "M2_l0": dict(exit_mode="none", g=g_pin,
                      horizon=int(design_w["horizon"]), dyadic_lambda=0.0,
                      dyadic_move_pull=1.0, vanishing=vanish["M2_l0"],
                      cells=len(xis), runs=100,
                      source=f"{root.name}/bug0006_remeasure_v2/m2_xi_runs.csv"),
        "M2_l1": dict(exit_mode="none", g=g_pin,
                      horizon=int(design_w["horizon"]), dyadic_lambda=1.0,
                      dyadic_move_pull=1.0, vanishing=vanish["M2_l1"],
                      cells=len(xis), runs=100,
                      source=f"{root.name}/bug0006_remeasure_v2/m2_xi_runs.csv"),
    }
    meta = dict(layout="pre_adr", root=root, xis=xis, cells=25,
                runs=int(len(runs)), tau_max=int(np.max(t["tau_max"]))
                if "tau_max" in t.columns else int(CANONICAL_TAU_MAX),
                gamma=float("nan"), g_pins=[g_pin],
                horizons=sorted({p["horizon"] for p in pins.values()}),
                archive=f"{root.name}/bug0006_remeasure_v2")
    return runs, pins, meta


def load_runs(data_root=None) -> tuple[pd.DataFrame, dict, dict]:
    """Tidy runs for the five tiers + per-tier pins + archive metadata."""
    root = data_root_path(data_root)
    layout = layout_of(root)
    if layout == "adr0003":
        runs, pins, meta = _load_cells_archive(root)
    else:
        runs, pins, meta = _load_pre_adr_archive(root)
    _require(set(runs["tier"]) == set(TIERS), "unexpected tier set")
    _require(int(runs["seed"].nunique()) >= SEEDS_PER_CELL,
             "fewer than 20 distinct seeds")
    return runs, pins, meta


def series_rows(runs: pd.DataFrame, xis) -> pd.DataFrame:
    """Mean / 95% CI per (panel, tier, xi) -- one row per plotted statistic."""
    rows = []
    for panel, metric in PANELS.items():
        col = VALUE_COL[panel]
        for tier in TIERS:
            for xi in xis:
                sel = runs[(runs["tier"] == tier) & (runs["xi"] == xi)][col]
                _require(len(sel) == SEEDS_PER_CELL,
                         f"{tier} xi={xi}: {len(sel)} runs, expected "
                         f"{SEEDS_PER_CELL}")
                mean, lo, hi, n = _mean_ci(sel)
                src = runs[(runs["tier"] == tier)
                           & (runs["xi"] == xi)]["source_file"]
                _require(src.nunique() == 1,
                         f"{tier} xi={xi}: mixed source files {set(src)}")
                rows.append(dict(
                    figure=FNAME, panel=panel, metric=metric,
                    statistic="mean", tier=tier, xi=xi, landscape=LANDSCAPE,
                    value=mean, ci_lo=lo, ci_hi=hi, n=n,
                    source_file=str(src.iloc[0]),
                ))
    return pd.DataFrame(rows)


def _value_table(series: pd.DataFrame) -> dict:
    """(panel, tier, xi) -> value, from the plotted aggregates only."""
    out = {}
    for _, r in series[series["statistic"] == "mean"].iterrows():
        out[(r["panel"], r["tier"], float(r["xi"]))] = float(r["value"])
    return out


def _corr(x, y) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    return float(np.corrcoef(a, b)[0, 1])


def annotation_rows(series: pd.DataFrame, runs: pd.DataFrame, pins: dict,
                    meta: dict) -> pd.DataFrame:
    """Every number on the canvas / in the note, derived from the plotted data."""
    V = _value_table(series)
    xis = list(meta["xis"])
    lo_knot, hi_knot = _knot(xis, min(xis)), _knot(xis, REPORT_HI_KNOT)
    plateau_xis = [x for x in xis if x <= 0.5]
    rows = []

    def add(panel: str, metric: str, value: float, note: str,
            n: int = 0) -> None:
        rows.append(dict(figure=FNAME, panel=panel, metric=metric,
                         statistic="annotation", tier="", xi=np.nan,
                         landscape=LANDSCAPE, value=float(value),
                         ci_lo=np.nan, ci_hi=np.nan, n=int(n),
                         source_file=note))

    # -- (a) macro: plateau, Stage-2 collapse, staggered rescue -------------
    plateau = [V[("a", t, xi)] for t in ENTRY_ARMS for xi in plateau_xis]
    collapse = [V[("a", t, hi_knot)] for t in ENTRY_ARMS]
    add("a", "entry_plateau_harvested_min", min(plateau),
        "min harvested share over entry arms (M1 fixed, M2 lam=0/1) at xi <= 0.5")
    add("a", "entry_plateau_harvested_max", max(plateau),
        "max harvested share over entry arms at xi <= 0.5")
    add("a", "stage2_harvest_min", min(collapse),
        f"min harvested share over entry arms at xi = {hi_knot:g}")
    add("a", "stage2_harvest_max", max(collapse),
        f"max harvested share over entry arms at xi = {hi_knot:g}")
    add("a", "m0_xi09_harvested", V[("a", "M0", hi_knot)],
        f"M0 harvested share at xi = {hi_knot:g}")
    add("a", "m1_fixed_harvested_min",
        min(V[("a", "M1_fixed", x)] for x in plateau_xis),
        "min M1 fixed_span harvested share at xi <= 0.5")
    add("a", "m1_fixed_harvested_max",
        max(V[("a", "M1_fixed", x)] for x in plateau_xis),
        "max M1 fixed_span harvested share at xi <= 0.5")
    add("a", "staggered_xi09_harvested", V[("a", "M1_staggered", hi_knot)],
        f"M1 age_hazard_staggered harvested share at xi = {hi_knot:g}")
    add("a", "staggered_rescue_pp",
        100.0 * (V[("a", "M1_staggered", hi_knot)] - V[("a", "M1_fixed", hi_knot)]),
        f"staggered minus M1 fixed harvested share at xi = {hi_knot:g}, "
        "percentage points")

    # -- (b) like-for-like alignment: endpoints, spread, reversal -----------
    for tier in TIERS:
        add("b", f"c_tm_xi01::{tier}", V[("b", tier, lo_knot)],
            f"time-mean c_ideal_tm, {tier}, xi = {lo_knot:g}")
        add("b", f"c_tm_xi09::{tier}", V[("b", tier, hi_knot)],
            f"time-mean c_ideal_tm, {tier}, xi = {hi_knot:g}")
    add("b", "alignment_spread_xi01",
        max(V[("b", t, lo_knot)] for t in TIERS)
        - min(V[("b", t, lo_knot)] for t in TIERS),
        f"across-tier spread of time-mean c_ideal_tm at xi = {lo_knot:g}")
    add("b", "alignment_spread_xi09",
        max(V[("b", t, hi_knot)] for t in TIERS)
        - min(V[("b", t, hi_knot)] for t in TIERS),
        f"across-tier spread of time-mean c_ideal_tm at xi = {hi_knot:g}")
    add("b", "m2_l0_alignment_rise", V[("b", "M2_l0", hi_knot)]
        - V[("b", "M2_l0", lo_knot)],
        "M2 lambda=0 time-mean c_ideal_tm change xi = "
        f"{lo_knot:g} -> {hi_knot:g}")
    add("b", "m2_l0_alignment_min", min(V[("b", "M2_l0", x)] for x in xis),
        "minimum M2 lambda=0 time-mean c_ideal_tm over the xi grid")

    # -- (b) ideal-channel correlations (run level = the report's number) ---
    for tier in TIERS:
        d = runs[runs["tier"] == tier]
        add("b", f"corr_xi_ctm_run::{tier}", _corr(d["xi"], d["c_ideal_tm"]),
            f"run-level Pearson corr(xi, c_ideal_tm) over all {len(d)} runs, "
            f"{tier}", n=len(d))
        cell = d.groupby("xi")["c_ideal_tm"].mean()
        add("b", f"corr_xi_ctm_cell::{tier}",
            _corr(cell.index, cell.values),
            f"Pearson corr(xi, tier x xi cell mean c_ideal_tm) over "
            f"{len(cell)} cells, {tier}", n=len(d))
    return pd.DataFrame(rows)


def check_against_cells(series: pd.DataFrame, data_root=None,
                        pins: dict | None = None) -> None:
    """Aggregation identity: plotted means == a fresh re-read of the cells.

    Replaces the pre-ADR assertion against
    ``tables/four_tier_xi_summary.csv`` (no counterpart in the ADR-0003
    archive).  Each point is recomputed here by re-opening its own per-cell
    CSV and averaging the raw metric column over the 20 seed rows -- a
    second route to the same number, independent of ``series_rows``.
    """
    root = data_root_path(data_root)
    meta_layout = layout_of(root)
    means = series[series["statistic"] == "mean"]
    if meta_layout == "adr0003":
        V = _value_table(series)
        checked = set()
        for f in sorted((root / "cells").glob("*.csv")):
            d = pd.read_csv(f)
            _require(len(d) == SEEDS_PER_CELL, f"{f.name}: {len(d)} rows")
            _require(d["tier"].nunique() == 1, f"{f.name}: mixed tier column")
            _require(d["xi"].nunique() == 1, f"{f.name}: mixed xi column")
            tier = str(d["tier"].iloc[0])
            xi = float(np.round(d["xi"].iloc[0], 2))
            for panel, col in VALUE_COL.items():
                key = (panel, tier, xi)
                _require(key in V, f"{f.name}: no plotted point for {key}")
                ref = float(d[col].mean())
                _require(abs(ref - V[key]) <= 1e-12,
                         f"cell-mean drift {tier} xi={xi} panel={panel}: "
                         f"plotted {V[key]:.12g} vs cell {ref:.12g}")
                checked.add(key)
        _require(len(checked) == len(V) == 2 * len(TIERS) * len(set(
            float(np.round(pd.read_csv(f)["xi"].iloc[0], 2))
            for f in (root / "cells").glob("*.csv"))),
            f"cell-mean identity covered {len(checked)} points, "
            f"plotted {len(V)}")
        return

    # pre-ADR: keep the archived tier table identity AND re-read the runs
    four = pd.read_csv(root / "bug0006_remeasure_v2" / "tables"
                       / "four_tier_xi_summary.csv")
    for _, r in means.iterrows():
        tid = LEGACY_TABLE_TIER[r["tier"]]
        q = four[(four["tier"] == tid) & (np.round(four["xi"], 2) == r["xi"])]
        _require(len(q) == 1, f"four_tier_xi_summary: no row {tid} xi={r['xi']}")
        ref = float(q[VALUE_COL[r["panel"]]].iloc[0])
        _require(abs(ref - float(r["value"])) <= 1e-9,
                 f"tier-table drift {tid} xi={r['xi']} panel={r['panel']}: "
                 f"{r['value']:.12g} vs {ref:.12g}")


# --------------------------------------------------------------- drawing ---
def _bands(ax) -> None:
    """F6's exact two-stage bands -- shaded regions, no boundary rules."""
    ax.axvspan(*BAND_STAGE1, color=PALETTE["gray_light"], alpha=0.16, zorder=0)
    ax.axvspan(*BAND_KINK, facecolor=PALETTE["purple_light"], alpha=0.16,
               hatch="///", edgecolor=PALETTE["gray"], linewidth=0, zorder=0)
    ax.axvspan(*BAND_STAGE2, color=PALETTE["gray_light"], alpha=0.32, zorder=0)


def _plotted_extent(ax) -> tuple[float, float]:
    vals = visual_qa._series_values(ax)
    parts = [v for k in ("line_values", "line_endpoints", "bar_values")
             for v in vals[k] if len(v)]
    if not parts:
        return 0.0, 1.0
    allv = np.concatenate(parts)
    return float(allv.min()), float(allv.max())


def _auto_ylim(ax, include=(), pad_lo: float = 0.08,
               pad_hi: float = 0.10, floor: float | None = None
               ) -> tuple[float, float]:
    """ylim from the rendered values (incl. error endpoints) + headroom."""
    lo, hi = _plotted_extent(ax)
    for v in include:
        lo, hi = min(lo, float(v)), max(hi, float(v))
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    lo, hi = lo - pad_lo * span, hi + pad_hi * span
    if floor is not None:
        lo = max(lo, float(floor))
    ax.set_ylim(lo, hi)
    return lo, hi


def _display_segments(ax) -> list:
    """Plotted geometry in DISPLAY pixels (bars, markers, error bars, refs).

    ``visual_qa._segments_from_artist`` returns data coordinates for lines
    and error bars, which cannot be compared against a display-space text
    bbox; this helper applies ``ax.transData`` first so the text-vs-data
    contract actually binds.
    """
    art = visual_qa.collect_panel_artists(ax)
    tr = ax.transData
    segs: list = []

    def _pts(xy):
        xy = np.asarray(xy, dtype=float)
        if xy.ndim == 1:
            xy = xy.reshape(-1, 2)
        return tr.transform(xy)

    for a in art["lines"]:
        pts = _pts(a.get_xydata())
        if len(pts) == 1:
            segs.append((tuple(pts[0]), tuple(pts[0])))
        else:
            segs.extend(zip(pts[:-1], pts[1:]))
    for cont in art["errorbars"]:
        dl, caps, bcs = visual_qa._eb_parts(cont)
        for bc in bcs:
            for s in bc.get_segments():
                p = _pts(s)
                segs.append((tuple(p[0]), tuple(p[-1])))
        for cap in caps:
            p = _pts(cap.get_xydata())
            segs.append((tuple(p[0]), tuple(p[-1] if len(p) > 1 else p[0])))
        if dl is not None:
            p = _pts(dl.get_xydata())
            if len(p) == 1:
                segs.append((tuple(p[0]), tuple(p[0])))
    for bc in art["bars"]:
        for patch in bc.patches:
            bb = patch.get_window_extent()
            segs.append(((bb.x0, bb.y0), (bb.x1, bb.y1)))
    for ln in art["refs"]:
        p = _pts(ln.get_xydata())
        segs.extend(zip(p[:-1], p[1:]))
    return segs


def _text_hits_data(ax, t) -> bool:
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = t.get_window_extent(renderer=r)
    return any(visual_qa._seg_hits_box(s, bb) for s in _display_segments(ax))


def audit_text_over_data(fig, axes) -> None:
    """Local strict text-vs-data gate in display space (zero tolerance)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    offenders = []
    for ax in axes:
        segs = _display_segments(ax)
        ab = ax.bbox
        for t in visual_qa._text_artists(fig, ax):
            if t is ax.title or t is ax.xaxis.label or t is ax.yaxis.label:
                continue
            bb = visual_qa._bbox(t, r)
            if bb is None:
                continue
            if bb.x1 < ab.x0 or bb.x0 > ab.x1 or bb.y1 < ab.y0 or bb.y0 > ab.y1:
                continue
            if any(visual_qa._seg_hits_box(s, bb) for s in segs):
                offenders.append(f"panel[{ax.get_title()[:24]}]: "
                                 f"{t.get_text()[:40]!r} over plotted data")
    _require(not offenders, "; ".join(offenders))


def _text_hits_text(ax, t) -> bool:
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = t.get_window_extent(renderer=r)
    others = [x for x in visual_qa._text_artists(fig, ax) if x is not t]
    for o in others:
        ob = visual_qa._bbox(o, r)
        if ob is not None and visual_qa._boxes_overlap(bb, ob):
            return True
    return False


def _anchors() -> tuple:
    """Ordered in-axes anchors: corners first (top-right preferred), then the
    mid band.  Every anchor is TESTED, never trusted -- the placement gate
    decides which one is actually free."""
    top = (0.97, 0.91, 0.85, 0.79, 0.73, 0.67, 0.61)
    bot = (0.03, 0.09, 0.15, 0.21, 0.27, 0.33)
    mid = (0.62, 0.58, 0.55, 0.49, 0.43, 0.37)
    out = ([(0.97, y, "right", "top") for y in top]
           + [(0.03, y, "left", "top") for y in top]
           + [(0.97, y, "right", "bottom") for y in bot]
           + [(0.03, y, "left", "bottom") for y in bot]
           + [(0.97, y, "right", "top") for y in mid]
           + [(0.03, y, "left", "top") for y in mid])
    return tuple(out)


def _place_text(ax, text: str, candidates=None, *, zorder: int = 6,
                sizes=(8, 7.4, 6.8, 6.4)):
    """First (anchor, size) where the block touches no data and no text.

    The canvas is drawn ONCE per call: segments and the other text boxes do
    not move while a candidate is added and removed, so the search is a
    static-geometry test, and the final ``audit_text_over_data`` re-checks
    the rendered canvas afterwards.
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    segs = _display_segments(ax)
    others = [bb for bb in (visual_qa._bbox(o, r)
                            for o in visual_qa._text_artists(fig, ax))
              if bb is not None]
    for size in sizes:
        for xf, yf, ha, va in (candidates or _anchors()):
            t = ax.text(xf, yf, text, transform=ax.transAxes,
                        fontsize=size, color=PALETTE["gray"],
                        ha=ha, va=va, zorder=zorder,
                        bbox=dict(boxstyle="square,pad=0.22",
                                  facecolor="white", alpha=0.72,
                                  edgecolor="none"))
            raw = t.get_window_extent(renderer=r)
            bb = type(raw).from_extents(raw.x0 - 3.0, raw.y0 - 3.0,
                                        raw.x1 + 3.0, raw.y1 + 3.0)
            hit = any(visual_qa._seg_hits_box(s, bb) for s in segs) or any(
                visual_qa._boxes_overlap(bb, ob) for ob in others)
            if not hit:
                return t
            t.remove()
    raise AssertionError(f"no free placement for callout {text[:40]!r}")


def _band_labels(ax, prefer=("top", "bottom")) -> None:
    """Stage labels, each placed where it touches neither data nor text."""
    slots = {"top": ((0.965, "top"), (0.90, "top")),
             "bottom": ((0.035, "bottom"), (0.10, "bottom"))}
    candidates = tuple(s for side in prefer for s in slots[side])
    for x, lab in ((0.45, "Stage 1"), (0.70, "kink"), (0.90, "Stage 2")):
        for yf, va in candidates:
            t = ax.text(x, yf, lab, transform=ax.get_xaxis_transform(),
                        ha="center", va=va, fontsize=HOUSE_STYLE["tick_size"],
                        color=PALETTE["gray"], zorder=5)
            if not _text_hits_data(ax, t) and not _text_hits_text(ax, t):
                break
            t.remove()
        else:
            raise AssertionError(f"no free slot for band label {lab!r}")


def place_figure_legend(fig, handles, labels, ncol: int, note_text, *,
                        gap: float = 8.0):
    """Figure legend between the x-axis labels and the footnote (measured)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    xl = min(ax.xaxis.label.get_window_extent(r).y0 for ax in fig.axes)
    fn = note_text.get_window_extent(r).y1
    y_top = (xl - gap) / fig.bbox.height
    leg = fig.legend(handles, labels, loc="upper center",
                     bbox_to_anchor=(0.5, y_top), ncol=ncol, frameon=False,
                     fontsize=HOUSE_STYLE["tick_size"] - 1, handlelength=2.2,
                     columnspacing=1.7)
    fig.canvas.draw()
    bb = leg.get_window_extent(r)
    _require(bb.y0 > fn + 1.0,
             f"figure legend overlaps the footnote ({bb.y0:.0f} <= {fn:.0f} px)")
    _require(bb.y1 < xl - 1.0,
             "figure legend overlaps the x-axis labels")
    fb = fig.bbox
    _require(bb.x0 >= fb.x0 + 1.0 and bb.x1 <= fb.x1 - 1.0,
             f"figure legend runs outside the canvas ({bb.x0:.0f}..{bb.x1:.0f} "
             f"of {fb.x0:.0f}..{fb.x1:.0f} px)")
    return leg


def _plot_tier(ax, s: pd.DataFrame, tier: str, xis) -> None:
    d = s[s["tier"] == tier].sort_values("xi")
    _require(len(d) == len(xis), f"{tier}: {len(d)} points, expected {len(xis)}")
    st = STYLE[tier]
    ax.errorbar(d["xi"].to_numpy(float), d["value"].to_numpy(float),
                yerr=np.vstack([d["value"] - d["ci_lo"],
                                d["ci_hi"] - d["value"]]),
                color=st["color"], linestyle=st["ls"], marker=st["marker"],
                markersize=4.6, linewidth=1.7, capsize=2.0, elinewidth=0.85,
                label=LABEL[tier], markerfacecolor=st["mfc"],
                markeredgecolor=st["color"], markeredgewidth=1.0, zorder=4)


# ---------------------------------------------------------------- figure ---
def build_figure(data_root=None):
    """Build the figure in memory: returns (fig, sidecar, runs, pins, meta)."""
    runs, pins, meta = load_runs(data_root)
    xis = list(meta["xis"])
    series = series_rows(runs, xis)
    check_against_cells(series, data_root, pins)
    anns = annotation_rows(series, runs, pins, meta)
    sidecar = pd.concat([series, anns], ignore_index=True)
    A = {r["metric"]: float(r["value"]) for _, r in anns.iterrows()}
    lo_knot = _knot(xis, min(xis))
    hi_knot = _knot(xis, REPORT_HI_KNOT)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.4, 5.9))
    fig.subplots_adjust(left=0.062, right=0.988, top=0.845, bottom=0.40,
                        wspace=0.21)
    fig.text(0.5, 0.955,
             "F6_v2 · Two-stage proxy failure — macro harvest and strictly "
             "like-for-like time-mean alignment, five tiers on the ADR-0003 "
             "clock (mgstd5)",
             ha="center", va="top",
             fontsize=HOUSE_STYLE["title_size"] + 1)

    for ax, panel, title, ylabel in (
            (ax_a, "a", "A · Macro efficiency: harvested knowledge share",
             "harvested knowledge share\n(final step · run mean)"),
            (ax_b, "b", "B · Like-for-like signal alignment (ideal channel)",
             "corr(prestige, ideal merit)\n(time mean · run mean)")):
        _bands(ax)
        s_metric = series[(series["metric"] == PANELS[panel])]
        for tier in TIERS:
            _plot_tier(ax, s_metric, tier, xis)
        ax.set_xlim(0.02, 1.0)
        ax.set_xticks([x for x in (0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9) if x in xis]
                      or [min(xis), 0.3, 0.5, 0.7, max(xis)])
        _cosmetics(ax, HOUSE_STYLE, title,
                   "visibility weight  $\\xi = 1 - c$", ylabel)

    # headroom is a layout quantity here: the top band carries the stage
    # labels in A and the computed five-tier dialogue in B
    _auto_ylim(ax_a, pad_lo=0.10, pad_hi=0.20)
    # panel B's lower pad is a layout allowance: the computed correlation
    # callout sits in the clean strip under the zero rule (both archives)
    _auto_ylim(ax_b, include=(0.0,), pad_lo=0.20, pad_hi=0.16)
    ax_b.axhline(0.0, color=PALETTE["gray"], linewidth=1.0, zorder=2)
    ticks_b = [t for t in ax_b.get_yticks()
               if ax_b.get_ylim()[0] < t < ax_b.get_ylim()[1]]
    ax_b.set_yticks(ticks_b)

    # callouts first (they need the largest free block), then band labels
    _place_text(ax_a, (
        f"entry arms, ξ ≤ 0.5: {A['entry_plateau_harvested_min'] * 100:.1f}–"
        f"{A['entry_plateau_harvested_max'] * 100:.1f}%  "
        f"(M1 fixed {A['m1_fixed_harvested_min'] * 100:.1f}–"
        f"{A['m1_fixed_harvested_max'] * 100:.1f}%, M2 lower)\n"
        f"Stage-2 collapse (ξ = {hi_knot:g}): "
        f"{A['stage2_harvest_min'] * 100:.1f}–"
        f"{A['stage2_harvest_max'] * 100:.1f}% (entry arms); "
        f"M0 {A['m0_xi09_harvested'] * 100:.1f}%\n"
        f"age_hazard_staggered {A['staggered_xi09_harvested'] * 100:.1f}% "
        f"({A['staggered_rescue_pp']:+.1f} pp vs M1 fixed)"))

    rows_b = [f"{SHORT[t]:<13s} {A[f'c_tm_xi01::{t}']:+.3f} → "
              f"{A[f'c_tm_xi09::{t}']:+.3f}" for t in TIERS]
    others = [A[f"corr_xi_ctm_run::{t}"] for t in TIERS if t != "M2_l0"]
    header = (f"c_ideal_tm, ξ = {_knot(xis, min(xis)):g} → {hi_knot:g}")
    _place_text(ax_b, "\n".join([header, *rows_b]))
    _place_text(ax_b, (
        f"run-level Corr(ξ, c_tm)\nM2 λ = 0: "
        f"{A['corr_xi_ctm_run::M2_l0']:+.3f} (only positive)\n"
        f"(others {max(others):+.3f}…{min(others):+.3f})"))
    # band labels once, in panel A: panel B's top-right band carries the
    # computed dialogue, and both panels share the same shading
    _band_labels(ax_a, prefer=("top", "bottom"))

    src = data_root_path(data_root)
    rel = src.relative_to(REPO) if src.is_relative_to(REPO) else src
    hz = {t: pins[t]["horizon"] for t in TIERS}
    exits = {t: pins[t]["exit_mode"] for t in TIERS}
    g_pins = sorted({pins[t]["g"] for t in TIERS})
    if meta["layout"] == "adr0003":
        lines_note = [
            f"Sources (read-only): {rel}/cells — {meta['cells']} cells x "
            f"{SEEDS_PER_CELL} seeds = {meta['runs']}, {len(xis)}-point xi grid, "
            f"{LANDSCAPE}, visibility reward on, logit + equal_split. ADR-0003 "
            f"clock: 1 step = 0.5 yr, tau_max = {meta['tau_max']} in every cell "
            f"(40-yr careers), prestige vanishing gamma = 0.5^(1/10) = "
            f"{meta['gamma']:.3f}, entry g = {pins['M0']['g']:g} (M0, no exit) / "
            f"{pins['M1_fixed']['g']:g} in every entry tier (ADR-0003 grid "
            f"0.125/0.25/0.5). Error bars: 95% CI = mean +/- 1.96 SE over "
            f"{SEEDS_PER_CELL} seeds.",
            f"Tier definitions read from the cells: M0 exit_mode none, g = "
            f"{pins['M0']['g']:g}; M1 exit_mode fixed_span / "
            f"age_hazard_staggered, g = {pins['M1_fixed']['g']:g}; M2 "
            f"exit_mode fixed_span, g = {pins['M2_l0']['g']:g}, dyadic pull = "
            f"{pins['M2_l0']['dyadic_move_pull']:g}, lambda = 0 / 1. Horizons "
            f"read from each cell's horizon column: M0/M2 {hz['M0']} steps, "
            f"both M1 tiers {hz['M1_fixed']} steps — the M1 levels are therefore "
            "not horizon-comparable with M0/M2 (shape and ordering only); "
            "fixed_span pins the population at N = 20 while "
            "age_hazard_staggered can grow, so that contrast is regime-level.",
            f"Stage 1 (xi <= 0.6), welfare kink [0.6, 0.8], Stage 2 (xi >= 0.8) "
            "are shaded regions only; no boundary rule crosses a series. Panel B "
            "plots the SAME statistic in every arm (time-mean c_ideal_tm), so no "
            "time-mean / final-step mixing. The ideal channel is not monotone "
            f"everywhere: M2 lambda = 0 rises "
            f"{A['m2_l0_alignment_rise']:+.3f} from xi = {min(xis):g} to "
            f"{hi_knot:g} (min {A['m2_l0_alignment_min']:+.3f}) and is the only "
            f"tier with a positive run-level Corr(xi, c_tm) = "
            f"{A['corr_xi_ctm_run::M2_l0']:+.3f}; the panel-B callout carries it "
            "as one computed number. Model-internal, not calibrated (owner "
            "ruling 2026-11-13).",
        ]
    else:
        exit_str = ", ".join(f"{SHORT[t]} {exits[t]}" for t in TIERS)
        lines_note = [
            f"Sources (read-only, PRE-ADR archives selected via data_root): "
            f"{rel}/bug0006_remeasure_v2 (w_v05, m2_xi) and "
            f"{rel}/turnover_v2 — {meta['runs']} runs over {len(xis) * 5} cells, "
            f"{SEEDS_PER_CELL} seeds each, {LANDSCAPE}, visibility reward on. "
            "OLD clock (gamma = 0.5 / 0.9, tau_max = 200, g = 0 / 0.1); this is "
            "the before/after comparison build, not the ADR-0003 figure. Error "
            "bars: 95% CI = mean +/- 1.96 SE over seeds.",
            f"Tier definitions read from the archives: {exit_str}; entry g = "
            f"{', '.join(f'{g:g}' for g in g_pins)}. Horizons read from the "
            f"archives: " + ", ".join(f"{SHORT[t]} {hz[t]}" for t in TIERS)
            + " steps — levels across horizons are not directly comparable.",
            "Stage 1 (xi <= 0.6), welfare kink [0.6, 0.8], Stage 2 (xi >= 0.8) "
            "are shaded regions only. Panel B plots the SAME statistic in every "
            "arm (time-mean c_ideal_tm). Model-internal, not calibrated.",
        ]
    note = fig.text(0.008, 0.006, "\n".join(textwrap.fill(t, width=178)
                                            for t in lines_note),
                    fontsize=HOUSE_STYLE["note_size"], color=PALETTE["gray"],
                    va="bottom")
    handles, labels = ax_a.get_legend_handles_labels()
    place_figure_legend(fig, handles, labels, len(TIERS), note)
    audit_text_over_data(fig, (ax_a, ax_b))
    visual_qa.assert_figure_clean(fig, [
        {"ax": ax_a, "metric": PANELS["a"], "expect_lines": len(TIERS),
         "points_per_line": len(xis), "sidecar": sidecar,
         "filters": {"statistic": "mean"}},
        {"ax": ax_b, "metric": PANELS["b"], "expect_lines": len(TIERS),
         "points_per_line": len(xis), "sidecar": sidecar,
         "filters": {"statistic": "mean"}},
    ])
    return fig, sidecar, runs, pins, meta


def build(outdir="output/report_notebook_v1", data_root=None) -> dict:
    """Render + persist the figure triple; ``data_root`` defaults to ADR-0003."""
    fig, sidecar, _runs, _pins, _meta = build_figure(data_root)
    out = Path(outdir)
    if not out.is_absolute():
        out = REPO / out
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
