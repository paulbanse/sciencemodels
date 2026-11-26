"""F2b_v2 - Career flows: per-capita harvest, alignment and population by entry
rate and exit rule (ADR-0003 clock, mgstd5).

Section 4.3 of the paper sketch.  Three panels against the entry rate g, split
by the ADR-0003 exit rule (``fixed_span`` vs ``age_hazard_staggered``), pooled
over curiosity (3 c x 20 seeds = 60 runs per point):

  (a) per-capita harvest  = harvested_fin / career_pop_final (terminal stock)
  (b) terminal Corr(prestige, ideal merit)  (c_ideal_fin)
  (c) final population    (career_pop_final)

WHY THE PANELS CHANGED.  The pre-ADR figure split clean (``mgstd5``) from noisy
(``noisypr5``) terrain.  The ADR-0003 career arm is ``mgstd5`` only -- there are
no ``noisypr5`` cells in ``data/career_flows_adr0003`` -- so that contrast
cannot be drawn from this archive and is not imitated.  The honest substitute
is the exit-rule split, which the new arm does carry and which carries the
ADR-0003 qualification (ii): under ``fixed_span`` the population is pinned
(one retirement per arrival), so per-capita harvest is FLAT in g, while under
``age_hazard_staggered`` the population grows and per-capita harvest falls.

ARCHIVE NAMING.  The ADR-0003 exit modes are ``fixed_span`` and
``age_hazard_staggered`` (pre-ADR names: ``none`` / ``age_hazard``); the entry
grid moved from g in {0.05, 0.1, 0.2} to {0.125, 0.25, 0.5}; one step is 0.5 yr
and a career is 40 yr (tau_max = 80).

``build(outdir, data_root)``: ``data_root`` defaults to the ADR-0003 archive;
pass ``data_root="data/career_flows_v2"`` to regenerate the pre-ADR figure
(``careerB_*.csv`` trajectories + ``*_career.csv`` sidecars).  That archive
has two landscapes and one exit rule, so the split becomes landscape there.

Model-internal, NOT calibrated.

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
        scripts/report_notebook/fig_F2b_v2_career_flows.py
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

BASENAME = "F2b_v2_career_flows_noisy_vs_clean"
DATA_ROOT = REPO / "data" / "career_flows_adr0003"

EXIT_MODES = ("fixed_span", "age_hazard_staggered")
LEGACY_GROUPS = ("mgstd5", "noisypr5")
Z95 = 1.96
PANELS = (("percap", "per-capita harvest"),
          ("align", "Corr(prestige, ideal merit)"),
          ("npop", "final population (agents)"))


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def data_root_path(data_root=None) -> Path:
    """Career-flow archive root (default: ``data/career_flows_adr0003``)."""
    root = Path(data_root) if data_root is not None else DATA_ROOT
    return root if root.is_absolute() else (REPO / root)


def _mean_ci(v) -> tuple[float, float, float, int]:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return m, m - Z95 * se, m + Z95 * se, n


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def _load_cells(root: Path) -> pd.DataFrame:
    files = sorted((root / "cells").glob("career_*.csv"))
    _require(len(files) == 18,
             f"{root}/cells: 18 career cells expected (3 c x 3 g x 2 exit), "
             f"found {len(files)}")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    _require(len(d) == 18 * 20, f"career cells: {len(d)} seed rows, expected 360")
    _require(sorted(d.exit_mode.unique().tolist()) == sorted(EXIT_MODES),
             f"exit grid {sorted(d.exit_mode.unique())}")
    _require(sorted(d.landscape.unique().tolist()) == ["mgstd5"],
             f"career arm must be mgstd5 only, got "
             f"{sorted(d.landscape.unique())}")
    _require(sorted(d.curiosity.unique().tolist()) == [0.3, 0.5, 0.7],
             f"curiosity grid {sorted(d.curiosity.unique())}")
    # per-capita harvest is the terminal harvested stock per surviving agent
    rel = (d.harvested_fin / d.career_pop_final
           - d.career_percap_harvest_final).abs().max()
    _require(rel < 1e-9, f"per-capita harvest identity violated (max {rel:g})")
    _require((d.n_final == d.career_pop_final).all(),
             "career_pop_final != n_final")
    out = d.assign(group=d.exit_mode,
                   percap=d.career_percap_harvest_final,
                   align=d.c_ideal_fin,
                   npop=d.career_pop_final,
                   align_tm=d.c_ideal_tm)
    return out


def _load_legacy(root: Path) -> pd.DataFrame:
    traj_files = sorted(p for p in root.glob("careerB_*.csv")
                        if not p.name.endswith("_career.csv"))
    _require(len(traj_files) == 18,
             f"{root}: 18 careerB_* trajectory files expected, "
             f"found {len(traj_files)}")
    frames = []
    for f in traj_files:
        side = f.with_name(f.stem + "_career.csv")
        _require(side.is_file(), f"missing career sidecar {side.name}")
        t = pd.read_csv(f)
        k = pd.read_csv(side)
        last_t = t[t.Step == t.Step.max()]
        last_k = k[k.Step == k.Step.max()]
        _require(len(last_t) == len(last_k),
                 f"{f.name}: {len(last_t)} terminal traj rows vs "
                 f"{len(last_k)} career rows")
        frames.append(pd.DataFrame({
            "cell_tag": [t.cell_tag.iloc[0]] * len(last_t),
            "curiosity": last_t.curiosity.to_numpy(float),
            "landscape": last_t.landscape.to_numpy(),
            "g": last_t.agent_generation_time.to_numpy(float),
            "seed": last_t.seed.to_numpy(),
            "percap": last_k.percap_harvest_cum.to_numpy(float),
            "align": last_t.Corr_prestige_ideal_merit.to_numpy(float),
            "npop": last_k.n_agents.to_numpy(float),
            "align_tm": np.nan,
        }))
    d = pd.concat(frames, ignore_index=True).assign(group=lambda x: x.landscape)
    _require(d.groupby("g").size().nunique() == 1, "unbalanced g cells")
    return d


def load(data_root=None) -> tuple[pd.DataFrame, str]:
    root = data_root_path(data_root)
    if sorted((root / "cells").glob("career_*.csv")):
        return _load_cells(root), "adr0003"
    if sorted(root.glob("careerB_*.csv")):
        return _load_legacy(root), "pre-adr"
    raise AssertionError(f"{root}: not a career-flow archive")


def summarise(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (grp, g), sub in d.groupby(["group", "g"], sort=True):
        rec = {"group": grp, "g": float(g), "n": int(len(sub))}
        for key in ("percap", "align", "npop", "align_tm"):
            if sub[key].notna().sum() < 2:
                rec[key] = rec[key + "_lo"] = rec[key + "_hi"] = float("nan")
                continue
            m, lo, hi, n = _mean_ci(sub[key])
            rec[f"{key}"] = m
            rec[f"{key}_lo"] = lo
            rec[f"{key}_hi"] = hi
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values(["group", "g"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def _sidecar(st: pd.DataFrame, layout: str, src: str) -> pd.DataFrame:
    rows = []
    keymap = {"percap": "f2b_percap", "align": "f2b_align", "npop": "f2b_npop"}
    stat = {"percap": "mean_harvest_per_agent",
            "align": "mean_corr_prestige_ideal",
            "npop": "mean_agents"}
    for key, metric in keymap.items():
        for r in st.itertuples():
            rows.append({
                "figure": BASENAME, "panel": key, "metric": metric,
                "statistic": stat[key], "series": r.group, "group": r.group,
                "g": r.g, "value": float(getattr(r, key)),
                "ci_lo": float(getattr(r, f"{key}_lo")),
                "ci_hi": float(getattr(r, f"{key}_hi")), "n": int(r.n),
                "source_file": src,
                "note": f"{layout}: curiosity-pooled mean over {r.n} runs, "
                        "95% CI = mean +/- 1.96 SE"})
    for r in st.itertuples():
        if not np.isfinite(r.align_tm):
            continue
        rows.append({
            "figure": BASENAME, "panel": "align_tm", "metric": "f2b_align_tm",
            "statistic": "mean_corr_prestige_ideal", "series": r.group,
            "group": r.group, "g": r.g, "value": float(r.align_tm),
            "ci_lo": float(r.align_tm_lo), "ci_hi": float(r.align_tm_hi),
            "n": int(r.n), "source_file": src,
            "note": "time-mean alignment variant (not plotted)"})
    return pd.DataFrame(rows)


def build(outdir="output/report_notebook_v1", data_root=None) -> dict:
    root = data_root_path(data_root)
    d, layout = load(data_root)
    st = summarise(d)
    canonical = EXIT_MODES if layout == "adr0003" else LEGACY_GROUPS
    present = set(d.group.tolist())
    groups = [g for g in canonical if g in present]
    _require(len(groups) == len(present),
             f"unexpected series {sorted(present - set(groups))}")
    g_levels = sorted(d.g.unique().tolist())
    n_tot = int(len(d))

    s = {**T.HOUSE_STYLE, "figsize": (9.9, 4.9)}
    fig, axes = plt.subplots(1, 3, figsize=s["figsize"])
    fig.subplots_adjust(left=0.095, right=0.985, top=0.752, bottom=0.275,
                        wspace=0.28)
    colors = {g: T.PALETTE[c] for g, c in
              zip(groups, ("purple", "orange", "green", "blue"))}
    markers = {g: m for g, m in zip(groups, ("o", "s", "D", "^"))}
    pos = np.arange(len(g_levels), dtype=float)

    handles = []
    for ax, (key, ylabel) in zip(axes, PANELS):
        lo_all = float(st[f"{key}_lo"].min())
        hi_all = float(st[f"{key}_hi"].max())
        vspan = hi_all - lo_all
        for gi, grp in enumerate(groups):
            sub = st[st.group == grp].sort_values("g").reset_index(drop=True)
            _require(len(sub) == len(g_levels), f"{grp}: g grid")
            ln = ax.errorbar(
                pos, sub[key],
                yerr=[sub[key] - sub[key + "_lo"], sub[key + "_hi"] - sub[key]],
                fmt=markers[grp] + "-", color=colors[grp], markersize=5,
                linewidth=1.9, capsize=3, elinewidth=1.2, label=grp)
            if key == "percap":
                handles.append(ln)
            fmt = {"percap": "{:.4f}", "align": "{:+.3f}",
                   "npop": "{:.0f}"}[key]
            vals = sub[key].to_numpy(float)
            # endpoint labels only: above a falling series, below a rising one,
            # pushed outside the line corridor at both ends of the axis
            # alternate the series' labels above/below so two curves that
            # nearly coincide never share a label band
            dy_b = 12.0 if gi % 2 == 0 else -12.0
            for x_, v, ha, dx in ((pos[0], vals[0], "right", -7.0),
                                  (pos[-1], vals[-1], "left", 7.0)):
                dy = dy_b
                ax.annotate(fmt.format(v), xy=(x_, v), xytext=(dx, dy),
                            textcoords="offset points", ha=ha, va="center",
                            fontsize=s["tick_size"] - 1.5, color=colors[grp])
        if key == "align":
            ax.axhline(0, color=T.PALETTE["gray"], linewidth=1.0)
            lo_all, hi_all = min(lo_all, 0.0), max(hi_all, 0.0)
            vspan = hi_all - lo_all
        # room for the endpoint labels above and below the extreme points
        ax.set_ylim(lo_all - 0.14 * vspan, hi_all + 0.10 * vspan)
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{g:g}" for g in g_levels])
        ax.set_xlim(-0.55, len(g_levels) - 1 + 0.55)
        T._cosmetics(ax, s, "", "entry rate g (newcomers per step)", ylabel)
        ax.yaxis.labelpad = 6.0

    # the per-capita panel carries the ADR-0003 qualification (ii): the
    # pinned (constant-N) group is identified from the data, never by order
    pc = st.pivot_table(index="g", columns="group", values="percap")
    nn = st.pivot_table(index="g", columns="group", values="npop")
    pinned = [g for g in groups if nn[g].nunique() == 1]
    moving = [g for g in groups if g not in pinned]
    if layout == "adr0003":
        _require(len(pinned) == 1 and len(moving) == 1,
                 f"expected one pinned and one growing exit mode, got "
                 f"pinned={pinned}, moving={moving}")
        pin, mov = pinned[0], moving[0]
        _require(float(nn[pin].iloc[-1]) <= float(nn[mov].iloc[0]),
                 "pinned population is not the smaller one")
        fig.suptitle(
            f"{pin} pins the population: per-capita harvest flat "
            f"({pc[pin].iloc[0]:.4f} \u2192 {pc[pin].iloc[-1]:.4f}); "
            f"{mov} falls "
            f"({pc[mov].iloc[0]:.4f} \u2192 {pc[mov].iloc[-1]:.4f})",
            fontsize=s["title_size"] - 1.0, y=0.982)
    else:
        fig.suptitle(
            "Pre-ADR career sweep: per-capita harvest falls with entry on "
            + " and ".join(groups), fontsize=s["title_size"] - 1.0, y=0.982)
    fig.legend(handles, list(groups), loc="upper center",
               bbox_to_anchor=(0.5, 0.915), ncol=len(groups),
               fontsize=s["tick_size"], frameon=False)

    if layout == "adr0003":
        split_note = ("The pre-ADR clean-vs-noisy landscape split is NOT drawn: "
                      "the ADR-0003 career arm is mgstd5 only. The exit-rule "
                      "split replaces it.")
    else:
        split_note = ("Pre-ADR archive: landscape split; one exit rule only, "
                      "so no exit-rule contrast.")
    note = textwrap.fill(
        f"source: {root.relative_to(REPO)}/cells/*.csv (c x g x group, 20 "
        f"seeds/cell, {n_tot} runs). ADR-0003 pins: 1 step = 0.5 yr, entry age "
        "25, tau_max = 80, gamma = 0.933, g in "
        "{" + ",".join(f"{g:g}" for g in g_levels) + "}, H = 200; per-capita "
        "harvest = harvested_fin / career_pop_final. " + split_note +
        (f" Qualification (ii): {pin} holds N at "
         f"{nn[pin].iloc[0]:.0f} (per-capita harvest flat, "
         f"{pc[pin].iloc[0]:.4f} \u2192 {pc[pin].iloc[-1]:.4f}) while {mov} "
         f"grows N {nn[mov].iloc[0]:.0f} \u2192 {nn[mov].iloc[-1]:.0f} "
         f"(per-capita harvest {pc[mov].iloc[0]:.4f} \u2192 "
         f"{pc[mov].iloc[-1]:.4f}). " if layout == "adr0003" else "") +
        "95% CI over runs; model-internal, NOT calibrated.", width=160)
    fig.text(0.006, 0.012, note, fontsize=s["note_size"],
             color=T.PALETTE["gray"], va="bottom", linespacing=1.45)

    sidecar = _sidecar(st, layout, f"{root.relative_to(REPO)}/cells/*.csv")
    panels = [{"ax": ax, "metric": f"f2b_{key}", "expect_lines": len(groups),
               "points_per_line": len(g_levels), "sidecar": sidecar}
              for ax, (key, _) in zip(axes, PANELS)]
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
    print(st.round(5).to_string(index=False))
    return {"png": str(png), "pdf": str(pdf), "data_csv": str(csv_path),
            "qa": report, "layout": layout, "sidecar_rows": len(sidecar)}


if __name__ == "__main__":
    build()
