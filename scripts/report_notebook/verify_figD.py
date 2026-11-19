"""Independent verifier for Figure D (F13_paradigm_refill_regimes).

``check()`` recomputes every plotted quantity and annotation in
``F13_paradigm_refill_regimes_data.csv`` from the RAW cell CSVs
(``data/interventions_v1/cells/cellC_*.csv``), asserts the expected input
counts (18 cells x 20 seeds = 360 runs; exactly the 3 refill regimes) and
compares at ``tol``.

It then re-runs the rendered-canvas contract: the figure is rebuilt in memory
via the builder's ``_figure`` helper and put through
``visual_qa.assert_figure_clean`` (artist counts incl. the panel-C diamond
series, plotted-vs-sidecar min/max, ylim headroom, text/text-title/text-data
collisions, legend duplicates) and ``assert_text_fits``.  A sidecar that
survives the numeric check but renders a collision (the v2 failure mode:
rotated labels into the panel-B title, callouts over bars) therefore fails
here.

``data_root`` accepts the archive directory (default
``data/interventions_v1/cells``) or its parent; both the recomputation and the
rebuild use it.  Empty return list = all checks pass.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FNAME = "F13_paradigm_refill_regimes"

REGIMES = ("off", "occupied", "frontier")
GS = (0.0, 0.1)
METRIC_COL = {
    "explored grid share": "explored_final",
    "mean residual knowledge on grid": "avg_knowledge_on_grid",
    "Corr(prestige, ideal merit) (final step)": "corr_prestige_ideal_merit_final",
    "Corr(prestige, ideal merit) (time mean)": "Mean_of_Corr_prestige_ideal_merit",
    "top-10% prestige share": "top10_final",
    "relative newcomer gap ((newcomer-incumbent)/incumbent)": "gap_ratio",
}


def _load(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("cellC_*.csv"))
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["source_file"] = f.name
        frames.append(d)
    c = pd.concat(frames, ignore_index=True)
    c["gap_ratio"] = ((c["newcomer_prestige_mean"] - c["incumbent_prestige_mean"])
                      / c["incumbent_prestige_mean"])
    return c


def _rerun_visual_contract(data_root, fails: list[str]) -> None:
    """Rebuild the figure in memory and assert the rendered canvas is clean."""
    import matplotlib.pyplot as plt

    import figD_paradigm_refill_regimes as builder
    import visual_qa
    from turnover_lifecycle_figure import assert_text_fits

    fig = None
    try:
        c = builder.load_runs(data_root)
        side = pd.concat([builder.series_rows(c, data_root),
                          builder.annotation_rows(c, data_root)],
                         ignore_index=True)
        fig, panels = builder._figure(side)
        w, h = fig.get_size_inches() * fig.dpi
        assert_text_fits(fig, w, h)
        visual_qa.assert_figure_clean(fig, panels)
    except AssertionError as exc:
        fails.append(f"visual contract: {exc}")
    except Exception as exc:  # noqa: BLE001 - a crash is a failure, never a pass
        fails.append(f"visual contract raised: {exc!r}")
    finally:
        if fig is not None:
            plt.close(fig)


def check(outdir: str = "output/report_notebook_v1", data_root=None,
          tol: float = 1e-9) -> list[str]:
    import figD_paradigm_refill_regimes as builder

    fails: list[str] = []
    root = builder.data_root_path(data_root)
    out = (REPO / outdir) if not Path(outdir).is_absolute() else Path(outdir)
    sidecar_path = out / f"{FNAME}_data.csv"
    if not sidecar_path.exists():
        return [f"missing sidecar: {sidecar_path}"]
    for ext in ("png", "pdf"):
        p = out / f"{FNAME}.{ext}"
        if not p.exists():
            fails.append(f"missing figure file: {p}")
    side = pd.read_csv(sidecar_path)
    c = _load(root)

    # ---- input counts -------------------------------------------------
    files = sorted(root.glob("cellC_*.csv"))
    if len(files) != 18:
        fails.append(f"cellC files: expected 18, got {len(files)}")
    if len(c) != 360:
        fails.append(f"runs: expected 360, got {len(c)}")
    if sorted(c["refill_mode"].unique()) != sorted(REGIMES):
        fails.append(f"refill_mode must be exactly {REGIMES}, got "
                     f"{sorted(c['refill_mode'].unique())}")
    if sorted(c["agent_generation_time"].unique().tolist()) != list(GS):
        fails.append("agent_generation_time grid changed")

    def cell(regime: str, g: float) -> pd.DataFrame:
        return c[(c["refill_mode"] == regime) & (c["agent_generation_time"] == g)]

    # ---- series rows --------------------------------------------------
    srows = side[side["statistic"] == "mean"]
    for _, r in srows.iterrows():
        col = METRIC_COL.get(r["metric"])
        if col is None:
            fails.append(f"unknown metric in sidecar: {r['metric']}")
            continue
        d = cell(str(r["refill_mode"]), float(r["g_entry"]))[col]
        if int(r["n"]) != len(d) or len(d) != 60:
            fails.append(f"n mismatch {r['metric']} {r['refill_mode']} "
                         f"{r['g_entry']}: raw {len(d)} vs sidecar {r['n']}")
            continue
        mean = float(d.mean())
        ci = 1.96 * float(d.std(ddof=1)) / np.sqrt(len(d))
        for label, got, want in (("value", r["value"], mean),
                                 ("ci_lo", r["ci_lo"], mean - ci),
                                 ("ci_hi", r["ci_hi"], mean + ci)):
            if abs(float(got) - want) > tol:
                fails.append(f"{r['metric']} {r['refill_mode']} "
                             f"{r['g_entry']} {label}: sidecar {got} vs "
                             f"recomputed {want}")

    # ---- annotation rows ---------------------------------------------
    ann = side[side["statistic"] == "annotation"].set_index("metric")
    m1 = c[c["agent_generation_time"] == 0.1]

    def m1mean(col: str, regime: str) -> float:
        return float(m1[m1["refill_mode"] == regime][col].mean())

    ex_off = m1mean("explored_final", "off")
    expected = {
        "occupied_exploration_collapse_pct_M1":
            100.0 * (m1mean("explored_final", "occupied") / ex_off - 1.0),
        "frontier_exploration_pct_M1":
            100.0 * m1mean("explored_final", "frontier"),
        "frontier_residual_ratio_vs_off_M1":
            m1mean("avg_knowledge_on_grid", "frontier")
            / m1mean("avg_knowledge_on_grid", "off"),
        "occupied_residual_ratio_vs_off_M1":
            m1mean("avg_knowledge_on_grid", "occupied")
            / m1mean("avg_knowledge_on_grid", "off"),
        "mean_prestige_inflation_occupied_M1":
            m1mean("mean_prestige", "occupied") / m1mean("mean_prestige", "off"),
        "mean_prestige_inflation_frontier_M1":
            m1mean("mean_prestige", "frontier") / m1mean("mean_prestige", "off"),
    }
    for regime in REGIMES:
        expected[f"corr_tm_M1_{regime}"] = m1mean(
            "Mean_of_Corr_prestige_ideal_merit", regime)
        expected[f"corr_final_M1_{regime}"] = m1mean(
            "corr_prestige_ideal_merit_final", regime)

    for key, want in expected.items():
        if key not in ann.index:
            fails.append(f"missing annotation row: {key}")
            continue
        got = float(ann.loc[key, "value"])
        if abs(got - want) > tol:
            fails.append(f"annotation {key}: sidecar {got} vs recomputed {want}")
    extra = set(ann.index) - set(expected)
    if extra:
        fails.append(f"unexpected annotation rows: {sorted(extra)}")

    # ---- rendered-canvas contract (visual_qa, re-run independently) ----
    _rerun_visual_contract(data_root, fails)
    return fails


if __name__ == "__main__":
    res = check()
    for f in res:
        print("FAIL:", f)
    print("all checks pass" if not res else "FAILURES PRESENT")
