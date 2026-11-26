"""D5 prototypes: spatial grid snapshots + temporal harvest/exploration.

Two exploratory figures for IE-0007 #7 and #8, built on ``Noisy2``:

1. **3-panel spatial snapshot** (IE-0007 #7) at ``t = 80`` for
   Stage 1 (``xi = 0.1``, scattered), the kink (``xi = 0.7``, clustering
   while exploring) and Stage 2 (``xi = 0.9``, collapsed cluster). Each panel
   is a heatmap of *remaining* knowledge with agent positions overlaid; all
   panels share one colour scale and one fixed landscape/agent seed.

2. **2-panel temporal dynamics** (IE-0007 #8), ``t = 0..H``: mean
   ``percentage_knowledge_harvested`` and ``explored_percentage`` across
   ``xi in {0.1, 0.5, 0.7, 0.9}``, with 95% CI bands over 20 seeds.

The script is deliberately parameterised for the post-ADR-0003 re-run::

    MPLBACKEND=Agg python3 scripts/fig_prototypes/build_snapshots_and_temporal.py \
        --vanishing 0.933 --entry 0.0 --horizon 200 \
        --out output/report_notebook_v1/prototypes

Defaults use the current model default retention factor
(``CANONICAL_VANISHING_FACTOR``), a closed cohort (``entry = 0``), and the
standard ``H = 200`` horizon.  It writes parameter-tagged PNG + PDF + the
underlying tidy CSVs; it never touches ``drafts/``.

Model-internal, not calibrated.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO / "scripts/report_notebook"
for _p in (REPO, REPORT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sciencemodels import landscapes  # noqa: E402
from sciencemodels.model import (  # noqa: E402
    CANONICAL_VANISHING_FACTOR,
    MyModel,
)

import diagram_templates as T  # noqa: E402
import visual_qa  # noqa: E402

#: The canonical Noisy2 bundle (two peaks buried in uniform noise).
NOISY2_PARAMS = {"prop_random": 5, "std": 5, "std_curiosity": 0.1}

#: IE-0007 #7 stages: (visibility weight xi, short label).
SNAPSHOT_STAGES = (
    (0.1, "Stage 1 - scattered"),
    (0.7, "Kink - clustering while exploring"),
    (0.9, "Stage 2 - collapsed cluster"),
)

#: IE-0007 #8 temporal arms.
TEMPORAL_XI = (0.1, 0.5, 0.7, 0.9)

#: One colour per temporal arm (house palette; four distinct hues).
ARM_COLORS = (
    T.PALETTE["purple"],
    T.PALETTE["orange"],
    T.PALETTE["green"],
    T.PALETTE["pink"],
)

DEFAULT_OUT = "output/report_notebook_v1/prototypes"


def _fmt_tag(value: float) -> str:
    """A compact, filename-safe number label (0p933, 0, 0p25)."""
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    text = text.replace(".", "p")
    return text if text else "0"


def _resolve_out(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _make_model(
    *,
    xi: float,
    seed: int,
    horizon: int,
    vanishing: float,
    entry: float,
    exit_mode: str,
    tau_max: int,
) -> MyModel:
    """One seeded M0-style run on Noisy2 (no dyadic layer, visibility on)."""
    return MyModel(
        n_agents=20,
        initial_curiosity=np.float64(1.0 - xi),  # model takes c, not xi
        epsilon=np.float64(0.1),
        harvest=np.float64(0.02),
        grid_size=20,
        cell_init_function=landscapes.noisyGaussian,
        use_distance=True,
        generation_params=dict(NOISY2_PARAMS),
        agent_generation_time=np.float64(entry),
        new_questions=False,
        agent_seed=int(seed),
        step_limit=int(horizon),
        agent_generation_function=lambda curiosity, _rng, _params: curiosity,
        vanishing_factor_for_prestige=np.float64(vanishing),
        exit_mode=exit_mode,
        tau_max=int(tau_max),
        use_visibility_reward=True,
        knowledge_normalization="share",
        prestige_accrual="harvested",
    )


def _initial_explored_share(model: MyModel) -> float:
    """Explored share at t=0 (the model attribute is set at the first update)."""
    return float(
        np.sum(model.grid.properties["explored"].data) / model.size**2
    )


def collect_snapshots(args, run_tag: str) -> tuple[pd.DataFrame, list[dict], float]:
    """Run the three stage panels at ``args.snapshot_step``.

    Returns (long tile data, per-panel metadata, shared colour maximum).
    """
    rows: list[dict] = []
    panels: list[dict] = []
    initial_max = 0.0

    for xi, label in SNAPSHOT_STAGES:
        model = _make_model(
            xi=xi,
            seed=args.snapshot_seed,
            horizon=args.horizon,
            vanishing=args.vanishing,
            entry=args.entry,
            exit_mode=args.exit_mode,
            tau_max=args.tau_max,
        )
        initial_max = max(
            initial_max,
            float(np.max(model.grid.properties["initial_knowledge"].data)),
        )
        for _ in range(args.snapshot_step):
            model.step()

        knowledge = np.asarray(
            model.grid.properties["knowledge"].data, dtype=float
        ).copy()
        positions = [tuple(int(v) for v in a.pos) for a in model._scientists]
        counts = Counter(positions)
        n_agents = len(positions)
        harvest = float(model.percentage_knowledge_harvested)
        explored = float(model.explored_percentage)
        panels.append(
            {
                "xi": xi,
                "c": 1.0 - xi,
                "label": label,
                "knowledge": knowledge,
                "counts": counts,
                "n_agents": n_agents,
                "harvest": harvest,
                "explored": explored,
                "unique_tiles": len(counts),
                "max_coloc": max(counts.values()) if counts else 0,
            }
        )
        for x in range(model.size):
            for y in range(model.size):
                rows.append(
                    {
                        "run_tag": run_tag,
                        "xi": xi,
                        "c": 1.0 - xi,
                        "stage": label,
                        "t": int(model.steps),
                        "seed": int(args.snapshot_seed),
                        "x": x,
                        "y": y,
                        "remaining_knowledge": float(knowledge[x, y]),
                        "agent_count": int(counts.get((x, y), 0)),
                        "harvest_share": harvest,
                        "explored_share": explored,
                    }
                )
        print(
            f"[snapshot] xi={xi:.1f}  N={n_agents:2d}  "
            f"harvest={harvest:6.1%}  explored={explored:6.1%}  "
            f"unique_tiles={len(counts):2d}  max_coloc={max(counts.values()):2d}"
        )

    if args.entry == 0.0:
        for panel in panels:
            if panel["n_agents"] != 20:
                raise AssertionError(
                    "closed-cohort snapshot must keep N=20, got "
                    f"{panel['n_agents']}"
                )
    return pd.DataFrame(rows), panels, initial_max


def collect_temporal(args) -> tuple[pd.DataFrame, dict]:
    """Run the four-arm, 20-seed temporal sweep; return long mean/CI rows."""
    n_steps = int(args.horizon)
    n_seeds = int(args.n_seeds)
    harvest = np.full((len(TEMPORAL_XI), n_seeds, n_steps + 1), np.nan)
    explored = np.full_like(harvest, np.nan)

    for arm, xi in enumerate(TEMPORAL_XI):
        for seed_idx in range(n_seeds):
            seed = int(args.seed0 + seed_idx)
            model = _make_model(
                xi=xi,
                seed=seed,
                horizon=n_steps,
                vanishing=args.vanishing,
                entry=args.entry,
                exit_mode=args.exit_mode,
                tau_max=args.tau_max,
            )
            harvest[arm, seed_idx, 0] = 0.0
            explored[arm, seed_idx, 0] = _initial_explored_share(model)
            for _ in range(n_steps):
                model.step()
                t = int(model.steps)
                harvest[arm, seed_idx, t] = float(
                    model.percentage_knowledge_harvested
                )
                explored[arm, seed_idx, t] = float(model.explored_percentage)
        print(
            f"[temporal] xi={xi:.1f}  seeds={n_seeds}  "
            f"final harvest={np.nanmean(harvest[arm, :, -1]):.1%}  "
            f"final explored={np.nanmean(explored[arm, :, -1]):.1%}"
        )

    t = np.arange(n_steps + 1)
    rows: list[dict] = []
    for arm, xi in enumerate(TEMPORAL_XI):
        for metric, values in (
            ("percentage_knowledge_harvested", harvest[arm]),
            ("explored_percentage", explored[arm]),
        ):
            mean = np.nanmean(values, axis=0)
            sd = np.nanstd(values, axis=0, ddof=1)
            half = student_t.ppf(0.975, n_seeds - 1) * sd / np.sqrt(n_seeds)
            for step in t:
                rows.append(
                    {
                        "xi": xi,
                        "c": 1.0 - xi,
                        "t": int(step),
                        "metric": metric,
                        "mean": float(mean[step]),
                        "ci_lo": float(mean[step] - half[step]),
                        "ci_hi": float(mean[step] + half[step]),
                        "n_seeds": n_seeds,
                    }
                )
    return pd.DataFrame(rows), {
        "harvest": harvest,
        "explored": explored,
        "t": t,
    }


def _plot_snapshots(
    panels: list[dict],
    initial_max: float,
    args,
    out_base: Path,
) -> plt.Figure:
    fig = plt.figure(figsize=(15.4, 5.6))
    gs = GridSpec(
        1,
        3,
        figure=fig,
        left=0.045,
        right=0.985,
        top=0.82,
        bottom=0.27,
        wspace=0.13,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    image = None
    for idx, (ax, panel) in enumerate(zip(axes, panels)):
        image = ax.imshow(
            panel["knowledge"],
            origin="lower",
            cmap="viridis",
            norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=initial_max),
            interpolation="nearest",
            aspect="equal",
        )
        xs = [pos[0] for pos in panel["counts"]]
        ys = [pos[1] for pos in panel["counts"]]
        sizes = [26 + 8 * (n - 1) for n in panel["counts"].values()]
        ax.scatter(
            xs,
            ys,
            s=sizes,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
            clip_on=False,
        )
        ax.set_title(
            f"({chr(97 + idx)}) {panel['label']}\n"
            f"$\\xi={panel['xi']:.1f}$ ($c={panel['c']:.1f}$)\n"
            f"harvest {panel['harvest']:.1%} · explored "
            f"{panel['explored']:.1%} · max cluster {panel['max_coloc']}",
            fontsize=10.2,
        )
        ax.set_xticks([0, 5, 10, 15])
        ax.set_yticks([0, 5, 10, 15])
        ax.set_xlabel("tile x", fontsize=9)
        ax.set_ylabel("tile y", fontsize=9)
        ax.tick_params(labelsize=8)

    cax = fig.add_axes([0.28, 0.125, 0.44, 0.022])
    cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    cbar.set_label(
        "remaining knowledge (share of initial landscape stock)",
        fontsize=9,
    )
    cbar.ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    cbar.ax.tick_params(labelsize=8)

    n_agents = panels[0]["n_agents"]
    fig.suptitle(
        f"Spatial snapshots at $t={args.snapshot_step}$: "
        "curiosity stages on Noisy2",
        fontsize=13,
        y=0.965,
    )
    fig.text(
        0.045,
        0.018,
        f"Noisy2, N={n_agents}, H={args.horizon}, "
        f"seed={args.snapshot_seed}, $\\gamma={args.vanishing:.3f}$, "
        f"$g={args.entry:g}$, exit={args.exit_mode}; white circles = agents "
        "(area grows with co-location); remaining knowledge = initial minus "
        "cumulative harvest, colour on a square-root scale. "
        "Model-internal, not calibrated.",
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=T.PALETTE["gray"],
    )
    fig.canvas.draw()
    visual_qa.assert_figure_clean(
        fig,
        [{"ax": ax} for ax in axes],
        strict=True,
    )
    fig.savefig(out_base.with_suffix(".png"), dpi=T.HOUSE_STYLE["dpi"])
    fig.savefig(out_base.with_suffix(".pdf"))
    return fig


def _plot_temporal(
    data: pd.DataFrame,
    args,
    out_base: Path,
) -> plt.Figure:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 4.9),
        sharex=True,
        gridspec_kw={"left": 0.07, "right": 0.98, "top": 0.84,
                     "bottom": 0.20, "wspace": 0.16},
    )
    specs = (
        ("percentage_knowledge_harvested", "(a) Harvested knowledge",
         "harvested knowledge share"),
        ("explored_percentage", "(b) Explored grid",
         "explored grid share"),
    )
    for ax, (metric, title, ylabel) in zip(axes, specs):
        panel = data[data["metric"] == metric]
        top = 0.0
        for color, xi in zip(ARM_COLORS, TEMPORAL_XI):
            arm = panel[panel["xi"] == xi].sort_values("t")
            t = arm["t"].to_numpy()
            mean = arm["mean"].to_numpy()
            lo = arm["ci_lo"].to_numpy()
            hi = arm["ci_hi"].to_numpy()
            top = max(top, float(np.nanmax(hi)))
            ax.fill_between(
                t,
                lo,
                hi,
                color=color,
                alpha=0.16,
                linewidth=0.0,
                zorder=1,
            )
            ax.plot(
                t,
                mean,
                color=color,
                linewidth=1.9,
                label=f"$\\xi={xi:.1f}$ ($c={1 - xi:.1f}$)",
                zorder=2,
            )
        if metric == "percentage_knowledge_harvested":
            ax.axhline(
                0.10,
                color=T.PALETTE["gray"],
                linestyle=":",
                linewidth=1.0,
                label="10% milestone",
                zorder=0,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("step $t$", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlim(0, args.horizon)
        if args.horizon >= 100:
            xstep = 25
        elif args.horizon >= 40:
            xstep = 10
        else:
            xstep = 5
        xticks = list(np.arange(0, args.horizon + 1, xstep, dtype=int))
        if xticks[-1] != args.horizon and args.horizon - xticks[-1] > xstep / 2:
            xticks.append(args.horizon)
        ax.set_xticks(xticks)
        bottom = -0.02 if metric == "percentage_knowledge_harvested" else 0.0
        ax.set_ylim(bottom, max(0.12, top * 1.14))
        ax.grid(axis="y", alpha=0.3, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(
        loc="upper left",
        fontsize=7.6,
        frameon=False,
        ncol=1,
    )
    fig.suptitle(
        "Temporal dynamics across visibility weight (Noisy2, "
        f"H={args.horizon}, {args.n_seeds} seeds, 95% CI)",
        fontsize=13,
        y=0.965,
    )
    fig.text(
        0.07,
        0.025,
        f"{args.tag}  Model-internal, not calibrated. "
        "Bands: mean $\\pm$ 95% CI across seeds; harvest milestone at 10%.",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=T.PALETTE["gray"],
    )
    fig.canvas.draw()
    visual_qa.assert_figure_clean(
        fig,
        [
            {"ax": axes[0], "legend_over_data_ok": True},
            {"ax": axes[1]},
        ],
        strict=True,
    )
    fig.savefig(out_base.with_suffix(".png"), dpi=T.HOUSE_STYLE["dpi"])
    fig.savefig(out_base.with_suffix(".pdf"))
    return fig


def _display_path(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def verify_outputs(
    snapshot_data: pd.DataFrame,
    temporal_data: pd.DataFrame,
    panels: list[dict],
    raw: dict,
    png_paths: list[Path],
    pdf_paths: list[Path],
    horizon: int,
) -> None:
    """Cheap, explicit checks so a pilot cannot pass with silent gaps."""
    for path in png_paths + pdf_paths:
        if not path.exists() or path.stat().st_size < 10_000:
            raise AssertionError(f"figure missing or too small: {path}")

    if len(snapshot_data) != len(SNAPSHOT_STAGES) * 20 * 20:
        raise AssertionError(
            f"snapshot data rows: {len(snapshot_data)}, expected 1200"
        )
    expected_agents = {panel["xi"]: panel["n_agents"] for panel in panels}
    got_agents = snapshot_data.groupby("xi")["agent_count"].sum()
    for xi, expected in expected_agents.items():
        if int(got_agents.loc[xi]) != expected:
            raise AssertionError(
                f"snapshot agent count at xi={xi}: "
                f"{int(got_agents.loc[xi])} != {expected}"
            )
    if not np.isfinite(snapshot_data["remaining_knowledge"]).all():
        raise AssertionError("non-finite remaining knowledge")
    if snapshot_data["remaining_knowledge"].min() < -1e-12:
        raise AssertionError("negative remaining knowledge")
    for panel in panels:
        if not (0.0 <= panel["harvest"] <= 1.0):
            raise AssertionError(f"harvest out of range: {panel['harvest']}")
        if not (0.0 <= panel["explored"] <= 1.0):
            raise AssertionError(f"explored out of range: {panel['explored']}")

    n_steps = int(temporal_data["t"].max())
    if n_steps != int(horizon):
        raise AssertionError(
            f"temporal horizon {n_steps} != requested {horizon}"
        )
    if len(temporal_data) != len(TEMPORAL_XI) * 2 * (n_steps + 1):
        raise AssertionError(
            f"temporal rows: {len(temporal_data)}, expected "
            f"{len(TEMPORAL_XI) * 2 * (n_steps + 1)}"
        )
    if temporal_data["n_seeds"].nunique() != 1:
        raise AssertionError("temporal seeds are not constant")
    if not np.isfinite(temporal_data[["mean", "ci_lo", "ci_hi"]].to_numpy()).all():
        raise AssertionError("non-finite temporal statistics")
    if (temporal_data["ci_lo"] > temporal_data["mean"] + 1e-12).any():
        raise AssertionError("CI lower bound exceeds mean")
    if (temporal_data["ci_hi"] < temporal_data["mean"] - 1e-12).any():
        raise AssertionError("CI upper bound below mean")
    for arm in range(len(TEMPORAL_XI)):
        for metric in ("percentage_knowledge_harvested", "explored_percentage"):
            values = raw["harvest" if metric.startswith("percentage") else "explored"][arm]
            if not np.isfinite(values).all():
                raise AssertionError(f"non-finite raw series: {metric}, arm {arm}")
    print("[verify] snapshot + temporal data and figure files OK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build IE-0007 #7 spatial snapshots and #8 temporal dynamics "
            "for Noisy2 (D5 prototypes)."
        )
    )
    parser.add_argument(
        "--vanishing",
        type=float,
        default=float(CANONICAL_VANISHING_FACTOR),
        help="prestige retention factor gamma (default: model canonical value)",
    )
    parser.add_argument(
        "--entry",
        type=float,
        default=0.0,
        help="agent_generation_time g (default 0, closed cohort M0)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=200,
        help="temporal horizon H in steps (default 200)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUT,
        help=f"output directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--exit-mode",
        type=str,
        default="none",
        choices=("none", "fixed_span", "age_hazard_sync",
                 "age_hazard_staggered"),
        help="ADR-0003 retirement mode (default none; matters only if entry>0)",
    )
    parser.add_argument(
        "--tau-max",
        type=int,
        default=80,
        help="ADR-0003 maximum career length in steps (default 80)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=20,
        help="temporal seeds (default 20)",
    )
    parser.add_argument(
        "--seed0",
        type=int,
        default=20261126,
        help="first temporal seed; seed i uses seed0+i (default 20261126)",
    )
    parser.add_argument(
        "--snapshot-seed",
        type=int,
        default=None,
        help="fixed landscape/agent seed for snapshots (default: seed0)",
    )
    parser.add_argument(
        "--snapshot-step",
        type=int,
        default=80,
        help="snapshot step t (default 80)",
    )
    args = parser.parse_args(argv)

    if not (0.0 < args.vanishing <= 1.0):
        parser.error("--vanishing must lie in (0, 1]")
    if args.entry < 0.0:
        parser.error("--entry must be non-negative")
    if args.horizon < 1:
        parser.error("--horizon must be at least 1")
    if args.n_seeds < 2:
        parser.error("--n-seeds must be at least 2 for a 95% CI")
    if args.snapshot_step < 1:
        parser.error("--snapshot-step must be at least 1")
    if args.snapshot_step > args.horizon:
        parser.error("--snapshot-step must not exceed --horizon")
    if args.snapshot_seed is None:
        args.snapshot_seed = args.seed0
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = _resolve_out(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = (
        f"Noisy2_H{args.horizon}_v{_fmt_tag(args.vanishing)}"
        f"_g{_fmt_tag(args.entry)}"
    )
    args.tag = (
        f"Noisy2, H={args.horizon}, $\\gamma={args.vanishing:.3f}$, "
        f"$g={args.entry:g}$, exit={args.exit_mode}"
    )
    snap_base = out_dir / (
        f"D5_spatial_snapshots_{tag}_seed{args.snapshot_seed}"
    )
    temp_base = out_dir / (
        f"D5_temporal_{tag}_n{args.n_seeds}"
    )

    print(
        f"[setup] Noisy2  H={args.horizon}  gamma={args.vanishing:.6g}  "
        f"g={args.entry:g}  exit={args.exit_mode}  out={out_dir}"
    )

    snapshot_data, panels, initial_max = collect_snapshots(args, tag)
    temporal_data, raw = collect_temporal(args)

    T.write_figure_data(snapshot_data, snap_base.with_suffix(".png"))
    T.write_figure_data(temporal_data, temp_base.with_suffix(".png"))

    fig_snap = _plot_snapshots(panels, initial_max, args, snap_base)
    fig_temp = _plot_temporal(temporal_data, args, temp_base)
    plt.close(fig_snap)
    plt.close(fig_temp)

    png_paths = [snap_base.with_suffix(".png"), temp_base.with_suffix(".png")]
    pdf_paths = [snap_base.with_suffix(".pdf"), temp_base.with_suffix(".pdf")]
    verify_outputs(
        snapshot_data,
        temporal_data,
        panels,
        raw,
        png_paths,
        pdf_paths,
        args.horizon,
    )
    for path in png_paths + pdf_paths:
        print(f"[output] {_display_path(path)}")
    for path in (snap_base, temp_base):
        print(f"[data]   {_display_path(path.with_suffix(''))}_data.csv")
    print("[done] D5 prototypes built and verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
