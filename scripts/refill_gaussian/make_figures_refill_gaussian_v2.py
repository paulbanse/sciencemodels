"""In-round diagnostic figures for refill-gaussian v2.

Reads only ``data/refill_gaussian_v2/`` and writes PNGs to
``data/refill_gaussian_v2/figures/``:

- ``tracking_dynamics.png``: per-mode mean trajectories (net centroid
  displacement from each seed's own step-0 centroid, top-5% dispersion,
  agent-to-peak distance) -- the traveling-vs-diffusion evidence.
- ``sabbatical_pairing.png``: p_reloc 0 vs 0.02 paired bars per mode x c on
  the three headline outcomes (H=800).

Plain ASCII figure text (matplotlib mathtext is fragile).  Every plotted
value comes from the run archive; nothing is hand-typed.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "refill_gaussian_v2"
FIGDIR = OUT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

MODES = ("gaussian_s3", "gaussian_s6", "frontier")
RELOCS = (0.0, 0.02)
MODE_COLOR = {"gaussian_s3": "#7b3294", "gaussian_s6": "#008837",
              "frontier": "#2166ac"}


def load_traj() -> pd.DataFrame:
    recs = []
    for p in sorted(glob.glob(str(OUT / "traj" / "cell_*.csv"))):
        name = Path(p).stem
        m = re.match(r"cell_refill-(gaussian_s\d+|frontier)_c(\d+)_gen(\d+)"
                     r"_reloc(\d+)_mgstd5", name)
        df = pd.read_csv(p)
        df["refill_mode"] = m.group(1)
        df["p_reloc"] = int(m.group(4)) / 100.0
        df["curiosity"] = int(m.group(2)) / 10.0
        recs.append(df)
    return pd.concat(recs, ignore_index=True)


def net_disp_from_start(df: pd.DataFrame) -> pd.DataFrame:
    """Per-seed minimal-image displacement from that seed's step-0 centroid."""
    n = 20
    out = []
    for (mode, reloc, c, seed), g in df.groupby(
            ["refill_mode", "p_reloc", "curiosity", "agent_seed"]):
        g = g.sort_values("step")
        x0, y0 = float(g["cx"].iloc[0]), float(g["cy"].iloc[0])
        dx = np.abs(g["cx"].to_numpy() - x0) % n
        dy = np.abs(g["cy"].to_numpy() - y0) % n
        dx = np.minimum(dx, n - dx)
        dy = np.minimum(dy, n - dy)
        out.append(pd.DataFrame(dict(
            refill_mode=mode, p_reloc=reloc, curiosity=c, agent_seed=seed,
            step=g["step"].to_numpy(), net=np.sqrt(dx * dx + dy * dy))))
    return pd.concat(out, ignore_index=True)


def fig_tracking(traj: pd.DataFrame) -> Path:
    net = net_disp_from_start(traj)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
    panels = [
        (net, "net", "net centroid displacement (tiles)"),
        (traj, "dispersion", "top-5% dispersion (tiles)"),
        (traj, "agent_peak_dist", "agent-to-peak distance (tiles)"),
    ]
    for ax, (frame, col, ylabel) in zip(axes, panels):
        for mode in MODES:
            for reloc, ls in ((0.0, "-"), (0.02, "--")):
                sub = (frame[(frame.refill_mode == mode)
                             & (frame.p_reloc == reloc)]
                       .groupby("step")[col].mean())
                ax.plot(sub.index, sub.to_numpy(), ls,
                        color=MODE_COLOR[mode], lw=1.4,
                        label=f"{mode} p={reloc:.2f}")
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 800)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("refill_gaussian_v2: peak-centroid tracking (mgstd5, "
                 "c pooled, g=0.1, 20 seeds per cell)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = FIGDIR / "tracking_dynamics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_pairing(panel: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
    metrics = [("explored_percentage_final", "explored share (final)"),
               ("late_window_flow_rate", "late-window flow rate"),
               ("corr_prestige_ideal_merit_final",
                "Corr(prestige, ideal merit) (final)")]
    cs = (0.3, 0.5, 0.7)
    width = 0.35
    x = np.arange(len(MODES))
    for ax, (metric, ylabel) in zip(axes, metrics):
        for i, c in enumerate(cs):
            offs = (i - 1) * width / 1.5
            for reloc, hatch in ((0.0, ""), (0.02, "///")):
                means = []
                for mode in MODES:
                    sub = panel[(panel.refill_mode == mode)
                                & (panel.initial_curiosity == c)
                                & (panel.p_reloc == reloc)][metric]
                    means.append(sub.mean())
                ax.bar(x + offs, means, width / 1.6,
                       color=[MODE_COLOR[m] if reloc > 0 else "#d9d9d9"
                              for m in MODES],
                       edgecolor=[MODE_COLOR[m] for m in MODES],
                       hatch=hatch, label=f"c={c:.1f} p={reloc:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in MODES], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.axhline(0.0, color="gray", lw=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, fontsize=6, ncol=3)
    fig.suptitle("refill_gaussian_v2: sabbatical pairing at H=800 "
                 "(open bars p_reloc=0, hatched p_reloc=0.02)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = FIGDIR / "sabbatical_pairing.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    traj = load_traj()
    panel = pd.concat([pd.read_csv(p) for p in
                       sorted(glob.glob(str(OUT / "cells" / "cell_*.csv")))],
                      ignore_index=True)
    p1 = fig_tracking(traj)
    p2 = fig_pairing(panel)
    print("wrote", p1)
    print("wrote", p2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
