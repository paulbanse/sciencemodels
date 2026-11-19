"""Independent verifier for Figure C (F9_v2_typology_annealing).

``check()`` recomputes every plotted quantity and annotation in
``F9_v2_typology_annealing_data.csv`` from the RAW archives
(``data/typology_v1/runs.csv`` and ``contrast_table.csv``), re-implementing
the 20,000-resample percentile bootstrap (seed 20261115) independently,
asserts the expected counts (480 runs; 24 conditions x 20 seeds) and the
``wedge_fin = c_ideal_fin - c_local_fin`` identity, and compares at ``tol``.

It then re-runs the rendered-canvas contract: the figure is rebuilt in memory
via the builder's ``_figure`` helper and put through
``visual_qa.assert_figure_clean`` (artist counts, plotted-vs-sidecar min/max,
ylim headroom, text/text-title/text-data collisions, legend duplicates) and
``assert_text_fits``.  A sidecar that survives the numeric check but renders a
collision (the v2 failure mode) therefore fails here.

``data_root`` accepts the archive directory (default ``data/typology_v1``) or
its ``cells`` subdirectory; both the recomputation and the rebuild use it.

Empty return list = all checks pass.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FNAME = "F9_v2_typology_annealing"
BOOT_SEED = 20261115
BOOT_N = 20000


def _boot_ci(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(x), size=(BOOT_N, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _paired(t: pd.DataFrame, col: str, a: tuple, z: tuple,
            match: list[str]) -> pd.Series:
    ga, gz = "gamma", "kappa"

    def _sel(arm):
        s = np.ones(len(t), dtype=bool)
        if arm[0] is not None:
            s &= (t[ga] == arm[0]).to_numpy()
        if arm[1] is not None:
            s &= (t[gz] == arm[1]).to_numpy()
        return s

    pa = t.loc[_sel(a)].set_index(match)[col]
    pz = t.loc[_sel(z)].set_index(match)[col]
    return pa - pz


def _rerun_visual_contract(data_root, fails: list[str]) -> None:
    """Rebuild the figure in memory and assert the rendered canvas is clean."""
    import matplotlib.pyplot as plt

    import figC_typology_annealing as builder
    import visual_qa
    from turnover_lifecycle_figure import assert_text_fits

    fig = None
    try:
        side = builder.series_rows(builder.load_runs(data_root), data_root)
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
    import figC_typology_annealing as builder

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
    t = pd.read_csv(root / "runs.csv")

    # ---- input counts and sign identity -------------------------------
    if len(t) != 480:
        fails.append(f"runs: expected 480, got {len(t)}")
    counts = t.groupby(["gamma", "kappa", "curiosity", "g_entry"]).size()
    if len(counts) != 24 or not (counts == 20).all():
        fails.append(f"expected 24 conditions x 20 seeds, got {counts.to_dict()}")
    ident = float((t["wedge_fin"]
                   - (t["c_ideal_fin"] - t["c_local_fin"])).abs().max())
    if ident >= 1e-12:
        fails.append(f"wedge_fin = c_ideal_fin - c_local_fin identity broken: "
                     f"max|err| = {ident}")
    t = t.assign(dr=-t["wedge_fin"])
    ilc = "ideal_loss_cum"
    match = ["curiosity", "g_entry", "seed"]
    match_k = ["kappa", "curiosity", "g_entry", "seed"]
    match_g = ["gamma", "curiosity", "g_entry", "seed"]

    pure_mean = float(t[(t["gamma"] == 0) & (t["kappa"] == 0)][ilc].mean())
    d_dest = _paired(t, ilc, (1.0, None), (0.0, None), match_k)
    d_cf = _paired(t, ilc, (None, 1.0), (None, 0.0), match_g)
    d_dr = _paired(t, "dr", (1.0, None), (0.0, None), match_k)
    d_ilc = _paired(t, ilc, (1.0, None), (0.0, None), match_k)
    arms = {"+Destruction (g=1)": (1.0, 0.0),
            "+Counterforce (k=1)": (0.0, 1.0),
            "Full (g=1,k=1)": (1.0, 1.0)}

    def want(row) -> tuple[float, float, float, int] | None:
        """(value, ci_lo, ci_hi, n) recomputed for one sidecar row."""
        m, st = row["metric"], row["statistic"]
        if st == "arm_delta":
            if row["arm"] == "Pure (0,0)":
                return 0.0, np.nan, np.nan, 120
            d = _paired(t, ilc, arms[row["arm"]], (0.0, 0.0), match)
            lo, hi = _boot_ci(d.to_numpy())
            return float(d.mean()), lo, hi, len(d)
        if st == "main_effect" and m == "ideal_loss_cum destruction_main":
            lo, hi = _boot_ci(d_dest.to_numpy())
            return float(d_dest.mean()), lo, hi, len(d_dest)
        if st == "main_effect" and m == "ideal_loss_cum counterforce_main":
            lo, hi = _boot_ci(d_cf.to_numpy())
            return float(d_cf.mean()), lo, hi, len(d_cf)
        if st == "level" and m == "pure_ideal_loss_cum_mean":
            return pure_mean, np.nan, np.nan, 120
        if st == "level" and m == "mean_prestige_all_runs":
            return float(t["mean_prestige"].mean()), np.nan, np.nan, 480
        if st == "annotation" and m == "destruction_pct_of_pure":
            return 100.0 * float(d_dest.mean()) / pure_mean, np.nan, np.nan, \
                len(d_dest)
        if st == "mean" and m.startswith("relational wedge"):
            sub = t[(t["g_entry"] == row["g_entry"])
                    & (t["curiosity"] == row["curiosity"])
                    & (t["gamma"] == row["gamma"])]["dr"]
            lo, hi = _boot_ci(sub.to_numpy())
            return float(sub.mean()), lo, hi, len(sub)
        if st == "paired_delta" and m == "paired wedge narrowing (gamma 1 - 0)":
            lo, hi = _boot_ci(d_dr.to_numpy())
            return float(d_dr.mean()), lo, hi, len(d_dr)
        if st == "paired_delta" and m.startswith("paired delta ideal_loss_cum"):
            if pd.isna(row["curiosity"]):
                lo, hi = _boot_ci(d_ilc.to_numpy())
                return float(d_ilc.mean()), lo, hi, len(d_ilc)
            sub = d_ilc[(d_ilc.index.get_level_values("curiosity")
                         == row["curiosity"])
                        & (d_ilc.index.get_level_values("g_entry")
                           == row["g_entry"])]
            lo, hi = _boot_ci(sub.to_numpy())
            return float(sub.mean()), lo, hi, len(sub)
        if st == "cross_check":
            ct = pd.read_csv(root / "contrast_table.csv")
            spec = {
                "ideal_loss_cum destruction_main (contrast_table cross-check)":
                    ("ideal_loss_cum",
                     "destruction_main (gamma 0->1, kappa matched)", 1.0),
                "ideal_loss_cum counterforce_main (contrast_table cross-check)":
                    ("ideal_loss_cum",
                     "counterforce_main (kappa 0->1, gamma matched)", 1.0),
                "paired wedge narrowing (gamma 1 - 0) (contrast_table "
                "cross-check)":
                    ("wedge_fin", "destruction_main (gamma 0->1, kappa matched)",
                     -1.0),
            }
            key = m if m in spec else None
            if key is None:
                for k in spec:
                    if k.startswith(m.split(" (contrast")[0]):
                        key = k
            if key is None:
                fails.append(f"cross-check row not understood: {m}")
                return None
            measure, contrast, sign = spec[key]
            r = ct[(ct["measure"] == measure) & (ct["contrast"] == contrast)]
            if len(r) != 1:
                fails.append(f"contrast_table row missing: {measure} {contrast}")
                return None
            r = r.iloc[0]
            return (sign * float(r["mean_delta"]),
                    np.nan, np.nan, int(r["n_pairs"]))
        fails.append(f"unhandled sidecar row: {m} / {st}")
        return None

    for _, row in side.iterrows():
        exp = want(row)
        if exp is None:
            continue
        value, lo, hi, n = exp
        checks = [("value", row["value"], value), ("n", row["n"], n)]
        if not (isinstance(lo, float) and np.isnan(lo)):
            checks += [("ci_lo", row["ci_lo"], lo), ("ci_hi", row["ci_hi"], hi)]
        for label, got, w in checks:
            if label == "n":
                if int(got) != int(w):
                    fails.append(f"{row['metric']} ({row['statistic']}): n "
                                 f"{got} != {w}")
            elif abs(float(got) - float(w)) > tol:
                fails.append(f"{row['metric']} ({row['statistic']}) {label}: "
                             f"sidecar {got} vs recomputed {w}")

    # ---- rendered-canvas contract (visual_qa, re-run independently) -----
    _rerun_visual_contract(data_root, fails)
    return fails


if __name__ == "__main__":
    res = check()
    for f in res:
        print("FAIL:", f)
    print("all checks pass" if not res else "FAILURES PRESENT")
