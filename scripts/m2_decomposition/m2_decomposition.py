"""M2 counterfactual ablation-share decomposition (workplan section 2.4,
item 14, former H4; IE-0007 #6).

Figure 8's method (scripts/report_notebook/fig_FA2_v2_arm_a_decomposition.py,
FA2_v2_drop_decomposition) transplanted to Tier M2: instead of decomposing
the entry-induced M0 -> M1 alignment drop into starvation vs. herding
shares, we decompose the dyadic-layer-induced M1 -> M2 alignment CHANGE
into three counterfactual ablation shares along a fixed chain:

    L0 = M1 baseline            (dyadic off, data/path_b_postbug/baseline_off.csv)
    L1 = link formation only    (dyadic on, dyadic_move_pull = 0, structural credit)
    L2 = + spatial co-movement  (dyadic_move_pull = 1.0, structural credit)
    L3 = + equal-split credit   (dyadic_move_pull = 1.0, equal_split)  = full M2

    gap   = L0 - L3                          (signed; negative = M2 lifts alignment)
    share_link   = (L0 - L1) / gap
    share_move   = (L1 - L2) / gap
    share_credit = (L2 - L3) / gap
    identity: share_link + share_move + share_credit == 100%  (asserted)

Metric: c_ideal_tm, the time-mean Corr(prestige, ideal_merit) reporter
(same quantity as Figure 15c of the paper sketch). The M2 archive carries
no terminal alignment column, so the time-mean is the archive-native
quantity; this differs from Figure 8's terminal metric and is disclosed in
the report.

Sources (read-only): data/path_b_postbug/ (manifest.json
gross_net-checked): baseline_off.csv (40 runs), cells_v1/ (800 runs,
dyadic_move_pull = 0), cells_v2/ (960 runs, pull 0.5/1.0),
m0_m2_probe/ (80 runs, g = 0 canary). All arms: c = 0.5, g = 0.1,
H = 200, 20 seeds (agent_seed drives agent and landscape RNG jointly).
Primary rows pool over the two formulations (logit, threshold), 2 cells
x 20 seeds = 40 runs per chain node; per-formulation rows are the
robustness twin. Shares carry seed-paired bootstrap 95% CIs (B = 10000,
rng seed 20261126); the paired design matches the archive's common
agent_seed grid.

Chain order matters (sequential decompositions are order-dependent), so
an alternative order (credit leg moved to pull = 0) is computed alongside.
The first leg is order-invariant by construction.

The report is GENERATED from scripts/m2_decomposition/report_template.md
by token replacement (@@TOKEN@@, one occurrence each); no computed number
is hand-typed into the report. Table fragments are written beside the
canonical CSVs as provenance.

NOT CALIBRATED: model-internal only (owner ruling 2026-11-13).

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
        scripts/m2_decomposition/m2_decomposition.py
"""
from __future__ import annotations

import json
import re
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
# house figure contract lives in scripts/report_notebook/ (one home per
# style module, per the figE/figF builders' docstring)
sys.path.insert(0, str(REPO / "scripts" / "report_notebook"))
import diagram_templates as T
import visual_qa

DATA = REPO / "data" / "path_b_postbug"
OUT_DATA = REPO / "data" / "m2_decomposition"
FIG_DIR = OUT_DATA / "figures"
TAB_DIR = OUT_DATA / "tables"
FIG_STEM = "m2_decomposition_v1"
FIG_PNG = FIG_DIR / f"{FIG_STEM}.png"
TEMPLATE = HERE / "report_template.md"
REPORT = REPO / "output" / "reports" / "m2_decomposition_v1.md"

METRIC = "c_ideal_tm"
BOOT_N = 10_000
BOOT_SEED = 20261126
LAMBDAS = (0.0, 1.0)
LANDS = ("mgstd5", "noisypr5")
FORMS = ("logit", "threshold")
SUM_TOL = 1e-9

_RE_V1 = re.compile(r"cell_lam(\d{3})_(logit|threshold)_(equal|struct)_(mgstd5|noisypr5)\.csv$")
_RE_V2 = re.compile(r"cell_lam(\d{3})_pull(\d{3})_(logit|threshold)_(equal|struct)_(mgstd5|noisypr5)\.csv$")


def _credit(tag: str) -> str:
    return "structural" if tag == "struct" else "equal_split"


def load_runs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run-level frame for every chain node plus the g=0 probe; manifest facts."""
    manifest = json.loads((DATA / "manifest.json").read_text())
    gross = manifest["gross_net"]

    base = pd.read_csv(DATA / "baseline_off.csv")
    assert set(base["dyadic_mode"]) == {"off"}, "baseline_off must be dyadic off"
    assert sorted(base["agent_seed"].unique()) == list(range(20)), "baseline seeds 0..19"
    assert len(base) == gross["baseline_off"]["rows_net"] == 40, "baseline row count"

    rows = []

    def add_block(files, regex, pull_from_name: bool):
        for f in sorted(files):
            m = regex.search(f.name)
            assert m, f"unparseable cell name {f.name}"
            if pull_from_name:
                lam = int(m.group(1)) / 100
                pull = int(m.group(2)) / 100
                form, credit, land = m.group(3), _credit(m.group(4)), m.group(5)
            else:
                lam = int(m.group(1)) / 100
                pull = 0.0
                form, credit, land = m.group(2), _credit(m.group(3)), m.group(4)
            d = pd.read_csv(f)
            assert len(d) == 20, f"{f.name}: {len(d)} rows"
            assert sorted(d["agent_seed"].unique()) == list(range(20)), f"{f.name}: seeds"
            for _, r in d.iterrows():
                rows.append(dict(lam=lam, pull=pull, form=form, credit=credit,
                                 land=land, seed=int(r["agent_seed"]),
                                 c=float(r[METRIC]), links=float(r["links_tm"])))

    v1 = list((DATA / "cells_v1").glob("*.csv"))
    v2 = list((DATA / "cells_v2").glob("*.csv"))
    assert len(v1) == gross["v1_cells"]["cells_gross"] == 40, "v1 cell count"
    assert len(v2) == gross["v2_cells"]["cells_gross"] == 48, "v2 cell count"
    add_block(v1, _RE_V1, pull_from_name=False)
    add_block(v2, _RE_V2, pull_from_name=True)

    runs = pd.DataFrame(rows).rename(columns={"c": METRIC})
    assert len(runs) == gross["v1_cells"]["rows_net"] + gross["v2_cells"]["rows_net"] == 1760
    assert set(runs.loc[runs.pull == 0.0, "lam"]) == {0.0, 0.25, 0.5, 0.75, 1.0}
    assert set(runs.loc[runs.pull > 0.0, "pull"]) == {0.5, 1.0}

    probe_rows = []
    for f in sorted((DATA / "m0_m2_probe").glob("cell_*.csv")):
        m = re.search(r"cell_lam(\d{3})_pull(\d{3})_(logit|threshold)_(equal|struct)_(mgstd5|noisypr5)", f.name)
        d = pd.read_csv(f)
        assert len(d) == 20
        for _, r in d.iterrows():
            probe_rows.append(dict(lam=int(m.group(1)) / 100, pull=int(m.group(2)) / 100,
                                   land=m.group(5), seed=int(r["agent_seed"]),
                                   c=float(r[METRIC]), links=float(r["links_tm"])))
    probe = pd.DataFrame(probe_rows).rename(columns={"c": METRIC})
    assert len(probe) == gross["m0_m2_probe"]["rows_net"] == 80
    return runs, probe, manifest


def node_levels(runs: pd.DataFrame, base: pd.DataFrame, land: str, lam: float,
                form: str | None) -> dict[str, np.ndarray]:
    """Per-seed chain-node levels (20 values each). form=None pools formulations."""
    b = base[base.land == land].sort_values("agent_seed")[METRIC if METRIC in base else "c_ideal_tm"].values
    assert len(b) == 20

    def sel(pull: float, credit: str) -> np.ndarray:
        s = runs[(runs.lam == lam) & (runs.pull == pull) & (runs.credit == credit)
                 & (runs.land == land)]
        if form is not None:
            s = s[s.form == form]
        # pooled over formulation: per-seed mean across the form cells
        return s.groupby("seed")[METRIC].mean().sort_index().values

    out = {"L0": b, "L1": sel(0.0, "structural"), "L2": sel(1.0, "structural"),
           "L3": sel(1.0, "equal_split"), "L1e": sel(0.0, "equal_split")}
    for k, v in out.items():
        assert len(v) == 20, f"{land} lam={lam} form={form}: node {k} n={len(v)}"
    return out


def shares_from(a0, a1, a2, a3, a1e=None) -> dict:
    """Signed ablation shares (%) for one chain; identity asserted when scalar."""
    gap = a0 - a3
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            "gap": gap,
            "link": 100.0 * (a0 - a1) / gap,
            "move": 100.0 * (a1 - a2) / gap,
            "credit": 100.0 * (a2 - a3) / gap,
        }
        if a1e is not None:
            out["credit_B"] = 100.0 * (a1 - a1e) / gap
            out["move_B"] = 100.0 * (a1e - a3) / gap
    return out


def boot_ci(nodes: dict[str, np.ndarray], rng: np.random.Generator):
    """Seed-paired bootstrap for the gap, the three leg level-diffs and shares."""
    idx = rng.integers(0, 20, size=(BOOT_N, 20))
    b = {k: v[idx].mean(axis=1) for k, v in nodes.items()}
    sh = shares_from(b["L0"], b["L1"], b["L2"], b["L3"], b.get("L1e"))
    legs = {"leg_link": b["L0"] - b["L1"], "leg_move": b["L1"] - b["L2"],
            "leg_credit": b["L2"] - b["L3"]}
    ci = {}
    for name, arr in {**{k: sh[k] for k in ("gap", "link", "move", "credit",
                                            "credit_B", "move_B") if k in sh},
                      **legs}.items():
        lo, hi = np.percentile(arr, [2.5, 97.5])
        ci[name] = (float(lo), float(hi))
    return ci


def fmt(x: float, digits: int = 1, signed: bool = True) -> str:
    s = f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"
    return s.replace("-", "-")  # ASCII minus already


def fmt_ci(t: tuple[float, float], digits: int = 1, signed: bool = True) -> str:
    return f"[{fmt(t[0], digits, signed)}, {fmt(t[1], digits, signed)}]"


def main() -> None:
    runs, probe, manifest = load_runs()
    base = pd.read_csv(DATA / "baseline_off.csv").rename(columns={"land": "land"})
    base["c_ideal_tm"] = base["c_ideal_tm"].astype(float)

    rng = np.random.default_rng(BOOT_SEED)

    # ---- per-formulation chains -------------------------------------
    rows = []
    for land in LANDS:
        for lam in LAMBDAS:
            for form in FORMS:
                nd = node_levels(runs, base, land, lam, form)
                lv = {k: float(v.mean()) for k, v in nd.items()}
                sh = shares_from(lv["L0"], lv["L1"], lv["L2"], lv["L3"], lv["L1e"])
                assert abs(sh["link"] + sh["move"] + sh["credit"] - 100.0) < SUM_TOL
                assert abs(sh["link"] + sh["credit_B"] + sh["move_B"] - 100.0) < SUM_TOL
                ci = boot_ci(nd, rng)
                rows.append(dict(scope="form", land=land, lam=lam, form=form,
                                 n_per_node=20,
                                 **{f"L{k[1:]}": lv[k] for k in ("L0", "L1", "L2", "L3", "L1e")},
                                 gap=sh["gap"], link=sh["link"], move=sh["move"],
                                 credit=sh["credit"], credit_B=sh["credit_B"],
                                 move_B=sh["move_B"],
                                 **{f"{k}_lo": v[0] for k, v in ci.items()},
                                 **{f"{k}_hi": v[1] for k, v in ci.items()}))

    # ---- pooled-over-formulation chains (primary) --------------------
    pooled = []
    for land in LANDS:
        for lam in LAMBDAS:
            nd = node_levels(runs, base, land, lam, None)
            lv = {k: float(v.mean()) for k, v in nd.items()}
            sh = shares_from(lv["L0"], lv["L1"], lv["L2"], lv["L3"], lv["L1e"])
            assert abs(sh["link"] + sh["move"] + sh["credit"] - 100.0) < SUM_TOL
            assert abs(sh["link"] + sh["credit_B"] + sh["move_B"] - 100.0) < SUM_TOL
            ci = boot_ci(nd, rng)
            pooled.append(dict(scope="pooled", land=land, lam=lam, form="pooled",
                               n_per_node=20,
                               **{f"L{k[1:]}": lv[k] for k in ("L0", "L1", "L2", "L3", "L1e")},
                               gap=sh["gap"], link=sh["link"], move=sh["move"],
                               credit=sh["credit"], credit_B=sh["credit_B"],
                               move_B=sh["move_B"],
                               **{f"{k}_lo": v[0] for k, v in ci.items()},
                               **{f"{k}_hi": v[1] for k, v in ci.items()}))

    vtab = pd.DataFrame(pooled + rows)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    vtab.to_csv(OUT_DATA / "verification_table.csv", index=False)

    # ---- structural checks ------------------------------------------
    checks = []

    def check(name: str, expected: str, found: str, ok: bool, *, assert_ok: bool = True):
        status = ("PASS" if ok else "FAIL") if assert_ok else "OBSERVED"
        checks.append(dict(check=name, expected=expected, found=found, status=status))
        if assert_ok:
            assert ok, f"check failed: {name}: {found} != {expected}"

    check("archive rows: baseline_off / cells_v1 / cells_v2 / m0_m2_probe",
          "40 / 800 / 960 / 80",
          f"{len(base)} / {int((runs.pull == 0).sum())} / {int((runs.pull > 0).sum())} / {len(probe)}",
          len(base) == 40 and int((runs.pull == 0).sum()) == 800
          and int((runs.pull > 0).sum()) == 960 and len(probe) == 80)

    max_sum_dev = float(np.abs(vtab.link + vtab.move + vtab.credit - 100.0).max())
    check("ablation identity: link + move + credit == 100%",
          f"max abs dev < {SUM_TOL:g}", f"max abs dev = {max_sum_dev:g}",
          max_sum_dev < SUM_TOL)

    max_sum_dev_b = float(np.abs(vtab.link + vtab.credit_B + vtab.move_B - 100.0).max())
    check("order-B identity: link + credit_B + move_B == 100%",
          f"max abs dev < {SUM_TOL:g}", f"max abs dev = {max_sum_dev_b:g}",
          max_sum_dev_b < SUM_TOL)

    # lambda-blindness of the link-only arm under structural credit (bit-level)
    p0 = runs[(runs.pull == 0.0) & (runs.credit == "structural")]
    spread = p0.groupby(["form", "land", "seed"])[METRIC].agg(lambda s: s.max() - s.min())
    check("pull=0 structural: c_ideal_tm exactly lambda-blind (5 lambda levels)",
          "max per-seed spread = 0.0", f"max per-seed spread = {float(spread.max()):.1f}",
          float(spread.max()) == 0.0)

    # links nevertheless form and vary strongly with lambda
    lk = p0[(p0.form == "logit") & (p0.land == "mgstd5")].groupby("lam")["links"].mean()
    links_lo, links_hi = float(lk.min()), float(lk.max())
    check("pull=0 structural: declared links vary with lambda (logit, mgstd5)",
          "links_tm varies across lambda", f"links_tm {links_lo:.2f} -> {links_hi:.2f}",
          links_hi > links_lo)

    # first leg identical across lambda in every pooled row (numerator invariance)
    num_spread = max(
        abs(float(vtab[(vtab.scope == "pooled") & (vtab.land == land)].L0.iloc[0]
                  - vtab[(vtab.scope == "pooled") & (vtab.land == land)].L1.iloc[0])
            - (float(vtab[(vtab.scope == "pooled") & (vtab.land == land)].L0.iloc[1]
                     - vtab[(vtab.scope == "pooled") & (vtab.land == land)].L1.iloc[1])))
        for land in LANDS)
    check("leg-1 level difference identical across lambda (pooled)",
          "max diff = 0.0", f"max diff = {num_spread:.1f}", num_spread == 0.0)

    # share rows reproduce exactly from the level columns of the emitted table
    rec_link = 100.0 * (vtab.L0 - vtab.L1) / vtab.gap
    rec_move = 100.0 * (vtab.L1 - vtab.L2) / vtab.gap
    rec_cred = 100.0 * (vtab.L2 - vtab.L3) / vtab.gap
    rec_dev = float(max((rec_link - vtab.link).abs().max(),
                        (rec_move - vtab.move).abs().max(),
                        (rec_cred - vtab.credit).abs().max()))
    check("verification table: shares reproduce from its own level columns",
          "max abs dev < 1e-9", f"max abs dev = {rec_dev:.1e}", rec_dev < 1e-9)

    # movement-leg smoothness: pull=0.5 sits between pull=0 and pull=1
    smooth_n = 0
    smooth_out = []
    for land in LANDS:
        for lam in LAMBDAS:
            for form in FORMS:
                g = runs[(runs.lam == lam) & (runs.land == land) & (runs.form == form)
                         & (runs.credit == "structural")]
                m = g.groupby("pull")[METRIC].mean()
                lo, mid, hi = float(m.loc[0.0]), float(m.loc[0.5]), float(m.loc[1.0])
                between = min(lo, hi) <= mid <= max(lo, hi)
                smooth_n += int(between)
                if not between:
                    dev = min(abs(mid - lo), abs(mid - hi))
                    smooth_out.append(f"{land}/{form}/lam{lam:.0f} dev {dev:.4f}")
    found = f"{smooth_n} of 8"
    if smooth_out:
        found += " (exceptions: " + "; ".join(sorted(smooth_out)) + ")"
    check("pull=0.5 level between pull=0 and pull=1 (movement leg, 8 cells)",
          "8 of 8", found, smooth_n == 8, assert_ok=False)

    # g=0 probe degeneracy: no links, lambda-blind
    pl = probe.links.abs().max()
    pp = probe.pivot_table(index=["land", "seed"], columns="lam", values=METRIC)
    pdiff = float((pp[0.0] - pp[1.0]).abs().max())
    check("g=0 probe (m0_m2_probe): no declared links, lambda-blind alignment",
          "links_tm = 0.0 and max lambda diff = 0.0",
          f"links_tm max = {pl:.1f}, max lambda diff = {pdiff:.1f}",
          pl == 0.0 and pdiff == 0.0)

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(OUT_DATA / "checks.csv", index=False)

    # ---- figure (house contract) ------------------------------------
    s = {**T.HOUSE_STYLE, "figsize": (11.0, 5.1)}
    fig, axes = plt.subplots(1, 2, figsize=s["figsize"])
    w = 0.26
    arms = [("link", "link formation only (pull = 0)", T.PALETTE["purple"]),
            ("move", "spatial co-movement (pull 0 -> 1)", T.PALETTE["orange"]),
            ("credit", "equal-split credit sharing", T.PALETTE["green"])]
    export_parts = []
    for ax, land in zip(axes, LANDS):
        sub = vtab[(vtab.scope == "pooled") & (vtab.land == land)].set_index("lam")
        x = np.arange(len(LAMBDAS))
        for i, (key, label, color) in enumerate(arms):
            vals = [float(sub.loc[lam, key]) for lam in LAMBDAS]
            bars = ax.bar(x + (i - 1) * w, vals, width=w, color=color, label=label)
            ax.bar_label(bars, labels=[f"{v:+.1f}%" for v in vals],
                         padding=3, fontsize=s["tick_size"] - 1)
            for lam, v in zip(LAMBDAS, vals):
                export_parts.append(dict(metric=f"m2share_{land}", landscape=land,
                                         lam=lam, arm=key, value=v, ci_lo=v, ci_hi=v,
                                         L0=float(sub.loc[lam, "L0"]),
                                         L1=float(sub.loc[lam, "L1"]),
                                         L2=float(sub.loc[lam, "L2"]),
                                         L3=float(sub.loc[lam, "L3"]),
                                         gap=float(sub.loc[lam, "gap"])))
        ax.axhline(0, color=T.PALETTE["gray"], linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([r"$\lambda = 0$" + "\n" + "prestige matching",
                            r"$\lambda = 1$" + "\n" + "closeness x local merit"],
                           fontsize=s["tick_size"])
        lo = min(float(sub[k].min()) for k, _, _ in arms)
        hi = max(float(sub[k].max()) for k, _, _ in arms)
        pad = max(30.0, 0.18 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
        T._cosmetics(ax, s,
                     f"{land}: mechanism shares of the M1 -> M2 gap",
                     "partner-choice rule (pooled over formulation)",
                     "share of the M1 -> M2 gap (%)")

    export = pd.DataFrame(export_parts)
    handles, labels = axes[0].get_legend_handles_labels()
    assert len(handles) == 3
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=3, fontsize=s["tick_size"], frameon=False)

    pl_pool = {land: vtab[(vtab.scope == "pooled") & (vtab.land == land)].set_index("lam")
               for land in LANDS}
    note = textwrap.fill(
        "source: data/path_b_postbug (baseline_off 40 runs = M1; cells_v1 800 runs "
        "pull=0; cells_v2 960 runs pull 0.5/1.0), chain M1 -> link-only (pull 0, "
        "structural) -> +co-movement (pull 1) -> +equal_split credit = full M2; "
        "gap = M1 - full; shares sum to 100% asserted (max dev < 1e-9). Metric: "
        "time-mean Corr(prestige, ideal_merit), c=0.5, g=0.1, H=200, 20 seeds x 2 "
        "formulations per node (pooled). M1/full (mgstd5): "
        + "; ".join(f"lam={lam:.0f} {pl_pool['mgstd5'].loc[lam, 'L0']:+.4f}/"
                    f"{pl_pool['mgstd5'].loc[lam, 'L3']:+.4f}" for lam in LAMBDAS)
        + "; noisypr5: "
        + "; ".join(f"lam={lam:.0f} {pl_pool['noisypr5'].loc[lam, 'L0']:+.4f}/"
                    f"{pl_pool['noisypr5'].loc[lam, 'L3']:+.4f}" for lam in LAMBDAS)
        + ". Model-internal, NOT calibrated.", width=155)
    fig.text(0.006, 0.012, note, fontsize=s["note_size"],
             color=T.PALETTE["gray"], va="bottom", linespacing=1.45)
    fig.tight_layout(rect=(0, 0.235, 1, 0.90))

    qa = visual_qa.assert_figure_clean(fig, [
        {"ax": axes[0], "metric": "m2share_mgstd5", "expect_bars": 6,
         "sidecar": export[export.landscape == "mgstd5"], "ci_cols": ("ci_lo", "ci_hi")},
        {"ax": axes[1], "metric": "m2share_noisypr5", "expect_bars": 6,
         "sidecar": export[export.landscape == "noisypr5"], "ci_cols": ("ci_lo", "ci_hi")},
    ])
    qa_failures = int(qa.get("_total", 0))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PNG, dpi=s["dpi"])
    fig.savefig(FIG_PNG.with_suffix(".pdf"))
    plt.close(fig)
    csv_path = T.write_figure_data(export, FIG_PNG)

    # ---- generated table fragments ---------------------------------
    def frag(tid: str, header: str, rows_md: list[str]) -> str:
        body = "\n".join([header] + rows_md)
        return f"<!-- TABLE:{tid} -->\n{body}\n<!-- /TABLE:{tid} -->\n"

    def cell(land: str, lam: float) -> str:
        return f"{land}, $\\lambda = {lam:.0f}$"

    P = {f"{r.land}_{r.lam:.0f}": r for r in vtab[vtab.scope == "pooled"].itertuples()}

    hdr_levels = ("| Cell | M1 (dyadic off) | Link only (pull 0) | +Movement (pull 1) | "
                  "Full (equal split) | Gap M1 -> full [95% CI] |\n"
                  "| :--- | ---: | ---: | ---: | ---: | :--- |")
    rows_levels = [
        f"| **{cell(land, lam)}** | {fmt(P[f'{land}_{lam:.0f}'].L0, 4)} | "
        f"{fmt(P[f'{land}_{lam:.0f}'].L1, 4)} | {fmt(P[f'{land}_{lam:.0f}'].L2, 4)} | "
        f"{fmt(P[f'{land}_{lam:.0f}'].L3, 4)} | {fmt(P[f'{land}_{lam:.0f}'].gap, 4)} "
        f"{fmt_ci((P[f'{land}_{lam:.0f}'].gap_lo, P[f'{land}_{lam:.0f}'].gap_hi), 4)} |"
        for land in LANDS for lam in LAMBDAS]
    (TAB_DIR / "table_levels.md").write_text(
        frag("levels", hdr_levels, rows_levels))

    hdr_shares = ("| Cell | Link formation | Co-movement | Credit sharing | \n"
                  "| :--- | :--- | :--- | :--- |")

    def share_cell(r, k: str) -> str:
        return (f"{fmt(getattr(r, k))} "
                f"{fmt_ci((getattr(r, k + '_lo'), getattr(r, k + '_hi')))}")

    rows_shares = [
        f"| **{cell(land, lam)}** | "
        + " | ".join(share_cell(P[f"{land}_{lam:.0f}"], k)
                     for k in ("link", "move", "credit")) + " |"
        for land in LANDS for lam in LAMBDAS]
    (TAB_DIR / "table_shares.md").write_text(
        frag("shares", hdr_shares, rows_shares))

    hdr_form = ("| Cell | Gap | Link formation | Co-movement | Credit sharing |\n"
                "| :--- | ---: | :--- | :--- | :--- |")
    rows_form = []
    for land in LANDS:
        for lam in LAMBDAS:
            for form in FORMS:
                r = vtab[(vtab.scope == "form") & (vtab.land == land)
                         & (vtab.lam == lam) & (vtab.form == form)].iloc[0]
                flag = " (near-degenerate)" if abs(r.gap) < 0.01 else ""
                rows_form.append(
                    f"| {land} $\\lambda = {lam:.0f}$ {form}{flag} | {fmt(r.gap, 4)} | "
                    f"{fmt(r.link)} | {fmt(r.move)} | {fmt(r.credit)} |")
    (TAB_DIR / "table_by_form.md").write_text(frag("by_form", hdr_form, rows_form))

    hdr_orderb = ("| Cell | Link formation | Credit sharing | Co-movement |\n"
                  "| :--- | :--- | :--- | :--- |")
    rows_orderb = [
        f"| **{cell(land, lam)}** | {fmt(P[f'{land}_{lam:.0f}'].link)} | "
        f"{fmt(P[f'{land}_{lam:.0f}'].credit_B)} | {fmt(P[f'{land}_{lam:.0f}'].move_B)} |"
        for land in LANDS for lam in LAMBDAS]
    (TAB_DIR / "table_order_b.md").write_text(frag("order_b", hdr_orderb, rows_orderb))

    hdr_checks = ("| Check | Expected | Found | Status |\n"
                  "| :--- | :--- | :--- | :--- |")
    rows_checks = [
        f"| {c['check']} | {c['expected']} | {c['found']} | **{c['status']}** |"
        for c in checks]
    (TAB_DIR / "table_checks.md").write_text(frag("checks", hdr_checks, rows_checks))

    # ---- report fill (token replacement, counts asserted) -----------
    p = pl_pool

    tokens = {
        "V_N_BASE": "40",
        "V_N_V1": "800",
        "V_N_V2": "960",
        "V_N_PROBE": "80",
        "V_A0_MG": fmt(p["mgstd5"].loc[0.0, "L0"], 4),
        "V_A1_MG": fmt(p["mgstd5"].loc[0.0, "L1"], 4),
        "V_A0_NP": fmt(p["noisypr5"].loc[0.0, "L0"], 4),
        "V_A1_NP": fmt(p["noisypr5"].loc[0.0, "L1"], 4),
        "V_LNKLEV_MG": fmt(float(p["mgstd5"].loc[0.0, "L0"] - p["mgstd5"].loc[0.0, "L1"]), 4),
        "V_LNKLEV_MG_CI": fmt_ci((
            float(vtab[(vtab.scope == "pooled") & (vtab.land == "mgstd5")].leg_link_lo.iloc[0]),
            float(vtab[(vtab.scope == "pooled") & (vtab.land == "mgstd5")].leg_link_hi.iloc[0])), 4),
        "V_LNKLEV_NP": fmt(float(p["noisypr5"].loc[0.0, "L0"] - p["noisypr5"].loc[0.0, "L1"]), 4),
        "V_LNKLEV_NP_CI": fmt_ci((
            float(vtab[(vtab.scope == "pooled") & (vtab.land == "noisypr5")].leg_link_lo.iloc[0]),
            float(vtab[(vtab.scope == "pooled") & (vtab.land == "noisypr5")].leg_link_hi.iloc[0])), 4),
        "V_GAP_MG_L0": fmt(p["mgstd5"].loc[0.0, "gap"], 4),
        "V_GAP_MG_L0_CI": fmt_ci((p["mgstd5"].loc[0.0, "gap_lo"], p["mgstd5"].loc[0.0, "gap_hi"]), 4),
        "V_GAP_MG_L1": fmt(p["mgstd5"].loc[1.0, "gap"], 4),
        "V_GAP_MG_L1_CI": fmt_ci((p["mgstd5"].loc[1.0, "gap_lo"], p["mgstd5"].loc[1.0, "gap_hi"]), 4),
        "V_GAP_NP_L0": fmt(p["noisypr5"].loc[0.0, "gap"], 4),
        "V_GAP_NP_L0_CI": fmt_ci((p["noisypr5"].loc[0.0, "gap_lo"], p["noisypr5"].loc[0.0, "gap_hi"]), 4),
        "V_GAP_NP_L1": fmt(p["noisypr5"].loc[1.0, "gap"], 4),
        "V_GAP_NP_L1_CI": fmt_ci((p["noisypr5"].loc[1.0, "gap_lo"], p["noisypr5"].loc[1.0, "gap_hi"]), 4),
        "V_LNK_MG_L0": fmt(p["mgstd5"].loc[0.0, "link"]),
        "V_MOV_MG_L0": fmt(p["mgstd5"].loc[0.0, "move"]),
        "V_MOV_MG_L0_CI": fmt_ci((p["mgstd5"].loc[0.0, "move_lo"], p["mgstd5"].loc[0.0, "move_hi"])),
        "V_CRD_MG_L0": fmt(p["mgstd5"].loc[0.0, "credit"]),
        "V_LNK_MG_L1": fmt(p["mgstd5"].loc[1.0, "link"]),
        "V_MOV_MG_L1": fmt(p["mgstd5"].loc[1.0, "move"]),
        "V_CRD_MG_L1": fmt(p["mgstd5"].loc[1.0, "credit"]),
        "V_MOV_MG_L1_B": fmt(p["mgstd5"].loc[1.0, "move_B"]),
        "V_CRD_MG_L1_B": fmt(p["mgstd5"].loc[1.0, "credit_B"]),
        "V_LNK_NP_L0": fmt(p["noisypr5"].loc[0.0, "link"]),
        "V_LNK_NP_L0_CI": fmt_ci((p["noisypr5"].loc[0.0, "link_lo"], p["noisypr5"].loc[0.0, "link_hi"])),
        "V_MOV_NP_L0": fmt(p["noisypr5"].loc[0.0, "move"]),
        "V_MOV_NP_L0_CI": fmt_ci((p["noisypr5"].loc[0.0, "move_lo"], p["noisypr5"].loc[0.0, "move_hi"])),
        "V_CRD_NP_L0": fmt(p["noisypr5"].loc[0.0, "credit"]),
        "V_LNK_NP_L1": fmt(p["noisypr5"].loc[1.0, "link"]),
        "V_LNK_NP_L1_CI": fmt_ci((p["noisypr5"].loc[1.0, "link_lo"], p["noisypr5"].loc[1.0, "link_hi"])),
        "V_MOV_NP_L1": fmt(p["noisypr5"].loc[1.0, "move"]),
        "V_MOV_NP_L1_CI": fmt_ci((p["noisypr5"].loc[1.0, "move_lo"], p["noisypr5"].loc[1.0, "move_hi"])),
        "V_CRD_NP_L1": fmt(p["noisypr5"].loc[1.0, "credit"]),
        "V_LAM_BLIND": fmt(float(spread.max()), 1, signed=False),
        "V_LINKS_LO": fmt(links_lo, 2, signed=False),
        "V_LINKS_HI": fmt(links_hi, 2, signed=False),
        "V_PROBE_LINKS": fmt(pl, 2, signed=False),
        "V_PROBE_DIFF": fmt(pdiff, 1, signed=False),
        "V_SMOOTH_N": str(smooth_n),
        "V_SUM_DEV": f"{max(max_sum_dev, max_sum_dev_b):.1e}",
        "V_QA_FAILS": str(qa_failures),
    }

    # the near-degenerate per-formulation cell (smallest pooled-adjacent gap)
    frm = vtab[vtab.scope == "form"]
    deg = frm.loc[frm.gap.abs().idxmin()]
    tokens["V_DEGEN_CELL"] = f"{deg.land} $\\lambda = {deg.lam:.0f}$ {deg.form}"
    tokens["V_DEGEN_GAP"] = fmt(float(deg.gap), 4)
    tokens["V_DEGEN_SHARES"] = (f"{fmt(float(deg.link))}% / {fmt(float(deg.move))}% / "
                                f"{fmt(float(deg.credit))}%")

    # per-formulation co-movement range at mgstd5, lambda = 0 (one prose token)
    mg0 = frm[(frm.land == "mgstd5") & (frm.lam == 0.0)]
    mv_lo, mv_hi = float(mg0.move.min()), float(mg0.move.max())
    tokens["V_MOV_MG_L0_FORMRANGE"] = (f"{fmt(mv_lo, 1, signed=False)}% to "
                                       f"{fmt(mv_hi, 1, signed=False)}%")

    text = TEMPLATE.read_text()
    for tok, val in tokens.items():
        marker = f"@@{tok}@@"
        n = text.count(marker)
        assert n >= 1, f"token {marker} not found in template"
        text = text.replace(marker, val)
    for tid in ("levels", "shares", "by_form", "order_b", "checks"):
        marker = f"@@T_{tid.upper()}@@"
        n = text.count(marker)
        assert n >= 1, f"table token {marker} not found in template"
        text = text.replace(marker,
                            (TAB_DIR / f"table_{tid}.md").read_text().strip())
    assert "@@" not in text, "unfilled token remains in report"
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    # shared-tree collision guard: output/reports/m2_decomposition_v1.md is a
    # first-committed deliverable of a sibling round (commit 681497d). Never
    # overwrite content this script did not generate (its generated reports
    # carry the <!-- TABLE:levels --> fragment markers); write the
    # reproduction report to a superseded-path fallback instead.
    target = REPORT
    if REPORT.exists() and "<!-- TABLE:levels -->" not in REPORT.read_text(encoding="utf-8"):
        target = REPORT.with_name("m2_decomposition_v1_superseded.md")
        print(f"COLLISION: {REPORT} holds foreign (first-committed) content; "
              f"writing this run's report to {target} instead.")
    target.write_text(text)

    print(f"QA failures: {qa_failures}")
    print(f"Wrote {FIG_PNG} / {FIG_PNG.with_suffix('.pdf')} / {csv_path}")
    print(f"Wrote {OUT_DATA / 'verification_table.csv'} and checks.csv")
    print(f"Wrote fragments under {TAB_DIR}")
    print(f"Wrote {target}")
    print()
    print(vtab[vtab.scope == "pooled"][["land", "lam", "L0", "L1", "L2", "L3", "gap",
                                        "link", "move", "credit"]].to_string(index=False))
    return None


if __name__ == "__main__":
    main()
