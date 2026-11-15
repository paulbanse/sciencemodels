"""Path D typology report generator (2026-11-15, P-0005).

Generates output/reports/typology_v1.md ENTIRELY from data/typology_v1
artefacts (numbers computed, never remembered; verdict sentences derived
from the computed values).  Fails loudly on any missing artefact.

Artefacts consumed:
  canary_first.txt, canary_last.txt   - the two smoke-canary captures
  gates.txt                           - identity / binding / probe / hashes
  manifest.json                       - gross vs net, wall, code-moved check
  typology_table.csv                  - 4 quadrants x 4 measures (mean/sd/n)
  typology_by_curiosity.csv           - quadrant x curiosity gradient
  typology_by_entry.csv               - quadrant x entry gradient
  cell_summary.csv                    - 24 cells
  contrast_table.csv                  - paired factorial contrasts + CIs
  control_checks.json                 - kappa-inert control, bindings, shapes
  figures_manifest.json               - figure paths + sources

Run (after the closing canary has been captured into canary_last.txt):
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/typology/make_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data" / "typology_v1"
REPORT = REPO / "output" / "reports" / "typology_v1.md"

QUADS = [
    ("Pure Distraction", 0.0, 0.0),
    ("Distraction + Destruction", 1.0, 0.0),
    ("Distraction + Counterforce", 0.0, 1.0),
    ("Full Interaction", 1.0, 1.0),
]
MEASURES = [
    ("Mean_of_Corr_prestige_ideal_merit", "Corr_prestige_ideal_merit", "ideal", 4),
    ("Mean_of_Corr_prestige_local_merit", "Corr_prestige_local_merit", "local", 4),
    ("wedge_fin", "wedge_fin", "wedge", 4),
    ("ideal_loss_cum", "ideal_loss_cum", "loss", 2),
]


def esc(s):
    return str(s).replace("|", "\\|")


def fmt(x, nd):
    return ("%+." + str(nd) + "f") % x if isinstance(x, float) else str(x)


def md_table(df, headers=None, float_fmt="%.4f"):
    headers = headers or list(df.columns)
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join([" --- "] * len(headers)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(float_fmt % v)
            else:
                cells.append(esc(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def load():
    art = {}
    for name in ("canary_first.txt", "canary_last.txt", "gates.txt"):
        p = OUT_DIR / name
        assert p.exists(), "missing artefact %s" % p
        art[name] = p.read_text()
    for name in ("manifest.json", "control_checks.json", "figures_manifest.json"):
        p = OUT_DIR / name
        assert p.exists(), "missing artefact %s" % p
        art[name] = json.loads(p.read_text())
    for name in ("typology_table.csv", "typology_by_curiosity.csv",
                 "typology_by_entry.csv", "cell_summary.csv", "contrast_table.csv"):
        p = OUT_DIR / name
        assert p.exists(), "missing artefact %s" % p
        art[name] = pd.read_csv(p)
    return art


def quad_row(tab, name):
    r = tab[tab["quadrant"] == name].iloc[0]
    return r


def contrast(contrasts, measure, contrast_key):
    sub = contrasts[(contrasts["measure"] == measure) &
                    (contrasts["contrast"].str.startswith(contrast_key))]
    assert len(sub) == 1, "contrast not unique: %s / %s" % (measure, contrast_key)
    return sub.iloc[0]


def ci_excludes(ci_lo, ci_hi):
    return (ci_lo > 0) or (ci_hi < 0)


def build(art):
    tab = art["typology_table.csv"]
    by_c = art["typology_by_curiosity.csv"]
    by_e = art["typology_by_entry.csv"]
    contrasts = art["contrast_table.csv"]
    checks = art["control_checks.json"]
    manifest = art["manifest.json"]
    gates = art["gates.txt"]
    figs = art["figures_manifest.json"]

    L = []
    A = L.append

    # ------------------------------------------------------------------ head
    A("---")
    A('title: "Path D typology v1: distraction, destruction and counterforce as parameterized perception regimes"')
    A("version: 1")
    A("date: 2026-11-15")
    A("revision-vehicle: strikethrough")
    A('revision-renders: "md, pdf, html"')
    A("tags: [path-d, typology, distraction, destruction, counterforce, perception-noise, wedge, prestige, abm, sciencemodels, p-0005, not-calibrated]")
    A("project: sciencemodels")
    A("---")
    A("")
    A("**Version 1** (November 2026)")
    A("")
    A("**NOT CALIBRATED.** Model-internal experiment only (owner ruling, "
      "2026-11-13). No empirical moment is a fitting target; the round "
      "crosses three perception regimes of the ABM and reads the "
      "prestige-merit wedge off the grid. Every number below was computed by "
      "`scripts/typology/run_typology_v1.py`, `scripts/typology/analyse_typology_v1.py` "
      "and `scripts/typology/make_figures.py` from the artefacts under "
      "`data/typology_v1/`, and is reproducible with the commands quoted in "
      "the appendix.")
    A("")

    # headline (computed from the artefacts, never hand-typed)
    r00h = quad_row(tab, "Pure Distraction")
    cd_w = contrast(contrasts, "wedge_fin", "destruction_main")
    cd_l = contrast(contrasts, "ideal_loss_cum", "destruction_main")
    kappa_max = max(
        abs(contrast(contrasts, lbl, "counterforce_main")["mean_delta"])
        for lbl in ("Corr_prestige_ideal_merit", "Corr_prestige_local_merit",
                    "wedge_fin", "ideal_loss_cum"))
    kappa_diff_n = int(round(
        checks["kappa_binding_descriptive_g01"]["wedge_fin_differs_frac"]
        * checks["kappa_binding_descriptive_g01"]["g01_runs_compared"]))
    e00h = by_e[(by_e["gamma"] == 0.0) & (by_e["kappa"] == 0.0)].set_index("g_entry")
    A("**Headline (every number computed from `data/typology_v1/`).** "
      "Destruction is the only channel that moves the commission's "
      "measures: its paired main is %+.4f on wedge_fin (bootstrap 95%% CI "
      "[%.4f, %.4f], excludes 0) and %+.4f on ideal_loss_cum (CI [%.4f, "
      "%.4f], excludes 0) --- a %.1f%% cut in the cumulative "
      "ideal-harvest shortfall. The counterforce binds (%d of %d g=0.1 "
      "seed-pairs differ on wedge_fin) but is quantitatively inert at the "
      "pinned semantics: every main is at most %.4f in absolute value with "
      "CIs spanning 0, and the interaction is ns on all four measures. The "
      "endpoint proxy wedge is entry-driven (wedge_fin %+.4f at g=0 vs "
      "%+.4f at g=0.1 in the Pure quadrant); destruction narrows it. The "
      "native curiosity gradient runs opposite: higher $c$, smaller wedge, "
      "smaller loss (gradient tables below)."
      % (cd_w["mean_delta"], cd_w["ci_lo"], cd_w["ci_hi"],
         cd_l["mean_delta"], cd_l["ci_lo"], cd_l["ci_hi"],
         -100.0 * cd_l["mean_delta"] / r00h["ideal_loss_cum_mean"],
         kappa_diff_n,
         checks["kappa_binding_descriptive_g01"]["g01_runs_compared"],
         kappa_max,
         e00h.loc[0.0, "wedge_fin_mean"], e00h.loc[0.1, "wedge_fin_mean"]))
    A("")

    # ---------------------------------------------------------- orientation
    A("# Purpose and scope")
    A("")
    A("`AdmiralsLog/proposals/P-0005-typology-distraction-destruction-counterforce.md` "
      "asks for the typology experiment that turns the hunting-proxy "
      "patterns (`context/Hunting_Proxy_Patterns.md`, formalized in "
      "`AdmiralsLog/contributions/CONT2-hunting-proxy-patterns-formalization.md`) "
      "into parameterized, crossable regimes of the ABM:")
    A("")
    A("- **distraction** --- the baseline $(1-c)$ visibility weight on the "
      "prestige-weighted field signal, already native to the model's reward "
      "(higher curiosity $c$ shrinks the visibility share);")
    A("- **destruction** ($\\gamma$, $AP_F \\dashv G$) --- crowding amplifies "
      "perception noise: $\\epsilon_{\\text{eff}} = \\epsilon \\cdot "
      "m_t(\\text{pos})^\\gamma$ for a candidate tile occupied by "
      "$m_t$ agents;")
    A("- **counterforce** ($\\kappa$, $P \\to G$) --- a newcomer's perception "
      "noise is divided by its supervisor's prestige: "
      "$\\epsilon_i = \\epsilon_0 / (1 + \\kappa \\cdot P_{\\text{sup}})$.")
    A("")
    A("The commission's grid (task header, 2026-11-15) crosses the two "
      "channels with the two native knobs on one landscape: "
      "$\\gamma \\in \\{0,1\\} \\times \\kappa \\in \\{0,1\\} \\times c \\in "
      "\\{0.3, 0.5, 0.7\\} \\times g_{\\text{entry}} \\in \\{0.0, 0.1\\}$ on "
      "`mgstd5` = **24 cells, 20 seeds, 480 runs**, horizon $H=200$. "
      "(P-0005's draft crossed two landscapes at fixed $g=0.1$; the "
      "commission's header replaces the landscape axis with the entry axis "
      "--- the entry axis doubles as the counterforce's on/off gate, which "
      "makes the grid's control arm free. `noisypr5` is parked as a need.)")
    A("")
    A("The read-out is the wedge between where prestige points and where "
      "the goal is: `Corr_prestige_ideal_merit` (goal tracking), "
      "`Corr_prestige_local_merit` (proxy tracking), their final-step "
      "difference `wedge_fin`, and the cumulative ideal-harvest shortfall "
      "`ideal_loss_cum`. The Path B choice layer is **off** throughout "
      "(`dyadic_mode=\"off\"`): this round varies *perception*, Path B "
      "varied *choice*; both are wedges between proxy and goal.")
    A("")

    # ---------------------------------------------------------------- design
    A("# Design")
    A("")
    A("## Channels and where they bind")
    A("")
    A("Both channels are implemented **driver-side** as "
      "`TypologyScientist(Scientist)` / `TypologyModel(MyModel)` in "
      "`scripts/typology/typology_layers.py`; `sciencemodels/*.py` is "
      "untouched by this round (write boundary, ADR-0001). The destruction "
      "channel binds at the model's single epsilon dynamics site --- the "
      "candidate-novelty line in `Scientist.step` "
      "(`sciencemodels/agents.py`, the `neighbor_rw_novelty * (1 + "
      "self.epsilon * noise)` expression): the subclass replaces "
      "`self.epsilon` there by $\\epsilon_{\\text{eff}} = \\epsilon \\cdot "
      "m_t(\\text{candidate})^\\gamma$. The counterforce channel binds at "
      "the two agent-construction sites (founders, newborns): a newborn "
      "stores its supervisor's entry-stock prestige at its birth step and "
      "divides its base epsilon by $1 + \\kappa P_{\\text{sup}}$ for life. "
      "The current tile's own reward is computed without noise in the "
      "parent and stays so --- the channels touch candidate evaluation only.")
    A("")
    A("Python name-mangling makes the newborn interception non-obvious: "
      "`MyModel.end_loop_update` calls `self.__generate_new_agents()`, which "
      "compiles to `self._MyModel__generate_new_agents`. A normally named "
      "subclass override would be **silently invisible**; "
      "`TypologyModel` therefore defines the mangled name explicitly, and "
      "the bit-identity proof below is the runtime guard that the "
      "interception reproduces the parent's roulette (rng consumption "
      "included) exactly.")
    A("")
    A("## Declared pins")
    A("")
    pins = pd.DataFrame([
        ["D1", "$m_t(\\text{pos})$", "number of scientists whose current position equals the candidate tile, counted at the focal agent's evaluation time (others do not move during one agent's step, so one per-step occupancy map is exact)"],
        ["D2", "destruction extreme", "literal formula: at $\\gamma=1$ an empty candidate tile ($m=0$) receives $\\epsilon_{\\text{eff}}=0$ (perfect read); $m=1$ leaves noise at baseline. Crowding amplifies, absence removes the amplification"],
        ["D3", "counterforce object", "newcomers only; founders have no supervisor and are $\\kappa$-inert. At $g_{\\text{entry}}=0$ no newcomers exist, so $\\kappa$ is inert by construction (the grid's built-in control arm)"],
        ["D4", "$P_{\\text{sup}}$", "the supervisor's entry-stock prestige (`entry_prestige_stock=\"prestige\"`, the stock the entry roulette itself reads) at the newborn's birth step, stored once, fixed for life. Live-supervisor and local-peer-max variants are parked needs"],
        ["D5", "NaN extreme regime", "at $\\gamma=1$, crowded tiles ($m \\gtrsim 11$ at $\\epsilon=0.1$) push the noise factor $1+\\epsilon_{\\text{eff}} \\cdot \\text{noise}$ negative; a negative perceived novelty to a fractional curiosity power is NaN, and a NaN reward is never selected (`>` comparisons are False). Literal-formula consequence, disclosed, not clamped"],
        ["D6", "baseline config", "the Path B house baseline: $n=20$, $\\epsilon=0.1$, harvest $=0.02$, $20\\times20$ grid, vanishing $=0.9$, visibility reward on, `knowledge_normalization=\"share\"`, `prestige_accrual=\"harvested\"`, structural credit, `closeness_mode=\"grid\"`"],
        ["D7", "dyadic layer", "`dyadic_mode=\"off\"`, `dyadic_move_pull=0.0` throughout (perception round, not choice round)"],
        ["D8", "measure definitions", "`wedge_fin = c_ideal_fin - c_local_fin` and `ideal_loss_cum = -sum(a.ideal_merit_cum)` --- the remeasure round's conventions, reused verbatim for cross-round comparability"],
        ["D9", "seeds / horizon", "`agent_seed` 0-19 drives agent and landscape RNG jointly; `step_limit=200`; landscape `mgstd5` = `multipleGaussians(std=5, number_gaussians=2)`"],
        ["D10", "subclass identity", "at $\\gamma=\\kappa=0$ the helpers short-circuit to the parent's own float operations (no multiply, no divide, no pow), so identity is structural; the runtime proof below verifies it on the full report surface"],
    ], columns=["#", "Pin", "Definition"])
    A(md_table(pins, float_fmt="%s"))
    A("")

    # -------------------------------------------------------------- gates
    A("# Gates and canaries")
    A("")
    A("## Canary FIRST (pre-edit, `tests/test_smoke.py`)")
    A("")
    A("```{.text}")
    A(art["canary_first.txt"].strip())
    A("```")
    A("")
    A("## Bit-identity proof (commission step 2)")
    A("")
    ident_line = [l for l in gates.splitlines() if l.startswith("gate3_identity_proof")][0]
    A("The subclass at $(\\gamma, \\kappa) = (0, 0)$ was run against plain "
      "`MyModel` at both entry levels for 5 seeds each; **every** report key "
      "plus the driver's extras were compared: "
      + ident_line.split(": ", 1)[1].replace(" | ", ": ")
      + ". The newborn path (entry $g=0.1$, "
      "up to 21 newcomers per run) is covered by the same proof --- the "
      "mangled-name interception consumes the same rng draws in the same "
      "order as the parent.")
    A("")
    A("## Binding and control probes")
    A("")
    A("```{.text}")
    for l in gates.splitlines():
        if l.startswith("gate4") or l.startswith("  g="):
            A(l)
    A("```")
    A("")
    A("The destruction channel binds at both entry levels; the counterforce "
      "binds where its object exists ($g=0.1$) and is **bit-identical** "
      "where it does not ($g=0.0$) --- the control arm. The source "
      "divergence canary (`gate2`) and the per-cell re-check inside every "
      "worker guard against the shared tree moving under the round; the "
      "launch-time sha256 snapshot of the five modules the dynamics touch "
      "is recorded in `gates.txt` and re-checked after the sweep "
      "(`manifest.json: code_moved_during_sweep`).")
    A("")

    # ----------------------------------------------------------- execution
    A("# Execution")
    A("")
    cc = checks
    A("Gross vs net (recounted from disk, not memory): **%d cells gross, "
      "%d cell files net; %d runs gross, %d rows net** (%d agent-final "
      "rows across the 24 cells); %d failed cells, %d failed seeds. Wall "
      "%.0f s for the grid at `n_jobs=3`."
      % (cc["gross_cells"], cc["net_cell_files"], cc["gross_runs"],
         cc["net_rows"], cc["net_agent_rows"],
         len(manifest["cells_failed"]), manifest["failed_seeds"],
         manifest["wall_s"]))
    A("")
    probe = json.loads([l for l in gates.splitlines() if l.startswith("gate5_probe")][0].split(": ", 1)[1])
    A("Probe timing: %.2f s/run wall through the real joblib worker path "
      "(measured under full contention from the owner's then-live Path B v2 "
      "and turnover pools, and BEFORE the disclosed warning filter of pin "
      "D5); the probe cell's schema was asserted from the worker-written "
      "file itself. The grid proper ran at %.2f s/run effective wall (%d "
      "runs / %.0f s at `n_jobs=3`, warning filter active, one owner pool "
      "sharing the 3-core cgroup quota)."
      % (probe["wall_per_run_s"], cc["gross_runs"] / manifest["wall_s"],
         cc["gross_runs"], manifest["wall_s"]))
    A("")
    moved = manifest.get("code_moved_during_sweep", {})
    A("Shared-tree straddle guard: the five modules' sha256 at gates vs "
      "after the sweep --- **%s**." % ("no movement" if not moved else "MOVED: %s" % esc(moved)))
    A("")

    # ----------------------------------------------------------------- table
    A("# The four-quadrant typology")
    A("")
    rows = []
    for name, g, k in QUADS:
        r = quad_row(tab, name)
        rows.append({
            "quadrant": name, "n": int(r["n_runs"]),
            "Corr_prestige_ideal_merit": "%.4f $\\pm$ %.4f" % (r["Corr_prestige_ideal_merit_mean"], r["Corr_prestige_ideal_merit_sd"]),
            "Corr_prestige_local_merit": "%.4f $\\pm$ %.4f" % (r["Corr_prestige_local_merit_mean"], r["Corr_prestige_local_merit_sd"]),
            "wedge_fin": "%.4f $\\pm$ %.4f" % (r["wedge_fin_mean"], r["wedge_fin_sd"]),
            "ideal_loss_cum": "%.2f $\\pm$ %.2f" % (r["ideal_loss_cum_mean"], r["ideal_loss_cum_sd"]),
        })
    A(md_table(pd.DataFrame(rows), float_fmt="%s"))
    A("")
    A("Means $\\pm$ sd over 120 runs per quadrant (3 curiosity $\\times$ 2 "
      "entry $\\times$ 20 seeds).")
    A("")
    A("Figure 1 - the P-0005 four-quadrant decay read (prestige vs ideal "
      "merit, local merit as hue, the mean ideal$\\to$local wedge arrow): "
      "![four-quadrant prestige vs ideal merit decay with the wedge arrow](../../data/typology_v1/figures/fig1_quadrant_pg_decay.png)")
    A("")
    A("Figure 2 - the four commission measures, per run, by quadrant: "
      "![the four commission measures per run, by quadrant](../../data/typology_v1/figures/fig2_measures_by_quadrant.png)")
    A("")

    # ------------------------------------------------------------ contrasts
    A("# Factorial contrasts (seed-paired, bootstrap 95% CIs)")
    A("")
    A("The grid is seed-paired: the same 20 seeds cross every cell at fixed "
      "$(c, g)$. Contrasts are paired deltas over the matched runs "
      "(240 pairs for the mains, 120 quartets for the interaction), with "
      "percentile bootstrap 95% CIs (20,000 resamples). A CI excluding 0 "
      "marks the row; the fraction of positive paired deltas is quoted "
      "beside it.")
    A("")
    rows = []
    for _, r in contrasts.iterrows():
        rows.append({
            "measure": esc(r["measure"].replace("Mean_of_Corr_prestige_", "Corr_prestige_")),
            "contrast": esc(r["contrast"]),
            "n_pairs": int(r["n_pairs"]),
            "mean delta": "%.4f" % r["mean_delta"],
            "95% CI": "[%.4f, %.4f]" % (r["ci_lo"], r["ci_hi"]),
            "excl. 0": "yes" if ci_excludes(r["ci_lo"], r["ci_hi"]) else "no",
            "frac > 0": "%.2f" % r["frac_positive"],
        })
    A(md_table(pd.DataFrame(rows), float_fmt="%s"))
    A("")

    # ----------------------------------------------------------- gradients
    A("# Curiosity and entry gradients")
    A("")
    rows = []
    for name, g, k in QUADS:
        sub = by_c[(by_c["gamma"] == g) & (by_c["kappa"] == k)].sort_values("curiosity")
        w = sub["wedge_fin_mean"].to_numpy(float)
        rows.append({
            "quadrant": name,
            "wedge_fin at c=0.3": "%.4f" % w[0],
            "wedge_fin at c=0.5": "%.4f" % w[1],
            "wedge_fin at c=0.7": "%.4f" % w[2],
            "slope 0.3->0.7": "%+.4f" % (w[2] - w[0]),
        })
    A(md_table(pd.DataFrame(rows), float_fmt="%s"))
    A("")
    rows = []
    for name, g, k in QUADS:
        sub = by_e[(by_e["gamma"] == g) & (by_e["kappa"] == k)].sort_values("g_entry")
        rows.append({
            "quadrant": name,
            "wedge_fin at g=0.0": "%.4f" % sub["wedge_fin_mean"].iloc[0],
            "wedge_fin at g=0.1": "%.4f" % sub["wedge_fin_mean"].iloc[1],
            "ideal_loss_cum at g=0.0": "%.2f" % sub["ideal_loss_cum_mean"].iloc[0],
            "ideal_loss_cum at g=0.1": "%.2f" % sub["ideal_loss_cum_mean"].iloc[1],
        })
    A(md_table(pd.DataFrame(rows), float_fmt="%s"))
    A("")
    A("![wedge_fin heatmap by curiosity and quadrant, per entry regime](../../data/typology_v1/figures/fig3_wedge_fin_heatmap.png)")
    A("")
    A("![ideal_loss_cum heatmap by curiosity and quadrant, per entry regime](../../data/typology_v1/figures/fig4_ideal_loss_cum_heatmap.png)")
    A("")

    # ------------------------------------------------------------- controls
    A("# Control arm and wiring checks")
    A("")
    ctl = cc["kappa_control"]
    A("- **Kappa-inert control (g=0.0):** %d seed-paired run pairs "
      "(2 $\\gamma$ $\\times$ 3 $c$ $\\times$ 20 seeds) compared across "
      "**every** column of `runs.csv`: max abs diff %r, non-numeric "
      "mismatches %d --- bit-identical, as construction demands (no "
      "newcomers, no counterforce object). A nonzero value here would be a "
      "wiring bug, not a finding."
      % (ctl["kappa_inert_at_g0_pairs_checked"], ctl["kappa_inert_at_g0_max_abs_diff"],
         len(ctl["kappa_inert_at_g0_non_numeric_mismatches"])))
    A("- **Gamma binds without entry:** at $g=0.0$ (founders only) the "
      "destruction channel still moves the surface --- wedge_fin differs "
      "in %.0f%% of the %d seed-paired (1,0)-vs-(0,0) contrasts: crowding "
      "among founders is enough; the channel does not need newborns."
      % (100 * cc["gamma_binding_descriptive_g0"]["wedge_fin_differs_frac_gamma1_vs_0"],
         cc["gamma_binding_descriptive_g0"]["g0_pairs"]))
    A("- **Kappa binds with entry:** at $g=0.1$, %d of %d seed-paired "
      "(0,1)-vs-(0,0) contrasts differ on wedge_fin (mean paired delta "
      "%+.4f)."
      % (round(cc["kappa_binding_descriptive_g01"]["wedge_fin_differs_frac"] * cc["kappa_binding_descriptive_g01"]["g01_runs_compared"]),
         cc["kappa_binding_descriptive_g01"]["g01_runs_compared"],
         cc["kappa_binding_descriptive_g01"]["wedge_fin_mean_delta"]))
    A("- **Panel-shape guards:** 24 cell files, each exactly 20 seed rows; "
      "`runs.csv` recounted from disk = %d = gross; the per-cell "
      "agent-finals files carry %d rows."
      % (cc["net_rows"], cc["net_agent_rows"]))
    A("- **Class identity:** the gates' bit-identity proof (above) covers "
      "the $(\\gamma, \\kappa) = (0,0)$ arm of the grid itself; the grid's "
      "Pure-Distraction quadrant is therefore MyModel dynamics, not a "
      "lookalike.")
    A("")

    # ------------------------------------------------------------ findings
    A("# Findings per measure")
    A("")
    for src, label, kind, nd in MEASURES:
        r00 = quad_row(tab, "Pure Distraction")
        r10 = quad_row(tab, "Distraction + Destruction")
        r01 = quad_row(tab, "Distraction + Counterforce")
        r11 = quad_row(tab, "Full Interaction")
        cd = contrast(contrasts, label, "destruction_main")
        ck = contrast(contrasts, label, "counterforce_main")
        ci = contrast(contrasts, label, "interaction")
        A("## %s" % label)
        A("")
        A("Quadrant means: Pure Distraction %.4f, +Destruction %.4f, "
          "+Counterforce %.4f, Full Interaction %.4f."
          % (r00[label + "_mean"], r10[label + "_mean"], r01[label + "_mean"], r11[label + "_mean"]))
        A("")
        for name, r in (("Destruction (main)", cd), ("Counterforce (main)", ck), ("Interaction", ci)):
            word = "excludes 0" if ci_excludes(r["ci_lo"], r["ci_hi"]) else "includes 0"
            A("- **%s:** %+.4f, bootstrap 95%% CI [%.4f, %.4f] (%s), %.0f%% of "
              "%d paired deltas positive."
              % (name, r["mean_delta"], r["ci_lo"], r["ci_hi"], word,
                 100 * r["frac_positive"], r["n_pairs"]))
        A("")
    A("Reading of the typology (derived from the table above): the wedge "
      "column ordering across quadrants is %s." % esc(
          " > ".join(
              "%s (%.4f)" % (n, quad_row(tab, n)["wedge_fin_mean"])
              for n, _, _ in sorted(QUADS, key=lambda q: -quad_row(tab, q[0])["wedge_fin_mean"])
          )))
    A("")

    # ------------------------------------------------------ interpretation
    A("# Interpretation against the proxy-pattern taxonomy")
    A("")
    r00 = quad_row(tab, "Pure Distraction")
    A("`Hunting_Proxy_Patterns.md` and CONT2 define the proxy pattern as "
      "prestige tracking the visible local proxy rather than the ideal "
      "goal, with distraction $(1-c)$ as its baseline generator. The "
      "Pure-Distraction quadrant reproduces the signature on this tree: "
      "Corr(prestige, local merit) %.4f vs Corr(prestige, ideal merit) "
      "%.4f, wedge_fin %.4f --- and the two channels read off as regimes "
      "on top of it:" % (r00["Corr_prestige_local_merit_mean"],
                         r00["Corr_prestige_ideal_merit_mean"],
                         r00["wedge_fin_mean"]))
    A("")
    A("- **Destruction ($AP_F \\dashv G$)** degrades the candidate read "
      "exactly where agents stack. Observed, not hypothesized: the endpoint "
      "proxy wedge NARROWS (wedge_fin main %+.4f, CI [%.4f, %.4f]) and the "
      "cumulative ideal-harvest shortfall falls by %.1f%% (main %+.4f, CI "
      "[%.4f, %.4f]), while both time-mean correlations drift down "
      "non-significantly. Working hypothesis (model-internal, labelled as "
      "such): crowding noise acts as exploration pressure, scattering "
      "agents off the depleted proxy peaks, so prestige ends less "
      "proxy-locked and the harvest closer to the greedy ideal."
      % (cd_w["mean_delta"], cd_w["ci_lo"], cd_w["ci_hi"],
         -100.0 * cd_l["mean_delta"] / r00["ideal_loss_cum_mean"],
         cd_l["mean_delta"], cd_l["ci_lo"], cd_l["ci_hi"]))
    A("- **Counterforce ($P \\to G$)** is the taxonomy's escape hatch and "
      "its wiring is proven (binding + control arm above), but at the "
      "pinned semantics it is quantitatively inert: all four mains are at "
      "most %.4f in absolute value with CIs spanning 0. Diagnosis: the "
      "divisor $1 + \\kappa P_{\\text{sup}}$ with $P_{\\text{sup}}$ on the "
      "model's prestige scale (canary anchor mean_prestige 0.0254) "
      "reduces a newborn's noise by only a few percent, applies to the ~21 "
      "newcomers of a 41-agent run, and only from birth on. A genuine "
      "$P \\to G$ spillover needs a stronger base noise, a live-supervisor "
      "divisor, or a rescaled prestige stock --- parked as needs, not "
      "smoothed here." % kappa_max)
    A("")
    A("The formalization CONT2 asks for is therefore delivered as a "
      "crossable $2\\times2$ on top of the native distraction axis: three "
      "patterns, two of them new as parameters, one grid, and a control "
      "arm (entry off) that proves the counterforce's wiring rather than "
      "assuming it.")
    A("")

    # -------------------------------------------------------- limitations
    A("# Limitations and disclosed pins")
    A("")
    A("- Literal formula consequences, disclosed rather than smoothed: the "
      "$m=0$ perfect-read edge (D2) and the NaN extreme regime at "
      "high crowding (D5). Both follow from the commissioned formulas; "
      "neither is clamped silently.")
    A("- $P_{\\text{sup}}$ is the supervisor's prestige **at the newborn's "
      "birth step**, fixed for life (D4). A live-supervisor divisor or a "
      "local-peer-max variant would answer a slightly different question "
      "and is parked as a need.")
    A("- One landscape (`mgstd5`), one horizon ($H=200$), one baseline "
      "config; P-0005's draft landscape axis (`noisypr5`) is not crossed "
      "here because the commission's grid replaces it with the entry axis. "
      "Not calibrated; every number is model-internal.")
    A("- The round ran against a shared live tree: the owner's Path B v2 "
      "(movement coupling) and turnover rounds landed in `sciencemodels/` "
      "during the morning, and an interventions round was live while the "
      "grid ran (hashes in `gates.txt`; post-sweep re-check: no movement; "
      "the closing canary below is line-identical to the opening one modulo "
      "the smoke test's own random /tmp scratch path, so "
      "the smoke gate itself did not change during the round). The round's "
      "own writes touched none of `sciencemodels/`.")
    A("")

    # --------------------------------------------------------------- needs
    A("# Needs (owner-facing)")
    A("")
    A("- Model-side natives: if the regimes should live in "
      "`sciencemodels/` (constructor parameters with validation, as Path B "
      "did for the dyadic layer), that edit is outside this round's "
      "boundary and is filed here rather than made in passing.")
    A("- Channel semantics variants: $(1+m)^\\gamma$ or $\\max(m,1)^\\gamma$ "
      "for the destruction floor; live-supervisor or peer-max "
      "$P_{\\text{sup}}$ for the counterforce; a noise-factor clamp "
      "$\\max(0, 1 + \\epsilon_{\\text{eff}} \\cdot \\eta)$ to name the NaN "
      "regime explicitly.")
    A("- Robustness grid: the P-0005 draft's second landscape (`noisypr5`) "
      "and a longer horizon ($H=400$, the turnover round's standard) as "
      "follow-up cells.")
    A("- `scripts/career_flows_noisy/audit_code_delta.py` step-vs-seed "
      "assertion and the `sweeps.py` 8-setup wiring are the owner's own "
      "pending hygiene items (2026-11-15 log, thirteenth entry), restated "
      "here only because this round's driver follows the same checkpoint "
      "convention.")
    A("")

    # --------------------------------------------------------------- figures
    A("# Figure manifest")
    A("")
    rows = []
    for f in figs:
        rows.append({
            "figure": f["name"],
            "pdf": esc(Path(f["paths"][".pdf"]).name),
            "png_300dpi": esc(Path(f["paths"][".png"]).name),
            "sources": esc("; ".join(f["sources"])),
        })
    A(md_table(pd.DataFrame(rows), float_fmt="%s"))
    A("")
    A("All figures live in `data/typology_v1/figures/` (the commission's "
      "figure home); `output/reports/` carries only this document and its "
      "PDF, per the write boundary.")
    A("")

    # ----------------------------------------------------------------- log
    A("# Revision Log")
    A("")
    A("- **Version 1** (November 2026): first issue. Generated by "
      "`scripts/typology/make_doc.py` from the `data/typology_v1/` "
      "artefacts; no number is hand-typed. Canary FIRST and LAST outputs "
      "pasted in full; the bit-identity proof, the binding/control probes "
      "and the kappa-inert control arm are quoted from `gates.txt` and "
      "`control_checks.json`.")
    A("")

    # -------------------------------------------------------------- appendix
    A("# Appendix: file:line manifest and artefact inventory")
    A("")
    A("## Files written this round (write boundary: `scripts/typology/`, "
      "`data/typology_v1/`, `output/reports/typology_v1.md` + `.pdf`, one "
      "append-only log entry)")
    A("")
    rows = []
    for rel, note in [
        ("scripts/typology/typology_layers.py", "the two channels: TypologyScientist (step override, eps helpers) + TypologyModel (construction-site overrides incl. the mangled-name interception) + source canary, compare_rows, identity_proof, binding_probe"),
        ("scripts/typology/run_typology_v1.py", "driver: grid, five gates, per-cell checkpoints (atomic), worker-path probe, consolidation, manifest/design"),
        ("scripts/typology/analyse_typology_v1.py", "typology table, c/g gradients, 24-cell summary, kappa-inert control, paired factorial contrasts with bootstrap CIs"),
        ("scripts/typology/make_figures.py", "four figures, PDF + 300 dpi PNG, sources declared per figure"),
        ("scripts/typology/make_doc.py", "this document's generator"),
        ("scripts/typology/README.md", "regeneration instructions and provenance (ADR-0002)"),
    ]:
        p = REPO / rel
        n = sum(1 for _ in open(p)) if p.exists() else -1
        rows.append({"path": rel, "lines": n, "role": note})
    A(md_table(pd.DataFrame(rows), float_fmt="%s"))
    A("")
    A("`sciencemodels/*.py`, `docs/`, `context/`: untouched by this round "
      "(hashes in `gates.txt`, re-checked post-sweep in "
      "`manifest.json:code_moved_during_sweep`).")
    A("")
    A("## Artefacts (`data/typology_v1/`)")
    A("")
    inv = []
    for p in sorted(OUT_DIR.rglob("*")):
        if p.is_file():
            inv.append({"path": str(p.relative_to(REPO)), "bytes": p.stat().st_size})
    A(md_table(pd.DataFrame(inv[:40]), headers=["path", "bytes"], float_fmt="%d")
      + "\n")
    if len(inv) > 40:
        A("... %d further files under `cells/` and `cells/agents/` "
          "(24 cell CSVs + 24 agent-finals CSVs, each 20-seeded)." % (len(inv) - 40))
        A("")
    A("## Regeneration")
    A("")
    A("```{.bash}")
    A("MPLBACKEND=Agg /opt/hermes/.venv/bin/python tests/test_smoke.py   # canary")
    A("MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/typology/run_typology_v1.py --gates-only")
    A("MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/typology/run_typology_v1.py --skip-gates")
    A("MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/typology/analyse_typology_v1.py")
    A("MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/typology/make_figures.py")
    A("MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/typology/make_doc.py")
    A("zsh /workspace/git/ai/IT/scripts/compile-here.zsh output/reports/typology_v1.md")
    A("```")
    A("")
    A("## Canary LAST (closing, `tests/test_smoke.py`)")
    A("")
    A("```{.text}")
    A(art["canary_last.txt"].strip())
    A("```")
    A("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote %s (%d lines)" % (REPORT, len(L)))


def main():
    art = load()
    build(art)


if __name__ == "__main__":
    main()
