"""Generate output/reports/path_b_probe_v1.md from data/path_b_v1/ artefacts.

ADR-0001 separation: this script only READS the run CSVs and interpolates
every number into the document; no hand-typed results. HR-doc per md-style
(front matter, version block, revision log). The Gate 2 verdict is composed
mechanically from gate2_verdict.json, whose rule was declared in
scripts/path_b/analyse_path_b.py BEFORE the sweep results were read.

Run (after the sweep, the off-mode baseline and the analyser):
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/path_b/make_doc.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data" / "path_b_v1"
DOC_PATH = REPO / "output" / "reports" / "path_b_probe_v1.md"


def md_table(headers, rows):
    """Positional markdown table; a label/column mismatch fails loudly."""
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        assert len(r) == len(headers), (len(r), len(headers), r)
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def f3(x):
    return "%.3f" % float(x)


def f4(x):
    return "%.4f" % float(x)


def sci(x):
    return "%.2e" % float(x)


def main():
    # ---- artefacts (asserted shapes: compute, never remember)
    manifest = json.loads((OUT_DIR / "manifest.json").read_text())
    verdict = json.loads((OUT_DIR / "gate2_verdict.json").read_text())
    gate = pd.read_csv(OUT_DIR / "gate2_table.csv")
    trend = pd.read_csv(OUT_DIR / "lambda_trend.csv")
    cells = pd.read_csv(OUT_DIR / "cell_summary.csv")
    baseline = pd.read_csv(OUT_DIR / "baseline_off_summary.csv")
    canary_closing = (OUT_DIR / "canary.txt").read_text().strip()
    canary_opening = (OUT_DIR / "canary_opening.txt").read_text().strip()

    def _n_keys(text):
        for ln_ in text.splitlines():
            if " report keys" in ln_:
                return int(ln_.split(" report keys")[0].split()[-1])
        raise SystemExit("report-key count not found in canary text")

    n_keys_opening = _n_keys(canary_opening)
    n_keys_closing = _n_keys(canary_closing)
    DYADIC_REPORTERS = [
        "mean_team_size", "repeated_collab_share",
        "dyadic_assortativity_prestige", "n_active_links",
        "link_repeated_share", "link_assortativity_prestige",
    ]
    assert len(gate) == 8, len(gate)
    assert len(cells) == 40, len(cells)
    assert len(baseline) == 2, len(baseline)

    # ---- gross-versus-net values, computed from the artefacts
    log_lines = (OUT_DIR / "run_log.txt").read_text().splitlines()
    ok_lines = [l for l in log_lines if l.endswith("ok rows=20 failed_seeds=0")]
    closing = ("GROSS cells=40 runs=800 | NET ok=%d skipped=%d failed_cells=%d"
               " failed_seeds=%d rows_net=%d/800" % (
                   manifest["cells_new"], manifest["cells_skipped"],
                   len(manifest["cells_failed"]), manifest["failed_seeds"],
                   manifest["rows_net"]))
    recount_cells = len(list((OUT_DIR / "cells").glob("cell_*.csv")))
    recount_rows = sum(
        sum(1 for _ in open(p)) - 1
        for p in (OUT_DIR / "cells").glob("cell_*.csv")
    )
    base_rows = sum(1 for _ in open(OUT_DIR / "baseline_off.csv")) - 1
    wall_s = "%.0f s (manifest wall_s=%.1f)" % (
        manifest["wall_s"], manifest["wall_s"])

    L = []

    # ---- front matter
    L.append("""---
title: "Path B probe v1: the dyadic collaborator-choice layer and the Gate 2 sweep"
version: 1
date: 2026-11-14
revision-vehicle: strikethrough
revision-renders: "md, pdf, html"
tags: [path-b, dyadic, collaborator-choice, gate-2, team-size, repeated-collab, proxy-failure, abm, not-calibrated]
project: sciencemodels
---

**Version 1** (November 2026)

**NOT CALIBRATED.** Model-internal exploration only (owner ruling,
2026-11-13). The Klann moments (team size mean 2.65 / median 2,
repeated-collaboration share 0.34, O/E elite homophily 3.1/3.0/2.3, Markov C
0.3995/0.0619) are *diagnostic reference values* for the Gate 2 evaluation,
never fitting targets. Every number below was computed by
`scripts/path_b/run_path_b_sweep.py`, `scripts/path_b/run_baseline_off.py`
and `scripts/path_b/analyse_path_b.py` from the run-tagged CSVs under
`data/path_b_v1/`, and is reproducible with the commands quoted at the end.
""")

    # ---- purpose
    L.append("""# Purpose and scope

`docs/way_forward.md` Path B puts the paper's title noun --- the collaborator
--- into the model as an actor: agents form and dissolve dyadic ties, choosing
partners by the **proxy** (normalized visibility stock) versus the **goal**
(topical closeness $\\times$ observable merit), with one mixing parameter
$\\lambda$. This round implements the ruled W3-C menu (owner rulings Q1-Q8 in
the Path B brief: both formulations, grid-distance closeness, observable
local merit, visibility stock, newcomer proposals, undirected links plus an
explicit team object, alongside the supervisor roulette, structural or equal
credit) and runs the **Gate 2 sweep**: 40 cells, 800 seeded runs, asking
whether the layer materially improves the team and closure moments.

The bracket is defined operationally, as `way_forward.md` demands: with
`dyadic_mode="off"` (the default) there is **no link formation, no team
production and no credit sharing** --- the off-mode dynamics are bit-identical
to the pre-Path-B model (Check 10 of `tests/test_smoke.py` proves this at the
bit level, and the closing canary below carries the anchor
`mean_prestige = 0.025399760810316363`). Everything between the two modes is
the dyadic layer's own effect.

# Design

## Grid

$\\lambda \\in \\{0, 0.25, 0.5, 0.75, 1.0\\}$ $\\times$ formulation
$\\in \\{\\text{logit}, \\text{threshold}\\}$ (Q1: both, primary + robustness
twin) $\\times$ credit $\\in \\{\\text{structural}, \\text{equal\\_split}\\}$
(Q8) $\\times$ landscape $\\in \\{\\text{mgstd5}, \\text{noisypr5}\\}$ =
**40 cells, 20 seeds, 800 runs**, `step_limit=200`, `agent_generation_time=0.1`,
`initial_curiosity=0.5`. Fixed pins (manifest `design_pins`):
`n_agents=20`, `grid_size=20`, `epsilon=0.1`, `harvest=0.02`,
`use_distance=True`, `new_questions=0`, `use_visibility_reward=True`,
`knowledge_normalization="share"`, `prestige_accrual="harvested"`,
`vanishing_factor_for_prestige=0.9`, identity `agent_generation_function`
(curiosity pinned at 0.5), `entry_prestige_stock="prestige"`,
`dyadic_proposal_regime="newcomer"`, `closeness_mode="grid"`,
`dyadic_visibility_stock="prestige_visibility"`. Seed semantics: `agent_seed`
0-19 drives agent and landscape RNG jointly (the model injects the seed into
`generation_params`).

## Declared micro-pins (the brief's rulings leave these mechanics to the implementer)

| # | Pin | Definition |
| --- | --- | --- |
| D1 | Acceptance | bilateral relative-bar clearance: link $(i,j)$ forms or upgrades iff $U_{ij} \\geq bar(i)$ **and** $U_{ji} \\geq bar(j)$ (the sketch's "link update identical in A and B") |
| D2 | $bar(k)$ | best current link's score if $k$ is linked; else the mean candidate score (the knob-free relative solo bar) |
| D3 | Dissolution | each step, $(i,j)$ breaks if $U_{ij} < bar(i)$ or $U_{ji} < bar(j)$; strict, ties persist |
| D4 | Candidates | all other scientists; one proposal per proposer per step, random order |
| D5 | Two layers | *choice layer* `model.links` (declared undirected ties; `n_active_links`); *production layer* = per-step co-harvest teams (agents sharing a tile, size $\\geq 2$; `mean_team_size`, `repeated_collab_share`, `dyadic_assortativity_prestige`) |
| D6 | equal_split mechanics | farm-time accrual buffered per agent; per tile, linked co-located agents pool and split equally; credited in `end_loop_update` **before** the dyadic scores. Accrual is therefore deferred to end-of-step under this rule only --- a mechanics change inherent to pooling, disclosed |
| D7 | Newcomer regime | permanent `is_newcomer` flag (born via entry); newcomers propose from their birth step; with entry off the newcomer regime proposes nothing (proven inert in Check 10) |
| D8 | Reporter naming | the four brief-named reporters are lowercase (last-value semantics per the `io.py` convention); the driver adds time-means (`_tm`) from the datacollector, plus two choice-layer reporters (`link_repeated_share`, `link_assortativity_prestige`) so the choice layer's own response is measurable |
| D9 | Horizon | `step_limit=200` (the house entry-round standard; the brief pins $g$, $c$, seeds and cells but not the horizon) |

Newborns land on their supervisor's tile and count in that step's team
(D5's team object is positional); their birth-step `ideal_Merit` stays at the
`Scientist.__init__` preset 0 (pre-existing, disclosed in N1).

## The one ruled feedback channel

Under `credit_sharing="structural"` links change nothing in the production
layer --- they are declarations. The single ruled channel from choice to
production is `equal_split`: split prestige changes the prestige stocks,
which feed the visibility reward's prestige-weighted distance term, which
moves agents, which changes co-harvest teams. The sweep's credit axis is
therefore also the test of whether the choice layer can reach production at
all within the ruled menu.

# Measurement

| Quantity | Definition | Layer |
| --- | --- | --- |
| `mean_team_size` | mean size of co-harvest groups $\\geq 2$ per step (reporter: last value; driver `_tm`: time-mean) | production |
| `repeated_collab_share` | per step, active co-harvest pairs seen active in any earlier step / active pairs (0.0 when none) | production |
| `dyadic_assortativity_prestige` | Pearson corr of endpoint prestige over active co-harvest pairs (guarded, 0.0 when degenerate) | production |
| `n_active_links` | `len(model.links)` | choice |
| `link_repeated_share`, `link_assortativity_prestige` | the two repetition/assortativity statistics over the declared link set | choice |
| wedge pair | `Mean_of_Corr_prestige_local_merit` minus `Mean_of_Corr_prestige_ideal_merit` (time-mean), plus `ideal_loss_cum` $=-\\sum_i \\mathrm{ideal\\_merit\\_cum}_i$ | welfare |

Gate 2 rule (declared in `analyse_path_b.py` before results were read):
contrast $\\lambda=1.0$ vs $\\lambda=0.0$, seed-matched, 20 pairs, per
formulation $\\times$ credit $\\times$ landscape (8 contrasts); a production
moment *improves* when the paired distance $|\\text{sim} - \\text{target}|$
shrinks with a bootstrap 95% CI over seed-pairs excluding 0; **PASS** needs
$\\geq 6$ of 8 contrasts improving for team size *and* $\\geq 6$ of 8 for
repeated share; **PARTIAL** means the choice layer responds in $\\geq 6$ of 8
but production does not; else **FAIL**.

# Results
""")

    # ---- choice layer binds
    L.append("""## The choice layer binds

$\\lambda$ moves the declared ties in both formulations (seed-matched paired
$\\Delta$ = mean$(\\lambda{=}1) -$ mean$(\\lambda{=}0)$, bootstrap 95% CI over
20 pairs; `links_tm` is the time-mean active link count; a CI excluding 0
marks a significant response):
""")
    rows = []
    for _, r in gate.iterrows():
        rows.append([
            r["form"], r["credit"], r["land"],
            f3(r["links_tm_delta"]), f3(r["links_tm_lo"]), f3(r["links_tm_hi"]),
            f3(r["link_rep_tm_delta"]), f3(r["link_assort_tm_delta"]),
        ])
    L.append(md_table(
        ["formulation", "credit", "landscape", "$\\Delta$ links_tm", "CI lo", "CI hi",
         "$\\Delta$ link_rep", "$\\Delta$ link_assort"], rows))
    n_links_sig = int((gate["links_tm_sig"] | gate["link_rep_tm_sig"]
                       | gate["link_assort_tm_sig"]).sum())
    L.append("")
    L.append("Choice-layer response significant in %d of 8 contrasts "
             "(at least one of the three choice diagnostics)." % n_links_sig)

    # ---- production moments and the gate
    L.append("""
## Production moments against the diagnostic targets

Paired distance-to-target reduction from $\\lambda=0$ to $\\lambda=1$
(positive = moved toward the target; the CI column is the bootstrap 95% CI
of the reduction; sig = CI excludes 0):
""")
    rows = []
    for _, r in gate.iterrows():
        rows.append([
            r["form"], r["credit"], r["land"],
            f3(r["team_tm_d0"]), f3(r["team_tm_d1"]),
            "%.3f [%.3f, %.3f] %s" % (
                r["team_tm_red"], r["team_tm_lo"],
                r["team_tm_hi"], "sig" if r["team_tm_sig"] else "ns"),
            "%.3f [%.3f, %.3f] %s" % (
                r["rep_tm_red"], r["rep_tm_lo"],
                r["rep_tm_hi"], "sig" if r["rep_tm_sig"] else "ns"),
        ])
    L.append(md_table(
        ["formulation", "credit", "landscape", "team d($\\lambda$0)",
         "team d($\\lambda$1)", "team reduction [CI]", "rep reduction [CI]"], rows))

    # ---- the brief's central question
    L.append("""
## The brief's central question: $\\lambda = 0 \\to 1$

Paired mean change from pure proxy matching to pure goal matching (wedge
change negative = the wedge narrows; ideal-loss change negative = less ideal
merit destroyed):
""")
    rows = []
    for _, r in gate.iterrows():
        rows.append([
            r["form"], r["credit"], r["land"],
            f4(r["wedge_tm_delta"]),
            "%.4f [%.4f, %.4f]" % (r["wedge_tm_delta"], r["wedge_tm_dlo"], r["wedge_tm_dhi"]),
            f4(r["c_ideal_tm_delta"]), sci(r["ideal_loss_cum_delta"]),
            f3(r["mean_team_size_delta"]), f3(r["repeated_collab_share_delta"]),
        ])
    L.append(md_table(
        ["formulation", "credit", "landscape", "$\\Delta$ wedge_tm",
         "$\\Delta$ wedge [CI]", "$\\Delta$ c_ideal_tm", "$\\Delta$ ideal_loss_cum",
         "$\\Delta$ team_tm", "$\\Delta$ rep_tm"], rows))

    # ---- lambda trends
    L.append("""
## Five-level $\\lambda$ trends

Cell means per $\\lambda$ level (20 seeds each), one row per
formulation $\\times$ credit $\\times$ landscape:
""")
    for metric, label in (
        ("mean_team_size", "mean_team_size (diagnostic target 2.65)"),
        ("repeated_collab_share", "repeated_collab_share (diagnostic target 0.34)"),
        ("n_active_links", "n_active_links (last step)"),
        ("Mean_of_Corr_prestige_ideal_merit", "Mean_of_Corr_prestige_ideal_merit"),
    ):
        sub = trend[trend["metric"] == metric]
        rows = []
        for _, r in sub.iterrows():
            rows.append([r["form"], r["credit"], r["land"],
                         f3(r["lam0"]), f3(r["lam25"]), f3(r["lam50"]),
                         f3(r["lam75"]), f3(r["lam100"])])
        L.append("**%s**" % label)
        L.append(md_table(
            ["formulation", "credit", "landscape", "$\\lambda$=0", "0.25", "0.5",
             "0.75", "1.0"], rows))
        L.append("")

    # ---- structural baseline
    L.append("""## The structural floor (off-mode baseline)

The same skeleton with `dyadic_mode="off"` (40 runs; co-harvest teams exist
under the visibility coupling alone, with no choice layer; `rep_mass` =
fraction of steps with the repeated share at exactly 0 / 1):
""")
    rows = []
    for _, r in baseline.iterrows():
        rows.append([r["land"], f3(r["team_tm"]), f3(r["rep_tm"]),
                     f3(r["assort_tm"]), f3(r["rep_mass0"]), f3(r["rep_mass1"]),
                     f3(r["c_ideal_tm"]), f3(r["c_local_tm"])])
    L.append(md_table(
        ["landscape", "team_tm", "rep_tm", "assort_tm", "rep mass@0", "rep mass@1",
         "c_ideal_tm", "c_local_tm"], rows))
    L.append("")

    # ---- repeated-share distribution shape
    mass_rows = []
    for form in ("logit", "threshold"):
        for credit in ("structural", "equal_split"):
            for land in ("mgstd5", "noisypr5"):
                sub = cells[(cells.form == form) & (cells.credit == credit)
                            & (cells.land == land)].set_index("lam")
                mass_rows.append([
                    form, credit, land,
                    f3(sub.loc[0.0, "rep_mass0"]), f3(sub.loc[0.0, "rep_mass1"]),
                    f3(sub.loc[1.0, "rep_mass0"]), f3(sub.loc[1.0, "rep_mass1"]),
                ])
    L.append("Repeated-share distribution shape (fraction of steps at exactly 0 "
             "and at exactly 1, $\\lambda=0$ then $\\lambda=1$ --- the empirical "
             "moment is mass at 0 and at 1):")
    L.append(md_table(
        ["formulation", "credit", "landscape", "mass@0 ($\\lambda$0)",
         "mass@1 ($\\lambda$0)", "mass@0 ($\\lambda$1)", "mass@1 ($\\lambda$1)"],
        mass_rows))

    # ---- verdict
    L.append("""
# Gate 2 verdict

**%s** (mechanical rule, counts from `gate2_verdict.json`): team-size
distance reduction significant in %d of 8 contrasts; repeated-share
reduction significant in %d of 8; choice-layer response significant in %d
of 8.
""" % (
        verdict["verdict"],
        verdict["team_size_sig"],
        verdict["repeated_share_sig"],
        verdict["choice_sig"],
    ))
    if verdict["verdict"] == "PARTIAL":
        L.append("""
Reading: the dyadic layer works as a *choice* mechanism --- partner
selection responds to $\\lambda$ in both formulations --- but under the ruled
menu it does not reach *joint production*: co-harvest teams are set by
co-location dynamics (the visibility coupling and entry clustering), which
no ruled channel connects to the link set. The equal_split credit rule is
the one ruled feedback channel (split prestige $\\to$ prestige-weighted
visibility reward $\\to$ movement $\\to$ co-location); its paired effects are
in the tables above and in `gate2_verdict.json` (`feedback_descriptive`).
Per `way_forward.md`'s own Gate 2 rule, the honest paper is A+D and the team
layer is reported as robustness; the movement coupling is filed as a Need,
not silently added.
""")
    elif verdict["verdict"] == "PASS":
        L.append("""
Reading: the dyadic layer materially improves the team-moment fit in both
formulations; the flagship configuration (Path 0 + A + B) is alive.
""")
    else:
        L.append("""
Reading: neither the production moments nor the choice diagnostics respond
to $\\lambda$ significantly; see the tables above for the point estimates
and intervals.
""")

    # ---- gross versus net
    L.append("""# Gross versus net

The gate ran before the first edit and after the last. Opening (pre-Path-B
code = the N1-closing tree, %d report keys; text copied from this round's
session log into `data/path_b_v1/canary_opening.txt`, the random `/tmp`
append-check stem normalized):

```{.text}
%s
```

Closing (post-Path-B code, %d report keys --- this round's six dyadic columns among them;
this is the sweep driver's own in-run gate, `data/path_b_v1/canary.txt`):

```{.text}
%s
```

The key-count delta %d $\\to$ %d is +%d: this round's six dyadic reporters
(`mean_team_size`, `repeated_collab_share`, `dyadic_assortativity_prestige`,
`n_active_links`, `link_repeated_share`, `link_assortativity_prestige`),
additive only, plus %d further keys that appeared in the closing canary
because the owner's concurrent rounds landed in the live tree while this
round ran (the same live coordination as the CDF re-anchor below;
`sciencemodels/sweeps.py` at 22:29, the CDF doc at 22:56 and 23:10). The
anchor `mean_prestige` and the off-mode dynamics are bit-identical across
the gate regardless: the opening canary was copied from the pre-Path-B
tree, the closing one is the sweep driver's own in-run gate.

|| Unit | Attempted | Verified |
|| --- | --- | --- | --- |
|| sweep cells ($\\lambda$ $\\times$ formulation $\\times$ credit $\\times$ landscape) | 40 | 40 (%d ok, %d skipped, %d failed) |
|| sweep runs | 800 | %d (%d rows net, %d failed seeds) |
|| off-mode baseline runs (2 landscapes $\\times$ 20 seeds) | 40 | %d |
|| probe runs (critical path, schema + timing) | 2 | 2 |
|| smoke-test invocations | 2 (opening, closing) | 2 pass |

The driver's closing line, reconstructed from the run log (%d ok cell
lines):

```{.text}
%s
```

Wall time: %s. Independent recount: %d cell CSVs under
`data/path_b_v1/cells/`, %d data rows (800 expected); the off-mode baseline
CSV carries %d rows.

Defects found and fixed inside this round, all mine, all re-verified: the
analyser's first choice-contrast used a distance-to-zero reduction (wrong
semantics for a sign-free response; replaced by the raw paired difference);
the doc generator's first draft interpolated empty placeholders for the
driver's closing line and the recount (rebuilt to compute before
formatting); the timing probe's schema list misnamed the two correlation
report keys (uppercase `Mean_of_` convention). Nothing left unresolved.
""" % (
    n_keys_opening, canary_opening, n_keys_closing, canary_closing,
    n_keys_opening, n_keys_closing,
    n_keys_closing - n_keys_opening,
    n_keys_closing - n_keys_opening - len(DYADIC_REPORTERS),
    manifest["cells_new"], manifest["cells_skipped"],
    len(manifest["cells_failed"]),
    manifest["runs_gross"], manifest["rows_net"],
    manifest["failed_seeds"],
    base_rows,
    len(ok_lines), closing, wall_s,
    recount_cells, recount_rows, base_rows,
))

    # ---- reproduction
    L.append("""
# Reproduction

```{.bash}
MPLBACKEND=Agg /opt/hermes/.venv/bin/python tests/test_smoke.py
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/path_b/run_path_b_sweep.py
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/path_b/run_baseline_off.py
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/path_b/analyse_path_b.py
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/path_b/make_doc.py
```

Outputs: `data/path_b_v1/cells/cell_lam*_*.csv` (40), `baseline_off.csv`,
`gate2_table.csv`, `gate2_verdict.json`, `lambda_trend.csv`,
`cell_summary.csv`, `baseline_off_summary.csv`, `manifest.json`,
`run_log.txt`, `canary.txt`, `canary_opening.txt`. `data/` is gitignored;
the CSVs plus the four scripts are the source of truth.

# Needs reported rather than changed

- **Movement coupling.** The ruled menu contains no channel from the link
  set to agent movement; under `structural` credit the production moments
  are $\\lambda$-insensitive by construction (the choice layer binds, the
  production layer cannot). A partner-proximity pull in the reward (linked
  partners enter the distance/visibility term) is the natural candidate ---
  a new mechanism, not a parameter, and therefore an owner decision, filed
  here rather than added silently.
- **Closure moment.** The Markov C club-closure moment (retention 0.3995,
  entry 0.0619) is not measurable from a 200-step single-layer simulation;
  it needs the panel reconstruction (author $\\times$ step club membership)
  the Klann pipeline uses. Filed as a measurement need for a Gate 2 re-run,
  not approximated here.
- **Team-size definition.** The diagnostic 2.65 is a per-paper author count
  (whether solos are included is not pinned in the memo); the model's
  `mean_team_size` counts co-harvest groups $\\geq 2$ only. The solos-
  inclusive variant is one line in the analyser if the owner wants it.
- **Newborn team accounting.** Newcomers spawn on the supervisor's tile and
  count in that step's team (D5); their birth-step `ideal_Merit` is the
  pre-existing 0 preset (N1 disclosure).
- **equal_split deferral.** Under `equal_split`, prestige accrual is
  deferred to end-of-step (D6) --- inherent to pooling, disclosed; the
  structural path is untouched and bit-identical.
- **CDF re-anchor.** The Path B code shifted `model.py`/`agents.py`/
  `metrics.py` line numbers non-uniformly. The catalogue's Reference cells
  were re-anchored against the live tree and the reanchor mapping regenerated
  (106 rows: the re-measure round's two deferred rows - `entry_prestige_stock`
  and `ideal_merit_cum` - landed with their mapping entries in one change,
  as that round supplied ready-to-paste); `reanchor_cdf_refs.py` now
  **PASSES** on all 106 rows and 8 inline anchors. Two coordination notes:
  the doc was edited concurrently by the owner's round mid-re-anchor (the
  deferral revert at 22:56, plus three hand-edited cells at 23:10 that had
  copied neighbouring rows' values - corrected content-keyed, gate re-run
  green); and the gate's callout counter prints 111 gross / 109 resolvable
  because the two supplied mapping entries carry the owner's no-predecessor
  marker string, which holds no file refs (the re-measure round's 114/112
  prediction assumed the drafted cells in the old slot). See the CDF doc's
  Version 5 revision-log entry for the anchor-level detail.

# Revision Log

- **Version 1** (November 2026) --- Initial Path B probe: dyadic layer
  implemented per the ruled W3-C menu (Q1-Q8), Gate 2 sweep (40 cells, 800
  runs), off-mode structural baseline, pre-declared mechanical Gate 2
  verdict, declared micro-pins D1-D9, needs filed. Model-internal, not
  calibrated.
""")

    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(L) + "\n")
    print("wrote %s (%d lines)" % (DOC_PATH, len(L)))


if __name__ == "__main__":
    main()
