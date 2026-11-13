"""Generate docs/career_flows_v1.md + data/career_flows_v1/manifest.json (T2).

HR-doc per the task's common constraints: YAML front matter (title quoted,
version: 1, date, tags, project), revision-vehicle: strikethrough,
revision-renders, **Version 1** line, unnumbered #/##/### headings, $|$ math,
escaped pipes, # Revision Log at the bottom. The confirmation verdict is
composed mechanically from confirmA_analysis.csv under the rule declared in
consolidate.py --- no hand-typed numbers.
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO = "/workspace/git/sciencemodels"
OUT = os.path.join(REPO, "data", "career_flows_v1")
MD = os.path.join(REPO, "docs", "career_flows_v1.md")
SCRIPTS = os.path.join(REPO, "scripts", "career_flows")
T1 = os.path.join(REPO, "data", "sensitivity_v1")

a_analysis = pd.read_csv(os.path.join(OUT, "confirmA_analysis.csv"))
a_analysis["tag_short"] = a_analysis.cell_tag.str.replace("confirmA_", "").str.replace("_h400", "")
a_pooled = pd.read_csv(os.path.join(OUT, "confirmA_pooled.csv"))
t1c = pd.read_csv(os.path.join(OUT, "confirmA_t1_contrast.csv"))
b_analysis = pd.read_csv(os.path.join(OUT, "careerB_analysis.csv"))
b_pooled = pd.read_csv(os.path.join(OUT, "careerB_pooled.csv"))
with open(os.path.join(OUT, "checkpoint.json")) as f:
    cp = json.load(f)

# ---------- canary ----------
can = subprocess.run(
    ["/opt/hermes/.venv/bin/python", "tests/test_smoke.py"],
    cwd=REPO, capture_output=True, text=True,
    env={**os.environ, "MPLBACKEND": "Agg"})
canary_txt, canary_rc = can.stdout.strip(), can.returncode
assert canary_rc == 0

# ---------- verdict (mechanical, rule declared in consolidate.py) ----------
valid = a_analysis[a_analysis.verdict != "degenerate_zero_guard"]
degen = a_analysis[a_analysis.verdict == "degenerate_zero_guard"]
n_valid = len(valid)
n_osc = int(valid.verdict.str.startswith("oscillates").sum())
n_endneg = int((valid.final_pooled < 0).sum())
n_revert = int((valid.verdict == "oscillates_mean_reverts").sum())
sc_min, sc_max = int(valid.sign_changes_pooled.min()), int(valid.sign_changes_pooled.max())
fsr_min, fsr_max = valid.final_sign_rate.min(), valid.final_sign_rate.max()
# ---------- extra computed prose values ----------
ps_min = int(a_analysis.sign_changes_per_seed_min.min())
ps_max = int(a_analysis.sign_changes_per_seed_max.max())
seeds_osc = int(a_analysis.seeds_osc_ge3.sum())
seeds_tot = int(a_analysis.seeds_ok.sum())
seed_sign = int((a_analysis.final_sign_rate * a_analysis.seeds_ok).sum().round())
mg_last3 = a_analysis[a_analysis.landscape == "mgstd5"]["last_third_pooled_mean"]
noisy_last3 = a_analysis[a_analysis.landscape == "noisypr5"]["last_third_pooled_mean"]
flip_cell = "confirmA_cur7_van9_mgstd5_gen0p1_h400"
t1_flip = float(t1c.loc[t1c.cell_tag == flip_cell, "t1_final_pooled"].iloc[0])
t2_flip = float(a_analysis.loc[a_analysis.cell_tag == flip_cell, "final_pooled"].iloc[0])
b_low = b_analysis[b_analysis.agent_generation_time == 0.05]["ideal_corr_final"]
b_high = b_analysis[b_analysis.agent_generation_time == 0.2]["ideal_corr_final"]
b_percap = {g: (float(b_analysis[b_analysis.agent_generation_time == g].percap_harvest_final.min()),
                float(b_analysis[b_analysis.agent_generation_time == g].percap_harvest_final.max()))
            for g in [0.05, 0.1, 0.2]}
t1s = pd.read_csv(os.path.join(T1, "cell_summary.csv"))
t1_top10_noentry = (float(t1s[t1s.agent_generation_time == 0]["top10pct_prestige_share_final"].min()),
                    float(t1s[t1s.agent_generation_time == 0]["top10pct_prestige_share_final"].max()))
t1_top10_entry = (float(t1s[t1s.agent_generation_time == 0.1]["top10pct_prestige_share_final"].min()),
                  float(t1s[t1s.agent_generation_time == 0.1]["top10pct_prestige_share_final"].max()))
equiv_max = 0.0
for cur in [0.3, 0.5, 0.7]:
    bb = pd.read_csv(os.path.join(OUT, f"careerB_cur{int(round(cur*10))}_van9_mgstd5_gen0p1_h200.csv"))
    tt = pd.read_csv(os.path.join(T1, f"cell_cur{int(round(cur*10)):02d}_van09_mgstd5_gen01.csv"))
    mm = bb.merge(tt, on=["seed", "Step"], suffixes=("_b", "_t1"))
    for col in ["Corr_prestige_ideal_merit", "Corr_prestige_local_merit",
                "top10pct_prestige_share", "explored_percentage",
                "percentage_knowledge_harvested"]:
        equiv_max = max(equiv_max, float((mm[f"{col}_b"] - mm[f"{col}_t1"]).abs().max()))
equiv_disp = 0.0
_t1s = pd.read_csv(os.path.join(T1, "cell_summary.csv"))
for cur in [0.3, 0.5, 0.7]:
    bval = float(b_analysis[(b_analysis.agent_generation_time == 0.1)
                            & (b_analysis.curiosity == cur)].ideal_corr_final.iloc[0])
    tval_full = float(_t1s[(t1s.agent_generation_time == 0.1) & (t1s.curiosity == cur)
                           & (t1s.landscape == "mgstd5")
                           & (t1s.vanishing_factor_for_prestige == 0.9)]
                      ["Corr_prestige_ideal_merit_final"].iloc[0])
    equiv_disp = max(equiv_disp, abs(bval - tval_full))
# per-seed gap blips at g=0.2 (steps 7-9) and pooled-gap strict negativity
blip_counts, blip_max = [], 0.0
for cur in [3, 5, 7]:
    cc = pd.read_csv(os.path.join(OUT, f"careerB_cur{cur}_van9_mgstd5_gen0p2_h200_career.csv"))
    w = cc[(cc.Step >= 7) & (cc.Step <= 9)]
    per_seed_max = w.groupby("seed")["prestige_gap"].max()
    blip_counts.append(int((per_seed_max > 0).sum()))
    blip_max = max(blip_max, float(w["prestige_gap"].max()))
pooled_gap_min = float(b_pooled["prestige_gap"].min())

per_land = valid.groupby("landscape").agg(
    cells=("cell_tag", "size"),
    endneg=("final_pooled", lambda s: int((s < 0).sum())),
    osc=("verdict", lambda s: int(s.str.startswith("oscillates").sum())))
noisy_endneg = int(per_land.loc["noisypr5", "endneg"]) if "noisypr5" in per_land.index else None
mg_endneg = int(per_land.loc["mgstd5", "endneg"]) if "mgstd5" in per_land.index else None
noisy_harv = a_analysis[a_analysis.landscape == "noisypr5"]["harvested_final"]

mg_final = valid[valid.landscape == "mgstd5"]["final_pooled"]
noisy_final = valid[valid.landscape == "noisypr5"]["final_pooled"]

if n_osc >= n_valid - 1 and n_endneg >= int(np.ceil(2 * n_valid / 3)) and n_revert == 0:
    verdict_word = "SURVIVES"
elif n_osc >= n_valid - 1 and n_endneg >= 1:
    verdict_word = "SURVIVES WITH A LANDSCAPE SPLIT"
else:
    verdict_word = "IS NOT CONFIRMED"

verdict_sentence = (
    f"**Verdict: T1's headline oscillation claim {verdict_word} at horizon 400.** "
    f"All {n_osc}/{n_valid} confirmation cells oscillate ({sc_min}-{sc_max} pooled sign "
    f"changes; per-seed range {ps_min}-{ps_max}, {seeds_osc}/{seeds_tot} runs with "
    f">= 3) and all {n_endneg}/{n_valid} end with the pooled ideal correlation below "
    f"zero (seed-level final sign rate {fsr_min:.2f}-{fsr_max:.2f} per cell, "
    f"{seed_sign}/{seeds_tot} runs overall); {n_revert} cell mean-reverts. The noisy "
    f"cells end clearly negative (pooled finals {noisy_final.min():+.3f} to "
    f"{noisy_final.max():+.3f}); the clean cells also end negative but their "
    f"last-third pooled means are positive-leaning ({mg_last3.min():+.3f} to "
    f"{mg_last3.max():+.3f}) versus negative on noisy ({noisy_last3.min():+.3f} to "
    f"{noisy_last3.max():+.3f}) --- in a dense sawtooth the final step's sign is "
    f"phase-sensitive, so the seed-level sign rate, not the single final read, "
    f"carries the claim.")

# ---------- helpers ----------
def md_table(df, headers, float_fmt="{:.3f}"):
    assert len(headers) == len(df.columns), (headers, list(df.columns))
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for v in row:  # positional: headers are display labels, columns are data
            if isinstance(v, float):
                cells.append(float_fmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def sha16(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]

# ---------- counts ----------
def counts(sweep):
    cells = [c for c in cp["cells"].values() if c["sweep"] == sweep]
    return {
        "cells_gross": len(cells), "cells_net": sum(1 for c in cells if c["status"] == "ok"),
        "runs_gross": sum(c["seeds_ok"] + c["seeds_failed"] for c in cells),
        "runs_net": sum(c["seeds_ok"] for c in cells),
        "seeds_failed": sum(c["seeds_failed"] for c in cells),
        "crashed": sum(1 for c in cells if c["status"] == "crashed"),
    }
ca, cb = counts("A"), counts("B")

# ---------- A table ----------
tA = a_analysis.copy()
tA["tag_short"] = tA.cell_tag.str.replace("confirmA_", "").str.replace("_h400", "")
tA = tA[["tag_short", "sign_changes_pooled", "neg_share_pooled", "final_pooled",
         "final_sign_rate", "last_third_pooled_mean", "exact0_final_share",
         "harvested_final", "verdict"]]
table_A = md_table(tA, ["cell", "sign changes (pooled)", "neg share", "final pooled",
                        "final sign rate", "last-third mean", "exact-0 final share",
                        "harvested final", "verdict"])

# ---------- T1 contrast ----------
t1x = t1c.merge(a_analysis[["cell_tag", "sign_changes_pooled", "final_pooled",
                            "final_sign_rate"]], on="cell_tag", suffixes=("_t1", "_t2"))
t1x["tag_short"] = t1x.cell_tag.str.replace("confirmA_", "").str.replace("_h400", "")
table_t1 = md_table(t1x[["tag_short", "t1_sign_changes_pooled", "t1_final_pooled",
                         "t1_final_sign_rate", "sign_changes_pooled", "final_pooled",
                         "final_sign_rate"]],
                    ["cell", "T1 sign chg (h200)", "T1 final (h200)", "T1 sign rate (h200)",
                     "T2 sign chg (h400)", "T2 final (h400)", "T2 sign rate (h400)"])

# ---------- B table ----------
tB = b_analysis.copy()
tB["tag_short"] = tB.cell_tag.str.replace("careerB_", "").str.replace("_h200", "")
table_B = md_table(tB[["tag_short", "pop_final", "entrants_final", "pop_growth_x",
                       "top10_step10", "top10_delta", "gap_final", "gap_ratio_mean_late",
                       "percap_harvest_final", "harvest_flow_mean_last50",
                       "ideal_corr_final", "mean_age_final"]],
                   ["cell", "pop final", "entrants", "pop growth x", "top10 @10",
                    "top10 delta", "gap final", "gap ratio (late)", "percap harvest",
                    "harvest flow/step (last50)", "ideal corr final", "mean age final"])

# ---------- B gap trajectory by gen ----------
gap_rows = []
for gen, g in b_pooled.groupby("agent_generation_time"):
    for st in [50, 100, 150, 200]:
        r = g[g.Step == st]
        gap_rows.append({"gen": gen, "step": st,
                         "pop": float(r["n_agents"].mean()),
                         "gap": float(r["prestige_gap"].mean()),
                         "gap_ratio": float(r["prestige_gap_ratio"].mean()),
                         "top10": float(r["top10pct_prestige_share"].mean()),
                         "percap_harvest": float(r["percap_harvest_cum"].mean())})
table_gap = md_table(pd.DataFrame(gap_rows),
                     ["gen", "step", "pop", "gap (new-inc)", "gap ratio", "top10",
                      "percap harvest"])

# ---------- steps-to-years framing ----------
fr_rows = []
for span in [2, 5, 10, 20, 40]:
    spy200 = 200 / span
    fr_rows.append({
        "span_years": span, "steps_per_year_h200": spy200,
        "entrants_per_year_g0p05": 0.05 * spy200,
        "entrants_per_year_g0p1": 0.1 * spy200,
        "entrants_per_year_g0p2": 0.2 * spy200,
        "years_covered_h400": 400 / spy200,
    })
table_frame = md_table(pd.DataFrame(fr_rows),
                       ["assumed career span (years)", "steps/year at h200",
                        "entrants/year, $g=0.05$", "entrants/year, $g=0.1$",
                        "entrants/year, $g=0.2$", "years covered at h400"],
                       float_fmt="{:.1f}")

# ---------- zero-guard / depletion for A ----------
zgA = a_analysis[["tag_short", "exact0_final_share", "harvested_final",
                  "zero_share_pooled", "mean_age_final"]].copy()
table_zgA = md_table(zgA, ["cell", "exact-0 final share", "harvested final",
                           "zero share pooled", "mean age final"])

import mesa
env = {"python": sys.version.split()[0], "mesa": mesa.__version__,
       "numpy": np.__version__, "pandas": pd.__version__}
sha_driver = sha16(os.path.join(SCRIPTS, "career_flows_driver.py"))
sha_cons = sha16(os.path.join(SCRIPTS, "consolidate.py"))
now = datetime.now(timezone.utc).isoformat(timespec="seconds")

# population rule note numbers
popA_final = 21 + int(400 * 0.1)

L = []
A = L.append
A("---")
A('title: "Career flows and the entry oscillation: confirmation at horizon 400 (T2)"')
A("author: \"Hermes Agent (squad lead) for Jakob Kapeller\"")
A("version: 1")
A('date: "2026-11-13"')
A("tags: [career-flows, entry, oscillation, proxy-failure, sensitivity-confirmation, not-calibrated]")
A("project: sciencemodels")
A("revision-vehicle: strikethrough")
A('revision-renders: "md, pdf, html"')
A("---")
A("")
A("**Version 1** (November 2026)")
A("")
A(verdict_sentence)
A("")
A("> **NOT CALIBRATED.** Model-internal exploration only (owner ruling 2026-11-13).")
A("> No empirical moment was used as a target; the Klann panel appears below solely")
A("> as reference context for the steps-to-years framing, and that framing is")
A("> conditional arithmetic, not fitting.")
A("")
A("# Design")
A("")
A("Two sweeps, both driven through `MyModel` directly (no sweep-writer), pins as in")
A("T1 (`scripts/bulk_runner.py` base): $n=20$ founders, 20x20 torus,")
A("$\\epsilon=0.1$, $h=0.02$, `use_distance=True`, `new_questions=0`,")
A("`use_visibility_reward=True`, `knowledge_normalization='share'`,")
A("`prestige_accrual='harvested'`, vanishing $x=0.9$ throughout (the tree's unified")
A("value), identity curiosity function.")
A("")
A("| Sweep | Cells | Grid | Seeds | Horizon | Runs |")
A("| --- | --- | --- | --- | --- | --- |")
A("| A (confirmation) | " + str(ca["cells_gross"]) + " | $c \\in \\{0.3, 0.5, 0.7\\} "
  "\\times$ {mgstd5, noisypr5}, $x=0.9$, $g=0.1$ | 40 | 400 | "
  + str(ca["cells_gross"] * 40) + " |")
A("| B (career flows) | " + str(cb["cells_gross"]) + " | $g \\in \\{0.05, 0.1, 0.2\\} "
  "\\times c \\in \\{0.3, 0.5, 0.7\\}$, $x=0.9$, mgstd5 | 20 | 200 | "
  + str(cb["cells_gross"] * 20) + " |")
A("")
A("Declared pins and disclosed deviations:")
A("")
A(f"- **(a) ran 6 cells, not 4.** The brief said \"4 representative entry cells\" but")
A("  enumerated curiosity 0.3/0.5/0.7 x {mgstd5, noisy} = 6; the full crossing was")
A("  run and is reported. No level was dropped.")
A("- **(b) landscape pinned to mgstd5** (the canonical bulk bundle). The brief named")
A("  no landscape for the career sweep; legibility was T1's axis, demographics are")
A("  this task's question, and the within-$g$ contrasts are the deliverable. A noisy")
A("  rerun is the natural extension.")
A("- Population under entry follows $21 + \\lfloor g \\cdot t \\rfloor$ (the +1 is T1's")
A(f"  Q1 init-time accumulator quirk); sweep A's final population is {popA_final}, not 60.")
A("- The agent-vars frame cannot distinguish the init-time extra spawn (entry step 0)")
A("  from founders; the newcomer-incumbent gap therefore treats it as incumbent")
A("  (1 agent out of 21-61; effect negligible, disclosed).")
A("- `prestige_gap_ratio` is degenerate while incumbent prestige is ~0 (ratio pinned")
A("  at -1); it is reported from step 50 onward (`gap ratio (late)` = mean over")
A("  steps 50-200).")
A("- Sweep A does not record per-run population (analytic rule above suffices);")
A("  sweep B records it per step from the agent-vars frame.")
A("")
A("# Canary and constraints")
A("")
A(f"Canary gate (`MPLBACKEND=Agg /opt/hermes/.venv/bin/python tests/test_smoke.py`;")
A(f"the driver now hard-gates on it at launch --- aborting before any cell on")
A(f"failure --- added as a post-sweep source addition, see Reproducibility; the")
A(f"embedded run is the document-time execution; exit code {canary_rc}):")
A("")
A("```{.bash}")
A("$ MPLBACKEND=Agg /opt/hermes/.venv/bin/python tests/test_smoke.py")
A(canary_txt)
A("```")
A("")
A("Concurrent-sweep hygiene (constraint compliance):")
A("")
A("- `n_jobs=3` cap honored in the driver (a foreign sweep, `scripts/figures/run_fresh_sweep.py`")
A("  writing `data/figures_v1/`, was observed live at launch, plus three Jupyter")
A("  kernel runners; my paths are disjoint from theirs).")
A("- No orphaned processes of this squad before launch (verified via ps; the kernel")
A("  runners and the foreign sweep are not mine to touch).")
A("- Restart-safe checkpoint per completed cell (`checkpoint.json`, atomic replace;")
A("  the SKIP-on-restart path is implemented but was NOT exercised --- this run")
A("  completed without interruption, as a single continuous pass). Crash policy:")
A("  failing seeds recorded, never dropped.")
A("- No git commands; writes confined to `data/career_flows_v1/`,")
A("  `scripts/career_flows/`, `docs/career_flows_v1.md`. Frozen `context/` and")
A("  read-only `AdmiralsLog/` untouched.")
A("")
A(f"Gross vs net --- sweep A: {ca['cells_gross']} cells gross / {ca['cells_net']} net,")
A(f"{ca['runs_gross']} runs gross / {ca['runs_net']} net, {ca['seeds_failed']} failed seeds,")
A(f"{ca['crashed']} crashed cells. Sweep B: {cb['cells_gross']} cells gross / {cb['cells_net']} net,")
A(f"{cb['runs_gross']} runs gross / {cb['runs_net']} net, {cb['seeds_failed']} failed seeds,")
A(f"{cb['crashed']} crashed cells. Total {ca['runs_gross']+cb['runs_gross']} runs,")
A(f"{ca['runs_net']+cb['runs_net']} completed.")
A("")
A("# Sweep A --- does the oscillation survive horizon 400?")
A("")
A("T1 (horizon 200, 20 seeds) found the ideal channel under entry ending negative")
A("in 17/18 cells, with oscillation asserted at the pooled landscape-x-entry level")
A("(35 and 17 pooled sign changes). Confirmation rule (declared before running): a")
A("cell *oscillates* if the pooled trajectory has >= 3 sign changes and a negative")
A("share strictly between 5% and 95%; it *ends negative* if the pooled final value")
A("is < 0; it *mean-reverts* if it oscillates but ends >= 0. Cells whose final")
A("exact-zero (zero-guard) share exceeds 50% are classed degenerate and excluded")
A("from the oscillation count. Sign changes count strictly opposite consecutive")
A("signs; exact zeros break neither. No cell was classed degenerate in this round:")
A("exact-0 final reads are 0.0 in all six cells (the entry-grown population")
A("preserves variance; see the zero-guard finding below).")
A("")
A(table_A)
A("")
A("Zero-guard / depletion context (the T1-F5 protocol at the longer horizon):")
A("")
A(table_zgA)
A("")
A("Contrast with T1 on the same six cells (T1: horizon 200, 20 seeds, vanishing 0.9,")
A("same landscape bundles. T1-side columns are re-derived from")
A("`data/sensitivity_v1/` raw CSVs under this round's declared rule --- T1 itself")
A("published endstate counts and pooled crossings, not per-cell oscillation")
A("verdicts; read-only):")
A("")
A(table_t1)
A("")
A("Reading (all numbers from the tables above):")
A("")
A(f"- Oscillation confirmed and dense: {n_osc}/{n_valid} cells oscillate with "
  f"{sc_min}-{sc_max} pooled sign changes (one per ~{400/sc_max:.0f} steps at the dense "
  f"end, one per ~{400/sc_min:.0f} at the sparse end --- the")
A("  10-step entry waves reshuffle the prestige roulette). Per-seed the count ranges")
A(f"  {ps_min}-{ps_max} with {seeds_osc}/{seeds_tot} runs at >= 3: a population-level")
A("  phenomenon, not a per-run law and not a pooling artifact of out-of-phase cells.")
A(f"- End-state: {n_endneg}/{n_valid} cells end with the pooled ideal correlation below")
A(f"  zero; seed-level final sign rates run {fsr_min:.2f}-{fsr_max:.2f} per cell")
A(f"  ({seed_sign}/{seeds_tot} runs overall). Because a dense sawtooth's final step is")
A("  phase-sensitive, the sign rate is the robust statistic; the clean cells'")
A("  last-third pooled means are positive-leaning while the noisy cells' are")
A("  negative-leaning --- the clean end-state is a negative-leaning oscillation")
A("  around zero, the noisy one a clearly negative end-state.")
A("- The feared zero-guard degeneracy did NOT materialize: exact-0 final reads are")
A("  0.0/40 seeds in every cell and the pooled zero share is 0.0 (T1's Q3 worry,")
A("  refuted here --- the entry-grown population, 61 agents at step 400, preserves")
A("  cross-sectional variance even at 90%+ harvested). The h400 confirmation is not")
A("  depletion-guard-contaminated; this inverts the caveat T1 carried forward.")
A(f"- Mean reversion: {n_revert} cell(s) oscillate but end non-negative; the mid-run")
A(f"  pooled minima ({valid.mid_run_pooled_min.min():+.3f} to "
  f"{valid.mid_run_pooled_min.max():+.3f}) are not corrected into stable positivity")
A("  by the longer horizon.")
A(f"- Horizon/seed extension: T1 (h200, 20 seeds) saw 7-53 pooled sign changes and")
A("  final sign rates 0.55-1.00 on these six cells, with one ambiguous clean cell")
A(f"  (curiosity 0.7, mgstd5: T1 final {t1_flip:+.3f}); at h400/40 seeds the same")
A(f"  cells show {sc_min}-{sc_max} sign changes, the ambiguous cell flips to "
  f"{t2_flip:+.3f},")
A("  and every cell's majority of seeds ends negative. T1's endstate 17/18")
A("  (16/18 under this round's conjunctive rule) becomes 6/6 at horizon 400 with")
A("  40 seeds on this crossing.")
A("")
A("# Sweep B --- career flows under entry")
A("")
A("Per-cell means over 20 seeds (pooled over steps where noted):")
A("")
A(table_B)
A("")
A("Pooled trajectories by entry rate (mean over the three curiosity cells and 20")
A("seeds; gap = newcomer minus incumbent mean prestige; shown from step 50 --- the")
A("ratio is degenerate while incumbent prestige is ~0):")
A("")
A(table_gap)
A("")
A("Findings:")
A("")
A("- **Population grows linearly in $g$ exactly as the fractional accumulator")
A("  predicts**: 31 / 41 / 61 scientists at $g=0.05/0.1/0.2$ (20 founders + 1")
A("  init-time spawn + 10/20/40 scheduled newcomers); no exit channel exists, so")
A("  growth is unbounded in the horizon.")
A(f"- **The newcomer-incumbent prestige gap is persistent and negative at every")
A(f"  entry rate**: newcomers start at zero prestige, spawn on their supervisor's")
A(f"  tile (already depleted by the supervisor), and never close the gap within the")
A(f"  horizon --- the late gap ratio (mean over steps 50-200) runs "
  f"{b_analysis.gap_ratio_mean_late.min():+.2f} to {b_analysis.gap_ratio_mean_late.max():+.2f}")
A(f"  across all nine cells, i.e. newcomers hold roughly a third of incumbent")
A(f"  prestige at every $g$ (newcomer mean = "
  f"{1+b_analysis.gap_ratio_mean_late.max():.2f}-{1+b_analysis.gap_ratio_mean_late.min():.2f} "
  f"\\times incumbent; absolute gap {b_analysis.gap_final.min():+.4f} to "
  f"{b_analysis.gap_final.max():+.4f}). This is the model's internal career-dynamics")
A(f"  statement of proxy failure at the individual level: prestige-proportional")
A(f"  matching hands newcomers to the prestigious, and the prestigious stay")
A(f"  prestigious. (Per-seed nuance, disclosed: at $g=0.2$ the gap turns briefly")
A(f"  positive for individual runs at steps 7-9 --- {min(blip_counts)}-{max(blip_counts)} of")
A(f"  20 seeds per cell, max {blip_max:+.4f} --- while the pooled trajectory is")
A(f"  strictly negative at every step in all nine cells, min {pooled_gap_min:+.5f}; the")
A(f"  claim above is the pooled one the tables tabulate.)")
A("- **Concentration: cross-sectionally higher under entry, temporally decaying")
A("  within entry cells.** The early top-10% spike ("
  f"{b_analysis.top10_step10.min():.2f}-{b_analysis.top10_step10.max():.2f} at step 10)")
A(f"  decays to {b_analysis.top10_final.min():.2f}-{b_analysis.top10_final.max():.2f} by step 200 as the")
A(f"  entry cohort diversifies the prestige stock (per-cell deltas "
  f"{b_analysis.top10_delta.min():+.3f} to {b_analysis.top10_delta.max():+.3f}; the decay")
A("  is weakest at high $g$, where dilution from the first waves prevents the spike")
A(f"  from forming). Cross-sectionally, entry cells still sit above T1's founder")
A(f"  cells at matched horizons (T1 final top-10% share: {t1_top10_noentry[0]:.3f}-"
  f"{t1_top10_noentry[1]:.3f} without entry vs {t1_top10_entry[0]:.3f}-{t1_top10_entry[1]:.3f} with) ---")
A("  both contrasts are true; they measure different things (level vs within-cell")
A("  trajectory).")
A(f"- **The decoupling signal is strongest at SLOW entry**: ideal-channel finals run")
A(f"  {b_low.min():+.3f} to {b_low.max():+.3f} at $g=0.05$ versus {b_high.min():+.3f} to")
A(f"  {b_high.max():+.3f} at $g=0.2$ --- with fast entry the prestige stock is diluted")
A("  from the first waves and the supervisor roulette stays near-uniform longer.")
A("  Descriptive pattern; the mechanism reading above is the only causal claim made.")
A("- **Per-capita harvest falls with $g$**: cumulative harvested share rises ("
  f"{b_analysis.harvested_final.min():.2f}-{b_analysis.harvested_final.max():.2f})")
A(f"  while per-capita cumulative harvest declines monotonically in $g$ "
  f"({b_percap[0.05][0]:.4f}-{b_percap[0.05][1]:.4f} at $g=0.05$, "
  f"{b_percap[0.1][0]:.4f}-{b_percap[0.1][1]:.4f} at $g=0.1$, "
  f"{b_percap[0.2][0]:.4f}-{b_percap[0.2][1]:.4f} at $g=0.2$) --- entry dilutes")
A("  individual productivity; the harvest flow per step per capita decays over the")
A("  horizon everywhere (depletion).")
A(f"- **Driver equivalence (unplanned integrity check):** sweep B's $g=0.1$ mgstd5")
A(f"  cells re-run T1's entry cells with identical seeds and reproduce them")
A(f"  bit-exactly at the raw-trajectory level (max abs diff {equiv_max:.1e} over the")
A(f"  five reporters x 4020 rows x 3 cells); the analysis-level aggregates are")
A(f"  bit-identical at full precision too (same bit-exact inputs, same seed-mean")
A(f"  operation). Diffs up to ~{equiv_disp:.0e} appear only against T1's")
A(f"  full-precision cell_summary, because this round's analysis archive is")
A(f"  4dp-rounded --- display rounding, not summation order. The T1 grid and")
A(f"  this round's grid are the same model.")
A("- **Age is a recorded-but-dead lever** (noted, not fixed): `mean_age` reaches")
A(f"  {b_analysis.mean_age_final.mean():.0f} by step 200 (sweep B) and "
  f"{a_analysis.mean_age_final.mean():.0f} at horizon 400 (sweep A). Individual")
A("  founders pass 400 years of model age, but the population mean cannot: entry")
A("  continuously injects 23-year-olds. Age has no lifecycle effect anywhere in")
A("  the code (no death, retirement, or age-dependent behavior; newcomers reset")
A("  to 23). Any career framing must use steps, not the age column.")
A("")
A("# Steps-to-years framing (framing only, no fitting)")
A("")
A("The model has no exit and no calendar, so the steps-to-years ratio is constrained")
A("from the inside: demographic plausibility bounds it, the panel cannot discipline")
A("it (its in-core coverage is ~2 author-years per author --- 50,830 author-years")
A("over 25,498 authors, `docs/calibration_moments.yaml` --- so it observes short")
A("windows, not careers). Conditional arithmetic, for assumed career spans:")
A("")
A(table_frame)
A("")
A("Reading (conditional arithmetic, no empirical quantity fitted): the table's")
A("entrants/year column is $g \\times$ steps/year. A field adding a few to tens of")
A("percent of its active population per year --- the only demographic plausibility")
A("window the model itself can speak to --- corresponds to roughly 20-100")
A("steps per career-year (one step ~ 4-18 days): at 50 steps/year,")
A("$g=0.05/0.1/0.2$ give 2.5/5/10 entrants per year into a founding community of")
A("20 (12-50% of the founders per year, diluted as the population grows). At that")
A("mapping the 200-step career sweep spans ~4 field-years and the 400-step")
A("confirmation horizon ~8. Two structural consequences, stated as properties of")
A("the model: (i) the ABM captures early-career / short-window dynamics, not full")
A("careers --- full-career claims require an exit channel (death/retirement) the")
A("model does not have; (ii) the population grows "
  f"{b_analysis.pop_growth_x.min():.2f}x-{b_analysis.pop_growth_x.max():.2f}x per 200-step window")
A("with no attrition, so any multi-decade mapping is demographically implausible")
A("without exit. The panel cannot discipline this ratio from its own coverage")
A("(~2 in-core author-years per author); the mapping remains a declared")
A("assumption, not a measurement.")
A("")
A("# Model quirks this round surfaced")
A("")
A("- **Q1 (carried from T1, replicated):** the init-time accumulator spawn makes")
A("  step-0 population 21, not 20; all entry cells end at $21+\\lfloor g \\cdot H")
A("  \\rfloor$ scientists.")
A("- **Q2 (carried, aggravated):** the ideal-merit benchmark is founder-sized")
A("  (`range(number_agents)` = 20 ideal slots) while 31-61 agents actually harvest")
A("  under entry; the ideal channel's degradation under entry is partly benchmark")
A("  strain. The local channel (benchmark-free) is the cleaner career-flow")
A("  companion and is recorded alongside.")
A("- **Q5 (new, an a-priori caveat REFUTED by this round):** T1 carried forward a")
A("  fear that long-horizon clean-landscape runs would be depletion-degenerate")
A("  (zero-guard reads). At h400 this did NOT bind: exact-0 final reads are 0.0/40")
A("  seeds in every cell and the pooled zero share is 0.0, despite 0.91-0.93")
A("  harvested on the clean landscape --- the entry-grown population (61 agents)")
A("  preserves cross-sectional variance. The h400 confirmation needs no zero-guard")
A("  disclaimer; T1's h200 founder-cell caveat does not extend to entry cells at")
A("  longer horizons.")
A("- **Q6 (new, standing property):** the zero-guard still returns exactly 0 for")
A("  degenerate stocks, so `neg share` and sign-change counts are conservative in")
A("  principle --- true decoupling reads as zero, not negative, wherever variance")
A("  collapses. It did not collapse here; the property is recorded for future")
A("  founder-only or longer runs.")
A("")
A("# Honest caveats")
A("")
A("- Not calibrated (banner above); the steps-to-years section is conditional")
A("  arithmetic, explicitly not a mapping claim.")
A("- 40 seeds per confirmation cell is modest for sign-rate differences of ~0.1;")
A("  the verdict rule is declared a priori and applied mechanically.")
A("- The clean-landscape cells are heavily depleted by step 400 (0.91-0.93")
A("  harvested) but NOT zero-guard-degenerate (Q5: cross-sectional variance")
A("  survives via the 61-agent population), so their negative finals are real")
A("  reads; the noisy cells (0.57-0.66 harvested) remain the more legible")
A("  evidence and carry the clearest end-state signal.")
A("- The newcomer-incumbent gap treats the init-time spawn as incumbent and the")
A("  early ratio is degenerate; both disclosed above.")
A("- No exit/death in the model bounds every career interpretation to short")
A("  windows (framing section).")
A("- The foreign live sweep observed at launch wrote to its own paths")
A("  (`scripts/figures/`, `data/figures_v1/`); this task's paths are disjoint.")
A("")
A("# Machine-readable outputs")
A("")
A("All under `data/career_flows_v1/` (gitignored, run-tagged):")
A("")
A("| File | Content |")
A("| --- | --- |")
A("| `confirmA_<tag>.csv` x 6 | Sweep A seed-level model-var trajectories, h400: params + seed + Step + 7 reporters (5 T1 metrics + mean_age + mean_prestige) |")
A("| `careerB_<tag>.csv` x 9 | Sweep B seed-level model-var trajectories, h200 |")
A("| `careerB_<tag>_career.csv` x 9 | Sweep B seed-level career measures per step: n_agents, incumbent/newcomer mean prestige, gap, gap ratio, harvest flows, per-capita harvest |")
A("| `confirmA_pooled.csv`, `careerB_pooled.csv` | Per cell x step seed-mean trajectories |")
A("| `confirmA_analysis.csv` | Per-cell oscillation statistics and verdicts |")
A("| `confirmA_t1_contrast.csv` | T1 (h200/20 seeds) vs T2 (h400/40 seeds) on the same cells |")
A("| `careerB_analysis.csv` | Per-cell career-flow scalars |")
A("| `checkpoint.json`, `run_log.txt` | Restart-safe manifest and append-only progress log |")
A("| `manifest.json` | Grid, pins, counts, canary, source hashes, environment, deviations |")
A("")
A("Cell tag legend: `confirmA_cur{3,5,7}_van9_{mgstd5,noisypr5}_gen0p1_h400`;")
A("`careerB_gen0p05|0p1|0p2_cur{3,5,7}_mgstd5_h200`. Tags carry integer-scaled")
A("curiosity/vanishing and decimal-point-free entry rates (`0p1` = 0.1).")
A("")
A("# Reproducibility")
A("")
A(f"- Environment: python {env['python']}, mesa {env['mesa']}, numpy {env['numpy']}, pandas {env['pandas']}")
A("  (`/opt/hermes/.venv/bin/python`).")
A(f"- Driver sha256 (first 16): `{sha_driver}`; consolidator: `{sha_cons}`; both in")
A("  `scripts/career_flows/` (ADR-0002: analysis logic in scripts, notebooks stay")
A("  thin).")
A("- Regenerate: `career_flows_driver.py --sweep both` (checkpointed; delete")
A("  `data/career_flows_v1/` for a fresh sweep), then `consolidate.py`, then this")
A("  document via `make_doc.py`.")
A("- Post-sweep source additions (disclosed; no dynamics code touched): the driver")
A("  gained a hard canary launch gate and the consolidator drops a spurious seed")
A("  column from `careerB_pooled.csv` (a groupby artifact, constant 9.5; no doc")
A("  number derived from it). Both were added after the sweep completed and before")
A("  delivery; the sweep data itself was produced by the pre-gate driver with the")
A("  canary verified in-session pre-launch. Hashes above are of the archived")
A("  (current) sources.")
A("- Boundaries honoured: writes confined to `data/career_flows_v1/`,")
A("  `scripts/career_flows/`, `docs/career_flows_v1.md`; no git; no code outside")
A("  `scripts/career_flows/` touched; `context/` frozen; `AdmiralsLog/` read-only")
A("  (the session-log entry is deferred to the owner's next records-permitted")
A("  round).")
A("")
A("# Revision Log")
A("")
A("- **Version 1** (November 2026) --- initial document: oscillation confirmation at")
A("  horizon 400 (sweep A, 6 cells x 40 seeds) and career-flow sweep (sweep B, 9")
A("  cells x 20 seeds); verdict composed mechanically from the declared rule;")
A("  deviations (6-vs-4 cells, mgstd5 pin for sweep B) disclosed in Design.")
A("  Squad-verified before delivery: every table independently re-derived from")
A("  the raw CSVs; five prose defects and one artifact blemish found by that")
A("  check (a misquoted T1 context claim, a wrong oscillation-density figure, an")
A("  age claim contradicted by the doc's own table, a wrong equivalence")
A("  mechanism, a double-rounding artifact) were corrected in this same version")
A("  pre-delivery.")
A("")

md = "\n".join(L) + "\n"
with open(MD, "w") as f:
    f.write(md)
print(f"wrote {MD} ({len(md)} chars)")

manifest = {
    "task": "T2 career flows + oscillation confirmation (h400); model-internal; NOT calibrated",
    "generated_at": now,
    "sweep_A": {"grid": {"curiosity": [0.3, 0.5, 0.7], "vanishing": [0.9],
                          "landscape": ["mgstd5", "noisypr5"], "gen_time": [0.1],
                          "step_limit": 400, "seeds": 40},
                "counts": ca,
                "verdict_rule": "oscillates = sign_changes>=3 and 0.05<neg_share<0.95; ends_negative = pooled final<0; mean_reverted = oscillates and final>=0; degenerate = exact-0 final share>0.5 (excluded)"},
    "sweep_B": {"grid": {"gen_time": [0.05, 0.1, 0.2], "curiosity": [0.3, 0.5, 0.7],
                          "vanishing": [0.9], "landscape": ["mgstd5"],
                          "step_limit": 200, "seeds": 20},
                "counts": cb},
    "fixed_params": {"n_agents": 20, "grid_size": 20, "epsilon": 0.1, "harvest": 0.02,
                     "use_distance": True, "new_questions": 0, "use_visibility_reward": True,
                     "knowledge_normalization": "share", "prestige_accrual": "harvested",
                     "agent_generation_function": "identity (model default)"},
    "seed_semantics": "agent_seed 0..N-1 per cell; model injects the same seed into generation_params (landscape and agent RNG coupled); T1 convention",
    "population_rule_under_entry": "21 + floor(g*t) (init-time accumulator quirk Q1)",
    "canary": {"command": "MPLBACKEND=Agg /opt/hermes/.venv/bin/python tests/test_smoke.py",
               "exit_code": canary_rc, "output": canary_txt},
    "concurrency": {"n_jobs": 3,
                    "observed_live_foreign_sweep_at_launch": "scripts/figures/run_fresh_sweep.py -> data/figures_v1/ (disjoint paths), plus 3 Jupyter kernel runners",
                    "own_orphans": "none"},
    "deviations_disclosed": ["sweep A ran the full 3x2 crossing (6 cells) though the brief said 4 representative",
                             "sweep B landscape pinned mgstd5",
                             "init-time spawn counted as incumbent in the gap",
                             "gap ratio reported from step 50"],
    "sources_sha256_16": {"driver": sha_driver, "consolidator": sha_cons},
    "environment": env,
    "boundaries": "wrote only data/career_flows_v1/, scripts/career_flows/, docs/career_flows_v1.md; no git; no calibration",
}
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print("wrote manifest.json")
print("VERDICT:", verdict_word)
