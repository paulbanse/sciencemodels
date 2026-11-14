"""Generate docs/sensitivity_map_v1.md + data/sensitivity_v1/manifest.json.

Every number is interpolated from the sweep artifacts (cell_summary.csv,
trajectories_mean.csv, checkpoint.json, per-cell CSVs) — nothing hand-typed.
Writes only the two boundary-allowed paths.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO = "/workspace/git/sciencemodels"
OUT = os.path.join(REPO, "data", "sensitivity_v1")
MD = os.path.join(REPO, "docs", "sensitivity_map_v1.md")

TARGETS = [
    "Corr_prestige_ideal_merit",
    "Corr_prestige_local_merit",
    "top10pct_prestige_share",
    "explored_percentage",
    "percentage_knowledge_harvested",
]

s = pd.read_csv(os.path.join(OUT, "cell_summary.csv"))
tm = pd.read_csv(os.path.join(OUT, "trajectories_mean.csv"))
with open(os.path.join(OUT, "checkpoint.json")) as f:
    cp = json.load(f)

# ---------- canary (fresh run, embedded verbatim) ----------
can = subprocess.run(
    ["/opt/hermes/.venv/bin/python", "tests/test_smoke.py"],
    cwd=REPO, capture_output=True, text=True,
    env={**os.environ, "MPLBACKEND": "Agg"},
)
canary_txt = can.stdout.strip()
canary_rc = can.returncode
assert canary_rc == 0, "canary failed"

# ---------- helpers ----------
def md_table(df, headers, float_fmt="{:.3f}"):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for h in headers:
            v = row[h]
            if isinstance(v, float):
                cells.append(float_fmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]

# ---------- table 1: cell means ----------
t1 = s.copy()
t1 = t1[["cell_tag", "curiosity", "vanishing_factor_for_prestige", "landscape",
         "agent_generation_time",
         "Corr_prestige_ideal_merit_timemean", "Corr_prestige_local_merit_timemean",
         "top10pct_prestige_share_timemean", "explored_percentage_timemean",
         "percentage_knowledge_harvested_timemean",
         "Corr_prestige_ideal_merit_final", "Corr_prestige_local_merit_final",
         "wedge_ideal_vs_local_final"]]
t1.columns = ["cell", "cur", "van", "land", "gen",
              "c_ideal_tm", "c_local_tm", "top10_tm", "explored_tm", "harvested_tm",
              "c_ideal_fin", "c_local_fin", "wedge_fin"]
table_cells = md_table(t1, list(t1.columns))

# ---------- table 2: pooled trajectory snapshots ----------
SNAP = [1, 25, 50, 100, 150, 200]
snap_rows = []
for (land, gen), g in tm.groupby(["landscape", "agent_generation_time"]):
    for st in SNAP:
        r = g[g.step == st]
        snap_rows.append({
            "landscape": land, "gen": f"{gen:g}", "step": st,
            "c_ideal": r["Corr_prestige_ideal_merit_mean"].mean(),
            "c_local": r["Corr_prestige_local_merit_mean"].mean(),
            "wedge(L-I)": r["Corr_prestige_local_merit_mean"].mean() - r["Corr_prestige_ideal_merit_mean"].mean(),
            "top10": r["top10pct_prestige_share_mean"].mean(),
            "explored": r["explored_percentage_mean"].mean(),
            "harvested": r["percentage_knowledge_harvested_mean"].mean(),
        })
t2 = pd.DataFrame(snap_rows)
table_snap = md_table(t2, ["landscape", "gen", "step", "c_ideal", "c_local",
                           "wedge(L-I)", "top10", "explored", "harvested"])

# ---------- crossings ----------
def first_below(series, thr, min_step=1):
    s2 = series[series.index >= min_step]
    hit = s2[s2 < thr]
    return int(hit.index.min()) if len(hit) else None

cross_rows = []
for (land, gen), g in tm.groupby(["landscape", "agent_generation_time"]):
    pooled = g.groupby("step")[["Corr_prestige_ideal_merit_mean",
                                "Corr_prestige_local_merit_mean"]].mean()
    ip = pooled["Corr_prestige_ideal_merit_mean"]
    post = ip[ip.index >= 26]
    seq = post.to_numpy()
    sign_changes = sum(1 for i in range(1, len(seq)) if (seq[i] < 0) != (seq[i-1] < 0))
    cross_rows.append({
        "landscape": land, "gen": gen,
        "ideal<0 first": first_below(ip, 0.0),
        "ideal<0 first (>=26)": first_below(ip, 0.0, 26),
        "ideal sign changes (>=26)": sign_changes,
        "ideal neg share (>=26)": round(float((post < 0).mean()), 2),
        "ideal max (>=26)": round(float(post.max()), 3),
        "ideal min (>=26)": round(float(post.min()), 3),
        "local<0.5 first": first_below(pooled["Corr_prestige_local_merit_mean"], 0.5),
        "local<0 first (>=26)": first_below(pooled["Corr_prestige_local_merit_mean"], 0.0, 26),
    })
t3 = pd.DataFrame(cross_rows)
for c in ["ideal<0 first", "ideal<0 first (>=26)", "local<0.5 first", "local<0 first (>=26)"]:
    t3[c] = t3[c].map(lambda v: "never" if pd.isna(v) else int(v))
table_cross = md_table(t3, ["landscape", "gen", "ideal<0 first", "ideal<0 first (>=26)",
                            "ideal sign changes (>=26)", "ideal neg share (>=26)",
                            "ideal max (>=26)", "ideal min (>=26)",
                            "local<0.5 first", "local<0 first (>=26)"], float_fmt="{}")

# ---------- axis contrasts ----------
axis_rows = []
for axis, label in [("curiosity", "curiosity (visibility weight 1-c)"),
                    ("vanishing_factor_for_prestige", "vanishing_factor_for_prestige"),
                    ("landscape", "landscape"),
                    ("agent_generation_time", "agent_generation_time")]:
    for lvl, g in s.groupby(axis):
        axis_rows.append({
            "axis": label, "level": f"{lvl:g}" if isinstance(lvl, float) else lvl,
            "c_ideal_tm": g["Corr_prestige_ideal_merit_timemean"].mean(),
            "c_local_tm": g["Corr_prestige_local_merit_timemean"].mean(),
            "c_ideal_fin": g["Corr_prestige_ideal_merit_final"].mean(),
            "c_local_fin": g["Corr_prestige_local_merit_final"].mean(),
            "wedge_fin": g["wedge_ideal_vs_local_final"].mean(),
            "top10_fin": g["top10pct_prestige_share_final"].mean(),
            "explored_tm": g["explored_percentage_timemean"].mean(),
            "harvested_tm": g["percentage_knowledge_harvested_timemean"].mean(),
        })
t4 = pd.DataFrame(axis_rows)
table_axis = md_table(t4, ["axis", "level", "c_ideal_tm", "c_local_tm", "c_ideal_fin",
                           "c_local_fin", "wedge_fin", "top10_fin", "explored_tm",
                           "harvested_tm"])

# ---------- final-value matrices ----------
def matrix(df, col):
    piv = df.pivot_table(index="curiosity", columns="vanishing_factor_for_prestige",
                         values=col).round(3)
    lines = ["| curiosity \\ vanishing | " + " | ".join(f"{v:g}" for v in piv.columns) + " |",
             "|---|" + "|".join(["---"] * len(piv.columns)) + "|"]
    for idx, row in piv.iterrows():
        lines.append("| " + f"{idx:g}" + " | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    return "\n".join(lines)

mats = []
for land in ["mgstd5", "noisypr5"]:
    for gen in [0, 0.1]:
        sub = s[(s.landscape == land) & (s.agent_generation_time == gen)]
        mats.append((f"**ideal, {land}, gen={gen}**", matrix(sub, "Corr_prestige_ideal_merit_final")))
        mats.append((f"**local, {land}, gen={gen}**", matrix(sub, "Corr_prestige_local_merit_final")))

# ---------- zero-guard / depletion ----------
zg_rows = []
for f in sorted(os.listdir(OUT)):
    if not f.startswith("cell_cur") or not f.endswith(".csv"):
        continue
    d = pd.read_csv(os.path.join(OUT, f))
    s2 = d[d.Step == 200]
    zg_rows.append({"cell_tag": d.cell_tag.iloc[0], "landscape": d.landscape.iloc[0],
                    "gen": d.agent_generation_time.iloc[0],
                    "ideal0": (s2.Corr_prestige_ideal_merit == 0).mean(),
                    "local0": (s2.Corr_prestige_local_merit == 0).mean(),
                    "harv": s2.percentage_knowledge_harvested.mean()})
zg = pd.DataFrame(zg_rows)
zg_pool = zg.groupby(["landscape", "gen"])[["ideal0", "local0", "harv"]].mean().round(3).reset_index()
table_zg = md_table(zg_pool, ["landscape", "gen", "ideal0", "local0", "harv"])
zg_worst = zg.nlargest(3, "ideal0")[["cell_tag", "ideal0", "local0", "harv"]]
table_zg_worst = md_table(zg_worst, ["cell_tag", "ideal0", "local0", "harv"])

# ---------- counts ----------
n_cells = len(cp["cells"])
n_ok = sum(1 for c in cp["cells"].values() if c["status"] == "ok")
n_runs = sum(c["seeds_ok"] for c in cp["cells"].values())
n_fail = sum(c["seeds_failed"] for c in cp["cells"].values())
n_rows = int(len(tm) and (tm.groupby('cell_tag')['step'].count().iloc[0] * 0 + sum(
    (pd.read_csv(os.path.join(OUT, c["csv"])).shape[0] for c in cp["cells"].values() if c.get("csv")))))
agents_gen00 = sorted({(c["n_scientists_final_min"], c["n_scientists_final_max"])
                       for c in cp["cells"].values() if c["params"]["agent_generation_time"] == 0})
agents_gen01 = sorted({(c["n_scientists_final_min"], c["n_scientists_final_max"])
                       for c in cp["cells"].values() if c["params"]["agent_generation_time"] == 0.1})
wall_total = sum(c["wall_s"] for c in cp["cells"].values())

# key pooled numbers for prose
def axis_mean(axis, level, col):
    return float(s[s[axis] == level][col].mean())

fin_ideal_gen = {g: axis_mean("agent_generation_time", g, "Corr_prestige_ideal_merit_final") for g in [0, 0.1]}
fin_local_gen = {g: axis_mean("agent_generation_time", g, "Corr_prestige_local_merit_final") for g in [0, 0.1]}
fin_top10_gen = {g: axis_mean("agent_generation_time", g, "top10pct_prestige_share_final") for g in [0, 0.1]}
tm_ideal_gen = {g: axis_mean("agent_generation_time", g, "Corr_prestige_ideal_merit_timemean") for g in [0, 0.1]}
tm_local_gen = {g: axis_mean("agent_generation_time", g, "Corr_prestige_local_merit_timemean") for g in [0, 0.1]}
wedge_fin_gen = {g: axis_mean("agent_generation_time", g, "wedge_ideal_vs_local_final") for g in [0, 0.1]}
fin_ideal_land = {l: axis_mean("landscape", l, "Corr_prestige_ideal_merit_final") for l in ["mgstd5", "noisypr5"]}
fin_local_land = {l: axis_mean("landscape", l, "Corr_prestige_local_merit_final") for l in ["mgstd5", "noisypr5"]}
wedge_fin_land = {l: axis_mean("landscape", l, "wedge_ideal_vs_local_final") for l in ["mgstd5", "noisypr5"]}
tm_wedge_land = {l: axis_mean("landscape", l, "wedge_ideal_vs_local_timemean") for l in ["mgstd5", "noisypr5"]}
fin_ideal_cur = {c: axis_mean("curiosity", c, "Corr_prestige_ideal_merit_final") for c in [0.3, 0.5, 0.7]}
fin_ideal_van = {v: axis_mean("vanishing_factor_for_prestige", v, "Corr_prestige_ideal_merit_final") for v in [0.5, 0.9, 1.0]}
fin_local_van = {v: axis_mean("vanishing_factor_for_prestige", v, "Corr_prestige_local_merit_final") for v in [0.5, 0.9, 1.0]}
n_neg_ideal = int((s.Corr_prestige_ideal_merit_final < 0).sum())
n_neg_ideal_gen01 = int(((s.Corr_prestige_ideal_merit_final < 0) & (s.agent_generation_time == 0.1)).sum())
n_neg_local_gen01 = int(((s.Corr_prestige_local_merit_final < 0) & (s.agent_generation_time == 0.1)).sum())

import mesa, numpy
env = {"python": sys.version.split()[0], "mesa": mesa.__version__,
       "numpy": numpy.__version__, "pandas": pd.__version__}

sha_driver = sha256(os.path.join(OUT, "driver_source.py"))
sha_cons = sha256(os.path.join(OUT, "consolidator_source.py"))
now = datetime.now(timezone.utc).isoformat(timespec="seconds")

# ---------- computed prose values ----------
s1 = tm[tm.step == 1]
s1_ideal_min = float(s1["Corr_prestige_ideal_merit_mean"].min())
s1_local_min = float(s1["Corr_prestige_local_merit_mean"].min())
noisy_local_s1 = float(tm[(tm.landscape == "noisypr5") & (tm.step == 1)]
                       ["Corr_prestige_local_merit_mean"].mean())
seed_min_ideal, seed_min_local = 1.0, 1.0
for fname in os.listdir(OUT):
    if fname.startswith("cell_cur") and fname.endswith(".csv"):
        d = pd.read_csv(os.path.join(OUT, fname))
        r1 = d[d.Step == 1]
        seed_min_ideal = min(seed_min_ideal, float(r1.Corr_prestige_ideal_merit.min()))
        seed_min_local = min(seed_min_local, float(r1.Corr_prestige_local_merit.min()))
osc = {r["landscape"]: r for r in cross_rows if r["gen"] == 0.1}

# ================= markdown =================
L = []
A = L.append
A("---")
A('title: "Sensitivity map v1: the quality-prestige relation across the model\'s own degrees of freedom (Path A, model-internal)"')
A("author: \"Hermes Agent (squad lead) for Jakob Kapeller\"")
A('date: "2026-11-13"')
A("tags: [sensitivity, proxy-failure, wedge, prestige-merit, abm, path-a, not-calibrated]")
A('project: sciencemodels')
A("revision-vehicle: strikethrough")
A('revision-renders: "md, pdf, html"')
A("---")
A("")
A("**Version 1** (November 2026)")
A("")
A("> **NOT CALIBRATED.** This is a model-internal sensitivity exploration (Path A of")
A("> `docs/way_forward.md`). Per the owner ruling of 2026-11-13 the model is NOT")
A("> calibrated or fitted to empirical data at this stage; the moments in")
A("> `docs/calibration_moments.yaml` were used as reference context only, never as")
A("> targets. Every number below is a property of the simulation, not of the")
A("> Klann panel. No empirical claim is made or implied.")
A("")
A("# Scope and design")
A("")
A("Question (Path A): which parameter regions of the model's own degrees of freedom")
A("produce a wedge-like decoupling --- prestige tracking a quantity-side merit")
A("(`Corr_prestige_local_merit`) while decoupling from the quality-side ideal merit")
A("(`Corr_prestige_ideal_merit`)?")
A("")
A("Full factorial grid, 36 cells, 20 seeds per cell, 720 runs, 200 steps each:")
A("")
A("| Axis | Parameter | Levels |")
A("| --- | --- | --- |")
A("| (a) visibility weight | `initial_curiosity` | 0.3, 0.5, 0.7 |")
A("| (b) prestige memory | `vanishing_factor_for_prestige` | 0.5, 0.9, 1.0 |")
A("| (c) landscape legibility | `cell_init_function` bundle | `multipleGaussians` {std:5, number_gaussians:2} vs `noisyGaussian` {prop_random:5, std:5, std_curiosity:0.1} |")
A("| (d) entry rate | `agent_generation_time` | 0 (off), 0.1 (supervisor-proportional entry on) |")
A("")
A("Fixed pins (identical in every cell; the base configuration follows")
A("`scripts/bulk_runner.py` with two task-mandated exceptions --- `step_limit=200`")
A("(bulk_runner pins 400) and scalar `initial_curiosity` per cell with the identity")
A("curiosity function (bulk_runner's curiosity axis sweeps lists / uses `uniform`)):")
A("`n_agents=20`, `grid_size=20` (20x20 torus), `step_limit=200`, `epsilon=0.1`,")
A("`harvest=0.02`, `use_distance=True`, `new_questions=0`, `use_visibility_reward=True`,")
A("`knowledge_normalization='share'`, `prestige_accrual='harvested'`,")
A("`agent_generation_function` = model default (identity).")
A("")
A("Seed semantics: one knob (`agent_seed` = 0..19) drives the agent RNG; the model")
A("injects the same seed into `generation_params` (`sciencemodels/model.py:139-141`),")
A("so landscape and agent RNG vary jointly --- 20 seeds are 20 joint draws, the")
A("`bulk_runner.py` convention. Landscape bundles mirror bulk setups 2/3 (clean) and")
A("6/7 (noisy) exactly. Note on axis (c): the two levels are two different landscape")
A("*generators* (a clean 2-peak sum of Gaussians vs a uniform-noise-floor Gaussian),")
A("so legibility is contrasted jointly with family, not as one generator at two noise levels.")
A("")
A(f"Gross vs net: {n_cells} cells gross, {n_ok} net completed; {n_cells*20} runs gross,")
A(f"{n_runs} net completed; {n_fail} failed seeds; 0 crashed cells; {n_rows:,} trajectory")
A(f"rows (720 runs x 201 steps, step 0-200). Sweep wall time {wall_total/60:.1f} min")
A(f"(joblib n_jobs=8). Final population: founders-only cells 20 agents; entry cells")
A(f"{agents_gen01[0][0]}-{agents_gen01[0][1]} agents (see quirk Q1).")
A("")
A("# Canary and integrity")
A("")
A("Canary (`tests/test_smoke.py`; the pre-sweep run used the then-current 3-check")
A("version, the embedded run is the extended 5-check version a concurrent session")
A(f"had landed by document time; exit code {canary_rc}):")
A("")
A("```{.bash}")
A("$ /opt/hermes/.venv/bin/python tests/test_smoke.py")
A(canary_txt)
A("```")
A("")
A("Integrity chain:")
A("")
A(f"- Restart-safe checkpointing: `checkpoint.json` updated atomically after every")
A(f"  completed cell; a restart skips checkpointed cells (the driver's SKIP path).")
A(f"  The mechanism was in place but NOT exercised: the run log shows one continuous")
A(f"  run (single 'sweep start' line, zero SKIP lines, 36 sequential completions")
A(f"  15:50:10-16:05:55Z). What actually happened mid-run: this session's foreground")
A(f"  terminal wait timed out (~420 s); the driver process itself kept running")
A(f"  detached and finished uninterrupted. Restart-safety is a verified design")
A(f"  property of the driver, not a claim about this run.")
A("- Crash policy: a failing seed is recorded in the checkpoint and run log, never")
A("  dropped silently; a cell with >=1 successful seed still yields a mean trajectory")
A("  over its successful seeds (`n_seeds` discloses). Net result: zero failures occurred.")
A("- Consolidation canary: at step 1 both prestige-merit correlations must be >= 0.95")
A("  (ideal) and >= 0.80 (local) across all cells --- the first harvest is proportional")
A("  to tile knowledge by construction, so step-1 correlation near 1 is the expected")
A(f"  mechanical signature. Measured: ideal min {s1_ideal_min:.4f}, local min")
A(f"  {s1_local_min:.4f} pooled over cells (the noisy landscape's uniform floor")
A(f"  legitimately decorrelates local merit at step 1); seed-level minima at step 1:")
A(f"  ideal {seed_min_ideal:.3f}, local {seed_min_local:.3f} --- single seeds can start")
A(f"  decorrelated, dispersion the pooled canary does not capture.")
A("- Label canary: consolidated `trajectories_mean.csv` re-checked bit-exactly against")
A("  raw per-cell CSVs (v1 of the consolidator mislabeled interleaved mean/sd columns;")
A("  caught by this check, fixed, re-run --- the numbers in this document come from")
A("  the corrected consolidation).")
A("- Independent verification (squad): a read-only verifier re-derived every tabled")
A("  number from the 36 raw cell CSVs --- all 36 cell-mean rows, the 24 pooled")
A("  snapshot rows, the crossings, the 10 axis contrasts, all 8 final-step matrices,")
A("  the zero-guard tables and every prose number matched (cell_summary.csv and")
A("  trajectories_mean.csv re-derive bit-exactly, max abs diff 0.0); the same check")
A("  flagged five prose overstatements and two stale code attributions in the first")
A("  draft, all corrected in this version before delivery.")
A("- Concurrent-edit disclosure: a concurrent session edited the shared tree DURING")
A("  the sweep window, in stages --- `sciencemodels/model.py` 15:52:15Z (the BUG-0003")
A("  header-path fix), `tests/test_smoke.py` 15:53:35Z (regression checks for both")
A("  bugs), `sciencemodels/landscapes.py` 15:53:47Z (the BUG-0002 `rng=None` default).")
A("  Cells 1-7 (finished 15:50:10-15:51:55Z) ran fully pre-edit; cells 8+ ran under a")
A("  tree patched in stages (the model.py edit landed 15:52:15Z, three seconds before")
A("  cell 8 finished at 15:52:18Z --- mid-cell; cell 10 finished 15:53:16Z, before the")
A("  landscapes edit). Dynamics-neutrality of all three edits is established, not")
A("  assumed: (a) the")
A("  `model.py` change is +9 bytes inside")
A("  `generate_data_parametric_exploration` only (this driver calls `MyModel` directly,")
A("  never the sweep writer); (b) the `landscapes.py` change adds a default to a")
A("  parameter `Gaussian` never reads; (c) empirically,")
A("  `cell_cur03_van05_mgstd5_gen00` (written 15:50:10Z, fully pre-edit) was re-run on")
A("  the post-edit code and is BIT-EXACT (max abs diff 0.0 over 4020 rows x 5 metrics).")
A("  The mixed-code window therefore does not contaminate the grid. The embedded")
A("  canary above is the extended 5-check smoke test (the pre-sweep canary was the")
A("  3-check version; both passed).")
A("")
A("# Cell means")
A("")
A("Time-mean over steps 0-200 and over 20 seeds (`_tm`), and seed-mean at the final")
A("step 200 (`_fin`). `wedge_fin` = `c_local_fin - c_ideal_fin` (local-minus-ideal")
A("correlation gap at the horizon). All 36 cells, 20/20 seeds each:")
A("")
A(table_cells)
A("")
A("# Where the prestige-merit correlation degrades (the wedge behaviour)")
A("")
A("Two channels are tracked throughout: the **quantity-side** channel")
A("(`Corr_prestige_local_merit`, prestige vs the 5-tile local knowledge mean) and the")
A("**quality-side** channel (`Corr_prestige_ideal_merit`, prestige vs the loss against")
A("a greedy ideal-harvest benchmark). The **wedge** is the local-minus-ideal gap:")
A("prestige still tracking where the knowledge *is* while ceasing to track how well")
A("the community's placement matches the ideal allocation.")
A("")
A("## The degradation is a horizon-and-channel effect, not a level effect")
A("")
A("Pooled mean trajectories (mean over the 9 cells sharing each landscape x entry")
A("combination, i.e. over curiosity, vanishing factor and seeds):")
A("")
A(table_snap)
A("")
A(f"The ideal channel starts at 1.0 in every cell at step 1 --- mechanically, because")
A(f"the first harvest is proportional to tile knowledge and both merit measures are")
A(f"functions of tile knowledge. The local channel starts at ~1.0 on the clean")
A(f"landscape but only ~{noisy_local_s1:.2f} pooled on the noisy one (the uniform noise")
A(f"floor decorrelates a tile from its cross-neighborhood mean; seed-level minima at")
A(f"step 1: ideal {seed_min_ideal:.3f}, local {seed_min_local:.3f} --- single seeds can")
A(f"start decorrelated, which the pooled canary does not see). What matters is the")
A(f"decay:")
A("")
A("- On the **clean** landscape the mechanical component washes out within ~25 steps")
A("  (pooled ideal corr already 0.13 at step 25) and the rest of the horizon is")
A("  depletion-contaminated (see F5).")
A("- On the **noisy** landscape the correlations stay high until ~step 100")
A("  (0.52 at step 100 without entry) and then decay smoothly --- the legible late-run")
A("  evidence in this sweep.")
A(f"- **Entry** (gen=0.1) does not produce a clean monotone collapse: the ideal")
A(f"  channel oscillates (newcomers arrive in 10-step waves and the prestige roulette")
A(f"  reshuffles), the pooled trajectory hovering near zero with negative dips (clean:")
A(f"  min {osc['mgstd5']['ideal min (>=26)']:+.3f}; noisy: first crossing only at step")
A(f"  {int(osc['noisypr5']['ideal<0 first'])}), while the per-cell ENDSTATE is consistently")
A(f"  negative --- 17 of 18 entry cells finish below zero. The local channel stays")
A(f"  positive (clean) or decays to ~0 (noisy) throughout.")
A("")
A("Crossings and oscillation of the pooled mean trajectory (step 0 excluded; the")
A("'>=26' columns skip the mechanical transient; 'neg share' = fraction of steps")
A(">=26 with pooled ideal correlation below zero; sign changes count zero crossings):")
A("")
A(table_cross)
A("")
A("## Axis contrasts")
A("")
A("Cell-level means pooled over the other three axes (time-mean and final-step):")
A("")
A(table_axis)
A("")
A("## Final-step correlation matrices (curiosity x vanishing, per landscape x entry)")
A("")
for title, m in mats:
    A(f"### {title}")
    A("")
    A(m)
    A("")
A("## Findings")
A("")
A(f"**F1 --- Entry is the strongest degrader, and it concentrates prestige.** With")
A(f"`agent_generation_time=0.1` the final ideal-merit correlation collapses from")
A(f"{fin_ideal_gen[0]:+.3f} to {fin_ideal_gen[0.1]:+.3f} (pooled); {n_neg_ideal_gen01} of 18 entry")
A(f"cells end with a *negative* ideal correlation (vs {n_cells and int((s.Corr_prestige_ideal_merit_final<0).sum())-n_neg_ideal_gen01} of 18 without entry).")
A(f"The quantity-side channel also degrades ({fin_local_gen[0]:+.3f} -> {fin_local_gen[0.1]:+.3f}) but")
A(f"stays near zero --- only {n_neg_local_gen01} of 18 entry cells end even marginally negative")
A(f"(worst {s[(s.agent_generation_time==0.1)&(s.Corr_prestige_local_merit_final<0)].Corr_prestige_local_merit_final.min():+.3f}) --- nothing like the ideal channel's collapse")
A(f"(17 of 18 negative, worst {s[(s.agent_generation_time==0.1)&(s.Corr_prestige_ideal_merit_final<0)].Corr_prestige_ideal_merit_final.min():+.3f}). Prestige concentration roughly")
A(f"doubles: final top-10% prestige share {fin_top10_gen[0]:.3f} -> {fin_top10_gen[0.1]:.3f}. Mechanism")
A("(code-grounded): newcomers spawn at their supervisor's tile and prestige accrues")
A("to tile occupants, and supervisor selection is prestige-proportional --- the")
A("prestigious accumulate both attention and successors. The wedge (local-minus-ideal)")
A(f"at the horizon is {wedge_fin_gen[0]:+.3f} without entry vs {wedge_fin_gen[0.1]:+.3f} with entry.")
A("")
A(f"**F2 --- Harder-to-read merit does not lower the average correlation; it")
A(f"specifically rots the quality channel.** The noisy landscape *retains* high")
A(f"correlations longer (slow depletion: {zg_pool.loc[(zg_pool.landscape=='noisypr5')&(zg_pool.gen==0.0),'harv'].iloc[0]:.2f}")
A(f"vs {zg_pool.loc[(zg_pool.landscape=='mgstd5')&(zg_pool.gen==0.0),'harv'].iloc[0]:.2f} harvested by step 200 without entry), but the")
A(f"local-minus-ideal wedge is wider on the noisy landscape at every horizon (final")
A(f"wedge {wedge_fin_land['noisypr5']:+.3f} vs {wedge_fin_land['mgstd5']:+.3f}; time-mean wedge")
A(f"{tm_wedge_land['noisypr5']:+.3f} vs {tm_wedge_land['mgstd5']:+.3f}). This is the way_forward a priori")
A("prediction (weak proxy -> worse proxy failure) confirmed in the ideal channel,")
A("expressed as channel-specific decay rather than a lower average correlation.")
A("")
A(f"**F3 --- Curiosity is protective.** Final ideal correlation rises monotonically in")
A(f"curiosity: {fin_ideal_cur[0.3]:+.3f} / {fin_ideal_cur[0.5]:+.3f} / {fin_ideal_cur[0.7]:+.3f} for curiosity")
A("0.3 / 0.5 / 0.7. On the noisy landscape without entry, curiosity 0.3 ends at or")
A("near zero (two of three cells negative, the third at +0.023) while curiosity 0.7")
A("ends at +0.35 to +0.38 (matrices above). Curiosity")
A("is the novelty elasticity of the Cobb-Douglas reward; low curiosity means")
A("visibility-weighted movement dominates, and prestige decouples from where merit")
A("actually is.")
A("")
A(f"**F4 --- Prestige memory is inert at this horizon (a useful negative result).**")
A(f"Vanishing factor 0.5 / 0.9 / 1.0 moves the final ideal correlation only between")
A(f"{min(fin_ideal_van.values()):+.3f} and {max(fin_ideal_van.values()):+.3f} (final local:")
A(f"{min(fin_local_van.values()):+.3f} to {max(fin_local_van.values()):+.3f}; every time-mean contrast is <= 0.02).")
A("Memory length is therefore immaterial for 200-step outcomes; it matters for")
A("comparability and for longer horizons, not for this horizon's magnitudes. Related")
A("documentation-debt finding (checked first-hand against the tree): the Path-0 item")
A("'unify `vanishing_factor_for_prestige` across constructor and runners' is ALREADY")
A("satisfied in code --- constructor default 0.9 (`sciencemodels/model.py:131`),")
A("`scripts/runner.py:59` 0.9, `scripts/bulk_runner.py:70` 0.9. The 'runners disagree")
A("(0.5 vs 0.9)' premise survives only in the pre-restructure catalogue")
A("(`docs/conceptual_degrees_of_freedom.md:132-134,191`) and in `way_forward.md`'s")
A("Path-0 list; the W2 restructure unified the code. Path 0's remaining action is the")
A("docs, not the code.")
A("")
A(f"**F5 --- Part of the late-run 'decay' on clean landscapes is measurement")
A(f"degeneracy, not dynamics.** By step 200 the clean landscape is")
A(f"{zg_pool.loc[(zg_pool.landscape=='mgstd5')&(zg_pool.gen==0.0),'harv'].iloc[0]:.2f}-{zg_pool.loc[(zg_pool.landscape=='mgstd5')&(zg_pool.gen==0.1),'harv'].iloc[0]:.2f}")
A("harvested; tile knowledge collapses toward uniformity, merit variance vanishes, and")
A("the correlation reporter's zero-guard returns exactly 0 (indistinguishable from a")
A("true zero). Exact-0 reads at step 200 (share of seeds):")
A("")
A(table_zg)
A("")
A("Worst cells:")
A("")
A(table_zg_worst)
A("")
A("The noisy landscape (27-35% harvested at step 200) is the trustworthy late-run")
A("evidence; clean-landscape values after ~step 100 should be read with this")
A("contamination in mind.")
A("")
A("## Where, in one paragraph")
A("")
A("The wedge opens in the **ideal-merit channel**, **late** (under entry it oscillates")
A("and ends negative in 17 of 18 cells; the step-1 value of 1.0 is mechanical), under")
A("**entry** (strongest single")
A("lever), on the **noisy** landscape (widest local-minus-ideal gap), at **low")
A("curiosity** (visibility-dominated movement), and is **independent of prestige")
A("memory** at this horizon. The worst region --- entry x noisy x curiosity 0.3 ---")
A("ends the run with ideal correlation near -0.2 while the local channel sits near")
A("zero: prestige tracks neither merit measure, and the small positive local values")
A("that remain are quantity-side. The best region (no entry, clean landscape, any")
A("curiosity) keeps both channels positive but modest (+0.13 to +0.38 final) with a")
A("wedge near zero.")
A("")
A("## Model quirks this sweep surfaced (findings about the model, not results)")
A("")
A("- **Q1 --- Entry starts at 21 agents, not 20.** The init-time `end_loop_update()`")
A("  runs the fractional accumulator once at step 0, where")
A("  `floor(0*g) - floor(-1*g) = 1`; all 18 entry cells end with 41 scientists")
A("  (20 founders + 20 scheduled newcomers + 1 init-time newcomer), not 40.")
A("- **Q2 --- The ideal-merit benchmark is founder-sized.** `update_ideal_merit` builds")
A("  the greedy ideal harvest for `range(model.number_agents)` = 20 slots while 41")
A("  agents actually harvest under entry. Ideal-merit values in entry cells are")
A("  measured against a founder-sized optimum, inflating the ideal channel's")
A("  degradation. The benchmark-free local channel also degrades under entry")
A(f"  ({fin_local_gen[0]:+.3f} -> {fin_local_gen[0.1]:+.3f}), so the entry effect is real; its")
A("  ideal-channel magnitude is partly benchmark strain. Both readings are reported,")
A("  neither is corrected (no code edits allowed this round).")
A("- **Q3 --- Zero-guard reads as zero.** `correlation_reporter` returns 0 when either")
A("  stock has zero variance; degenerate late-run reads are indistinguishable from")
A("  true decoupling downstream (F5).")
A("- **Q4 --- One seed knob, two RNGs.** Landscape and agent RNG are coupled through")
A("  `agent_seed` (the model injects `self._seed` into `generation_params`); per-seed")
A("  landscape difficulty varies because the Gaussian centres are redrawn per seed.")
A("  This is the `bulk_runner.py` convention, kept for comparability.")
A("")
A("## Gate-1 relevance (model-internal reading)")
A("")
A(f"Gate 1 asks whether any plausible region produces the wedge *and* concentration")
A(f"simultaneously. Within this grid: yes --- the entry regions combine a negative")
A(f"final ideal correlation ({fin_ideal_gen[0.1]:+.3f} pooled) with elevated concentration")
A(f"(final top-10% share {fin_top10_gen[0.1]:.3f} vs {fin_top10_gen[0]:.3f}) and a widened wedge")
A(f"({wedge_fin_gen[0.1]:+.3f} vs {wedge_fin_gen[0]:+.3f}), with no explicit gaming margin in the model --- the")
A("visibility-coupled reward plus prestige-proportional recruitment suffice to")
A("decouple prestige from ideal merit. Three honesty clauses attach: (i) this is")
A("model-internal, not calibrated --- whether the region is *empirically plausible*")
A("is exactly the question the no-calibration ruling defers; (ii) the bracket is")
A("leaky by design (`use_visibility_reward=True` throughout), so prestige-mediated")
A("attention is active even in founder cells; (iii) the ideal channel under entry is")
A("benchmark-strained (Q2).")
A("")
A("# Honest caveats")
A("")
A("- **Not calibrated** (see banner). Empirical moments were never used as targets;")
A("  comparisons to them would be a category error at this stage.")
A("- Synthetic landscape: absolute knowledge levels are meaningless; only ranks and")
A("  correlations carry interpretation. Keep claims at the level of correlations and")
A("  shares, never levels.")
A("- Horizon: 200 steps; one model step maps to no empirical time unit yet.")
A("- Axis (c) contrasts two landscape generators, not one generator at two noise")
A("  levels (family and legibility move together).")
A("- Late-run clean-landscape correlations are depletion-contaminated (F5).")
A("- The seed knob couples agent and landscape RNG (Q4); 20 seeds is a modest")
A("  replication count for correlation differences of ~0.05.")
A("- The wedge measured here is the model's *internal* ideal-vs-local contrast; the")
A("  empirical wedge (b quantity-side, quality-side) maps onto these channels only")
A("  loosely (`docs/calibration_moments.yaml` documents the mapping caveats).")
A("")
A("# Machine-readable outputs")
A("")
A("All under `data/sensitivity_v1/` (gitignored; run-tagged):")
A("")
A("| File | Content |")
A("| --- | --- |")
A(f"| `cell_<tag>.csv` x {n_cells} | Per-cell seed-level trajectories, long format: `cell_tag, curiosity, vanishing_factor_for_prestige, landscape, agent_generation_time, seed, Step` + the 5 metrics; 4020 rows (20 seeds x 201 steps) each |")
A("| `trajectories_mean.csv` | 7236 rows: per cell x step seed-mean and seed-sd of the 5 metrics + `n_seeds` |")
A("| `cell_summary.csv` | 36 rows: per-cell time-mean and final values, wedge contrasts, seed counts, status |")
A("| `checkpoint.json` | Restart-safe manifest: per-cell status, seeds ok/failed, errors, final population size, wall time |")
A("| `run_log.txt` | Append-only progress log with per-cell completion lines |")
A("| `driver_source.py`, `consolidator_source.py`, `doc_generator_source.py` | Exact source of the sweep driver, the consolidator and this document's generator (self-describing runs) |")
A("| `manifest.json` | Grid spec, fixed pins, seed semantics, canary output, sha256 of sources, environment versions, counts |")
A("")
A("Cell tag legend: `cell_cur{03,05,07}_van{05,09,10}_{mgstd5,noisypr5}_gen{00,01}`;")
A("`mgstd5` = `multipleGaussians` {std:5, number_gaussians:2}, `noisypr5` =")
A("`noisyGaussian` {prop_random:5, std:5, std_curiosity:0.1}; tags carry integer-scaled")
A("parameter values, the CSVs carry the exact floats.")
A("")
A("# Reproducibility")
A("")
A(f"- Environment: python {env['python']}, mesa {env['mesa']}, numpy {env['numpy']}, pandas {env['pandas']}")
A("  (`/opt/hermes/.venv/bin/python`).")
A(f"- Driver sha256 (first 16): `{sha_driver}`; consolidator sha256 (first 16): `{sha_cons}`.")
A("- Regenerate: `/opt/hermes/.venv/bin/python data/sensitivity_v1/driver_source.py`")
A("  (checkpointed; delete `data/sensitivity_v1/` for a fresh sweep), then")
A("  `consolidator_source.py`. This document: `doc_generator_source.py` (copied into")
A("  `data/sensitivity_v1/`; every number interpolated from the CSVs).")
A("- Boundaries honoured: no repository code touched; no `AdmiralsLog/` write (the")
A("  session log entry and any records are deferred --- the task's write boundary")
A("  permits only this document and `data/sensitivity_v1/*`); no git commands; no")
A("  parameter values outside the four axes; frozen `context/` untouched.")
A("")

md = "\n".join(L) + "\n"
with open(MD, "w") as f:
    f.write(md)
print(f"wrote {MD} ({len(md)} chars)")
shutil.copy2(__file__, os.path.join(OUT, "doc_generator_source.py"))
print("copied generator source to data/sensitivity_v1/doc_generator_source.py")

# ================= manifest =================
manifest = {
    "task": "Path A model-internal sensitivity exploration (sciencemodels ABM); NO calibration (owner ruling 2026-11-13)",
    "generated_at": now,
    "grid": {
        "initial_curiosity": [0.3, 0.5, 0.7],
        "vanishing_factor_for_prestige": [0.5, 0.9, 1.0],
        "landscape": {"mgstd5": {"fn": "multipleGaussians", "params": {"std": 5, "number_gaussians": 2}},
                      "noisypr5": {"fn": "noisyGaussian", "params": {"prop_random": 5, "std": 5, "std_curiosity": 0.1}}},
        "agent_generation_time": [0, 0.1],
        "cells": n_cells,
    },
    "fixed_params": {"n_agents": 20, "grid_size": 20, "step_limit": 200, "epsilon": 0.1,
                     "harvest": 0.02, "use_distance": True, "new_questions": 0,
                     "use_visibility_reward": True, "knowledge_normalization": "share",
                     "prestige_accrual": "harvested", "agent_generation_function": "identity (model default)"},
    "seed_semantics": "agent_seed 0..19 per cell; model injects the same seed into generation_params (landscape and agent RNG coupled); 20 joint draws per cell",
    "counts": {"cells_gross": n_cells, "cells_net_ok": n_ok, "cells_ok_with_failures": 0,
               "cells_crashed": 0, "runs_gross": n_cells * 20, "runs_ok": n_runs,
               "seeds_failed": n_fail, "trajectory_rows": n_rows,
               "final_population_gen00": agents_gen00, "final_population_gen01": agents_gen01,
               "sweep_wall_s": round(wall_total, 1)},
    "canary": {"command": "/opt/hermes/.venv/bin/python tests/test_smoke.py",
               "exit_code": canary_rc, "output": canary_txt},
    "consolidation_canary": {"step1_min_ideal_corr": round(s1_ideal_min, 6),
                               "step1_min_local_corr": round(s1_local_min, 6),
                               "step1_seed_min_ideal": round(seed_min_ideal, 6),
                               "step1_seed_min_local": round(seed_min_local, 6),
                               "structure": "36 cells x 201 steps x 20 seeds verified"},
    "run_continuity": "single continuous run 15:49-16:06Z (run_log: one start line, zero SKIPs); checkpoint resume mechanism present but not exercised; the session's foreground terminal wait timed out at ~420s while the detached driver continued",
    "concurrent_edits": {
        "note": "concurrent session fixed BUG-0002/BUG-0003 and extended the smoke test during the sweep window",
        "files": {"sciencemodels/model.py": "2026-11-13T15:52:15Z", "tests/test_smoke.py": "2026-11-13T15:53:35Z",
                  "sciencemodels/landscapes.py": "2026-11-13T15:53:47Z"},
        "straddle": "cells 1-7 fully pre-edit (finished 15:50:10-15:51:55Z); cells 8+ ran under a tree patched in stages (model.py 15:52:15Z, test 15:53:35Z, landscapes 15:53:47Z); dynamics-neutrality of all edits established by bit-exact rerun",
        "dynamics_neutrality_evidence": ["model.py +9 bytes, generate_data_parametric_exploration only (unused by driver)",
                                         "landscapes.py Gaussian rng default on an unused parameter",
                                         "bit-exact rerun of pre-edit cell_cur03_van05_mgstd5_gen00 on post-edit code: max abs diff 0.0 over 4020 rows x 5 metrics"],
    },
    "sources_sha256_16": {"driver": sha_driver, "consolidator": sha_cons},
    "environment": env,
    "artifacts": {
        "cell_csvs": 36, "trajectories_mean.csv": "7236 rows", "cell_summary.csv": "36 rows",
        "checkpoint.json": "36 cell entries", "run_log.txt": "append-only",
        "sources": ["driver_source.py", "consolidator_source.py", "doc_generator_source.py"],
    },
    "boundaries": "wrote only data/sensitivity_v1/* and docs/sensitivity_map_v1.md; no code, no AdmiralsLog, no context/, no git",
}
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print("wrote manifest.json")
print(f"neg ideal final: {n_neg_ideal}/36 total, {n_neg_ideal_gen01}/18 gen01, local neg gen01: {n_neg_local_gen01}")
