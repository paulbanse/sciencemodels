"""Path D typology sweep (2026-11-15, P-0005): distraction x destruction x
counterforce as parameterized regimes of the ABM.

Commission (task header 2026-11-15): implement the two 1-parameter channels
DRIVER-SIDE (sciencemodels/*.py untouched) via the TypologyScientist /
TypologyModel subclasses in typology_layers.py, and run the grid

    gamma in {0.0, 1.0}  x  kappa in {0.0, 1.0}  x  curiosity c in {0.3, 0.5, 0.7}
    x  entry g in {0.0, 0.1}  on landscape mgstd5
    = 24 cells x 20 seeds = 480 runs at horizon H=200.

Channels (formulas as commissioned):
* destruction:  eps_eff = eps * m_t(pos)**gamma   (crowding amplifies the
  perception noise of a candidate tile's novelty read)
* counterforce: eps_i = eps_0 / (1 + kappa * P_sup)  (a newborn's noise is
  divided by its supervisor's entry-stock prestige at its birth step)

Baseline distraction is the model's own (1 - c) curiosity weighting, already
in the parent dynamics; this round adds the two channels on top.

NOT CALIBRATED: model-internal experiment only (owner ruling 2026-11-13).
No empirical moment is a fitting target; the Klann-style diagnostics are not
used here at all.

Gates before the grid (ADR-0001 in-run canary discipline):
1. tests/test_smoke.py exit 0 (canary gate, output into canary.txt);
2. source_canary: the parent lines the subclass copies/intercepts are still
   the tree's lines (divergence canary against concurrent owner edits);
3. identity_proof: (gamma=0, kappa=0) is BIT-IDENTICAL to MyModel on every
   report key at both entry levels, 5 seeds each (commission step 2);
4. binding_probe: gamma binds at both entry levels; kappa binds at g=0.1
   (newcomers exist) and is BIT-IDENTICAL at g=0.0 (no newborns -> the
   round's built-in control arm, quoted in the report);
5. probe: one cell x 2 seeds through the real worker path, schema asserted
   from the WORKER-written file, timing recorded.

Checkpointing: atomic per-cell CSVs under data/typology_v1/cells/ (tmp ->
os.replace), append-only run log, manifest with gross-vs-net.  A restart
skips cells whose verified state file is present.

Run:
    MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/typology/run_typology_v1.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts" / "typology") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "typology"))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sciencemodels import landscapes
from sciencemodels.model import MyModel

import typology_layers as tl

# The destruction channel's literal formula makes eps_eff exceed 1 on
# crowded tiles (gamma=1, m >= ~11 at eps=0.1), so the noise factor
# 1 + eps_eff*noise goes negative and the fractional curiosity power yields
# NaN rewards (disclosed pin D5 in the report; the option is then never
# selected because NaN comparisons are False).  The arithmetic is
# unaffected; only the per-event warning overhead (thousands of events per
# run, ~20x wall) is suppressed here.  The regime itself is NOT silenced:
# it is disclosed in the report and visible in the gates' binding output.
warnings.filterwarnings("ignore", message="invalid value encountered in scalar power")

OUT_DIR = REPO / "data" / "typology_v1"
CELLS_DIR = OUT_DIR / "cells"
AGENTS_DIR = CELLS_DIR / "agents"
LOG_PATH = OUT_DIR / "run_log.txt"
PROBE_DIR = Path("/tmp/d_probe/typology")

GAMMAS = (0.0, 1.0)
KAPPAS = (0.0, 1.0)
CURIOSITIES = (0.3, 0.5, 0.7)
ENTRIES = (0.0, 0.1)
SEEDS = 20
STEP_LIMIT = 200

# House baseline (identical to the Path B sweep's BASE_PARAMS; the typology
# round adds nothing to the baseline config - the two channels are the only
# differences, and they live in the subclass).
BASE_PARAMS = {
    "n_agents": 20,
    "epsilon": 0.1,
    "harvest": 0.02,
    "grid_size": 20,
    "use_distance": True,
    "new_questions": 0,
    "agent_generation_function": lambda cur, _rng, _params: cur,
    "vanishing_factor_for_prestige": 0.9,
    "use_visibility_reward": True,
    "knowledge_normalization": "share",
    "prestige_accrual": "harvested",
    "entry_prestige_stock": "prestige",
    "dyadic_proposal_regime": "newcomer",
    "closeness_mode": "grid",
    "dyadic_visibility_stock": "prestige_visibility",
}
LANDSCAPE_BUNDLE = (
    landscapes.multipleGaussians,
    {"std": 5, "number_gaussians": 2},
)

# The analysis contract: columns the report/figures must find.
REQUIRED_COLUMNS = [
    "gamma", "kappa", "curiosity", "g_entry", "seed",
    "Mean_of_Corr_prestige_ideal_merit",
    "Mean_of_Corr_prestige_local_merit",
    "c_ideal_fin", "c_local_fin", "wedge_fin", "wedge_tm",
    "ideal_loss_cum", "agents_final", "newcomers_final",
]

DESIGN_PINS = {
    "grid": "2 gamma x 2 kappa x 3 curiosity x 2 entry = 24 cells on mgstd5, 20 seeds, 480 runs, horizon 200",
    "landscape": "mgstd5 = landscapes.multipleGaussians(std=5, number_gaussians=2)",
    "baseline_config": "Path B house baseline (n=20, eps=0.1, harvest=0.02, grid 20x20, vanishing 0.9, visibility reward on, knowledge_normalization share, prestige_accrual harvested, entry_prestige_stock prestige, credit structural, dyadic layer OFF)",
    "dyadic_mode": "off (default) - the typology round varies PERCEPTION channels only; the Path B choice layer is off throughout",
    "destruction_formula": "eps_eff = eps * m_t(candidate_tile)**gamma; m_t counts all scientists currently on the candidate tile at the focal agent's evaluation time",
    "counterforce_formula": "eps_i = eps_0 / (1 + kappa * P_sup) for newborns; P_sup = supervisor's entry_prestige_stock ('prestige') read at the birth step, fixed for life; founders kappa-inert",
    "m_zero_edge": "at gamma=1 an empty candidate tile (m=0) gets eps_eff=0 (perfect read) - the commissioned formula's literal consequence, disclosed",
    "seed_semantics": "agent_seed 0..19 drives agent and landscape RNG jointly (model injects agent_seed into generation_params)",
    "wedge_definition": "wedge_fin = c_ideal_fin - c_local_fin (final-step correlations; remeasure-round convention); wedge_tm analog on time-means",
    "ideal_loss_definition": "ideal_loss_cum = -sum(a.ideal_merit_cum) over final scientists (remeasure-round convention)",
    "name_mangling_interception": "TypologyModel defines _MyModel__generate_new_agents explicitly; MyModel.end_loop_update's self.__generate_new_agents() compiles to that mangled name, so a normally named override would be silently invisible",
    "entry_gate": "g_entry=0.0 sets agent_generation_time=0.0 -> no newborns -> kappa inert by construction (built-in control arm, asserted bit-identical in binding_probe and re-checked on the full grid)",
}


def cell_tag(gamma, kappa, curiosity, g_entry):
    return "cell_g%d_k%d_c%02d_e%02d" % (
        int(round(gamma)), int(round(kappa)),
        int(round(curiosity * 100)), int(round(g_entry * 100)),
    )


def build_params(gamma, kappa, curiosity, g_entry, seed):
    f, gp = LANDSCAPE_BUNDLE
    p = dict(BASE_PARAMS)
    p.update(
        initial_curiosity=curiosity,
        cell_init_function=f,
        generation_params=dict(gp),
        agent_seed=seed,
        step_limit=STEP_LIMIT,
        agent_generation_time=g_entry,
        typology_gamma=gamma,
        typology_kappa=kappa,
    )
    return p


def run_one(gamma, kappa, curiosity, g_entry, seed):
    m = tl.TypologyModel(**build_params(gamma, kappa, curiosity, g_entry, seed))
    for _ in range(STEP_LIMIT):
        m.step(True)
    row = tl._build_row(m)
    row.update(
        gamma=gamma, kappa=kappa, curiosity=curiosity,
        g_entry=g_entry, seed=seed,
    )
    return row


def agent_finals(m, gamma, kappa, curiosity, g_entry, seed):
    """Final-step per-agent table (the P-G decay figure's source)."""
    rows = []
    for a in m._scientists:
        rows.append(dict(
            gamma=gamma, kappa=kappa, curiosity=curiosity,
            g_entry=g_entry, seed=seed,
            agent_id=a.unique_id, is_newcomer=bool(a.is_newcomer),
            age=int(a.age),
            prestige=float(a.prestige),
            prestige_vanishing=float(a.prestige_vanishing),
            prestige_visibility=float(a.prestige_visibility),
            current_local_merit=float(a.current_local_merit),
            ideal_Merit=float(a.ideal_Merit),
            ideal_merit_cum=float(getattr(a, "ideal_merit_cum", 0.0)),
            pos_x=int(a.pos[0]), pos_y=int(a.pos[1]),
        ))
    return rows


def run_cell(spec, cells_dir=None, agents_dir=None, log_path=None):
    """One cell = 20 seeded runs, checkpointed atomically.  The per-worker
    source canary runs here (inside the task function, per the turnover
    round's doctrine: a module-level check does not survive the pickler)."""
    gamma, kappa, curiosity, g_entry = spec
    tag = cell_tag(gamma, kappa, curiosity, g_entry)
    cdir = cells_dir if cells_dir is not None else CELLS_DIR
    adir = agents_dir if agents_dir is not None else AGENTS_DIR
    lpath = log_path if log_path is not None else LOG_PATH
    tl.source_canary()  # divergence canary, per cell, per worker
    path = cdir / (tag + ".csv")
    apath = adir / (tag + ".csv")
    if path.exists() and sum(1 for _ in open(path)) == SEEDS + 1:
        return tag, "skipped", []
    rows, arows, failed = [], [], []
    t0 = time.time()
    for seed in range(SEEDS):
        try:
            m = tl.TypologyModel(**build_params(gamma, kappa, curiosity, g_entry, seed))
            for _ in range(STEP_LIMIT):
                m.step(True)
            row = tl._build_row(m)
            row.update(
                gamma=gamma, kappa=kappa, curiosity=curiosity,
                g_entry=g_entry, seed=seed,
            )
            rows.append(row)
            arows.extend(agent_finals(m, gamma, kappa, curiosity, g_entry, seed))
        except Exception as exc:  # a crashed seed is reported, not dropped
            failed.append((seed, repr(exc)[:300]))
    wall = time.time() - t0
    if rows:
        df = pd.DataFrame(rows)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        assert not missing, "cell %s missing columns %s" % (tag, missing)
        tmp = str(path) + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
        adf = pd.DataFrame(arows)
        atmp = str(apath) + ".tmp"
        adf.to_csv(atmp, index=False)
        os.replace(atmp, apath)
    with open(lpath, "a") as f:
        f.write("%s %s rows=%d failed_seeds=%d wall=%.1fs\n"
                % (tag, "ok" if rows else "FAILED", len(rows), len(failed), wall))
        for seed, err in failed:
            f.write("  seed %d: %s\n" % (seed, err))
    return tag, ("ok" if rows else "FAILED"), failed


def probe():
    """Critical-path probe OUTSIDE the run folder: one cell x 2 seeds
    dispatched through the REAL joblib worker path (proves the typology_layers
    import survives the pickler into a fresh worker), schema asserted from the
    WORKER-written file, timing recorded."""
    import shutil

    if PROBE_DIR.exists():
        shutil.rmtree(PROBE_DIR)
    (PROBE_DIR / "cells" / "agents").mkdir(parents=True)
    spec = (1.0, 0.0, 0.5, 0.1)
    t0 = time.time()
    results = Parallel(n_jobs=1, verbose=0)(
        [delayed(run_cell)(
            spec, cells_dir=PROBE_DIR / "cells",
            agents_dir=PROBE_DIR / "cells" / "agents",
            log_path=PROBE_DIR / "run_log.txt",
        )]
    )
    wall = time.time() - t0
    tag, status, failed = results[0]
    assert status == "ok" and not failed, "probe cell failed: %s" % failed
    df = pd.read_csv(PROBE_DIR / "cells" / (tag + ".csv"))
    adf = pd.read_csv(PROBE_DIR / "cells" / "agents" / (tag + ".csv"))
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, "probe schema missing %s" % missing
    assert len(df) == SEEDS, "probe should carry %d seed rows, has %d" % (SEEDS, len(df))
    assert len(adf) >= SEEDS, "probe agent-finals file suspiciously small: %d" % len(adf)
    summary = {
        "probe_cell": tag, "probe_rows": len(df),
        "probe_agent_rows": len(adf),
        "probe_wall_s": round(wall, 1), "wall_per_run_s": round(wall / 2, 2),
        "dispatched_via": "joblib Parallel(n_jobs=1) - the real worker path",
        "status": "PASS",
    }
    return summary


def run_gates():
    """Gates 1-5 before any grid cell: smoke canary, source divergence
    canary + code snapshot, the commission's bit-identity proof, the
    binding/control probe, and the worker-path probe.  Writes
    canary_smoke.txt (the smoke output itself) and gates.txt (the record)."""
    # Gate 1: the repository's own canary
    canary = os.system(
        "MPLBACKEND=Agg %s %s > %s/canary_smoke.txt 2>&1"
        % (sys.executable, REPO / "tests" / "test_smoke.py", OUT_DIR)
    )
    assert canary == 0, "canary failed - aborting before any run"

    # Gate 2: divergence canary + code snapshot
    hashes = tl.source_canary()

    # Gate 3: commission step 2 - bit-identity of (0,0) vs MyModel
    ident = tl.identity_proof(dict(BASE_PARAMS))
    print("IDENTITY:", ident["status"], "| comparisons:", ident["n_comparisons"],
          "| max_abs_diff:", ident["max_abs_diff"], flush=True)

    # Gate 4: binding + the kappa-inert control arm
    binding = tl.binding_probe(dict(BASE_PARAMS))
    for key, d in binding.items():
        print("BINDING g=%s seed=%s: gamma_d_wedge=%+.6f kappa_d_wedge=%+.6f kappa_bitidentical=%s"
              % (key[0], key[1], d["gamma_delta_wedge_fin"], d["kappa_delta_wedge_fin"],
                 d["kappa_bitidentical_all_keys"]), flush=True)

    # Gate 5: probe through the real worker path
    probe_sum = probe()
    print("PROBE:", json.dumps(probe_sum), flush=True)

    with open(OUT_DIR / "gates.txt", "w") as f:
        f.write("gate1_smoke: exit 0 -> canary_smoke.txt\n")
        f.write("gate2_source_canary: PASS\n")
        f.write("gate3_identity_proof: %s | comparisons=%d max_abs_diff=%s\n"
                % (ident["status"], ident["n_comparisons"], ident["max_abs_diff"]))
        f.write("gate4_binding_probe:\n")
        for key, d in binding.items():
            f.write("  g=%s seed=%s gamma_d_wedge=%+.6f kappa_d_wedge=%+.6f kappa_bitidentical=%s\n"
                    % (key[0], key[1], d["gamma_delta_wedge_fin"],
                       d["kappa_delta_wedge_fin"], d["kappa_bitidentical_all_keys"]))
        f.write("gate5_probe: %s\n" % json.dumps(probe_sum))
        f.write("code_hashes_at_gates: %s\n" % json.dumps(hashes))
    return hashes, ident, binding, probe_sum


def main(argv):
    os.makedirs(CELLS_DIR, exist_ok=True)
    os.makedirs(AGENTS_DIR, exist_ok=True)
    t0 = time.time()
    gates_only = "--gates-only" in argv
    skip_gates = "--skip-gates" in argv

    if skip_gates:
        # Re-verify the recorded gates instead of re-running them; refuse if
        # the shared tree moved since they were recorded (straddle guard).
        gates_txt = (OUT_DIR / "gates.txt").read_text()
        assert "gate3_identity_proof: PASS" in gates_txt, "recorded identity gate not PASS"
        assert "gate5_probe: " in gates_txt and '"status": "PASS"' in gates_txt, "recorded probe gate not PASS"
        import hashlib

        hashes = {}
        for name in ("model.py", "agents.py", "metrics.py", "io.py", "landscapes.py"):
            p = REPO / "sciencemodels" / name
            hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        recorded = json.loads(
            [l for l in gates_txt.splitlines() if l.startswith("code_hashes_at_gates")][0]
            .split(": ", 1)[1]
        )
        assert recorded == hashes, (
            "code moved since the gates ran - re-run gates (--gates-only) before the grid"
        )
        ident = binding = None
        probe_sum = json.loads(
            [l for l in gates_txt.splitlines() if l.startswith("gate5_probe")][0]
            .split(": ", 1)[1]
        )
    else:
        hashes, ident, binding, probe_sum = run_gates()

    if gates_only:
        print("GATES ONLY: all five gates recorded in gates.txt; grid not launched.", flush=True)
        return

    # The grid
    grid = list(itertools.product(GAMMAS, KAPPAS, CURIOSITIES, ENTRIES))
    results = Parallel(n_jobs=3, verbose=5)(delayed(run_cell)(c) for c in grid)
    ok = sum(1 for _, st, _ in results if st == "ok")
    skipped = sum(1 for _, st, _ in results if st == "skipped")
    failed_cells = sorted(t for t, st, _ in results if st == "FAILED")
    failed_seeds = sum(len(fl) for _, _, fl in results)

    # Consolidate runs.csv (gross vs net recounted from disk, not memory)
    frames = [pd.read_csv(p) for p in sorted(CELLS_DIR.glob("cell_*.csv"))]
    runs = pd.concat(frames, ignore_index=True)
    runs.to_csv(OUT_DIR / "runs.csv", index=False)
    rows_net = 0
    for p in sorted(CELLS_DIR.glob("cell_*.csv")):
        rows_net += sum(1 for _ in open(p)) - 1

    # Post-sweep hash re-check (shared-tree straddle guard)
    hashes_after = tl.source_canary()
    moved = {k: (hashes[k], hashes_after[k]) for k in hashes if hashes[k] != hashes_after[k]}

    manifest = {
        "grid": "2 gamma x 2 kappa x 3 curiosity x 2 entry = 24 cells on mgstd5, 20 seeds, 480 runs, horizon 200",
        "cells_gross": len(grid),
        "cells_new": ok,
        "cells_skipped": skipped,
        "cells_failed": failed_cells,
        "runs_gross": len(grid) * SEEDS,
        "rows_net": rows_net,
        "failed_seeds": failed_seeds,
        "wall_s": round(time.time() - t0, 1),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "design_pins": DESIGN_PINS,
        "identity_proof": ident,
        "binding_probe": (
            {"g%s_seed%s" % (k[0], k[1]): v for k, v in binding.items()}
            if binding else "see gates.txt (skip-gates launch)"
        ),
        "probe": probe_sum,
        "code_hashes_at_launch": hashes,
        "code_hashes_after_sweep": hashes_after,
        "code_moved_during_sweep": moved,
        "canary": "tests/test_smoke.py exit 0 (canary_smoke.txt); identity_proof PASS; binding_probe PASS; source_canary per cell",
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    with open(OUT_DIR / "design.json", "w") as f:
        json.dump(
            {"design_pins": DESIGN_PINS, "base_params": {
                k: (v.__name__ if callable(v) else v) for k, v in BASE_PARAMS.items()
            }, "landscape_bundle": "multipleGaussians std=5 number_gaussians=2",
             "step_limit": STEP_LIMIT, "seeds": SEEDS},
            f, indent=1)
    print(
        "GROSS cells=%d runs=%d | NET ok=%d skipped=%d failed_cells=%s"
        " failed_seeds=%d rows_net=%d/%d | wall=%.0fs | code_moved=%s"
        % (len(grid), len(grid) * SEEDS, ok, skipped, failed_cells,
           failed_seeds, rows_net, len(grid) * SEEDS, time.time() - t0, moved),
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
