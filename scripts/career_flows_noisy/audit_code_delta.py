"""Measure the code delta between this round's sweep and the current model tree.

The sweep ran 2026-11-13T23:54-00:05 UTC. While it ran, a sibling round landed
an ideal-merit pairing audit and an entry-prestige-stock change
(`sciencemodels/{io,metrics,model,plotting}.py` and the smoke test were modified
between 23:54 and 00:29). This round's numbers therefore belong to the
pre-audit code. Instead of forcing the old values, this script re-runs a
sample of the same cells and seeds through the CURRENT code and measures the
delta on every reporter the analysis uses.

Sample: five seeds of the control cell (mgstd5, c=0.5, g=0.1, seeds 0-4) plus
three seeds of one noisy cell (c=0.5, g=0.1, seeds 0-2), both compared against
the CSVs this round already wrote.

Comparison grain: `(seed, Step)`, i.e. step level.  The archive stores one row
per seed per step (201 steps x 20 seeds per cell) and `run_one` returns the same
step-level frame for a re-run seed, so both sides are already at that grain;
the only thing that has to be established is that the two sides are paired on
the *same* (seed, Step) keys rather than on row order.  Step level is also the
grain the audit is asking about: the deltas quoted downstream are per-step
maxima (0.729684 on the control cell, 0.548018 on the noisy one), and the
reading the report draws from them -- a large per-step maximum with a small
mean shift is a re-pairing, not a dynamics change -- is only visible at step
level.  Aggregating to one row per seed would destroy that signal, so instead
of collapsing either side this script filters the archive to the re-run seeds
and verifies the `(seed, Step)` key sets match exactly before subtracting.
A row-count difference is reported as the grain error it is, with the rows-per-
seed on both sides, rather than as a bare "603 vs 4020".

Writes `code_delta.csv` and `code_delta.json`.  `--out` selects the destination
and defaults to the archived run folder; pass a scratch directory (e.g.
`/tmp/code_delta_check`) to verify without touching the archived outputs, which
another round owns.

    /opt/hermes/.venv/bin/python scripts/career_flows_noisy/audit_code_delta.py
    /opt/hermes/.venv/bin/python scripts/career_flows_noisy/audit_code_delta.py \
        --out /tmp/code_delta_check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

REPO = "/workspace/git/sciencemodels"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "career_flows_noisy"))

import numpy as np
import pandas as pd

from driver import FIXED, MODEL_TARGETS, career_measures_from_agent_vars, run_one
from sciencemodels import landscapes as L

# read side: the archived run, always read-only
ARCHIVE = os.path.join(REPO, "data", "career_flows_noisy_v1")
# write side: historical default, overridable with --out
DEFAULT_OUT = ARCHIVE

CASES = [
    # (landscape label, cell tag already on disk, gen_time, seeds to re-run)
    ("mgstd5", "control_cur5_van9_mgstd5_gen0p1_h200", 0.1, list(range(5))),
    ("noisypr5", "noisyB_cur5_van9_noisypr5_gen0p1_h200", 0.1, [0, 1, 2]),
]
GP = {
    "mgstd5": {"std": 5, "number_gaussians": 2},
    "noisypr5": {"prop_random": 5, "std": 5, "std_curiosity": 0.1},
}
FN = {"mgstd5": L.multipleGaussians, "noisypr5": L.noisyGaussian}
CAREER_COLS = ["n_agents", "incumbent_mean_prestige", "newcomer_mean_prestige",
               "prestige_gap", "prestige_gap_ratio", "percap_harvest_cum"]
# the grain the two sides are paired on: one row per seed per step on each side
KEY = ["seed", "Step"]


def sha16(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _rows_per_seed(df):
    return {int(k): int(v) for k, v in df.groupby("seed").size().items()}


def pair_step_level(now, old, tag, what, seeds):
    """Pair a re-run frame with its archive frame on `(seed, Step)`.

    Both sides are step-level by construction (archive: 201 rows per seed;
    re-run: `run_one` returns a 201-row frame per seed), so this does not
    aggregate anything -- it makes the pairing explicit and checks it, instead
    of trusting row order.  Row order alone would silently misalign the two
    sides if one of them ever lost, gained or reordered a step, and a bare
    row-count check cannot tell a genuine code delta from a grain mismatch.

    Raises with the rows-per-seed on both sides when the counts differ, so a
    step-level-vs-seed-level comparison names itself as a grain error.
    """
    for who, df in (("re-run", now), ("archive", old)):
        missing = [k for k in KEY if k not in df.columns]
        if missing:
            raise RuntimeError(f"{tag}: {what} {who} side lacks key columns {missing}")
        dup = int(df.duplicated(subset=KEY).sum())
        if dup:
            raise RuntimeError(
                f"{tag}: {what} {who} side has {dup} duplicate {KEY} rows; "
                "expected exactly one row per seed per step")
    now = now.sort_values(KEY).reset_index(drop=True)
    old = old.sort_values(KEY).reset_index(drop=True)
    if len(now) != len(old):
        now_pse, old_pse = _rows_per_seed(now), _rows_per_seed(old)
        raise RuntimeError(
            f"{tag}: {what} grain mismatch: re-run {len(now)} rows vs archive "
            f"{len(old)} rows for seeds {list(seeds)}; rows per seed: re-run "
            f"{now_pse}, archive {old_pse}. Both sides must be step-level (one "
            "row per seed per Step); a seed-level aggregate on either side is a "
            "grain error, not a numeric delta")
    bad = np.flatnonzero((now[KEY].to_numpy() != old[KEY].to_numpy()).any(axis=1))
    if len(bad):
        i = int(bad[0])
        raise RuntimeError(
            f"{tag}: {what} key mismatch at row {i}: re-run "
            f"{now[KEY].iloc[i].to_dict()} vs archive {old[KEY].iloc[i].to_dict()} "
            f"({len(bad)} of {len(now)} rows differ)")
    return now, old


def max_abs_delta(now, old, col):
    return float(np.max(np.abs(now[col].to_numpy() - old[col].to_numpy())))


def main():
    ap = argparse.ArgumentParser(description="Diff re-run reporters against the archived run.")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="directory for code_delta.csv/.json (default: the archived "
                         f"run folder {DEFAULT_OUT}); use a scratch dir to verify "
                         "without overwriting the archived outputs")
    args = ap.parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for land, tag, gen, seeds in CASES:
        model_now, career_now = [], []
        for s in seeds:
            status, payload, n_final, av = run_one(
                FN[land], dict(GP[land]), 0.5, 0.9, gen, 200, s)
            if status != "ok":
                raise RuntimeError(f"{tag} seed {s} failed:\n{payload[-1500:]}")
            model_now.append(payload)
            cm = career_measures_from_agent_vars(av, payload)
            cm.insert(0, "seed", s)
            career_now.append(cm)
        now_m = pd.concat(model_now, ignore_index=True)
        now_c = pd.concat(career_now, ignore_index=True)
        old_m = pd.read_csv(os.path.join(ARCHIVE, f"{tag}.csv"))
        old_c = pd.read_csv(os.path.join(ARCHIVE, f"{tag}_career.csv"))
        # the archive cells hold all 20 seeds; the audit re-runs a subset
        old_m = old_m[old_m["seed"].isin(seeds)]
        old_c = old_c[old_c["seed"].isin(seeds)]
        now_m, old_m = pair_step_level(now_m, old_m, tag, "model", seeds)
        now_c, old_c = pair_step_level(now_c, old_c, tag, "career", seeds)
        rec = {"cell": tag, "seeds_rerun": len(seeds), "rows": len(now_m),
               "rows_per_seed": int(now_m.groupby("seed").size().iloc[0]),
               "_delta_keys": []}
        for c in MODEL_TARGETS + ["Corr_prestige_local_merit"]:
            if c not in now_m.columns:
                continue
            d = max_abs_delta(now_m, old_m, c)
            rec[f"model_max_abs_delta_{c}"] = d
            rec[f"model_mean_old_{c}"] = round(float(old_m[c].mean()), 6)
            rec[f"model_mean_new_{c}"] = round(float(now_m[c].mean()), 6)
            rec["_delta_keys"].append(f"model_max_abs_delta_{c}")
        for c in CAREER_COLS:
            d = max_abs_delta(now_c, old_c, c)
            rec[f"career_max_abs_delta_{c}"] = d
            rec[f"career_mean_old_{c}"] = round(float(old_c[c].mean()), 6)
            rec[f"career_mean_new_{c}"] = round(float(now_c[c].mean()), 6)
            rec["_delta_keys"].append(f"career_max_abs_delta_{c}")
        rec["identical"] = all(rec[k] == 0.0 for k in rec["_delta_keys"])
        rec["max_abs_delta_overall"] = max(rec[k] for k in rec["_delta_keys"])
        del rec["_delta_keys"]
        rows.append(rec)

    out = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "code_delta.csv")
    json_path = os.path.join(out_dir, "code_delta.json")
    out.to_csv(csv_path, index=False)
    summary = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "comparison_grain": "seed x Step (step-level; archive and re-run are both "
                            "one row per seed per Step, paired on those keys)",
        "archive_dir": ARCHIVE,
        "sources_sha256_16_now": {p: sha16(os.path.join(REPO, p)) for p in
                                  ["sciencemodels/metrics.py", "sciencemodels/model.py",
                                   "sciencemodels/io.py", "tests/test_smoke.py"]},
        "cases": rows,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=1)
    for r in rows:
        print(f"\n{r['cell']}  seeds_rerun={r['seeds_rerun']} rows={r['rows']}")
        for k, v in r.items():
            if k.endswith(tuple(f"_delta_{c}" for c in MODEL_TARGETS + CAREER_COLS + ["Corr_prestige_local_merit"])):
                print(f"   {k}: {v}")
    print(f"\nwrote {csv_path}\nwrote {json_path}")
    if os.path.abspath(out_dir) == os.path.abspath(ARCHIVE):
        print("NOTE: wrote into the archived run folder (default); pass --out to keep it untouched.")


if __name__ == "__main__":
    main()
