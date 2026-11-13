"""Repoint the six notebooks' data-loading cells at the figures-v1 sweep.

The carve-out for the figures-v1 validation permits exactly one kind of notebook
edit: the data-loading path cells. This script applies those edits as literal
string replacements on the notebook JSON (so nothing but the path tokens
changes), asserts each replacement happens the expected number of times, and
re-parses every file afterwards.

Legacy path -> figures-v1 artifact mapping
------------------------------------------
The legacy notebooks each loaded their own single-run CSVs (different
visibility/normalisation settings, no unified sweep). The fresh data is the
8-setup bulk design, so the four-landscape "compare" notebooks are repointed at
the curiosity-sweep members of each family (setups 0/2/4/6: randominit, 2
gaussians, 6 gaussians, noisy gaussian); the parameter differences the legacy
files encoded (visibility reward on/off, harmonised knowledge) are NOT
reproduced -- that is a finding for the report, not something a path edit may
fix.

Run from anywhere:

    /opt/hermes/.venv/bin/python scripts/figures/point_notebooks_at_figures_v1.py
"""
from __future__ import annotations

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NB = os.path.join(REPO, "notebooks")

FOUR_LEGACY_VIS = [
    ("data/testrun_visibilityReward_seed4_std5_number_gaussians2.csv", "run_number2"),
    ("data/testrun_visibilityReward_seed4_std5_number_gaussians6.csv", "run_number4"),
    ("data/testrun_visibilityReward_prop_random5_seed4_std5_std_curiosity0.1.csv", "run_number6"),
    ("data/testrun_visibilityReward_seed4.csv", "run_number0"),
]
FOUR_LEGACY_NORM = [
    ("testrun_harmonizedKnowledge_seed4_std5_number_gaussians2.csv", "run_number2"),
    ("testrun_harmonizedKnowledge_seed4_std5_number_gaussians6.csv", "run_number4"),
    ("testrun_harmonizedKnowledge_prop_random5_seed4_std5_std_curiosity0.2.csv", "run_number6"),
    ("testrun_harmonizedKnowledge_seed4.csv", "run_number0"),
]

EDITS = {
    "analyze_data.ipynb": [
        ("end_report.csv", "data/figures_v1/run_number0.csv", 1),
    ],
    "compare_landscapes_Prestige.ipynb": [
        (old, f"data/figures_v1/{new}.csv", 1) for old, new in FOUR_LEGACY_VIS
    ],
    "compare_landscapes_flexible_search.ipynb": [
        (old, f"data/figures_v1/{new}.csv", 1) for old, new in FOUR_LEGACY_VIS
    ],
    "compare_landscapes_normalizedknowledge.ipynb": [
        (old, f"data/figures_v1/{new}.csv", 1) for old, new in FOUR_LEGACY_NORM
    ],
    "compare_landscapes_corr-prestige-merit.ipynb": [
        ("run_name = 'large_run_with_ideal'", "run_name = 'figures_v1'", 1),
    ],
    "longitudinalcomparisions.ipynb": [
        ("data/agent_long_random.csv", "data/figures_v1/agent_long_random.csv", 1),
        ("data/agent_long_MGauss1.csv", "data/figures_v1/agent_long_MGauss1.csv", 1),
        ("data/agent_long_MGauss4.csv", "data/figures_v1/agent_long_MGauss4.csv", 1),
        ("data/agent_long_NGauss.csv", "data/figures_v1/agent_long_NGauss.csv", 1),
        ('data/agent_long_random\\")', 'data/figures_v1/longitudinal_random\\")', 1),
    ],
}


def main():
    failures = []
    for name, edits in EDITS.items():
        path = os.path.join(NB, name)
        with open(path) as f:
            text = f.read()
        before = text
        applied = []
        for old, new, expected in edits:
            n = text.count(old)
            if n != expected:
                failures.append(f"{name}: {old!r} found {n}x, expected {expected}")
                continue
            text = text.replace(old, new)
            applied.append((old, new))
        if text != before:
            json.loads(text)  # the file must stay valid notebook JSON
            with open(path, "w") as f:
                f.write(text)
        print(f"{name}: {len(applied)}/{len(edits)} path edits applied")
        for old, new in applied:
            print(f"    {old}  ->  {new}")
    if failures:
        print("\nFAILED:", *failures, sep="\n  ")
        sys.exit(1)
    print("\nall notebook data-path edits applied")


if __name__ == "__main__":
    main()
