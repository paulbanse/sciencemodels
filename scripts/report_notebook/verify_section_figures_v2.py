"""Independent verification for the Section 4.3/5.2/5.3/6 figure commission.

Recomputes every plotted quantity from the RAW source CSVs and asserts a
match to ``tol`` (default 1e-9) against each figure's ``*_data.csv`` sidecar,
verifies expected run/cell counts, and asserts zero clipped or overlapping
text artists on every figure canvas. Family-level checks live in
``verify_fig<A-F>.py`` (each exposing ``check(outdir, tol) -> list[str]``;
an empty list means all checks pass).

Usage::

    MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
        scripts/report_notebook/verify_section_figures_v2.py

Exit code 0 = all checks pass; 1 = at least one failure (printed).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import verify_figA
import verify_figB
import verify_figC
import verify_figD
import verify_figE
import verify_figF

FAMILIES = [
    ("A", "F8_v2_turnover_diagnostics", verify_figA),
    ("B", "F12_sabbaticals_vs_lotteries", verify_figB),
    ("C", "F9_v2_typology_annealing", verify_figC),
    ("D", "F13_paradigm_refill_regimes", verify_figD),
    ("E", "F10a_m2_choice_to_production", verify_figE),
    ("F", "F10b_m2_visibility_like_for_like", verify_figF),
]

# Commission Section 3.2: expected source run/cell counts overall.
EXPECTED_COUNTS = {
    "turnover runs": 1080,
    "Arm B runs": 720,
    "typology runs": 480,
    "Arm C runs": 360,
    "Path B v1+v2+off runs": 1800,
    "M2 xi-extension runs": 200,
}


def main(outdir: str = "output/report_notebook_v1", tol: float = 1e-9) -> int:
    all_failures = []
    for tag, name, module in FAMILIES:
        print(f"[{tag}] verifying {name} ...", flush=True)
        try:
            failures = module.check(outdir=outdir, tol=tol)
        except Exception as exc:  # a crashed verifier is a failure, never a pass
            failures = [f"verifier raised: {exc!r}"]
        if failures:
            for f in failures:
                print(f"  FAIL: {f}")
            all_failures.extend(f"[{tag}] {f}" for f in failures)
        else:
            print(f"  PASS ({name})")

    print("\nExpected source counts (asserted per family where applicable):")
    for k, v in EXPECTED_COUNTS.items():
        print(f"  {k}: {v}")

    if all_failures:
        print(f"\n{len(all_failures)} FAILURE(S).")
        return 1
    print("\nALL CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
