"""Aggregate entry point for the Section 4.3/5.2/5.3/6 figure commission.

Regenerates all six publication figures (PNG @ 180 dpi + vector PDF + tidy
``*_data.csv`` sidecar per figure) into ``output/report_notebook_v1/``:

    F8_v2_turnover_diagnostics        (Section 4.3, family A)
    F12_sabbaticals_vs_lotteries      (Section 5.2, family B)
    F9_v2_typology_annealing          (Section 5.2, family C)
    F13_paradigm_refill_regimes       (Section 5.3, family D)
    F10a_m2_choice_to_production      (Section 6, family E)
    F10b_m2_visibility_like_for_like  (Section 6, family F)

Each family lives in its own module under ``scripts/report_notebook/`` and
exposes ``build(outdir) -> dict`` with keys ``png``/``pdf``/``data_csv``.
This aggregator only orchestrates; all values are computed inside the family
modules from the raw run archives (zero hardcoded numbers).

Usage::

    MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
        scripts/report_notebook/section_figures_v2.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figA_turnover_diagnostics
import figB_sabbaticals_vs_lotteries
import figC_typology_annealing
import figD_paradigm_refill_regimes
import figE_m2_choice_to_production
import figF_m2_visibility_like_for_like

FAMILIES = [
    ("A", "F8_v2_turnover_diagnostics", figA_turnover_diagnostics),
    ("B", "F12_sabbaticals_vs_lotteries", figB_sabbaticals_vs_lotteries),
    ("C", "F9_v2_typology_annealing", figC_typology_annealing),
    ("D", "F13_paradigm_refill_regimes", figD_paradigm_refill_regimes),
    ("E", "F10a_m2_choice_to_production", figE_m2_choice_to_production),
    ("F", "F10b_m2_visibility_like_for_like", figF_m2_visibility_like_for_like),
]


def main(outdir: str = "output/report_notebook_v1") -> dict:
    results = {}
    for tag, name, module in FAMILIES:
        print(f"[{tag}] building {name} ...", flush=True)
        out = module.build(outdir=outdir)
        missing = [k for k in ("png", "pdf", "data_csv") if not out.get(k)]
        if missing:
            raise RuntimeError(f"[{tag}] {name}: missing outputs {missing}")
        for k in ("png", "pdf", "data_csv"):
            if not os.path.exists(out[k]):
                raise RuntimeError(f"[{tag}] {name}: {k} not written: {out[k]}")
        results[name] = out
        print(f"[{tag}] ok: {out['png']}", flush=True)
    print(f"\nAll {len(FAMILIES)} figure families built.")
    return results


if __name__ == "__main__":
    main()
