"""Aggregate entry point for the Section 4.1/4.3/5.2/5.3/6 figure set (v3).

Regenerates every publication figure (PNG @ 180 dpi + vector PDF + tidy
``*_data.csv`` sidecar) into ``output/report_notebook_v1/``:

    F1_v2_t1_postbug                  (Section 4.1, post-BUG-0006 T1)
    F8_v2_turnover_diagnostics        (Section 4.3, pre-fix diagnostics)
    F8_v3_turnover_diagnostics        (Section 4.3, post-BUG-0006 + staggered)
    F6_v2_two_stage_proxy_failure     (Section 4.1/2.3 Figure 1 refresh)
    F12_sabbaticals_vs_lotteries      (Section 5.2, family B)
    F9_v2_typology_annealing          (Section 5.2, family C)
    F13_paradigm_refill_regimes       (Section 5.3, family D)
    F10a_m2_choice_to_production      (Section 6, family E)
    F10b_v2_m2_visibility_like_for_like  (Section 6, post-BUG-0006)
    F10b_m2_visibility_like_for_like  (Section 6, pre-fix, kept for contrast)

Each family lives in its own module exposing ``build(outdir, data_root=None)``
with keys ``png``/``pdf``/``data_csv``; every builder runs the shared
``visual_qa.assert_figure_clean`` contract internally (rendered artists +
collision-free layout), so a clean build is also a clean canvas.

Usage::

    MPLBACKEND=Agg /opt/hermes/.venv/bin/python \
        scripts/report_notebook/section_figures_v2.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fig_F1_v2_t1_postbug
import fig_F6_v2
import figA_turnover_diagnostics
import figA_v3_turnover_postbug
import figB_sabbaticals_vs_lotteries
import figC_typology_annealing
import figD_paradigm_refill_regimes
import figE_m2_choice_to_production
import figF_m2_visibility_like_for_like
import figF_v2_m2_visibility_postbug

FAMILIES = [
    ("F1_v2", "F1_v2_t1_postbug", fig_F1_v2_t1_postbug),
    ("A", "F8_v2_turnover_diagnostics", figA_turnover_diagnostics),
    ("A3", "F8_v3_turnover_diagnostics", figA_v3_turnover_postbug),
    ("F6_v2", "F6_v2_two_stage_proxy_failure", fig_F6_v2),
    ("B", "F12_sabbaticals_vs_lotteries", figB_sabbaticals_vs_lotteries),
    ("C", "F9_v2_typology_annealing", figC_typology_annealing),
    ("D", "F13_paradigm_refill_regimes", figD_paradigm_refill_regimes),
    ("E", "F10a_m2_choice_to_production", figE_m2_choice_to_production),
    ("F10b_v2", "F10b_v2_m2_visibility_like_for_like",
     figF_v2_m2_visibility_postbug),
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
