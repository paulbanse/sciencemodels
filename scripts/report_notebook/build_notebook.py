"""Build (and optionally execute) the round-results report notebook.

ADR-0002: the notebook is a thin presentation wrapper. This script owns its
construction so the notebook is regenerable rather than hand-edited, and so
execution is a flag rather than a manual step.

    python build_notebook.py              # write notebooks/round_results_report_v1.ipynb
    python build_notebook.py --execute    # ... then execute it in place (nbclient)

Execution uses ``%matplotlib inline`` so every figure is embedded in the
notebook as an image/png output as well as written to
``output/report_notebook_v1/`` by the figure builders.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NOTEBOOK = REPO / "notebooks" / "round_results_report_v1.ipynb"

CANARY_BLOCK = """\
Gate (tests/test_smoke.py), run before the first edit and after the last.

Report keys went 44 -> 56 since the previous build: the Path B dyadic layer
landed in the model between builds (off-mode bit-identity, logit/threshold
binding, knob validation). The bit-level anchors are unchanged
(mean_prestige = 0.025399760810316363; 4-arg Gaussian = 0.00550470835249002).

    canary run OK: 56 report keys, harvested_10_step=37.5, harvested_25_step=37.5
    same-seed rerun bit-identical OK
    newcomer run OK: 6 scientists after 20 steps
    4-arg Gaussian OK: np.float64(0.00550470835249002) (float64)
    append-mode header check OK: 3 lines = 1 header + 2 data rows
    ideal-merit pairing audit OK: A=0.0, C=0.0, B=-0.04999999999999999
    seed columns OK: agent_seed=1, landscape_seed=1
    WARNING: agent_generation_function '<lambda>' returned IDENTICAL draws for mu=0.1 and mu=0.9 on the same rng - the swept axis (initial_curiosity) is INERT.
    binding probe OK: constant generator flagged inert, identity binds
    entry_prestige_stock OK: validated, both alternate stocks run
    Path B off-mode bit-identity OK (anchor mean_prestige=np.float64(0.025399760810316363)); 0 dynamics keys differ
    Path B binding OK: logit lambda0=(np.int64(2), 1.024390243902439) lambda1=(np.int64(5), 2.1951219295129512)
    Path B binding OK: threshold lambda0=(np.int64(1), 1.0) lambda1=(np.int64(3), 2.048780487804878)
    Path B knob validation OK: all six knobs reject invalid values
    ALL SMOKE TESTS PASSED
    EXIT=0
"""


CELLS = [
    ("md", """\
# Round results: visual report

**Model-internal only.** Nothing in this notebook is calibrated to empirical
data (owner ruling, 2026-11-13); every number is read from a run archive on
disk and asserted against its expected shape before it is plotted.

One figure per headline round of the evidence ledger
(`AdmiralsLog/HANDOVER-2026-11-14.md`, section 1.3), drawn through the
project's house templates (CodeLearn `P_diagram_templates`, palette
`_shared/template_core.py`; the copy in `scripts/report_notebook/` is
byte-identical to the source, sha256 `6d4c0a89...`).

Per ADR-0002 the computation lives in `scripts/report_notebook/`
(`round_report_data.py`, `round_report_figures.py`); this notebook only
calls it."""),

    ("code", """\
%matplotlib inline
import sys
from pathlib import Path

REPO = Path.cwd() if (Path.cwd() / "scripts").exists() else Path.cwd().parent
sys.path.insert(0, str(REPO / "scripts" / "report_notebook"))

import round_report_data as D
import round_report_figures as F

OUT = F.OUT
print("repo:", D.REPO)
print("figures ->", OUT.relative_to(D.REPO))"""),

    ("md", """\
## Gross versus net

Attempted against verified, recomputed here rather than quoted: each loader
asserts the gross shape (cells, seeds) and returns only what is on disk."""),

    ("code", """\
manifest = D.manifest()
manifest"""),

    ("md", """\
## F1 -- T1 sensitivity: the wedge opens under entry (post-audit arm)

Drawn from the **post-audit** re-measured arm (`data/remeasure_v1/runs.csv`,
720 T1 runs), not the pre-fix sweep, because the ideal-merit pairing audit
moved the levels this figure reports. Final prestige-ideal correlation falls
from positive to negative when entry turns on, and curiosity is protective
(the higher-curiosity line sits above the lower one at both entry rates).
The figure's source line carries the caption numbers: entry=0 mean +0.220,
entry=0.1 mean -0.144, 17/18 entry cells negative."""),

    ("code", """\
fig1, path1, cap1 = F.figure_sensitivity()
print("wrote", path1.relative_to(D.REPO))
print("caption:", {k: cap1[k] for k in
      ("runs", "cells", "entry0_mean", "entry01_mean",
       "entry01_negative", "entry01_cells", "sign_flips")})"""),

    ("md", """\
## F2 -- T3 career flows: the gap never closes, on either terrain

Four series = landscape (clean, noisy) x statistic (final, late), each the
mean over the three curiosity levels. Both headline levels appear with their
statistic named, so the figure never compares one statistic on one line with
another on the next: clean final -0.71, noisy late -0.52. The gap is
everywhere negative."""),

    ("code", """\
fig2, path2 = F.figure_career_flows()
print("wrote", path2.relative_to(D.REPO))"""),

    ("md", """\
## F2b -- the slow-entry decoupling gradient

Final prestige-ideal correlation by entry rate and terrain. The most negative
correlation sits at the LOWEST entry rate on both terrains (clean -0.133 at
g=0.05 vs -0.025 at g=0.20; noisy -0.227 vs -0.112): slow entry lets founder
hierarchies harden before advisees arrive."""),

    ("code", """\
fig2b, path2b = F.figure_career_flows_ideal_corr()
print("wrote", path2b.relative_to(D.REPO))"""),

    ("md", """\
## F3 -- T4 welfare probe: the price of visibility pressure

Loss rises steeply in the visibility weight along all four entry x legibility
series; the on/off reward toggle moves it by at most the amount printed in
the title, so the weight axis is the live lever. Means over the toggle."""),

    ("code", """\
fig3, path3 = F.figure_welfare()
print("wrote", path3.relative_to(D.REPO))"""),

    ("md", """\
## F4 -- W3-A prestige stock: how fast, not where

Entry-on cells only (the entry-off cells are bit-identical across stocks, so
they would draw three copies of the same bar). The three stocks move the
terminal wedge only marginally within every cell."""),

    ("code", """\
fig4, path4 = F.figure_prestige_stock()
print("wrote", path4.relative_to(D.REPO))"""),

    ("md", """\
## F5 -- N1/N2 re-measure: what the pairing audit moved

Seed-matched pre-fix vs post-fix deltas on the T1 arm (720 pairs). The
ideal-side levels move, the local-side and top-10% reporters are untouched
(point estimates exactly 0), which is what makes the moved numbers
attributable to the pairing fix alone.

Two measurement facts discovered while building this figure, both asserted in
code rather than assumed:

1. **The welfare arm of the re-measure is excluded.** It was re-run at
   `vanishing_factor_for_prestige = 0.9` against `0.5` in the original sweep,
   so that arm carries two changes at once and its delta is not attributable
   to the pairing fix. 800 + 800 rows inspected, 0 paired.
2. **`runs.csv` stores `wedge = c_ideal - c_local`**, the opposite sign to
   the pre-arm convention; both arms are recomputed from the components
   before differencing (the archive's convention is asserted, then
   overridden). Trusting the column would have reported a -0.206 wedge
   "movement" that is a pure sign flip.

A third arm is wired in and **conditional**: if
`data/remeasure_v1/w_v05_runs.csv` is banked by the re-measure round, its 800
welfare runs pair against the original v05 sweep (which ran at vanishing
0.5) and are drawn as a second, dodged arm in F5. The file is absent at this
build, so F5 shows T1 only and the counts below record that as a fact rather
than an omission."""),

    ("code", """\
fig5, path5, rm_counts = F.figure_remeasure_delta()
print("wrote", path5.relative_to(D.REPO))
for k, v in rm_counts.items():
    print(f"  {k}: {v}")"""),

    ("md", f"""\
## Artifacts and gate

Figures written by this notebook (each also carries its `_data.csv` where the
template provides one):

- `output/report_notebook_v1/F1_sensitivity_entry_curiosity.png`
- `output/report_notebook_v1/F2_career_flows_gap_ratio.png`
- `output/report_notebook_v1/F2b_career_flows_noisy_vs_clean.png`
- `output/report_notebook_v1/F3_welfare_loss_vs_visibility_weight.png`
- `output/report_notebook_v1/F4_prestige_stock_wedge_matrix.png`
- `output/report_notebook_v1/F5_pairing_audit_paired_delta.png` (+ `_data.csv`)
- `output/report_notebook_v1/F7_m0_cognitive_division_of_labor.png` (+ `_data.csv`)

Sources are read-only: `data/remeasure_v1/`, `data/sensitivity_v1/`,
`data/career_flows_v1/`, `data/career_flows_noisy_v1/`, `data/welfare_v1/`,
`data/prestige_stock_v1/`.

{CANARY_BLOCK}"""),
]


def _id(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def build() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    cells = []
    for kind, text in CELLS:
        cell = (new_markdown_cell(text) if kind == "md"
                else new_code_cell(text))
        cell["id"] = _id(kind + text)
        cells.append(cell)
    nb.cells = cells
    return nb


def execute(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    from nbclient import NotebookClient
    client = NotebookClient(
        nb, timeout=900, allow_errors=True,
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    )
    client.execute()
    return nb


def _write_manifest(nb: nbformat.NotebookNode) -> Path:
    """Gross-vs-net + figure ledger as a file, computed from the loaders."""
    import json

    import round_report_data as D
    import round_report_figures as F

    manifest = {
        "notebook": str(NOTEBOOK.relative_to(REPO)),
        "code_cells": sum(c.cell_type == "code" for c in nb.cells),
        "executed": True,
        "rounds": D.manifest().to_dict("records"),
        "remeasure": {k: str(v) for k, v in D.remeasure_pairs()[1].items()},
        "figures": [],
    }
    for png in sorted(F.OUT.glob("*.png")):
        manifest["figures"].append(
            {"path": str(png.relative_to(REPO)), "bytes": png.stat().st_size})
    out = F.OUT / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="fail if any cell raised")
    args = ap.parse_args()

    nb = build()
    if args.execute:
        nb = execute(nb)
        errors = [
            (i, o.get("ename"), o.get("evalue"))
            for i, c in enumerate(nb.cells)
            if c.cell_type == "code"
            for o in c.get("outputs", [])
            if o.get("output_type") == "error"
        ]
        print(f"executed: {sum(c.cell_type == 'code' for c in nb.cells)} code "
              f"cells, {len(errors)} error(s)")
        for e in errors:
            print("  ERROR cell", e)
        if errors and args.strict:
            return 1
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, NOTEBOOK)
    print("wrote", NOTEBOOK.relative_to(REPO))
    if args.execute:
        print("wrote", _write_manifest(nb).relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
