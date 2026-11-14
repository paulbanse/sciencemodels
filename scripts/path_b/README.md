# Path B round sources (`path_b_v1`)

Driver, analysis, doc-generator and re-anchor sources for the Path B dyadic
collaborator-choice round (2026-11-14). Data outputs live in
`data/path_b_v1/`; the report is `output/reports/path_b_probe_v1.md`.

- `run_path_b_sweep.py` — Gate 2 sweep driver: 40 cells (5 lambda x 2
  formulation x 2 credit x 2 landscape), 20 seeds each, 800 runs, canary
  gate, atomic per-cell checkpoints, skip-on-restart, manifest.
- `run_baseline_off.py` — off-mode structural baseline (dyadic_mode="off",
  2 landscapes x 20 seeds): the structural floor and the lambda-response
  counterfactual.
- `analyse_path_b.py` — Gate 2 analyser with the pre-declared mechanical
  verdict rule (declared before results were read); emits
  `gate2_table.csv`, `lambda_trend.csv`, `gate2_verdict.json`,
  `cell_summary.csv`, `baseline_off_summary.csv`.
- `make_doc.py` — programmatic generator of
  `output/reports/path_b_probe_v1.md`; every number is interpolated from
  the artefact CSVs, none hand-typed.
- `reanchor_cdf_apply.py` — one-shot applicator that landed the CDF
  re-anchor: doc-side cell rebuild from the doc's own segment structure,
  the re-measure round's two deferred rows + mapping entries inserted in
  one change, prose-only citations re-anchored, status note and Version 5
  revision-log entry. Ran against `docs/conceptual_degrees_of_freedom.md`
  and `scripts/cdf_reanchor/reanchor_cdf_refs.py`; the gate
  (`python3 scripts/cdf_reanchor/reanchor_cdf_refs.py`) verifies the
  result and PASSES (106 rows, 8 inline anchors).
- `reanchor_pairing_probe.py` — order-preserving content-validated pairing
  of doc rows with mapping rows; located the two owner-added rows the v3.0
  mapping never tracked (since resolved by the deferral/landing mechanism).
- `reanchor_doc_classify.py` / `reanchor_doc_refcheck.py` — coverage
  classifiers: table cells vs prose refs, and the gate's blind spot
  (prose-only citations), used to close the prose side of the re-anchor.

Scratch originals of the last four lived under `/tmp` (one-shot round
sources; archived here for provenance per ADR-0002).
