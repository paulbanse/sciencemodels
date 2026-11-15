# sciencemodels

Here is a project to model scientific exploration: an agent-based model (Mesa)
of scientists searching an epistemic landscape, studying the interplay of
quality and prestige and (in planned extensions) collaborator choice under
proxy failure.

## Layout

- `sciencemodels/` - the model package: `agents`, `model`, `landscapes`,
  `metrics`, `viz`, `io`, `plotting`, and `sweeps` (the shared sweep machinery
  of the career-flow experiment family: fixed configuration, landscape presets,
  reporter list and the two measurement functions, imported by both arms).
- `scripts/` - entry points and round drivers. Base entry points are
  `runner.py` (interactive single run) and `bulk_runner.py` (parametric sweep);
  analysis rounds live in their own subfolders (`career_flows/`,
  `career_flows_noisy/`, `figures/`, `welfare/`, `path_b/`, `remeasure/`,
  `prestige_stock/`, `shimfix/`, `report_notebook/`). Round drivers read the
  model package and write run-tagged CSVs; they never write into each other's
  folders.
- `notebooks/` - thin presentation notebooks over `sciencemodels.plotting`.
- `tests/` - `test_smoke.py`, the seeded canary (a script, not a pytest suite;
  see below).
- `data/` - run archives, one folder per round (`data/<round>_v1/`), all
  gitignored. `data/legacy_2025_2026/` quarantines the loose pre-restructure
  CSVs (Nov 2025 - Jan 2026) with their own provenance note.
- `output/reports/` - round reports (HR-docs with compiled PDFs).
- `drafts/` - paper drafts and manuscript sketches (`drafts/paper_sketch_v1.md`, with superseded drafts in `drafts/archive/`).
- `docs/` - long-standing conceptual documents; `AdmiralsLog/` - project
  records. `IDEA.md` is the orientation document; `AdmiralsLog/decisions/`
  is authoritative for anything already decided.
- `context/` - frozen background literature (read-only, additive only).

## To run

Install the pinned dependencies with `pip install -r requirements.txt`, then
run `python scripts/runner.py` (interactive single run) or
`python scripts/bulk_runner.py` (parametric bulk sweep).

The verification gate is the seeded smoke test, run as a script:

```{.bash}
MPLBACKEND=Agg python tests/test_smoke.py
```

`pytest -q` is *not* the gate in this repository: the smoke test carries no
`test_*` functions, so pytest collects nothing and exits 5. Run the script
directly until a pytest wrapper exists.

`requirements.txt` pins exact versions. It was refreshed on 2026-11-14 for
`pandas` and `joblib`; the pins for `numpy`, `scipy` and `matplotlib` still
describe the older project environment and do not match the current container
(see `AdmiralsLog/logs/2026-11.md`, 2026-11-14) - rebuild the environment from
the lock or refresh the lock from the environment, but do not mix the two.
