# sciencemodels

Here is a project to model scientific exploration: an agent-based model (Mesa)
of scientists searching an epistemic landscape, studying the interplay of
quality and prestige and (in planned extensions) collaborator choice under
proxy failure.

## Layout

- `sciencemodels/` - the model package (agents, model, landscapes, metrics,
  viz, io, plotting)
- `scripts/` - entry points (`runner.py`, `bulk_runner.py`)
- `notebooks/` - thin presentation notebooks over `sciencemodels.plotting`
- `tests/` - seeded smoke test
- `context/` - frozen background literature; `docs/` - documentation;
  `AdmiralsLog/` - project records (`IDEA.md` is the orientation document)

## To run

Install needed dependencies via pip using the `requirements.txt` file
(generated with pipreqs), then use python to run `scripts/runner.py`
(interactive single run) or `scripts/bulk_runner.py` (parametric bulk sweep).
The smoke test runs via `python tests/test_smoke.py`.
