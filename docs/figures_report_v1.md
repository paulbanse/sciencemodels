---
title: "Figures Regeneration v1: Six Notebooks on Fresh Sweep Data"
author: "Hermes Agent"
version: 1
date: 2026-11-13
tags: [figures, notebooks, validation, restructure, abm, proxy-failure]
project: sciencemodels
revision-vehicle: strikethrough
revision-renders: "md, pdf, html"
---

**Version 1** (November 2026)

The model is **not calibrated** to empirical data; per the owner ruling of
2026-11-13 the moments of `docs/calibration_moments.yaml` are an option menu,
and every figure here is model-internal exploration.

# Purpose and Scope

This report documents the regeneration of the six analysis notebooks' figures on
freshly simulated data, which doubles as an end-to-end validation of the
ADR-0002 restructure: thin notebooks over `sciencemodels.plotting`, the
`scripts/` entry points, and the legacy-column shim in
`sciencemodels/plotting.py`. It records what was run, what was produced, what
failed and why, and whether the restructure holds under real data.

Scope: a fresh 8-setup parametric sweep at `repeats_per_setting=10`,
`step_limit=200`; one longitudinal companion dataset for the trajectory
notebook; headless execution of all six notebooks; the collected figures; and
the verdict. No model code, no analysis code outside the three driver scripts in
`scripts/figures/`, and no notebook edit beyond the declared data-loading path
cells. Nothing is fitted or calibrated.

# Environment and Canary

Interpreter `/opt/hermes/.venv/bin/python` (mesa 3.2.0, numpy 2.3.2,
matplotlib 3.10.6, joblib, pandas; notebook execution through nbclient 0.11.0).
The canary was run before any other step and passed:

```{.text}
canary run OK: 40 report keys, harvested_10_step=37.5, harvested_25_step=37.5
same-seed rerun bit-identical OK
newcomer run OK: 6 scientists after 20 steps
4-arg Gaussian OK: np.float64(0.00550470835249002) (float64)
/tmp/smoke_append_z4g5ts44/append_header_check.csv w
now processing param set  1 out of 1
/tmp/smoke_append_z4g5ts44/append_header_check.csv a
now processing param set  1 out of 1
append-mode header check OK: 3 lines = 1 header + 2 data rows
ALL SMOKE TESTS PASSED
```

The canary also exercises the two open bug candidates from the W2 round and
covers both the 4-argument `landscapes.Gaussian` call and the `filename +
".csv"` header check, so those two are no longer live defects in this tree.

# Fresh Sweep

Run parameters: 8 setups (4 landscape families x swept/fixed curiosity market,
`chosen_setup = 0..7` as in `scripts/bulk_runner.py`), `repeats_per_setting=10`,
`step_limit=200`, `change_landscape_seed=True`, `n_jobs=3`. Driver:
`scripts/figures/run_fresh_sweep.py`; outputs `data/figures_v1/run_number<k>.csv`
plus `data/figures_v1/sweep_manifest.json`.

Every setup was written through the production interface
`generate_data_parametric_exploration` into a temporary file and renamed only
after its row count matched the gross expectation, so no partial file can pass
as complete.

| Setup | Landscape family | Curiosity regime | Gross runs | Net runs | Worker time (s) |
| --- | --- | --- | --- | --- | --- |
| 0 | randominit | swept 0.0-1.0 (11 values) | 110 | 110 | 220.0 |
| 1 | randominit | fixed 0.5, uniform draw | 10 | 10 | 36.6 |
| 2 | MultG2 | swept 0.0-1.0 (11 values) | 110 | 110 | 225.5 |
| 3 | MultG2 | fixed 0.5, uniform draw | 10 | 10 | 22.5 |
| 4 | MultG6 | swept 0.0-1.0 (11 values) | 110 | 110 | 230.4 |
| 5 | MultG6 | fixed 0.5, uniform draw | 10 | 10 | 20.3 |
| 6 | Noisy2 | swept 0.0-1.0 (11 values) | 110 | 110 | 106.7 |
| 7 | Noisy2 | fixed 0.5, uniform draw | 10 | 10 | 17.9 |
| **all** | 4 families x 2 regimes | | **480** | **480** | 879.9 worker-seconds |

Gross equals net: 480 runs attempted, 480 completed, wall clock roughly six
minutes with `n_jobs=3` (manifest written 2026-11-13T21:11:18).

Data integrity checks, all computed on the eight files: 480 rows in total; per
file 110 rows (swept setups) or 10 rows (fixed setups); duplicate
`(initial_curiosity, generation_params)` rows: zero; missing values in
`harvested_10_step` and `Mean_of_Corr_prestige_local_merit`: zero.

Descriptive means over the ten repeats, stated as description only --- no
parameter is fitted and no moment is targeted:

| Setup | n | mean corr(prestige, local merit) | sd | mean corr(prestige, ideal merit) | mean 10% harvest step | mean explored share | mean top-10% prestige share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 randominit, swept | 110 | 0.288 | 0.129 | 0.373 | 105.30 | 0.453 | 0.116 |
| 1 randominit, fixed | 10 | 0.387 | 0.099 | 0.496 | 74.80 | 0.534 | 0.117 |
| 2 MultG2, swept | 110 | 0.211 | 0.248 | 0.221 | 39.85 | 0.337 | 0.121 |
| 3 MultG2, fixed | 10 | 0.424 | 0.243 | 0.594 | 15.20 | 0.333 | 0.125 |
| 4 MultG6, swept | 110 | 0.477 | 0.309 | 0.493 | 60.32 | 0.360 | 0.131 |
| 5 MultG6, fixed | 10 | 0.702 | 0.188 | 0.743 | 31.00 | 0.386 | 0.132 |
| 6 Noisy2, swept | 110 | 0.533 | 0.274 | 0.510 | 84.11 | 0.438 | 0.156 |
| 7 Noisy2, fixed | 10 | 0.613 | 0.128 | 0.575 | 56.40 | 0.487 | 0.153 |

The longitudinal companion (`scripts/figures/make_longitudinal_data.py`) holds
four agent-trajectory files (`data/figures_v1/agent_long_{random,MGauss1,
MGauss4,NGauss}.csv`, 200 steps each) and 22 per-run model frames in
`data/figures_v1/longitudinal_random/` (11 curiosities x 2 agent seeds), gross 26
artifacts, net 26.

# Notebook Data-Path Edits

Only data-loading path cells were edited, applied as literal string
replacements by `scripts/figures/point_notebooks_at_figures_v1.py` (each
replacement asserted to occur exactly once; every file re-parsed as valid
notebook JSON afterwards; the diff against the pre-edit copies contains nothing
but the path tokens). Nineteen path tokens across six notebooks:

| Notebook | Cell | Edit |
| --- | --- | --- |
| `analyze_data.ipynb` | 0 | `end_report.csv` -> `data/figures_v1/run_number0.csv` |
| `compare_landscapes_Prestige.ipynb` | 0 | four `data/testrun_visibilityReward_*.csv` -> `data/figures_v1/run_number{0,2,4,6}.csv` |
| `compare_landscapes_flexible_search.ipynb` | 0 | four `data/testrun_visibilityReward_*.csv` -> `data/figures_v1/run_number{0,2,4,6}.csv` |
| `compare_landscapes_normalizedknowledge.ipynb` | 0 | four `testrun_harmonizedKnowledge_*.csv` -> `data/figures_v1/run_number{0,2,4,6}.csv` |
| `compare_landscapes_corr-prestige-merit.ipynb` | 0 | `run_name = 'large_run_with_ideal'` -> `run_name = 'figures_v1'` |
| `longitudinalcomparisions.ipynb` | 2 | `data/agent_long_random` -> `data/figures_v1/longitudinal_random` |
| `longitudinalcomparisions.ipynb` | 3 | four `data/agent_long_*.csv` -> the same names under `data/figures_v1/` |

Mapping rule: the four-landscape "compare" notebooks point at the
curiosity-swept member of each family (setups 0, 2, 4, 6). The legacy files
their cells named encoded other parameter settings (visibility reward switched
on for the `testrun_visibilityReward_*` set, harmonised knowledge for the
`testrun_harmonizedKnowledge_*` set); those parameter differences are **not**
reproduced, because only paths were in scope. The distinctiveness of the legacy
files therefore collapses onto one common data set --- a finding, listed below,
not a licence to change parameters inside a path edit.

# Notebook Execution

Harness: `scripts/figures/execute_notebooks.py`. It builds a throwaway mirror
(`/tmp/figs_exec`: copies of `notebooks/`, symlinks to `sciencemodels/`,
`data/`, `docs/`, `scripts/`) so execution never writes into the repository,
prepends one in-memory capture cell that replaces `plt.show` with a `savefig`
(under `MPLBACKEND=Agg` every notebook otherwise produces no artifact at all),
executes with `allow_errors=True` so one broken cell does not hide the rest, and
removes the capture cell again before saving the executed notebook. Figures are
named `<notebook>_fig<NN>_<cellhash>.png`, the suffix tying each file to its
producing cell.

| Notebook | Code cells | Executed | Figures | Failing cells | Verdict |
| --- | --- | --- | --- | --- | --- |
| `analyze_data.ipynb` | 6 | clean | 4 | - | all figures regenerated |
| `compare_landscapes_Prestige.ipynb` | 8 | clean | 13 | - | all figures regenerated |
| `compare_landscapes_corr-prestige-merit.ipynb` | 8 | errors | 13 | 10, 11 | harvest/exploration panels regenerated; all six correlation panels missing (schema name) |
| `compare_landscapes_flexible_search.ipynb` | 8 | clean | 13 | - | all figures regenerated |
| `compare_landscapes_normalizedknowledge.ipynb` | 3 | clean | 11 | - | all figures regenerated |
| `longitudinalcomparisions.ipynb` | 7 | errors | 3 | 2 | trajectory panels regenerated; mean-corr-over-time panel missing (schema name) |

Six notebooks attempted, four executed clean, 57 figures produced. Execution
times under concurrent load in the shared tree: 59.2, 84.3, 107.1, 80.9, 64.3
and 79.3 seconds.

The failing cells, quoted verbatim (no fix applied):

```{.python}
# compare_landscapes_corr-prestige-merit.ipynb, cell 10
plot_smt("Mean_of_Corr_prestige_localMerit", datasets)
plot_smt("Mean_of_Corr_prestige_visibility_local_merit", datasets)
plot_smt("Mean_of_Corr_prestige_vanishing_local_merit", datasets)
# KeyError: 'Mean_of_Corr_prestige_localMerit'
```

```{.python}
# compare_landscapes_corr-prestige-merit.ipynb, cell 11
plot_smt("Mean_of_Corr_prestige_idealMerit", datasets)
plot_smt("Mean_of_Corr_prestige_visibility_ideal_merit", datasets)
plot_smt("Mean_of_Corr_prestige_vanishing_ideal_merit", datasets)
# KeyError: 'Mean_of_Corr_prestige_idealMerit'
```

```{.python}
# longitudinalcomparisions.ipynb, cell 2
plot_all_curiosities("data/figures_v1/longitudinal_random")
# KeyError: 'corr_prestige_localMerit'
```

# Figure Inventory

All files live under `data/figures_v1/figures/` (3.7 MB, 57 PNG, 110 dpi).

| Notebook | Cell | Figure files |
| --- | --- | --- |
| `analyze_data` | 1 | `analyze_data_fig00_dbfb72e5.png`, `analyze_data_fig01_dbfb72e5.png` |
| `analyze_data` | 2 | `analyze_data_fig02_ba4dac06.png` |
| `analyze_data` | 4 | `analyze_data_fig03_182a374a.png` |
| `compare_landscapes_Prestige` | 3 | `..._Prestige_fig00_23a976a4.png`, `fig01_23a976a4`, `fig02_23a976a4` |
| `compare_landscapes_Prestige` | 5 | `..._fig03_00a1f72e.png`, `fig04_00a1f72e` |
| `compare_landscapes_Prestige` | 7 | `..._fig05_48f519c3.png` through `fig10_48f519c3` (6 files) |
| `compare_landscapes_Prestige` | 9 | `..._fig11_79d5ecc7.png`, `fig12_79d5ecc7` |
| `compare_landscapes_corr-prestige-merit` | 3 | `..._fig00_23a976a4.png`, `fig01_23a976a4`, `fig02_23a976a4` |
| `compare_landscapes_corr-prestige-merit` | 5 | `..._fig03_00a1f72e.png`, `fig04_00a1f72e` |
| `compare_landscapes_corr-prestige-merit` | 7 | `..._fig05_48f519c3.png` through `fig10_48f519c3` (6 files) |
| `compare_landscapes_corr-prestige-merit` | 9 | `..._fig11_79d5ecc7.png`, `fig12_79d5ecc7` |
| `compare_landscapes_flexible_search` | 3 | `..._flexible_search_fig00_23a976a4.png`, `fig01_23a976a4`, `fig02_23a976a4` |
| `compare_landscapes_flexible_search` | 5 | `..._fig03_00a1f72e.png`, `fig04_00a1f72e` |
| `compare_landscapes_flexible_search` | 7 | `..._fig05_48f519c3.png` through `fig10_48f519c3` (6 files) |
| `compare_landscapes_flexible_search` | 9 | `..._fig11_79d5ecc7.png`, `fig12_79d5ecc7` |
| `compare_landscapes_normalizedknowledge` | 2 | `..._normalizedknowledge_fig00_f42fee1e.png` through `fig10_f42fee1e` (11 files) |
| `longitudinalcomparisions` | 3 | `longitudinalcomparisions_fig00_4437abd4.png` |
| `longitudinalcomparisions` | 4 | `longitudinalcomparisions_fig01_26c69d43.png` |
| `longitudinalcomparisions` | 5 | `longitudinalcomparisions_fig02_c2e5e423.png` |

Full machine-readable inventory: `data/figures_v1/notebook_run_manifest.json`
(per notebook: executed flag, seconds, every figure path with its cell, and
every failing cell with traceback tail). Executed notebook artifacts:
`data/figures_v1/executed/<notebook>.ipynb`.

# Verdict on the Restructure

The thinning works. All six notebooks execute as thin wrappers: the bootstrap
cell resolves the repo root from either the notebook directory or the repo root,
the plotting helpers come from `sciencemodels.plotting`, and no notebook carries
its own copy of `plot_smt`. The notebooks produced 57 figures without any code
edit beyond the data paths.

The column shim works exactly as documented --- in both directions. Fresh CSVs
carry `harvested_10_step` and `harvested_25_step`, the notebooks still request
`harvested_50_step` and `harvested_90_step`, and `_resolve_column` maps the
request onto the column present (`harvested_50_step -> harvested_10_step`,
verified by direct call and by four notebooks plotting those series without
error). Reading a legacy CSV through `plotting.read_csv` renames
`harvested_50_step -> harvested_10_step` as documented.

The shim is **incomplete**, and that is the one real breakage. It covers the
harvest family only. The correlation family drifted twice over: the current
model emits `Mean_of_Corr_prestige_local_merit` (snake_case, from the
datacollector reporter names through `build_report`), the notebook requests
`Mean_of_Corr_prestige_localMerit` (camelCase), and the legacy CSVs in `data/`
carry a third spelling (`corr_prestige_localMerit`, `mean_corr_prestige_localMerit`).
Nothing maps between them, so the corr plots have no data source in either
direction: the fresh data has the numbers under names no cell asks for.
`plot_all_curiosities` fails the same way on `corr_prestige_localMerit`.

So: restructure validated for layout, import path, plotting helper, sweep entry
point and the harvest-column shim; refuted for shim coverage of the corr metric
family. Four of six notebooks are end-to-end green; the two failures are
schema-name drift, not restructure damage, and they are reported rather than
repaired because the task's carve-out allows path edits only.

# Findings and Open Items

- **Shim coverage gap (blocking the corr notebook).** Extend the shim with the
  correlation family, or rename the notebook call sites to the current schema;
  a three-way alias table (`Mean_of_Corr_prestige_local_merit`,
  `Mean_of_Corr_prestige_localMerit`, `mean_corr_prestige_localMerit`) is the
  minimum. Owner decision, because it touches shared analysis code.
- **`plot_all_curiosities` has no working data source.** It needs a folder of
  model-level per-step frames keyed by `curiosity<c>` and a column
  `corr_prestige_localMerit`. The legacy layout had `data/agent_long_random` as
  a *file*, not a directory, so this cell was already dead before this
  regeneration; the fresh companion folder exists now, and the column name is
  the remaining gap.
- **Two notebooks pointed at data that does not exist in the repo.**
  `analyze_data.ipynb` read a repo-root `end_report.csv`, and
  `compare_landscapes_normalizedknowledge.ipynb` read four root-level
  `testrun_harmonizedKnowledge_*.csv` files; there is no such file at the repo
  root (one noisy-knowledge variant sits in `data/`). Both cells were dead
  before the regeneration and are live now.
- **`compare_landscapes_corr-prestige-merit.ipynb` was entirely dead.** Its
  legacy path `data/large_run_with_ideal/` does not exist in this checkout
  (removed with the restructure); cells 0-9 only became runnable because the
  fresh sweep writes exactly that eight-file design under `figures_v1`.
- **The legacy per-config distinctions are lost.** `Prestige` and
  `flexible_search` now plot identical data (both point at the same four
  setups), and `normalizedknowledge` plots the same four setups again with a
  different metric list. Reproducing the legacy settings
  (`use_visibility_reward`, `knowledge_normalization`) needs a parameterised
  sweep design, not a path edit.
- **The setup table is duplicated.** `scripts/figures/run_fresh_sweep.py`
  repeats the 8-setup design of `scripts/bulk_runner.py`, because the legacy
  runner is a script that launches its own sweep at import and cannot be
  imported from. ADR-0002 wants parameter setups in one place per experiment
  family; the fix is to lift the setup table and grid builder into the package
  and let both entry points call it.
- **Threshold columns saturate at `1.25 x step_limit`.** With `step_limit=200`
  the sentinel is 250. Rows at that value: `harvested_10_step` 51 of 480
  (10.6%), `harvested_25_step` 184 of 480 (38.3%), `explored_50_step` 385 of 480
  (80.2%), `explored_90_step` 480 of 480 (100.0%), `weighted_50_step` 65 of 480,
  `weighted_90_step` 398 of 480. The two `90_step` panels are therefore flat at
  the sentinel in this run: a reading caveat, not a defect.
- **Shared tree, concurrent load.** Other sessions' artifacts
  (`scripts/career_flows/`, `scripts/welfare/`, `docs/welfare_probe_v1.md`,
  moved bug records) were written while this round ran; none was touched here.
  Notebook execution ran under that load, which is why the six runs took 59 to
  107 seconds rather than the 20 to 35 seconds measured on an idle tree.

# Reproduction

```{.bash}
cd /workspace/git/sciencemodels
MPLBACKEND=Agg /opt/hermes/.venv/bin/python tests/test_smoke.py
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/figures/run_fresh_sweep.py
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/figures/make_longitudinal_data.py
/opt/hermes/.venv/bin/python scripts/figures/point_notebooks_at_figures_v1.py
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/figures/execute_notebooks.py
```

Each driver is restart-safe: the sweep skips a setup whose final CSV already
holds its gross row count, the longitudinal driver skips present artifacts, and
the notebook harness rebuilds its mirror and clears nothing in the repo. Re-running
`point_notebooks_at_figures_v1.py` is a no-op once the paths are rewired (it
requires exactly one occurrence per replacement and reports the miss).

# Limitations

- Not calibrated, and not calibrated by design: no empirical moment of
  `docs/calibration_moments.yaml` is targeted, no parameter is fitted. The
  descriptive table is description.
- Ten repeats per setting and 200 steps: enough to regenerate the canonical
  figures, not enough for distributional claims. Uncertainty bands in the
  figures are the sd over ten runs.
- The figures are captured from `plt.show` under `MPLBACKEND=Agg`; the notebook
  artifacts saved under `data/figures_v1/executed/` therefore carry no embedded
  images, and each figure's PNG is the canonical artifact instead.
- The legacy per-config parameter settings are not reproduced (path edits only).
- The HR-doc compile to PDF was skipped: this task's write boundary covers
  `docs/figures_report_v1.md` only, so the PDF was neither produced nor staged.
  Compile with `zsh /workspace/git/ai/IT/scripts/compile-here.zsh docs/figures_report_v1.md`
  when the boundary allows it.

# Revision Log

- **Version 1** (November 2026) --- Fresh 8-setup sweep (480/480 runs) plus a
  longitudinal companion dataset, headless execution of all six notebooks (4
  clean, 2 with schema-name failures quoted verbatim), 57 figures inventoried
  with their producing cells, and the restructure verdict: layout and
  harvest-column shim validated, correlation-family shim missing.
