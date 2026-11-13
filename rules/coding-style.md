---
title: "Rules - Coding style and reproducibility"
author: "Jakob Kapeller"
date: "2026-11-13"
tags: [rules, coding-style, reproducibility, notebooks]
project: "sciencemodels"
---

Normative policy for code in this repository. Decided with the owner
2026-11-13 (onboarding round); the target layout is the restructure direction
ruled the same day (package split along the conceptual levels, `scripts/`
entry points, thin notebooks, seeded smoke test, one branch-PR).

# Notebooks vs scripts division

- Analysis logic lives in **scripts**: importable modules that run headless,
  diff in review and can be tested. Notebooks are **thin presentation
  wrappers** that call those scripts and plot; a notebook cell that computes
  something does not belong in a notebook.
- One shared plotting/analysis helper serves all notebooks; copy-paste cells
  are not maintained in parallel.
- Parameter setups live in one place per experiment family (the runner
  scripts), not duplicated per notebook.

# Model code organisation

- Modules follow the conceptual levels of the model: agents, model,
  landscapes, metrics, visualization, io, sweeps. The experiment interface
  (`generate_data_parametric_exploration`) stays API-stable; interface changes
  are discussed first.
- No dead code kept as an active alternative: commented-out variants are
  catalogued as degrees of freedom (see
  `docs/conceptual_degrees_of_freedom.md`) or removed.

# Reproducibility

- Every RNG is seeded: landscape generation and agent behaviour alike
  (landscape seeds and agent seeds are separate knobs).
- Dependencies are pinned in `requirements.txt`; after any regeneration
  (pipreqs or manual), a miss-check of actual imports against the file is
  mandatory (the 2026-11-13 miss: joblib, pandas).
- Relative data paths only; run outputs carry their parameter tags in the
  filename and land under `data/` (gitignored).
- A reported number is computed, not remembered; report the run that
  produced it.

# File hygiene

- Filenames: no spaces, `snake_case`; record naming follows the
  `AdmiralsLog/` charter.
- Generated outputs are never hand-edited; notebooks are saved without stored
  outputs where the tooling allows.
- One smoke test (seeded run asserting the datacollector schema and run
  reproducibility) guards the model core.
