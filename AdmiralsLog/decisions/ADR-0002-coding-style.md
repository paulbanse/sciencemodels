---
title: "ADR-0002 - Coding style and reproducibility (notebooks vs scripts)"
author: "Jakob Kapeller"
date: "2026-11-13"
tags: [adr, coding-style, reproducibility, notebooks, scripts]
project: "sciencemodels"
---

- **Status:** proposed (drafted 2026-11-13 from the owner's request for general coding principles; reworked the same day per the owner's correction that the principles belong as ADRs in `decisions/`, not in a `rules/` folder; acceptance pending a ruling on this text)
- **Date:** 2026-11-13
- **Supersedes:** -
- **Related:** ADR-0001, `docs/conceptual_degrees_of_freedom.md`, `mesa_model.py`, `runner.py`, `bulk_runner.py`
- **Spawned:** the code-restructure workplan item (this ADR is its specification)

## Context

The owner asked for general principles of code organisation, "the usual
notebooks vs scripts division for instance", to govern the project before the
workplan starts. The current code base violates this division: analysis logic
lives only in notebooks, one plotting helper is copy-pasted across five of
them, and `mesa_model.py` mixes six concerns in 905 lines. The restructure
direction (package split along the conceptual levels, `scripts/` entry
points, thin notebooks, seeded smoke test, one branch-PR) was ruled at the
2026-11-13 onboarding battery; this ADR fixes the principles it must obey.

## Decision

Standing clauses testable against any file:

- **Notebooks vs scripts division.** Analysis logic lives in scripts:
  importable modules that run headless, diff in review and can be tested.
  Notebooks are thin presentation wrappers that call those scripts and plot;
  a notebook cell that computes something does not belong in a notebook. One
  shared plotting/analysis helper serves all notebooks; copy-paste cells are
  not maintained in parallel. Parameter setups live in one place per
  experiment family (the runner scripts), not duplicated per notebook.
- **Model code organisation.** Modules follow the conceptual levels of the
  model: agents, model, landscapes, metrics, visualization, io, sweeps. The
  experiment interface (`generate_data_parametric_exploration`) stays
  API-stable; interface changes are discussed first. Commented-out
  alternatives are catalogued as degrees of freedom (see
  `docs/conceptual_degrees_of_freedom.md`) or removed; dead code is not kept
  as a silent second implementation.
- **Reproducibility.** Every RNG is seeded: landscape generation and agent
  behaviour alike, as separate knobs. Dependencies are pinned in
  `requirements.txt`; after any regeneration a miss-check of actual imports
  against the file is mandatory (2026-11-13 miss: joblib, pandas). Relative
  data paths only; run outputs carry their parameter tags in the filename and
  land under `data/` (gitignored). A reported number is computed, not
  remembered; the run that produced it is named.
- **File hygiene.** Filenames carry no spaces and use `snake_case`; record
  naming follows the `AdmiralsLog/` charter. Generated outputs are never
  hand-edited. A seeded smoke test asserting the datacollector schema and run
  reproducibility guards the model core.

## Consequences

The restructure workplan item implements these clauses: the package split,
the `scripts/` and `notebooks/` folders, one shared analysis helper and the
smoke test. Until it lands, new analysis code follows the division from its
first line even in the old layout.

## Enforcement

A review that finds computing cells in notebooks, duplicated helpers,
unseeded RNGs or unpinned new dependencies reports the violation in the month
log; the restructure PR is checked against every clause before merge. This
draft stays at the `decisions/` top level until the owner rules on its text.
