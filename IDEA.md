---
title: "IDEA - sciencemodels"
author: "Jakob Kapeller"
date: "November 2026"
tags: [idea, charter, sciencemodels, abm, proxy-failure]
project: "sciencemodels"
---

Here is a project to model scientific exploration: an agent-based model (Mesa)
of scientists searching an epistemic landscape, studying the interplay of
quality and prestige, with planned extensions to collaborator choice under
proxy failure. Everything under `/workspace/git/sciencemodels` (host:
`~/Git/sciencemodels`) belongs to it. This file is the authoritative
orientation document; `AdmiralsLog/decisions/` is authoritative for anything
already decided.

# Purpose and Scope

- The project builds and studies an agent-based model of scientific
  exploration: scientists move on a toroidal epistemic landscape, harvest
  knowledge, accrue prestige and visibility, and the model measures how far
  prestige tracks merit. The research question is proxy failure in science,
  currently bracketed to the quality-prestige interplay, with collaborator
  choice as the planned extension (see `AdmiralsLog/way_forward.md`).
- Belongs here: the model code and runners, analysis notebooks and scripts,
  the background-literature corpus (`context/`), documentation (`docs/`), and
  the project records (`AdmiralsLog/`, which also holds the governance ADRs).
- Does not belong here: the empirical data and analyses of the sibling project
  `/workspace/git/niklasklann-project` (referenced read-only; connection memo
  in `docs/connections_niklas.md`); container-wide environment facts
  (`/workspace/AGENTS.md`).
- Current state: the November 2026 exploration round is distilled in
  `AdmiralsLog/way_forward.md` and `AdmiralsLog/HANDOVER-2026-11-14.md`; status details live in `AdmiralsLog/`, never here.

# Agent Behaviour

- Autonomy (ruled 2026-11-13): experiments, sweeps, analysis runs and document
  authoring proceed without asking. Code restructuring and standing decisions
  ask first. Git: commit locally if it helps the task; never push.
- Helper dispatch follows the scenario the owner prescribes in each request
  (2026-11-13); there is no standing handover rule.
- Session start: read this file, `AdmiralsLog/README.md`, the current month's
  log and the ADRs touching the area; report purpose, state of the workplan,
  open decisions and one proposed next action before acting.
- Never: push to any remote (pushing is the owner's action from the host);
  modify existing files in `context/` (frozen); overwrite run outputs of
  another session; hand-edit generated artifacts; report a number not
  actually computed (evidence-first).
- Documents follow `md-style` / `md-style-corrections`; records are MO-docs,
  first-level plans and reviews are HR-docs with a revision vehicle and a
  compiled PDF. English is the document language.

# Folder Structure

| Path | What it is | Status |
| --- | --- | --- |
| `sciencemodels/` | Model package: agents, model, landscapes, metrics, viz, io, plotting | live (restructured 2026-11-13) |
| `scripts/` | Entry points: `runner.py` (single animated run), `bulk_runner.py` (parametric sweep) | live |
| `notebooks/` | Thin presentation notebooks over `sciencemodels.plotting` | live |
| `tests/` | `test_smoke.py`, the seeded canary | live |
| `old_experiments/` | One legacy notebook from earlier experiment generations | archive |
| `context/` | Background literature corpus with `FROZEN.md` | frozen, read-only (see Frozen Zones) |
| `data/` | Run outputs (CSV), gitignored | generated |
| `docs/` | Documentation and supplementary documents | live |
| `drafts/` | Paper drafts and manuscript sketches (`drafts/archive/` holds superseded drafts) | live |
| `output/reports/` | Round reports and generated outputs (owner directive 2026-11-14) | live |
| `AdmiralsLog/` | Records: logs, decisions, bugs, ideas, unexplored, proposals, archive | live |
| `AdmiralsLog/contributions/` | External contribution boundary (Franz, Paul, Niklas material) | live |
| `helpers/` | Supporting tooling (created when its first file arrives) | planned |

# Frozen Zones

Read-only for humans and agents alike; the convention (`FROZEN.md` marker,
never edit in place) is defined in `/workspace/AGENTS.md`, the inventory is
here.

| Path | Why | Authorisation |
| --- | --- | --- |
| `context/` | Declared frozen 2026-09-22 by Jakob Kapeller / Niklas Klann (`context/FROZEN.md`) | Additive-only new files under explicit user authorization; edits to existing files need both named authors |

# Software and Environment

- Python 3.12 (host Mac); dependencies pinned in `requirements.txt` (pip).
  Project environment: `.venv` on the host, `/opt/hermes/.venv` in the
  container (mesa 3.2.0, numpy 2.3.2, scipy 1.16.1, matplotlib 3.10.6;
  joblib and pandas added 2026-11-13 as `>=` floors after a pipreqs miss).
- Run: `python scripts/runner.py` (single animated run, matplotlib UI);
  `python scripts/bulk_runner.py` (parametric sweep via joblib `Parallel`,
  writes run-tagged CSVs under `data/<folder>/`). Smoke test:
  `python tests/test_smoke.py`.
- Notebooks via Jupyter; analysis reads CSVs from `data/`.
- Data is tracked: none. `data/`, `*.csv`, `*.pdf`, `__pycache__/` are
  gitignored; the literature corpus and documentation are tracked.
- Required environment variables or credentials: none.
- Container-wide inventory (Nix, Julia, uv) deliberately not repeated; it
  lives in `/workspace/AGENTS.md`.

# Rules for Folders, Files and Repos

- Git: `origin` is `https://github.com/paulbanse/sciencemodels.git` (a
  collaborator's GitHub). Agents commit locally under the identity their
  environment configures and **never push**; pushing is the owner's action
  from the host. Commit by pathspec in shared trees; re-check
  `git status` immediately before committing; never switch branches unasked.
  Current working branch is `jokoexplore` (exploration round); the default
  branch is `main`.
- Frozen zones: as above; `context/` changes follow their authorisation row.
- Document placement: plans and reviews in the `AdmiralsLog/` first level
  (HR-docs), records in `AdmiralsLog/` subfolders (MO-docs), documentation in
  `docs/` (HR-docs; long-standing conceptual documents only --- plan-like
  documents move to the `AdmiralsLog/` first level at the latest when a
  workplan supersedes them, owner ruling 2026-11-13); round reports and
  generated outputs in `output/reports/` (owner directive 2026-11-14); governance policy lives
  in the ADRs of `AdmiralsLog/decisions/` (normative, drafted as ADR-0001 and
  ADR-0002).
- Naming: no spaces in filenames; record ids per the `AdmiralsLog/` charter;
  run outputs carry their parameter tags in the filename.
- Generated artifacts: run outputs (`data/`) and compiled PDFs are never
  committed (gitignored); generated reports are never hand-edited.

# AdmiralsLog Usage

- All plans live in the first level (workplans, reviews, summaries, guides);
  no plan lives in a subfolder. The index is `AdmiralsLog/README.md`.
- Traceability: a record that produced something downstream notes it in
  `Spawned:`; the produced record cites the origin under `Related:`.
- Triggers: an ADR for a conceptual choice discussed and ruled; a BUG for a
  pinned fault; a DE for an abandoned approach; an IE for a captured idea; a
  UE for a parked strategic item; a P for a route or direction awaiting the
  owner's ruling; an archive move when a first-level document is superseded.
- Ids are consecutive and never reused; counters currently: next ADR-0003,
  BUG-0007, CONT8, DE-0001, IE-0007, UE-0004, P-0012.
- Every session leaves one entry in `AdmiralsLog/logs/<YYYY-MM>.md`
  (append-only).

# Open Questions and Known Gaps

- Branch and PR policy for the ruled code restructure: which branch it lands
  on and who opens the pull request (agents never push). One coordinated
  branch-PR was the ruled direction.
- `requirements.txt` exact pins for `joblib` and `pandas` should be
  regenerated from the host environment (currently `>=` floors).
- Model-level caveats (time-scale mapping, identifiability, the soft empirical
  wedge) are tracked in `AdmiralsLog/way_forward.md` and deliberately not duplicated
  here.
