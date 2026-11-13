---
title: "ADR-0001 - Multiagent coordination"
author: "Jakob Kapeller"
date: "2026-11-13"
tags: [adr, coordination, multiagent, git, autonomy, records]
project: "sciencemodels"
---

- **Status:** proposed (drafted 2026-11-13 from the onboarding battery ruling; reworked the same day per the owner's correction that the general principles belong as ADRs in `decisions/`, not in a `rules/` folder; acceptance pending a ruling on this text)
- **Date:** 2026-11-13
- **Supersedes:** -
- **Related:** `IDEA.md`, `AGENTS.md`, `AdmiralsLog/README.md`, ADR-0002
- **Spawned:** the code-restructure workplan item (direction ruled 2026-11-13, plan pending)

## Context

Several humans (Jakob Kapeller, Franz, Paul, Niklas) and several AI agents
of different owners work in this shared repository. The onboarding battery of
2026-11-13 ruled the records layout and the autonomy level; the owner then
corrected the same day that the general principles are to live as ADRs in
`AdmiralsLog/decisions/` and not in a separate `rules/` folder (retired as
redundant). This ADR is the coordination contract; coding conventions are
ADR-0002.

## Decision

Standing clauses testable against any file or session action:

- **One records vocabulary.** `IDEA.md` points, `AdmiralsLog/` records decide,
  logs narrate. `AdmiralsLog/` is the sole records container with the ruled
  subfolder set: `logs/`, `decisions/{accepted,rejected}/`,
  `bugs/{resolved,deadends}/`, `ideas/{embraced,rejected}/`,
  `unexplored/{beneficial,dismissed}/`, `proposals/`, `archive/`,
  `contributions/`, `skills/`. No parallel records system; no policy outside
  the ADRs of `decisions/` binds anything. Plans live in the `AdmiralsLog/`
  first level; documentation lives in `docs/`.
- **Session write discipline.** Every session that changes documents leaves
  one entry in `AdmiralsLog/logs/<YYYY-MM>.md` (append-only). An ADR is
  drafted only for a conceptual choice the owner actually discussed and
  ruled; unruly drafts sit at the `decisions/` top level as
  `Status: proposed`; battery answers and chat instructions are recorded as
  dated words in the log, not dressed up as ADRs. Records carry `Spawned:`
  downstream and `Related:` upstream.
- **Git discipline.** `origin` is a collaborator's GitHub
  (`paulbanse/sciencemodels`). Agents commit locally as
  `perplexio <jakob.kapeller@jku.at>` when it helps the task and **never
  push**; pushing is the owner's action from the host. In shared trees:
  re-check `git status` immediately before committing, commit by pathspec
  naming only this session's paths. No branch switches without asking; no
  history rewrite; nothing moved, renamed or deleted without explicit
  approval.
- **Autonomy.** Experiments, sweeps, analysis runs and document authoring
  proceed without asking. Code restructuring and standing decisions ask
  first: the restructure direction ruled 2026-11-13 (package split along the
  conceptual levels, `scripts/` entry points, thin notebooks, seeded smoke
  test, one branch-PR) is an act for the workplan, not authorization to move
  files before the plan exists.
- **Concurrent-run safety.** Sweeps and runs write run-tagged outputs under
  `data/` (gitignored); a run never overwrites another run's outputs;
  generated outputs are never hand-edited; committed artifacts change only
  through explicit, attributable edits.
- **Foreign agents and frozen zones.** A foreign agent is greeted by
  `AGENTS.md`, which hands it to `IDEA.md`; collaborators' material lands in
  `AdmiralsLog/contributions/`, named by source and date. The frozen zone
  `context/` is additive-only under explicit user authorization; existing
  files are never modified.

## Consequences

The coordination contract is one document; the records structure it names was
created at onboarding with stubs in every folder. The push ban means the
owner carries every branch publication; agents report when a local commit is
ready to be pushed. The restructure waits for its workplan item, which this
ADR spawns.

## Enforcement

A session that pushes, forks the records vocabulary, restructures code
without the workplan item, or edits frozen files violates this ADR; the
incident is noted in the month log and the change is reported for reversal.
This draft stays at the `decisions/` top level until the owner rules on its
text.
