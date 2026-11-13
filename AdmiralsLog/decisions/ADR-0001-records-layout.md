---
title: "ADR-0001 - Records layout and authority split for the multi-player environment"
author: "Jakob Kapeller"
date: "2026-11-13"
tags: [adr, admiralslog, layout, governance]
project: "sciencemodels"
---

- **Status:** proposed (drafted 2026-11-13 from the onboarding battery ruling; acceptance pending a ruling on this text)
- **Date:** 2026-11-13
- **Supersedes:** -
- **Related:** `IDEA.md`, `AdmiralsLog/README.md`, `rules/`
- **Spawned:** none

## Context

The repository is being onboarded as a multi-player research environment
(human collaborators Franz, Paul, Niklas plus AI agents of different owners).
The onboarding battery of 2026-11-13 confirmed the `AdmiralsLog/` subfolder
set: standard records set plus `contributions/` (research project),
`rules/` (governance policy) and `skills/` (index pointer to the house
conventions).

## Decision

Standing clauses testable against any file or action:

- `AdmiralsLog/` is the sole records container of this project; no parallel
  records vocabulary (`docs/decisions`, `logs/` at root and the like) is
  created.
- The subfolder set is: `logs/`, `decisions/{accepted,rejected}/`,
  `bugs/{resolved,deadends}/`, `ideas/{embraced,rejected}/`,
  `unexplored/{beneficial,dismissed}/`, `proposals/`, `archive/`,
  `contributions/`, `skills/`.
- `rules/` is the sole home of normative governance policy; a policy outside
  `rules/` does not bind. `docs/` holds documentation only. Plans and reviews
  live in the `AdmiralsLog/` first level.
- `contributions/` is the external contribution boundary; foreign material is
  filed there, named by source and date.
- `skills/` carries index pointers only (internal project; no vendored
  snapshots).

## Consequences

The skeleton was created with stub files in every folder (git cannot track
empty directories). Register counters start at ADR-0003, BUG-0001, DE-0001,
IE-0001, UE-0001, P-0001. The placement of `docs/way_forward.md` (plan-like
document in `docs/`) is deliberately NOT settled here; it remains an open
question in `IDEA.md`.

## Enforcement

A session filing a record outside the named folders, or creating a second
records vocabulary, violates this ADR; the record is moved and the incident
noted in the month log. A proposed copy of this ADR stays at the
`decisions/` top level until the owner rules on its text.
