---
title: "Rules - Multiagent coordination"
author: "Jakob Kapeller"
date: "2026-11-13"
tags: [rules, coordination, multiagent, git]
project: "sciencemodels"
---

Normative policy for all humans and AI agents working in this repository.
Decided with the owner 2026-11-13 (onboarding round); standing clauses are
proposed for ruling as ADR-0002.

# One records vocabulary

- `IDEA.md` points, `AdmiralsLog/` records decide, logs narrate. No parallel
  records system is created; no document restates another's content.
- Settled decisions are binding (`decisions/accepted/` and
  `decisions/rejected/`); a task colliding with one names it instead of
  silently deviating.

# Session write discipline

- Every session that changes documents leaves one entry in
  `AdmiralsLog/logs/<YYYY-MM>.md`: intent, documents touched, decisions taken,
  dead ends, next step. Entries are append-only.
- An ADR is drafted only for a conceptual choice the owner actually discussed
  and ruled on; unruly drafts sit at the `decisions/` top level as
  `Status: proposed`. Battery answers and chat instructions are recorded as
  dated words in the log, not dressed up as ADRs.
- Records that produce something downstream carry `Spawned:`; the produced
  record cites the origin under `Related:`.

# Shared-tree git discipline

- `origin` is a collaborator's GitHub (`paulbanse/sciencemodels`). Agents
  commit locally as `perplexio <jakob.kapeller@jku.at>` and **never push**;
  pushing is the owner's action from the host. If a commit is ready to
  publish, say so; do not push.
- Re-check `git status` immediately before every commit (other sessions may
  be mid-flight); commit by pathspec naming only this session's paths.
- No branch switches without asking. Never rewrite history.
- Nothing is moved, renamed or deleted without explicit approval.

# Concurrent-run safety

- Sweeps and runs write run-tagged outputs under `data/` (gitignored); a run
  never overwrites another run's outputs, and generated outputs are never
  hand-edited.
- Committed artifacts (documents, code) change only through an explicit,
  attributable edit.

# Foreign agents and contributors

- A foreign agent is greeted by `AGENTS.md`, which hands it to `IDEA.md`;
  collaborators' material lands in `AdmiralsLog/contributions/`, named by
  source and date.
- Frozen zones (`context/`) are additive-only under explicit user
  authorization; existing files are never modified.
