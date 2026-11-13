---
title: "ADR-0002 - Agent autonomy and git discipline"
author: "Jakob Kapeller"
date: "2026-11-13"
tags: [adr, autonomy, git, multiagent-coordination]
project: "sciencemodels"
---

- **Status:** proposed (drafted 2026-11-13 from the onboarding battery ruling; acceptance pending a ruling on this text)
- **Date:** 2026-11-13
- **Supersedes:** -
- **Related:** `IDEA.md`, `rules/multiagent-coordination.md`, ADR-0001
- **Spawned:** none

## Context

Multiple agents (Hermes, Claude Code, Codex and others) and multiple humans
work in this shared repository. The onboarding battery of 2026-11-13 ruled
the autonomy level: "Experiments, sweeps and documents freely; code
restructuring and standing decisions ask first; commit locally, never push".

## Decision

Standing clauses testable against any session action:

- An agent may run experiments and sweeps, run analysis, and author or revise
  documents without prior confirmation.
- Code restructuring (layout, module splits, renames) and standing decisions
  require asking first. The restructure direction ruled 2026-11-13 (package
  split along the conceptual levels, `scripts/` entry points, thin notebooks,
  seeded smoke test, one branch-PR) is an act for the workplan, not by itself
  an authorization to move files before the workplan exists.
- Git: agents commit locally as `perplexio <jakob.kapeller@jku.at>` when it
  helps the task, by pathspec in shared trees, after re-checking
  `git status`. Agents never push: `origin` is a collaborator's GitHub
  (`paulbanse/sciencemodels`) and pushing is the owner's action from the
  host. No branch switches without asking.
- The frozen zone `context/` is additive-only under explicit user
  authorization; existing files are never modified.

## Consequences

Sessions may produce runs, analyses and documents freely; anything that
changes code structure waits for the workplan item. The push ban means the
owner carries every branch publication; agents report when a local commit is
ready to be pushed.

## Enforcement

A session that pushes, restructures code without the workplan item, or edits
frozen files violates this ADR; the incident is noted in the month log and
the change is reported for reversal. A proposed copy stays at the
`decisions/` top level until the owner rules on its text.
