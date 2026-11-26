---
title: "ADR-0004 - Co-Command: Two Admirals (Jakob Kapeller & Paul Banse)"
author: "Jakob Kapeller"
date: "2026-11-27"
tags: [adr, governance, co-command, two-admirals, ownership, coordination]
project: "sciencemodels"
---

- **Status:** accepted (ruled 2026-11-27)
- **Date:** 2026-11-27
- **Supersedes:** -
- **Related:** `IDEA.md`, `AGENTS.md`, `AdmiralsLog/decisions/accepted/ADR-0001-multiagent-coordination.md`
- **Spawned:** `IDEA.md` co-command wording update

## Context

`IDEA.md` and `ADR-0001` were originally drafted referring to "the owner" in
the singular, describing `origin` (`paulbanse/sciencemodels`) as "a
collaborator's GitHub," and placing Paul Banse's material under the external
contribution boundary (`AdmiralsLog/contributions/`). In reality,
`sciencemodels` is jointly led by **two Admirals** at Starfleet Command:
**Jakob Kapeller** and **Paul Banse**.

Because multiple AI agents and helpers operate across this shared repository
on behalf of either Admiral, ambiguous ownership wording creates avoidable
friction (e.g., treating one Admiral's commits as "external contributions" or
overwriting authorship/review records).

## Decision

1. **Equal Co-Command (Two Admirals).** `sciencemodels` recognizes **two
   Admirals** with equal standing and authority: **Jakob Kapeller** and
   **Paul Banse**. Both hold full owner authority over conceptual rulings,
   ADRs, `AdmiralsLog/` records, `drafts/`, code architecture, and pushing to
   `origin`.
2. **First-class records, not external contributions.** Commits, ideas,
   reviews, and proposals authored by either Admiral belong directly in the
   live project records (`AdmiralsLog/ideas/`, `proposals/`, `decisions/`,
   `drafts/`), never in the external contribution boundary
   (`AdmiralsLog/contributions/`, which remains for external/sibling-project
   material such as Franz or Niklas artifacts).
3. **Agent protocol under two Admirals.**
   - Every AI agent working in this repository treats rulings from **either**
     Admiral as binding.
   - An agent never silently overwrites, deletes, or re-attributes an edit or
     record authored by the other Admiral.
   - If an instruction from one Admiral conflicts with a standing ruling or
     recent edit by the other Admiral, the agent **flags the divergence
     explicitly** for joint resolution rather than resolving it silently.
   - Both Admirals' parallel helper sessions follow `ADR-0001`'s worktree and
     `logs/inbox/` isolation rules.

## Consequences

- `IDEA.md` updates its references from singular "owner" / "collaborator's
  GitHub" to the two-Admiral co-command model while keeping the agent push ban
  intact (only the two Admirals push from their hosts).
- `ADR-0001` remains authoritative for technical git, worktree, and execution
  hygiene; `ADR-0004` is authoritative for command authority.

## Enforcement

Any agent session that silently overwrites an Admiral's record or treats Paul
Banse's commits as external/unprivileged contributions violates this ADR and
must be flagged in the month log.
