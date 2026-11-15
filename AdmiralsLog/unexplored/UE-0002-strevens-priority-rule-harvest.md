---
title: "UE-0002 - Strevens (2003) priority-rule (winner-take-all) vs. proportional co-harvest sharing"
author: "Antigravity"
date: "2026-11-15"
tags: [unexplored, priority-rule, strevens, harvest-sharing, herding, path-c]
project: "sciencemodels"
status: unexplored
---

- **Status:** unexplored
- **Date:** 2026-11-15
- **Related:** `docs/related_publications.md` (Strevens 2003; Kitcher 1990), `docs/conceptual_degrees_of_freedom.md`, `AdmiralsLog/contributions/CONT1-depletion-counterforces-refills.md`

## The unexplored path

In `sciencemodels/model.py`, when $m$ scientists occupy the same tile, `knowledge_normalization = "share"` divides the harvested knowledge equally ($h \cdot k_{xy} / m$) or `raw` grants each occupant $h \cdot k_{xy}$. In the classic economics and philosophy of science literature (Merton 1957; Dasgupta & David 1994; Strevens 2003), scientific credit is governed instead by the **Priority Rule** (winner-take-all for the first discoverer, or a tournament prize among co-occupants).

Testing a `priority` harvest regime — where only the first visitor to a tile (or a single tournament winner among co-occupants) receives the novelty prestige credit while latecomers harvest zero epistemic credit — would test Strevens' (2003) hypothesis directly on our spatial grid: **does priority competition endogenously deter visibility herding and improve collective exploration without requiring exogenous sabbatical shocks (`p_reloc`)?**
