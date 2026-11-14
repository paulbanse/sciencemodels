---
title: "BUG-0005 - bulk_runner setup comment places the 6-Gaussian preset wrongly"
author: "Hermes Agent"
date: "2026-11-14"
tags: [bug, comments, bulk-runner]
project: "sciencemodels"
---

- **Status:** open
- **Date:** 2026-11-14
- **Related:** `scripts/bulk_runner.py:22`
- **Spawned:** none

## Symptom

The setup comment in `scripts/bulk_runner.py` places the 6-Gaussian preset
at setup 2, while the code at that case builds 2 Gaussians (the 6-Gaussian
preset lives at setups 4-5). The comment invites a misread of which landscape
a setup number produces --- the same comment-vs-code drift as the former
runner.py "6 gaussians" note (fixed 2026-11-13).

## Cause

Legacy comment drift, pinned by the shim/catalogue round (2026-11-14).

## Fix

Not yet applied. Candidate: correct the comment to match the case values.
