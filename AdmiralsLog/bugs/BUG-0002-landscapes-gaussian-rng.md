---
title: "BUG-0002 - Latent TypeError: landscapes.Gaussian called without rng in longitudinal notebook"
author: "Hermes Agent"
date: "2026-11-13"
tags: [bug, landscapes, notebooks]
project: "sciencemodels"
---

- **Status:** open
- **Date:** 2026-11-13
- **Related:** `notebooks/longitudinalcomparisions.ipynb` (cell 6), `sciencemodels/landscapes.py` (Gaussian signature)
- **Spawned:** none

## Symptom

`Gaussian` is defined as `Gaussian(posX, posY, size, args, rng)` but
`notebooks/longitudinalcomparisions.ipynb` cell 6 calls it with four
arguments (no `rng`), which raises `TypeError` when that cell executes.

## Cause

Pre-existing quirk of the legacy notebook, preserved verbatim by the W2
restructure (pure-move rule); `Gaussian` never uses the passed `rng`
internally (it rebuilds from `args["seed"]`), so the parameter is a
decoy in this call path.

## Fix

Not yet applied. Candidate: give `rng` a default (`rng=None`) in
`Gaussian`, or pass `None` at the call site.
