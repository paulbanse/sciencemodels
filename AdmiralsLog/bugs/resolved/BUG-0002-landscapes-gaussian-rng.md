---
title: "BUG-0002 - Latent TypeError: landscapes.Gaussian called without rng in longitudinal notebook"
author: "Hermes Agent"
date: "2026-11-13"
tags: [bug, landscapes, notebooks]
project: "sciencemodels"
---

- **Status:** resolved (2026-11-13, direct fix with proof --- no ADR involved)
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

Applied 2026-11-13 (T2 helper round): `rng` gains a `None` default across the
generator family (`randominit`, `Gaussian`, `noisyGaussian`,
`multipleGaussians`), and the notebook call site now passes the cell's own
`rng`.

## Resolution

Verified first-hand by the oversight session, 2026-11-13: `Gaussian(5, 5, 10,
{"seed": 4, "std": 5})` returns `np.float64(0.00550470835249002)` without
error; `notebooks/longitudinalcomparisions.ipynb` cell 6 executes verbatim
(`M.sum() = 10.132118364232971`); the extended smoke test carries a
regression assert (4-argument call must return a finite float) and passes;
`sciencemodels/landscapes.py` remains CRLF on all 68 lines (byte-checked).
