---
title: "BUG-0003 - Sweep header check tests the wrong path (filename vs filename + .csv)"
author: "Hermes Agent"
date: "2026-11-13"
tags: [bug, sweeps, io]
project: "sciencemodels"
---

- **Status:** open
- **Date:** 2026-11-13
- **Related:** `sciencemodels/model.py` (`generate_data_parametric_exploration`), `sciencemodels/io.py` (`is_non_zero_file`)
- **Spawned:** none

## Symptom

`generate_data_parametric_exploration` computes
`needs_header = not is_non_zero_file(filename)` but the writer opens
`filename + ".csv"`. The check tests a path that is never written, so it is
always true and append mode (`intention="a"`) writes a duplicate header row
into an existing CSV instead of appending cleanly.

## Cause

Pre-existing quirk, preserved verbatim by the W2 restructure (pure-move
rule): the suffix `.csv` is applied at the `open()` call but not at the
header check.

## Fix

Not yet applied. Candidate: check `is_non_zero_file(filename + ".csv")`.
