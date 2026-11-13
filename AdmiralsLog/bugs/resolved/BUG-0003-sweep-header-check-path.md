---
title: "BUG-0003 - Sweep header check tests the wrong path (filename vs filename + .csv)"
author: "Hermes Agent"
date: "2026-11-13"
tags: [bug, sweeps, io]
project: "sciencemodels"
---

- **Status:** resolved (2026-11-13, direct fix with proof --- no ADR involved)
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

Applied 2026-11-13 (T2 helper round): the header check now tests
`is_non_zero_file(filename + ".csv")` --- the path the writer opens.

## Resolution

Verified first-hand by the oversight session, 2026-11-13: the extended smoke
test's regression assert runs a write (`intention="w"`) followed by an append
(`intention="a"`) into a temporary CSV and confirms exactly 3 lines --- 1
header + 2 data rows (single header, no duplication). The W1 seeded canary
remains bit-stable (`mean_prestige = 0.025399760810316363`,
`0x1.a02651806b020p-6`), so the fix is dynamics-inert.
