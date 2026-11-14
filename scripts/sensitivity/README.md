# scripts/sensitivity/

Archived sources of the T1 sensitivity round (Path 0), run on 2026-11-13.
These are verbatim copies rescued from the gitignored `data/sensitivity_v1/`
folder: driver code belongs in `scripts/` rather than in `data/` per
ADR-0002, so the snapshots live here and the originals are left untouched.

The round's data output stays in `data/sensitivity_v1/`; its written report is
`output/reports/sensitivity_map_v1.md`.

| File | sha256 (copy) | Bytes | sha256 (original in `data/sensitivity_v1/`) |
| --- | --- | --- | --- |
| `driver_source.py` | `159eade38e914022bd647aa7b660b6c32cd183e66bb3f6ef17666ba6aa436847` | 8828 | `159eade38e914022bd647aa7b660b6c32cd183e66bb3f6ef17666ba6aa436847` |
| `consolidator_source.py` | `6cfd107039862f202a189978a5e5f34ae57607acf3b2ffd1eb286583ccb03ec6` | 4544 | `6cfd107039862f202a189978a5e5f34ae57607acf3b2ffd1eb286583ccb03ec6` |
| `doc_generator_source.py` | `65ade27e763a84724f8171eb0069c6336e4c28ed000e597a8a344f6a1f812416` | 39896 | `65ade27e763a84724f8171eb0069c6336e4c28ed000e597a8a344f6a1f812416` |

Copy and original hash identically for all three files, so the copies are
proven byte-identical to the archived originals.

What each file did:

- `driver_source.py` - the sweep driver. Enumerated the sensitivity cells
  (curiosity, vanguard share, measurement noise, generation), ran the model
  for each cell, and wrote the per-cell trajectory and summary CSVs plus the
  checkpoint and run log.
- `consolidator_source.py` - the consolidation and analysis step. Read the
  per-cell outputs, reduced them to the cell summary and the mean
  trajectories, and produced the comparison quantities the map is drawn
  from.
- `doc_generator_source.py` - the report generator. Turned the consolidated
  tables into the written round report `output/reports/sensitivity_map_v1.md`.
