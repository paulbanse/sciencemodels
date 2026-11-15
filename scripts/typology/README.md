# scripts/typology — Path D typology round (P-0005, 2026-11-15)

Driver-side implementation of the Path D typology experiment: distraction
(the native $(1-c)$ baseline), destruction ($\gamma$, $AP_F \dashv G$) and
counterforce ($\kappa$, $P \to G$) as crossable perception regimes of the
ABM. `sciencemodels/*.py` is NOT touched by this round; the two channels
live in a clean subclass pair.

## One home per measure (ADR-0002)

| File | Owns |
| --- | --- |
| `typology_layers.py` | the two channels (`TypologyScientist`, `TypologyModel`), the source divergence canary, `compare_rows`, the bit-identity proof, the binding/control probe |
| `run_typology_v1.py` | the five gates, the 24-cell grid (atomic per-cell checkpoints, `n_jobs=3`), consolidation, manifest/design |
| `analyse_typology_v1.py` | the 4-quadrant typology table, curiosity/entry gradients, the kappa-inert control arm, paired factorial contrasts with bootstrap CIs |
| `make_figures.py` | the four house-style figures (PDF + 300 dpi PNG) |
| `make_doc.py` | `output/reports/typology_v1.md` (every number from the artefacts) |

## Regeneration

```{.bash}
MPLBACKEND=Agg /opt/hermes/.venv/bin/python tests/test_smoke.py          # canary
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/typology/run_typology_v1.py --gates-only
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/typology/run_typology_v1.py --skip-gates
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/typology/analyse_typology_v1.py
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/typology/make_figures.py
MPLBACKEND=Agg /opt/hermes/.venv/bin/python scripts/typology/make_doc.py
zsh /workspace/git/ai/IT/scripts/compile-here.zsh output/reports/typology_v1.md
```

A restart skips cells whose verified state file is present (20 seed rows +
header); the `--skip-gates` launch re-verifies the recorded gates and
refuses to run if the shared tree's `sciencemodels/` hashes moved since
the gates ran (re-run `--gates-only` then).

## Key pins (full list: `data/typology_v1/design.json`, report section Design)

- Destruction: `eps_eff = eps * m_t(candidate)**gamma`, `m_t` = scientists
  on the candidate tile at evaluation time.
- Counterforce: `eps_i = eps_0 / (1 + kappa * P_sup)` for newborns;
  `P_sup` = supervisor's entry-stock prestige at the birth step, fixed for
  life; founders kappa-inert; kappa inert by construction at g_entry=0.0
  (the grid's built-in control arm, asserted bit-identical).
- Newborn interception defines the mangled name
  `_MyModel__generate_new_agents` explicitly (a normally named override is
  silently invisible to `MyModel.end_loop_update`).
- Model-internal only, NOT calibrated (owner ruling 2026-11-13).

## Artefacts

All outputs land in `data/typology_v1/` (cells/, cells/agents/,
runs.csv, typology_table.csv, contrast_table.csv, control_checks.json,
figures/, gates.txt, canary_smoke.txt, manifest.json, design.json).
`output/reports/typology_v1.md` (+ `.pdf`) is the round's document.
