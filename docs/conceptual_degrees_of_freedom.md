---
title: "Conceptual Degrees of Freedom: Catalogue of Modelling Assumptions"
version: 3.0
date: 2026-11-13
tags: [sciencemodels, assumptions, sensitivity-analysis, proxy-failure]
project: sciencemodels
revision-vehicle: strikethrough
revision-renders: "md, pdf, html"
---

**Version 3** (November 2026)
**Version 2** (November 2026)
**Version 1** (November 2026)

v3 status note (2026-11-13): every `file:line` reference in the catalogue was
re-anchored against the live tree. The 104 table rows carry 109 reference
callouts (a row that names two files counts twice); each anchor is the line
that holds the token the entry quotes, checked with `grep -n` and re-checked by
`scripts/cdf_reanchor/reanchor_cdf_refs.py`, which fails if an anchor stops
matching. Two rows are marked **unresolvable** in place, with the reason: the
landscape-signature comment block and the `rationRewards` helper no longer
exist anywhere in the package. Rows whose *assumption* the Path 0 round removed
are struck and restated with the rule that replaced them, so the entry documents
the current code rather than a superseded one. Line numbers are as of this
check and will drift with every edit.

Re-verified at 2026-11-14 01:20 against the working tree after a concurrent
round rewrote `sciencemodels/model.py` and `sciencemodels/metrics.py` (shifts
of 1 to 64 lines, non-uniform) and this project's correlation-shim round
edited `sciencemodels/plotting.py`: 44 reference cells were re-resolved, the
row on the plotting horizon was restated because the hard-coded horizon it
described no longer exists, and the verifier now passes on all 109 resolvable
anchors. Anchors describe the working tree at that timestamp, so they match
the next commit that carries those files; if that round lands differently,
re-run `scripts/cdf_reanchor/reanchor_cdf_refs.py` and re-resolve.

The restructure is not a one-to-one rename. Code that the old `mesa_model.py`
held now lives in six modules, so an old line number does not identify its new
module by itself:

| Old module | Live module | What moved there |
|---|---|---|
| `mesa_model.py` | `sciencemodels/model.py` | model class, grid and knowledge initialisation, harvest, newcomer generation, new questions, sweep entry point |
| `mesa_model.py` | `sciencemodels/agents.py` | the `Scientist` class: construction, reward, step rule, prestige accrual |
| `mesa_model.py` | `sciencemodels/metrics.py` | `local_merit`, ideal merit, correlation reporter, milestone goals |
| `mesa_model.py` | `sciencemodels/io.py` | CSV writing, report assembly, header extraction |
| `mesa_model.py` | `sciencemodels/viz.py` | 3D surface, landscape plot, step animation |
| `landscapefunctions.py` | `sciencemodels/landscapes.py` | landscape and curiosity generators, line for line |
| `runner.py` | `scripts/runner.py` | interactive entry point and its parameter block |
| `bulk_runner.py` | `scripts/bulk_runner.py` | bulk sweep entry point and its parameter block |
| notebook plotting cells | `sciencemodels/plotting.py` | shared `plot_smt`, `read_csv`, correlation helpers |

v2 status note (2026-11-13): entries cited pre-restructure paths
(`mesa_model.py` etc.); the code moved to the `sciencemodels/` package the
same day (content preserved, see `5bd5c83`), so references map by module.
The Path 0 hygiene round (`33421ff`) resolved several entries: prestige
accrual now takes the harvested flow $h \cdot k$ (legacy via
`prestige_accrual: 'post_tile'`); `new_questions` is split from
`knowledge_normalization`; milestones are renamed `harvested_10_step` /
`harvested_25_step`; newcomer curiosity runs through
`agent_generation_function`; founder tiles count as explored; the vanishing
factor is unified at 0.9 (the disagreement rows below are stale and struck).
~~A full re-anchor of `file:line` references to the package layout is pending.~~

# Purpose

This catalogue collects every place where the code fixes a parameter value, threshold, functional form, initialization rule or normalization. Each entry gives the `file:line` reference, what is assumed, the current value or formula, and what the assumption conceptually controls. Entries marked *dead / commented* are alternatives left in the code; they document intended degrees of freedom even where currently inactive.

Sections follow the causal chain of the model: epistemic environment → agent decision-making → reward and proxy structure → prestige dynamics → demographics → measurement and reporting. The final section ranks the levers for the paper question (proxy failure via collaborator choice).

# Epistemic environment

## Grid and topology

| Reference | Assumption | Current value / form | Conceptual role |
|---|---|---|---|
| `sciencemodels/model.py:121,147` | World is a single square grid of side `grid_size` | constructor parameter `grid_size`, recorded as `self.size` | Total size of the knowledge space; fixes ratio of agents to tiles |
| `sciencemodels/model.py:46` | Grid is a torus | `MultiGrid(self.size, self.size, torus=True)` | No borders or center; every tile topologically equivalent |
| `scripts/runner.py:51`, `scripts/bulk_runner.py:62` | Grid side 20 | `grid_size: 20` (400 tiles) | Absolute scale of exploration tasks in all standard runs |
| `sciencemodels/model.py:47-64` | Knowledge lives in property layers | `knowledge`, `previous_knowledge`, `initial_knowledge`, `explored` | Separates the epistemic state from agents; enables snapshot comparisons |

## Landscape generation

| Reference | Assumption | Current value / form | Conceptual role |
|---|---|---|---|
| `sciencemodels/landscapes.py:13-15` | `randominit`: i.i.d. uniform tile values | $k_{xy} \sim U(0,1)$ | Null landscape: no structure, no correlation between neighbors |
| `sciencemodels/landscapes.py:18-29` | `Gaussian`: one bump, separable in x and y, drawn on the torus circle | $k_{xy} = f_{vM}(x;\mu_x,\kappa)\cdot f_{vM}(y;\mu_y,\kappa)$ with von Mises pdfs; $\mu_x,\mu_y$ drawn from `args["seed"]` | A single coherent research area; separability rules out rotated/elliptical ridges |
| `sciencemodels/landscapes.py:23` | Landscape is deterministic given the seed | `rng_fix = np.random.default_rng(args["seed"])`; per-cell `rng` argument ignored by `Gaussian` | Landscapes are reproducible and comparable across agent populations |
| `sciencemodels/landscapes.py:19-22` | Width parameter `std` (tile units) maps to von Mises concentration | `kappa = 1 / (args["std"] / size)`, i.e. $\kappa = size/std$, unless `kappa` given directly | How broad a knowledge peak is relative to the grid |
| `sciencemodels/landscapes.py:41-53` | `multipleGaussians`: unnormalized sum of several bumps | `sum` of `number_gaussians` Gaussians with sub-seeds drawn from the master seed | Multi-topic landscape; unnormalized sum means narrow peaks can tower over broad ones |
| `scripts/runner.py:19-21,31,34` | Landscape presets | setup 1: `std: 2, number_gaussians: 2`; setup 2: `std: 5, number_gaussians: 4`; ~~comment says "6 gaussians" — comment and value disagree~~ the runner's setup comment now agrees with the value (4 gaussians), and the stale "6 gaussians" comment survives in `scripts/bulk_runner.py:22`, whose setup 2 builds 2 Gaussians | Concrete landscape choices for interactive runs |
| `scripts/bulk_runner.py:21-23,34,37` | Landscape presets for bulk runs | `std: 5` with `number_gaussians: 2` (setups 2-3) or `6` (setups 4-5) | Landscape families compared in the large sweep; the setup comment on line 22 misplaces the 6-Gaussian preset at setup 2 |
| `sciencemodels/landscapes.py:32-38` | `noisyGaussian`: uniform noise floor plus amplified Gaussian | $k_{xy} = U(0,1) + prop\_random \cdot G(x,y)$ | Landscape where signal must be teased out of noise; `prop_random` sets signal-to-noise |
| `scripts/runner.py:38`, `scripts/bulk_runner.py:41` | Signal amplification | `prop_random: 5` | Bump is 5x the noise floor; ordering of peaks barely legible |
| `scripts/runner.py:40`, `scripts/bulk_runner.py:43` | Curiosity spread passed via generation params | `std_curiosity: 0.1` | Coupled to the Beta curiosity generator (see decision-making) |
| **unresolvable** (block deleted) | Landscape functions share one signature | `(posX, posY, size, args, rng)` — the comment block that documented it (`landscapefunctions.py:56-64`) was not carried into `sciencemodels/landscapes.py`; lines 56-64 there now hold the curiosity helpers, and the convention survives only implicitly in the four `def` lines (13, 18, 32, 41) | Any callable with this signature can be swapped in — the landscape family is a free design axis |

## Knowledge initialization and normalization

| Reference | Assumption | Current value / form | Conceptual role |
|---|---|---|---|
| `sciencemodels/model.py:80-87` | Two mutually exclusive normalization schemes | min-max rescale to $[0.05, 0.95]$ when `knowledge_normalization == "minmax"`, else share-of-total $k_{xy}/\sum k$ | Absolute scale of knowledge values, hence of rewards and prestige |
| `sciencemodels/model.py:85-87` | Floor and ceiling of min-max rescale | `* 0.95 + 0.05` | No tile is ever fully empty or maximal; 0.05 floor keeps novelty and reward bounded away from zero |
| `sciencemodels/model.py:80-83` | Share-of-total alternative | `val / self.total_initial_knowledge` | Makes tile values sum to 1; small absolute rewards |
| `sciencemodels/model.py:146,161,343` | ~~One flag controls two unrelated things~~ two flags are now separate: `knowledge_normalization` selects the scheme, `new_questions` the refill | `self.new_questions = new_questions` (144), `self.knowledge_normalization = ...` (159), `if self.new_questions:` (320) | ~~Conceptual conflation: normalization cannot be varied independently of refills~~ split on 2026-11-13 (Path 0); normalization is now an independent lever |
| `sciencemodels/model.py:96` | Initial total knowledge reset after normalization | `self.total_initial_knowledge = np.float64(1.0)` | Makes `percentage_knowledge_harvested` a plain share |
| `sciencemodels/model.py:89-94` | Initial state snapshot | `initial_knowledge` and `previous_knowledge` copy the normalized values | Basis for weighted-exploration and ideal-merit benchmarks |

## Harvest and knowledge depletion

| Reference | Assumption | Current value / form | Conceptual role |
|---|---|---|---|
| `sciencemodels/model.py:303-307` | Harvesting is multiplicative depletion | $k_{xy} \leftarrow (1-h)\,k_{xy}$ per agent visit | Diminishing returns at a tile; crowding costs |
| `scripts/runner.py:50`, `scripts/bulk_runner.py:61` | Harvest rate | `harvest: 0.02` | Speed of depletion vs. exploration; strength of competition over good tiles |
| `sciencemodels/model.py:303-307,342-349` | No regrowth except the refill rule | only `_generate_new_questions` restores knowledge | Knowledge is a depletable stock; the landscape is effectively finite |
| `sciencemodels/model.py:283-286` | Population-relative knowledge normalizer | `avg_current_agent_knowledge = mean(agents' last_tile_knowledge)` | Reward is relative to what the community currently sits on, not to an absolute scale |

## New-questions refill

| Reference | Assumption | Current value / form | Conceptual role |
|---|---|---|---|
| `sciencemodels/model.py:342-349` | Refill trigger threshold | occupied tiles with $k < 0.025$ get $k \leftarrow U(0,1)$ | How depleted a tile must be before it becomes a "new question"; 0.025 is a hard-coded constant |
| `sciencemodels/model.py:344` | Only occupied tiles are refilled | `occupied_pos = list(set(agent positions))` | New questions appear only where scientists already are — no exogenous discovery elsewhere |

# Agent decision-making

## Initialization

| Reference | Assumption | Current value / form | Conceptual role |
|---|---|---|---|
| `sciencemodels/agents.py:41` | Default age at construction | `self.age = 23` | Starting career stage (implies PhD-start metaphor) |
| `sciencemodels/model.py:105` | Initial agents' ages drawn uniformly | `rng.choice(np.arange(23, 50))` → integers 23-49 | Age heterogeneity at $t=0$ |
| `sciencemodels/agents.py:175` | Age increments each step | `self.age += 1` | Time in the model = career time |
| `sciencemodels/agents.py:41,175` (class 21-183) | Age has no functional effect | no death, retirement, or age-dependent behavior anywhere | *Dead lever*: age is recorded but does nothing |
| `sciencemodels/model.py:104` | Initial positions uniform over the grid | `coords = (rng.integers(0, size), rng.integers(0, size))` | No correlation between starting position and landscape |
| `sciencemodels/model.py:103` | Homogeneous noise sensitivity | all agents get the model-level `epsilon` | One noise level for the whole population |
| `sciencemodels/agents.py:38-40` | Zero initial prestige stocks | `prestige = prestige_visibility = prestige_vanishing = 0` | Everyone starts equal in the prestige race |
| `sciencemodels/model.py:129-131` | Curiosity constant across agents by default | `agent_generation_function = lambda cur, rng, params: cur` | Default population is homogeneous in curiosity |
| `sciencemodels/landscapes.py:57-64` | Beta-distributed curiosity option | `beta(mu, rng, params)`: Beta with mean `mu`, std from `params["std_curiosity"]` (default `0.1`); if `mu` outside $(0,1)$ returns `mu` constant | Heterogeneous population with tunable spread; degenerate means fall back to homogeneity |
| `sciencemodels/landscapes.py:67-68` | Uniform curiosity option | `uniform(mu, rng, params) = U(0,1)`, ignoring `mu` | Maximally heterogeneous population |
| `scripts/bulk_runner.py:49-54` | Curiosity sweep vs. mixed population | even setups: fixed curiosity over `curiosities = [k/10 for k in range(11)]`; odd setups: `landscapes.uniform` with `initial_curiosity: 0.5` | The main experimental contrast: homogeneous sweeps vs. mixed populations |
| `scripts/runner.py:46` | Minimum population size | comment `# minimum 3` with `n_agents: 3` | Visibility and prestige averages run over "other" agents; needs a few peers |
| `scripts/bulk_runner.py:57` | Bulk population size | `n_agents: 20` | Community size in the large sweep |

## Decision rule

| Reference | Assumption | Current value / form | Conceptual role |
|---|---|---|---|
| `sciencemodels/agents.py:142-144` | Choice set: von Neumann neighborhood, center excluded | `get_neighborhood(self.pos, moore=False, include_center=False)` | 4 neighbors + implicit stay; local, bounded search |
| `sciencemodels/agents.py:150-170` | Greedy best-of with stay fallback | move iff `best_reward - total_current_reward > 0` | Strict improvement rule; ties and losses mean staying — a status-quo bias |
| `sciencemodels/agents.py:156-159` | Decision noise is multiplicative on the neighbor's novelty | $k' = k\,(1 + \epsilon\,U(-1,1))$, redrawn per neighbor per step | Perception noise; $\epsilon$ scales its amplitude |
| `scripts/runner.py:49`, `scripts/bulk_runner.py:60` | Noise level | `epsilon: 0.1` | 10% relative misperception of novelty |
| `sciencemodels/agents.py:146-149` | Current tile evaluated without noise | `current_rw_novelty = compute_knowledge(self.pos)` then unnoised reward | Asymmetric noise: the status quo is seen clearly, options are noisy — biases toward staying |
| `sciencemodels/agents.py:145` | *Dead / commented*: random walk alternative | `# optionpos = self.model.rng.choice(neighbors_nodes)` | Marks the random-move null model as an intended option |
| `sciencemodels/agents.py:134` | *Dead / commented*: additive reward alternative | `# reward = self.curiosity * Novelty / avg + (1 - self.curiosity) * visibility` | Linear weighting instead of Cobb-Douglas — a free functional-form choice |
| `sciencemodels/agents.py:129-132` | *Dead / commented*: three network-effect designs | (1) agents dragged toward others (current), (2) crowding changes tile attractiveness, (3) prestige-weighted pull | Explicit list of intended variants for the social term |
| `sciencemodels/model.py:390-394` | Asynchronous update in random order | `temp.shuffle_do("step")` each step | Ordering effects averaged out; earlier movers can deplete tiles for later ones |
| `sciencemodels/agents.py:89-90` | Docstring admits exploratory status | "This part is just a test, we change the network reward by..." | The visibility reward itself is framed as a testbed, not a settled assumption |

# Reward and proxy structure

| Reference | Assumption | Current value / form | Conceptual role |
|---|---|---|---|
| `sciencemodels/agents.py:125-127` | Cobb-Douglas style reward | $R = \left(k_{novelty}/\bar{k}\right)^{c} \cdot v^{1-c}$ with $c$ = curiosity | Curiosity is an elasticity: trade-off between knowledge and visibility; both inputs must be positive |
| `sciencemodels/agents.py:93,125` | Novelty normalized by population average | divide by `avg_current_agent_knowledge` | Reward measures relative novelty, not absolute knowledge |
| `sciencemodels/agents.py:91-95` | `use_distance=False` alternative functional form | $R = c \cdot k_{novelty}/\bar{k}$, no visibility term | Curiosity becomes a linear weight rather than an exponent — a wholly different parametrization |
| `sciencemodels/agents.py:83-84` | Visibility formula with +1 smoothing | $v = (\overline{d}_{global} + 1)/(d_{own} + 1)$ | Ratio of own average distance to the population average; +1 avoids division by zero and dampens extremes when agents cluster |
| `sciencemodels/model.py:369-378` | Global distance reference recomputed each step | `agent_avg_distance` = mean over agents of mean pairwise Manhattan-torus distance | Normalizer that makes visibility relative to the community's spread |
| `sciencemodels/agents.py:48-62` | Distance metric: Manhattan on a torus | `dx = min(dx, n-dx); dy = min(dy, n-dy); return dx + dy` | Wrapping, non-diagonal metric; defines what "close" means for visibility and clustering |
| `sciencemodels/agents.py:117-121`, `sciencemodels/model.py:154` | `use_visibility_reward` switches the visibility denominator | if true: $(\overline{d}_{global}+1)/(\overline{d}_{prestige}+1)$ with prestige-weighted mean distance; else plain visibility | Whether social proximity is weighted by peers' prestige (the proxy channel in the reward) |
| `sciencemodels/agents.py:103-115` | Prestige weights come from `prestige_vanishing` | `np.average(dists, weights=prest)` with `prest = a.prestige_vanishing` | Which prestige stock shapes perceived visibility; only applied when total prestige > 0 (line 111) |
| `sciencemodels/agents.py:73` | *Dead / commented*: pairwise alternative | "we could also use individual distances instead of agent's visibility" | Per-perceiver visibility as an intended variant |
| `sciencemodels/agents.py:123-124` | *Dead / commented*: field- vs. agent-visibility question | "differentiate between visibility of a field vs. visibility of an agent?" | Conceptual degree of freedom about what the visibility attaches to |
| **unresolvable** (helper deleted) | *Dead code*: normalized reward comparison helper | `rationRewards`: $\Delta R / (\|R_{new}\| + \|R_{old}\|)$ with zero-guard — the helper was dropped in the restructure and `grep -rn "rationRewards" sciencemodels/ scripts/` returns nothing | Alternative bounded comparison rule for the decision step; the option itself is gone, not merely inactive |
| `sciencemodels/model.py:195-203` | Degenerate-configuration warning | prints if `use_distance` is false and curiosity, noise and visibility reward all inactive | Documents the assumption that at least one of these levers must be on for movement to be meaningful |

# Prestige dynamics

| Reference | Assumption | Current value / form | Conceptual role |
|---|---|---|---|
| `sciencemodels/agents.py:172-183` | Prestige accrues from the harvested flow | `farm()` depletes first (line 173), then `increase_prestige(knowledge_before * harvest)` (line 181); legacy post-harvest value stays available as `prestige_accrual: "post_tile"` (line 183) | ~~Prestige measures roughly the tile's standing knowledge, not the harvested flow $h\cdot k$; a measure-choice assumption on what counts as achievement~~ since 2026-11-13 prestige measures the harvested flow; the measure choice remains a lever through `prestige_accrual` |
| `sciencemodels/agents.py:66-72` | `prestige_visibility` accrual | $\text{mean}_{i \neq self}\left[ value^{c_i} \cdot visibility_{self}^{1-c_i} \right]$ | Peer-perceived prestige: each other agent applies its own curiosity elasticity to the value; uses the earner's visibility, not the perceiver's distance |
| `sciencemodels/agents.py:73` | *Dead / commented*: individual distances in `prestige_visibility` | comment on line 73 | Alternative weighting of the same accrual |
| `sciencemodels/agents.py:74-81`, `sciencemodels/model.py:132` | Vanishing prestige: exponential forgetting | $P_v \leftarrow x\cdot P_v + value$; `x = vanishing_factor_for_prestige`; $x = 1$ disables decay | How much past achievement still counts; ~~$x = 0$ (constructor default) keeps only the latest harvest~~ constructor default now 0.9 (unified 2026-11-13) |
| `scripts/runner.py:59` | Vanishing factor in interactive runs | ~~`vanishing_factor_for_prestige: 0.5`~~ 0.9 (unified 2026-11-13) | Half-life of one step in the prestige stock |
| `scripts/bulk_runner.py:70` | Vanishing factor in bulk runs | `vanishing_factor_for_prestige: 0.9` | Long memory in the large sweep --- ~~different from `runner.py`~~ now the unified standard (2026-11-13) |
| `sciencemodels/metrics.py:53-97` | "Ideal merit" benchmark | greedy allocation of all agents to highest tiles with the same harvest factor; loss per agent matched by sorting, averaged per site | Counterfactual measure of how much an agent's placement costs the community |
| `sciencemodels/metrics.py:69,78-79` | Ideal benchmark assumes sequential multiplicative depletion | `know * (1 - harvest) ** redone` for stacking agents; ideal list depletes the same way | Fair comparison basis depends on the harvest functional form |
| `sciencemodels/metrics.py:55,107` | Snapshot timing | ideal merit computed from `previous_knowledge`, then current knowledge copied into it | Measures this step's harvest against the state at the step's start |

# Demographics (entry and supervisor choice)

| Reference | Assumption | Current value / form | Conceptual role |
|---|---|---|---|
| `sciencemodels/model.py:309-313` | Entry rate via fractional accumulator | `agents_to_generate = floor(steps·t) - floor((steps-1)·t)` with `t = agent_generation_time`; $t \le 0$ switches generation off | Population growth rate; smooth fractional scheduling |
| `scripts/runner.py:54`, `scripts/bulk_runner.py:65` | Generation off in standard runs | `agent_generation_time: 0` | The supervisor/collaborator channel is currently inactive in the standard configurations |
| `sciencemodels/model.py:315-326` | Supervisor drawn proportional to cumulative prestige | roulette wheel over `a.prestige` (raw prestige stock) | Core proxy mechanism: prestigious scientists attract newcomers regardless of research quality |
| `sciencemodels/model.py:321-326` | Edge case: zero total prestige | `rng.uniform(0, 0)` → `val <= p` matches the first agent | Deterministic fallback to the first-listed scientist |
| `sciencemodels/model.py:328-331` | ~~New agents' curiosity fixed to `initial_curiosity`~~ newcomers draw curiosity through `agent_generation_function`, like founders | `w = self.agent_generation_function(...)` then `Scientist(self, w, self.epsilon)` — ~~bypasses `agent_generation_function`~~ | ~~Newcomers do not inherit or draw curiosity like founders — inconsistency between founder and successor populations~~ removed 2026-11-13 (Path 0): founder and successor populations draw curiosity the same way |
| `sciencemodels/model.py:336-337` | Newcomers spawn at the supervisor's exact position | `coords = sup.pos` | Local clustering around prestige; no spatial dispersal |
| `sciencemodels/agents.py:41`, `sciencemodels/model.py:105` | Newcomers' age stays at the default 23 | random age draw runs only in `_init_agents` for founders | Successor cohorts are age-homogeneous |
| `sciencemodels/model.py:103,331` | Newcomers inherit the model-wide `epsilon` | same noise for everyone | No heterogeneity in rationality/noise |
| `scripts/runner.py:54` | *Mislabel*: rate comment | comment says "if 0.1, 10 new agent per generation" while $t = 0.1$ yields 0.1 agent per step (1 per 10 steps) | Easy-to-misread entry-rate semantics |

# Measurement and reporting

| Reference | Assumption | Current value / form | Conceptual role |
|---|---|---|---|
| `sciencemodels/metrics.py:35-50` | Local merit = mean knowledge of a 5-tile cross | center + 4 von Neumann neighbors, `moore=False` | "Quality of one's spot" benchmark for prestige-merit correlations |
| `sciencemodels/metrics.py:53-97` | Ideal merit = loss vs. greedy optimum | see prestige section | Second, stricter merit benchmark |
| `sciencemodels/metrics.py:110-118` | Correlation reporter | Pearson `np.corrcoef`; returns `0` when either stock has zero variance | Treats degenerate cases as "zero correlation" rather than undefined; Pearson assumes linear association |
| `sciencemodels/model.py:261-266` | Six reported prestige-merit correlations | prestige / prestige\_visibility / prestige\_vanishing × local / ideal merit | The main outcome family for proxy failure |
| `sciencemodels/metrics.py:125-132` | Milestone thresholds in `update_step_goals` | explored and weighted milestones at 0.5 and 0.9; `harvested_10_step` at 0.10 and `harvested_25_step` at 0.25 of knowledge harvested | Speed measures; ~~the harvested milestone names do not match their thresholds~~ the 2026-11-13 rename made the names match their thresholds, and `sciencemodels/plotting.py:37-41` keeps a read-time shim for the old column names |
| `sciencemodels/model.py:184` | Unreached-milestone sentinel | `_default_steps_thresholds = step_limit * 1.25` (float) | Marks "never reached" in a numeric column; sentinel value is a modelling constant |
| `sciencemodels/model.py:128` | Default horizon | `step_limit: 400` | Length of a run; sets the scale of all time-based measures |
| `scripts/runner.py:57` | Interactive horizon | `step_limit: 200` | Shorter demo runs |
| `scripts/bulk_runner.py:68` | Bulk horizon | `step_limit: 400` | Standard sweep horizon |
| `sciencemodels/model.py:225-250` | Inequality measures | top-1% and top-10% prestige share via `np.percentile(..., 99 / 90)` | Prestige concentration reporting |
| `sciencemodels/model.py:359-368` | Exploration measures | `explored_percentage = tiles touched / size²`; weighted variant weights tiles by `initial_knowledge` | Coverage vs. knowledge-weighted coverage |
| `sciencemodels/model.py:107,293` | `explored` is set at placement | `new_place` marks the destination tile (line 277); `_init_agents` marks each founder's tile directly (line 106) | ~~Founder tiles count as unexplored until revisited or until a generation sweep re-marks positions — a measurement artifact~~ fixed 2026-11-13: founder tiles are marked explored at placement |
| `sciencemodels/model.py:379-382` | Harvested share | $(1 - \sum k_{current})$ since total is normalized to 1 | Overall productivity measure |
| `sciencemodels/io.py:25-44` | Report key convention | lowercase-keyed model vars take the last value; uppercase-keyed take the run mean (`Mean_of_` prefix) | Reporting semantics of the bulk CSV |
| `sciencemodels/model.py:506-514` | Header extraction via dummy model | `all_params["step_limit"] = 1` throw-away run | CSV schema derived from a live run |
| `sciencemodels/model.py:530-535`, `scripts/bulk_runner.py:82-83` | Replication and seeding | `repeats_per_setting = 40`, `agent_seed = repeat`, `change_landscape_seed = True` | 40 replications varying both agent and landscape seed per parameter cell |
| `scripts/bulk_runner.py:50` | Curiosity grid | `[k/10 for k in range(11)]` = 0.0, 0.1, ..., 1.0 | Resolution of the main experimental sweep |
| `scripts/bulk_runner.py:97` | Parallelism | `Parallel(n_jobs=10)` | Throughput only, but determines run ordering on disk |
| `sciencemodels/viz.py:48-53,60-67` | Plot jitter | offset `(unique_id/n_agents - 0.5)²` added to scatter positions | Cosmetic display choice |
| `sciencemodels/viz.py:84-87,46` | Color scale choice | `vmin=0, vmax=1` when `knowledge_normalization == "minmax"`, auto otherwise; *commented* `# , vmin=0, vmax=1` alternative on line 46 | Visual comparability across frames |
| `notebooks/compare_landscapes_corr-prestige-merit.ipynb` (cell 0) | Sweep semantics: 8 run files = 4 landscapes × fixed/mixed curiosity | `run_number0`-`run_number7` mapping; plots group by `initial_curiosity` | Analysis-level assumption that curiosity is the x-axis of interest |
| `notebooks/compare_landscapes_flexible_search.ipynb`, `notebooks/compare_landscapes_normalizedknowledge.ipynb` (cell 0) | Filename-encoded setups | `std5`, `number_gaussians2/6`, `prop_random5`, `std_curiosity0.1` (0.2 in the normalized-knowledge notebook) | Legacy parameter bundles; `std_curiosity` 0.1 vs. 0.2 differs across experiment generations. The bundles now survive only in the `data/` CSV filenames, and one notebook name gained an underscore (`flexible search` → `flexible_search`) |
| `sciencemodels/plotting.py:172,204` | ~~Horizon hard-coded in plotting~~ Horizon derived from the frames read | ~~`length=401` default in `plot_avg_time_vs_corr_prest_merit`, `length=1601` at the `plot_all_curiosities` call~~ `length=None` default, resolved to `max(Step) + 1` over the frames actually read; an explicit `length` is still honoured | ~~Assumes 400- or 1600-step longitudinal runs~~ the 2026-11-13 horizon fix removed the assumption: the 1601-step horizon over 201-step companion frames padded the mean series with NaN and emitted 30,803 `Mean of empty slice` warnings; the helper still lives in the package, not the notebook |
| `sciencemodels/viz.py:26-39,42-55` | Landscape displayed as 3D surface / heatmap | `Vizualize_3d`, `plot` | Visualization assumptions only |

# Overview: most consequential levers for the paper question

The paper question is proxy failure through collaborator choice: whether a community that rewards visible prestige ends up matching newcomers to prestigious-but-shallow supervisors instead of high-merit ones. Ranked by directness of impact:

- Supervisor-proportional entry (`sciencemodels/model.py:315-326`) is the proxy channel itself. It is currently switched off (`agent_generation_time: 0` in both runners), so any proxy-failure claim requires activating and sweeping the entry rate.
- Which prestige stock weights supervisor choice and visibility (`prestige` vs. `prestige_vanishing` vs. `prestige_visibility`; `sciencemodels/agents.py:103-115`, `sciencemodels/model.py:315`) decides what the community treats as a quality signal — the Goodhart target.
- The prestige accrual rule (`sciencemodels/agents.py:172-183`): ~~prestige flows from post-harvest tile knowledge rather than harvested output~~ prestige accrues from the harvested flow $h \cdot k$ since 2026-11-13 (Path 0; legacy rule via `prestige_accrual: 'post_tile'`). Changing this changes what "achievement" means and thus the proxy's informativeness.
- The social term in the reward: `use_visibility_reward` and the prestige weighting of distance (`sciencemodels/agents.py:117-121`) determine how strongly agents chase proximity to prestigious peers at the cost of knowledge.
- The Cobb-Douglas elasticity (curiosity, `sciencemodels/agents.py:125-127`) and the curiosity distribution (constant vs. Beta vs. uniform, `sciencemodels/model.py:131`, `sciencemodels/landscapes.py:57-68`) govern how much agents trade knowledge for visibility and how heterogeneous the population is.
- The vanishing factor (`sciencemodels/agents.py:74-81`) sets the memory of prestige; ~~note the runners disagree (0.5 vs 0.9) and the constructor default (0.0) is a third regime~~ unified at 0.9 on 2026-11-13 (Path 0); sweep it deliberately per `docs/sensitivity_map_v1.md` (found inert at the tested horizon, $\Delta \le 0.02$).
- The harvest rate (`sciencemodels/model.py:303-307`) drives depletion, crowding and hence the cost of herding onto prestigious locations.
- Landscape family and shape parameters (`sciencemodels/landscapes.py`; `std`, `kappa`, `prop_random`, `number_gaussians`) control how legible research quality is — proxy failure should worsen when merit is harder to see.
- Knowledge normalization (min-max $[0.05, 0.95]$ vs. share-of-total, `sciencemodels/model.py:79-86`) and the 0.025 refill threshold (`sciencemodels/model.py:346`) rescale every reward and prestige flow; ~~the `new_questions` flag conflates normalization with refills and should be disentangled before any sensitivity study~~ the conflation was split on 2026-11-13 (Path 0): `knowledge_normalization` governs the scheme, `new_questions` the refill.
- Decision noise $\epsilon$ and the noise asymmetry (`sciencemodels/agents.py:156-159,146-149`) plus the strict-improvement stay rule (`sciencemodels/agents.py:166`) set how boundedly rational the search is and bias agents toward inertia.

Secondary but non-negligible: entry-position clustering (`sciencemodels/model.py:336`), newcomer curiosity now drawn through `agent_generation_function` (`sciencemodels/model.py:328-331`), the exploratory-vs-exploited merit benchmarks (`local_merit`, `ideal_Merit`), and ~~the harvested-milestone threshold mismatch (`mesa_model.py:550-551`) which affects reported speed measures but not the dynamics~~ the harvested-milestone thresholds, whose names and values now agree (`sciencemodels/metrics.py:130-131`).

# Revision Log

- **Version 1** (November 2026) --- Initial assumption catalogue, 104 entries.
- **Version 2** (November 2026) --- Status note on the restructure and the Path 0 fixes; stale entries struck (vanishing-factor disagreement, post-harvest prestige accrual, the new_questions conflation); full `file:line` re-anchor pending.
- **Version 3** (November 2026) --- Re-anchored all 104 table rows (109 reference callouts) to the post-restructure tree: agent code to `sciencemodels/agents.py`, model loop to `sciencemodels/model.py`, metrics to `sciencemodels/metrics.py`, CSV/report writing to `sciencemodels/io.py`, plots to `sciencemodels/viz.py`, notebook helpers to `sciencemodels/plotting.py`, entry points to `scripts/`. Every anchor verified by `grep -n` and by `scripts/cdf_reanchor/reanchor_cdf_refs.py`. Two rows marked unresolvable with reasons (deleted signature comment block, deleted `rationRewards` helper). Stale Path 0 assumptions restated in place: prestige accrual, `new_questions`/`knowledge_normalization` split, newcomer curiosity, founder `explored` marking, harvested milestone names. Incidental corrections: `runner.py`'s landscape-setup comment now agrees with its value and the stale "6 gaussians" comment is pinned to `scripts/bulk_runner.py:22`; the `compare_landscapes_flexible search.ipynb` filename corrected to its live underscore spelling. Heading levels raised from `##`/`###` to `#`/`##` to match `md-style`.
