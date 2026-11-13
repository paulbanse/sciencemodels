---
title: "Conceptual Degrees of Freedom: Catalogue of Modelling Assumptions"
version: 2.0
date: 2026-11-13
tags: [sciencemodels, assumptions, sensitivity-analysis, proxy-failure]
project: sciencemodels
revision-vehicle: strikethrough
revision-renders: "md, pdf, html"
---

**Version 2** (November 2026)
**Version 1** (November 2026)

v2 status note (2026-11-13): entries cite pre-restructure paths
(`mesa_model.py` etc.); the code moved to the `sciencemodels/` package the
same day (content preserved, see `5bd5c83`), so references map by module.
The Path 0 hygiene round (`33421ff`) resolved several entries: prestige
accrual now takes the harvested flow $h \cdot k$ (legacy via
`prestige_accrual: 'post_tile'`); `new_questions` is split from
`knowledge_normalization`; milestones are renamed `harvested_10_step` /
`harvested_25_step`; newcomer curiosity runs through
`agent_generation_function`; founder tiles count as explored; the vanishing
factor is unified at 0.9 (the disagreement rows below are stale and struck).
A full re-anchor of `file:line` references to the package layout is pending.

## Purpose

This catalogue collects every place where the code fixes a parameter value, threshold, functional form, initialization rule or normalization. Each entry gives the `file:line` reference, what is assumed, the current value or formula, and what the assumption conceptually controls. Entries marked *dead / commented* are alternatives left in the code; they document intended degrees of freedom even where currently inactive.

Sections follow the causal chain of the model: epistemic environment → agent decision-making → reward and proxy structure → prestige dynamics → demographics → measurement and reporting. The final section ranks the levers for the paper question (proxy failure via collaborator choice).

## Epistemic environment

### Grid and topology

| Reference | Assumption | Current value / form | Conceptual role |
|---|---|---|---|
| `mesa_model.py:305,328` | World is a single square grid of side `grid_size` | constructor parameter | Total size of the knowledge space; fixes ratio of agents to tiles |
| `mesa_model.py:231` | Grid is a torus | `MultiGrid(self.size, self.size, torus=True)` | No borders or center; every tile topologically equivalent |
| `runner.py:42`, `bulk_runner.py:51` | Grid side 20 | `grid_size: 20` (400 tiles) | Absolute scale of exploration tasks in all standard runs |
| `mesa_model.py:228-249` | Knowledge lives in property layers | `knowledge`, `previous_knowledge`, `initial_knowledge`, `explored` | Separates the epistemic state from agents; enables snapshot comparisons |

### Landscape generation

| Reference | Assumption | Current value / form | Conceptual role |
|---|---|---|---|
| `landscapefunctions.py:13-15` | `randominit`: i.i.d. uniform tile values | $k_{xy} \sim U(0,1)$ | Null landscape: no structure, no correlation between neighbors |
| `landscapefunctions.py:18-29` | `Gaussian`: one bump, separable in x and y, drawn on the torus circle | $k_{xy} = f_{vM}(x;\mu_x,\kappa)\cdot f_{vM}(y;\mu_y,\kappa)$ with von Mises pdfs; $\mu_x,\mu_y$ drawn from `args["seed"]` | A single coherent research area; separability rules out rotated/elliptical ridges |
| `landscapefunctions.py:23` | Landscape is deterministic given the seed | `rng_fix = np.random.default_rng(args["seed"])`; per-cell `rng` argument ignored by `Gaussian` | Landscapes are reproducible and comparable across agent populations |
| `landscapefunctions.py:19-22` | Width parameter `std` (tile units) maps to von Mises concentration | `kappa = 1 / (args["std"] / size)`, i.e. $\kappa = size/std$, unless `kappa` given directly | How broad a knowledge peak is relative to the grid |
| `landscapefunctions.py:41-53` | `multipleGaussians`: unnormalized sum of several bumps | `sum` of `number_gaussians` Gaussians with sub-seeds drawn from the master seed | Multi-topic landscape; unnormalized sum means narrow peaks can tower over broad ones |
| `runner.py:10,21,24` | Landscape presets | setup 1: `std: 2, number_gaussians: 2`; setup 2: `std: 5, number_gaussians: 4` (comment says "6 gaussians" — comment and value disagree) | Concrete landscape choices for interactive runs |
| `bulk_runner.py:22-26` | Landscape presets for bulk runs | `std: 5` with `number_gaussians: 2` (setups 2-3) or `6` (setups 4-5) | Landscape families compared in the large sweep |
| `landscapefunctions.py:32-38` | `noisyGaussian`: uniform noise floor plus amplified Gaussian | $k_{xy} = U(0,1) + prop\_random \cdot G(x,y)$ | Landscape where signal must be teased out of noise; `prop_random` sets signal-to-noise |
| `runner.py:28`, `bulk_runner.py:30` | Signal amplification | `prop_random: 5` | Bump is 5x the noise floor; ordering of peaks barely legible |
| `runner.py:30`, `bulk_runner.py:32` | Curiosity spread passed via generation params | `std_curiosity: 0.1` | Coupled to the Beta curiosity generator (see decision-making) |
| `landscapefunctions.py:56-64` (comment block) | Landscape functions share one signature | `(posX, posY, size, args, rng)` | Any callable with this signature can be swapped in — the landscape family is a free design axis |

### Knowledge initialization and normalization

| Reference | Assumption | Current value / form | Conceptual role |
|---|---|---|---|
| `mesa_model.py:265-272` | Two mutually exclusive normalization schemes, switched by `new_questions` | min-max rescale to $[0.05, 0.95]$ if `new_questions` truthy; else share-of-total $k_{xy}/\sum k$ | Absolute scale of knowledge values, hence of rewards and prestige |
| `mesa_model.py:270-272` | Floor and ceiling of min-max rescale | `* 0.95 + 0.05` | No tile is ever fully empty or maximal; 0.05 floor keeps novelty and reward bounded away from zero |
| `mesa_model.py:266-268` | Share-of-total alternative | `val / self.total_initial_knowledge` | Makes tile values sum to 1; small absolute rewards |
| `mesa_model.py:327,265,533` | One flag controls two unrelated things | `new_questions` selects both the normalization scheme and the refill mechanism | Conceptual conflation: normalization cannot be varied independently of refills |
| `mesa_model.py:281` | Initial total knowledge reset after normalization | `self.total_initial_knowledge = np.float64(1.0)` | Makes `percentage_knowledge_harvested` a plain share |
| `mesa_model.py:274-279` | Initial state snapshot | `initial_knowledge` and `previous_knowledge` copy the normalized values | Basis for weighted-exploration and ideal-merit benchmarks |

### Harvest and knowledge depletion

| Reference | Assumption | Current value / form | Conceptual role |
|---|---|---|---|
| `mesa_model.py:503-507` | Harvesting is multiplicative depletion | $k_{xy} \leftarrow (1-h)\,k_{xy}$ per agent visit | Diminishing returns at a tile; crowding costs |
| `runner.py:41`, `bulk_runner.py:50` | Harvest rate | `harvest: 0.02` | Speed of depletion vs. exploration; strength of competition over good tiles |
| `mesa_model.py:507` | No regrowth except the refill rule | only `_generate_new_questions` restores knowledge | Knowledge is a depletable stock; the landscape is effectively finite |
| `mesa_model.py:441-444` | Population-relative knowledge normalizer | `avg_current_agent_knowledge = mean(agents' last_tile_knowledge)` | Reward is relative to what the community currently sits on, not to an absolute scale |

### New-questions refill

| Reference | Assumption | Current value / form | Conceptual role |
|---|---|---|---|
| `mesa_model.py:532-539` | Refill trigger threshold | occupied tiles with $k < 0.025$ get $k \leftarrow U(0,1)$ | How depleted a tile must be before it becomes a "new question"; 0.025 is a hard-coded constant |
| `mesa_model.py:534` | Only occupied tiles are refilled | `occupied_pos = list(set(agent positions))` | New questions appear only where scientists already are — no exogenous discovery elsewhere |

## Agent decision-making

### Initialization

| Reference | Assumption | Current value / form | Conceptual role |
|---|---|---|---|
| `mesa_model.py:59` | Default age at construction | `self.age = 23` | Starting career stage (implies PhD-start metaphor) |
| `mesa_model.py:290` | Initial agents' ages drawn uniformly | `rng.choice(np.arange(23, 50))` → integers 23-49 | Age heterogeneity at $t=0$ |
| `mesa_model.py:206` | Age increments each step | `self.age += 1` | Time in the model = career time |
| `mesa_model.py:45,206` (whole class) | Age has no functional effect | no death, retirement, or age-dependent behavior anywhere | *Dead lever*: age is recorded but does nothing |
| `mesa_model.py:289` | Initial positions uniform over the grid | `coords = (rng.integers(0, size), rng.integers(0, size))` | No correlation between starting position and landscape |
| `mesa_model.py:288` | Homogeneous noise sensitivity | all agents get the model-level `epsilon` | One noise level for the whole population |
| `mesa_model.py:56-58` | Zero initial prestige stocks | `prestige = prestige_visibility = prestige_vanishing = 0` | Everyone starts equal in the prestige race |
| `mesa_model.py:313-315` | Curiosity constant across agents by default | `agent_generation_function = lambda cur, rng, params: cur` | Default population is homogeneous in curiosity |
| `landscapefunctions.py:57-64` | Beta-distributed curiosity option | `beta(mu, rng, params)`: Beta with mean `mu`, std from `params["std_curiosity"]` (default `0.1`); if `mu` outside $(0,1)$ returns `mu` constant | Heterogeneous population with tunable spread; degenerate means fall back to homogeneity |
| `landscapefunctions.py:67-68` | Uniform curiosity option | `uniform(mu, rng, params) = U(0,1)`, ignoring `mu` | Maximally heterogeneous population |
| `bulk_runner.py:38-43` | Curiosity sweep vs. mixed population | even setups: fixed curiosity over `curiosities = [k/10 for k in range(11)]`; odd setups: `landscapefunctions.uniform` with `initial_curiosity: 0.5` | The main experimental contrast: homogeneous sweeps vs. mixed populations |
| `runner.py:37` | Minimum population size | comment `# minimum 3` with `n_agents: 3` | Visibility and prestige averages run over "other" agents; needs a few peers |
| `bulk_runner.py:46` | Bulk population size | `n_agents: 20` | Community size in the large sweep |

### Decision rule

| Reference | Assumption | Current value / form | Conceptual role |
|---|---|---|---|
| `mesa_model.py:174-176` | Choice set: von Neumann neighborhood, center excluded | `get_neighborhood(self.pos, moore=False, include_center=False)` | 4 neighbors + implicit stay; local, bounded search |
| `mesa_model.py:182-198` | Greedy best-of with stay fallback | move iff `best_reward - total_current_reward > 0` | Strict improvement rule; ties and losses mean staying — a status-quo bias |
| `mesa_model.py:188-190` | Decision noise is multiplicative on the neighbor's novelty | $k' = k\,(1 + \epsilon\,U(-1,1))$, redrawn per neighbor per step | Perception noise; $\epsilon$ scales its amplitude |
| `runner.py:40`, `bulk_runner.py:49` | Noise level | `epsilon: 0.1` | 10% relative misperception of novelty |
| `mesa_model.py:178-181` | Current tile evaluated without noise | `current_rw_novelty = compute_knowledge(self.pos)` then unnoised reward | Asymmetric noise: the status quo is seen clearly, options are noisy — biases toward staying |
| `mesa_model.py:177` | *Dead / commented*: random walk alternative | `# optionpos = self.model.rng.choice(neighbors_nodes)` | Marks the random-move null model as an intended option |
| `mesa_model.py:152` | *Dead / commented*: additive reward alternative | `# reward = self.curiosity * Novelty / avg + (1 - self.curiosity) * visibility` | Linear weighting instead of Cobb-Douglas — a free functional-form choice |
| `mesa_model.py:147-150` | *Dead / commented*: three network-effect designs | (1) agents dragged toward others (current), (2) crowding changes tile attractiveness, (3) prestige-weighted pull | Explicit list of intended variants for the social term |
| `mesa_model.py:599-601` | Asynchronous update in random order | `temp.shuffle_do("step")` each step | Ordering effects averaged out; earlier movers can deplete tiles for later ones |
| `mesa_model.py:107-108` | Docstring admits exploratory status | "This part is just a test, we change the network reward by..." | The visibility reward itself is framed as a testbed, not a settled assumption |

## Reward and proxy structure

| Reference | Assumption | Current value / form | Conceptual role |
|---|---|---|---|
| `mesa_model.py:143-145` | Cobb-Douglas style reward | $R = \left(k_{novelty}/\bar{k}\right)^{c} \cdot v^{1-c}$ with $c$ = curiosity | Curiosity is an elasticity: trade-off between knowledge and visibility; both inputs must be positive |
| `mesa_model.py:111,143` | Novelty normalized by population average | divide by `avg_current_agent_knowledge` | Reward measures relative novelty, not absolute knowledge |
| `mesa_model.py:109-113` | `use_distance=False` alternative functional form | $R = c \cdot k_{novelty}/\bar{k}$, no visibility term | Curiosity becomes a linear weight rather than an exponent — a wholly different parametrization |
| `mesa_model.py:101-102` | Visibility formula with +1 smoothing | $v = (\overline{d}_{global} + 1)/(d_{own} + 1)$ | Ratio of own average distance to the population average; +1 avoids division by zero and dampens extremes when agents cluster |
| `mesa_model.py:576-585` | Global distance reference recomputed each step | `agent_avg_distance` = mean over agents of mean pairwise Manhattan-torus distance | Normalizer that makes visibility relative to the community's spread |
| `mesa_model.py:66-80` | Distance metric: Manhattan on a torus | `dx = min(dx, n-dx); dy = min(dy, n-dy); return dx + dy` | Wrapping, non-diagonal metric; defines what "close" means for visibility and clustering |
| `mesa_model.py:136-139,317` | `use_visibility_reward` switches the visibility denominator | if true: $(\overline{d}_{global}+1)/(\overline{d}_{prestige}+1)$ with prestige-weighted mean distance; else plain visibility | Whether social proximity is weighted by peers' prestige (the proxy channel in the reward) |
| `mesa_model.py:121-133` | Prestige weights come from `prestige_vanishing` | `np.average(dists, weights=prest)` with `prest = a.prestige_vanishing` | Which prestige stock shapes perceived visibility; only applied when total prestige > 0 (line 129) |
| `mesa_model.py:91` | *Dead / commented*: pairwise alternative | "we could also use individual distances instead of agent's visibility" | Per-perceiver visibility as an intended variant |
| `mesa_model.py:141-142` | *Dead / commented*: field- vs. agent-visibility question | "differentiate between visibility of a field vs. visibility of an agent?" | Conceptual degree of freedom about what the visibility attaches to |
| `mesa_model.py:27-36` | *Dead code*: normalized reward comparison helper | `rationRewards`: $\Delta R / (|R_{new}| + |R_{old}|)$ with zero-guard | Alternative bounded comparison rule for the decision step |
| `mesa_model.py:355-363` | Degenerate-configuration warning | prints if `use_distance` is false and curiosity, noise and visibility reward all inactive | Documents the assumption that at least one of these levers must be on for movement to be meaningful |

## Prestige dynamics

| Reference | Assumption | Current value / form | Conceptual role |
|---|---|---|---|
| `mesa_model.py:204,207-210,82-83` | Prestige accrues from the tile's post-harvest value | `farm()` depletes first (line 204), then `increase_prestige(last_tile_knowledge)` where `last_tile_knowledge = (1-h)·k` | Prestige measures roughly the tile's standing knowledge, not the harvested flow $h\cdot k$; a measure-choice assumption on what counts as achievement |
| `mesa_model.py:84-90` | `prestige_visibility` accrual | $\text{mean}_{i \neq self}\left[ value^{c_i} \cdot visibility_{self}^{1-c_i} \right]$ | Peer-perceived prestige: each other agent applies its own curiosity elasticity to the value; uses the earner's visibility, not the perceiver's distance |
| `mesa_model.py:91` | *Dead / commented*: individual distances in `prestige_visibility` | comment on line 91 | Alternative weighting of the same accrual |
| `mesa_model.py:92-99,316` | Vanishing prestige: exponential forgetting | $P_v \leftarrow x\cdot P_v + value$; `x = vanishing_factor_for_prestige`; $x = 1$ disables decay | How much past achievement still counts; ~~$x = 0$ (constructor default) keeps only the latest harvest~~ constructor default now 0.9 (unified 2026-11-13) |
| `runner.py:50` | Vanishing factor in interactive runs | ~~`vanishing_factor_for_prestige: 0.5`~~ 0.9 (unified 2026-11-13) | Half-life of one step in the prestige stock |
| `bulk_runner.py:59` | Vanishing factor in bulk runs | `vanishing_factor_for_prestige: 0.9` | Long memory in the large sweep --- ~~different from `runner.py`~~ now the unified standard (2026-11-13) |
| `mesa_model.py:453-497` | "Ideal merit" benchmark | greedy allocation of all agents to highest tiles with the same harvest factor; loss per agent matched by sorting, averaged per site | Counterfactual measure of how much an agent's placement costs the community |
| `mesa_model.py:469,476-477` | Ideal benchmark assumes sequential multiplicative depletion | `know * (1 - harvest) ** redone` for stacking agents; ideal list depletes the same way | Fair comparison basis depends on the harvest functional form |
| `mesa_model.py:455,497` | Snapshot timing | ideal merit computed from `previous_knowledge`, then current knowledge copied into it | Measures this step's harvest against the state at the step's start |

## Demographics (entry and supervisor choice)

| Reference | Assumption | Current value / form | Conceptual role |
|---|---|---|---|
| `mesa_model.py:509-513` | Entry rate via fractional accumulator | `agents_to_generate = floor(steps·t) - floor((steps-1)·t)` with `t = agent_generation_time`; $t \le 0$ switches generation off | Population growth rate; smooth fractional scheduling |
| `runner.py:45`, `bulk_runner.py:54` | Generation off in standard runs | `agent_generation_time: 0` | The supervisor/collaborator channel is currently inactive in the standard configurations |
| `mesa_model.py:514-523` | Supervisor drawn proportional to cumulative prestige | roulette wheel over `a.prestige` (raw prestige stock) | Core proxy mechanism: prestigious scientists attract newcomers regardless of research quality |
| `mesa_model.py:515-523` | Edge case: zero total prestige | `rng.uniform(0, 0)` → `val <= p` matches the first agent | Deterministic fallback to the first-listed scientist |
| `mesa_model.py:525` | New agents' curiosity fixed to `initial_curiosity` | `Scientist(self, self.initial_curiosity, self.epsilon)` — bypasses `agent_generation_function` | Newcomers do not inherit or draw curiosity like founders — inconsistency between founder and successor populations |
| `mesa_model.py:526-527` | Newcomers spawn at the supervisor's exact position | `coords = sup.pos` | Local clustering around prestige; no spatial dispersal |
| `mesa_model.py:59,290` | Newcomers' age stays at the default 23 | random age draw only runs in `_init_agents` for founders | Successor cohorts are age-homogeneous |
| `mesa_model.py:288,525` | Newcomers inherit the model-wide `epsilon` | same noise for everyone | No heterogeneity in rationality/noise |
| `runner.py:45` | *Mislabel*: rate comment | comment says "if 0.1, 10 new agent per generation" while $t = 0.1$ yields 0.1 agent per step (1 per 10 steps) | Easy-to-misread entry-rate semantics |

## Measurement and reporting

| Reference | Assumption | Current value / form | Conceptual role |
|---|---|---|---|
| `mesa_model.py:155-170` | Local merit = mean knowledge of a 5-tile cross | center + 4 von Neumann neighbors, `moore=False` | "Quality of one's spot" benchmark for prestige-merit correlations |
| `mesa_model.py:453-497` | Ideal merit = loss vs. greedy optimum | see prestige section | Second, stricter merit benchmark |
| `mesa_model.py:367-375` | Correlation reporter | Pearson `np.corrcoef`; returns `0` when either stock has zero variance | Treats degenerate cases as "zero correlation" rather than undefined; Pearson assumes linear association |
| `mesa_model.py:423-428` | Six reported prestige-merit correlations | prestige / prestige\_visibility / prestige\_vanishing × local / ideal merit | The main outcome family for proxy failure |
| `mesa_model.py:545-552` | Milestone thresholds in `_update_step_goals` | explored and weighted milestones at 0.5 and 0.9; "harvested\_50" actually at 0.10 and "harvested\_90" at 0.25 of knowledge harvested | Speed measures; the harvested milestone names do not match their thresholds |
| `mesa_model.py:344` | Unreached-milestone sentinel | `_default_steps_thresholds = step_limit * 1.25` (float) | Marks "never reached" in a numeric column; sentinel value is a modelling constant |
| `mesa_model.py:312` | Default horizon | `step_limit: 400` | Length of a run; sets the scale of all time-based measures |
| `runner.py:48` | Interactive horizon | `step_limit: 200` | Shorter demo runs |
| `bulk_runner.py:57` | Bulk horizon | `step_limit: 400` | Standard sweep horizon |
| `mesa_model.py:387-412` | Inequality measures | top-1% and top-10% prestige share via `np.percentile(..., 99 / 90)` | Prestige concentration reporting |
| `mesa_model.py:566-575` | Exploration measures | `explored_percentage = tiles touched / size²`; weighted variant weights tiles by `initial_knowledge` | Coverage vs. knowledge-weighted coverage |
| `mesa_model.py:280,451,291` | "Explored" is set on placement in `new_place`, not at founders' initial placement | initial agents placed via `grid.place_agent` directly (line 291) | Founder tiles count as unexplored until revisited or until a generation sweep re-marks positions (lines 529-530) — a measurement artifact |
| `mesa_model.py:586-589` | Harvested share | $(1 - \sum k_{current})$ since total is normalized to 1 | Overall productivity measure |
| `mesa_model.py:645-664` | Report key convention | lowercase-keyed model vars take the last value; uppercase-keyed take the run mean (`Mean_of_` prefix) | Reporting semantics of the bulk CSV |
| `mesa_model.py:856-864` | Header extraction via dummy model | `all_params["step_limit"] = 1` throw-away run | CSV schema derived from a live run |
| `mesa_model.py:881-886` | Replication and seeding | `repeats_per_setting = 40` (`bulk_runner.py:70`), `agent_seed = repeat`, `change_landscape_seed = True` (`bulk_runner.py:71`) | 40 replications varying both agent and landscape seed per parameter cell |
| `bulk_runner.py:39` | Curiosity grid | `[k/10 for k in range(11)]` = 0.0, 0.1, ..., 1.0 | Resolution of the main experimental sweep |
| `bulk_runner.py:87` | Parallelism | `Parallel(n_jobs=10)` | Throughput only, but determines run ordering on disk |
| `mesa_model.py:627-630,670-675` | Plot jitter | offset `(unique_id/n_agents - 0.5)²` added to scatter positions | Cosmetic display choice |
| `mesa_model.py:693-696` | Color scale choice | `vmin=0, vmax=1` for `new_questions` mode, auto otherwise; *commented* `# , vmin=0, vmax=1` alternative on line 624 | Visual comparability across frames |
| `compare_landscapes_corr-prestige-merit.ipynb` (cell 0) | Sweep semantics: 8 run files = 4 landscapes × fixed/mixed curiosity | `run_number0`-`run_number7` mapping; plots group by `initial_curiosity` | Analysis-level assumption that curiosity is the x-axis of interest |
| `compare_landscapes_flexible search.ipynb`, `compare_landscapes_normalizedknowledge.ipynb` (cell 0) | Filename-encoded setups | `std5`, `number_gaussians2/6`, `prop_random5`, `std_curiosity0.1` (0.2 in the normalized-knowledge notebook) | Legacy parameter bundles; `std_curiosity` 0.1 vs 0.2 differs across experiment generations |
| `longitudinalcomparisions.ipynb` (cell 1) | Horizon hard-coded in plotting | `length=401` default, `length=1601` in `plot_all_curiosities` | Assumes 400- or 1600-step longitudinal runs |
| `mesa_model.py:605-618` | Landscape displayed as 3D surface / heatmap | `Vizualize_3d`, `plot` | Visualization assumptions only |

## Overview: most consequential levers for the paper question

The paper question is proxy failure through collaborator choice: whether a community that rewards visible prestige ends up matching newcomers to prestigious-but-shallow supervisors instead of high-merit ones. Ranked by directness of impact:

- Supervisor-proportional entry (`mesa_model.py:514-523`) is the proxy channel itself. It is currently switched off (`agent_generation_time: 0` in both runners), so any proxy-failure claim requires activating and sweeping the entry rate.
- Which prestige stock weights supervisor choice and visibility (`prestige` vs. `prestige_vanishing` vs. `prestige_visibility`; `mesa_model.py:121-133,514`) decides what the community treats as a quality signal — the Goodhart target.
- The prestige accrual rule (`mesa_model.py:204-210`): ~~prestige flows from post-harvest tile knowledge rather than harvested output~~ prestige accrues from the harvested flow $h \cdot k$ since 2026-11-13 (Path 0; legacy rule via `prestige_accrual: 'post_tile'`). Changing this changes what "achievement" means and thus the proxy's informativeness.
- The social term in the reward: `use_visibility_reward` and the prestige weighting of distance (`mesa_model.py:136-139`) determine how strongly agents chase proximity to prestigious peers at the cost of knowledge.
- The Cobb-Douglas elasticity (curiosity, `mesa_model.py:143-145`) and the curiosity distribution (constant vs. Beta vs. uniform, `mesa_model.py:315`, `landscapefunctions.py:57-68`) govern how much agents trade knowledge for visibility and how heterogeneous the population is.
- The vanishing factor (`mesa_model.py:92-99`) sets the memory of prestige; ~~note the runners disagree (0.5 vs 0.9) and the constructor default (0.0) is a third regime~~ unified at 0.9 on 2026-11-13 (Path 0); sweep it deliberately per `docs/sensitivity_map_v1.md` (found inert at the tested horizon, $\Delta \le 0.02$).
- The harvest rate (`mesa_model.py:503-507`) drives depletion, crowding and hence the cost of herding onto prestigious locations.
- Landscape family and shape parameters (`landscapefunctions.py`; `std`, `kappa`, `prop_random`, `number_gaussians`) control how legible research quality is — proxy failure should worsen when merit is harder to see.
- Knowledge normalization (min-max $[0.05, 0.95]$ vs. share-of-total, `mesa_model.py:265-272`) and the 0.025 refill threshold (`mesa_model.py:536`) rescale every reward and prestige flow; ~~the `new_questions` flag conflates normalization with refills and should be disentangled before any sensitivity study~~ the conflation was split on 2026-11-13 (Path 0): `knowledge_normalization` governs the scheme, `new_questions` the refill.
- Decision noise $\epsilon$ and the noise asymmetry (`mesa_model.py:188-190,178-181`) plus the strict-improvement stay rule (`mesa_model.py:198`) set how boundedly rational the search is and bias agents toward inertia.

Secondary but non-negligible: entry-position clustering (`mesa_model.py:526`), newcomer curiosity bypassing `agent_generation_function` (`mesa_model.py:525`), the exploratory-vs-exploited merit benchmarks (`local_merit`, `ideal_Merit`), and the harvested-milestone threshold mismatch (`mesa_model.py:550-551`) which affects reported speed measures but not the dynamics.

# Revision Log

- **Version 1** (November 2026) --- Initial assumption catalogue, 104 entries.
- **Version 2** (November 2026) --- Status note on the restructure and the Path 0 fixes; stale entries struck (vanishing-factor disagreement, post-harvest prestige accrual, the new_questions conflation); full `file:line` re-anchor pending.
