---
title: "Empirical Discipline for the Proxy-Failure ABM: What the Klann Co-Authorship Data Can Calibrate and Validate"
version: 1.0
date: 2026-11-13
tags: [connections, calibration, validation, proxy-failure, bibliometrics, abm]
project: sciencemodels
revision-vehicle: strikethrough
revision-renders: "md, pdf, html"
---

**Version 1** (November 2026)

This memo surveys the reviewed repository `/workspace/git/niklasklann-project`
(Niklas Klann, MA thesis *Collaboration Patterns in Economics*, EconLit Top-30
economics journals 1990--2018, extended to 2023 for citations) and proposes
concrete ways its data and measures could empirically discipline our Mesa ABM
of proxy failure with focus on collaborator choice. The reviewed repo was
inspected read-only; all numbers below were either read from its committed
documents or recomputed from its canonical panel files directly.

# Purpose

Our ABM posits a mechanism: scientists choose locations on a knowledge
landscape by reward $= (\text{novelty}/\overline{k})^{c} \cdot
\text{visibility}^{1-c}$, prestige accrues from harvested knowledge, and
visible prestigious collaborators act as a proxy that can decouple from deep
research quality (Goodhart / Campbell / John et al. 2023). The claim is
currently illustrated, not disciplined. The Klann project offers an unusually
good counterpart world: a real co-authorship system where the *same*
distinction is operationalised --- proxy-side output ($P_{out}$ = future paper
count) versus goal-side quality ($G$ = age- and cohort-normalised citation
percentile, column `citedby_pct_age_norm`) --- plus prestige measures, network
structure, and career dynamics on one author-year panel.

# Repository inventory

## Data assets

The Tier-3 canonical datasets in `data/processed/` (ADR-0014) are the only
files downstream estimators read; they are the right interface for us too.

| File | Grain | Sample size | Key columns |
|---|---|---|---|
| `panel_author_year_canonical.parquet` | author x year, 1990--2023 | 81,862 rows x 62 cols, 35,021 authors | `coauth_rank`, `coauth_prestige`, `own_prestige`, `future5`, `citedby_pct_age_norm`, `career_age`, `n_authors`, `repeated_collab_share`, `mean_topical_closeness`, `treated`, `event_time` |
| `paper_metadata_canonical.parquet` | paper | 60,256 rows x 14 cols | DOI, journal, year, JEL top-levels, `citedby_scopus`, `citedby_scopus_pct`, `topic_id`, author list |
| `citations.csv` | record | 50,210 rows | DOI, Scopus and OpenAlex citation counts, cohort percentiles (year x journal), provenance |
| `rds_jel_final.csv` | record x JEL code | 153,672 rows, 48,002 papers | `jel_code`, `jel_top_level`, label source |
| `author_paradigm.csv` | author | 34,271 rows | `het_share`, `top_share`, first year in corpus |
| `external_institution_ranks.csv` | institution | 615 rows | RePEc IDEAS rank, score |
| `affiliation_openalex.csv` | affiliation string | 9,668 rows | modal OpenAlex institution (ROR), `confident` flag, RePEc rank |
| `econlit_author_affil_full.RDS` | paper x author x affiliation | 113,693 rows | raw thesis pipeline core (journal, affiliation, `affil_weight`) |
| `top30_2019_2023.csv` | paper | 12,255 rows | panel-extension papers 2019--2023 |

The estimation window is `in_core_window` ($t \le 2013$): 50,830 author-years,
25,498 authors (mean 1.99 observations per author). `CHECKSUMS.json` pins
row and column counts with SHA-256 hashes.

## Notebooks

The `notebooks/` family are thin wrappers over `scripts/cc_*.py` (ADR-0002);
all analysis logic lives in the scripts.

- `cc_01_data_exploration` --- builds the dual-outcome panel, reports coverage of every candidate outcome and team control; the goal column `citedby_pct_age_norm` is NaN where an author has no papers in $t{+}1..t{+}5$.
- `cc_02_dual_outcome` --- estimates the dual-outcome equation (Poisson, log link, author and year FE) twice: proxy `future5` against goal `citedby_pct_age_norm`, plus a labelled `citedby_count` divergence check. Reproduces the M1 pilot: pooled $\beta_1(\text{coauth\_rank}) = -0.0001$ ($p < 0.0001$, $N = 50{,}830$) versus within-author $\beta_1 \approx 0$ ($p = 0.79$, $N = 32{,}293$).
- `cc_03_event_study` --- event study around the first prestige-oriented collaboration $A_{PF}$; estimator call present but blocked pending treatment flags in older revisions (flags now exist on the panel).
- `cc_04_treadmill` --- proxy-treadmill dynamics (input escalation, gradient steepening); estimator stub in older revisions.
- `cc_05_plots` --- dual-outcome wedge figures, once per goal column.
- `cc_06`--`cc_11` extend this with JEL coverage, name-conflation audit, co-author-rank circularity audit, descriptives, and flagship diagnostics; `notebooks/ma/` holds the thesis-replication notebooks (dyad and paper models).

## Framework documents and model reports

- `docs/model_framework_author_level_EN.md` --- the estimating equation $Y_{i,t} = b_0 + b_1 P + b_2 K + b_3 D + \alpha_i + \lambda_t + \varepsilon$ with full distribution tables for every variable on the estimation window.
- `docs/variable_registry.md` --- every column, its construction, coverage, and known gaps (the single best data dictionary).
- `docs/CONT-proxy-failure-ma-thesis.md` --- the proxy-failure storyline: assortative structure, Markov club closure, transfers, and the counterfactual design that would state the wedge causally.
- `output/models/` --- committed model reports: `M1_dual_outcome_pilot.md`, `event_study_summary.md`, `joint_wedge_contrast.md`, plus variants (spec-curve, Romano-Wolf, conflation sensitivity).

# Empirical quantities the project measures

- Prestige as past productivity: `own_prestige` (cumulative own papers before $t$), `coauth_prestige` (mean prior stock of year-$t$ co-authors), `past_coauth_output` (mean prior stock of the cumulative partner set), `coauth_rank` / `coauth_rank_pct` (institutional standing of partners, in-sample and RePEc-external).
- Quantity output (the proxy): `future5`, papers in $t{+}1..t{+}5$ --- median 1, mean 1.35, p90 4, 48.5% zeros on the core window.
- Quality output (the goal): `citedby_pct_age_norm`, the within-journal x year-cohort citation percentile of the author's $t{+}1..t{+}5$ papers, itself normalised by career age --- median 0.56, mean 0.54, coverage 25,653 of 50,830 core author-years.
- Proxy-goal alignment: pooled corr(`future5`, `citedby_pct_age_norm`) is only 0.15 (Spearman 0.14) on the common sample --- quantity and quality are related but far from identical, which is exactly the gap a proxy-failure model needs to reproduce.
- Co-authorship network structure: 5 x 5 rank-category O/E matrix (dyad-corrected elite cells 3.1 / 3.0 / 2.3 versus paper-level 2.8 / 2.8 / 2.1), team size (median 2, mean 2.65; collaboration index rising 1.6 to 2.4 over 1990--2018, collaborative share 46% to 79%), `repeated_collab_share` (mean 0.34), Markov chain C on elite network access (retention about 0.70, entry about 0.05; 12,417 authors, 40,037 transitions).
- Complementarity and diversity: `mean_topical_closeness` (JEL-cosine fit with partners, median 0.79), `div_rank_sd` / `div_rank_gini` / `div_jel_entropy` / `div_upward_share`, `jel_shannon` author breadth.
- Concentration: top-1% of authors hold 9.8% of career papers and 12.7% of the cumulative prestige stock; top-10% hold 40.8% and 55.5% respectively; thesis pipeline Gini about 0.85.
- Treatment dynamics: the $A_{PF}$ event design (first elite partnership with low topical fit) records 2,661 treatments under definition B, of which 1,166 form the observable event sample; the joint wedge contrast (Gelman-Stern test) gives a quantity-quality coefficient difference of $-0.03$ SD with 95% CI $[-0.103, +0.044]$ --- a bound, not a rejection.

# Mapping model variables to empirical counterparts

| ABM quantity | Empirical counterpart | Source column or statistic |
|---|---|---|
| Agent prestige (`prestige`, `prestige_visibility`, `prestige_vanishing`) | accumulated scientific standing | `own_prestige`, `past_coauth_output` (author-year panel) |
| Visibility of a collaborator | partner institutional standing | `coauth_rank_pct`, `coauth_rank_ext_pct` |
| `current_local_merit` (quantity merit) | realised output rate | `future5`, `n_papers_t` |
| `ideal_Merit` (goal merit) | age-normalised citation impact | `citedby_pct_age_norm` |
| `Corr_prestige_local_merit` | pooled corr(prestige, future output) | 0.50 (own\_prestige), 0.19 (coauth rank pct) |
| `Corr_prestige_ideal_merit` | pooled corr(prestige, citation impact) | 0.07 (both prestige measures) |
| `Corr_prestige_visibility_*` | corr(partner standing, outcome) | 0.19 vs 0.075 (quantity vs quality) |
| Top-1% / top-10% prestige shares | concentration of output and stock | 9.8% / 40.8% (papers), 12.7% / 55.5% (stock) |
| `percentage_knowledge_harvested` | system output flow | papers per author-year (mean 1.15 in $t$) |
| `explored_percentage` | topical breadth of the field | `div_jel_entropy`, `jel_shannon`, `div_upward_share` |
| `avg_distance_between_agents` | topical distance between collaborators | `mean_topical_closeness` (median 0.79) |
| Agent entry (supervisor-proportional generation) | entry into the Top-30 system | Markov C entry 0.05, first-year mass (median career age 1) |
| Reward normalisation by $\overline{k}$ | cohort and age normalisation | `citedby_pct_age_norm` construction (year x journal percentiles) |
| Curiosity heterogeneity | career-age gradient of exploration | diversity and K by `career_age` bins |

# Calibration and validation options

Each option names the moments it uses, whether they are calibration targets or
held-out validation moments, and which ABM parameters they discipline. The
practical recipe for all options is identical: run the model, rebuild a
simulated author-year panel with the same variable definitions
(`cc_data_prep.build_cc_panel()` is the template), and match a moment vector by
simulated method of moments or approximate Bayesian computation.

## Option: calibrate the prestige wedge itself

- Calibration moments: pooled corr(prestige, quantity) about 0.50 and corr(prestige, quality) about 0.07 on the same sample (own\_prestige version; the co-author-rank version is 0.19 versus 0.075).
- Validation moments: the within-author near-zero prestige slope (M1b, $p = 0.79$) and the joint wedge bound $\lvert \Delta \rvert < 0.103$ SD from `joint_wedge_contrast.md`.
- Disciplines: the reward exponent $1-c$ (visibility weight), the coupling of harvested knowledge to prestige accrual, and any prestige-bias parameter in collaborator choice.
- Why it matters: this is the model's core claim expressed as two numbers the simulation must hit simultaneously; hitting one but not the other is the empirical signature of proxy failure, and the data pin both.

## Option: calibrate concentration and cumulative advantage

- Calibration moments: top-1% and top-10% shares of career output (9.8% and 40.8%) and of prestige stock (12.7% and 55.5%), directly comparable to `top1pct_prestige_share` and `top10pct_prestige_share` in the Mesa DataCollector.
- Validation moments: the output Gini (about 0.85 in the thesis pipeline) and the full career-output distribution (median 1, p90 5, p99 16 papers per author).
- Disciplines: `harvest` (the per-step extraction rate), `epsilon` (noise in the reward measure), and the supervisor-proportional generation rule (which sets the skew of agent endowments).
- Why it matters: heavy-tailed prestige is the observable footprint of the model's cumulative-advantage mechanism; if simulated concentration is flat, the mechanism is missing regardless of the correlation fit.

## Option: calibrate partner-choice structure

- Calibration moments: team-size distribution (median 2, mean 2.65) and `repeated_collab_share` (mean 0.34, mass at 0 and at 1), plus the growth of the collaboration index from 1.6 to 2.4.
- Validation moments: the dyad-corrected O/E elite homophily cells (3.1 / 3.0 / 2.3) and Markov chain C (retention 0.70, entry 0.05).
- Disciplines: the visibility mechanism in partner selection, any assortativity or mixing parameter the collaborator-choice extension gains, and `initial_curiosity` via the exploration-exploitation mix.
- Why it matters: homophily and club closure are the structural facts that make a prestige proxy *possible*; if the simulated network does not reproduce them, the proxy channel has no stage to act on.

## Option: calibrate career flows and the output zero-gate

- Calibration moments: 48.5% zero mass in `future5`, papers-per-active-year mean 1.15, career length within the Top-30 window (median 1 year observed, p90 12).
- Validation moments: the `future5` upper tail (p90 = 4, max 26) and the `citedby_pct_age_norm` distribution (median 0.56, roughly symmetric on 0--1).
- Disciplines: `agent_generation_time` (entry rate), `harvest`, and the exit rule if one is added; it also anchors how many steps of the model one empirical year maps to.
- Why it matters: the zero-inflated, short-observed-career shape is what the simulation's population must look like *before* any prestige comparison is meaningful.

## Option: hold dynamics out as the falsification battery

- Calibration moments: none --- this option spends no moments on fitting.
- Validation moments: the $A_{PF}$ event-study paths (treatment = first elite, low-fit collaboration; 2,661 treated, 1,166 observable events), treadmill escalation of `coauth_rank` over careers and calendar time, and the rarity of elite entry (0.05 per transition).
- Disciplines: nothing at fit time; it tests the fitted model's dynamic claims about proxy failure --- whether a first prestige-oriented collaboration in the simulation produces the same quantity-quality divergence pattern seen in the panel.
- Why it matters: a proxy-failure model earns its keep on dynamics, not levels; this is the one option that can genuinely reject the mechanism.

# Moment menu

| Moment | Empirical value | Role |
|---|---|---|
| corr(prestige, future5), pooled | 0.50 | calibration |
| corr(prestige, citedby\_pct\_age\_norm), pooled | 0.07 | calibration |
| corr(prestige, future5), within-author | about 0 | validation |
| Joint wedge contrast | -0.03 SD, CI -0.103 to +0.044 | validation (bound) |
| Top-1% output share | 9.8% | calibration |
| Top-10% output share | 40.8% | calibration |
| Top-1% / top-10% prestige-stock shares | 12.7% / 55.5% | calibration |
| Output Gini | about 0.85 | validation |
| future5 zero share | 48.5% | calibration |
| Team size mean | 2.65 | calibration |
| Repeated-collaboration share | 0.34 | calibration |
| O/E elite homophily (dyad-corrected) | 3.1 / 3.0 / 2.3 | validation |
| Markov C retention / entry | 0.70 / 0.05 | validation |
| corr(future5, citedby\_pct\_age\_norm) | 0.15 | validation |
| Mean topical closeness with partners | 0.79 | validation |
| Elite-entry event treatments | 2,661 recorded, 1,166 observable | validation |

# Data-availability caveats

- Author identity is name-confounded (BUG-1); the identity layer estimates about 10.7% merge error. Any moment built on career stocks inherits this, and it biases concentration measures downward.
- The panel is short: mean 1.99 observations per author in the estimation window, median observed career 1 year. Within-author moments (the most model-relevant kind) are estimated on a selected minority (32,293 of 50,830 rows).
- `citedby_pct_age_norm` covers only about 50% of core author-years (no $t{+}1..t{+}5$ papers, no citation match). OpenAlex counts cover 96.3% of articles, Scopus 75.2%, either 92.6%; the AER pre-2000 gap is a known defect (P-0001).
- Right-censoring: `future5` and the goal column require $t \le 2013$; the 2019--2023 block of the panel is deliberately sparse (identity, `n_papers_t` only).
- The in-sample prestige rank is partly circular (it is built from the same publication records the outcome counts); the RePEc external rank bounds the objection but covers only 48.4% of author-paper rows.
- The joint wedge contrast conditions on publishing at all (0% zeros in its sample versus 48.5% in the core window) --- intensive-margin moments only.
- Solo papers are excluded from the Markov C chain, truncating the lower tail of the entry distribution.
- No gender, PhD-origin network, or field-mobility measures; the JEL layer supplies topical breadth but the hybrid K window uses mild look-ahead (ADR-0003).
- EconLit raw extracts are AEA-licensed with no redistribution; any shared calibration bundle must ship derived moments, not raw rows (the canonical panel plus `CHECKSUMS.json` is the intended interface).
- Two structural mismatches to keep in view: the model's knowledge landscape is synthetic (no empirical counterpart for absolute knowledge values, only for normalised ranks and percentiles), and the model has no explicit team-production step --- collaborator choice exists in the data at the paper level and in the model only through visibility coupling, so Options that need team-size moments first require a small model extension (a co-authorship or team-formation step).

# Suggested sequence

- Start with Option (wedge) and Option (concentration): both run on the existing model code with no extension, and together they discipline the reward and prestige-accrual parameters.
- Add Option (career flows) to anchor the time scale and the population.
- Option (partner choice) motivates a co-authorship extension of the ABM; it is the highest-value extension because it puts the model's central actor --- the collaborator --- on the stage the data actually observe.
- Reserve Option (dynamics) untouched until the levels fit; then let the event study and the treadmill decide whether the mechanism survives.

# Revision Log

- **Version 1** (November 2026) --- Initial memo connecting the Klann data to the ABM.