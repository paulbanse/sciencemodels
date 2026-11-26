---
title: "Way Forward: Exploration Paths from the Proxy-Failure ABM to a Paper"
version: 5.0
date: 2026-11-15
revision-vehicle: strikethrough
revision-renders: "md, pdf, html"
tags: [way-forward, proxy-failure, collaborator-choice, abm, research-design]
project: sciencemodels
---

**Version 5** (November 2026)
**Version 4** (November 2026)
**Version 3** (November 2026)
**Version 2** (November 2026)
**Version 1** (November 2026)

This document distills the exploration round of November 2026 --- the code
audit (`conceptual_degrees_of_freedom.md`), the literature review
(`related_publications.md`), and the empirical-connection memo
(`connections_niklas.md`) --- into a decision document: which exploration
paths could turn the current model into a publishable paper, in what order,
and what would count as failure. It is deliberately a menu with gates, not a
plan with commitments.

# Where we stand

The model today is an illustration, not yet a discipline. It is a Mesa ABM on
a toroidal epistemic landscape: agents search tiles by a Cobb-Douglas reward
$R = (k/\overline{k})^{c} \cdot v^{1-c}$ that trades relative novelty against
visibility, harvest depletes tiles, prestige accrues from what agents sit on,
and the reporting layer measures how far prestige tracks merit
(`Corr_prestige_*` against local and ideal merit). Three facts from the audit
should govern every decision below.

- The collaborator channel is deliberately bracketed. ~~The proxy channel is switched off.~~ Supervisor-proportional entry
  (~~`mesa_model.py:514-523`~~ `sciencemodels/model.py:389-420`) --- the literal "choose the prestigious
  collaborator" mechanism --- does not run (`agent_generation_time = 0` in
  both `scripts/runner.py` and `scripts/bulk_runner.py`) **by design**: the project sequenced
  the quality-prestige interplay before introducing collaboration as a
  distinct element, and wants the impact of alternative modelling choices
  understood first (author clarification, 2026-11-13). The sequencing is
  sound and is the justification for the phase order below; the bracket
  bounds every claim rather than being a defect. One caveat: the bracket is
  partial --- the visibility reward already couples agents socially
  (`use_visibility_reward = True` weights proximity by peers' prestige), so
  prestige-mediated *attention* is active while dyadic *choice*, team
  production, and credit sharing are not.
- Collaborator choice exists only by implication *(pre-Path-B baseline; explicit dyadic layer added in `sciencemodels/` and probed in `output/reports/path_b_probe_v1.md`, 2026-11-14, gated off by default)*. The visibility term pulls
  agents toward prestigious company, and prestige-proportional supervisors
  assign newcomers, but there is no dyadic choice, no team production, and no
  co-authorship object in the baseline model. The data we want to discipline against
  (Klann panel) observe collaboration at the paper level. This structural
  mismatch is the single most consequential gap.
- ~~Several measurement choices bend the very quantity we want to study.
  Prestige accrues from the post-harvest tile value rather than the harvested
  flow (`mesa_model.py:204-210`); the `new_questions` flag conflates knowledge
  normalization with the refill mechanism; the prestige memory constant
  disagrees across configurations (0.0 / 0.5 / 0.9); newcomers bypass the
  curiosity generator. These are fixable in days and each changes the
  prestige-merit correlation we report.~~ *(Resolved in Path 0 / `33421ff` and the N1/N2 pairing audit `d3503ac`; verified in `output/reports/remeasure_v1.md`.)*

Against this, the exploration round produced three assets: a verified map of
~~34~~ 46 related studies (`docs/related_publications.md` v2.0) showing the niche is real and unoccupied (no existing model
combines landscape search, visibility-proxy incentives, and partner selection
scored against an ideal allocation); a ~~104~~ 106-entry catalogue of every modelling
assumption with its conceptual role (`docs/conceptual_degrees_of_freedom.md` v5.0); and an empirical anchor world --- the
Klann co-authorship panel --- whose prestige wedge (corr(prestige, quantity)
$\approx$ 0.50 versus corr(prestige, quality) $\approx$ 0.07) is precisely the
signature the model claims to explain.

# The claims the paper could make

Three candidate contributions, in descending order of distinctiveness:

- Proxy failure is *relational*: it can arise endogenously from partner
  selection under a visibility proxy, without any agent gaming a metric
  consciously. No published model demonstrates this (gap 1 and 2 of the
  literature review).
- The Matthew effect can be *quantified against ground truth*: ideal merit
  separates earned prestige from self-reinforcing visibility exactly, which
  bibliometric data never can (gap 3).
- The mechanism is *empirically disciplined*: the model must hit the observed
  prestige wedge and concentration moments simultaneously, and survive
  held-out dynamic tests (the Klann moment menu).

A paper needs at least the first plus one of the other two. Claim two alone is
a methods note; claim three alone is a calibration exercise.

# Exploration paths

## Path 0: Measurement hygiene (prerequisite, not a path)

**Completed 2026-11-13 (W1 round, `33421ff`): every item below is done** ---
prestige accrual, normalization split, vanishing unification (0.9), milestone
renames, newcomer curiosity routing --- plus founder-explored accounting and
BUG-0001 (the newcomer path was not merely off, it crashed).

Fix the audit findings that bias the quality-prestige interplay itself --- these are measurement questions about the Phase 1 object of study, not consequences of the collaboration bracket:

- Prestige accrues from harvested flow $h \cdot k$ (or report both variants as
  a measurement sensitivity).
- Disentangle normalization from refill: two flags where `new_questions` now
  serves both.
- Unify `vanishing_factor_for_prestige` across constructor and runners; pick
  one regime, sweep it deliberately.
- Route newcomer curiosity through `agent_generation_function`; repair the
  milestone-threshold names; note the founder-tile "explored" artifact.

Cost: small. Value: without this, every correlation we report is partly an
artifact of inconsistent defaults.

## Path A: The wedge --- explore the mechanism against the documented empirical moments

**Owner ruling (2026-11-13): no calibration of the model to empirical data at
this point.** The empirical moments stay documented options
(`connections_niklas.md`, `docs/calibration_moments.yaml`); this path is
model-internal exploration --- which parameter regions produce a wedge-like
decoupling --- not a fit to the data.

No model extension. Run the existing model with entry activated, and ask
whether the empirical signature emerges: a high prestige-quantity correlation
together with a low prestige-quality correlation (0.50 vs 0.07), plus the
observed concentration (top-10% shares 40.8% output / 55.5% prestige stock).
Sweep the visibility weight $1-c$, the prestige stock that weights attraction
(`prestige` vs `prestige_vanishing` vs `prestige_visibility`), the vanishing
factor, and landscape legibility (`std`, `prop_random`). A priori prediction:
the wedge widens as merit becomes harder to see --- proxy failure worsens when
the landscape is noisy, which is exactly the Braganza/John "weak proxy"
argument made dynamic.

- Moments: the wedge pair and concentration shares (calibration); within-author
  flatness and the joint wedge bound (validation).
- Deliverable: an existence proof with quantitative fit --- "this mechanism
  generates the observed decoupling at plausible parameter values".
- Deliverable (second): a structured sensitivity map of how the alternative
  modelling choices in `conceptual_degrees_of_freedom.md` shape the
  quality-prestige relation --- the bracket's own justification (understand
  the impact of alternative choices before adding collaborations) turned
  into a result.
- Main risk: identifiability. Curiosity and the visibility weight move the
  same moments; mitigation is the moment battery (they act differently on
  concentration and exploration breadth) and a formal sensitivity profile
  over the catalogued degrees of freedom.

## Path B: Collaborator choice made explicit --- the differentiating extension

First step: activate the existing demographic channel --- set
`agent_generation_time > 0`, sweep the entry rate, and route newcomer
curiosity through `agent_generation_function` (the cheapest explicit
collaborator element, already present in code). Then add a minimal dyadic
layer: agents occasionally form or dissolve collaborative
links, choosing partners by visibility (the proxy) versus topical closeness or
merit (the goal); co-production lets linked agents harvest jointly or share
credit. This puts the paper's title noun --- the collaborator --- into the
model as an actor, and unlocks the team moments the Klann panel observes:
team-size distribution (mean 2.65), repeated-collaboration share (0.34),
rank-homophily O/E cells (3.1 / 3.0 / 2.3), and Markov club closure
(retention ~~0.70, entry 0.05~~ 0.3995, entry 0.0619; Markov A top-10% is 0.71 / 0.054).

- Moments: team structure (calibration); homophily and closure (validation).
- Deliverable: the flagship paper --- proxy failure *through* collaborator
  choice, empirically disciplined on both the wedge and the network.
- Main risk: parsimony. The extension adds parameters exactly where the data
  are thinnest (paper-level links, no PhD-origin ties). Keep the dyadic layer
  to one decision rule with one mixing parameter, or the model becomes
  unfalsifiable by accumulation.

## Path C: Welfare and interventions --- the normative paper

Use ideal merit as the planner's benchmark to price visibility pressure: how
much ideal merit is destroyed per unit of visibility weight, and where
exploration collapses. Then test interventions in the Smaldino et al. (2019)
spirit: randomized collaborator acceptance, caps on visibility-weighted
attribution, rewarding epistemic distance in partner choice.

- Deliverable: welfare curves plus intervention ranking; policy resonance with
  the John et al. counterforce discussion and the DobKap pluralism agenda.
- Position: strongest as a section of the flagship, or as a standalone if
  Path A shows a sharp collapse threshold (a "phase diagram" figure is a
  paper on its own in a metascience outlet).

## Path D: Typology formalized --- the theory-companion note

Instantiate the `Hunting_Proxy_Patterns` typology as parameterized variants of
one model: distraction vs destruction ($AP_F$ crowding out $A$ vs directly
harming $G$), direct vs confounded $P$-$G$ links, counterforce on/off. Show
analytically or numerically how the $P$-$G$ correlation decays in each cell,
and where the counterforce (strong $P \to G$) stabilizes honest signalling.

- Deliverable: a short theory note that formally classifies proxy failure ---
  the natural companion piece to John et al. (2023) and the March 2026
  Kapeller-Paul-Braganza note, and the fastest route to a first publication.
- Risk: without empirical or simulation grounding it stays a comment; pair it
  with one Path A figure to give it weight.

# How the empirical data disciplines each path

The Klann panel moment menu (`connections_niklas.md`) assigns each statistic a
role; the mapping in brief:

| Path | Calibration moments | Held-out validation |
|---|---|---|
| A | wedge pair (0.50 / 0.07), top-1% and top-10% shares | within-author flatness, joint wedge bound |
| B | team size 2.65, repeated collab 0.34 | O/E homophily 3.1/3.0/2.3, Markov ~~0.70/0.05~~ 0.3995/0.0619 (C) & 0.71/0.054 (A) |
| C | none (welfare is model-internal) | intervention analogues, event-study paths |
| D | none | sign patterns of the $P$-$G$ decay by typology cell |

The discipline is real but soft at one point: the joint wedge contrast is a
bound (CI $[-0.103, +0.044]$), not a rejection, so the empirical target says
"the wedge is at most this wide", and the model must respect the bound rather
than match a point. The event study around first prestige-oriented
collaborations (2,661 treatments) and the treadmill dynamics stay untouched
until the levels fit --- they are the one battery that can genuinely reject
the mechanism.

# Decision gates

- Gate 1 (after Path 0 plus a Path A sweep): does any plausible parameter
  region produce the wedge *and* the concentration moments simultaneously?
  If no, the visibility-coupling mechanism is too weak and the model needs an
  explicit $AP_F$ action (a "gaming" margin) --- that failure is itself a
  result and redirects to Path D.
- Gate 2 (after the Path B prototype): does the dyadic layer materially
  improve the moment fit, especially the team and closure moments? If yes,
  the flagship paper is B+A; if no, the honest paper is A+D (mechanism plus
  typology), and the team layer is reported as a robustness extension.
- Gate 3 (last): run the falsification battery. If the fitted model gets the
  levels but wrong dynamics (no wedge emergence around prestige-oriented
  moves, wrong treadmill direction), the mechanism story must be rewritten
  before submission, not after the first referee report.

# Risks and honest caveats

- Identifiability: several catalogued levers move the same moments. Mitigate
  with the full sensitivity profile, and by preferring claims about *regions*
  of parameter space over point estimates.
- Synthetic landscape: absolute knowledge values have no empirical
  counterpart; only normalised ranks do. Keep empirical claims at the level
  of correlations and shares, never levels.
- Time scale: one model step maps to no empirical year yet. The career-flow
  moments (48.5% zero output, median observed career 1 year) are what anchor
  this; without them, dynamic claims are uncalibrated.
- Data thinness: roughly two observations per author in the estimation
  window; within-author moments rest on a selected minority. State this where
  it bites.
- The bracket is leaky: with `use_visibility_reward = True`, prestige already
  shapes attention even without dyadic choice. Define the bracket
  operationally --- no link formation, no team production, no credit sharing
  --- in one paragraph early, or referees will read collaborator choice into
  the Phase 1 results.
- The just-so risk: an ABM that tells a plausible story about everything
  explains nothing. The pre-declared moment menu, the held-out dynamics, and
  the Gate 1 kill criterion are the defense.

# Paper skeletons

- **Phase-1 paper** --- "Does Prestige Track Quality? Modelling the
  Merit-Prestige Nexus under Alternative Modelling Choices": Path 0 + A
  standalone, with the sensitivity map as its second figure. The smallest
  publishable unit and the direct product of the bracket rationale; the
  empirical wedge is its headline. Outlets: JEI, JASSS, Scientometrics.
- **Flagship** --- "Prestige as a Proxy: How Collaborator Choice Degrades the
  Quality Signal in Science": Path 0 + A + B, welfare section from C, one
  typology figure from D. Outlets: Research Policy, Journal of Economic
  Issues, JASSS.
- **Fast companion** --- "The Proxy Treadmill in Collaboration Networks: A
  Typology and a Model": Path D + one calibrated figure from A. Outlets:
  JEI, Research Policy (comment/innovation), Royal Society Open Science
  (metascience).

# Next actions

**Status as of 2026-11-15 (Version 5):** All four initial exploration actions below and the four Wave-4 rounds (`F1`–`F4`: 198 cells / 3,960 runs across `output/reports/path_b_probe_v2.md`, `output/reports/interventions_v1.md`, `output/reports/typology_v1.md`, and `output/reports/turnover_v1.md`) are **completed**. Proposals `P-0001`–`P-0006` are `exercised`. The synthesized results live in `AdmiralsLog/contributions/CONT1..CONT5`, rejected mechanism hypotheses and active refinements in `AdmiralsLog/ideas/IE-0001..IE-0005`, strategic unexplored paths in `AdmiralsLog/unexplored/UE-0001..UE-0003`, and the next experimental queue in `AdmiralsLog/proposals/P-0007..P-0010`:

- ~~Run the Path 0 hygiene pass and re-run one `bulk_runner` setup to confirm
  the reported correlations move as predicted (this doubles as the audit's
  validation).~~ *(Done: `33421ff`, `d3503ac`, `output/reports/remeasure_v1.md`.)*
- ~~Prototype Path A on the existing sweep machinery with entry activated; the
  first figure to look at is the wedge pair against the visibility weight.~~ *(Done: `output/reports/sensitivity_map_v1.md`, `career_flows_v1.md`, `career_flows_noisy_v1.md`, `prestige_stock_v1.md`, `welfare_probe_v1.md`, `turnover_v1.md`, `notebooks/round_results_report_v1.ipynb`.)*
- ~~Sketch the Path B decision rule on paper (one paragraph of pseudocode)
  before coding; the design trade-off (parsimony vs team moments) deserves a
  decision, not an accident.~~ *(Done: `AdmiralsLog/archive/path_b_dyadic_sketch_v1.md`, `output/reports/path_b_probe_v1.md`, and `output/reports/path_b_probe_v2.md`.)*
- ~~Extract the moment menu into a machine-readable calibration spec (YAML) so
  every later run is scored against the same targets.~~ *(Done: `docs/calibration_moments.yaml`.)*

# Revision Log

- **Version 1** (November 2026) --- Initial distillation from the exploration round of 2026-11-13.
- **Version 2** (November 2026) --- Author clarification folded in: the collaborator bracket is deliberate sequencing, not a defect; Path 0 retitled measurement hygiene with entry activation moved to the front of Path B; added the sensitivity-map deliverable, the leaky-bracket caveat, and a Phase-1 paper skeleton.
- **Version 3** (November 2026) --- Owner rulings folded: no calibration to empirical data at this point (Path A reframed as model-internal exploration of documented options); docs/ holds long-standing conceptual documents, plans move to the AdmiralsLog/ first level when a workplan supersedes them (this file is scheduled to move with the first workplan).
- **Version 4** (November 2026) --- Relocated from `docs/way_forward.md` to `AdmiralsLog/way_forward.md` per owner directive (2026-11-14); pre-restructure `mesa_model.py` anchors and Markov C `0.70/0.05` figures updated via strikethrough; initial Next Actions marked completed with pointers to `output/reports/{remeasure_v1,path_b_probe_v1,paper_phase1_v1}.md`, `AdmiralsLog/contributions/CONT1..CONT3`, and `AdmiralsLog/proposals/P-0001..P-0006`.
- **Version 5** (November 2026) --- Updated after Wave-4 execution rounds `F1`–`F4` (`output/reports/{path_b_probe_v2,interventions_v1,typology_v1,turnover_v1}.md`, 3,960 runs); marked `P-0001..P-0006` exercised and linked `CONT1..CONT5`, `P-0007..P-0010`, `IE-0001..IE-0005`, and `UE-0001..UE-0003`.
