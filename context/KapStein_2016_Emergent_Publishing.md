---
title: "Kapeller & Steinerberger (2016) – Emergent Phenomena in Scientific Publishing: A Simulation Exercise"
version: 1.0
date: 2026-11-13
tags: [KapStein-2016, simulation, peer-review, journal-rankings, emergent-phenomena, proxyeconomics, sciencemodels]
project: sciencemodels
---

**Jakob Kapeller, Stefan Steinerberger**  
*Research Policy*, 45 (2016), 1945–1952 (short communication). DOI: 10.1016/j.respol.2016.08.004

## Core Claim

The paper simulates the academic production process — creation, submission, evaluation and publication of papers — under deliberately **overly optimistic assumptions**: authors, referees and journals are completely objective, journal rankings are completely efficient, and opportunistic behavior, feuds and distorting factors are absent. Even so, the process gives rise to **universal emergent phenomena for purely mathematical reasons**: the efficiency of scientific publishing is delicate and very unstable, and already a tiny amount of noise fundamentally alters basic properties of the academic production process for the worse. The results supply "an additional point of departure for a critical assessment of the inner routines of academic institutions" without invoking human fallibility, hot topics, or grant/tenure pressure.

## Motivation

- Peer review is known to lack robustness (Gans & Shepherd 1994) and objectivity (Bedeian 2003) and can create shrewd incentives (Macdonald & Kam 2010; Day 2015).
- Citation metrics are contested: skewed attention distributions (Solla-Price 1965), internal-validity problems (relative impact used as a proxy for quality; Amin & Mabe 2000), conceptual biases (Kapeller 2010), and reactive behavior (Reedijk & Moed 2006).
- The paper deliberately takes the opposite tack: assume these problems **do not exist** and explore what properties of scientific discourse remain — the residual phenomena are then structural, not behavioral.

## Model Setup

### Two scenarios

| Scenario | Assumptions |
|-----------|-------------|
| Idealized | Authors have perfect insight into their paper's quality, submit to the appropriate journal, referees judge perfectly. |
| Noisy | Authors have only approximate understanding, act strategically in submissions; referees guess true quality up to a small error. |

In the idealized scenario the journal landscape is well ordered, papers land in journals whose reputation closely matches paper quality, and rejection rates are low. The focus is on the **noisy** scenario; "universality" means the phenomena are independent of the actual parameter values and must be described qualitatively. The noisy structures are stable under parameter perturbation; the idealized scenario is highly sensitive to even very slight changes.

### Scientists and manuscripts

- $N = 200$ scientists $sci_1, \dots, sci_N$ with skill drawn randomly, $N(0.5, 0.2)$ (mean $0.5$, standard deviation $0.2$).
- Each scientist writes one paper whose quality fluctuates around her skill:

$$qual_j = sci_j\,(1 + N(0, 0.2))$$

- Even "weak" scientists occasionally produce outstanding work; outstanding scientists occasionally produce much worse papers — quality is skill plus Gaussian fluctuation.

### Journals and submissions

- $k = 25$ journals, each publishing the best 12 papers submitted to it per period: 300 slots for 200 papers (the average journal receives roughly 8 submissions). This is intentionally **overly optimistic** — abundant slots — so results are no artefact of journal scarcity.
- Journal prestige $jour_1, \dots, jour_k$ is a real number (eventually a moving average of published quality).
- Submission process: the scientist estimates her paper's quality,

$$est_j = qual_j \quad \text{(idealized)}, \qquad est_j = qual_j\,(1 + N(0, 0.2)) \quad \text{(noisy)}.$$

  The noisy estimate is still very optimistic — a typical error of only 20%.
- She submits to the journal whose reputation is closest to $a_1 \cdot est_j$; after rejection, to the one closest to $a_2 \cdot est_j$; after a second rejection, $a_3 \cdot est_j$. At most three submissions. Parameters:

$$(a_1, a_2, a_3) = (1, 0.9, 0.8) \ \text{(idealized)}, \qquad (a_1, a_2, a_3) = (1.1, 1, 0.9) \ \text{(noisy)}.$$

  Idealized authors are modest and accept a slightly lower tier after rejection (a2, a3 barely matter since rejection is rare); noisy authors are slightly more ambitious at first. Journal reputation is universally agreed upon and available to all scientists.

### Evaluation

- Each journal requests a referee report per submission; refereeing is double-blind (the referee does not know the author's skill) and aims at maximum accuracy:

$$ref_j = qual_j \quad \text{(idealized)}, \qquad ref_j = qual_j\,(1 + N(0, 0.2)) \quad \text{(noisy)}.$$

- Multiple referees are subsumed by a Gaussian identity: if one report has error $N(\mu, \sigma)$, the average of $k$ independent reports satisfies

$$\frac{ref_1 + ref_2 + \dots + ref_k}{k} \sim N\!\left(\mu, \frac{\sigma}{\sqrt{k}}\right),$$

  so requesting more referee reports is modeled by changing a parameter.
- Journals accept the "best" papers (max 12 per period) in up to three acceptance rounds, then close for the period.

### Journal re-evaluation

Journal reputation is updated by a weighted average of current reputation (80%) and the average intrinsic quality of papers published in the last round (20%):

$$jour_j \leftarrow \frac{4 \cdot jour_j}{5} + \frac{\overline{qual_j}}{5},$$

  unchanged if the journal published nothing. Journals publishing above their reputation rise; reputable journals publishing weak papers slowly lose standing. Journal rankings are thus assumed **completely efficient and objective in both scenarios**. After re-evaluation the cycle restarts (scientist skills fixed, reputations updated). The initial journal landscape is an equidistant partition of probability mapped through the inverse CDF of $N(0, 0.2)$; as in models of this type, the initial landscape is irrelevant after sufficient simulation time (confirmed in experiments).

### Scope remarks

- The model **drastically underestimates** intrinsic randomness: all variables rest on undistorted observations of "true" quality; personal feuds, research trends and competing field sizes are ignored. Focus is on phenomena arising already from a minimum of noise.
- Empirical plausibility: closest to fast-moving natural-science subdisciplines with many groups on the same questions (three rejections ≈ a year passed ≈ results outdated).

## Summary of Results

| \# | Finding | Content |
|----|---------|---------|
| 1 | Top journals | Very few top journals, publishing almost exclusively excellent papers; their position is stable over time. |
| 2 | Clustering | Outside the top, many journals are virtually indistinguishable; their ranking is not stable. |
| 3 | Variation | In any non-top journal the quality of published papers varies considerably; almost all journals feature surprisingly good and surprisingly bad contributions relative to their ranking. |
| 4 | Rejection rate | Under ideal parameters, rejection probability decreases with paper quality; under even slightly imperfect conditions it **increases** with quality (until the very best papers, where it drops sharply). |
| 5 | Journals as bottleneck | The journal landscape deviates from the quality of papers produced; the misfit acts as a bottleneck creating unnecessarily high rejection rates and slowing publication, especially for very good but not stellar papers (top 20%, not top 5%). |

## Universally Emergent Phenomena I: Patterns of Quality

The transition from idealized to noisy is not gradual: a very small deviation from ideal parameters has a huge impact; all phenomena below are stable under perturbing parameters or changing distributions.

### Top journals

Only the ranking of top journals is stable in the noisy scenario (Figs. 2–4); all other journal rankings are volatile (idealized rankings are perfectly stable, Fig. 1). Mechanism: scientists aim at higher-tier journals, so the best journals receive many submissions, and volume compensates referee error — the journal can afford to publish only those papers where both authors and referees believe the paper to be excellent. A referee misjudging a mediocre paper upward is as likely as misjudging a stellar paper downward to merely "excellent"; with many submissions the top journal selects on the joint event. The result mirrors practice: most disciplines know a very small number of traditionally outstanding journals.

### Clustering

Outside the top there is a large herd of very similar journals (Figs. 2, 4). Mechanism: ambition (authors prefer prestigious outlets) plus erroneous judgment by authors and referees; "middle of the pack" journals lack a clear enough profile to attract top-journal submission volumes and are thus more exposed to random quality fluctuations. Occasionally a journal gets lucky with outstanding submissions and separates from the herd (or loses its special status and merges back).

### Variation

Journal reputation is a good proxy for paper quality in the idealized scenario — but in the noisy scenario **this relation fails drastically**. Top journals feature almost exclusively good or outstanding papers; all other journals show a surprising variety. In the idealized scenario the "worst" journal's papers cluster tightly in the worst-quality region (Fig. 5); in the noisy scenario the mean quality there **goes up** and a tail appears — some papers in the least ranked journal are far above average (Fig. 6).

### Empirical illustration

Many model assumptions (e.g. "intrinsic quality") have a Platonic component, but for the first two observations a rough comparison is possible: Impact Factor series for all journals of three subject categories (full populations on the left, the 9 journals around the median on the right, Web of Science data; the empirical figure shows these Impact Factor plots). The comparison remains imperfect since Impact Factors are themselves a contested indicator of journal quality.

## Universally Emergent Phenomena II: Rejections and Bottlenecks

### Rejection rate

Rejection here means the paper remains unpublished after three attempted submissions. Idealized: high rejection of low-quality papers, sharply decreasing once minimal standards are met — the system works. Noisy: the clustering effect creates many mediocre journals, so rejection of below-average papers becomes negligible (many appropriate outlets; little chance all are full). Instead, rejection probability **increases with paper quality** until the paper is among the very best, then decays — a "bad" referee report downgrades "stellar and revolutionizing" to "outstanding and important", which is still likely to be accepted. Figs. 8–9 plot paper density against rejection rates in both scenarios.

### The NIPS experiment

A comparable empirical bleakness: at NIPS 2014 (top machine-learning conference), 10% of submissions (166 papers) were reviewed by both independent committees (Cortes & Lawrence 2015). A paper accepted by one committee had a **more than 50% likelihood of being rejected by the other**. With an overall acceptance quota of 22.5%, completely randomized decisions would imply 77.5% cross-committee rejection; the observed outcome is thus actually much more drastic than the model's near-reliable top-tier prediction.

### Journals as bottleneck

Excellent journals publish excellent papers (they can be picky), but the converse fails drastically: excellent papers appear in a wide variety of outlets or are never published. Since $sci_j \sim N(0.5, 0.2)$ and $q_j = sci_j(1 + N(0, 0.2))$, the paper-quality distribution is essentially Gaussian and an ideal journal landscape would match it — every paper a suitable outlet. Outside the idealized scenario this hardly occurs (Figs. 10–11): papers in the top 20% but outside the top 5% face a profound lack of journals and often settle lower, while the low end faces an abundance of outlets. In the idealized scenario most of the best 20% of papers land in the best 20% of journals; in the noisy scenario many (roughly 40%) of the best papers **do not get published at all**. This dynamics acts as a bottleneck slowing the publication process.

## Robustness and Model Variations

The model is remarkably robust; natural extensions left open: editorial desk rejection, dynamically adapting journal strategies, evolving scientist skill, multi-author papers, social network formation.

### Intensity of randomness

All base errors are $1 + N(0, 0.2) = N(1, 0.2)$ (unified to avoid extra parameters). Smaller variance moves the model gradually — not sensitively — toward idealized. With self-estimation and referee error tightly clustered at $N(1, 0.05)$, homogenization of journals happens slower but is equally inevitable (Fig. 12, left). A more conservative reputation update (current-issue weight 20% → 5%) yields a tighter middle cluster but leaves the noisy pattern intact (Fig. 12, right).

### Eight variations

Three independent sources of uncertainty:

1. Scientists unsure about their work's intrinsic value: $est_j = qual_j(1 + N(0, 2))$ (printed as $N(0, 2)$; evidently a typo for $N(0, 0.2)$, cf. §5.1).
2. Scientists unsure where to submit (**ambition**).
3. Referees unsure about submitted quality: $ref_j = qual_j(1 + N(0, 2))$ (same typo).

All eight presence/absence combinations were simulated (landscape in Fig. 13). Findings:

- The only scenario not already described is a **downward trend in journal quality with only a few journals surviving**: most journals' reputation tends to zero; the few with nonzero reputation differ strongly from each other. Cause: ambition floods high-tier journals with low-quality submissions; a journal's fate depends on its "defence mechanism" against such floods — and decisively **not** on having good referees but on whether **self-estimation error** exists. If authors are uncertain about their own quality, submissions arrive with mixed quality instead of as bulk of similar papers, which protects the journal population.
- Perfect self-understanding + modest submission (no ambition) reproduces the idealized scenario **regardless of referee error**: every journal receives submissions precisely at its level, so referee errors are inconsequential.
- **Ambition** causes journals to receive many below-level submissions, driving the perverse rejection-rate and bottleneck effects of Section 4; with self-estimation error, journals are quickly overwhelmed and more likely to reject genuinely appropriate papers.
- **Self-estimation error** both aggravates ambition's effects and creates the noisy journal-quality patterns of Section 3; even without ambition it suffices to partially collapse the idealized scenario.
- **Referee error** has no profound structural impact; it merely intensifies or weakens the intensity of the observed unexpected outcomes.

## Conclusion

- In a perfect world (accurate quality estimates, no strategic behavior) peer review is a viable tool for objectively clustering research into outlets of different quality.
- Even minimal deviation from ideal conditions drastically affects outcomes: tiny misjudgements and minimal strategic considerations produce journal clustering and high quality variability among mediocre journals.
- The rejection-probability profile is highly idiosyncratic: overall rejection rate **increases with manuscript quality for a majority of papers**, tightly coupled to the mismatch between paper quality and journal quality (the bottleneck).
- These phenomena add a new dimension to the traditional criticism of academic publishing: even in the absence of human fallibility, hot topics and grant/tenure pressure, potentially harmful phenomena emerge for purely mathematical reasons.

## Relevance to the Project

- Direct instance of **proxy failure**: journal reputation is used as a proxy for paper quality; the paper shows the proxy fails "drastically" outside the top journals (its own §3.3 wording) even when the ranking mechanism is completely objective — i.e. proxy divergence without gaming.
- Complements the Braganza (2020) equilibrium-sample-size model and the John et al. (2023) proxy-failure framework: here the emergent phenomenon is generated by noise plus ambition, not by deliberate gaming — an argument that proxy failure is an inherent risk of the evaluation architecture itself.
- Relevant to "Consequences of Collaboration"-type arguments: prestige hierarchies (of journals, and by extension of co-authors) cannot carry the informational load assigned to them, structurally.

## Caveats

- Model randomness is unrealistically low by design; the claim is only that curious phenomena arise *already* from minimal noise, not that the magnitudes are calibrated.
- "Intrinsic quality" of papers and journals is Platonic and unobservable in practice; empirical validation is restricted to qualitative Impact Factor patterns, themselves contested.
- Rejection rates and slot numbers (300 slots for 200 papers) are deliberately unrepresentative of real scarcity.
- Journal rankings assumed completely efficient and objective — the noise is in estimation and ambition, not in the rankings themselves; a fortiori results hold if rankings are also distorted.
- No welfare analysis or policy proposals: the paper is diagnostic, offering critique-anchoring phenomena rather than fixes.
- Typographic quirk: the error variances in §5.2 are printed as $N(0, 2)$ although §5.1 fixes all errors at $N(0, 0.2)$.
