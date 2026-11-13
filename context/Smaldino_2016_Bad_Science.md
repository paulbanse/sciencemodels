---
title: "Smaldino & McElreath (2016) – The Natural Selection of Bad Science"
version: 1.0
date: 2026-11-13
tags: [Smaldino-2016, cultural-evolution, statistical-power, incentives, Campbell-law, simulation, false-discoveries, proxyeconomics, sciencemodels]
project: sciencemodels
---

**Paul E. Smaldino, Richard McElreath**  
*Royal Society Open Science*, 3: 160384 (2016). DOI: 10.1098/rsos.160384

## Core Claim

Poor research design and data analysis persist despite perennial calls for improvement because **incentives favour them**: publication is a principal factor for career advancement, so methods that produce publishable findings are naturally selected — the **natural selection of bad science**. This dynamic requires **no conscious strategizing** — no deliberate cheating nor loafing — only selection on consequences. Some normative methods of analysis have almost certainly been selected to further publication instead of discovery. Improving science therefore requires a shift away from correcting misunderstandings and towards **rewarding understanding**: institutional change, not more guidelines. Supported by a 60-year meta-analysis of statistical power and a cultural-evolutionary simulation model.

## The Argument: Selection Logic

- Campbell's Law (epigraph): "The more any quantitative social indicator is used for social decision-making, the more subject it will be to corruption pressures and the more apt it will be to distort and corrupt the social processes it is intended to monitor" (Campbell 1976, p. 49). Practitioner gloss (Terry McGlynn, tweet): "Number and IF [impact factor] of pubs are what counts."
- Science is a **cultural activity** evolving by variation-selection-retention (Campbell 1965; Popper; Hull; Boyd & Richerson; Mesoudi). Laboratory methods propagate directly (via PhD students starting labs) and indirectly (prestige-biased adoption).
- Darwin's three conditions for natural selection map onto methodology: (i) research practices **vary**; (ii) variation has **consequences** — habits leading to publication lead to competitive positions; (iii) variation is **heritable** — apprentices acquire habits from mentors, peers and successful role models.
- Therefore, when researchers are rewarded primarily for publishing, publication-promoting habits are selected — even though such habits can directly undermine scientific progress. Not a new argument; the paper strengthens it empirically (power meta-analysis) and analytically (formal model). Formalisation has a diagnostic function: if the argument cannot work in theory, it cannot be the correct explanation; formalising also yields clarity and intervention targets.

## Institutional Incentives for Scientific Researchers

### Competition and quantity pressure

- Publication rates have steadily increased (Larsen & von Ins; Bornmann & Mutz), partly via more multi-author collaboration (Nabout et al.; Wardil & Hauert), but possibly also via changing incentives.
- Brischoux & Angelier: junior researchers hired by the French CNRS in evolutionary biology 2005–2013 show persistent increases in publications at hiring — almost twice as many as 10 years earlier (22 in 2013 vs 12.5 in 2005).
- Universities produce far more PhDs than permanent positions (Cyranoski et al.; Schillebeeckx et al.; Powell; Kerr); graduates from non-elite universities face additional job-market discrimination (Clauset et al.). Tenure-track positions at major research universities carry prestige, flexibility and creative freedom and remain desirable; competition continues for grants, promotions, prestige and student placement.
- Peter Higgs: "Today I wouldn't get an academic job... I don't think I would be regarded as productive enough" (The Guardian, 6 Dec 2013). Speculative, but evidence shows only a small proportion of scientists produce the bulk of output and citation impact (Ioannidis, Boyack & Klavans), and most new tenure-track positions go to graduates of a few elite universities with very high publication rates.
- **Innovation rhetoric**: 1974–2014, the frequency of 'innovative', 'groundbreaking', 'novel' in PubMed abstracts rose by 2500% or more (Vinkers et al.) — implausible as real innovation growth; a response to pressure to stand out.
- **Volume**: publication count dominates search-committee decisions (the McGlynn quote). Metrics like the h-index (Hirsch: largest $h$ such that one has $h$ publications with $\geq h$ citations each) combine output and impact, but volume alone counts, especially early-career. The argument applies to **all fields in which the number of published papers, however scaled, benchmarks success**.

### Metric gaming (Goodhart)

- Whenever a quantitative metric is a proxy for a social behaviour, it becomes open to exploitation and corruption (Campbell; Goodhart 1975; Lucas 1976): "when a measure becomes a target, it ceases to be a good measure."
- h-index manipulation: artificial self-citation inflation (Bartneck & Kokkelmans); the "Google Scholar experiment" (López-Cózar et al.) — six fake papers under a fake author, posted to a university server, dramatically raised the authors' Google Scholar h-indices once indexed. Gaming strategies may be invented by cheaters but **propagate through their consequences** (selection, not intention, spreads them).
- Footnote: h-index incentives may also encourage high-risk hypothesizing on 'hot' topics, since being corrected still increases citation counts.

### Publication bias and the false-positive route

- Positive results supporting novel hypotheses are more publishable than negatives, especially in high-impact journals (e.g. *Journal of Personality and Social Psychology* refused failed replications of precognition it had published; Aldhous). Many with non-supporting results never submit (Franco et al. 2014, "file drawer"); some fail to report hypotheses that produced nothing (Franco et al. 2015 underreporting), which further raises false-positive prevalence (Simmons et al.).
- Two routes to positive results: (a) firm theoretical grounding + sufficient power — effortful and slow (Ioannidis 2005; Button et al.); (b) techniques that drive up false-positive rates — faster output *and* more publishable results. Route (b) is doubly advantaged under publication incentives.
- Replication rarely catches poor studies: <1% of psychological research is ever replicated (Makel et al.); failed replications are disputed (Bissell; Schnall; Kahneman); even discredited work keeps being cited (Campanario). Once published, a false discovery permanently feeds the producer's metrics.
- False discoveries arise from fraud and p-hacking (Aschwanden) but also from well-intentioned practice: the dead Atlantic salmon showing "neural responses to emotional stimuli" (Bennett et al.) and "pre-cognitive" students predicting a random number generator (Wagenmakers et al.) — both from standard field methods. Usually false discoveries are not identifiable at a glance; poor or absent theory makes hypotheses nearly random, lowering the probability that a positive result is real (Ioannidis 2005; Gigerenzer 1998); post-data interpretation via biased analysis selection (Gelman & Loken) and HARKing (Kerr 1998) generate further false positives.
- Campbell's Law applied: incentivising publication count makes researchers modify methods to maximize **publishable results, not rigour**. Crucially, no direct strategizing is needed — the giraffe-neck analogy: necks lengthened not by stretching but because longer-necked animals monopolized food and produced more offspring; likewise, methodological traditions spread because their bearers transmit them more successfully. Incentives shape both innovation patterns and selection; strategic innovation need not be common for fitter strategies to spread rapidly.

## Case Study: Statistical Power Has Not Improved

- Power = probability a test correctly rejects a false null (Cohen 1992), given sample size, effect size and false-positive rates. (Footnote: distinguished from "power" generally — the Gestalt probability that one's methods return a positive result given a true effect.)
- Many effects in biomedical/behavioural/social sciences are small (Richard et al.; Aguinis et al.; Kühberger et al.), so high power matters. Low-powered experiments are cheaper (small subject pools, expensive equipment, longitudinal data). Low power clearly causes false negatives; less obviously, it **also increases false-discovery rates and inflates reported effect sizes** (Ioannidis 2005; Button et al.; Lakens & Evers) via weak noise-muting.
- The incentive contrast: under purely cooperative incentives to reveal true causal models, increasing power is good; under publish-or-perish with positive-results bias, **low-power studies are an efficient strategy** — cheap and farmable for significant results, especially when hypotheses only predict "a difference" (Meehl). Conflict of interest between population and individual: the population benefits from high power more than individuals do, and science has **no invisible hand** reconciling them.
- History of exhortation without effect: Cohen (1962) first flagged alarmingly low power; Sedlmeier & Gigerenzer (1989) and Rossi (1990) reviewed 25 power reviews of 1960–1984 — power was low and had **not improved** since Cohen; Vankov et al. (2014) found the same persists.
- **New meta-analysis (this paper)**: Google Scholar search among citers of Sedlmeier & Gigerenzer with terms 'statistical power' and 'review'; all reviews of statistical power of published papers in social, behavioural and biological sciences collected; **19 studies from 16 papers, 1992–2014** (methods/sources in appendix). Combined with earlier data (44 reviews, 1960–2011, figure 1), power for small effects $d = 0.2$ at $\alpha = 0.05$ is very low (**mean 0.24** — three misses in four) and shows **no increase over six decades** ($R^2 = 0.00097$). Caveats: data far from a complete field picture, and power may be **over**estimated because only published (sufficiently powered to pass review) results are observable. Footnote: genuine short-run power gains may be quelled by publication bias — higher-powered studies merely generate more (unpublishable) null results, so low-powered labs keep the advantage.
- Why does low power persist? Two explanation classes: (1) **direct strategic response** (Vankov et al.: "it makes more sense to use finite resources to generate as many publications as possible"); (2) **selection on misunderstanding** — traditions believing poor methods achieve scientific goals crowd out alternatives in the job market and journal space; notable: the common denominator of statistical misunderstandings is that they all make positive results (true or false) more likely — misunderstandings that hurt careers are rare. Reality is likely a mix; the working assumption is that most researchers have internalised honest norms. But historical/evolutionary inference from sparse data cannot resolve the dynamics (Smaldino, Calanchini & Pickett) — hence the formal model.

## An Evolutionary Model of Science

Purpose: verify the logic of the selection argument, refine falsifiable predictions. Evolutionary currency of fitness = number of publications; successful labs spawn more progeny that inherit their methods. Foundation: the analytical McElreath & Smaldino (2015) model (novel + previously tested hypotheses, communication into a literature, exact posterior probability a hypothesis is true given the publication record); here extended to a finite heterogeneous population of $N$ labs.

### Core assumptions

- Each lab has a characteristic **power** $W$ — ability to positively identify true associations; not just formal statistical power but the **entire chain of inference**; can be very high even at low sample size.
- Increasing power also increases the false-positive rate **unless effort is exerted** (signal-detection logic: declaring support for every association gives perfect power but floods false positives; requiring stronger evidence lowers both $\alpha$ and $W$ unless effort grows the dataset). Alternatives (signal threshold/noise as lab traits) are formally translatable; the chosen formalization encodes a **behavioural hypothesis**: researchers reason as if hypotheses were true and select methods that make true effects easy to find — maintaining "effective power" while exaggerating false-positive rates. (Footnote: low-power/high-$\alpha$ methods can themselves be effortful; substituting **rigour** for effort is invited.)
- Increasing effort **decreases productivity** — rigorous research takes longer.

### Structure: two stages

**Science stage** (per time step, each lab, in random order):

1. *Investigation decision*: a new investigation begins with probability $h(e_i) = 1 - \eta \log_{10} e_i$ (4.1), where $e_i \in [1, 100]$ is the lab's characteristic effort and $\eta$ the extent to which effort lowers the research rate. All simulations use $\eta = 0.2$ (keeping $h \geq 0$); labs start at $e_0 = 75$ with $h \approx 0.63$ ($h = 0.625$). Results hold for any monotone decreasing non-negative $h$.
2. *Hypothesis selection*: with probability $r_i$ the lab attempts a **replication** (a hypothesis supported at least once in the published literature); otherwise a novel hypothesis, true with probability $b$ (base rate; possibly 0.1 in some fields, likely much lower in others).
3. *Investigation*: lab $i$ has power $W_i = \Pr(+|T)$; the false-positive rate is convex in power and depends on effort:

$$\alpha_i = \frac{W_i}{1 + (1 - W_i)\, e_i} \qquad (4.2)$$

   (figure 2: curves for $e = 1, 10, 75$; runs initialized at $e_0 = 75$ such that $\alpha = 0.05$ when power $= 0.8$). The **false discovery rate** additionally depends on the base rate $b$ — rarer true hypotheses mean more false among positives (Ioannidis 2005; McElreath & Smaldino 2015). Effort and power are independent in the model (a narrative convenience: effort is the sum of behaviours that raise power without raising $\alpha$; sensible when power is maintained by contingent procedures).
4. *Communication*: a true hypothesis yields a positive result with probability $W_i$; a false one with probability $\alpha_i$. **Positive novel results are always publishable; negative novel results never are**; both confirmatory and disconfirmatory replications are published with probabilities $c_{R+}$ and $c_{R-}$. Pay-offs (prestige, funding, students) accumulate over a lab's lifetime: $V_N$ for a novel result, $V_{R+}/V_{R-}$ for published replications, plus $V_{O+}$ when others replicate one's novel finding and $V_{O-}$ (punitive) when it fails to replicate.

**Evolution stage** (end of each time step):

- *Death*: a random sample of $d = 10$ labs is drawn; the oldest dies (ties broken at random) — age coarsely tracks fragility.
- *Birth*: an independent random sample of $d$ labs; the one with the **highest accumulated pay-off** reproduces — skewing reproduction toward older and more successful labs, but since age does not correlate with pay-off accumulation rate, selection favours strategies that raise pay-offs **most quickly**.
- *Imperfect inheritance*: a new lab (age 0) inherits the parent's power, effort and replication rate, each "mutating" independently with probability $\mu_w, \mu_e, \mu_r$ by a Gaussian increment (mean 0, s.d. $\sigma_w, \sigma_e, \sigma_r$), truncated to the parameter range.
- Equal expected lifespans are unrealistic but conservative: dropout unrelated to publication is just noise in the death rule; dropout caused by publication makes the model a **muted** demonstration of selection strength.
- The literature is truncated to the 1 million most recent results at each cycle (fields have fewer relevant papers; 90% of psychology replications target work <10 years old; data availability declines with article age — Vines et al.) — computationally motivated but empirically defensible.

### Simulation runs

50 runs per parameter set; each run 1 million time steps with data collected every 2000 steps; the **false discovery rate (FDR)** among published positives tracked along with evolvable traits. Goal: illustrative, not exploratory. Java source code in the electronic supplementary material.

| Parameter | Definition | Values tested |
|-----------|-------------|---------------|
| $N$ | number of labs | 100 |
| $b$ | base rate of true hypotheses | 0.1 |
| $r_0$ | initial replication rate | $\{0, 0.01, 0.2, 0.5\}$ |
| $e_0$ | initial effort | 75 |
| $w_0$ | initial power | 0.8 |
| $\eta$ | effort influence on productivity | 0.2 |
| $c_{R+}$ | probability of publishing positive replication | 1 |
| $c_{R-}$ | probability of publishing negative replication | 1 |
| $V_N$ | pay-off, novel result | 1 |
| $V_{R+}$ | pay-off, positive replication | 0.5 |
| $V_{R-}$ | pay-off, negative replication | 0.5 |
| $V_{O+}$ | pay-off, one's novel result replicated | 0.1 |
| $V_{O-}$ | pay-off, one's novel result fails to replicate | $-100$ |
| $d$ | labs sampled per death/birth event | 10 |
| $\mu_r, \mu_e, \mu_w$ | mutation probabilities | $\{0, 0.01\}$ |
| $\sigma_r, \sigma_e, \sigma_w$ | mutation s.d. ($r$, $e$, $w$) | 0.01, 1, 0.01 |

## Simulation Results

### The natural selection of bad science

Ignore replication ($r_0 = \mu_r = 0$). Let power evolve under constant effort ($\mu_e = 0$, $\mu_w = 0.01$): **both $W$ and $\alpha$ go to unity** (figure 3) — all results positive, hence publishable. Proof: the probability a novel test yields a publishable positive is

$$\Pr(+) = \Pr(+|T)\Pr(T) + \Pr(+|F)\Pr(F) = bW + (1 - b)\, \frac{W}{1 + (1 - W)e} \qquad (5.1)$$

which is strictly increasing in $W$ (by differentiation); selection for higher discovery rates pushes power upward until matched by countervailing forces. Unrealistic as stated (labs could never openly support everything; unbounded power is impractical), but institutions can hold **measurable** aspects of power (e.g. statistical power) high — while false positives are notoriously hard for peer reviewers to assess (MacArthur; Eyre-Walker & Stoletzki). Then publication rates can still rise by **cutting effort**: with power fixed and effort evolvable ($\mu_w = 0$, $\mu_e = 0.01$), selection favours low-effort labs, raising output at the cost of more false discoveries (figure 4). Incentives for publication quantity select for the continued degradation of scientific practices.

### The ineffectuality of replication

Could easy-to-publish replication plus catastrophic failure-to-replicate penalties deter bad methods? Runs: $r_0 = 0.01$ (empirical psychology rate, Makel et al.), $\mu_w = 0$, $\mu_r = \mu_e = 0.01$; all replications publishable and worth half a novel result; confirming replication adds 10% ($V_{O+} = 0.1$); failing to replicate is catastrophic ($V_{O-} = -100$ — justified: retractions slash citations of prior work (Lu et al.); "money-back guarantee" proposals (Rosenblatt) could be highly punitive). Conditions were **extremely favourable** to replication, since false positives are the results most at risk of failing to replicate.

Findings (figures 5–6):

- Mean replication rate evolved slowly but steadily **up to ~0.08** (weak selection for it: guaranteed publication at half value).
- Evolved replication **could not stave off effort collapse**: low effort raised $\alpha$ until novel hypotheses became more likely than not to yield positives, so cutting effort beat raising replication.
- Even **enforced** replication at 25% or 50% ($\mu_r = 0$, initialized high) only slowed the decline of effort; effort still bottomed out.
- Emphatic caveat: the authors are **not** advocating punitive failure-to-replicate outcomes (even good research fails to replicate); the point is that even such harshness would not stop selection for lower quality.

### Why replication is not sufficient

Replication penalises poor-method labs, but unless **every** published study is replicated several times (ideal but implausible), some labs escape. With finite careers and high network connectivity, marginal returns to the top tier of publications are orders of magnitude above a merely respectable record (Clauset et al.; Ioannidis et al.; Frank). Because power, effort and replication are entangled in the full model, a **simplified non-evolutionary model** isolates the mechanism: power fixed at 0.8; labs are all high-effort (H) or low-effort (L); population 50% H. As printed: $e_H = 0.75$ ($\alpha = 0.05$, $h = 0.625$) and $e_L = 0.15$ ($\alpha = 0.2$, $h = 0.765$) [evidently typos for $e_H = 75$ and $e_L = 15$, which are the values consistent with equation (4.2) and the quoted $\alpha$ and $h$]. Ten baseline steps without replication build a literature; then 100 steps (expected lab lifespan) with each investigation a replication with probability $r$; pay-off distributions over 50 runs (figure 7, $r = 0, 0.01, 0.05$, 110 steps total).

- Without replication, low effort has an unambiguous advantage.
- With replication, **mean** pay-off of high-effort labs can exceed low-effort (they are punished less) — but the **very highest performers are low-effort labs** that neither produced caught false positives nor got caught: the laws of probability guarantee some escapees, and the right tail of the pay-off distribution is theirs.
- When top labs disproportionately shape funding and student placement, this small tail advantage **cascades** into continuous selection for lower effort and more false discoveries.

## Discussion

- **Incentives drive cultural evolution**: publication-quantity incentives, absent countervailing forces, select methods producing the most publishable results → natural selection of poor methods and rising FDR. First illustration of the *evolutionary logic* of inevitable degradation under existing incentives.
- Collateral damage beyond false discoveries: years-long scrupulous work on hard problems loses to shallow high-output work; researchers drawn to complex questions may be jobless.
- **Good science as a public good**: cooperation (quality research at personal fitness cost) vs free-riding. Standard cooperation mechanisms (reciprocity, monitoring, punishment; Rand & Nowak) fail here because cooperator benefits must be measurable in **fitness units**; rigorous coalitions may out-produce, and collaboration is rising — but collaboration can itself be a quantity strategy (small contributions to many projects). Cooperation yields knowledge but not fitness in the cultural evolution of methodology; **purely bottom-up solutions are insufficient** (though attitude change remains vital).
- **Institutional change** is hard (large-scale coordination, costly early adoption; North; Aoki) but needed. Tighter journal standards merely shift gaming opportunities (Martin); incentives to game persist while success is tied to publication. Punishing failures to replicate is inadvisable: important discoveries often look unlikely first; false positives cannot be eliminated; failed replications may themselves be false negatives; replication is difficult or impossible in some fields (large clinical samples, historical data) — over-praising replication favours some fields over others.
- **Changing the fitness landscape** (the incentives for success) is drastic but necessary. Obstacles: funders want 'deliverables' (papers), pressuring output even if hiring criteria change (Ioannidis 2014); evaluators face time/cognitive constraints; researcher quality is a high-dimensionality optimization problem with many ways to be good. Simple metrics (publication rates, impact factor) persist because they give clear unambiguous comparisons — **precisely the qualities that make them exploitable**. The true fitness landscape is multidimensional; support institutions enabling routes to success beyond CV length; resist "the temptation or social pressure to paper count."
- **Robustness**: interpretation holds if (i) negative results are harder to publish or (ii) lowering effort increases output rate. Parameters chosen to err **against** the hypothesis (very high replication rate, extremely punitive failure); no full sensitivity analysis presented; code provided for verification. Model treats all novel publications as equal; adding journal prestige would add selection for **highly surprising results** (more likely false). Evidence: statistical power in high-impact journals is no better (Brembs et al.), while retraction rates correlate with impact factor (Fang & Casadevall) — partly attention, but high-IF journals also reject competent work deemed insufficiently novel.
- **Against excessive cynicism**: many committees evaluate quality, not quantity; most scientists are truth-driven (framing everyone as Homo economicus is over-cynical). The model needs **no ill intent and no incentive-responsiveness** — selection alone favours high-volume labs; active social learning (success-biased copying) would only accelerate the evolution. Plenty of good science persists: publication volume is rarely the only career determinant (topic importance, work quality, peer esteem vary by discipline; where disciplinary norms encode quality selection, they can counteract volume incentives). Where success is largely determined by output metrics, "those who care about quality research should be on high alert."
- **Closing**: quantitative metrics used as proxies to evaluate and reward scientists become open to exploitation whenever exploitation is easier than improving research. Any real solution is patchwork: real merit takes time to manifest, scrutinizing quality is costly, competition is stiff, reviewers need workable assessment, and individuals differ on excellence — simple metrics save time, energy and ambiguity, but their long-term costs are likely great. Institutions must be made to incentivize meaningful, reproducible science.

## Statements

- **Ethics**: meta-analysis used only previously published data; "all simulated scientists were humanely euthanized."
- **Data accessibility**: meta-analysis data from published material (methods/sources in ESM); Java model code in ESM.
- **Funding**: no specific grant. Contributions: both authors conceived/designed and wrote; P.E.S. ran the meta-analysis and model code.

## Relevance to the Project

- The canonical formal statement of **selection-based proxy failure** in science: publication count (and metrics built on it) is the proxy, discovery is the goal; Campbell's Law is quoted as epigraph — direct bridge to John et al. (2023) and the proxyeconomics framing.
- Companion to Braganza (2020): Braganza derives equilibrium sample sizes from publication-bias incentives; Smaldino & McElreath derive evolutionary deterioration of methods from career incentives — two formalizations of the same "incentives over intentions" mechanism.
- For the collaborator-choice agenda: prestige-bearing proxies (co-author rank, journal name) are subject to the same dynamics — signalling games where selection propagates proxy-hacking behavior without any agent intending it; the "top-tier tail advantage" mechanism predicts rank-homophily strategies spreading in co-authorship networks.
- The claim that replication only slows (not stops) deterioration is directly testable in the model class used in this project's Mesa implementation.

## Caveats

- The power meta-analysis is a convenience sample (Google Scholar snowball from one seed paper), may overestimate power, and lacks field-level completeness.
- Model assumptions (positive novel results always publishable, negatives never; binary truth of hypotheses; equal pay-offs per novel publication) are stylised; the authors state robustness conditions but publish no full sensitivity analysis.
- Effort/power independence is a modelling convenience, acknowledged as unnatural from a pure signal-detection view.
- Cultural-evolution framing asserts heritability of methods only "partly"; the strength of prestige-biased copying is not parameterised.
- Policy discussion is normative rather than model-derived — no welfare criterion is formalised inside the model.
