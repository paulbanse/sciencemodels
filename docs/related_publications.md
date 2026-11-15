---
title: "Proxy Failure in the Epistemic Marketplace: A Literature Review of Agent-Based Models of Science, Metric Gaming, and Collaborator Choice"
version: 2.1
date: 2026-11-14
project: sciencemodels
tags:
  - agent-based-modeling
  - proxy-failure
  - goodharts-law
  - metascience
  - epistemic-landscape
  - collaborator-choice
  - team-science
  - science-of-science
  - network-formation
  - funding-lotteries
  - peer-review
revision-vehicle: strikethrough
revision-renders: "md, pdf, html"
---

**Version 2.1** (November 2026)
**Version 2** (November 2026)
**Version 1** (November 2026)

v2 status note (2026-11-14): this version closes the publication gaps left by
v1 in two directions. First, it gives the project's **own four foundational
texts** a dedicated section --- the two Dobusch--Kapeller paradigm papers, the
Kapeller--Steinerberger publishing simulation, and the Kapeller--Paul--Braganza
typology note that Path D formalises --- because the model's claims rest on
them and v1 cited them nowhere. Second, it adds **eight studies from the
epistemic-network, funding-lottery and peer-review literatures** that the
handovers and proposals rely on but the review did not carry: Strevens on the
priority rule, Avin's funding-lottery analysis and its simulation, the two
Squazzoni peer-review simulations and the Squazzoni--Bravo--Takács incentive
experiment, Bianchi et al.'s resource-constrained peer-review ABM, and the two
network-formation foundations (Bala--Goyal, Jackson--Rogers) that Path B's
dyadic layer stands on. Every added bibliographic record was resolved through
its DOI against Crossref on 2026-11-14, not transcribed from a reading list;
where a handed-over citation named the wrong venue or paper type, the record
below carries the verified one and the correction is stated in place.

# Introduction

This review maps the intellectual terrain around the `sciencemodels` project: an agent-based model (ABM) of scientific exploration designed to illustrate *proxy failure* in science — the systematic degradation of the link between what science rewards and what science is for. In the model, scientists move on a two-dimensional toroidal grid of knowledge tiles (an epistemic landscape), harvest knowledge from the tiles they visit, and accrue two distinct currencies: *prestige* (deep, quality-related harvest) and *visibility* (how conspicuous one's position and output are to others). Each scientist's reward mixes novelty-seeking (curiosity) with the visibility of their position in Cobb-Douglas form, $R_i = (k_i/\overline{k})^{c_i} \cdot V_i^{1-c_i}$ where $k_i/\overline{k}$ is the novelty of the chosen tile relative to the community average and $V_i$ is visibility (the curiosity $c_i$ is an elasticity), and the model's normative yardstick is *ideal merit*: an agent's harvest compared with the allocation an omniscient social planner would have produced. The paper's central angle is the **collaborator-choice problem**: when visibility acts as a proxy for research quality, scientists rationally choose prestigious-but-shallow co-authors, and the resulting selection pressure pushes visibility ($V$) and true quality ($Q$) apart — precisely the dynamic formalized by John et al. (2023) under the label *proxy failure* and long known in the policy literature as Goodhart's law and Campbell's law.

The review opens with the project's own foundations and then runs through three thematic clusters. Section one collects the project's own four foundational texts, which v1 of this review cited nowhere. The first external cluster covers agent-based and epistemic-landscape models of science, from Kitcher's division of cognitive labor to network epistemology and modern reviews of the ABM-of-science genre. The second covers metric gaming, proxy failure, and misaligned incentive structures, including the John et al. target article, the "natural selection of bad science" tradition, and the empirical literatures on questionable research practices --- with the funding-lottery debate added in v2 because it is the one intervention literature our Path C work runs into directly. The third covers collaborator choice, team science, co-authorship, and prestige: why teams dominate knowledge production, how superstar scientists structure knowledge diffusion, and how cumulative advantage shapes who works with whom --- including, in v2, the two network-formation models Path B has to stand on. A closing section identifies what our ABM can contribute that these literatures have not.

Every entry below was verified against publisher or bibliographic database records (title, authors, year, volume, issue, pages, DOI or stable URL): the v1 corpus at its preparation, the v2 additions on 2026-11-14 by resolving each DOI through Crossref and comparing the returned title, container, volume, issue and pages against the cited record. No citation is reproduced from memory alone, and candidates that could not be verified were dropped in favour of verified replacements --- the one item with no published record at all is marked as an unpublished working note instead of being dressed with invented metadata.

# The project's own foundations: four texts this model formalises

The review below surveys the outside literature. These four texts are the
project's own foundations and belong in the same record: the model's question,
its reward architecture and its typology of failure are derived from them, and
v1 of this review cited none of them.

## Dobusch and Kapeller (2009): paradigmatic dominance as path-dependent lock-in

Leonhard Dobusch and Jakob Kapeller, "'Why is Economics not an Evolutionary Science?' New Answers to Veblen's Old Question", *Journal of Economic Issues*, 43(4), 867--898 (2009). DOI: 10.2753/JEI0021-3624430403.

The paper reframes Veblen's question from the content of economic theory to the evolution of the discipline itself, and answers it with path-dependence theory: three positive-feedback mechanisms --- coordination/network effects, institutional complementarity, and learning with irreversible investment --- make the neoclassical position self-reinforcing, while strategic *amplifiers* (winning the prize, closing disciplinary borders, assimilating critics) work on top of them. Citation metrics enter as a *critical juncture* that institutionalised the dominance already in place. The original citation analysis gives the asymmetry numbers: over 1989--2008 the top-ten heterodox journals cite the top-ten orthodox journals roughly eleven times more often than the reverse (4,229 imports against 385 exports), and 64.22% of citations into the orthodox top ten are intra-network against 19.3% in the heterodox top ten.

**Link to our ABM.** (i) The paper's amplifiers are *strategic* pressure on a quality proxy, which is what our agents do unintentionally through the visibility term $(1-c_i)$ --- the model's contribution is to show the same feedback needs no strategy at all. (ii) The citation-transfer asymmetry is the empirical shadow of our prestige/visibility split: prestige is what accumulates through the network, merit is what the landscape holds. (iii) Lock-in is the macro counterpart of our stationary clusters: visibility-weighted movement concentrates agents where prestige already sits, and the aggregate is a frozen hierarchy. (iv) The paper has no model; its mechanisms are the qualitative targets our prestige--merit correlation is meant to measure.

## Dobusch and Kapeller (2012): interested pluralism and the evaluation proxy

Leonhard Dobusch and Jakob Kapeller, "Heterodox United vs. Mainstream City? Sketching a Framework for Interested Pluralism in Economics", *Journal of Economic Issues*, 46(4), 1035--1058 (2012). DOI: 10.2753/JEI0021-3624460410.

The constructive companion to the 2009 paper proposes *interested pluralism*: a pluralist (meta-)paradigm built from ecumenical principles rather than demarcation criteria, strong enough to act as a single competitor to the mainstream. Its analytical core is a taxonomy --- three styles of pluralism (selfish, disinterested, interested) and six types of relation between statements from different traditions, from identical to contradictory --- plus an evaluative reform: content-oriented evaluation instead of citation counting, i.e. explicitly an anti-Goodhart redesign of the quality proxy.

**Link to our ABM.** (i) The evaluative reform defines our intervention space: everything that de-targets the visibility proxy (Path C, and the lottery/damping experiments of `P-0003`) is an instance of it. (ii) The six-type taxonomy is a ready coding scheme for a multi-tradition landscape --- several knowledge peaks, with tradition labels deciding which peak an agent will even consider. (iii) Their testable conjecture, that citation-count regimes measurably reduce cross-tradition engagement, is directly runnable in our setting: does a higher visibility weight reduce visits to peaks far from an agent's starting region?

## Kapeller and Steinerberger (2016): emergent pathologies in publishing

Jakob Kapeller and Stefan Steinerberger, "Emergent phenomena in scientific publishing: A simulation exercise", *Research Policy*, 45(10), 1945--1952 (2016). DOI: 10.1016/j.respol.2016.08.004.

A short communication that simulates the full production chain --- creation, submission, evaluation, publication, journal re-ranking --- under deliberately optimistic assumptions: objective authors, objective referees, efficient rankings, no opportunism. Even so, five phenomena emerge for purely mathematical reasons: the top of the journal hierarchy is stable while the middle is a volatile herd; within-journal quality varies wildly outside the top; rejection probability *increases* with paper quality once noise is even slightly present; and the journal landscape becomes a bottleneck in which a large share of the best papers (about 40% of the top 20% in the reported runs) never appears. A tiny amount of noise, not misconduct, produces the pathology.

**Link to our ABM.** (i) It is the closest methodological sibling: a simulation whose distortions are emergent rather than behavioural, which is exactly our claim about prestige. (ii) Its journal ladder is our prestige hierarchy with visible rungs, and its "volume compensates referee error" argument is a mechanism we do not model --- a candidate extension. (iii) It supports our no-gaming premise: proxy-oriented action $AP_F$ need not be conscious, which is why our baseline can generate a wedge with fully honest agents (Path D's distraction channel). (iv) Its bottleneck finding is a welfare statement of the same family as our ideal-merit loss and belongs beside it in the Path C section.

## Kapeller, Paul, and Braganza (2026): the causal grammar of proxy failure

Jakob Kapeller, Paul (surname not given in the source note), and Oliver Braganza, "Hunting the Causal Patterns Behind Proxy Failure", work-in-progress note, March 2026 (translation in `context/Hunting_Proxy_Patterns.md`; no venue or DOI --- cite as an unpublished note).

The note fixes the vocabulary the project uses: a proxy $P$, a goal $G$, an aligned action $A$, a proxy-oriented action $AP_F$ motivated by $M$; three possible $P$--$G$ structures (direct, confounded through a third factor, or no link at all); and two failure channels --- **distraction** ($AP_F$ crowds out $A$, indirect) and **destruction** ($AP_F \dashv G$, the direct "blue path"). It also names the **counterforce**: where $P \to G$ holds, raising the proxy genuinely helps the goal, which can stabilise honest signalling but need not dominate.

**Link to our ABM.** (i) This is the text Path D formalises, and the mapping is already worked out: `AdmiralsLog/contributions/CONT2-hunting-proxy-patterns-formalization.md` maps each typology element onto model objects, and `P-0005` specifies the $2 \times 2 \times 3$ regime diagram that sweeps the three channels ($\gamma$ for destruction, $\kappa$ for the counterforce). (ii) Our baseline model is the **pure-distraction cell**: $AP_F$ is visibility-weighted movement, $G$ is ideal merit, and no agent harms a tile directly. (iii) The conditional counterforce is the one channel the current code cannot express --- which is why $\kappa$ enters as a one-parameter extension rather than a reinterpretation of an existing knob.

# Agent-based and epistemic-landscape models of science

Computational models of science have a fifty-year history, but the modern ABM tradition — agents searching landscapes, exchanging results, and dividing cognitive labor — crystallizes in the following works. These are the direct methodological ancestors of our toroidal knowledge grid.

## Kitcher (1990): The division of cognitive labor

Philip Kitcher, "The Division of Cognitive Labor", *The Journal of Philosophy*, 87(1), 5–22 (1990). DOI: 10.2307/2026796.

Kitcher asks how a community of rational scientists should distribute itself across competing research lines when individual incentives pull everyone toward the most promising approach. Using an informal mathematical model rather than simulation, he shows that a community of purely selfish, payoff-maximizing scientists will converge too heavily on the currently dominant theory: the private rationality of chasing the best-currently-funded line produces a socially suboptimal division of labor, because the option value of exploring alternative ("bold") lines is a public good no individual fully captures. Kitcher then sketches mechanisms — norm-driven motivation, tolerance for mavericks — by which a community can approximate the optimal distribution. The paper is the founding statement of the "division of cognitive labor" problem and the reason epistemic landscape models exist at all: it poses precisely the gap between individually rational allocation and collectively ideal allocation that our *ideal merit* metric is designed to measure. Relevance to our ABM: ideal merit is a Kitcher-style social-optimum benchmark, and our curiosity term can be read as an institutionalized tolerance for unexplored tiles that selfish prestige-maximization would neglect.

## Weisberg and Muldoon (2009): Epistemic landscapes and the division of cognitive labor

Michael Weisberg and Ryan Muldoon, "Epistemic Landscapes and the Division of Cognitive Labor", *Philosophy of Science*, 76(2), 225–252 (2009). DOI: 10.1086/644786.

Weisberg and Muldoon translate Kitcher's philosophical problem into an explicit agent-based model: scientists occupy positions on an epistemic landscape — a grid of research approaches whose "significance" (the height of the landscape) is initially unknown — and move around it to find the most epistemically significant approaches. They compare three search strategies: a "tortoise" strategy (follow-the-neighbor, exploiting local information), a "maverick" strategy (jump to unexplored regions), and a "follower" strategy. Their headline result is that communities containing mavericks — agents willing to explore neglected regions — often outperform communities of more conventionally rational searchers, though the result is sensitive to landscape topology and information assumptions. The paper is the canonical reference for the grid-based epistemic landscape ABM and for the normative claim that diversity of search heuristics is epistemically valuable. Relevance to our ABM: our toroidal knowledge grid is a direct descendant of their landscape, and our curiosity parameter is a continuous, payoff-weighted generalization of the maverick–follower distinction.

## Zollman (2007): The communication structure of epistemic communities

Kevin J. S. Zollman, "The Communication Structure of Epistemic Communities", *Philosophy of Science*, 74(5), 574–587 (2007). DOI: 10.1086/525605.

Zollman investigates whether the social network through which scientists share results affects a community's ability to track the truth. Using a Bayesian-agent simulation (a bandit-like model in which agents repeatedly choose between theories and share noisy experimental results over different network structures — complete, cycle, and star), he finds that sparsely connected communities often converge on the truth more reliably than highly connected ones: rapid information flow lets early misleading evidence cascade and lock the community into a false consensus, the so-called "Zollman effect". The paper founded *network epistemology*, the study of how network structure interacts with individual learning. Relevance to our ABM: visibility in our model plays exactly the role of Zollman's communication channel — it determines whose harvest others observe — so the visibility–quality decoupling we study is a story about *which signals propagate* through an epistemic network, not merely how many.

## Zollman (2010): The epistemic benefit of transient diversity

Kevin J. S. Zollman, "The Epistemic Benefit of Transient Diversity", *Erkenntnis*, 72(1), 17–35 (2010). DOI: 10.1007/s10670-009-9194-6.

Extending his earlier network results, Zollman examines how communities divide their labor among competing approaches when agents can observe one another's payoffs and reallocate their efforts accordingly. The key finding is that a community benefits even from diversity that is *transient*: the fact that some scientists briefly pursue unpromising-looking approaches before switching gives the community insurance against cascades of misleading early evidence. But this benefit is fragile — when payoffs are observable and switching is cheap and immediate, the diversity collapses too quickly to help; some institutional friction that sustains diversity is epistemically valuable. The paper links the division-of-cognitive-labor literature to the network-epistemology literature formally. Relevance to our ABM: our scientists switch tiles and collaborators in response to observed rewards; Zollman's transient-diversity result predicts exactly when visibility-driven imitation will prematurely collapse exploration of low-visibility regions of the knowledge grid.

## Hong and Page (2004): Groups of diverse problem solvers

Lu Hong and Scott E. Page, "Groups of diverse problem solvers can outperform groups of high-ability problem solvers", *Proceedings of the National Academy of Sciences*, 101(46), 16385–16389 (2004). DOI: 10.1073/pnas.0403723101.

Hong and Page build a formal framework of functionally diverse problem-solving agents, each equipped with a representation of the problem and a repertoire of heuristics ("tools"). Through computational experiments and analytical results they show a counterintuitive theorem: under broad conditions, a group of randomly selected agents with diverse heuristics outperforms a group composed of the individually best-performing problem solvers, because high-ability solvers tend to have redundant toolkits while diverse groups cover more of the problem space. The result is one of the most-cited formal arguments for cognitive diversity in teams. Relevance to our ABM: it predicts a *second* cost of prestige-driven collaborator choice beyond quality dilution — if prestigious co-authors are selected for visibility rather than complementary skill, the effective diversity of search heuristics on the knowledge grid falls, and the team's coverage of unexplored tiles shrinks even when individual prestige rises.

## Thurner and Hanel (2011): Peer review and selection of the average

Stefan Thurner and Rudolf Hanel, "Peer-review in a world with rational scientists: Toward selection of the average", *The European Physical Journal B* (2011). DOI: 10.1140/epjb/e2011-20545-7.

Thurner and Hanel model scientists as rational agents choosing how much research effort to invest when output is filtered by peer review and career selection. In their stylized world, scientists trade off the cost of effort against the probability of passing review thresholds; when reviewers and institutions cannot observe quality directly and instead select on relative position in the output distribution, the equilibrium pulls effort toward the safest publishable increment. Their analysis warns that peer review functioning as a *selection device* tends toward "selection of the average": high-risk, high-payoff strategies are penalized because their output distribution is wider, so the population of surviving scientists becomes progressively more conservative. Relevance to our ABM: this is the peer-review analogue of our visibility channel — when the filter (review, hiring, co-author selection) rewards a proxy, the population evolves to optimize the proxy, and our model adds the social layer (who teams with whom) that theirs leaves implicit.

## Smaldino and Epstein (2015): Social conformity despite individual preferences for distinctiveness

Paul E. Smaldino and Joshua M. Epstein, "Social conformity despite individual preferences for distinctiveness", *Royal Society Open Science*, 2(3), 140437 (2015). DOI: 10.1098/rsos.140437.

Smaldino and Epstein demonstrate, formally and in an agent-based model, that a population of agents each of whom has a *fixed need to be distinctive* can nevertheless self-organize into complete social conformity. Each agent adjusts an observable position toward its ideal, defined in standard deviations from the population mean; because everyone's ideal is relative to the same moving distribution, individual distinctiveness-seeking produces aggregate conformity, with occasional departures such as divergence or the outsized influence of extremist minorities. The paper is a striking demonstration that individually "original" motivations do not guarantee collective diversity — an emergent-conformity result of exactly the kind visible in fashion-like academic incentives. Relevance to our ABM: our reward mixes curiosity with visibility of position; if visibility is socially defined (standing out *relative* to others), Smaldino and Epstein's dynamics predict that curiosity-driven scientists may still cluster on the same visible tiles, and the paper motivates modeling distinctiveness as a relative, not absolute, payoff.

## Payette (2011): Agent-based models of science

Nicolas Payette, "Agent-Based Models of Science", in A. Scharnhorst, K. Börner, and P. van den Besselaar (Eds.), *Models of Science Dynamics: Encounters Between Complexity Theory and Information Sciences*, pp. 127–157. Springer (2011). DOI: 10.1007/978-3-642-23068-4_4.

Payette's chapter surveys the ABM-of-science genre and offers a taxonomy of how agent-based models have been used to study scientific communities: models of the division of cognitive labor, models of norm transmission and reward structures, models of citation and communication, and models of team formation. Crucially, it also articulates the methodological standards the genre aspires to — models should make explicit the micro-assumptions about scientists' motivations and the evaluation criteria by which "good science" is assessed — and catalogs recurring weaknesses, including ad hoc payoff functions and the difficulty of grounding agent rules empirically. Relevance to our ABM: the chapter is our methodological checklist; our explicit separation of prestige and visibility currencies, and the use of ideal merit as an external evaluation criterion, responds directly to Payette's complaint that models of science rarely define what counts as success independently of what agents are rewarded for.

## Šešelja (2022): Agent-based models of scientific interaction

Dunja Šešelja, "Agent-based models of scientific interaction", *Philosophy Compass*, 17(7), e12855 (2022). DOI: 10.1111/phc3.12855.

Šešelja reviews the recent wave of ABMs of scientific interaction — models in which communities of scientists exchange information, form beliefs, and divide labor under varying social structures — organized around three questions: how social network structure affects collective belief, how cognitive diversity and division of labor affect problem-solving, and how norms and incentive structures shape research practice. The review highlights that the literature has matured from proof-of-concept "how-possible" models toward models that engage empirical patterns, and it identifies open problems: connecting model assumptions to empirical measures of scientific behavior, and modeling the co-evolution of incentives and epistemic outcomes. Relevance to our ABM: our model sits precisely at the intersection of the second and third strands — division of labor (tile allocation) under incentive structures (visibility-weighted reward) — and the review's call for empirically anchored incentive dynamics defines our contribution target.

## Strevens (2003): the priority rule as an allocation device

Michael Strevens, "The Role of the Priority Rule in Science", *The Journal of Philosophy*, 100(2), 55--79 (2003). DOI: 10.5840/jphil2003100224.

Strevens asks what the priority rule --- first to discover takes the credit, however close the runners-up were --- is *for*, and answers with a welfare argument: treat any reward scheme for scientific endeavour as a device for allocating resources across research programmes so as to maximise the benefit to society. The priority system, he shows, is especially well suited to finding an efficient allocation in precisely those situations characteristic of science in which success subsequent to the first success brings little additional social benefit --- the winner-take-all benefit structure. Reward for firstness therefore need not be a distortion; in the right benefit environment it is an allocation mechanism, and the paper derives both its virtues and its distinctive pathologies (racing, secrecy) from that structure.

**Relevance to our ABM:** this is the strongest published counterweight to a naive reading of our own result. Our model measures how far prestige strays from ideal merit, but Strevens supplies the conditions under which *prestige-based* allocation is the efficient rule --- and those conditions are parameterisable in our landscape: legible peaks (low `std`, low `prop_random`) and a winner-take-all harvest technology are exactly when our wedge closes. His paper turns our headline from an indictment into a comparative question: prestige allocation tracks merit when firstness carries the social value and fails when it does not, which is the model-internal claim our sensitivity map should test. It also gives the theoretical reason why removing prestige entirely (the $\alpha = 0$ lottery of `P-0003`) is not a free improvement.

## Avin (2015): funding science by lottery on a dynamic epistemic landscape

Shahar Avin, "Funding Science by Lottery", in *Recent Developments in Philosophy of Science: EPSA13 Helsinki* (European Studies in Philosophy of Science), Springer, 111--126 (2015). DOI: 10.1007/978-3-319-23015-3_9.

Avin argues that a formal random element in grant allocation can *raise* expected efficiency. Grant peer review is biased towards the familiar present, while the causal effect of a research project changes over time, so a lottery counteracts that bias by buying exploration of the unfamiliar. The argument is supported by a computer simulation of different funding mechanisms on a hypothetical **dynamic epistemic landscape**, illustrated with the historical route to the structure of DNA.

*Correction to the handed-over citation:* the brief for this round named "Avin (2018 / 2019)" for the simulation. The simulation is this 2015 EPSA13 chapter; the 2018/2019 journal article of the same author (next cluster) is a policy synthesis of the lottery debate and contains no model. Both are listed, each described as what it is.

**Relevance to our ABM:** (i) it is the closest published precedent for an epistemic-landscape model in which a funding mechanism is the treatment --- but its allocation is over *projects*, never over *partners*, which leaves our collaborator channel untouched and keeps our niche claim intact. (ii) It is the literature anchor for the $\alpha = 0$ pure placement lottery in `P-0003`: our prestige-damped entry weights $P(\text{sup}_i) \propto \text{prestige}_i^{\alpha}$ interpolate between Avin's randomisation ($\alpha = 0$) and the baseline prestige roulette ($\alpha = 1$). (iii) Its efficiency argument is a movement in the same direction as our maverick-relocation shock (`P-0003`, relocation probability $p_{\text{reloc}}$): both trade exploitation of the visible for expected exploration, and both should therefore be scored against the same ideal-merit loss.

## Squazzoni and Gandelli (2013): an agent-based model of referee behaviour

Flaminio Squazzoni and Claudio Gandelli, "Opening the Black-Box of Peer Review: An Agent-Based Model of Scientist Behaviour", *Journal of Artificial Societies and Social Simulation*, 16(2), 3 (2013). No DOI registered for this issue; stable URL: https://www.jasss.org/16/2/3.html.

The paper opens the review process into an explicit interaction protocol: peer review is modelled as evaluation under knowledge asymmetry and bias, with authors and referees playing repeated games in several interaction scenarios. Its central result is that reciprocity between the parties does not by itself improve review quality --- it can amplify evaluation bias --- and has a positive effect only when reciprocity is disciplined by disinterested standards of fairness.

**Relevance to our ABM:** (i) the referee is the one actor our model does not have; if Path B makes credit sharing explicit, evaluation becomes the natural next object, and this is the template for coding it without adding a second free-floating parameter. (ii) Their conditional result --- reciprocity helps only under a fairness regime --- is structurally the same conditional we measure: a visibility reward improves collective outcomes only below the collapse knee $\xi^{*} \in [0.6, 0.8]$, and harms above it. (iii) Methodologically, JASSS is the natural venue for the ABM-only paper and its conventions (model documentation, replication tables) are worth obeying from the first submission.

## Bianchi, Grimaldo, Bravo, and Squazzoni (2018): peer review as a cooperation dilemma

Federico Bianchi, Francisco Grimaldo, Giangiacomo Bravo, and Flaminio Squazzoni, "The peer review game: an agent-based model of scientists facing resource constraints and institutional pressures", *Scientometrics*, 116(3), 1401--1420 (2018). DOI: 10.1007/s11192-018-2825-4.

Peer review is modelled as a cooperation dilemma inside a game-theoretic framework. Scientists split effort between publishing and reviewing, react to acceptance and rejection of their own manuscripts and to the fairness of the review they received, may be realistic or over-confident about their own work, and respond to the competitive pressure of their institutional context. Publication bias and publication quality depend mainly on reviewer motivations, and context pressure cuts both ways: excessive competition degrades the minimisation of bias, yet some competition is instrumental to quality --- especially when refereeing is sustained by reciprocity motives.

**Relevance to our ABM:** (i) this is the closest ABM to our effort-allocation problem, but its currency is effort between tasks and its success measure is publication outcome, whereas ours is epistemic merit against a planner's benchmark --- the yardstick it lacks is our contribution. (ii) Its non-monotone competition result (too much and too little competition both hurt) is the kind of interior optimum our Path D counterforce $\kappa$ is designed to produce, and the two predictions should be compared rather than merged. (iii) Its institutional-pressure channel is a candidate Path D *destruction* mechanism --- crowding consumes the capacity to evaluate honestly, not the knowledge itself.

# Metric gaming, proxy failure, and misaligned incentives in science

This cluster supplies the theoretical core of our paper. It runs from the unifying "proxy failure" framework through institutional critiques of metrics to the empirical and modeling literatures on how incentive structures select for bad science.

## John, Caldwell, McCoy, and Braganza (2024): Dead rats, dopamine, performance metrics, and peacock tails

Yohan J. John, Leigh Caldwell, Dakota E. McCoy, and Oliver Braganza, "Dead rats, dopamine, performance metrics, and peacock tails: Proxy failure is an inherent risk in goal-oriented systems", *Behavioral and Brain Sciences*, 47, e67, 1–56 (published online 26 June 2023). DOI: 10.1017/S0140525X23002753.

This target article is the conceptual anchor of our project. John and colleagues argue that Goodhart's law, Campbell's law, the cobra effect, goal displacement, and the McNamara fallacy are instances of one structural phenomenon they call **proxy failure**: whenever incentivization or selection is based on an imperfect proxy of an underlying goal, a pressure arises that tends to make the proxy a worse approximation of the goal. They distinguish inferential variants (regressional and extremal Goodhart, where optimization moves into regions where proxy–goal correlation breaks down) from control variants (causal and adversarial Goodhart, where agents act to decouple proxy from goal), work through neuroscience, economics, and ecology case studies, and stress that proxy failure is usually *partial divergence*, not total collapse — the "proxy treadmill". Their Hanoi rat-bounty vignette (bounties on rat tails leading to rat farming) is the emblem of adversarial Goodhart. Relevance to our ABM: our model is a controlled instantiation of adversarial Goodhart in science — visibility is the proxy, ideal merit is the goal, and collaborator choice is the causal intervention by which agents decouple the two.

## Braganza (2020): Economically rational sample-size choice

Oliver Braganza, "A simple model suggesting economically rational sample-size choice drives irreproducibility", *PLoS ONE*, 15(3), e0229615 (2020). DOI: 10.1371/journal.pone.0229615.

Braganza formalizes a minimal model in which scientists choose sample sizes by balancing the cost of experimentation against the income (career reward) from publication, under positive publication bias. He shows that economically rational agents systematically choose sample sizes that are too small: the individually optimal, cost-biased sample size yields low statistical power and a low positive predictive value, generating irreproducibility *without any fraud or even any strategic gaming* — the incentive gradient alone does the damage. The model also demonstrates why well-measured but naive interventions (demanding larger samples by fiat) can backfire when the underlying reward function is unchanged. Relevance to our ABM: Braganza is the closest formal precedent for our approach — a minimal, incentive-only model of a Goodhart-like distortion in science — and his result that unglamorous effort (statistical power, like deep exploration) is under-provided because its costs are private while its benefits are diffuse maps directly onto our under-investment in high-ideal-merit, low-visibility tiles.

## Smaldino and McElreath (2016): The natural selection of bad science

Paul E. Smaldino and Richard McElreath, "The natural selection of bad science", *Royal Society Open Science*, 3(9), 160384 (2016). DOI: 10.1098/rsos.160384.

Smaldino and McElreath argue that the replication crisis is not primarily a story of individual moral failure but of *cultural evolution*: when institutions select scientists and funding on the basis of publishable output (a proxy for quality), methods that are likely to produce publishable results — low power, high flexibility, p-hacking-friendly designs — are differentially retained regardless of whether individual scientists intend to game anything. They formalize this with an agent-based model of a population of labs with heritable methodological traits, showing that even rare initial rates of poor practice grow under realistic reward structures, and that "bad science" is an evolutionarily stable response to the incentive environment. Relevance to our ABM: their model gives the evolutionary backdrop for ours; where they model selection on publication-driven method quality, we model the *social* selection channel — choice of collaborators — and can ask whether prestige-as-proxy reproduces the same selection for shallow but visible harvests.

## Smaldino, Turner, and Contreras Kallens (2019): Open science and modified funding lotteries

Paul E. Smaldino, Matthew A. Turner, and Pablo Contreras Kallens, "Open science and modified funding lotteries can impede the natural selection of bad science", *Royal Society Open Science*, 6(7), 190194 (2019). DOI: 10.1098/rsos.190194.

This follow-up to Smaldino and McElreath (2016) tests interventions inside the same evolutionary model of science. The authors ask which institutional reforms can slow or reverse the selection for exploitable metrics: they evaluate open-science practices (which reduce the payoff to flexible, result-contingent reporting) and modified funding lotteries (which decouple funding decisions from exploitably noisy metrics). Their simulations show that both classes of intervention can impede the natural selection of bad science — lotteries by removing the selection pressure on the proxy, open science by making the proxy harder to game — though each has limits and unintended consequences (e.g., lotteries can reduce incentives for genuinely high-quality differentiation if designed poorly). Relevance to our ABM: these are candidate *policy levers* we can implement as model experiments — randomizing collaborator acceptance, or suppressing visibility signals — to see whether ideal merit recovers when the visibility proxy is de-targeted.

## Edwards and Roy (2017): Academic research in the 21st century

Marc A. Edwards and Siddhartha Roy, "Academic Research in the 21st Century: Maintaining Scientific Integrity in a Climate of Perverse Incentives and Hypercompetition", *Environmental Engineering Science*, 34(1), 51–61 (2017). DOI: 10.1089/ees.2016.0223.

Edwards and Roy document how the reward structure of academic science has shifted toward hypercompetition and quantitative performance metrics (impact factors, h-indices, citation counts, grant dollars), and argue that this shift creates systematic, quantifiable pressure toward poor scientific practices and even misconduct. Combining institutional analysis with illustrative cases, they show that when career survival depends on metrics, the metrics cease to measure quality — the Goodhart dynamic — and they warn of a "positive feedback loop" in which declining integrity erodes public trust and further tightens metric control. They propose concrete countermeasures: deemphasizing bibliometrics in evaluation, funding stability, and greater transparency. Relevance to our ABM: our visibility currency is the model-world counterpart of the bibliometrics they indict, and their feedback loop is a macro pattern our model should reproduce — rising measured productivity alongside falling ideal merit.

## Osterloh and Frey (2015): Ranking games

Margit Osterloh and Bruno S. Frey, "Ranking Games", *Evaluation Review*, 39(1), 102–129 (2015). DOI: 10.1177/0193841X14524957.

Osterloh and Frey analyze research rankings — bibliometric league tables governing hiring, funding, and prestige — as *games* in which the ranked players respond strategically to the ranking rule. Their analytical treatment catalogs the distortions rankings induce: gaming of the indicator, neglect of unmeasured tasks (teaching, replication, mentoring), homogenization of research portfolios, and erosion of cooperation, since rankings make colleagues rivals. They distinguish the intended function of rankings (information for outsiders) from their actual use (resource allocation), arguing that the mismatch converts informational signals into targets, and evaluate remedies such as decoupling evaluation from rankings and fostering intrinsic motivation. Relevance to our ABM: the ranking-game framing is exactly our collaborator-choice story — visibility is the ranking, co-author selection is the strategic move, and their prediction that rankings erode cooperation gives us a falsifiable pattern to test against our simulation's network of collaborations.

## Muller (2018): The Tyranny of Metrics

Jerry Z. Muller, *The Tyranny of Metrics*. Princeton University Press (2018). ISBN: 978-0-691-17495-2. URL: https://press.princeton.edu/books/hardcover/9780691174952/the-tyranny-of-metrics

Muller's book is the most widely read synthesis of the metric-fixation critique across education, medicine, business, and government. His argument is historical and case-based: the demand for standardized, comparable performance measures grows with distrust of professional judgment; the resulting metric fixation produces goal displacement (hitting the number replaces doing the job), short-termism, and the erosion of intrinsic motivation, and it reliably invites gaming because the proxy is cheaper to move than the goal. He catalogues recurring failure modes — measuring the easily measurable, substituting targets for missions, rewarding those who game the system — and proposes a diagnostic discipline: use metrics as supplements to, never substitutes for, experienced judgment. Relevance to our ABM: Muller supplies the canonical vocabulary (goal displacement, cream-skimming, gaming) that we map onto model mechanisms; in particular, his claim that metric fixation selects for the appearance of performance is the individual-level story behind our agents' drift toward visible-but-shallow tiles.

## Brembs (2019): Reliable novelty — new should not trump true

Björn Brembs, "Reliable novelty: New should not trump true", *PLoS Biology*, 17(2), e3000117 (2019). DOI: 10.1371/journal.pbio.3000117.

Brembs argues that current incentive structures over-reward novelty at the expense of reliability: because journal prestige and citations track novelty and "sexiness" of results, while reliability (robustness, replication, careful measurement) is slow and unglamorous, the reward gradient actively selects against the very property science most needs. He reviews the evidence that journal rank correlates poorly with reliability, and that novelty-seeking incentives contribute to the replication crisis, then sketches a redesign: reward reliability and robustness explicitly (e.g., through registered reports, post-publication review, and decoupling publication from journal brand). Relevance to our ABM: the novelty-versus-reliability trade-off is structurally identical to our curiosity-versus-visibility mix; our ideal-merit metric can quantify how much "reliable harvest" is lost when visibility dominates the reward.

## Brembs, Button, and Munafò (2013): Deep impact — unintended consequences of journal rank

Björn Brembs, Katherine Button, and Marcus Munafò, "Deep impact: unintended consequences of journal rank", *Frontiers in Human Neuroscience*, 7, 291 (2013). DOI: 10.3389/fnhum.2013.00291.

This paper assembles the empirical evidence on journal prestige (impact factor and related rank measures) as a proxy for quality. Reviewing studies of methodological quality, reliability, retraction rates, and citation behavior, the authors find that journal rank correlates weakly with methodological soundness but strongly with the *appearance* of significance and novelty — and that high-ranked journals exhibit disproportionate rates of retraction and irreproducibility. They argue that using journal rank as a quality proxy is not merely noisy but actively harmful, since it incentivizes exactly the practices (overclaiming, flexible analysis) that populate top venues. Relevance to our ABM: journal rank is the macro-level visibility proxy our agents would face in the real world; this paper's evidence that prestige and reliability decouple is the empirical warrant for building a model in which the visibility–quality correlation *endogenously* degrades under selection.

## Fanelli (2012): Negative results are disappearing

Daniele Fanelli, "Negative results are disappearing from most disciplines and countries", *Scientometrics*, 90(3), 891–904 (2012). DOI: 10.1007/s11192-011-0494-7.

Fanelli provides large-scale empirical evidence that positive-outcome bias is worsening. He analyzes more than 4,600 papers published across all disciplines between 1990 and 2007, measuring how often studies that declared they "tested" a hypothesis reported positive support for it. The overall frequency of positive results grew by more than 22% over the period, with the strongest increases in the social and some biomedical sciences, and with significant country-level variation. Ruling out simple methodological artifacts, he concludes that the trend indicates either declining pioneering research or declining objectivity in how results are produced and published — consistent with scientists rationally steering toward confirmatory, publishable outputs. Relevance to our ABM: this is the aggregate fingerprint of proxy failure at the literature level; our model should show the analogous signature — a rising share of "positive" (high-visibility) harvest reports even as ideal merit stagnates.

## Ioannidis (2005): Why most published research findings are false

John P. A. Ioannidis, "Why Most Published Research Findings Are False", *PLoS Medicine*, 2(8), e124 (2005). DOI: 10.1371/journal.pmed.0020124.

Ioannidis's landmark paper derives, from explicit probabilistic assumptions, conditions under which a published research claim is more likely false than true. Using a simple framework built on study power, bias, the number of competing teams working on a question, and the ratio of true to no relationships probed, he proves a set of corollaries: smaller studies, smaller effects, greater flexibility in design and analysis, financial interests, and scientific competition all push the positive predictive value down; and "hot" fields with many teams chasing the same target are especially prone to false discoveries. The paper reframed the replication crisis as a predictable statistical consequence of the incentive environment. Relevance to our ABM: Ioannidis's competition corollary — more teams chasing visible targets lowers the reliability of what emerges — is a claim our model can produce endogenously, since visibility attracts agents to the same tiles and inflates apparent harvest without increasing ideal merit.

## Kerr (1998): HARKing — hypothesizing after the results are known

Norbert L. Kerr, "HARKing: Hypothesizing After the Results are Known", *Personality and Social Psychology Review*, 2(3), 196–217 (1998). DOI: 10.1207/s15327957pspr0203_4.

Kerr names and analyzes a widespread questionable research practice: HARKing — presenting a hypothesis derived from one's results as though it were an a priori prediction. He defines its variants (e.g., dropping a failed original hypothesis and reporting a post hoc one as if predicted), enumerates the epistemic costs (inflated confidence, corrupted error rates, misleading theory accumulation), considers the social-psychological pressures that encourage it (the premium on confirmatory, clean narratives), and discusses remedies through norms, disclosure requirements, and methodological training. The paper is the classic statement that incentive-driven flexibility in *reporting* degrades the evidentiary record even absent fabrication. Relevance to our ABM: HARKing is the individual-level strategy our visibility-seeking agents would adopt if we modeled output reporting; it motivates treating the *reported* harvest (visible) and the *true* harvest (merit-bearing) as distinct quantities in the model.

## Simmons, Nelson, and Simonsohn (2011): False-positive psychology

Joseph P. Simmons, Leif D. Nelson, and Uri Simonsohn, "False-Positive Psychology: Undisclosed Flexibility in Data Collection and Analysis Allows Presenting Anything as Significant", *Psychological Science*, 22(11), 1359–1366 (2011). DOI: 10.1177/0956797611417632.

Simmons, Nelson, and Simonsohn demonstrate experimentally that ordinary, undisclosed research flexibility — optional stopping, adding covariates, dropping conditions, pooling measures — can produce statistically significant support for arbitrarily false claims (their demonstrations include "evidence" that listening to a song makes listeners literally younger). They show analytically that these practices raise the real false-positive rate far above the nominal 5%, and they propose concrete author guidelines (four simple requirements) to reinstate error control. The paper was a catalyst for the open-science reforms of the 2010s. Relevance to our ABM: it quantifies the *return* to gaming strategies in the environments our agents inhabit — where reported significance is the visibility proxy, the marginal payoff to flexibility is huge, and only norm-level constraints (our potential policy interventions) can lower it.

## John, Loewenstein, and Prelec (2012): Measuring the prevalence of questionable research practices

Leslie K. John, George Loewenstein, and Dražen Prelec, "Measuring the Prevalence of Questionable Research Practices With Incentives for Truth Telling", *Psychological Science*, 23(5), 524–532 (2012). DOI: 10.1177/0956797611430953.

John, Loewenstein, and Prelec survey more than 2,000 psychologists about their own engagement in questionable research practices, using an anonymous elicitation format supplemented by monetary incentives for honest reporting. Admissions were strikingly common: a substantial majority admitted to practices such as selectively reporting studies that worked, collecting more data after peeking at results, or excluding outliers post hoc; incentives for truth-telling significantly increased admission rates, indicating that ordinary self-reports underestimate prevalence. The authors also documented hypocrisy — respondents rated others' identical practices more harshly than their own — and found that self-reported engagement correlated with perceived career pressure. Relevance to our ABM: this is the empirical distribution of gaming propensities our agents' behavior should be calibrated against; the career-pressure correlation is direct evidence that the reward slope ($\beta$ on visibility in our model) drives the choice of shallow-but-positive strategies.

## Avin (2019): mavericks and lotteries --- the policy synthesis

Shahar Avin, "Mavericks and lotteries", *Studies in History and Philosophy of Science Part A*, 76, 13--23 (2019; online December 2018). DOI: 10.1016/j.shpsa.2018.11.006.

A review and analytical synthesis of the funding-by-lottery proposal as it emerged independently in several academic corners and as it has actually been implemented: the New Zealand Health Research Council "Explorer Grants" (2013), the Volkswagen Foundation "Experiment!" changes (2017), and the New Zealand Science for Technological Innovation "Seed Projects" (2017) all now use random selection as part of grant allocation. Avin reconstructs the arguments, shows how they support and constrain each other, and identifies where competing arguments demand further analysis or data. There is no model here; the model is the 2015 chapter above.

**Relevance to our ABM:** (i) it supplies the *institutional* justification for the lottery arms of `P-0003` --- our simulation tests a mechanism that already has policy precedent, which strengthens the external-validity framing of Path C. (ii) Its central normative worry, that lotteries reward the merely unconventional rather than the good, is exactly what our ideal-merit yardstick can price: a lottery that raises exploration but lowers realised harvest is visible in our welfare curves in a way it is not in the policy debate. (iii) It documents *why* the proposal keeps recurring --- proxy critique of peer review --- which is the same critique this project formalises.

## Squazzoni, Bravo, and Takács (2013): does paying for review improve it?

Flaminio Squazzoni, Giangiacomo Bravo, and Károly Takács, "Does incentive provision increase the quality of peer review? An experimental study", *Research Policy*, 42(1), 287--294 (2013). DOI: 10.1016/j.respol.2012.04.014.

An incentivised laboratory experiment rather than a simulation: reviewers are paid for the *quality* of their reports (accuracy against the editors' information), and the paper asks whether the incentive actually improves reviewing. The answer is conditional --- the form and the salience of the incentive matter, and payment can shift effort without improving accuracy --- which makes it the cleanest evidence that attaching a reward to a proxy for epistemic quality does not mechanically improve the epistemic goal.

*Correction to the handed-over citation:* the brief placed this study in JASSS as an ABM. The record is a *Research Policy* experiment by the same authors; the JASSS ABM by two of them is Squazzoni and Gandelli (2013), listed in the previous cluster.

**Relevance to our ABM:** (i) it is the experimental counterpart of our central mechanism: a reward aimed at a proxy ($P$) fails to move, or moves only conditionally, the goal ($G$). Our ABM's contribution is to show the same conditional structure arising *without* any agent being paid for gaming. (ii) It constrains Path C's intervention menu: paying or penalising evaluators is not a reliable lever, so the interventions worth simulating are allocation-level ones (lotteries, damping, relocation), which is where `P-0003` concentrates. (iii) It is a useful honesty anchor for the paper: when we claim that visibility rewards distort behaviour, there is experimental evidence that such rewards can fail to do what they are meant to do.

# Collaborator choice, team science, and prestige

Our paper's focal mechanism — choosing co-authors whose visibility substitutes for their depth — sits at the intersection of three literatures: cumulative advantage and the Matthew effect, the empirics of team science, and models of team formation.

## Merton (1968): The Matthew effect in science

Robert K. Merton, "The Matthew Effect in Science", *Science*, 159(3810), 56–63 (1968). DOI: 10.1126/science.159.3810.56.

Merton's presidential address to the AAAS identifies and theorizes the "Matthew effect" (from Matthew 25:29: "unto every one that hath shall be given"): eminent scientists receive disproportionately great credit for modest contributions, while unknown scientists receive disproportionately little credit for substantial ones. Through case studies of simultaneous discoveries (including the credit disputes around oxygen, the calculus, and natural selection) and an analysis of the reward and communication systems of science, Merton shows that the allocation of credit is a self-reinforcing process — recognition attracts resources, attention, and further recognition — and discusses both the functional and dysfunctional consequences of this cumulative asymmetry. Relevance to our ABM: the Matthew effect is the engine that makes visibility worth cultivating; in our model it appears as preferential attention to already-visible positions, and it converts early noise in visibility into durable prestige gaps even when harvests are equal.

## Laband and Tollison (2000): Intellectual collaboration

David N. Laband and Robert D. Tollison, "Intellectual Collaboration", *Journal of Political Economy*, 108(3), 632–661 (2000). DOI: 10.1086/262132.

Laband and Tollison treat co-authorship as an economic exchange and ask what it reveals about the division of intellectual labor. Analyzing data on journal submissions and co-authorship patterns, they distinguish "true" intellectual collaboration (joint production where co-authors materially share the work) from "ghost" collaboration driven by reciprocal credit-giving — arrangements where individuals co-author each other's papers to multiply publication counts. Their evidence shows that co-authorship is partly governed by exchange motives: authors trade co-authorship credits across papers, and the practice persists because it is individually rational within a reward system that counts publications. Relevance to our ABM: this is the earliest empirical articulation of our central mechanism — co-authorship chosen for its *returns* (credit, visibility) rather than its epistemic complementarity — and it motivates modeling collaborator choice as a market in which visibility is the traded good.

## Newman (2004): Coauthorship networks and patterns of scientific collaboration

Mark E. J. Newman, "Coauthorship networks and patterns of scientific collaboration", *Proceedings of the National Academy of Sciences*, 101(suppl. 1), 5200–5205 (2004). DOI: 10.1073/pnas.0307545100.

Newman constructs coauthorship networks from three bibliographic databases (biomedical, physics, and computer science) and characterizes their structural signatures: the number of papers and co-authors per scientist, clustering and degree distributions, typical scientist-to-scientist distances, and the variation of collaboration patterns across fields and over time. The networks are highly clustered and show "small-world" properties, with heavy-tailed productivity distributions and strong homophily by geography and field. The paper established the empirical baseline for treating collaboration as a network to be measured, not merely counted. Relevance to our ABM: our model's emergent collaboration network can be validated against Newman's statistics — clustering, path lengths, degree distribution — so that the network cost of prestige-driven collaborator choice (e.g., bridge erosion, echo chambers of visibility) can be stated in empirically grounded terms.

## Guimerà, Uzzi, Spiro, and Amaral (2005): Team assembly mechanisms

Roger Guimerà, Brian Uzzi, Jarrett Spiro, and Luís A. Nunes Amaral, "Team assembly mechanisms determine collaboration network structure and team performance", *Science*, 308(5722), 697–702 (2005). DOI: 10.1126/science.1106340.

Guimerà and colleagues study how the mechanisms by which teams assemble determine both the structure of collaboration networks and the performance of the teams within them. Analyzing longitudinal publication data and building a model of team self-assembly, they show that a combination of newcomers and experienced team players — rather than either alone — is characteristic of high-performing teams and of the community structure of collaboration networks. Teams assembled through mechanisms that mix fresh participants with repeat collaborators generate networks with the "small-world" community structure observed empirically, and such mixtures predict team success. Relevance to our ABM: it supplies a benchmark for *healthy* team assembly; if visibility-driven selection systematically replaces complementary newcomers with prestigious repeaters, our model should show both a structural signature (network fragmentation around prestige hubs) and a performance loss relative to the newcomer–veteran mixture Guimerà et al. identify as optimal.

## Wuchty, Jones, and Uzzi (2007): The increasing dominance of teams

Stefan Wuchty, Benjamin F. Jones, and Brian Uzzi, "The Increasing Dominance of Teams in Production of Knowledge", *Science*, 316(5827), 1036–1039 (2007). DOI: 10.1126/science.1136099.

Using 19.9 million papers over five decades and 2.1 million patents, Wuchty, Jones, and Uzzi document a fundamental shift in knowledge production: teams now dominate solo authors across virtually every field — sciences, engineering, social sciences, arts and humanities, and patents — and team-authored work has a growing advantage in citations, with team products increasingly occupying the highest-impact tail where solo work once reigned. The trend is universal and accelerating, suggesting that the nature of the problems and tools of modern science structurally favors collaboration. Relevance to our ABM: since collaborator choice is our focal decision, the empirical reality that nearly all consequential science is team science raises the stakes of the choice problem — a proxy failure in partner selection now affects the dominant mode of knowledge production, not a niche.

## Jones, Wuchty, and Uzzi (2008): Multi-university research teams

Benjamin F. Jones, Stefan Wuchty, and Brian Uzzi, "Multi-University Research Teams: Shifting Impact, Geography, and Stratification in Science", *Science*, 322(5905), 1259–1262 (2008). DOI: 10.1126/science.1158357.

This companion study shows that teamwork increasingly spans university boundaries: multi-university collaborations are the fastest-growing authorship structure, produce the highest-impact work, and have grown despite the friction of distance (implying large underlying returns to collaboration). Crucially for our purposes, the pattern is stratified — elite universities play a dominant role in these networks and capture a disproportionate share of the credit and impact, reinforcing geographic and institutional inequality in science. Relevance to our ABM: this is direct evidence that collaborator choice is structured by *institutional prestige*, not only by epistemic fit; our agents' preference for prestigious partners has a real-world analogue in the elite-university coupling the paper documents, and the stratification result is a macro pattern we can treat as a validation target.

## Azoulay, Graff Zivin, and Wang (2010): Superstar extinction

Pierre Azoulay, Joshua S. Graff Zivin, and Jialan Wang, "Superstar Extinction", *The Quarterly Journal of Economics*, 125(2), 549–589 (2010). DOI: 10.1162/qjec.2010.125.2.549.

Azoulay, Graff Zivin, and Wang estimate the causal knowledge spillovers generated by academic superstars by exploiting the premature deaths of 112 eminent life scientists. Using difference-in-differences designs on the publication trajectories of collaborators, they find that the death of a superstar causes a persistent, substantial decline (roughly 5–8% in output) in the productivity of his or her collaborators — far more than the loss of a single co-author's labor could explain. The effect is mediated by the superstar's role in shaping the research direction, visibility, and network access of those around them: superstars are genuine hubs of knowledge diffusion, not merely credited figureheads. Relevance to our ABM: this validates one half of the prestige-as-proxy story — prestigious collaborators really do confer visibility and momentum — which is exactly why the proxy is tempting to use, and why its failure is so costly: agents rationally chase partners who are real conduits of attention, while the proxy erodes as a measure of depth.

## Petersen et al. (2014): Reputation and impact in academic careers

Alexander M. Petersen, Santo Fortunato, Raj K. Pan, Kimmo Kaski, Orion Penner, Armando Rungi, Massimo Riccaboni, H. Eugene Stanley, and Fabio Pammolli, "Reputation and impact in academic careers", *Proceedings of the National Academy of Sciences*, 111, 15316–15321 (2014). DOI: 10.1073/pnas.1323111111.

Petersen and colleagues analyze large-scale career-longitudinal publication and citation data to disentangle the roles of reputation, cumulative advantage, and scientific impact. They find that the Matthew effect is quantifiable and strong: early-career recognition (e.g., institutional prestige and early citations) raises the probability of future high-impact work being attributed and attended to, and the system shows "double cumulative advantage" — success begets resources begets success — while the intrinsic impact of a given paper is only weakly predictable from prior reputation. The authors model the feedback between reputation and attention explicitly and discuss how career evaluation could damp the runaway accumulation of credit. Relevance to our ABM: it offers an empirical mapping between our prestige and visibility variables and measurable career outcomes, and its finding that reputation exceeds impact in predicting attention is the precise statistical form of proxy failure we model in collaborator choice.

## Milojević (2014): Principles of scientific research team formation and evolution

Staša Milojević, "Principles of scientific research team formation and evolution", *Proceedings of the National Academy of Sciences*, 111(11), 3984–3989 (2014). DOI: 10.1073/pnas.1309723111.

Milojević develops the first comprehensive model of team size dynamics in science, validated on 154,221 astronomy articles spanning 50 years. She shows that team size distributions have changed shape, not merely scale: small "core" teams form by a Poisson process, while large "extended" teams arise through cumulative advantage — teams grow by accumulating new members proportional to members' past productivity. The model reproduces the empirical evolution of team size distributions (including the power-law tail of very large teams) and offers an analytical decomposition of the two modes of knowledge production. Relevance to our ABM: her cumulative-advantage growth rule gives us an empirically grounded micro-rule for team growth; if visibility (rather than productivity) drives member accumulation in our model, we can quantify how much of the observed growth in team size is prestige-driven rather than production-driven.

## Wu, Wang, and Evans (2019): Large teams develop and small teams disrupt

Lingfei Wu, Dashun Wang, and James A. Evans, "Large teams develop and small teams disrupt science and technology", *Nature*, 566, 378–382 (2019). DOI: 10.1038/s41586-019-0941-9.

Analyzing more than 50 million papers, patents, and software products from 1954 to 2014, Wu, Wang, and Evans show that small and large teams play fundamentally different epistemic roles: small teams disrupt science and technology with novel questions and risky, recombining ideas, while large teams develop and refine established problems with incremental, reliable work. The disruptiveness gradient is systematic across fields and eras and holds after controlling for many confounds. Relevance to our ABM: this is the strongest empirical warrant for treating team composition as an epistemic variable — if visibility-driven collaborator choice systematically inflates team size and homogeneity, our model predicts a shift from disruptive toward developmental work, and ideal merit (which values coverage of unexplored tiles) should fall even as visible output rises.

## Smith, Vacca, Krenz, and McCarty (2021): Great minds think alike, or do they often differ?

Thomas Bryan Smith, Raffaele Vacca, Till Krenz, and Christopher McCarty, "Great minds think alike, or do they often differ? Research topic overlap and the formation of scientific teams", *Journal of Informetrics*, 15(1), 101104 (2021). DOI: 10.1016/j.joi.2020.101104.

Smith and colleagues test how the similarity of researchers' topic profiles predicts future co-authorship. Using topic modelling of six years of publications at a large U.S. research university (approximately 14,300 articles, 12,500 collaborations, 3,400 authors) combined with exponential random graph models, they find a robust inverted-U relationship between topic overlap and collaboration: probability of collaboration rises with similarity up to a "sweet spot" at low-to-moderate overlap, then falls sharply as topics become redundant. The pattern replicates across disciplines, genders, and intra- versus interdisciplinary collaboration, supporting the view that scientists seek collaborators to augment their scientific and technical human capital rather than to mirror themselves. Relevance to our ABM: it identifies the empirically realistic baseline for complementarity-driven partner choice — the foil against which prestige-driven choice is a distortion — and gives us a target shape (inverted-U in epistemic distance) to reproduce or deviate from as visibility weight $\beta$ increases.

## Bala and Goyal (2000): a noncooperative model of network formation

Venkatesh Bala and Sanjeev Goyal, "A Noncooperative Model of Network Formation", *Econometrica*, 68(5), 1181--1229 (2000). DOI: 10.1111/1468-0262.00155.

The canonical model in which a network is the equilibrium of individual linking decisions: each agent chooses with whom to form a link, pays for it, and receives a payoff that flows through the network with decay, so that information (or benefit) reaching an agent from a distant node is worth less than from a neighbour. The paper characterises the stable and efficient architectures and shows how the decay and cost parameters select between them --- including the tension between what agents find it worth building and the architecture that maximises collective payoff.

**Relevance to our ABM:** this is the formal backbone of Path B's dyadic layer. Our link utility $U_{ij} = (1-\lambda)\tilde V_j + \lambda\sqrt{\tilde C_{ij}\tilde M_j}$ is a two-component variant of their payoff: proxy value plus goal-relevant value, with $\lambda$ the mixing parameter. What Bala--Goyal supply and we currently lack is (i) an explicit *cost* of maintaining a tie, without which our network can only grow, and (ii) the decay logic --- a reason why distant partners are worth less --- which maps onto our grid distance or topical overlap $\tilde C_{ij}$. If the owner rules for a decay term in Q2, this is the parameterisation to borrow, and its stable-vs-efficient architecture distinction is exactly the Gate 2 question (does the visibility-driven network differ from the merit-driven one?).

## Jackson and Rogers (2007): how random are social networks?

Matthew O. Jackson and Brian W. Rogers, "Meeting Strangers and Friends of Friends: How Random Are Social Networks?", *American Economic Review*, 97(3), 890--915 (2007). DOI: 10.1257/aer.97.3.890.

A dynamic model of network formation in which nodes meet new partners by two channels at once: uniformly at random ("meeting strangers") and through their existing neighbours ("friends of friends"). The mixture of the two, plus the number of links formed per period, endogenously generates degree distributions that run from the uniform-random graph through the small-world range to a scale-free tail --- and the paper shows that this hybrid reproduces observed social-network degree distributions far better than either pure process.

**Relevance to our ABM:** (i) it tells us what a network-formation rule must contain before our team moments mean anything: a random mixing channel as well as a preferential one, because preferential attachment alone will manufacture the concentration the Klann panel actually shows. (ii) The random channel is the natural place to implement Q5's proposal regimes (universal vs newcomer-driven) without adding a second free parameter --- heterogeneity in who meets whom comes from the process, not from an extra knob. (iii) It sharpens Path B's parsimony risk: a two-channel meeting process already generates the observable structure, so any additional visibility preference must earn its place by moving the held-out moments (O/E homophily 3.1/3.0/2.3, Markov closure 0.70/0.05).

# Master table of studies

| Cluster | Study | Year | Venue | DOI or URL |
|---|---|---|---|---|
| A | Kitcher, "The Division of Cognitive Labor" | 1990 | The Journal of Philosophy, 87(1), 5–22 | 10.2307/2026796 |
| A | Weisberg and Muldoon, "Epistemic Landscapes and the Division of Cognitive Labor" | 2009 | Philosophy of Science, 76(2), 225–252 | 10.1086/644786 |
| A | Zollman, "The Communication Structure of Epistemic Communities" | 2007 | Philosophy of Science, 74(5), 574–587 | 10.1086/525605 |
| A | Zollman, "The Epistemic Benefit of Transient Diversity" | 2010 | Erkenntnis, 72(1), 17–35 | 10.1007/s10670-009-9194-6 |
| A | Hong and Page, "Groups of diverse problem solvers can outperform groups of high-ability problem solvers" | 2004 | PNAS, 101(46), 16385–16389 | 10.1073/pnas.0403723101 |
| A | Thurner and Hanel, "Peer-review in a world with rational scientists" | 2011 | The European Physical Journal B | 10.1140/epjb/e2011-20545-7 |
| A | Smaldino and Epstein, "Social conformity despite individual preferences for distinctiveness" | 2015 | Royal Society Open Science, 2(3), 140437 | 10.1098/rsos.140437 |
| A | Payette, "Agent-Based Models of Science" | 2011 | In Models of Science Dynamics, Springer, 127–157 | 10.1007/978-3-642-23068-4_4 |
| A | Šešelja, "Agent-based models of scientific interaction" | 2022 | Philosophy Compass, 17(7), e12855 | 10.1111/phc3.12855 |
| A | Strevens, "The Role of the Priority Rule in Science" | 2003 | The Journal of Philosophy, 100(2), 55–79 | 10.5840/jphil2003100224 |
| A | Avin, "Funding Science by Lottery" | 2015 | In Recent Developments in Philosophy of Science: EPSA13 Helsinki, Springer, 111–126 | 10.1007/978-3-319-23015-3_9 |
| A | Squazzoni and Gandelli, "Opening the Black-Box of Peer Review" | 2013 | Journal of Artificial Societies and Social Simulation, 16(2), 3 | jasss.org/16/2/3.html |
| A | Bianchi, Grimaldo, Bravo, and Squazzoni, "The peer review game" | 2018 | Scientometrics, 116(3), 1401–1420 | 10.1007/s11192-018-2825-4 |
| B | John, Caldwell, McCoy, and Braganza, "Dead rats, dopamine, performance metrics, and peacock tails" | 2024 | Behavioral and Brain Sciences, 47, e67 | 10.1017/S0140525X23002753 |
| B | Braganza, "A simple model suggesting economically rational sample-size choice drives irreproducibility" | 2020 | PLoS ONE, 15(3), e0229615 | 10.1371/journal.pone.0229615 |
| B | Smaldino and McElreath, "The natural selection of bad science" | 2016 | Royal Society Open Science, 3(9), 160384 | 10.1098/rsos.160384 |
| B | Smaldino, Turner, and Contreras Kallens, "Open science and modified funding lotteries can impede the natural selection of bad science" | 2019 | Royal Society Open Science, 6(7), 190194 | 10.1098/rsos.190194 |
| B | Edwards and Roy, "Academic Research in the 21st Century" | 2017 | Environmental Engineering Science, 34(1), 51–61 | 10.1089/ees.2016.0223 |
| B | Osterloh and Frey, "Ranking Games" | 2015 | Evaluation Review, 39(1), 102–129 | 10.1177/0193841X14524957 |
| B | Muller, The Tyranny of Metrics | 2018 | Princeton University Press | 978-0-691-17495-2 |
| B | Brembs, "Reliable novelty: New should not trump true" | 2019 | PLoS Biology, 17(2), e3000117 | 10.1371/journal.pbio.3000117 |
| B | Brembs, Button, and Munafò, "Deep impact: unintended consequences of journal rank" | 2013 | Frontiers in Human Neuroscience, 7, 291 | 10.3389/fnhum.2013.00291 |
| B | Fanelli, "Negative results are disappearing from most disciplines and countries" | 2012 | Scientometrics, 90(3), 891–904 | 10.1007/s11192-011-0494-7 |
| B | Ioannidis, "Why Most Published Research Findings Are False" | 2005 | PLoS Medicine, 2(8), e124 | 10.1371/journal.pmed.0020124 |
| B | Kerr, "HARKing: Hypothesizing After the Results are Known" | 1998 | Personality and Social Psychology Review, 2(3), 196–217 | 10.1207/s15327957pspr0203_4 |
| B | Simmons, Nelson, and Simonsohn, "False-Positive Psychology" | 2011 | Psychological Science, 22(11), 1359–1366 | 10.1177/0956797611417632 |
| B | John, Loewenstein, and Prelec, "Measuring the Prevalence of Questionable Research Practices" | 2012 | Psychological Science, 23(5), 524–532 | 10.1177/0956797611430953 |
| B | Avin, "Mavericks and lotteries" | 2019 | Studies in History and Philosophy of Science Part A, 76, 13–23 | 10.1016/j.shpsa.2018.11.006 |
| B | Squazzoni, Bravo, and Takács, "Does incentive provision increase the quality of peer review?" | 2013 | Research Policy, 42(1), 287–294 | 10.1016/j.respol.2012.04.014 |
| C | Merton, "The Matthew Effect in Science" | 1968 | Science, 159(3810), 56–63 | 10.1126/science.159.3810.56 |
| C | Laband and Tollison, "Intellectual Collaboration" | 2000 | Journal of Political Economy, 108(3), 632–661 | 10.1086/262132 |
| C | Newman, "Coauthorship networks and patterns of scientific collaboration" | 2004 | PNAS, 101(suppl. 1), 5200–5205 | 10.1073/pnas.0307545100 |
| C | Guimerà, Uzzi, Spiro, and Amaral, "Team assembly mechanisms determine collaboration network structure and team performance" | 2005 | Science, 308(5722), 697–702 | 10.1126/science.1106340 |
| C | Wuchty, Jones, and Uzzi, "The Increasing Dominance of Teams in Production of Knowledge" | 2007 | Science, 316(5827), 1036–1039 | 10.1126/science.1136099 |
| C | Jones, Wuchty, and Uzzi, "Multi-University Research Teams" | 2008 | Science, 322(5905), 1259–1262 | 10.1126/science.1158357 |
| C | Azoulay, Graff Zivin, and Wang, "Superstar Extinction" | 2010 | The Quarterly Journal of Economics, 125(2), 549–589 | 10.1162/qjec.2010.125.2.549 |
| C | Petersen et al., "Reputation and impact in academic careers" | 2014 | PNAS, 111, 15316–15321 | 10.1073/pnas.1323111111 |
| C | Milojević, "Principles of scientific research team formation and evolution" | 2014 | PNAS, 111(11), 3984–3989 | 10.1073/pnas.1309723111 |
| C | Wu, Wang, and Evans, "Large teams develop and small teams disrupt science and technology" | 2019 | Nature, 566, 378–382 | 10.1038/s41586-019-0941-9 |
| C | Smith, Vacca, Krenz, and McCarty, "Great minds think alike, or do they often differ?" | 2021 | Journal of Informetrics, 15(1), 101104 | 10.1016/j.joi.2020.101104 |
| C | Bala and Goyal, "A Noncooperative Model of Network Formation" | 2000 | Econometrica, 68(5), 1181–1229 | 10.1111/1468-0262.00155 |
| C | Jackson and Rogers, "Meeting Strangers and Friends of Friends" | 2007 | American Economic Review, 97(3), 890–915 | 10.1257/aer.97.3.890 |
| D | Dobusch and Kapeller, "'Why is Economics not an Evolutionary Science?'" | 2009 | Journal of Economic Issues, 43(4), 867–898 | 10.2753/JEI0021-3624430403 |
| D | Dobusch and Kapeller, "Heterodox United vs. Mainstream City?" | 2012 | Journal of Economic Issues, 46(4), 1035–1058 | 10.2753/JEI0021-3624460410 |
| D | Kapeller and Steinerberger, "Emergent phenomena in scientific publishing" | 2016 | Research Policy, 45(10), 1945–1952 | 10.1016/j.respol.2016.08.004 |
| D | Kapeller, Paul, and Braganza, "Hunting the Causal Patterns Behind Proxy Failure" | 2026 | Unpublished working note (translation in `context/`) | no DOI |

Clusters: A = agent-based and epistemic-landscape models of science; B = metric gaming, proxy failure, and misaligned incentives; C = collaborator choice, team science, and prestige; D = the project's own foundations. **46 rows**: the 34 studies of v1, the 8 studies added in v2 (four in A, two in B, two in C), and the four foundations rows (D). Every entry was verified against publisher or bibliographic database records --- v1 during its preparation, the v2 additions through their DOIs against Crossref on 2026-11-14 --- and the one item with no published record (the Kapeller--Paul--Braganza note) is marked as an unpublished working note rather than given invented metadata.

# Mapping of the added studies to model objects

The brief for v2 asked for the new entries to be mapped onto the model's
parameters. These are the objects that actually exist; where the brief named a
parameter the code does not have, the nearest real object is named and the
mismatch stated.

| Study | Model object, experiment or premise | Path / proposal | Status in the tree |
|---|---|---|---|
| Strevens (2003) | Visibility weight $1-c$ read as the price of firstness; legibility knobs `std`, `prop_random`; `ideal_merit` as the social-benefit benchmark | Path A / C | knobs exist; the comparative test (prestige allocation efficient under legible, winner-take-all landscapes) is a Path A sweep not yet run |
| Avin (2015) | `P-0003` entry lottery, $P(\text{sup}_i) \propto \text{prestige}_i^{\alpha}$: $\alpha = 0$ is Avin's randomisation, $\alpha = 1$ the baseline roulette (`model.py:396-407`) | Path C, `P-0003` | proposed; driver-side wrapper, baseline bit-identity at $\alpha = 1$ |
| Avin (2019) | Same lottery arms, scored against the collapse knee $\xi^{*} \in [0.6, 0.8]$ of the welfare probe | Path C, `P-0003` | proposed; the welfare diagnostic it would extend is complete |
| Squazzoni and Gandelli (2013) | A referee actor --- absent from the model; the natural home is a Path B extension once credit sharing is explicit | Path B | absent |
| Bianchi, Grimaldo, Bravo and Squazzoni (2018) | Destruction channel $\gamma$: crowding noise $\epsilon_{\text{eff}} = \epsilon \cdot m^{\gamma}$ with $m$ agents on a tile; their non-monotone competition result is the shape $\kappa$ should produce | Path D, `P-0005` | proposed ($\gamma, \kappa \in \{0, 1\}$) |
| Squazzoni, Bravo and Takács (2013) | The reward-for-proxy test in its simplest form: what the visibility weight does to $\mathrm{corr}(\text{prestige}, \text{ideal merit})$ | Path A / C | measured --- the existing wedge and its sensitivity map |
| Bala and Goyal (2000) | Path B link utility with a tie cost and a decay of partner value with distance; $\lambda$ mixes proxy and goal components | Path B, `output/reports/path_b_probe_v1.md` (menu archived at `AdmiralsLog/archive/path_b_dyadic_sketch_v1.md`) | $\lambda$ specified in the archived menu and implemented in the Gate 2 sweep; cost and decay not implemented |
| Jackson and Rogers (2007) | Two-channel meeting process (random and friend-of-friend), making team concentration a property of the process rather than a free knob; hosts Q5's proposal regimes | Path B | proposed |
| Dobusch and Kapeller (2009) | The prestige / visibility split itself; stationary-cluster lock-in as the macro counterpart of the wedge | Path A | measured (oscillating clusters, 16/18 conjunctive cells) |
| Dobusch and Kapeller (2012) | The intervention space --- de-targeting the evaluation proxy --- and multi-tradition landscapes as several labelled peaks | Path C; Path D | partially (`P-0003`) |
| Kapeller and Steinerberger (2016) | Premise: the pathology is emergent, not behavioural (no agent games anything); the prestige hierarchy as a visible ladder | premise of the whole model | premise |
| Kapeller, Paul and Braganza (2026) | Distraction ($AP_F$ crowds out $A$) is the baseline model; destruction ($\gamma$) and counterforce ($\kappa$) are the two extensions | Path D, `P-0005`, `CONT2` | baseline measured; extensions proposed |

Notes on the parameter names in the brief:

- `visibility_weight` is a **derived column**, not a constructor argument: it
  is $1 - $ `initial_curiosity`, written by the welfare sweep
  (`scripts/welfare/run_welfare_sweep.py:145`) and used in the collapse-knee
  analysis. In the model proper the object is the curiosity elasticity $c$ in
  the Cobb-Douglas reward (`sciencemodels/agents.py:135-137`).
- `entry_prestige_stock` **is** a constructor parameter
  (`sciencemodels/model.py:41,143`) with the alternative stocks
  `prestige`, `prestige_vanishing`, `prestige_visibility`; the audit of which
  stock drives the supervisor channel is `output/reports/prestige_stock_v1.md`.
- $\lambda$ is the Path B mixing parameter
  (`output/reports/path_b_probe_v1.md`, which implements the menu archived at
  `AdmiralsLog/archive/path_b_dyadic_sketch_v1.md`), $\gamma$ and $\kappa$ are the Path D
  extensions specified in `P-0005`, and the lottery exponent $\alpha$ is
  `P-0003`.
- **`tau` does not exist** in the code, the sketch or any proposal. The two
  objects that could carry that role are the **time-scale mapping** left open
  as Q10 in `AdmiralsLog/HANDOVER-2026-11-14.md` (how many steps one paper or
  one collaboration lasts) and **Bala--Goyal's decay** of partner value with
  distance, which Path B would introduce. Neither is implemented; the brief's
  `tau` is left unnamed rather than mapped onto something that is not it.

# Gaps and opportunities

The three literatures above are rich but largely disconnected, and the seams between them are exactly where our ABM operates. The most consequential gaps are as follows.

- **Collaborator choice is rarely modeled as proxy-driven selection.** The ABM-of-science literature (Kitcher; Weisberg and Muldoon; Zollman) models search and communication; the incentive literature (Smaldino and McElreath; Braganza; John et al.) models selection on output proxies; the team-science literature (Wuchty; Guimerà; Milojević; Smith et al.) models who teams with whom. Almost none of it models the *decision to partner* as a Goodhart-like optimization of a visibility proxy in place of a quality goal. Our model's explicit two-currency architecture (prestige versus visibility) and its ideal-merit benchmark make this mechanism measurable for the first time.

- **Formal proxy-failure work lacks a social network channel.** John et al. (2023) derive proxy failure abstractly and illustrate it with institutional examples; Braganza (2020) formalizes one behavioral margin (sample size). Neither models the co-authorship network as the causal pathway by which a proxy degrades. In our ABM, the proxy failure is *relational*: agents select partners by visibility, the selected partners amplify each other's visibility, and the visibility–quality correlation degrades endogenously through the network — a testable instantiation of adversarial Goodhart.

- **The Matthew effect is documented but seldom micro-founded against an epistemic ground truth.** Merton (1968) and Petersen et al. (2014) show that attention begets attention, but real bibliometric data cannot cleanly separate genuine quality from accumulated credit. Our ideal-merit metric supplies the missing ground truth: we can measure exactly how much of an agent's prestige is attributable to harvest and how much to self-reinforcing visibility, quantifying the Matthew effect as a ratio rather than an inference.

- **Normative benchmarks are missing from incentive models.** Payette (2011) notes that models of science rarely define success independently of agent rewards. Our toroidal landscape with an omniscient planner's allocation gives us an external standard, letting us state results as welfare losses: how much ideal merit is destroyed per unit of visibility weight in the reward $R_i = (k_i/\overline{k})^{c} \cdot V_i^{1-c}$, and at what curiosity level $c$ the community's exploration of the knowledge grid collapses.

- **Interventions can be tested cheaply.** Smaldino, Turner, and Contreras Kallens (2019) evaluate lotteries and open science in a population model; our framework can test their analogues in the social domain --- randomizing collaborator acceptance, capping co-author visibility effects, or rewarding epistemic distance in partner choice --- and check whether the inverted-U team-complementarity pattern of Smith et al. (2021) and the newcomer--veteran mixture of Guimerà et al. (2005) re-emerge as visibility pressure is relaxed.

- **The lottery literature has a landscape model, but no partners.** Avin (2015) already simulates funding mechanisms on a dynamic epistemic landscape, and Avin (2019) documents three funding agencies that run lotteries in practice. So the institutional half of our Path C is anticipated in the literature; what is not anticipated is allocation over *partners* rather than over projects. This is the strongest support for `P-0003`'s lottery arms and, at the same time, the sharpest statement of what our contribution is not.

- **Prestige-based allocation has a serious defence, not only a critique.** Strevens (2003) argues that rewarding firstness is an allocation device that is efficient when later success adds little social value. A review that cites only the Goodhart strand would be arguing against a straw position: our ideal-merit yardstick should be used to test Strevens' conditions (legible landscapes, winner-take-all harvest) rather than to presuppose that prestige allocation is distortion. If the wedge closes exactly where his conditions hold, our result and his theorem support each other.

- **Network formation has a canonical toolkit we have not yet used.** Bala and Goyal (2000) supply the cost and decay structure our Path B link utility lacks, and Jackson and Rogers (2007) show that a two-channel meeting process --- random plus friend-of-friend --- reproduces observed degree distributions without any prestige preference at all. Path B must therefore demonstrate that visibility-driven choice moves the held-out network moments *beyond* what a prestige-free meeting process already generates; otherwise the flagship's network claim is an artefact of preferential attachment we introduced ourselves.

- **Team-size and disruption dynamics are not linked to incentive proxies.** Wu, Wang, and Evans (2019) show that small teams disrupt and large teams develop; Milojević (2014) shows cumulative advantage drives team growth. Whether prestige-driven partner choice inflates team size and thereby shifts a field from disruptive to developmental is an open empirical question our model can address directly as a mechanism study.

In short, the literature supplies the pieces --- landscapes, incentive selection, network formation, and a formal theory of proxy failure --- but no existing model assembles them into a single system in which scientists choose collaborators under a visibility proxy and are scored against an ideal allocation. Avin's lottery model comes closest on the institutional side and Bianchi et al.'s peer-review game closest on the allocation side, but neither makes partner choice the proxy channel and neither scores outcomes against a ground-truth benchmark. That assembly remains the contribution of the `sciencemodels` project, and the studies above define both its empirical anchors and its validation targets.

# Revision Log

- **Version 1** (November 2026) --- Initial literature review, 34 verified studies in three clusters, every record checked against publisher or bibliographic-database data; master table; gap analysis and the niche claim.
- **Version 2** (November 2026) --- Added the project's own four foundations as a dedicated section with "Link to our ABM" bullets (Dobusch--Kapeller 2009 and 2012, Kapeller--Steinerberger 2016, Kapeller--Paul--Braganza 2026, the last cross-referenced to `AdmiralsLog/contributions/CONT2-hunting-proxy-patterns-formalization.md` and `P-0005`). Added eight studies that the v2 brief identified as gaps: Strevens (2003), Avin (2015) and (2019), Squazzoni--Gandelli (2013), Bianchi--Grimaldo--Bravo--Squazzoni (2018), Squazzoni--Bravo--Takács (2013), Bala--Goyal (2000), Jackson--Rogers (2007); master table extended to 46 rows with a new cluster D. Added the mapping section that ties every added study to a model object, an experiment or a premise, and states which brief-named parameters do not exist (`tau`). Two handed-over citations corrected against Crossref: the Squazzoni--Bravo--Takács study is a *Research Policy* experiment, not a JASSS ABM, and Avin's simulation is the 2015 EPSA13 chapter, not the 2018/2019 article. Three gap bullets added (lotteries, Strevens' defence of prestige allocation, network formation) and the niche claim restated against the new evidence.
- **Version 2.1** (15 November 2026) --- Errata pass: the Path B design menu left `docs/` and is archived at `AdmiralsLog/archive/path_b_dyadic_sketch_v1.md` (source and PDF together) once its probe landed. The Bala--Goyal row and the lambda note now cite `output/reports/path_b_probe_v1.md`, the executed Gate 2 sweep, alongside the archived menu, and the Bala--Goyal status reads that lambda is implemented in that sweep rather than only specified. No study, table row or claim changed.
