---
title: "IE-0007  - Draft review proposals"
author: "Paul"
date: "2026-11-24"
tags: [idea]
project: "sciencemodels"
status: idea
---

- **Status:** idea
- **Date:** 2026-11-24
- **Related:** 

## The idea

1. **Trying to make the framing closer to real academia:**
   - *Mechanism:* In the redaction of the article, try justifying design parameter values regarding using the general context. 
   - *Examples:* Let's say a step in the model corresponds to 6 month in an academic career. When agents are not immortals, one could say that then the generation parameter and generational replacement should allow for a full generation to be renewed within 80 steps, and people should retire for the population to stay relatively constant. Similarly, in science, an article looses importance in one's career as it gets older. The scale of the vanishing parameter can be estimated and justified in the draft based on real life situations, for example one could say that the prestige from an article published 5 years ago is half of its initial prestige. Justifications like these should be investigated for most of the parameters that are unjustified in the model.
   - *Pro:* It gives a new way to justify some parameters that were before unjustified and increases the likelihood to be relevant for a reader in academia
   - *Cons:* It requires a bit more explanation in the article

2. **The vanishing parameter tested is too big:**
   - *Mechanism:* testing vanishing values of 0.5, 0.9 and 1.0 creates extremely strong vanishment, lower values such as 0.1 for example would be more realistic. 
   - *Pros:* this will allow having some longer term signal on prestige which makes it more likely to correlate with ideal Merit
   - *Cons:* not yet identified
   
3. **The use of distance= 0 version is exposed confusingly in the article:**
   - *Mechanism:* In the draft, the simulations where distance = 0 appear are called simulations with "jump" but there is no mention of a jump mechanism. If distance = 0 is useful it has to be justified, if the only thing distance = 0 does is to de-activate visibility, then it is hard to justify using it with different level of curiosity.
   - *Pros:* Potentially delete the distance = 0 option, or make it usefull for the reader
   - *Cons:* difficult to forsee given that distance = 0 is not explained

4. **Standardize simulation procedure:**
   - *Mechanism:* a paragraph should be written that summarizes what we use as a standard simulation procedure. How many steps in a simulation, what is the stopping criterion. Values of curiosity can be picked to be representative of the stage1, kink, stage2 situations (for example 0.5, 0.7, 0.9) in order to allow for representation along a different x axis.
   - *Examples:* Figure 4 first panel shows the result of the simulation at 10% knowledge harvested. One could wonder why was 10% used in that situation, if there's a justification it is not present
   - *Pros:* helps the reader understand what is the normal situation and which parameters are used in a single section
   - *Cons:* uses space in the potential article.

5. **The use of the word "lab" is not justified:**
   - *Mechanism:* the article uses the word lab probably in reference to Smaldino, but this usage is unclear and needs to be justified or deleted
   - *Pros:* Deletion would make the article less confusing, and smaller
   - *Cons:* 
   
6. **Figure 9 method is interesting:**
   - *Mechanism:* the method used to produce figure 9 is very interesting, it questions how beneficial is the fact that the new researchers are created on the cell of their supervisor. And it shows that despite putting the researcher on a cell that's for sure already explored, it also puts it into a cell that's close to interesting research spots. This is relevant in real life and an interesting question, I would suggests other design questions of the model to be investigated this way. 
   - *Pros:* 
   - *Cons:* 

7. **Figure 1: Simulation grid examples**
   - *Mechanism:* Figure 1 identifies different phases (Stage 1,2 and kink) with very few data point, I would advise having at least 2 point per phase to identify a bit more precisely the transition. Once these stages are properly identified, I would advise adding in Figure 1 three panels with representative examples of research behaviors during simulations, potentially in a noisy landscape: In Stage 1, researchers would be scattered accross the board, in the kink phase, researcher would be grouping but still exploring, and in stage 3 researchers would be exploring much less.
   - *Pros:* More instutitive visualization of our work
   - *Cons:* It could look a bit basic.

8. **Missing Visualization**
   - *Mechanism:* There is only one temporal visualization of simulation and it's to look at the evolution of corr(prestige, ideal Merit). Investigate the different possible temporal visualizations, for example Figure 2 could use the standardize procedure (see point 4 of this file) and show the temporal dynamics of the different simulations instead of the end point. Or one could investigate the evolution of curiosity along time in populations that reproduce for example.
   - *Pros:* It will give a more instinctive understanding of the work
   - *Cons:* uses space

## Code-verified assessment

- **#2 is inverted by the code.** `vanishing_factor_for_prestige` is applied in `Scientist.increase_prestige`: the accumulated stock is multiplied by $x$ and the new accrual added (`sciencemodels/agents.py:84-91`, `temp = temp * x; temp += value`). $x$ is a retention factor (1.0 = no forgetting; 0.5 = halve the stock per accrual), not a decay rate. The tested values 0.5/0.9/1.0 already span strong-to-no forgetting; lowering to 0.1 would strengthen forgetting, not weaken it, and weakening it means raising $x$ toward 1.0. It also acts per accrual event (roughly per step), so a "half after five years" reading needs a step-to-time mapping.
- **#3 is confirmed, with a stronger consequence.** With `use_distance=0`, `compute_all_rewards` returns `curiosity * novelty / avg_current_agent_knowledge` and zero visibility (`sciencemodels/agents.py:123-126`), while `Scientist.step` still evaluates only the 4-neighbour von Neumann neighbourhood (`sciencemodels/agents.py:194-196`). This is local search with visibility off, not jump search. Because curiosity is a common multiplicative constant across all candidate rewards, it cancels in the comparison; under `use_distance=0` curiosity cannot alter movement (only `epsilon` noise can), so the draft's "different levels of curiosity" comparison is mechanically empty unless the reward is reformulated.

