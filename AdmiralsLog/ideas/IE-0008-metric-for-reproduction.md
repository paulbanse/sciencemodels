---
title: "IE-0008 - Metric for reproduction"
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

1. **Reproduction in the model should probably not be based on prestige:**
   - *Mechanism:* Using prestige as the metric for reproduction means using the goal to reward agents, in the context of the work it probably makes more sense to use a proxy measure or a mixed proxy-goal measure
   - *Examples:* One could use current visibility of a scientist, visibility multiplied by (normalized) prestige, or a balance between prestige and visibility depending on the current mean curiosity of the agents, ...
   - *Pro:* It will allow to investigate if science is selecting for people who do not have the best curiosity value
   - *Cons:* It opens the door to criticism on which metric we use. 

## Code-verified assessment

The visibility-based supervisor channel already exists. `entry_prestige_stock` is validated with `"prestige_visibility"` as an allowed value (`sciencemodels/model.py:183-190`), and `__generate_new_agents` weights supervisor draws by `getattr(a, entry_prestige_stock)` (`sciencemodels/model.py:411-423`). Setting `entry_prestige_stock="prestige_visibility"` (default `"prestige"`, `sciencemodels/model.py:144`) delivers the visibility reproduction option with no new code; a multiplicative visibility $\times$ normalised-prestige metric would still need a new stock option. The heritability premise does not hold, though: newcomers receive `self.initial_curiosity` via `agent_generation_function` (`sciencemodels/model.py:425-427`; default identity at `sciencemodels/model.py:137-139`), not the supervisor's curiosity. Curiosity is currently non-heritable, so reproduction cannot select for or against it; the proposed test requires a transmission/mutation channel first.
