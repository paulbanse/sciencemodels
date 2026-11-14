"""Metric computations that separate cleanly from the model class.

Extracted verbatim from the pre-restructure ``mesa_model.py``:

- ``local_merit``        (was ``Scientist.local_merit``)
- ``update_ideal_merit`` (was ``MyModel.updateIdealMerit``)
- ``correlation_reporter`` (was a closure defined inside ``MyModel.__init__``)
- ``update_step_goals``  (was ``MyModel._update_step_goals``, milestone logic)

Each function takes the model (or the relevant objects) as an argument; the
original methods on ``Scientist`` / ``MyModel`` remain as thin delegates so
the public API is unchanged.

Entanglement notes (methods that could not become pure functions without a
logic change):

- ``update_ideal_merit`` mutates the model's ``previous_knowledge`` property
  layer in place while computing the ideal harvest and writes each agent's
  ``ideal_Merit`` attribute; it is therefore a function *of the model* and is
  driven from ``MyModel.updateIdealMerit``.
- ``update_step_goals`` writes the model's milestone attributes
  (``explored_50_step`` ... ``harvested_25_step``); same treatment.
- ``local_merit`` needs ``model.grid`` and ``model.compute_knowledge``; it is
  a function of the model and a position, with ``Scientist.local_merit``
  delegating to it.
- The core dynamics (``step``, ``farm``, ``end_loop_update``, ``new_place``,
  ``compute_knowledge``, ``update_knowledge``, newcomer generation and new
  questions) remain methods on ``MyModel``: they *are* the model class.
"""
from __future__ import annotations

import numpy as np


def local_merit(model, pos) -> np.float64:
    neighbors = model.grid.get_neighborhood(
        pos, moore=False, include_center=False
    )

    # positions: center + 4-neighbors
    positions = [pos, *neighbors]

    # compute into a NumPy array (float64)
    knowledge = np.fromiter(
        (model.compute_knowledge(p) for p in positions),
        dtype=np.float64,
        count=len(positions),
    )

    return knowledge.mean()


def update_ideal_merit(model):
    """ computes the loss for each agent compared to the best position they could have in the ideal harvest situation"""
    temp_knowledge_map= model.grid.properties["previous_knowledge"].data
    
    # compute the harvest made
    IdealHarvestList = []
    L = [(a.pos, a) for a in model.agents]
    L = sorted(L, key=lambda L: L[0])  #sort for positions so same positions are together
    ActualHarvestList = []
    pastpos, redone = -1,0
    for kpos,k in L:
        know = temp_knowledge_map[kpos]
        if kpos == pastpos:
            redone += 1
        else:
            redone = 0
        know = know * ((1- model.harvest) ** redone)
        pastpos = kpos
        ActualHarvestList.append([know*model.harvest, kpos, k])
    
    #compute ideal harvest: greedy allocation, one slot per agent in the
    #model (under entry the population exceeds number_agents, so the slot
    #count follows len(model.agents))
    for k in range(len(model.agents)):
        pos = np.unravel_index(temp_knowledge_map.argmax(), temp_knowledge_map.shape) 
        IdealHarvestList.append((temp_knowledge_map[pos]*model.harvest,pos))
        temp_knowledge_map[pos] = temp_knowledge_map[pos] * (1- model.harvest)
    
    #compute loss per agent: sorted-to-sorted pairing - the k-th largest
    #actual harvest is matched against the k-th largest greedy-ideal
    #harvest. (Pre-fix this paired every actual harvest against the ideal
    #list's static head, which only advanced on an exact-zero loss, so
    #below-head agents were measured against the single best tile.)
    ActualHarvestList = sorted(ActualHarvestList, key= lambda L: L[0], reverse=True)
    IdealHarvestList = sorted(IdealHarvestList, key= lambda L: L[0], reverse=True)
    for k in range(len(model.agents)):
        loss = ActualHarvestList[k][0] - IdealHarvestList[k][0]
        ActualHarvestList[k] = ActualHarvestList[k] + [loss]
    
    # average loss per position
    decided_pos = []
    for (kharvested, kpos, k, loss) in ActualHarvestList:
        if kpos not in decided_pos:
            agents_on_site = [a[2] for a in ActualHarvestList if a[1]== kpos]
            loss_on_site = sum([a[3] for a in ActualHarvestList if a[1]== kpos])
            for a in agents_on_site:
                a.ideal_Merit = loss_on_site/(len(agents_on_site))
            decided_pos.append(kpos)
    
    # cumulative per-agent ideal-merit loss over the run (welfare-round
    # need); lazy init keeps agents.py outside the edit boundary
    for a in model.agents:
        a.ideal_merit_cum = getattr(a, "ideal_merit_cum", 0.0) + a.ideal_Merit
    
    model.grid.properties["previous_knowledge"].data = model.grid.properties["knowledge"].data.copy()


def correlation_reporter(merit_type, prestige_type):
    def reporter(m):
        Prest = [getattr(a, prestige_type) for a in m.agents]
        Mert = [getattr(a, merit_type) for a in m.agents]
        if  np.std(Prest) > 0 and np.std(Mert) > 0:
            return np.corrcoef(Prest,Mert)[0, 1]
        else:
            return 0
    return reporter


def update_step_goals(model) -> None:
    sentinel = model._default_steps_thresholds
    steps = model.steps

    milestones = (
        ("explored_50_step", "explored_percentage", 0.5),
        ("explored_90_step", "explored_percentage", 0.9),
        ("weighted_50_step", "explored_weighted_by_initial_knowledge", 0.5),
        ("weighted_90_step", "explored_weighted_by_initial_knowledge", 0.9),
        ("harvested_10_step", "percentage_knowledge_harvested", 0.10),
        ("harvested_25_step", "percentage_knowledge_harvested", 0.25),
    )

    for step_attr, metric_attr, threshold in milestones:
        if (
            getattr(model, step_attr) == sentinel
            and getattr(model, metric_attr) >= threshold
        ):
            setattr(model, step_attr, steps)
