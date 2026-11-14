"""Agent level: the Scientist agent.

Extracted verbatim from the pre-restructure ``mesa_model.py`` (class
``Scientist``). The only change is that ``local_merit`` delegates to
``sciencemodels.metrics.local_merit`` instead of computing inline.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import mesa
import numpy as np
from mesa.space import Coordinate

from . import metrics

if TYPE_CHECKING:
    from .model import MyModel


class Scientist(mesa.Agent):
    curiosity: np.float64
    epsilon: np.float64
    prestige: np.float64
    prestige_visibility: np.float64
    prestige_vanishing: np.float64
    age: int
    last_tile_knowledge: np.float64
    model: MyModel
    visibility: np.float64
    current_local_merit: np.float64
    pos: Coordinate

    def __init__(self, model, curiosity, epsilon):
        super().__init__(model)
        self.curiosity = curiosity
        self.epsilon = epsilon
        self.prestige = np.float64(0.0)
        self.prestige_visibility = np.float64(0.0)
        self.prestige_vanishing = np.float64(0.0)
        self.age = 23
        self.last_tile_knowledge = np.float64(0.0)
        self.model = model
        self.visibility = 0
        self.current_local_merit = 0
        self.ideal_Merit= 0

    def compute_distance(self, other_pos: tuple[int, int]) -> int:
        x1, y1 = self.pos
        x2, y2 = other_pos
        n = self.model.size  # torus width/height

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # on a torus: shortest wrap distance in each dimension
        dx = min(dx, n - dx)
        dy = min(dy, n - dy)

        return (
            dx + dy
        )  # Manhattan distance on a torus

    def increase_prestige(self, value):
        self.prestige += value
        self.prestige_visibility += np.mean(
            [
                value ** (a.curiosity) * self.visibility ** (1 - a.curiosity)
                for a in self.model._scientists
                if a != self
            ]
        )
        # idea: we could also use individual distances instead of agent's visibility in the above specification.
        x = self.model.vanishing_factor_for_prestige
        if x != 1:
            temp = self.prestige_vanishing
            temp = temp * x  # vanishes past values
            temp += value  # adds the new one
            self.prestige_vanishing = temp
        else:
            self.prestige_vanishing += value

    def _compute_visibility(self, avg_agent_distance: np.float64) -> np.float64:
        return (self.model.agent_avg_distance + 1.0) / (avg_agent_distance + 1.0)

    def compute_all_rewards(
        self, pos: Coordinate, novelty: np.float64
    ) -> tuple[np.float64, np.float64]:
        """This part is just a test, we change the network reward by just the avg distance of
        all agents divided the average distance of the current agent"""
        if not self.model.use_distance:
            return (
                self.curiosity * novelty / self.model.avg_current_agent_knowledge,
                np.float64(0.0),
            )

        others = [a for a in self.model._scientists if a is not self]
        dists = np.fromiter(
            (a.compute_distance(pos) for a in others),
            dtype=np.float64,
            count=len(others),
        )
        prest = np.fromiter(
            (a.prestige_vanishing for a in others), dtype=np.float64, count=len(others)
        )

        avg_agent_distance = dists.mean()
        visibility = self._compute_visibility(avg_agent_distance)

        avg_agent_distance_p = avg_agent_distance
        if prest.sum() > 0.0:
            avg_agent_distance_p = np.average(
                dists,
                weights=prest,
            )

        visibility_used = visibility
        if self.model.use_visibility_reward:
            visibility_used = (self.model.agent_avg_distance + 1) / (
                avg_agent_distance_p + 1
            )

        # differentiate between visibility of a field vs. visibility of an agent?
        # (would not this formulation stay constant across neihbouring cells?; answer: no, it won't)
        reward = (novelty / self.model.avg_current_agent_knowledge) ** (
            self.curiosity
        ) * visibility_used ** (1 - self.curiosity)

        # currently we use only the above. however, we could employ three options:
        # (1) the above, where I am dragged towards others,
        # (2) an alternative version, where closeness of others impacts on the attractiveness of grids,
        # (3) some version of either (1) or (2) where the pull effect is weighted by prestige
        # else:
        #    reward = self.curiosity * Novelty/ self.model.avgcurrentAgentKnowledge + (1-self.curiosity) * visibility
        return reward, visibility

    def local_merit(self, pos) -> np.float64:
        return metrics.local_merit(self.model, pos)

    def step(self):
        """pick a random node and check if according to the agent preference it is better to move to that node"""
        neighbors_nodes = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False
        )
        # optionpos = self.model.rng.choice(neighbors_nodes)
        current_rw_novelty = self.model.compute_knowledge(self.pos)
        total_current_reward, currentVis = self.compute_all_rewards(
            self.pos, current_rw_novelty
        )
        best_reward = total_current_reward
        optionpos = self.pos  # fallback to current position
        best_vis = currentVis

        for neighbor in neighbors_nodes:
            neighbor_rw_novelty = self.model.compute_knowledge(neighbor)
            noise = self.rng.uniform(-1.0, 1.0)
            total_reward, visibility = self.compute_all_rewards(
                neighbor, neighbor_rw_novelty * (1 + self.epsilon * noise)
            )

            if total_reward > best_reward:
                best_reward = total_reward
                optionpos = neighbor
                best_vis = visibility

        if best_reward - total_current_reward > 0:
            self.model.new_place(self, optionpos)
            self.visibility = best_vis
        else:
            self.visibility = currentVis

        knowledge_before = self.model.compute_knowledge(self.pos)
        self.model.farm(self.pos)  # farming also updates AvgAgentKnowledge

        self.age += 1
        self.last_tile_knowledge = self.model.grid.properties["knowledge"].data[
            self.pos
        ]
        if self.model.prestige_accrual == "harvested":
            # accrue the harvested flow (knowledge just before farming * harvest rate)
            self.increase_prestige(knowledge_before * self.model.harvest)
        else:  # "post_tile": legacy rule, accrues the post-harvest tile value
            self.increase_prestige(self.last_tile_knowledge)
