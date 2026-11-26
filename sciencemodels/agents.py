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


#: ADR-0003 canonical clock: 1 step = 0.5 academic years (6 months).
AGE_AT_ENTRY = np.float64(25.0)
AGE_AT_RETIREMENT = np.float64(65.0)
AGE_INCREMENT_PER_STEP = np.float64(0.5)
STEPS_PER_YEAR = np.float64(2.0)


class Scientist(mesa.Agent):
    curiosity: np.float64
    epsilon: np.float64
    prestige: np.float64
    prestige_visibility: np.float64
    prestige_vanishing: np.float64
    age: np.float64
    last_tile_knowledge: np.float64
    model: MyModel
    visibility: np.float64
    current_local_merit: np.float64
    pos: Coordinate
    is_newcomer: bool
    harvest_profile: np.ndarray

    def __init__(self, model, curiosity, epsilon):
        super().__init__(model)
        self.curiosity = curiosity
        self.epsilon = epsilon
        self.prestige = np.float64(0.0)
        self.prestige_visibility = np.float64(0.0)
        self.prestige_vanishing = np.float64(0.0)
        self.age = AGE_AT_ENTRY
        self.last_tile_knowledge = np.float64(0.0)
        self.model = model
        self.visibility = 0
        self.current_local_merit = 0
        self.ideal_Merit = np.nan  # BUG-0006: unscored until update_ideal_merit runs
        # Path B dyadic state. The zero/false initialisation is
        # dynamics-neutral (no rng, no feedback into the step rule): every
        # reader is guarded behind dyadic_mode != "off" or
        # closeness_mode == "profile_overlap".
        self.is_newcomer = False
        self.harvest_profile = np.zeros(model.size**2, dtype=np.float64)
        self._pending_prestige = np.float64(0.0)
        self._pending_tile = None

    @property
    def tenure_steps(self) -> np.float64:
        """Career tenure in model steps under the ADR-0003 clock.

        Entry age is 25 and one step is six months, so
        ``tenure_steps = 2 * (age - 25)``; at age 65 a scientist has
        completed ``tau_max = 80`` steps.
        """
        return np.float64(
            STEPS_PER_YEAR * (self.age - AGE_AT_ENTRY)
        )

    @property
    def tenure(self) -> np.float64:
        """Alias for :attr:`tenure_steps` (P-0009's ``agent.tenure``)."""
        return self.tenure_steps

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

    def partner_proximity(self, pos: Coordinate) -> np.float64:
        """Mean inverse torus distance from ``pos`` to this agent's linked partners.

        ``1/(d+1)``: a co-located partner scores 1.0, the term decays with
        distance and never reaches 0 for a finite grid. Returns ``0.0`` when
        the agent holds no active link, which leaves the dyadic movement
        multiplier at exactly 1.0.

        Path B v2 (movement coupling). The link set is updated at the END of a
        step (``metrics.dyadic_step``), so the pull reads the previous step's
        links: a one-step lag, disclosed in the round's report.
        """
        partners = self.model.linked_partners(self)
        if not partners:
            return np.float64(0.0)
        total = np.float64(0.0)
        for partner in partners:
            total = total + np.float64(1.0) / np.float64(
                partner.compute_distance(pos) + 1
            )
        return total / np.float64(len(partners))

    def compute_all_rewards(
        self, pos: Coordinate, novelty: np.float64
    ) -> tuple[np.float64, np.float64]:
        """This part is just a test, we change the network reward by just the avg distance of
        all agents divided the average distance of the current agent"""
        if not self.model.use_distance:
            reward = self.curiosity * novelty / self.model.avg_current_agent_knowledge
            visibility = np.float64(0.0)
            return self._apply_move_pull(pos, reward), visibility

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
        # (3) is the Path B v2 movement coupling: see _apply_move_pull.
        return self._apply_move_pull(pos, reward), visibility

    def _apply_move_pull(self, pos: Coordinate, reward: np.float64) -> np.float64:
        """Path B v2: multiply a candidate tile's reward by the partner pull.

        ``reward *= 1 + dyadic_move_pull * partner_proximity(pos)`` makes tiles
        nearer the agent's declared partners more attractive, which is the one
        channel from the choice layer (``model.links``) to movement and hence
        to the co-harvest production layer. Skipped entirely when the pull is
        0.0 or the agent has no link, so ``dyadic_move_pull=0.0`` is
        bit-identical to Path B v1 (not merely equal to printed precision):
        no multiplication by 1.0 is performed at all.
        """
        if self.model.dyadic_mode == "off" or self.model.dyadic_move_pull <= 0.0:
            return reward
        proximity = self.partner_proximity(pos)
        if proximity <= 0.0:
            return reward
        return reward * (
            np.float64(1.0) + self.model.dyadic_move_pull * proximity
        )

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
        self.model.farm(self.pos, self)  # farming also updates AvgAgentKnowledge

        self.age += AGE_INCREMENT_PER_STEP
        self.last_tile_knowledge = self.model.grid.properties["knowledge"].data[
            self.pos
        ]
        if self.model.credit_sharing == "equal_split":
            # Path B Q8: buffer this step's accrual; the model pools and
            # splits it among linked co-located agents in end_loop_update.
            # (Accrual is therefore deferred to end-of-step under this rule
            # only - a mechanics change inherent to pooling, disclosed.)
            if self.model.prestige_accrual == "harvested":
                self._pending_prestige = np.float64(
                    knowledge_before * self.model.harvest
                )
            else:  # "post_tile": legacy rule, accrues the post-harvest tile value
                self._pending_prestige = np.float64(self.last_tile_knowledge)
            self._pending_tile = self.pos
        elif self.model.prestige_accrual == "harvested":
            # accrue the harvested flow (knowledge just before farming * harvest rate)
            self.increase_prestige(knowledge_before * self.model.harvest)
        else:  # "post_tile": legacy rule, accrues the post-harvest tile value
            self.increase_prestige(self.last_tile_knowledge)
