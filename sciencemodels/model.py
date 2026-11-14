"""Model level: the Mesa model and the parametric-exploration entry point.

Extracted verbatim from the pre-restructure ``mesa_model.py`` (class
``MyModel`` and module function ``generate_data_parametric_exploration``).
Methods whose bodies live in ``sciencemodels.metrics``, ``sciencemodels.io``
or ``sciencemodels.viz`` are thin delegates with unchanged names and
signatures; the core dynamics (``step``, ``farm``, ``end_loop_update``,
``new_place``, ``compute_knowledge``, ``update_knowledge``, newcomer
generation, new questions) remain methods here — they are the model class.
"""
from __future__ import annotations

import csv
import itertools
import os
from typing import Any, Callable

import mesa
import numpy as np
from mesa.space import Coordinate, PropertyLayer
from numpy.random import Generator

from . import io, metrics, viz
from .agents import Scientist


class MyModel(mesa.Model):
    number_agents: int
    agent_generation_time: np.float64
    new_questions: bool
    size: int
    steps: int
    harvest: np.float64
    initial_curiosity: np.float64
    epsilon: np.float64
    use_distance: bool
    _scientists: list[Scientist]
    total_initial_knowledge: np.float64
    avg_current_agent_knowledge: np.float64
    agent_avg_distance: np.float64
    entry_prestige_stock: str
    dyadic_mode: str
    dyadic_lambda: np.float64
    dyadic_proposal_regime: str
    credit_sharing: str
    closeness_mode: str
    dyadic_visibility_stock: str
    links: set

    def _init_grid(
        self, cell_init_function: Callable[[int, int, int, Any, Generator]]
    ):
        self.grid = mesa.space.MultiGrid(self.size, self.size, torus=True)
        self.grid.add_property_layer(
            PropertyLayer(
                "knowledge", self.size, self.size, np.float64(0.0), dtype=np.float64
            )
        )
        self.grid.add_property_layer(
            PropertyLayer(
                "previous_knowledge", self.size, self.size, np.float64(0.0), dtype=np.float64
            )
        )
        self.grid.add_property_layer(
            PropertyLayer(
                "initial_knowledge", self.size, self.size, np.float64(0.0), dtype=np.float64,
            )
        )
        self.grid.add_property_layer(
            PropertyLayer("explored", self.size, self.size, False, dtype=bool)
        )
        
        for pos_x, pos_y in list(itertools.product(range(self.size), range(self.size))):
            initial_value = cell_init_function(
                pos_x, pos_y, self.size, self.generation_params, self.rng
            )
            self.grid.properties["knowledge"].data[pos_x, pos_y] = initial_value
            self.total_initial_knowledge += initial_value

        min_knowledge, max_knowledge = (
            np.min(self.grid.properties["knowledge"].data),
            np.max(self.grid.properties["knowledge"].data),
        )

        for pos_x, pos_y in list(itertools.product(range(self.size), range(self.size))):
            val = self.grid.properties["knowledge"].data[pos_x, pos_y]
            if self.knowledge_normalization == "share":
                self.grid.properties["knowledge"].data[pos_x, pos_y] = (
                    val / self.total_initial_knowledge
                )
            else:  # "minmax"
                self.grid.properties["knowledge"].data[pos_x, pos_y] = (
                    val - min_knowledge
                ) / (max_knowledge - min_knowledge) * 0.95 + 0.05

            self.grid.properties["initial_knowledge"].data[pos_x, pos_y] = (
                self.grid.properties["knowledge"].data[pos_x, pos_y]
            )
            self.grid.properties["previous_knowledge"].data[pos_x, pos_y] = (
                self.grid.properties["knowledge"].data[pos_x, pos_y]
            )
            self.grid.properties["explored"].data[pos_x, pos_y] = False
        self.total_initial_knowledge = np.float64(1.0)

    def _init_agents(self, agent_generation_function):
        for _ in range(self.number_agents):
            w = agent_generation_function(
                self.initial_curiosity, self.rng, self.generation_params
            )
            a = Scientist(self, w, self.epsilon)
            coords = (self.rng.integers(0, self.size), self.rng.integers(0, self.size))
            a.age = self.rng.choice(np.arange(23, 50))
            self.grid.place_agent(a, coords)
            self.grid.properties["explored"].data[coords] = True
            a.last_tile_knowledge = self.grid.properties["knowledge"].data[coords]
            a.current_local_merit = a.local_merit(a.pos)

        self._scientists = [
            agent for agent in self.agents if isinstance(agent, Scientist)
        ]

    def __init__(
        self,
        n_agents: int,
        initial_curiosity: np.float64,
        epsilon: np.float64,
        harvest: np.float64,
        grid_size: int,
        cell_init_function: Callable[[int, int, int, Dict, Generator], np.float64],
        use_distance: bool,
        generation_params={"seed": 0},
        agent_generation_time: np.float64 = np.float64(-1.0),
        new_questions: bool = False,
        agent_seed: int = 0,
        step_limit: int = 400,
        agent_generation_function: Callable[
            [np.float64, Generator, Dict], np.float64
        ] = lambda cur, rng, params: cur,
        vanishing_factor_for_prestige: np.float64 = np.float64(0.9),
        use_visibility_reward: bool = True,
        knowledge_normalization: str = "share",
        prestige_accrual: str = "harvested",
        entry_prestige_stock: str = "prestige",
        dyadic_mode: str = "off",
        dyadic_lambda: float = 0.5,
        dyadic_proposal_regime: str = "newcomer",
        credit_sharing: str = "structural",
        closeness_mode: str = "grid",
        dyadic_visibility_stock: str = "prestige_visibility",
    ):
        """parameters :  number of agents, number of connections, curiosity, noise intensity, harvest,  size, initCellfunc"""
        super().__init__(seed=agent_seed)

        self.generation_params = generation_params | {
            "initCellFunc": cell_init_function.__name__, "seed": self._seed
        }
        self.number_agents = n_agents
        self.agent_generation_time = agent_generation_time
        self.new_questions = new_questions
        self.size = grid_size
        self.harvest = harvest
        self.initial_curiosity = initial_curiosity
        self.epsilon = epsilon
        self.use_distance = use_distance
        self.step_limit = step_limit
        self.vanishing_factor_for_prestige = vanishing_factor_for_prestige
        self.use_visibility_reward = use_visibility_reward
        self.agent_generation_function = agent_generation_function
        if knowledge_normalization not in ("share", "minmax"):
            raise ValueError(
                "knowledge_normalization must be 'share' or 'minmax', got "
                + repr(knowledge_normalization)
            )
        self.knowledge_normalization = knowledge_normalization
        if prestige_accrual not in ("harvested", "post_tile"):
            raise ValueError(
                "prestige_accrual must be 'harvested' or 'post_tile', got "
                + repr(prestige_accrual)
            )
        self.prestige_accrual = prestige_accrual
        if entry_prestige_stock not in (
            "prestige", "prestige_vanishing", "prestige_visibility"
        ):
            raise ValueError(
                "entry_prestige_stock must be 'prestige', 'prestige_vanishing'"
                " or 'prestige_visibility', got " + repr(entry_prestige_stock)
            )
        self.entry_prestige_stock = entry_prestige_stock
        if dyadic_mode not in ("off", "logit", "threshold"):
            raise ValueError(
                "dyadic_mode must be 'off', 'logit' or 'threshold', got "
                + repr(dyadic_mode)
            )
        if not (0.0 <= dyadic_lambda <= 1.0):
            raise ValueError(
                "dyadic_lambda must be in [0, 1], got " + repr(dyadic_lambda)
            )
        if dyadic_proposal_regime not in ("newcomer", "universal"):
            raise ValueError(
                "dyadic_proposal_regime must be 'newcomer' or 'universal', got "
                + repr(dyadic_proposal_regime)
            )
        if credit_sharing not in ("structural", "equal_split"):
            raise ValueError(
                "credit_sharing must be 'structural' or 'equal_split', got "
                + repr(credit_sharing)
            )
        if closeness_mode not in ("grid", "profile_overlap"):
            raise ValueError(
                "closeness_mode must be 'grid' or 'profile_overlap', got "
                + repr(closeness_mode)
            )
        if dyadic_visibility_stock not in (
            "prestige_visibility", "prestige", "prestige_vanishing", "entry"
        ):
            raise ValueError(
                "dyadic_visibility_stock must be 'prestige_visibility',"
                " 'prestige', 'prestige_vanishing' or 'entry', got "
                + repr(dyadic_visibility_stock)
            )
        self.dyadic_mode = dyadic_mode
        self.dyadic_lambda = np.float64(dyadic_lambda)
        self.dyadic_proposal_regime = dyadic_proposal_regime
        self.credit_sharing = credit_sharing
        self.closeness_mode = closeness_mode
        self.dyadic_visibility_stock = dyadic_visibility_stock
        # Path B dyadic state: the declared (choice) link set, the per-step
        # co-harvest (production) teams, and the four dyadic statistics.
        # With dyadic_mode="off" no rng is consumed and no dynamic state
        # changes - the pre-Path-B dynamics are bit-identical.
        self.links = set()
        self.teams = []
        self._ever_active_pairs = set()
        self._ever_active_links = set()
        self._dyadic_stats = {
            "mean_team_size": np.float64(0.0),
            "repeated_collab_share": np.float64(0.0),
            "dyadic_assortativity_prestige": np.float64(0.0),
            "n_active_links": 0,
            "link_repeated_share": np.float64(0.0),
            "link_assortativity_prestige": np.float64(0.0),
        }

        self.total_initial_knowledge = np.float64(0.0)
        self.avg_current_agent_knowledge = np.float64(0.0)
        self.agent_avg_distance = np.float64(0.0)

        self.explored_weighted_by_initial_knowledge = np.float64(0.0)
        self.percentage_knowledge_harvested = np.float64(0.0)
        self.explored_percentage = np.float64(0.0)
        self._default_steps_thresholds = step_limit * (1.25)
        self.explored_50_step = self._default_steps_thresholds
        self.explored_90_step = self._default_steps_thresholds
        self.weighted_50_step = self._default_steps_thresholds
        self.weighted_90_step = self._default_steps_thresholds
        self.harvested_10_step = self._default_steps_thresholds
        self.harvested_25_step = self._default_steps_thresholds

        self._init_grid(cell_init_function)
        self._init_agents(agent_generation_function)

        if not (
            self.use_distance
            or initial_curiosity == 0
            or epsilon == 0
            or use_visibility_reward
        ):
            print(
                "Agents will NOT use distance information when choosing where to go, curiosity and noise will have similar effects"
            )

        self.update_knowledge()
        # updateIdealMerit is deliberately NOT called here: it runs inside
        # end_loop_update() below, and a second init-time call would
        # double-count step 0 in the ideal_merit_cum accumulator.
        # The reporters below are the model-side counterparts of the empirical
        # moments documented in docs/calibration_moments.yaml. That file is an
        # OPTION MENU only: per the owner's ruling (2026-11-13) the model is
        # NOT calibrated to empirical data at this stage; the moments and their
        # model counterparts are documented options for later use.
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Step": lambda m: m.steps,
                "mean_age": lambda m: m.agents.agg("age", np.mean),
                "mean_prestige": lambda m: m.agents.agg("prestige", np.mean),
                "mean_vanishing_prestige": lambda m: m.agents.agg(
                    "prestige_vanishing", np.mean
                ),
                "mean_visibility_prestige": lambda m: m.agents.agg(
                    "prestige_visibility", np.mean
                ),
                "top1pct_prestige_share": lambda m: (
                    np.sum(
                        [
                            a.prestige
                            for a in m.agents
                            if a.prestige
                            >= np.percentile([ag.prestige for ag in m.agents], 99)
                        ]
                    )
                    / np.sum([a.prestige for a in m.agents])
                    if np.sum([a.prestige for a in m.agents]) > 0
                    else 0
                ),
                "top10pct_prestige_share": lambda m: (
                    np.sum(
                        [
                            a.prestige
                            for a in m.agents
                            if a.prestige
                            >= np.percentile([ag.prestige for ag in m.agents], 90)
                        ]
                    )
                    / np.sum([a.prestige for a in m.agents])
                    if np.sum([a.prestige for a in m.agents]) > 0
                    else 0
                ),
                "avgcurrentAgentKnowledge": lambda m: m.avg_current_agent_knowledge,
                "explored_percentage": lambda m: m.explored_percentage,
                "explored_weighted_by_initial_knowledge": lambda m: m.explored_weighted_by_initial_knowledge,
                "total_initial_knowledge": lambda m: m.total_initial_knowledge,
                "avg_knowledge_on_grid": lambda m: np.mean(
                    m.grid.properties["knowledge"].data, dtype=np.float64
                ),
                "best_knowledge": lambda m: np.max(m.grid.properties["knowledge"].data),
                "avg_distance_between_agents": lambda m: m.agent_avg_distance,
                "percentage_knowledge_harvested": lambda m: m.percentage_knowledge_harvested,
                "Corr_prestige_local_merit": metrics.correlation_reporter("current_local_merit", "prestige"),
                "Corr_prestige_ideal_merit": metrics.correlation_reporter("ideal_Merit", "prestige"),
                "Corr_prestige_visibility_local_merit": metrics.correlation_reporter("current_local_merit", "prestige_visibility"),
                "Corr_prestige_visibility_ideal_merit": metrics.correlation_reporter("ideal_Merit", "prestige_visibility"),
                "Corr_prestige_vanishing_local_merit": metrics.correlation_reporter("current_local_merit", "prestige_vanishing"),
                "Corr_prestige_vanishing_ideal_merit": metrics.correlation_reporter("ideal_Merit", "prestige_vanishing"),
                "Ideal_merit_cum": lambda m: np.mean(
                    [getattr(a, "ideal_merit_cum", 0.0) for a in m.agents]
                ),
                "mean_team_size": lambda m: m._dyadic_stats["mean_team_size"],
                "repeated_collab_share": lambda m: m._dyadic_stats["repeated_collab_share"],
                "dyadic_assortativity_prestige": lambda m: m._dyadic_stats["dyadic_assortativity_prestige"],
                "n_active_links": lambda m: m._dyadic_stats["n_active_links"],
                "link_repeated_share": lambda m: m._dyadic_stats["link_repeated_share"],
                "link_assortativity_prestige": lambda m: m._dyadic_stats["link_assortativity_prestige"],
            },
            agent_reporters={
                "prestige": "prestige",
                "prestige_vanishing": "prestige_vanishing",
                "prestige_visibility": "prestige_visibility",
                "last_tile_knowledge": "last_tile_knowledge",
                "curiosity": "curiosity",
                "local_merit": "current_local_merit",
                "ideal_merit_cum": "ideal_merit_cum",
                "is_newcomer": "is_newcomer",
            },
        )
        self.end_loop_update()

    def update_knowledge(self):
        self.avg_current_agent_knowledge = np.mean(
            [agent.last_tile_knowledge for agent in self._scientists], dtype=np.float64
        )

    def new_place(self, agent1, coords, newAgent=False):
        """function used to modify an agent location and update the surrounding agent's distance list"""
        if not (newAgent):
            self.grid.remove_agent(agent1)
        self.grid.place_agent(agent1, coords)
        self.grid.properties["explored"].data[coords] = True
    
    def updateIdealMerit(self):
        """ computes the loss for each agent compared to the best position they could have in the ideal harvest situation"""
        return metrics.update_ideal_merit(self)

    def compute_knowledge(self, pos: Coordinate) -> np.float64:
        posX, posY = pos
        return self.grid.properties["knowledge"].data[posX, posY]

    def farm(self, pos, agent=None):
        knowledge_before = self.grid.properties["knowledge"].data[pos]
        self.grid.properties["knowledge"].data[pos] = (
            knowledge_before * (1 - self.harvest)
        )
        if agent is not None and self.closeness_mode == "profile_overlap":
            # Path B Q2: per-agent flow-weighted harvest profile, consumed
            # by the profile_overlap closeness mode only
            agent.harvest_profile[pos[0] * self.size + pos[1]] += (
                knowledge_before * self.harvest
            )
        self.update_knowledge()

    def __generate_new_agents(self):
        if self.agent_generation_time > 0:
            agents_to_generate: int = np.floor(
                self.steps * self.agent_generation_time
            ) - np.floor((self.steps - 1) * self.agent_generation_time)
            entry_stock = self.entry_prestige_stock
            prestige_agent_tuples = [
                (getattr(a, entry_stock), a) for a in self._scientists
            ]
            totalPrestige = sum([getattr(a, entry_stock) for a in self._scientists])
            supervisors = []
            for _ in range(int(agents_to_generate)):
                val = self.rng.uniform(0, totalPrestige)
                for p, a in prestige_agent_tuples:
                    if val <= p:
                        supervisors.append(a)
                        break
                    val -= p
            for sup in supervisors:
                w = self.agent_generation_function(
                    self.initial_curiosity, self.rng, self.generation_params
                )
                a = Scientist(self, w, self.epsilon)
                # accumulator init: covers the birth-step collection and
                # final-wave newcomers, for whom no updateIdealMerit follows
                # their birth (agents.py is outside this round's edit boundary)
                a.ideal_merit_cum = np.float64(0.0)
                # Path B Q5: permanent newcomer flag (proposal regime)
                a.is_newcomer = True
                coords = sup.pos
                self.grid.place_agent(a, coords)
                self._scientists.append(a)
            for agent in self._scientists:
                self.grid.properties["explored"].data[agent.pos] = True

    def _redistribute_equal_split(self):
        """Path B Q8: pool each tile's buffered prestige accrual among the
        tile's LINKED co-located agents and credit the equal split. Agents
        with no link to any co-located peer keep their own accrual. Called
        once per step from end_loop_update, before the dyadic scores, so
        partner scores see the split prestige."""
        by_tile = {}
        for a in self._scientists:
            if a._pending_tile is None:
                continue
            by_tile.setdefault(a._pending_tile, []).append(a)
        for agents_on_tile in by_tile.values():
            linked = [
                a
                for a in agents_on_tile
                if any(
                    frozenset((a.unique_id, b.unique_id)) in self.links
                    for b in agents_on_tile
                    if b is not a
                )
            ]
            if len(linked) >= 2:
                pool = np.float64(
                    sum(a._pending_prestige for a in linked) / len(linked)
                )
                for a in linked:
                    a.increase_prestige(pool)
                for a in agents_on_tile:
                    if a not in linked:
                        a.increase_prestige(a._pending_prestige)
            else:
                for a in agents_on_tile:
                    a.increase_prestige(a._pending_prestige)
            for a in agents_on_tile:
                a._pending_prestige = np.float64(0.0)
                a._pending_tile = None

    def _generate_new_questions(self):
        if self.new_questions:
            occupied_pos = list(set([agent.pos for agent in self.agents]))
            for pos_x, pos_y in occupied_pos:
                if self.grid.properties["knowledge"].data[pos_x, pos_y] < 0.025:
                    self.grid.properties["knowledge"].data[pos_x, pos_y] = (
                        self.rng.random()
                    )

    def _update_step_goals(self) -> None:
        return metrics.update_step_goals(self)

    def end_loop_update(self):
        if self.credit_sharing == "equal_split":
            # Path B Q8: pool and split this step's buffered accrual among
            # linked co-located agents BEFORE the dyadic scores are computed
            self._redistribute_equal_split()
        self.updateIdealMerit()
        self.__generate_new_agents()
        if self.dyadic_mode != "off":
            metrics.dyadic_step(self)
        self._generate_new_questions()

        self.explored_percentage = np.sum(self.grid.properties["explored"].data) / (
            self.size**2
        )
        self.explored_weighted_by_initial_knowledge = (
            np.sum(
                self.grid.properties["explored"].data
                * (self.grid.properties["initial_knowledge"].data)
            )
            / self.total_initial_knowledge
        )
        self.agent_avg_distance = np.mean(
            [
                np.mean(
                    [a.compute_distance(b.pos) for b in self._scientists if a != b],
                    dtype=np.float64,
                )
                for a in self._scientists
            ],
            dtype=np.float64,
        )
        self.percentage_knowledge_harvested = (
            self.total_initial_knowledge
            - np.sum(self.grid.properties["knowledge"].data)
        ) / self.total_initial_knowledge

        for agent in self._scientists:
            agent.current_local_merit = agent.local_merit(agent.pos)

        # Path B Q6: co-harvest teams and the four dyadic statistics;
        # computed in every mode (the structural baseline needs no links)
        metrics.compute_teams(self)

        self.datacollector.collect(self)
        self._update_step_goals()

    def step(self, endupdate=True):
        # compute everything and let agents take the decision
        temp = self.agents.select(lambda a: isinstance(a, Scientist))
        temp.random = self.rng
        temp.shuffle_do("step")
        if endupdate:
            self.end_loop_update()

    def Vizualize_3d(self):
        return viz.Vizualize_3d(self)

    def plot(self):
        return viz.plot(self)

    def compute_distanceToAgents(self):
        Distances = np.zeros((self.size, self.size))
        for X in range(self.size):
            for Y in range(self.size):
                dist = 0
                for agent2 in self.agents:
                    dist += agent2.compute_distance((X, Y))
                Distances[X, Y] = dist
        return Distances
    
    def get_report(self):
        return io.build_report(self)

    def animate_steps(self, dynamic_plot=True, csv_name="data", end_report_file=""):
        return viz.animate_steps(self, dynamic_plot=dynamic_plot, csv_name=csv_name, end_report_file=end_report_file)

    def run_mode_for_bulk(self, longitudinal=False):
        for k in range(self.step_limit):
            self.step(True)
        if longitudinal == False:
            return self.get_report()
        else:
            return self.datacollector.get_agent_vars_dataframe()
    

def probe_agent_generation_binding(
    agent_generation_function, rng, params, mu_lo=0.1, mu_hi=0.9
):
    """Binding probe for a swept agent_generation_function.

    Calls the generator twice with different ``mu`` on the SAME rng and
    compares the draws. Identical draws mean the generator ignores both
    ``mu`` and the rng state (a constant): the swept axis is inert and the
    probe warns loudly. Generators that read ``mu`` (identity) or advance
    the rng (uniform) pass. Disclosed limitation: a generator that ignores
    ``mu`` but still draws from ``rng`` (e.g. ``landscapes.uniform``) is
    mu-inert yet passes this two-run identity check - the sweep-level guard
    for that case is comparing the first completed cells along the axis, as
    the welfare round did.
    """
    d1 = agent_generation_function(mu_lo, rng, params)
    d2 = agent_generation_function(mu_hi, rng, params)
    inert = bool(d1 == d2)
    if inert:
        print(
            "WARNING: agent_generation_function "
            + repr(
                getattr(
                    agent_generation_function,
                    "__name__",
                    agent_generation_function,
                )
            )
            + " returned IDENTICAL draws for mu=" + repr(mu_lo) + " and mu="
            + repr(mu_hi)
            + " on the same rng - the swept axis (initial_curiosity) is INERT."
        )
    return inert


def generate_data_parametric_exploration(
    filename,
    param_grid,
    repeats_per_setting=10,
    change_landscape_seed=False,
    intention="w",
    skip_to=0,
    longitudinal=False,
):
    import itertools

    param_as_lists = [a for a in param_grid.keys() if type(param_grid[a]) == list]
    varying_params = [a for a in param_as_lists if len(param_grid[a]) > 1]
    # Create a list of all combinations of parameters
    param_combinations = list(
        itertools.product(
            *[
                param_grid[k] if k in param_as_lists else [param_grid[k]]
                for k in param_grid.keys()
            ]
        )
    )

    # Create a list of parameter names
    param_names = list(param_grid.keys())
    # Binding guard (welfare-round need): probe a swept
    # agent_generation_function for mu-inertness before burning the grid.
    if callable(param_grid.get("agent_generation_function")):
        probe_agent_generation_binding(
            param_grid["agent_generation_function"], np.random.default_rng(0), {}
        )
    # Loop through all combinations of parameters
    param_range = range(len(param_combinations))
    if skip_to > 0:
        param_range = range(skip_to - 1, skip_to)
        print("skipping to", skip_to, "out of", len(param_combinations))

    if not (longitudinal):
        print(filename + ".csv", intention)
        needs_header = not (io.is_non_zero_file(filename + ".csv"))
        f = open(filename + ".csv", intention)
        writer = csv.writer(f)
        if needs_header or intention == "w":
            # create a dummy model to get the header
            param_set = param_combinations[0]
            all_params = {param_names[i]: param_set[i] for i in range(len(param_names))}
            all_params["step_limit"] = 1
            model = MyModel(**all_params)
            row = model.run_mode_for_bulk()
            writer.writerow(sorted(row.keys()))
            del model

    if longitudinal:
        os.makedirs(filename, exist_ok=True)
        f = open(f"{filename}/" + "fixed_params.txt", "w")
        f.write(
            str(
                {k: param_grid[k] for k in param_grid.keys() if k not in varying_params}
            )
        )
        f.close()

    for idx in param_range:
        param_set = param_combinations[idx]
        all_params = {param_names[i]: param_set[i] for i in range(len(param_names))}
        print("now processing param set ", idx + 1, "out of", len(param_combinations))
        for repeat in range(repeats_per_setting):
            all_params["agent_seed"] = repeat
            if change_landscape_seed:
                all_params["generation_params"] = all_params.get(
                    "generation_params", {}
                ) | {"seed": repeat}
            model = MyModel(**all_params)
            row = model.run_mode_for_bulk(longitudinal=longitudinal)
            if longitudinal:
                row.to_csv(
                    f"{filename}/run_"
                    + "_".join(
                        [
                            f"{key}{value}"
                            for key, value in all_params.items()
                            if key in varying_params + ["agent_seed"]
                        ]
                    )
                    + ".csv",
                    index=False,
                )
            else:
                writer.writerow([row[k] for k in sorted(row.keys())])
            del model
    f.close()
