from __future__ import annotations
from mesa.space import Coordinate, PropertyLayer
import mesa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
from matplotlib import cm
import csv

# self.model.grid.move_agent(self, new_position)
import threading
import os

pause = False


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def rationRewards(currentRw, newRw):
    """Usefull function to give a normalized output"""
    total = abs(newRw) + abs(currentRw)
    if total == 0:
        return 0
    else:
        if total < 0.0001:
            print("total reward very small", total)
        diff = newRw - currentRw
        return diff / total


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
        self.visibility = np.float64(0.0)
        self.current_local_merit = np.float64(0.0)

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
        )  # Manhattan distance on a torus    def incrPrestige(self, value):

    def increase_prestige(self, value):
        self.prestige += value
        self.prestige_visibility += np.mean(
            [
                value ** (a.curiosity) * self.visibility ** (1 - a.curiosity)
                for a in self.model.scientists
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

    def compute_all_rewards(self, pos, novelty):
        """This part is just a test, we change the network reward by just the avg distance of
        all agents divided the average distance of the current agent"""
        if not self.model.use_distance:
            return self.curiosity * novelty / self.model.avgcurrentAgentKnowledge, 0
        avg_agent_distance = np.mean(
            [a.compute_distance(pos) for a in self.model.scientists if a != self]
        )
        avg_all_distance = self.model.agent_avg_distance
        visibility = (avg_all_distance + 1) / (avg_agent_distance + 1)
        if sum([a.prestige_vanishing for a in self.model.scientists if a != self]) == 0:
            avg_agent_distance_p = avg_agent_distance
        else:
            avg_agent_distance_p = np.average(
                [a.compute_distance(pos) for a in self.model.scientists if a != self],
                weights=[
                    a.prestige_vanishing for a in self.model.scientists if a != self
                ],
            )
        visibility_p = (avg_all_distance + 1) / (avg_agent_distance_p + 1)
        # differentiate between visibility of a field vs. visibility of an agent? (would not this formulation stay constant across neihbouring cells?; answer: no, it won't)
        if not self.model.use_visibility_reward:
            reward = (novelty / self.model.avgcurrentAgentKnowledge) ** (
                self.curiosity
            ) * visibility ** (1 - self.curiosity)
        else:
            reward = (novelty / self.model.avgcurrentAgentKnowledge) ** (
                self.curiosity
            ) * (visibility_p) ** (1 - self.curiosity)

        # currently we use only the above. however, we could employ three options:
        # (1) the above, where I am dragged towards others,
        # (2) an alternative version, where closeness of others impacts on the attractiveness of grids,
        # (3) some version of either (1) or (2) where the pull effect is weighted by prestige
        # else:
        #    reward = self.curiosity * Novelty/ self.model.avgcurrentAgentKnowledge + (1-self.curiosity) * visibility
        return reward, visibility

    def local_merit(self, pos) -> np.float64:
        neighbors = self.model.grid.get_neighborhood(
            pos, moore=False, include_center=False
        )

        # positions: center + 4-neighbors
        positions = [pos, *neighbors]

        # compute into a NumPy array (float64)
        knowledge = np.fromiter(
            (self.model.compute_knowledge(p) for p in positions),
            dtype=np.float64,
            count=len(positions),
        )

        return knowledge.mean()

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
            noise = 2 * (self.model.rng.random() - 0.5)
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

        self.model.farm(self.pos)  # farming also updates AvgAgentKnowledge

        self.age += 1
        self.last_tile_knowledge = self.model.grid.properties["knowledge"].data[
            self.pos
        ]
        self.increase_prestige(self.last_tile_knowledge)


class MyModel(mesa.Model):
    number_connection: int
    number_agents: int
    agent_generation_rate: np.float64
    new_questions: bool
    size: int
    steps: int
    harvest: np.float64
    initial_curiosity: np.float64
    epsilon: np.float64
    use_distance: bool
    scientists: list[Scientist]

    def __init__(
        self,
        n_agents,
        n_connection,
        initial_curiosity,
        epsilon,
        harvest,
        sizeGrid,
        initCellFunc,
        use_distance,
        generation_params={"seed": 0},
        agent_generation_rate=np.float64(-1.0),
        new_questions=False,
        agent_seed=0,
        step_limit=400,
        AgentGenerationFunc=lambda cur, rng, params: cur,
        vanishing_factor_for_prestige=0,
        use_visibility_reward=True,
    ):
        """parameters :  number of agents, number of connections, curiosity, noise intensity, harvest,  size, initCellfunc"""
        super().__init__()
        self.number_connection = n_connection
        self.number_agents = n_agents
        self.agent_generation_rate = agent_generation_rate
        self.new_questions = new_questions
        self.size = sizeGrid
        self.steps = 0
        self.harvest = harvest
        self.initial_curiosity = initial_curiosity
        self.epsilon = epsilon
        self.use_distance = use_distance
        self.grid = mesa.space.MultiGrid(sizeGrid, sizeGrid, torus=True)
        self.grid.add_property_layer(
            PropertyLayer(
                "knowledge", sizeGrid, sizeGrid, np.float64(0.0), dtype=np.float64
            )
        )
        self.grid.add_property_layer(
            PropertyLayer(
                "initial_knowledge",
                sizeGrid,
                sizeGrid,
                np.float64(0.0),
                dtype=np.float64,
            )
        )
        self.grid.add_property_layer(
            PropertyLayer("explored", sizeGrid, sizeGrid, False, dtype=bool)
        )
        self.totalInitialKnowledge = 0
        self.step_limit = step_limit
        self.vanishing_factor_for_prestige = vanishing_factor_for_prestige
        self.use_visibility_reward = use_visibility_reward

        self.avgcurrentAgentKnowledge = 0
        self.agent_avg_distance = 0
        self.generation_params = generation_params | {
            "initCellFunc": initCellFunc.__name__
        }
        self.rng = np.random.default_rng(agent_seed)
        self.explored_weighted_by_initial_knowledge = 0
        self.percentage_knowledge_harvested = 0
        self.explored_percentage = 0
        self._default_steps_thresholds = step_limit * (1.25)
        self.explored_50_step = self._default_steps_thresholds
        self.explored_90_step = self._default_steps_thresholds
        self.weighted_50_step = self._default_steps_thresholds
        self.weighted_90_step = self._default_steps_thresholds
        self.harvested_50_step = self._default_steps_thresholds
        self.harvested_90_step = self._default_steps_thresholds

        if not (
            self.use_distance
            or initial_curiosity == 0
            or epsilon == 0
            or use_visibility_reward
        ):
            print(
                "Agents will NOT use distance information when choosing where to go, curiosity and noise will have similar effects"
            )

        self._seed = agent_seed

        for posX, posY in list(itertools.product(range(sizeGrid), range(sizeGrid))):
            initial_value = initCellFunc(posX, posY, sizeGrid, generation_params)
            self.grid.properties["knowledge"].data[posX, posY] = initial_value
            self.totalInitialKnowledge += initial_value
        min_knowledge, max_knowledge = (
            np.min(self.grid.properties["knowledge"].data),
            np.max(self.grid.properties["knowledge"].data),
        )
        for posX, posY in list(itertools.product(range(sizeGrid), range(sizeGrid))):
            val = self.grid.properties["knowledge"].data[posX, posY]
            if not self.new_questions:
                self.grid.properties["knowledge"].data[posX, posY] = (
                    val / self.totalInitialKnowledge
                )
            else:
                self.grid.properties["knowledge"].data[posX, posY] = (
                    val - min_knowledge
                ) / (max_knowledge - min_knowledge) * 0.95 + 0.05

            self.grid.properties["initial_knowledge"].data[posX, posY] = (
                self.grid.properties["knowledge"].data[posX, posY]
            )
            self.grid.properties["explored"].data[posX, posY] = False
        self.totalInitialKnowledge = 1

        for _ in range(n_agents):
            w = AgentGenerationFunc(initial_curiosity, self.rng, generation_params)
            a = Scientist(self, w, epsilon)
            coords = (self.rng.integers(0, sizeGrid), self.rng.integers(0, sizeGrid))
            a.age = self.rng.choice(np.arange(23, 50))
            self.grid.place_agent(a, coords)
            a.last_tile_knowledge = self.grid.properties["knowledge"].data[coords]
            a.current_local_merit = a.local_merit(a.pos)

        self.scientists = [
            agent for agent in self.agents if isinstance(agent, Scientist)
        ]
        self.updateknowledge()
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
                "avgcurrentAgentKnowledge": lambda m: m.avgcurrentAgentKnowledge,
                "explored_percentage": lambda m: m.explored_percentage,
                "explored_weighted_by_initial_knowledge": lambda m: m.explored_weighted_by_initial_knowledge,
                "total_initial_knowledge": lambda m: m.totalInitialKnowledge,
                "avg_knowledge_on_grid": lambda m: np.mean(
                    m.grid.properties["knowledge"].data
                ),
                "best_knowledge": lambda m: np.max(m.grid.properties["knowledge"].data),
                "avg_distance_between_agents": lambda m: m.agent_avg_distance,
                "percentage_knowledge_harvested": lambda m: m.percentage_knowledge_harvested,
                "corr_prestige_localMerit": lambda m: (
                    np.corrcoef(
                        [a.prestige for a in m.agents],
                        [a.current_local_merit for a in m.agents],
                    )[0, 1]
                    if np.std([a.prestige for a in m.agents]) > 0
                    and np.std([a.current_local_merit for a in m.agents]) > 0
                    else 0
                ),
            },
            agent_reporters={
                "prestige": "prestige",
                "prestige_vanishing": "prestige_vanishing",
                "prestige_visibility": "prestige_visibility",
                "last_tile_knowledge": "last_tile_knowledge",
                "curiosity": "curiosity",
                "localMerit": "current_local_merit",
            },
        )
        self.end_loop_update()

    def updateknowledge(self):
        self.avgcurrentAgentKnowledge = np.mean(
            [agent.last_tile_knowledge for agent in self.scientists]
        )

    def new_place(self, agent1, coords, newAgent=False):
        """function used to modify an agent location and update the surrounding agent's distance list"""
        if not (newAgent):
            self.grid.remove_agent(agent1)
        self.grid.place_agent(agent1, coords)
        self.grid.properties["explored"].data[coords] = True

    def compute_knowledge(self, pos: Coordinate) -> np.float64:
        posX, posY = pos
        value = self.grid.properties["knowledge"].data[posX, posY]
        return value

    def farm(self, pos):
        self.grid.properties["knowledge"].data[pos] = self.grid.properties[
            "knowledge"
        ].data[pos] * (1 - self.harvest)
        self.updateknowledge()

    def __generate_new_agents(self):
        if self.agent_generation_rate > 0:
            agents_to_generate: int = np.floor(
                self.steps * self.agent_generation_rate
            ) - np.floor(self.steps * self.agent_generation_rate)
            prestige_agent_tuples = [(a.prestige, a) for a in self.scientists]
            totalPrestige = sum([a.prestige for a in self.scientists])
            supervisors = []
            for _ in range(agents_to_generate):
                val = self.rng.uniform(0, totalPrestige)
                for p, a in prestige_agent_tuples:
                    if val <= p:
                        supervisors.append(a)
                        break
                    val -= p
            for sup in supervisors:
                a = Scientist(self, self.initial_curiosity, self.epsilon)
                coords = sup.pos
                self.grid.place_agent(a, coords)
                self.scientists.append(a)
            for agent in self.scientists:
                self.grid.properties["explored"].data[agent.pos] = True

    def __generate_new_questions(self):
        if self.new_questions:
            occupied_pos = list(set([agent.pos for agent in self.agents]))
            for posX, posY in occupied_pos:
                if self.grid.properties["knowledge"].data[posX, posY] < 0.025:
                    self.grid.properties["knowledge"].data[posX, posY] = (
                        self.rng.random()
                    )

    def __update_step_goals(self) -> None:
        sentinel = self._default_steps_thresholds
        steps = self.steps

        milestones = (
            ("explored_50_step", "explored_percentage", 0.5),
            ("explored_90_step", "explored_percentage", 0.9),
            ("weighted_50_step", "explored_weighted_by_initial_knowledge", 0.5),
            ("weighted_90_step", "explored_weighted_by_initial_knowledge", 0.9),
            ("harvested_50_step", "percentage_knowledge_harvested", 0.10),
            ("harvested_90_step", "percentage_knowledge_harvested", 0.25),
        )

        for step_attr, metric_attr, threshold in milestones:
            if (
                getattr(self, step_attr) == sentinel
                and getattr(self, metric_attr) >= threshold
            ):
                setattr(self, step_attr, steps)

    def end_loop_update(self):
        self.__generate_new_agents()
        self.__generate_new_questions()

        self.explored_percentage = np.sum(self.grid.properties["explored"].data) / (
            self.size**2
        )
        self.explored_weighted_by_initial_knowledge = (
            np.sum(
                self.grid.properties["explored"].data
                * (self.grid.properties["initial_knowledge"].data)
            )
            / self.totalInitialKnowledge
        )
        self.agent_avg_distance = np.mean(
            [
                np.mean([a.compute_distance(b.pos) for b in self.scientists if a != b])
                for a in self.scientists
            ]
        )
        self.percentage_knowledge_harvested = (
            self.totalInitialKnowledge - np.sum(self.grid.properties["knowledge"].data)
        ) / self.totalInitialKnowledge

        for agent in self.scientists:
            agent.current_local_merit = agent.local_merit(agent.pos)

        self.datacollector.collect(self)
        self.__update_step_goals()

    def step(self, endupdate=True):
        # compute everything and let agents take the decision
        temp = self.agents.select(lambda a: isinstance(a, Scientist))
        temp.random = self.rng
        temp.shuffle_do("step")
        if endupdate:
            self.end_loop_update()

    def Vizualize_3d(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Make data.
        X = np.arange(0, self.size, 1)
        Y = np.arange(0, self.size, 1)
        X, Y = np.meshgrid(X, Y)
        Z = self.grid.properties["knowledge"].data[X, Y]

        # Plot the surface.
        surf = ax.plot_surface(
            X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False
        )
        plt.show()

    def plot(self):
        epistemicGrid = self.grid.properties["knowledge"].data
        Scientists = [a.pos for a in self.agents]
        fig, ax = plt.subplots(ncols=1, figsize=(15, 5.5))
        im = ax.imshow(epistemicGrid)  # , vmin=0, vmax=1)
        cbar = ax.figure.colorbar(im, ax=ax)
        PosX = [
            k[0] + (k.unique_id / self.number_agents - 0.5) ** 2 for k in Scientists
        ]
        PosY = [
            k[1] + (k.unique_id / self.number_agents - 0.5) ** 2 for k in Scientists
        ]
        ax.scatter(PosY, PosX, color="r", alpha=0.5)
        plt.show()

    def compute_distanceToAgents(self):
        Distances = np.zeros((self.size, self.size))
        for X in range(self.size):
            for Y in range(self.size):
                dist = 0
                for agent2 in self.agents:
                    dist += agent2.compute_distance((X, Y))
                Distances[X, Y] = dist
        return Distances

    def animate_steps(self, dynamic_plot=True, csv_name="data", end_report_file=""):
        epistemicGrid = self.grid.properties["knowledge"].data
        PosX = [
            k.pos[0] + (k.unique_id / self.number_agents - 0.5) ** 2
            for k in self.agents
        ]
        PosY = [
            k.pos[1] + (k.unique_id / self.number_agents - 0.5) ** 2
            for k in self.agents
        ]
        PosAge = [k.prestige for k in self.agents]
        Rounds = [self.steps]
        cbar = None

        if dynamic_plot:
            fig, (ax, ax3, ax2) = plt.subplots(ncols=3, figsize=(10, 6))

            def onClick(event):
                global pause
                pause ^= True

            fig.canvas.mpl_connect("button_press_event", onClick)
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", "5%", "5%")
            div2 = make_axes_locatable(ax3)
            cax2 = div2.append_axes("right", "5%", "5%")
            if self.new_questions:
                im = ax.imshow(epistemicGrid, vmin=0, vmax=1)
            else:
                im = ax.imshow(epistemicGrid)  # , vmin=0, vmax=1)
            im = ax3.imshow(
                self.compute_distanceToAgents()
            )  # , vmin=0, vmax=2*self.size)
            ax3.set_title("Distance to all agents")
            ax2.set_title("Measures over time")
            ax.set_title("Epistemic Landscape")
            cbar = ax.figure.colorbar(im, cax=cax)
            scat = ax.scatter(
                PosY, PosX, c=PosAge, alpha=0.9, cmap="OrRd", edgecolors="k"
            )

        # measures
        explorelist = [[0 for k in range(self.size)] for j in range(self.size)]
        listForAvg = []
        for agent in self.agents:
            Xx, Yy = agent.pos
            explorelist[Xx][Yy] = 1
            listForAvg.append(self.grid.properties["knowledge"].data[Xx, Yy])

        explorePercentage = sum([sum(k) for k in explorelist]) / (self.size**2)
        bestKnowledge = np.max(self.grid.properties["knowledge"].data)
        avgMap = np.mean(self.grid.properties["knowledge"].data)
        avgAgent = self.avgcurrentAgentKnowledge
        Val1 = [explorePercentage]
        Val2 = [avgMap / bestKnowledge]
        Val3 = [avgAgent / bestKnowledge]

        def update(frame, cbar=cbar):
            if not (pause):
                self.step(True)
                # measure
                Rounds.append(self.steps)
                listForAvg = []
                for agent in self.agents:
                    Xx, Yy = agent.pos
                    explorelist[Xx][Yy] = 1
                    listForAvg.append(self.grid.properties["knowledge"].data[Xx, Yy])
                explorePercentage = self.datacollector.get_model_vars_dataframe().iloc[
                    -1
                ]["explored_percentage"]
                bestKnowledge = np.max(self.grid.properties["knowledge"].data)
                avgMap = np.mean(self.grid.properties["knowledge"].data)
                avgAgent = self.avgcurrentAgentKnowledge
                Val1.append(explorePercentage)
                Val2.append(avgMap / bestKnowledge)
                Val3.append(avgAgent / bestKnowledge)
                if dynamic_plot:
                    cax.cla()
                    cax2.cla()
                    ax2.cla()
                    ax2.plot(Rounds, Val1, label="exploration percentage")
                    ax2.plot(Rounds, Val2, label="Avg Map / Best Tile")
                    ax2.plot(Rounds, Val3, label="Avg Agent / Best Tile")
                    ax2.legend()
                    PosX = [
                        k.pos[0] + (k.unique_id / self.number_agents - 0.5) ** 2
                        for k in self.agents
                    ]
                    PosY = [
                        k.pos[1] + (k.unique_id / self.number_agents - 0.5) ** 2
                        for k in self.agents
                    ]
                    PosAge = [k.curiosity for k in self.agents]
                    data = np.stack([PosY, PosX]).T
                    scat.set_offsets(data)
                    scat.set_array(np.array(PosAge))
                    scat.set_clim(vmin=min(PosAge), vmax=max(PosAge))

                    epistemicGrid = self.grid.properties["knowledge"].data

                    im = ax.imshow(epistemicGrid)
                    cbar = ax.figure.colorbar(im, cax=cax)
                    im = ax3.imshow(self.compute_distanceToAgents())
                    cbar2 = ax3.figure.colorbar(im, cax=cax2)

                if self.step_limit == frame + 1 and dynamic_plot:
                    threading.Timer(
                        0.1, lambda: plt.close(fig)
                    ).start()  # 👈 Delayed close

        if dynamic_plot:
            ani = animation.FuncAnimation(
                fig=fig, func=update, frames=range(self.step_limit), interval=200
            )
            plt.show()
        else:
            for k in range(self.step_limit):
                update(k)

        if csv_name != "":
            self.datacollector.get_agent_vars_dataframe().to_csv(
                "data/agent_" + csv_name + ".csv"
            )
            self.datacollector.get_model_vars_dataframe().to_csv(
                "data/model_" + csv_name + ".csv"
            )
            print(
                "data saved to ",
                "agent_" + csv_name + ".csv",
                "model_" + csv_name + ".csv",
            )
        if end_report_file != "":
            a = self.datacollector.get_model_vars_dataframe().iloc[-1].to_dict()
            b = {
                k: self.__getattribute__(k)
                for k in self.__dict__
                if (
                    type(self.__getattribute__(k)) in [int, float, str, list, dict]
                    and k[0] != "_"
                )
            }
            c = {
                "mean_corr_prestige_localMerit": self.datacollector.get_model_vars_dataframe()[
                    "corr_prestige_localMerit"
                ].mean()
            }
            row = {**a, **b, **c}
            needs_header = not (is_non_zero_file("data/" + end_report_file))
            with open("data/" + end_report_file, "a") as f:
                writer = csv.writer(f)
                if needs_header:
                    writer.writerow(sorted(row.keys()))
                row = [row[k] for k in sorted(row.keys())]
                writer.writerow(row)
            print("end report saved to ", end_report_file)
        del self

    def run_mode_for_bulk(self, longitudinal=False):
        for k in range(self.step_limit):
            self.step(True)

        if longitudinal == False:
            a = self.datacollector.get_model_vars_dataframe().iloc[-1].to_dict()
            b = {
                k: self.__getattribute__(k)
                for k in self.__dict__
                if (
                    type(self.__getattribute__(k)) in [int, float, str, list, dict]
                    and k[0] != "_"
                )
            }
            c = {
                "mean_corr_prestige_localMerit": self.datacollector.get_model_vars_dataframe()[
                    "corr_prestige_localMerit"
                ].mean()
            }
            row = {**a, **b, **c}
            return row
        else:
            return self.datacollector.get_model_vars_dataframe()


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
    # Loop through all combinations of parameters

    param_range = range(len(param_combinations))
    if skip_to > 0:
        param_range = range(skip_to - 1, skip_to)
        print("skipping to", skip_to, "out of", len(param_combinations))

    if not (longitudinal):
        print(filename + ".csv", intention)
        needs_header = not (is_non_zero_file(filename))
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
