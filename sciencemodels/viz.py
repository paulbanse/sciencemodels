"""Visualization level: landscape plots and the step animation.

Extracted verbatim from the pre-restructure ``mesa_model.py`` (methods
``MyModel.Vizualize_3d``, ``MyModel.plot``, ``MyModel.animate_steps`` and the
module-level ``pause`` toggle). ``self`` became the ``model`` parameter; the
CSV/report-writing tails of ``animate_steps`` now call ``sciencemodels.io``
(the same code, in the io module). ``MyModel`` keeps thin ``plot`` /
``Vizualize_3d`` / ``animate_steps`` delegates so ``model.animate_steps(...)``
works unchanged.
"""
from __future__ import annotations

import threading

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import io

pause = False


def Vizualize_3d(model):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.arange(0, model.size, 1)
    Y = np.arange(0, model.size, 1)
    X, Y = np.meshgrid(X, Y)
    Z = model.grid.properties["knowledge"].data[X, Y]

    # Plot the surface.
    surf = ax.plot_surface(
        X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    plt.show()


def plot(model):
    epistemicGrid = model.grid.properties["knowledge"].data
    Scientists = [a.pos for a in model.agents]
    fig, ax = plt.subplots(ncols=1, figsize=(15, 5.5))
    im = ax.imshow(epistemicGrid)  # , vmin=0, vmax=1)
    cbar = ax.figure.colorbar(im, ax=ax)
    PosX = [
        k[0] + (k.unique_id / model.number_agents - 0.5) ** 2 for k in Scientists
    ]
    PosY = [
        k[1] + (k.unique_id / model.number_agents - 0.5) ** 2 for k in Scientists
    ]
    ax.scatter(PosY, PosX, color="r", alpha=0.5)
    plt.show()


def animate_steps(model, dynamic_plot=True, csv_name="data", end_report_file=""):
    epistemicGrid = model.grid.properties["knowledge"].data
    PosX = [
        k.pos[0] + (k.unique_id / model.number_agents - 0.5) ** 2
        for k in model.agents
    ]
    PosY = [
        k.pos[1] + (k.unique_id / model.number_agents - 0.5) ** 2
        for k in model.agents
    ]
    PosAge = [k.prestige for k in model.agents]
    Rounds = [model.steps]
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
        if model.knowledge_normalization == "minmax":
            im = ax.imshow(epistemicGrid, vmin=0, vmax=1)
        else:
            im = ax.imshow(epistemicGrid)  # , vmin=0, vmax=1)
        im = ax3.imshow(
            model.compute_distanceToAgents()
        )  # , vmin=0, vmax=2*self.size)
        ax3.set_title("Distance to all agents")
        ax2.set_title("Measures over time")
        ax.set_title("Epistemic Landscape")
        cbar = ax.figure.colorbar(im, cax=cax)
        scat = ax.scatter(
            PosY, PosX, c=PosAge, alpha=0.9, cmap="OrRd", edgecolors="k"
        )

    # measures
    explorelist = [[0 for k in range(model.size)] for j in range(model.size)]
    listForAvg = []
    for agent in model.agents:
        Xx, Yy = agent.pos
        explorelist[Xx][Yy] = 1
        listForAvg.append(model.grid.properties["knowledge"].data[Xx, Yy])

    explorePercentage = sum([sum(k) for k in explorelist]) / (model.size**2)
    bestKnowledge = np.max(model.grid.properties["knowledge"].data)
    avgMap = np.mean(model.grid.properties["knowledge"].data)
    avgAgent = model.avg_current_agent_knowledge
    Val1 = [explorePercentage]
    Val2 = [avgMap / bestKnowledge]
    Val3 = [avgAgent / bestKnowledge]

    def update(frame, cbar=cbar):
        if not (pause):
            model.step(True)
            # measure
            Rounds.append(model.steps)
            listForAvg = []
            for agent in model.agents:
                Xx, Yy = agent.pos
                explorelist[Xx][Yy] = 1
                listForAvg.append(model.grid.properties["knowledge"].data[Xx, Yy])
            explorePercentage = model.datacollector.get_model_vars_dataframe().iloc[-1]["explored_percentage"]
            bestKnowledge = np.max(model.grid.properties["knowledge"].data)
            avgMap = np.mean(model.grid.properties["knowledge"].data)
            avgAgent = model.avg_current_agent_knowledge
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
                    k.pos[0] + (k.unique_id / model.number_agents - 0.5) ** 2
                    for k in model.agents
                ]
                PosY = [
                    k.pos[1] + (k.unique_id / model.number_agents - 0.5) ** 2
                    for k in model.agents
                ]
                PosAge = [k.curiosity for k in model.agents]
                data = np.stack([PosY, PosX]).T
                scat.set_offsets(data)
                scat.set_array(np.array(PosAge))
                scat.set_clim(vmin=min(PosAge), vmax=max(PosAge))

                epistemicGrid = model.grid.properties["knowledge"].data

                im = ax.imshow(epistemicGrid)
                cbar = ax.figure.colorbar(im, cax=cax)
                im = ax3.imshow(model.compute_distanceToAgents())
                cbar2 = ax3.figure.colorbar(im, cax=cax2)

            if model.step_limit == frame + 1 and dynamic_plot:
                threading.Timer(
                    0.1, lambda: plt.close(fig)
                ).start()  # 👈 Delayed close

    if dynamic_plot:
        ani = animation.FuncAnimation(
            fig=fig, func=update, frames=range(model.step_limit), interval=200
        )
        plt.show()
    else:
        for k in range(model.step_limit):
            update(k)

    if csv_name != "":
        io.save_run_csvs(model, csv_name)
    if end_report_file != "":
        io.write_end_report(model, end_report_file)
    del model
