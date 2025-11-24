from mesa.space import PropertyLayer
import mesa
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import csv
# self.model.grid.move_agent(self, new_position)
import threading
import pandas as pd
from pandas import DataFrame
import os
from copy import deepcopy
from itertools import product

pause = False


def is_non_zero_file(fpath):  
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

def rationRewards(currentRw,newRw):
    '''Usefull function to give a normalized output'''
    total = (abs(newRw) + abs(currentRw))
    if total == 0:
        return 0
    else:
        if total < 0.0001:
            print("total reward very small", total)
        diff = (newRw - currentRw)
        return diff/total


class Scientist(mesa.Agent):
    def __init__(self, model, curiosity,epsilon):
        super().__init__(model)
        self.curiosity = curiosity
        self.epsilon = epsilon
        self.prestige = 0
        self.prestige_visibility = 0
        self.prestige_vanishing = 0
        self.age = 23
        self.lastTileKnowledge = 0
        self.visibility = 0
        self.model = model


    def computeDistance(self,otherAgentPos):
        X1,Y1 = self.pos
        X2,Y2 = otherAgentPos
        Size = self.model.size
        dX1 = abs((X2 - X1)%Size)
        dX2 = abs((X1 - X2)%Size)
        dX = min(dX1,dX2)
        dY1 = abs((Y2 - Y1)%Size)
        dY2 = abs((Y1 - Y2)%Size)
        dY = min(dY1,dY2)
        return (dX + dY)
    
    def incrPrestige(self, value):
        self.prestige += value
        self.prestige_visibility += value**(1-self.model.visibility_factor_for_prestige) * self.visibility **(self.model.visibility_factor_for_prestige)
        x = self.model.vanishing_factor_for_prestige
        if x != 1:
            temp = self.prestige_vanishing 
            temp = temp *x #vanishes past values
            temp += value  # adds the new one
            self.prestige_vanishing = temp
        else:
            self.prestige_vanishing += value

 
    def computeAllrewards2(self, pos, Novelty):
        """ This part is just a test, we change the network reward by just the avg distance of
         all agents divided the average distance of the current agent"""
        if self.model.use_distance == False:
            return self.curiosity *Novelty/ self.model.avgcurrentAgentKnowledge
        avgAgentDistance = np.mean([a.computeDistance(pos) for a in self.model.agents if a != self])
        avgAllDistance = self.model.agent_avg_distance

        return self.curiosity * Novelty/ self.model.avgcurrentAgentKnowledge + (1-self.curiosity) * (avgAllDistance+1)/(avgAgentDistance+1)

    def step(self):
        '''pick a random node and check if according to the agent preference it is better to move to that node'''
        neighbors_nodes = self.model.grid.get_neighborhood(self.pos, moore = False, include_center=False)
        optionpos = self.model.rng.choice(neighbors_nodes)

        currentRwNovelty = self.model.computeRewardKnowledge(self.pos)
        newRwNovelty = self.model.computeRewardKnowledge(optionpos)

        noise = 2*(self.model.rng.random()-0.5 ) 
        totCurrentReward = self.computeAllrewards2(self.pos,currentRwNovelty) + self.epsilon*noise

        noise = 2*(self.model.rng.random()-0.5 )
        totNewReward = self.computeAllrewards2(optionpos,newRwNovelty) + self.model.error_imbalance *self.epsilon*noise

        if totNewReward - totCurrentReward > 0 :
            self.model.new_place(self, optionpos)
            
        self.model.Farm(self.pos) #farming also updates AvgAgentKnowledge

        self.age += 1
        self.lastTileKnowledge = self.model.grid.properties["knowledge"].data[self.pos]
        
        self.incrPrestige(self.lastTileKnowledge) 
        print("did update prestige", self.prestige_vanishing, self.prestige, "value was", self.lastTileKnowledge)



class MyModel(mesa.Model):
    def __init__(self, n_agents, n_connection, initial_curiosity, epsilon, harvest, sizeGrid, initCellFunc, use_distance, generation_params = {"seed" :0}, agent_generation_rate = -1, constant_population = 1, agent_seed = 0,step_limit = 400, AgentGenerationFunc = lambda cur,seed: cur, visibility_factor_for_prestige = 0, vanishing_factor_for_prestige = 0):
        ''' parameters :  number of agents, number of connections, curiosity, noise intensity, harvest,  size, initCellfunc '''
        super().__init__()
        self.number_connection = n_connection
        self.number_agents = n_agents
        self.agent_generation = agent_generation_rate
        self.constant_population = constant_population
        self.size = sizeGrid
        self.steps = 0
        self.harvest = harvest
        self.initial_curiosity = initial_curiosity
        self.epsilon = epsilon
        self.use_distance = use_distance
        self.grid = mesa.space.MultiGrid(sizeGrid, sizeGrid, torus=True)
        self.grid.add_property_layer(PropertyLayer("knowledge",  sizeGrid,sizeGrid,0.0, dtype=float) )
        self.grid.add_property_layer(PropertyLayer("initial_knowledge",  sizeGrid,sizeGrid,0.0, dtype=float) )
        self.grid.add_property_layer(PropertyLayer("explored",  sizeGrid,sizeGrid,False, dtype=bool) )
        self.totalInitialKnowledge = 0
        self.error_imbalance = 5
        self.step_limit = step_limit
        self.visibility_factor_for_prestige = visibility_factor_for_prestige
        self.vanishing_factor_for_prestige = vanishing_factor_for_prestige

        self.avgcurrentAgentKnowledge = 0  
        self.agent_avg_distance = 0
        self.generation_params = generation_params | {"initCellFunc": initCellFunc.__name__}
        self.rng = random.Random(agent_seed)
        
        self.explored_weighted_by_initial_knowledge = 0
        self.percentage_knowledge_harvested = 0
        self.explored_percentage = 0
        # stored_percentage_names = ["explored_percentage", "explored_weighted_by_initial_knowledge","percentage_knowledge_harvested"]
        # stored_percentage = [[0.5,-1],[0.9,-1]]
        # self.stored_percentage = [(k, deepcopy(stored_percentage)) for k in stored_percentage_names]
        self.explored_50_step = -1
        self.explored_90_step = -1
        self.weighted_50_step = -1
        self.weighted_90_step = -1
        self.harvested_50_step = -1
        self.harvested_90_step = -1

        if self.use_distance == False:
            print("Agents will NOT use distance information when choosing where to go, curiosity and noise will have similar effects")

        self._seed = agent_seed
        for posX,posY in list(itertools.product(range(sizeGrid), range(sizeGrid))):
            initial_value = initCellFunc(posX,posY, sizeGrid, generation_params)
            self.grid.properties["knowledge"].data[posX,posY] = initial_value
            self.totalInitialKnowledge += initial_value

        for posX,posY in list(itertools.product(range(sizeGrid), range(sizeGrid))):
            self.grid.properties["knowledge"].data[posX,posY] = self.grid.properties["knowledge"].data[posX,posY]/self.totalInitialKnowledge
            self.grid.properties["initial_knowledge"].data[posX,posY] = self.grid.properties["knowledge"].data[posX,posY]
            self.grid.properties["explored"].data[posX,posY] = False
        self.totalInitialKnowledge = 1
            
        

        for _ in range(n_agents):
            w = AgentGenerationFunc(initial_curiosity,self._seed + _, generation_params)
            a = Scientist(self, w,epsilon)
            coords = (self.rng.randrange(0, sizeGrid), self.rng.randrange(0, sizeGrid))
            a.age = self.rng.randint(23,50)
            self.grid.place_agent(a, coords)
            a.lastTileKnowledge = self.grid.properties["knowledge"].data[coords]

        
        self.updateknowledge()
        self.datacollector = mesa.DataCollector(
            model_reporters={"Step": lambda m: m.steps,
                             "mean_age": lambda m: m.agents.agg("age", np.mean),
                             "mean_prestige": lambda m: m.agents.agg("prestige", np.mean),
                             "mean_vanishing_prestige": lambda m:  m.agents.agg("prestige_vanishing", np.mean),
                             "mean_visibility_prestige": lambda m:  m.agents.agg("prestige_visibility", np.mean),
                             "avgcurrentAgentKnowledge": lambda m: m.avgcurrentAgentKnowledge,
                             "explored_percentage": lambda m: m.explored_percentage,
                             "explored_weighted_by_initial_knowledge": lambda m: m.explored_weighted_by_initial_knowledge,
                             "total_initial_knowledge": lambda m: m.totalInitialKnowledge,
                             "avg_knowledge_on_grid": lambda m: np.mean(m.grid.properties["knowledge"].data),
                             "best_knowledge": lambda m: np.max(m.grid.properties["knowledge"].data),
                             "avg_distance_between_agents": lambda m: m.agent_avg_distance,
                             "percentage_knowledge_harvested": lambda m: m.percentage_knowledge_harvested,
                             },
            agent_reporters={"prestige":   "prestige",
                             "lastTileKnowledge": "lastTileKnowledge",
                             'curiosity': "curiosity"
                             })
        self.endLoopUpdate()

    def updateknowledge(self):
        self.avgcurrentAgentKnowledge = np.mean([agent.lastTileKnowledge for agent in self.agents])


            

    def new_place(self, agent1,coords, newAgent = False):
        '''function used to modify an agent location and update the surrounding agent's distance list'''
        if not(newAgent): 
            self.grid.remove_agent(agent1)
        self.grid.place_agent(agent1, coords)
        self.grid.properties["explored"].data[coords] = True

        

        for agent2 in self.agents:
            if agent2 != agent1:
                dist = agent1.computeDistance(agent2.pos)

    def computeRewardKnowledge(self,pos):
        posX,posY = pos
        return (self.grid.properties["knowledge"].data[posX,posY])
    
    def Farm(self, pos):
        self.grid.properties["knowledge"].data[pos] = self.grid.properties["knowledge"].data[pos] * (1- self.harvest)
        self.updateknowledge()
    
    def endLoopUpdate(self):
        if self.agent_generation > 0:
            prevNew = int((self.steps-1) // self.agent_generation)
            newNew = int(self.steps // self.agent_generation)
            ListPA  = [(a.prestige, a) for a in self.agents]
            totalPrestige = sum([a.prestige for a in self.agents])
            supervisors = []
            for k in range(prevNew, newNew):
                val = self.rng.uniform(0, totalPrestige)
                for p, a in ListPA:
                    if val <= p:
                        supervisors.append(a)
                        break
                    val -= p

            for sup in supervisors:    
                a = Scientist(self, self.initial_curiosity,self.epsilon)
                coords = sup.pos
                self.grid.place_agent(a, coords)
            for agent in self.agents:
                self.grid.properties["explored"].data[agent.pos] = True
        self.explored_percentage = np.sum(self.grid.properties["explored"].data)/(self.size**2)
        self.explored_weighted_by_initial_knowledge = np.sum(self.grid.properties["explored"].data * (self.grid.properties["initial_knowledge"].data))/self.totalInitialKnowledge
        self.agent_avg_distance = np.mean([ np.mean([a.computeDistance(b.pos) for b in self.agents if a != b]) for a in self.agents])
        self.percentage_knowledge_harvested = (self.totalInitialKnowledge - np.sum(self.grid.properties["knowledge"].data))/self.totalInitialKnowledge

        self.datacollector.collect(self)
        if self.explored_50_step == -1 and self.explored_percentage >= 0.5: 
            self.explored_50_step = self.steps
        if self.explored_90_step == -1 and self.explored_percentage >= 0.9: 
            self.explored_90_step = self.steps
        if self.weighted_50_step == -1 and self.explored_weighted_by_initial_knowledge >= 0.5: 
            self.weighted_50_step = self.steps
        if self.weighted_90_step == -1 and self.explored_weighted_by_initial_knowledge >= 0.9: 
            self.weighted_90_step = self.steps
        if self.harvested_50_step == -1 and self.percentage_knowledge_harvested >= 0.1: 
            self.harvested_50_step = self.steps
        if self.harvested_90_step == -1 and self.percentage_knowledge_harvested >= 0.25: 
            self.harvested_90_step = self.steps

        #for (name,lst_per) in self.stored_percentage:
        #    value = self.__getattribute__(name)
        #    for per in lst_per:
        #        if per[1] == -1 and value >= per[0]:
        #            per[1] = self.steps
            

    def step(self, endupdate = True):
        # compute everything and let agents take the decision 
        temp = self.agents.select(lambda a: type(a) == Scientist)
        temp.random = self.rng
        temp.shuffle_do("step")
        if endupdate:
            self.endLoopUpdate()

                
    def Vizualize_3d(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Make data.
        X = np.arange(0, self.size, 1)
        Y = np.arange(0, self.size, 1)
        X, Y = np.meshgrid(X, Y)
        Z = self.grid.properties["knowledge"].data[X,Y]

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        plt.show()

    def plot(self):
        epistemicGrid = self.grid.properties["knowledge"].data
        Scientists = [a.pos for a in self.agents]
        fig, ax = plt.subplots(ncols = 1, figsize=(15, 5.5))
        im = ax.imshow(epistemicGrid)#, vmin=0, vmax=1)
        cbar = ax.figure.colorbar(im, ax=ax)
        PosX = [k[0]+(k.unique_id /self.number_agents-0.5)**2 for k in Scientists]
        PosY = [k[1]+(k.unique_id/self.number_agents-0.5)**2 for k in Scientists]
        ax.scatter(PosY, PosX, color = 'r', alpha = 0.5)
        plt.show()
    
    def computeDistanceToAgents(self):
        Distances = np.zeros((self.size,self.size))
        for X in range(self.size):
            for Y in range(self.size):
                dist = 0
                for agent2 in self.agents:
                    dist += agent2.computeDistance((X,Y))
                Distances[X,Y] = dist
        return Distances

    def animate_steps(self, dynamic_plot = True, csv_name= "data", end_report_file = ""):


        epistemicGrid = self.grid.properties["knowledge"].data
        PosX = [k.pos[0]+(k.unique_id/self.number_agents-0.5)**2 for k in self.agents]
        PosY = [k.pos[1]+(k.unique_id/self.number_agents-0.5)**2 for k in self.agents]
        PosAge = [k.prestige for k in self.agents]
        Rounds = [self.steps]
        cbar = None

        if dynamic_plot:
            fig, (ax,ax3,ax2) = plt.subplots(ncols = 3, figsize=(10, 6))
            def onClick(event):
                global pause
                pause ^= True
            fig.canvas.mpl_connect('button_press_event', onClick)
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', '5%', '5%')
            div2 = make_axes_locatable(ax3)
            cax2 = div2.append_axes('right', '5%', '5%')
            im = ax.imshow(epistemicGrid)#, vmin=0, vmax=1)
            im= ax3.imshow(self.computeDistanceToAgents())#, vmin=0, vmax=2*self.size)
            ax3.set_title('Distance to all agents')
            ax2.set_title('Measures over time')
            ax.set_title('Epistemic Landscape')
            cbar = ax.figure.colorbar(im, cax=cax)
            scat = ax.scatter(PosY, PosX, c = PosAge, alpha = 0.9, cmap = 'OrRd',edgecolors="k")

        #measures
        explorelist = [[0 for k in range(self.size)] for j in range(self.size)]
        listForAvg = []
        for agent in self.agents:
            Xx, Yy  = agent.pos
            explorelist[Xx][Yy] = 1
            listForAvg.append(self.grid.properties["knowledge"].data[Xx,Yy])
        
        explorePercentage = sum([ sum(k)for k in explorelist])/(self.size**2)
        bestKnowledge = np.max(self.grid.properties["knowledge"].data)
        avgMap = np.mean(self.grid.properties["knowledge"].data)
        avgAgent = self.avgcurrentAgentKnowledge
        Val1 = [explorePercentage]
        Val2 = [avgMap/bestKnowledge]
        Val3 = [avgAgent/bestKnowledge]

        def update(frame, cbar = cbar):

            if not(pause):
                self.step(True)
                #measure 
                Rounds.append(self.steps)
                listForAvg = []
                for agent in self.agents:
                    Xx, Yy  = agent.pos
                    explorelist[Xx][Yy] = 1
                    listForAvg.append(self.grid.properties["knowledge"].data[Xx,Yy])
                explorePercentage = self.datacollector.get_model_vars_dataframe().iloc[-1]["explored_percentage"]
                bestKnowledge = np.max(self.grid.properties["knowledge"].data)
                avgMap = np.mean(self.grid.properties["knowledge"].data)
                avgAgent = self.avgcurrentAgentKnowledge
                Val1.append(explorePercentage)
                Val2.append(avgMap/bestKnowledge)
                Val3.append(avgAgent/bestKnowledge)
                if dynamic_plot:
                    cax.cla()
                    cax2.cla()
                    ax2.cla()
                    ax2.plot(Rounds, Val1, label = 'exploration percentage')
                    ax2.plot(Rounds, Val2, label = 'Avg Map / Best Tile')
                    ax2.plot(Rounds, Val3, label = 'Avg Agent / Best Tile')
                    ax2.legend()
                    PosX = [k.pos[0]+(k.unique_id/self.number_agents-0.5)**2 for k in self.agents ]
                    PosY = [k.pos[1]+(k.unique_id/self.number_agents-0.5)**2 for k in self.agents ]
                    PosAge = [k.curiosity for k in self.agents]
                    data = np.stack([PosY,PosX]).T
                    scat.set_offsets(data)
                    scat.set_array(np.array(PosAge))
                    scat.set_clim(vmin=min(PosAge), vmax=max(PosAge))
                    
                    epistemicGrid = self.grid.properties["knowledge"].data
                    im = ax.imshow(epistemicGrid)
                    cbar = ax.figure.colorbar(im, cax=cax)
                    im= ax3.imshow(self.computeDistanceToAgents())
                    cbar2 = ax3.figure.colorbar(im, cax=cax2)

                if self.step_limit == frame+1 and dynamic_plot:
                    threading.Timer(0.1, lambda: plt.close(fig)).start()  # 👈 Delayed close



        if dynamic_plot:
            ani = animation.FuncAnimation(fig=fig, func=update, frames=range(self.step_limit), interval=200)
            plt.show()
        else:
            for k in range(self.step_limit):
                update(k)

        if csv_name != "":
            self.datacollector.get_agent_vars_dataframe().to_csv("agent_"+csv_name+".csv")
            self.datacollector.get_model_vars_dataframe().to_csv("model_"+csv_name+".csv")
        if end_report_file != "":
            a = self.datacollector.get_model_vars_dataframe().iloc[-1].to_dict()
            b = {k: self.__getattribute__(k) for k in self.__dict__ if (type(self.__getattribute__(k)) in [int,float,str,list, dict] and k[0] != '_') }
            row = {**a, **b}
            needs_header = not(is_non_zero_file(end_report_file))
            with open(end_report_file, 'a') as f:
                writer = csv.writer(f)
                if needs_header:
                    writer.writerow(sorted(row.keys()))
                row = [row[k] for k in sorted(row.keys())]
                writer.writerow(row)
        del self

    def run_mode_for_bulk(self):
            for k in range(self.step_limit):
                self.step(True)

            a = self.datacollector.get_model_vars_dataframe().iloc[-1].to_dict()
            b = {k: self.__getattribute__(k) for k in self.__dict__ if (type(self.__getattribute__(k)) in [int,float,str,list, dict] and k[0] != '_') }
            row = {**a, **b}
            return row


def plot_normalized_values(csv_name, size):
    modeldf = pd.read_csv("model_"+csv_name+".csv")
    agentdf = pd.read_csv("agent_"+csv_name+".csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(modeldf['Step'], modeldf['explored_percentage'], label = 'exploration percentage')
    ax.plot(modeldf['Step'], modeldf['avg_knowledge']/modeldf['best_knowledge'], label = 'Avg Map / Best Tile')
    ax.plot(modeldf['Step'], modeldf['avgcurrentAgentKnowledge']/modeldf['best_knowledge'], label = 'Avg Agent / Best Tile')
    ax.plot(modeldf['Step'], modeldf['avg_distance_between_agents']/size*(2)**(1/2), label = 'Avg Distance / max distance')
    ax.legend()
    plt.show()

def plot_tileknowledge_per_prestige():
    agentdf = pd.read_csv("agent_data.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.title("agents tile vs their prestige")
    #normalize lastTileKnowledge by the best knowledge in the model on the same step
    modeldf = pd.read_csv("model_data.csv")
    best_knowledge_per_step = dict(zip(modeldf['Step'], modeldf['best_knowledge']))
    agentdf['lastTileKnowledge_normalized'] = agentdf.apply(lambda row: row['lastTileKnowledge']/best_knowledge_per_step[row['Step']], axis=1)
    ax.scatter(agentdf['prestige'], agentdf['lastTileKnowledge_normalized'], alpha = 0.5, c = agentdf['Step'], cmap = 'viridis')
    cbar = plt.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax)
    cbar.set_ticks([k/10 for k in range(10)])
    max_step = max(agentdf['Step'])
    if max_step > 10:
        cbar.ax.set_yticklabels([k* (max_step//10) for k in range(10)])
    
    cbar.set_label('Step')
    ax.set_xlabel('Prestige')
    ax.set_ylabel('Agent Tile Knowledge/Best Tile Knowledge')
    plt.show()
        

def generate_data_parametric_exploration(filename, param_grid, repeats_per_setting = 10, change_landscape_seed = False, intention = "w", skip_to = 0):
    import itertools
    param_as_lists =  [a for a in param_grid.keys() if type(param_grid[a]) == list]
    varying_params = [a for a in param_as_lists if len(param_grid[a]) > 1]
    # Create a list of all combinations of parameters

    param_combinations = list(itertools.product(*[param_grid[k] if k in param_as_lists else [param_grid[k]] for k in param_grid.keys()]))

    # Create a list of parameter names
    param_names = list(param_grid.keys())
    # Loop through all combinations of parameters

    param_range = range(len(param_combinations))
    if skip_to > 0:
        param_range = range(skip_to-1, skip_to )
        print("skipping to", skip_to, "out of", len(param_combinations))
    print(filename, intention)
    needs_header = not(is_non_zero_file(filename))
    with open(filename, intention) as f:
        writer = csv.writer(f)
        if needs_header or intention == "w":
            #create a dummy model to get the header
            param_set = param_combinations[0]
            all_params = {param_names[i]: param_set[i] for i in range(len(param_names))}
            all_params['step_limit'] = 1
            model = MyModel(**all_params)
            row = model.run_mode_for_bulk()
            writer.writerow(sorted(row.keys()))
            del model
        
        for idx in param_range:
            param_set = param_combinations[idx]
            all_params = {param_names[i]: param_set[i] for i in range(len(param_names))}

            print("now processing param set ", idx+1, "out of", len(param_combinations))
            for repeat in range(repeats_per_setting):
                all_params["agent_seed"] = repeat 
                if change_landscape_seed:
                    all_params["generation_params"] = all_params.get("generation_params", {}) | {"seed": repeat}
                model = MyModel(**all_params)
                row = model.run_mode_for_bulk()
                writer.writerow([row[k] for k in sorted(row.keys())])
                del model