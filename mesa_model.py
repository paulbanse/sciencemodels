from mesa.space import PropertyLayer
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
import pandas as pd
import os

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
        self.model = model
        self.visibility = 0
        self.current_localMerit = 0
        self.ideal_Merit= 0


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
        self.prestige_visibility += np.mean([value**(a.curiosity) * self.visibility **(1-a.curiosity) for a in self.model.agents if a != self])
        # idea: we could also use individual distances instead of agent's visibility in the above specification.
        x = self.model.vanishing_factor_for_prestige
        if x != 1:
            temp = self.prestige_vanishing 
            temp = temp *x #vanishes past values
            temp += value  # adds the new one
            self.prestige_vanishing = temp
        else:
            self.prestige_vanishing += value

 
    def computeAllrewards(self, pos, Novelty):
        """ This part is just a test, we change the network reward by just the avg distance of
         all agents divided the average distance of the current agent"""
        if self.model.use_distance == False:
            return self.curiosity *Novelty/ self.model.avgcurrentAgentKnowledge, 0
        avgAgentDistance = np.mean([a.computeDistance(pos) for a in self.model.agents if a != self])
        avgAllDistance = self.model.agent_avg_distance
        visibility = (avgAllDistance+1)/(avgAgentDistance+1)
        if sum([a.prestige_vanishing for a in self.model.agents if a != self]) == 0:
            avgAgentDistanceP = avgAgentDistance
        else:
            avgAgentDistanceP = np.average([a.computeDistance(pos) for a in self.model.agents if a != self],
                                       weights=[a.prestige_vanishing for a in self.model.agents if a != self])
        visibilityP = (avgAllDistance+1)/(avgAgentDistanceP+1)
        # differentiate between visibility of a field vs. visibility of an agent? (would not this formulation stay constant across neihbouring cells?; answer: no, it won't)
        if self.model.use_visibility_reward == False:
            reward = (Novelty/self.model.avgcurrentAgentKnowledge)**(self.curiosity) * visibility**(1-self.curiosity)
        else:
            reward = (Novelty/self.model.avgcurrentAgentKnowledge)**(self.curiosity) * (visibilityP)**(1-self.curiosity)
        # currently we use only the above. however, we could employ three options: (1) the above, where I am dragged towards others, (2) an alternative version, where closeness of others impacts on the attractiveness of grids, (3) some version of either (1) or (2) where the pull effect is weighted by prestige
        # else:
        #    reward = self.curiosity * Novelty/ self.model.avgcurrentAgentKnowledge + (1-self.curiosity) * visibility
        return reward, visibility
    
    def computeLocalMerit(self, pos):
        # Get the four direct neighbors
        neighbors = self.model.grid.get_neighborhood(pos, moore=False, include_center=False)
        # Get knowledge values for all positions (center + neighbors)
        knowledge_values = [self.model.computeKnowledge(pos)]  # center position
        for neighbor_pos in neighbors:
            knowledge_values.append(self.model.computeKnowledge(neighbor_pos))
        # Return the average
        self.current_localMerit = np.mean(knowledge_values)



    
    def step(self):
        '''pick a random node and check if according to the agent preference it is better to move to that node'''
        neighbors_nodes = self.model.grid.get_neighborhood(self.pos, moore = False, include_center=False)
        # optionpos = self.model.rng.choice(neighbors_nodes)
        currentRwNovelty = self.model.computeKnowledge(self.pos)
        totCurrentReward, currentVis = self.computeAllrewards(self.pos, currentRwNovelty)

        best_reward = totCurrentReward
        optionpos = self.pos  # fallback to current position
        best_vis = currentVis

        for neighbor in neighbors_nodes:
            neighborRwNovelty = self.model.computeKnowledge(neighbor)
            noise = 2*(self.model.rng.random()-0.5)
            totReward, vis = self.computeAllrewards(neighbor, neighborRwNovelty*(1 + self.epsilon*noise))
    
            if totReward > best_reward:
                best_reward = totReward
                optionpos = neighbor
                best_vis = vis
        
        if best_reward - totCurrentReward > 0:
            self.model.new_place(self, optionpos)
            self.visibility = best_vis
        else:
            self.visibility = currentVis

        self.model.Farm(self.pos) #farming also updates AvgAgentKnowledge
        
        self.age += 1
        self.lastTileKnowledge = self.model.grid.properties["knowledge"].data[self.pos]
        self.incrPrestige(self.lastTileKnowledge) 



class MyModel(mesa.Model):
    def __init__(self, n_agents, initial_curiosity, epsilon, harvest, sizeGrid, 
                 initCellFunc, use_distance, 
                 generation_params = {"seed" :0}, 
                 agent_generation_time = -1, 
                 new_questions = 0,
                 agent_seed = 0,step_limit = 400,
                 AgentGenerationFunc = lambda cur,seed,params: cur,
                 vanishing_factor_for_prestige = 0,
                 use_visibility_reward = True):
        ''' parameters :  number of agents, number of connections, curiosity, noise intensity, harvest,  size, initCellfunc '''
        super().__init__(seed = agent_seed)
        self.number_agents = n_agents
        self.agent_generation = agent_generation_time
        self.new_questions = new_questions
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
        self.grid.add_property_layer(PropertyLayer("previous_knowledge",  sizeGrid,sizeGrid,0.0, dtype=float) )
        self.totalInitialKnowledge = 0
        self.step_limit = step_limit
        self.vanishing_factor_for_prestige = vanishing_factor_for_prestige
        self.use_visibility_reward = use_visibility_reward

        self.avgcurrentAgentKnowledge = 0
        self.agent_avg_distance = 0
        self.generation_params = generation_params | {"initCellFunc": initCellFunc.__name__}
        
        self.explored_weighted_by_initial_knowledge = 0
        self.percentage_knowledge_harvested = 0
        self.explored_percentage = 0
        self._default_steps_thresholds = step_limit*(1.25)
        self.explored_50_step = self._default_steps_thresholds
        self.explored_90_step = self._default_steps_thresholds
        self.weighted_50_step = self._default_steps_thresholds
        self.weighted_90_step = self._default_steps_thresholds
        self.harvested_50_step = self._default_steps_thresholds
        self.harvested_90_step = self._default_steps_thresholds

        if self.use_distance == False and not(initial_curiosity == 0 or epsilon == 0 or use_visibility_reward ):
            print("Agents will NOT use distance information when choosing where to go, curiosity and noise will have similar effects")

        self._seed = agent_seed
        
        for posX,posY in list(itertools.product(range(sizeGrid), range(sizeGrid))):
            initial_value = initCellFunc(posX,posY, sizeGrid, generation_params)
            self.grid.properties["knowledge"].data[posX,posY] = initial_value
            self.totalInitialKnowledge += initial_value
        min_knowledge, max_knowledge = np.min(self.grid.properties["knowledge"].data), np.max(self.grid.properties["knowledge"].data)
        for posX,posY in list(itertools.product(range(sizeGrid), range(sizeGrid))):
            val = self.grid.properties["knowledge"].data[posX,posY]
            if self.new_questions == 0:
                self.grid.properties["knowledge"].data[posX,posY] = val /self.totalInitialKnowledge
            if self.new_questions > 0:
                self.grid.properties["knowledge"].data[posX,posY] = val /self.totalInitialKnowledge * 0.5 + 1/sizeGrid**2 *0.5
                
            self.grid.properties["initial_knowledge"].data[posX,posY] = self.grid.properties["knowledge"].data[posX,posY]
            self.grid.properties["previous_knowledge"].data[posX,posY] = self.grid.properties["knowledge"].data[posX,posY]
            self.grid.properties["explored"].data[posX,posY] = False
        self.totalInitialKnowledge = 1
            
        
        for _ in range(n_agents):
            w = AgentGenerationFunc(initial_curiosity,self.rng, generation_params)
            a = Scientist(self, w,epsilon)
            coords = (self.rng.integers(0, sizeGrid), self.rng.integers(0, sizeGrid))
            a.age = self.rng.integers(23,50)
            self.grid.place_agent(a, coords)
            a.lastTileKnowledge = self.grid.properties["knowledge"].data[coords]
            a.computeLocalMerit(a.pos)

        
        self.updateAvgknowledge()
        self.updateIdealMerit()
        def correlation_reporter(merit_type, prestige_type):
            def reporter(m):
                Prest = [getattr(a, prestige_type) for a in m.agents]
                Mert = [getattr(a, merit_type) for a in m.agents]
                if  np.std(Prest) > 0 and np.std(Mert) > 0:
                    return np.corrcoef(Prest,Mert)[0, 1]
                else:
                    return 0
            return reporter
         
        self.datacollector = mesa.DataCollector(
            model_reporters={"step": lambda m: m.steps,
                             "mean_age": lambda m: m.agents.agg("age", np.mean),
                             "mean_prestige": lambda m: m.agents.agg("prestige", np.mean),
                             "mean_vanishing_prestige": lambda m:  m.agents.agg("prestige_vanishing", np.mean),
                             "mean_visibility_prestige": lambda m:  m.agents.agg("prestige_visibility", np.mean),
                             "top1pct_prestige_share": lambda m: (np.sum([a.prestige for a in m.agents if a.prestige >= np.percentile([ag.prestige for ag in m.agents], 99)]) / np.sum([a.prestige for a in m.agents])
                             if np.sum([a.prestige for a in m.agents]) > 0 else 0),
                             "top10pct_prestige_share": lambda m: (np.sum([a.prestige for a in m.agents if a.prestige >= np.percentile([ag.prestige for ag in m.agents], 90)]) / np.sum([a.prestige for a in m.agents])
                             if np.sum([a.prestige for a in m.agents]) > 0 else 0),
                             "avgcurrentAgentKnowledge": lambda m: m.avgcurrentAgentKnowledge,
                             "explored_percentage": lambda m: m.explored_percentage,
                             "explored_weighted_by_initial_knowledge": lambda m: m.explored_weighted_by_initial_knowledge,
                             "total_initial_knowledge": lambda m: m.totalInitialKnowledge,
                             "avg_knowledge_on_grid": lambda m: np.mean(m.grid.properties["knowledge"].data),
                             "best_knowledge": lambda m: np.max(m.grid.properties["knowledge"].data),
                             "avg_distance_between_agents": lambda m: m.agent_avg_distance,
                             "percentage_knowledge_harvested": lambda m: m.percentage_knowledge_harvested,
                             "Corr_prestige_localMerit": correlation_reporter("current_localMerit", "prestige"),
                             "Corr_prestige_idealMerit": correlation_reporter("ideal_Merit", "prestige"),
                             "Corr_prestigeVisibility_localMerit": correlation_reporter("current_localMerit", "prestige_visibility"),
                             "Corr_prestigeVisibility_idealMerit": correlation_reporter("ideal_Merit", "prestige_visibility"),
                             "Corr_prestigevanishing_localMerit": correlation_reporter("current_localMerit", "prestige_vanishing"),
                             "Corr_prestigevanishing_idealMerit": correlation_reporter("ideal_Merit", "prestige_vanishing"),
                            }, # "corr_prestige_localMerit": lambda m: (np.corrcoef([a.prestige for a in m.agents],[a.current_localMerit for a in m.agents])[0, 1] if  np.std([a.prestige for a in m.agents]) > 0 and np.std([a.current_localMerit for a in m.agents]) > 0 else 0)

            agent_reporters={"prestige":   "prestige",
                             "prestige_vanishing":   "prestige_vanishing",
                             "prestige_visibility":   "prestige_visibility",
                             "lastTileKnowledge": "lastTileKnowledge",
                             'curiosity': "curiosity",                  
                             "localMerit": "current_localMerit",
                             "idealMerit": "ideal_Merit",
                             })
        self.endLoopUpdate()

    def updateAvgknowledge(self):
        self.avgcurrentAgentKnowledge = np.mean([agent.lastTileKnowledge for agent in self.agents])

    def updateIdealMerit(self):
        """ first computes the difference between the current agent positionning and the ideal one computed from the knowledge in last round    
        """
        temp_knowledge_map= self.grid.properties["previous_knowledge"].data
        
        # compute the harvest made
        IdealHarvestList = []
        L = [(a.pos, a) for a in self.agents]
        L = sorted(L, key=lambda L: L[0])  #sort for positions so same positions are together
        ActualHarvestList = []
        pastpos, redone = -1,0
        for kpos,k in L:
            know = temp_knowledge_map[kpos]
            if kpos == pastpos:
                redone += 1
            else:
                redone = 0
            know = know * ((1- self.harvest) ** redone)
            pastpos = kpos
            ActualHarvestList.append([know*self.harvest, kpos, k])
        #print("at step", self.steps, "total harvest", sum([k[0] for k in ActualHarvestList]), np.sum(temp_knowledge_map) - np.sum(self.grid.properties["knowledge"].data))
        
        for k in range(self.number_agents):
            pos = np.unravel_index(temp_knowledge_map.argmax(), temp_knowledge_map.shape) 
            IdealHarvestList.append((temp_knowledge_map[pos]*self.harvest,pos))
            temp_knowledge_map[pos] = temp_knowledge_map[pos] * (1- self.harvest)
        #print("ideal harvest", sum([k[0] for k in IdealHarvestList]))

        #compute loss per agent
        ActualHarvestList = sorted(ActualHarvestList, key= lambda L: L[0])
        for k in range(len(self.agents)):
            loss = ActualHarvestList[k][0] - IdealHarvestList[0][0]
            if loss == 0:
                IdealHarvestList = IdealHarvestList[1:]
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

        self.grid.properties["previous_knowledge"].data = self.grid.properties["knowledge"].data.copy()
            


    def new_place(self, agent1,coords, newAgent = False):
        '''function used to modify an agent location and update the surrounding agent's distance list'''
        if not(newAgent): 
            self.grid.remove_agent(agent1)
        self.grid.place_agent(agent1, coords)
        self.grid.properties["explored"].data[coords] = True
        for agent2 in self.agents:
            if agent2 != agent1:
                dist = agent1.computeDistance(agent2.pos)

    def computeKnowledge(self,pos):
        posX,posY = pos
        value = self.grid.properties["knowledge"].data[posX,posY]
        return (value)
    
    
    def Farm(self, pos):
        self.grid.properties["knowledge"].data[pos] = self.grid.properties["knowledge"].data[pos] * (1- self.harvest)
        self.updateAvgknowledge()
    
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
        if self.new_questions > 0:
            occupied_pos = list(set([ agent.pos for agent in self.agents]))
            self.rng.shuffle(occupied_pos)
            for posX,posY in occupied_pos:
                if self.grid.properties["knowledge"].data[posX,posY] < 1/self.size**2 *0.5:
                    self.grid.properties["knowledge"].data[posX,posY] = self.totalInitialKnowledge - np.sum(self.grid.properties["knowledge"].data)
        self.explored_percentage = np.sum(self.grid.properties["explored"].data)/(self.size**2)
        self.explored_weighted_by_initial_knowledge = np.sum(self.grid.properties["explored"].data * (self.grid.properties["initial_knowledge"].data))/self.totalInitialKnowledge
        self.agent_avg_distance = np.mean([ np.mean([a.computeDistance(b.pos) for b in self.agents if a != b]) for a in self.agents])
        self.percentage_knowledge_harvested = (self.totalInitialKnowledge - np.sum(self.grid.properties["knowledge"].data))/self.totalInitialKnowledge
        
        
        for agent in self.agents:
                agent.computeLocalMerit(agent.pos)
        self.updateIdealMerit() #note that we update the ideal merit now because it will be used next generation to know how "good" did the agent position

        self.datacollector.collect(self)
        if self.explored_50_step == self._default_steps_thresholds and self.explored_percentage >= 0.5: 
            self.explored_50_step = self.steps
        if self.explored_90_step == self._default_steps_thresholds and self.explored_percentage >= 0.9: 
            self.explored_90_step = self.steps
        if self.weighted_50_step == self._default_steps_thresholds and self.explored_weighted_by_initial_knowledge >= 0.5: 
            self.weighted_50_step = self.steps
        if self.weighted_90_step == self._default_steps_thresholds and self.explored_weighted_by_initial_knowledge >= 0.9: 
            self.weighted_90_step = self.steps
        if self.harvested_50_step == self._default_steps_thresholds and self.percentage_knowledge_harvested >= 0.1: 
            self.harvested_50_step = self.steps
        if self.harvested_90_step == self._default_steps_thresholds and self.percentage_knowledge_harvested >= 0.25: 
            self.harvested_90_step = self.steps



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
            if self.new_questions > 0:
                im = ax.imshow(epistemicGrid, vmin=0, vmax=1)
            else:
                im = ax.imshow(epistemicGrid)#, vmin=0, vmax=1)
            im= ax3.imshow(self.computeDistanceToAgents())#, vmin=0, vmax=2*self.size)
            ax3.set_title('Distance to all agents')
            ax2.set_title('Measures over time')
            ax.set_title('Epistemic Landscape')
            cbar = ax.figure.colorbar(im, cax=cax)
            #scat = ax.scatter(PosY, PosX, c = PosAge, alpha = 0.9, cmap = 'OrRd',edgecolors="k")
            scat = ax.scatter(PosY, PosX, c = ["r"] + ['b']*(self.number_agents-1), alpha = 0.9, cmap = 'OrRd',edgecolors="k")

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
        Val1, Val2, Val3 = [0],[0],[0]
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
                # explorePercentage = self.datacollector.get_model_vars_dataframe().iloc[-1]["explored_percentage"]
                # bestKnowledge = np.max(self.grid.properties["knowledge"].data)
                # avgMap = np.mean(self.grid.properties["knowledge"].data)
                # avgAgent = self.avgcurrentAgentKnowledge
                # Val1.append(explorePercentage)
                # Val2.append(avgMap/bestKnowledge)
                # Val3.append(avgAgent/bestKnowledge)
                Val1.append(self.agents[0].current_localMerit)
                Val2.append(10*self.agents[0].ideal_Merit)
                Val3.append(self.agents[0].prestige_vanishing)

                if dynamic_plot:
                    cax.cla()
                    cax2.cla()
                    ax2.cla()
                    ax2.plot(Rounds, Val1, label = "localMerit")#'exploration percentage')
                    ax2.plot(Rounds, Val2, label = "10 * loss (ideal Merit)")#'Avg Map / Best Tile')
                    ax2.plot(Rounds, Val3, label = "vanishing prestige") #'Avg Agent / Best Tile')
                    ax2.legend()
                    PosX = [k.pos[0]+(k.unique_id/self.number_agents-0.5)**2 for k in self.agents ]
                    PosY = [k.pos[1]+(k.unique_id/self.number_agents-0.5)**2 for k in self.agents ]
                    PosAge = [k.curiosity for k in self.agents]
                    data = np.stack([PosY,PosX]).T
                    scat.set_offsets(data)
                    #scat.set_array(np.array(PosAge))
                    #scat.set_clim(vmin=min(PosAge), vmax=max(PosAge))
                    
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
            self.datacollector.get_agent_vars_dataframe().to_csv("data/agent_"+csv_name+".csv")
            self.datacollector.get_model_vars_dataframe().to_csv("data/model_"+csv_name+".csv")
            print("data saved to ", "agent_"+csv_name+".csv", "model_"+csv_name+".csv")
        if end_report_file != "":
            a = {k: self.datacollector.get_model_vars_dataframe()[k].iloc[-1] for k in self.datacollector.get_model_vars_dataframe().columns if k[0].islower()}
            b = {k: self.__getattribute__(k) for k in self.__dict__ if (type(self.__getattribute__(k)) in [int,float,str,list, dict] and k[0] != '_') }
            c = {"Mean_of_"+k: self.datacollector.get_model_vars_dataframe()[k].mean() for k in self.datacollector.get_model_vars_dataframe().columns if k[0].isupper()}
            row = {**a, **b, **c}
            needs_header = not(is_non_zero_file("data/"+end_report_file))
            with open("data/"+end_report_file, 'a') as f:
                writer = csv.writer(f)
                if needs_header:
                    writer.writerow(sorted(row.keys()))
                row = [row[k] for k in sorted(row.keys())]
                writer.writerow(row)
            print("end report saved to ", end_report_file)
        del self

    def run_mode_for_bulk(self, longitudinal = False):
            for k in range(self.step_limit):
                self.step(True)

            if longitudinal == False:
                a = {k: self.datacollector.get_model_vars_dataframe()[k].iloc[-1] for k in self.datacollector.get_model_vars_dataframe().columns if k[0].islower()}
                b = {k: self.__getattribute__(k) for k in self.__dict__ if (type(self.__getattribute__(k)) in [int,float,str,list, dict] and k[0] != '_') }
                c = {"Mean_of_"+k: self.datacollector.get_model_vars_dataframe()[k].mean() for k in self.datacollector.get_model_vars_dataframe().columns if k[0].isupper()}
                row = {**a, **b, **c}
                return row
            else:
                return self.datacollector.get_agent_vars_dataframe()
    



def generate_data_parametric_exploration(filename, param_grid, repeats_per_setting = 10, change_landscape_seed = False, intention = "w", skip_to = 0, longitudinal = False):
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


    if not(longitudinal):
        print(filename + ".csv", intention)
        needs_header = not(is_non_zero_file(filename))
        f = open(filename + ".csv", intention)
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
    
    if longitudinal:
        os.makedirs(filename, exist_ok=True)
        f = open(f"{filename}/" +"fixed_params.txt", "w")
        f.write(str({k: param_grid[k] for k in param_grid.keys() if k not in varying_params}))
        f.close()



    for idx in param_range:
        param_set = param_combinations[idx]
        all_params = {param_names[i]: param_set[i] for i in range(len(param_names))}

        print("now processing param set ", idx+1, "out of", len(param_combinations))
        for repeat in range(repeats_per_setting):
            all_params["agent_seed"] = repeat 
            if change_landscape_seed:
                all_params["generation_params"] = all_params.get("generation_params", {}) | {"seed": repeat}
            model = MyModel(**all_params)
            row = model.run_mode_for_bulk(longitudinal=longitudinal)
            if longitudinal:
                row.to_csv(f"{filename}/run_" + "_".join([f"{key}{value}" for key, value in all_params.items() if key in varying_params + ['agent_seed']])+'.csv', index=False)
            else:
                writer.writerow([row[k] for k in sorted(row.keys())])
            del model
    f.close()