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
# self.model.grid.move_agent(self, new_position)

def rationRewards(currentRw,newRw):
    '''Usefull function to give a normalized output'''
    total = (abs(newRw) + abs(currentRw))
    if total == 0:
        return 0
    else:
        diff = (newRw - currentRw)
        return diff/total


class Scientist(mesa.Agent):
    def __init__(self, model, curiosity,epsilon):
        super().__init__(model)
        self.curiosity = curiosity
        self.epsilon = epsilon
        self.distanceList = []
        self.maxDistance = 0
        self.prestige = 1
        self.age = 23
        self.lastTileKnowledge = 0

    def modifyDistanceList(self,dist,agent2):
        '''We store who is closest to who for each scientists, so the can compete to be in the closest proximity to as many scientists as possible'''
        if dist < self.maxDistance:
            self.distanceList[-1] = (dist, agent2)
            sublist = [ a for a,b in self.distanceList]
            ind =   np.argsort(sublist)
            self.distanceList = [ self.distanceList[a]  for a in ind]
            self.maxDistance = self.distanceList[-2][0]   #here importantly len(distance list) = number of connection +1 so the max is the penultimate agent 

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

    def computeAllrewards(self, Novelty, Network):
        return self.curiosity * Novelty/ self.model.avgAgentKnowledge + (1-self.curiosity) * Network/self.model.number_connection

    def step(self):
        '''pick a random node and check if according to the agent preference it is better to move to that node'''
        neighbors_nodes = self.model.grid.get_neighborhood(self.pos, moore = False, include_center=False)
        optionpos = random.choice(neighbors_nodes)

        currentRwNovelty = self.model.computeRewardKnowledge(self.pos)
        newRwNovelty = self.model.computeRewardKnowledge(optionpos)

        currentRwNetwork = self.model.computeRewardSpatial(self, self.pos)
        newRwNetwork = self.model.computeRewardSpatial(self, optionpos)

        noise = 2*(random.random()-0.5 ) 

        totCurrentReward = self.computeAllrewards(currentRwNovelty, currentRwNetwork)
        totNewReward = self.computeAllrewards(newRwNovelty, newRwNetwork)

        
        prob =    (1- self.epsilon) *(rationRewards(totCurrentReward, totNewReward) ) + self.epsilon*noise
        if prob > 0 :
            self.model.new_place(self, optionpos)
            
        self.model.Farm(self.pos) #farming also updates AvgAgentKnowledge

        self.age += 1
        self.lastTileKnowledge = self.model.grid.properties["knowledge"].data[self.pos]

        for dist, agent2 in self.distanceList[:-1]: #for the moment prestige is only based on who cites you regardless of your scientific production
            agent2.incrPrestige(1)


class MyModel(mesa.Model):
    def __init__(self, n_agents, n_connection, initial_curiosity, epsilon, harvest, sizeGrid, initCellFunc, generation_params = {"seed" :0}, agent_generation_rate = -1, constant_population = 1):
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
        self.grid = mesa.space.MultiGrid(sizeGrid, sizeGrid, torus=True)
        self.grid.add_property_layer(PropertyLayer("knowledge",  sizeGrid,sizeGrid,0.0, dtype=float) )

        random.seed(generation_params["seed"])
        self._seed = generation_params["seed"]
        for posX,posY in list(itertools.product(range(sizeGrid), range(sizeGrid))):
            self.grid.properties["knowledge"].data[posX,posY] = initCellFunc(posX,posY, sizeGrid, generation_params)
        

        for _ in range(n_agents):
            a = Scientist(self, initial_curiosity,epsilon)
            coords = (self.random.randrange(0, sizeGrid), self.random.randrange(0, sizeGrid))
            a.age = self.random.randint(23,50)
            self.grid.place_agent(a, coords)
            a.lastTileKnowledge = self.grid.properties["knowledge"].data[coords]

        

        #initialize the distance list
        N = len(self.agents.select(lambda a: type(a) == Scientist))
        DistanceList = {a : [] for a in self.agents.select(lambda a: type(a) == Scientist) }
        for agent1 in self.agents:
            for agent2 in self.agents:
                if agent2 != agent1:
                    dist = agent1.computeDistance(agent2.pos)

                    if len(DistanceList[agent1]) < self.number_connection+1: # We keep more than the minimum so we can replace 
                        DistanceList[agent1].append((dist,agent1))
                    else:
                        sublist =[a for a,b in DistanceList[agent1]]
                        if dist < max(sublist):
                            
                            position =np.argmax(sublist) 
                            DistanceList[agent1][position] = (dist,agent1)
            DistanceList[agent1] = sorted(DistanceList[agent1])

        for agent1 in self.agents:
            agent1.distanceList = DistanceList[agent1]
            agent1.maxDistance = max(DistanceList[agent1])[0]
        
        self.updateAvgAgentKnowledge()
        self.datacollector = mesa.DataCollector(
            model_reporters={"mean_age": lambda m: m.agents.agg("age", np.mean)},
            agent_reporters={"age": "age"})

    def updateAvgAgentKnowledge(self):
        self.avgAgentKnowledge = np.mean([agent.lastTileKnowledge for agent in self.agents])

    def new_place(self, agent1,coords, newAgent = False):
        '''function used to modify an agent location and update the surrounding agent's distance list'''
        if not(newAgent): 
            self.grid.remove_agent(agent1)
        self.grid.place_agent(agent1, coords)

        agent1.distanceList = [ (2*self.size,agent1)  for a in range(self.number_connection +1)]
        agent1.maxDistance = 2*self.size
        

        for agent2 in self.agents:
            if agent2 != agent1:
                dist = agent1.computeDistance(agent2.pos)
                agent2.modifyDistanceList(dist,agent1)
                agent1.modifyDistanceList(dist,agent2)


    def computeRewardSpatial(self, agent1, pos1):
        RwSpatial = 0
        for agent2 in self.agents:
            dist = agent2.computeDistance(pos1)
            if dist < agent2.maxDistance: # if the agent is super close it is selected
                RwSpatial += 1
            else:
                closest_agents = [ b for a,b in agent2.distanceList]
                if agent1 in closest_agents and dist <= agent2.distanceList[-1][0]: # if the agent is already in the list, it competes with the one closest outside the list
                                                                                    # if two agents have the same distance, advantages goes to the one already in the list
                    RwSpatial += 1
        return RwSpatial
                

    def computeRewardKnowledge(self,pos):
        posX,posY = pos
        return (self.grid.properties["knowledge"].data[posX,posY])
    
    def Farm(self, pos):
        self.grid.properties["knowledge"].data[pos] = self.grid.properties["knowledge"].data[pos] * (1- self.harvest)
        self.updateAvgAgentKnowledge()
    
    def endLoopUpdate(self):
        
        self.datacollector.collect(self)
        if self.agent_generation > 0:
            prevNew = int((self.steps-1) // self.agent_generation)
            newNew = int(self.steps // self.agent_generation)
            print(prevNew, newNew)
            ListPA  = [(a.prestige, a) for a in self.agents]
            totalPrestige = sum([a.prestige for a in self.agents])
            supervisors = []
            for k in range(prevNew, newNew):
                val = self.random.uniform(0, totalPrestige)
                for p, a in ListPA:
                    if val <= p:
                        supervisors.append(a)
                        break
                    val -= p

            for sup in supervisors:    
                a = Scientist(self, self.initial_curiosity,self.epsilon)
                coords = sup.pos
                self.grid.place_agent(a, coords)
            

    def step(self, endupdate = True):
        # compute everything and let agents take the decision 
        self.agents.select(lambda a: type(a) == Scientist).shuffle_do("step")
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
        PosX = [k[0]+(random.random()-0.5)**2 for k in Scientists]
        PosY = [k[1]+(random.random()-0.5)**2 for k in Scientists]
        ax.scatter(PosY, PosX, color = 'r', alpha = 0.5)
        plt.show()
    
    def animate_steps(self):
        epistemicGrid = self.grid.properties["knowledge"].data
        fig, (ax,ax2) = plt.subplots(ncols = 2, figsize=(10, 6))
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')
        im = ax.imshow(epistemicGrid)#, vmin=0, vmax=1)
        cbar = ax.figure.colorbar(im, cax=cax)
        PosX = [k.pos[0]+(random.random()-0.5)**2 for k in self.agents]
        PosY = [k.pos[1]+(random.random()-0.5)**2 for k in self.agents]
        PosAge = [k.age for k in self.agents]
        scat = ax.scatter(PosY, PosX, c = PosAge, alpha = 0.5, cmap = 'OrRd')

        Rounds = [self.steps]

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
        avgAgent = self.avgAgentKnowledge
        Val1 = [explorePercentage]
        Val2 = [avgMap/bestKnowledge]
        Val3 = [avgAgent/bestKnowledge]

        def update(frame, cbar = cbar):
            self.step(True)
            #measure 
            Rounds.append(self.steps)
            listForAvg = []
            for agent in self.agents:
                Xx, Yy  = agent.pos
                explorelist[Xx][Yy] = 1
                listForAvg.append(self.grid.properties["knowledge"].data[Xx,Yy])
            explorePercentage = sum([ sum(k)for k in explorelist])/(self.size**2)
            bestKnowledge = np.max(self.grid.properties["knowledge"].data)
            avgMap = np.mean(self.grid.properties["knowledge"].data)
            avgAgent = self.avgAgentKnowledge
            Val1.append(explorePercentage)
            Val2.append(avgMap/bestKnowledge)
            Val3.append(avgAgent/bestKnowledge)

            self.endLoopUpdate()
            cax.cla()
            ax2.cla()
            ax2.plot(Rounds, Val1, label = 'exploration percentage')
            ax2.plot(Rounds, Val2, label = 'Avg Map / Best Tile')
            ax2.plot(Rounds, Val3, label = 'Avg Agent / Best Tile')
            ax2.legend()
            PosX = [k.pos[0]+(random.random()-0.5)**2 for k in self.agents ]
            PosY = [k.pos[1]+(random.random()-0.5)**2 for k in self.agents ]
            PosAge = [k.age for k in self.agents]
            data = np.stack([PosY,PosX,PosAge]).T
            scat.set_offsets(data)
            epistemicGrid = self.grid.properties["knowledge"].data
            im = ax.imshow(epistemicGrid)
            cbar = ax.figure.colorbar(im, cax=cax)

        ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=300)
        plt.show()


