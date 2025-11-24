
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
# self.model.grid.move_agent(self, new_position)

def randominit(posX,posY,size, args):
    return random.random()

def Gaussian(posX,posY,size, args):
    '''Requires args to contain (kappa or std) and seed'''
    if "kappa" in args.keys():
        kappa = args['kappa']
    else :
        kappa = 1/(args['std']/size)

    state = random.getstate()
    random.seed(args["seed"])
    muX = random.random()*2*np.pi
    muY = random.random()*2*np.pi
    random.setstate(state)

    valX = vonmises.pdf(posX*2*np.pi/size, loc=muX, kappa=kappa)
    valY = vonmises.pdf(posY*2*np.pi/size, loc=muY, kappa=kappa)
    return valX * valY

def noisyGaussian(posX,posY,size, args):
    '''Requires args to contain prop_random'''
    prop_random = args['prop_random']
    a = random.random()
    b = Gaussian(posX, posY, size, args)
    return a+prop_random*b

def multipleGaussians(posX,posY,size, args):
    '''Requires args to contain (number_gaussian) and seed'''
    number_gaussians  = args["number_gaussians"]
    state = random.getstate()
    random.seed(args["seed"])
    AllSeeds = [random.random() for k in range(number_gaussians)]
    random.setstate(state)
    tot = 0
    tempDict = copy.deepcopy(args)
    for k in range(number_gaussians):
        tempDict["seed"] = AllSeeds[k]
        tot += Gaussian(posX,posY,size, tempDict)
    return tot

## additional function for agent curiosity 
def beta(mu, seed, params): 
    if mu > 0 and mu < 1 :
        np.random.seed(seed)
        std_beta = params.get("std_curiosity", 0.1)
        alpha =  ((1 - mu) / std_beta**2 - 1 / mu) * mu ** 2
        beta = alpha * (1 / mu - 1)
        return np.random.beta(alpha, beta)
    else:
        return mu

