import random
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import Generator
from scipy.stats import vonmises
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
# self.model.grid.move_agent(self, new_position)


def randominit(posX, posY, size, args, rng: Optional[np.random.Generator]):
    rng = rng or np.random.default_rng()
    return rng.random()


def Gaussian(posX, posY, size, args, rng: Optional[np.random.Generator]):
    if "kappa" in args.keys():
        kappa = args["kappa"]
    else:
        kappa = 1 / (args["std"] / size)

    rng = rng or np.random.default_rng()
    muX = rng.random() * 2 * np.pi
    muY = rng.random() * 2 * np.pi

    valX = vonmises.pdf(posX * 2 * np.pi / size, loc=muX, kappa=kappa)
    valY = vonmises.pdf(posY * 2 * np.pi / size, loc=muY, kappa=kappa)
    return valX * valY


def noisyGaussian(posX, posY, size, args, rng: Optional[np.random.Generator]):
    """Requires args to contain prop_random"""
    prop_random = args["prop_random"]
    rng = rng or np.random.default_rng()
    a = rng.random()
    b = Gaussian(posX, posY, size, args, rng)
    return a + prop_random * b


def multipleGaussians(posX, posY, size, args, rng: Optional[np.random.Generator]):
    """Requires args to contain (number_gaussian) and seed"""
    number_gaussians = args["number_gaussians"]

    rng = rng or np.random.default_rng()
    AllSeeds = [rng.random() for k in range(number_gaussians)]

    tot = 0
    tempDict = copy.deepcopy(args)
    for k in range(number_gaussians):
        tempDict["seed"] = AllSeeds[k]
        tot += Gaussian(posX, posY, size, tempDict, rng)
    return tot


## additional function for agent curiosity
def beta(mu, rng: Generator, params):
    if mu > 0 and mu < 1:
        std_beta = params.get("std_curiosity", 0.1)
        alpha = ((1 - mu) / std_beta**2 - 1 / mu) * mu**2
        beta = alpha * (1 / mu - 1)
        return rng.beta(alpha, beta)
    else:
        return mu


def uniform(_mu, rng: Generator, _params):
    return rng.uniform(0.0, 1.0)
