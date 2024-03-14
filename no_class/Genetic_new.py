import random
import sys
import math
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tomli

import Bz_Field
import COV
import Resistance
import Plot

from deap import base, algorithms
from deap import creator
from deap import tools
from COV import COV_circle, COV_square, COV_piecewise_linear
from utilities import index_of_element, Radii_in_coords

import warnings

warnings.filterwarnings("error")

# PARAMETERS
# Algorith parameters
discreteness = 3  # number of digits encoding the position of 1 wire
ngen = 80
pop_size = 50
probability_of_mutation = 0.05
tournSel_k = 4
CXPB = 0.4
MUTPB = 0.04

# Geometric parameters
r_max = 0.125
r_min = 0.0125
minimal_gap = 0.003
spacing = 1.5
calculation_area = 0.5
cp = 100
height = 0.03

# Physical parameters
I = 1
freq = 6.78e6

ind_length = discreteness * int((r_max - r_min) // minimal_gap)  # number of digits encoding a chromosome

toolbox = base.Toolbox()  # create toolbox for genetic algorithm

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


def zero_or_one(p: float):
    """
    Returns 1 with probability p and 0 with probability 1 - p.
    @param p: probability that the function will return 1
    @return: 0 or 1
    """
    rand = random.random()
    if rand > p:
        return 0
    else:
        return 1


toolbox.register("ZeroOrOne", zero_or_one, p=0.8)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.ZeroOrOne, ind_length)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

pop = toolbox.population(n=pop_size)


def decode(ind: list):
    """
    Decodes the individual chromosome into a list of radii of sequential coil turns
    @param ind: chromosome
    @return: list of radii

    First, the function iterates over a chromosome and sequentially counts the 1s in the chromosome.
    If it encounters a 0 after a 1 it checks whether the amount of 1s is divisible by :var:'discreteness', so
    that the chromosome can be split into an integer number of coil turns. The preceding 1s are replaced with
    0s until the amount of 1s is divisible by :var:'discreteness'. In the beginning a 0 is appended to
    :var:'ind' so that when the iteration reaches the end it runs a final check. The 0 is deleted after the
    iteration. The resulting 1s positions are recorded in the :var:'ones_placement' list.

    Secondly, every consecutive :var:'discreteness' number of 1s are mapped to a specific wire location, or
    a radius. Each :var:'discreteness'-d 1 in :var:'ones_placement' starting with the :var:'discreteness'//2-d
    is the center of a wire.
    """
    radii = []

    # Append a 0 to :var:'ind'
    ind = ind + [0]
    # Counter of 1s
    cnt = 0
    # List of 1s placement
    ones_placement = []  # list of positions of 1s in ::ind::

    # Iterate over :var:'ind'
    for i in range(ind_length):
        if ind[i] == 1:
            cnt += 1
            ones_placement.append(i)
        if ind[i] == 1 and ind[i + 1] == 0 and cnt % discreteness != 0:
            j = i
            # Replace preceding 1s with 0s until :var:'cnt' is divisible by :var:'discreteness'
            while cnt % discreteness != 0:
                ind[j] = 0
                cnt -= 1
                ones_placement.pop()
                j -= 1

    # Remove the 0 from the end of :var:'ind'
    ind.pop()

    # Map each :var:'discreteness' number of 1s to a radius
    for i in range(discreteness // 2, len(ones_placement), discreteness):
        radii.append(r_min + ones_placement[i] * (r_max - r_min) / ind_length)
    # If the :var:'radii' array ends up being empty, add a 0 to it
    if not radii:
        radii.append(0)
    return radii


def determine_Bz(ind: list):
    """
    Calculates the Z-component of magnetic inductance Bz at the specified height above the coil encoded
    by :var:'ind' in the specified calculation area.
    @param ind: chromosome
    @return: numpy.ndarray(2, 2) Bz
    """
    R = decode(ind)
    return Bz_Field.Bz_circular_contour(R=R,
                                        I=I,
                                        spacing=spacing,
                                        cp=cp,
                                        height=height)


def determine_COV(ind: list):
    """
    Calculates the coefficient (COV) of variation of the Bz of the coil, encoded by :var:'ind'.
    @param ind: chromosome
    @return: tuple (cov, )
    """
    bz = determine_Bz(ind)
    try:
        cov = COV.COV_circle(Bz=bz,
                             max_coil_r=r_max,
                             spacing=spacing,
                             P=calculation_area)
    # Return 1 if COV is incalculable
    except RuntimeWarning:
        cov = 1
    return cov,


def max_min_bz_ratio(ind: list):
    """
    Calculates (max(Bz) - min(Bz)) / max(Bz) for Bz inside the calculation area.
    @param ind: chromosome
    @return: tuple ((max(Bz) - min(Bz)) / max(Bz), )
    """
    bz = determine_Bz(ind)
    bz_max_min = COV.max_min_bz_circular(Bz=bz,
                                         max_coil_r=r_max,
                                         spacing=spacing,
                                         P=calculation_area)
    bz_min = bz_max_min[0]
    bz_max = bz_max_min[1]
    try:
        return abs((bz_max - bz_min) / bz_max),
    except RuntimeWarning:
        return 1,


def check_feasibility(ind: list):
    """
    Checks whether the coil encoded by the :var:'ind' has enough coil turns.
    @param ind: chromosome
    @return: boolean
    """
    return len(decode(ind)) == 9


# registering objective function with constraint
toolbox.register("evaluate", max_min_bz_ratio)  # provide the objective function here
# toolbox.decorate("evaluate",
#                  tools.DeltaPenalty(check_feasibility, 0.5))  # constraint on the objective function

# registering basic processes using built-in functions in DEAP
toolbox.register("select", tools.selTournament, tournsize=tournSel_k)  # selection strategy
toolbox.register("mate", tools.cxTwoPoint)  # strategy for crossover, this classic two point crossover
toolbox.register("mutate", tools.mutFlipBit,
                 indpb=probability_of_mutation)  # mutation strategy with probability of mutation

# register statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register('Min', np.min)
stats.register('Max', np.max)
stats.register('Avg', np.mean)
stats.register('Std', np.std)

# run the GA using a built-in DEAP algorithm
pop, logbook = algorithms.eaSimple(pop,
                                   toolbox,
                                   cxpb=CXPB,
                                   mutpb=MUTPB,
                                   ngen=ngen,
                                   stats=stats,
                                   verbose=True)

hall_of_fame = tools.HallOfFame(1)
hall_of_fame.update(pop)
print(decode(hall_of_fame[0]))
print(len(decode(hall_of_fame[0])))
print(max_min_bz_ratio(hall_of_fame[0]))
print(determine_COV(hall_of_fame[0]))
print(list(map(len, list(map(decode, pop)))))
