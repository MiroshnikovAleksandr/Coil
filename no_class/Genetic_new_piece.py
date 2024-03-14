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
coords = [[-0.125, 0.0],
          [-0.0625, 0.108253125],
          [0.0625, 0.108253125],
          [0.125, 0.0],
          [0.0625, -0.108253125],
          [-0.0625, -0.108253125]]
c_min = 0.1
c_max = 1
minimal_gap = 0.003
spacing = 1.5
calculation_area = 0.5
cp = 100
height = 0.03

# Physical parameters
I = 1
freq = 6.78e6


def MaxSide(starting_coords: list):
    """
    Calculates the coefficient for :var:'minimal_gap' based on the coil geometry.
    @param starting_coords: list of outer coil coordinates
    @return: float coefficient
    """
    sides = [(starting_coords[i], starting_coords[i + 1]) for i in range(len(starting_coords) - 1)]
    sides.append((starting_coords[-1], starting_coords[0]))
    sorted_sides = sorted(sides,
                          key=lambda x: np.sqrt((x[0][0] - x[1][0]) ** 2 + (x[0][1] - x[1][1]) ** 2),
                          reverse=True)
    max_side = sorted_sides[0]
    k = (max_side[0][1] - max_side[1][1]) / (max_side[0][0] - max_side[1][0])
    bounds_const = np.sqrt(1 + k ** 2) / (
        abs(max_side[0][1] - k * max_side[0][0]))
    return bounds_const


ind_length = discreteness * int((c_max - c_min) // (MaxSide(coords) * minimal_gap))  # number of digits encoding a chromosome

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
    Decodes the individual chromosome into a list of coefficients of sequential coil turns
    @param ind: chromosome
    @return: list of coeffs

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
    coeffs = []

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
        coeffs.append(0.1 + ones_placement[i] * (c_max - c_min) / ind_length)
    # If the :var:'coeffs' array ends up being empty, add a 0 to it
    if not coeffs:
        coeffs.append(0)
    return coeffs


def determine_Bz(ind: list):
    """
    Calculates the Z-component of magnetic inductance Bz at the specified height above the coil encoded
    by :var:'ind' in the specified calculation area.
    @param ind: chromosome
    @return: Bz
    """
    R = decode(ind)
    return Bz_Field.Bz_piecewise_linear_contour(R=R,
                                                coords=coords,
                                                I=I,
                                                spacing=spacing,
                                                cp=cp,
                                                height=height)


def determine_COV(ind: list):
    """
    Calculates the coefficient (COV) of variation of the Bz of the coil, encoded by :var:'ind'.
    @param ind: chromosome
    @return: COV
    """
    bz = determine_Bz(ind)
    return COV.COV_piecewise_linear(Bz=bz,
                                    coords=coords,
                                    spacing=spacing,
                                    P=calculation_area)


def fitness_func(ind: list):
    """
    Calculates the fitness function of an individual for the genetic algorithm.
    @param ind: chromosome
    @return: tuple (cov, )
    """
    try:
        cov = determine_COV(ind)
    # Return 1 if COV is incalculable
    except RuntimeWarning:
        cov = 1
    return cov,


def check_feasibility(ind: list):
    """
    Checks whether the coil encoded by the :var:'ind' has enough coil turns.
    @param ind: chromosome
    @return: boolean
    """
    return len(decode(ind)) > 4


# registering objective function with constraint
toolbox.register("evaluate", fitness_func)  # provide the objective function here
# toolbox.decorate("evaluate",
#                  tools.DeltaPenalty(check_feasibility, 1.5))  # constraint on the objective function

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
print(fitness_func(hall_of_fame[0]))
print(list(map(len, list(map(decode, pop)))))
