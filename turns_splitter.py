from deap import base, algorithms
from deap import creator
from deap import tools

import random
import sys
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

turns = [0.49986530221352876, 0.4957422334496335, 0.4913159397152264, 0.48650341844713946, 0.46856754015057495,
         0.4194663585872939, 0.35410272529314335, 0.28729411095991536, 0.15641894155542901, 0.07312009035069601]
turns = [0.5, 0.498, 0.496, 0.494, 0.4349677419354839, 0.3703329864724246, 0.2621789802289282, 0.05]
turns = [0.5, 0.498, 0.496, 0.4653548387096774, 0.409681581685744, 0.33337148803329864, 0.19536108220603537]


def main(turns):
    # GLOBAL CONSTANTS
    INDIVIDUAL_SIZE = 50
    POPULATION_SIZE = 400
    P_CROSSOVER = 0.9
    P_MUTATION = 0.1
    MAX_GENERATIONS = 100

    # GLOBAL VARIABLES
    LENGTH = len(turns)

    hall_of_fame = tools.HallOfFame(1)

    creator.create("SumMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.SumMin)

    toolbox = base.Toolbox()
    toolbox.register("randomOrder", random.sample, range(LENGTH), LENGTH)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.randomOrder, 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    population = toolbox.population(n=POPULATION_SIZE)

    def minimal_sum(individual):
        sum1 = sum(individual[0][:(len(individual[0]) // 2)])
        sum2 = sum(individual[0][(len(individual[0]) // 2):])
        return [abs(sum1 - sum2)]

    def cxOrdered(ind1, ind2):
        for x1, x2 in zip(ind1, ind2):
            tools.cxOrdered(x1, x2)
        return ind1, ind2

    def mutShuffleIndexes(ind, indpb):
        for x in ind:
            tools.mutShuffleIndexes(x, indpb=indpb)
        return ind,

    toolbox.register("evaluate", minimal_sum)
    toolbox.register("select", tools.selTournament, tournsize=4)
    toolbox.register("mate", cxOrdered)
    toolbox.register("mutate", mutShuffleIndexes, indpb=0.3)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('Min', np.min)
    stats.register('Avg', np.mean)

    population, logbook = algorithms.eaSimple(population,
                                              toolbox,
                                              cxpb=P_CROSSOVER,
                                              mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS,
                                              stats=stats,
                                              halloffame=hall_of_fame,
                                              verbose=True)

    print(hall_of_fame[0][0])
    first_turns = [turns[i] for i in hall_of_fame[0][0][(len(turns) // 2):]]
    second_turns = [turns[i] for i in hall_of_fame[0][0][:(len(turns) // 2)]]
    print(f'{first_turns}, {second_turns}')
    print(f'{sum(first_turns) * 2 * math.pi}, {sum(second_turns) * 2 * math.pi}, {sum(turns) * 2 * math.pi}')
    print(1 - (sum(first_turns) / sum(second_turns)))


def normal_solution(turns):

    # This code works best if the :turns: array is sorted, which it is

    # n = func(freq)
    n = 2

    res = []
    for i in range(n):
        res.append([])

    for turn in turns:
        res[0].append(turn)
        res = sorted(res, key=lambda x: sum(x))

    for i in range(n):
        print(sum(res[i]) * 2 * math.pi)
    print(sum(turns) * 2 * math.pi)


normal_solution(turns)
