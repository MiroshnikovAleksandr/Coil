from deap import base, algorithms
from deap import creator
from deap import tools

import random
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import Field_functions as ff
import DEAP_Field_newnew as DF

# Genetic
no_of_generations = 50  # максимальное количество поколений
population_size = 50  # количество индивидуумов в популяции
size_of_variable = 8  # длина подлежащей оптимизации битовой строки
probability_of_mutation = 0.1  # вероятность мутации индивидуума
tournSel_k = 4
CXPB, MUTPB = 0.4, 0.04  # вероятность мутации и срещивания

# Geometrical & Electrical parameters.toml
a_max = 0.5  # [m] Max coil radius
a_min = 0.05  # [m] Min coil radius
I = 1  # [A] Current
spacing = 1.5  # spacing for calculation domain
cp = 30  # Calculation domain points
minimal_gap = 0.002

creator.create('FitnessMin_var', base.Fitness, weights=(-1.0,))
creator.create('Variables', list, fitness=creator.FitnessMin_var)

toolbox = base.Toolbox()
toolbox.register("ZeroOrOne", random.randint, 0, 1)
toolbox.register("variables", tools.initRepeat, creator.Variables, toolbox.ZeroOrOne, size_of_variable)
toolbox.register("population_of_variables", tools.initRepeat, list, toolbox.variables)

pop = toolbox.population_of_variables(n=population_size)


def decode(variables):
    return int(''.join(str(x) for x in variables), 2)


def objective_fxn(variables):
    return DF.main(decode(variables))


toolbox.register("evaluate", objective_fxn)  # privide the objective function here

# registering basic processes using bulit in functions in DEAP
toolbox.register("mate", tools.cxTwoPoint)  # strategy for crossover, this classic two point crossover
toolbox.register("mutate", tools.mutFlipBit,
                 indpb=probability_of_mutation)  # mutation strategy with probability of mutation
toolbox.register("select", tools.selTournament, tournsize=tournSel_k)  # selection startegy

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register('Min', np.min)
stats.register('Max', np.max)
stats.register('Avg', np.mean)
stats.register('Std', np.std)

pop, logbook = algorithms.eaSimple(pop,
                                   toolbox,
                                   cxpb=CXPB,
                                   mutpb=MUTPB,
                                   ngen=no_of_generations,
                                   stats=stats,
                                   verbose=True)

minFitnessValues, meanFitnessValues = logbook.select('Min', 'Avg')

# using select method in logbook object to extract the argument/key as list
plt.plot(logbook.select('Min'))
plt.show()
