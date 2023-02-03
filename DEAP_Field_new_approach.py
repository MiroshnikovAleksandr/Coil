from deap import base, algorithms
from deap import creator
from deap import tools

import random
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import Field_functions as ff

# Setup

# Genetic
no_of_generations = 50  # how many generations there will be
population_size = 50  # amount if individuals in a population
size_of_individual = 100  # amount of coils
probability_of_mutation = 0.1  # mutation probability for mutation strategy
tournSel_k = 4  # number of contestants in the crossing over tournament
no_of_variables = 10  # Количество переменных (радиусов витков)
CXPB, MUTPB = 0.4, 0.04  # crossing over and mutation probability

# Geometrical & Electrical parameters
a_max = 0.5  # [m] Max coil radius
a_min = 0.05  # [m] Min coil radius
I = 1  # [A] Current
spacing = 1.5  # spacing for calculation domain
cp = 30  # Calculation domain points
minimal_gap = 0.002

# Displaying the random seed in the output
seed = random.randrange(sys.maxsize)
rng = random.Random(seed)
print("Seed was:", seed)

# Creating DEAP classes and registering GA functions
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, size_of_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

population = toolbox.population(n=population_size)


def decode(individual):
    len_of_radius = int(size_of_individual/no_of_variables)

    radiuses = []
    for i in range(0, size_of_individual, len_of_radius):
        radius = individual[i:i + len_of_radius]
        radius = ''.join(str(x) for x in radius)
        radiuses.append(radius)

    radiuses_decimal = list(map(lambda x: int(x, 2), radiuses))
    precision = (a_max - a_min) / ((2 ** len_of_radius) - 1)
    radiuses_normalized = [x * precision + a_min for x in radiuses_decimal]

    return radiuses_normalized


def objective_fxn(individual):
    r_i = decode(individual)
    Bz = ff.Bz(a_max, a_min, no_of_variables, I, spacing, cp, r_i)

    height = 0.015  # [m]
    COV = ff.COV_circ(Bz, a_max, height, spacing)

    obj_function_value = COV

    return [obj_function_value]


def feasible(individual):
    r_i = decode(individual)

    for x in r_i:
        for y in r_i:
            if abs(x - y) > minimal_gap:
                return False
            else:
                return True


# registering objective function with respect to constraints
toolbox.register('evaluate', objective_fxn)  # provide the objective function here
# toolbox.decorate('evaluate', tools.DeltaPenalty(feasible, 1000))

# registering DEAP built in GA methods
toolbox.register("mate", tools.cxTwoPoint)  # strategy for crossover, this classic two point crossover
toolbox.register("mutate", tools.mutFlipBit,
                 indpb=probability_of_mutation)  # mutation strategy with probability of mutation
toolbox.register("select", tools.selTournament, tournsize=tournSel_k)  # selection strategy

# registering statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register('Min', np.min)
stats.register('Max', np.max)
stats.register('Avg', np.mean)
stats.register('Std', np.std)

# implementing the GA
population, logbook = algorithms.eaSimple(population,
                                          toolbox,
                                          cxpb=CXPB,
                                          mutpb=MUTPB,
                                          ngen=no_of_generations,
                                          stats=stats,
                                          verbose=True)

minFitnessValues, meanFitnessValues = logbook.select('Min', 'Avg')

plt.title("Minimum values of f(x,y) Reached Through Generations",fontsize=20,fontweight='bold')
plt.xlabel("Generations", fontsize=18, fontweight='bold')
plt.ylabel("Value of Himmelblau's Function", fontsize=18, fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.plot(logbook.select('Min'))

# displaying the best individual and its value
hall_of_fame = tools.HallOfFame(1)
hall_of_fame.update(population)
print(decode(hall_of_fame[0]))
print(objective_fxn(hall_of_fame[0])[0])
#print(COV(hall_of_fame[0]))

df = pd.DataFrame(sorted(decode(hall_of_fame[0])))
df.to_excel('hall_of_fame.xlsx')


# plt.show()
