from deap import base, algorithms
from deap import creator
from deap import tools

import random
import sys
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import Field_functions as ff


# plt.style.use(['science', 'ieee'])

# Setup


def main():
    # Genetic
    no_of_generations = 50  # максимальное количество поколений
    population_size = 50  # количество индивидуумов в популяции
    # size_of_individual = 1000 - (1000 % no_of_variables) # длина подлежащей оптимизации битовой строки
    probability_of_mutation = 0.1  # вероятность мутации индивидуума
    tournSel_k = 4
    CXPB, MUTPB = 0.4, 0.04  # вероятность мутации и срещивания

    # Geometrical & Electrical parameters
    a_max = 0.5  # [m] Max coil radius
    a_min = 0.05  # [m] Min coil radius
    I = 1  # [A] Current
    spacing = 1.5  # spacing for calculation domain
    cp = 30  # Calculation domain points

    seed = random.randrange(sys.maxsize)
    rng = random.Random(seed)
    print("Seed was:", seed)

    # one tuple or a pair of lower bound and upper bound for each variable
    # bounds = [(a_min,a_max),(a_min,a_max),(a_min,a_max),(a_min,a_max),(a_min,a_max),(a_min,a_max),(a_min,a_max),(a_min,a_max),(a_min,a_max),(a_min,a_max)]

    # bounds = []
    minimal_gap = 0.002
    # for i in range(no_of_variables):
    #    bounds.append((a_min, a_max-i*minimal_gap))

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    def individualCreator(min_var, max_var, r_min, r_max):
        individual = [float(random.randint(min_var, max_var))]
        for i in range(int(individual[0])):
            individual.append(random.uniform(r_min, r_max))
        return individual

    toolbox = base.Toolbox()
    toolbox.register("ZeroOrOne", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.ZeroOrOne, random.randint(50, 1000))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    pop = toolbox.population(n=population_size)

    def bounds_fn(ind):
        ind = [0]*(10 - (len(ind) % 10)) + ind

        bounds = [(a_min, a_max)]
        len_chromosome = len(ind)
        len_chromosome_one_var = 10
        no_of_variables = len_chromosome // len_chromosome_one_var
        array_of_chromosomes_one = []

        for i in range(0, len_chromosome, len_chromosome_one_var):
            array_of_chromosomes_one.append(''.join(str(xi) for xi in ind[i:i + len_chromosome_one_var]))

        array_of_chromosomes_one_decimal_sorted = list(map(lambda x: int(x, 2), array_of_chromosomes_one))

        zipped = zip(array_of_chromosomes_one, array_of_chromosomes_one_decimal_sorted)
        zipped_sorted = sorted(zipped, key=lambda tup: tup[1], reverse=True)

        array_of_chromosomes_one = [x[0] for x in zipped_sorted]
        array_of_chromosomes_one_decimal_sorted = [x[1] for x in zipped_sorted]

        sorted_individual = list(''.join(s for s in array_of_chromosomes_one))

        precision = (a_max - a_min) / ((2 ** len_chromosome_one_var) - 1)

        radiuses = [x * precision + a_min for x in array_of_chromosomes_one_decimal_sorted]

        for i in range(1, no_of_variables):
            bounds.append((a_min, radiuses[i - 1] - i * minimal_gap))

        return [sorted_individual, bounds]

    def decode_all_x(individual):
        len_chromosome_one_var = 10
        bound_index = 0
        x = []

        individual = bounds_fn(individual)[0]
        len_chromosome = len(individual)

        bounds = bounds_fn(individual)[1]

        for i in range(0, len_chromosome, len_chromosome_one_var):
            # converts binary to decimal using 2**place_value
            chromosome_string = ''.join((str(xi) for xi in individual[i:i + len_chromosome_one_var]))
            binary_to_decimal = int(chromosome_string, 2)

            lb = bounds[bound_index][0]
            ub = bounds[bound_index][1]
            precision = (ub - lb) / ((2 ** len_chromosome_one_var) - 1)
            decoded = (binary_to_decimal * precision) + lb
            x.append(decoded)
            bound_index += 1

        return x

    def objective_fxn(individual):
        # decoding chromosome to get decoded x in a list
        r_i = decode_all_x(individual)
        Bz = ff.Bz(a_max, a_min, len(r_i), I, spacing, cp, r_i)

        height = 0.015  # [m]
        COV = ff.COV_circ(Bz, a_max, height, spacing)

        obj_function_value = COV
        return [obj_function_value]

    # def cross(ind1, ind2, min_var, max_var):
    #     new_ind1 = [round(tools.cxSimulatedBinaryBounded(ind1[0], ind2[0], low=min_var, up=max_var)[0])]
    #     new_ind2 = [round(tools.cxSimulatedBinaryBounded(ind1[0], ind2[0], low=min_var, up=max_var)[1])]
    #     for i in range(new_ind1[0]):
    #
    #     return tuple(new_ind1, new_ind2)

    def mutate(ind, Indpb):
        p = random.random()
        if p <= 0.5:
            ind += [(i - (i - 1)) * random.randint(0, 1) for i in range(10)]
        else:
            ind = ind[:-10]
        ind = tools.mutFlipBit(ind, indpb=Indpb)
        return ind,

    # registering objetive function with constraint
    toolbox.register("evaluate", objective_fxn)  # privide the objective function here
    # toolbox.decorate("evaluate", tools.DeltaPenalty(check_feasiblity, 1000, penalty_fxn)) # constraint on our
    # objective function

    # registering basic processes using built in functions in DEAP
    toolbox.register("select", tools.selTournament, tournsize=tournSel_k)  # selection strategy
    toolbox.register("mate",
                     tools.cxMessyOnePoint)  # strategy for crossover, this classic two point crossover
    #toolbox.register("mutate", mutate,
    #                Indpb=probability_of_mutation)  # mutation strategy with probability of mutation
    toolbox.register("mutate", tools.mutFlipBit, indpb=probability_of_mutation)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('Min', np.min)
    stats.register('Max', np.max)
    stats.register('Avg', np.mean)
    stats.register('Std', np.std)

    # logbook = tools.Logbook()

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
    # plt.plot(minFitnessValues)

    plt.title("Minimum values of f(x,y) Reached Through Generations", fontsize=20, fontweight='bold')
    plt.xlabel("Generations", fontsize=18, fontweight='bold')
    plt.ylabel("Value of Himmelblau's Function", fontsize=18, fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    hall_of_fame = tools.HallOfFame(1)
    hall_of_fame.update(pop)
    print(decode_all_x(hall_of_fame[0]))
    print(objective_fxn(hall_of_fame[0])[0])

    length = 2*math.pi*np.sum(np.array(decode_all_x(hall_of_fame[0])))
    print(f'Total length = {length} m.')
    #print(bounds_fn(hall_of_fame[0]))
    # plt.show()

    df = pd.DataFrame(decode_all_x(hall_of_fame[0]))
    df.to_excel('hall_of_fame.xlsx')

    return objective_fxn(hall_of_fame[0])


if __name__ == '__main__':
    main()
