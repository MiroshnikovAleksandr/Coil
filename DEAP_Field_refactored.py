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
import tomllib

with open('parameters.toml', 'rb') as toml:
    parameters = tomllib.load(toml)


class Genetic:

    def __init__(self, params):
        self.toolbox = None
        self.logbook = None
        self.pop = None
        self.no_of_generations = params['gen']['no_of_generations']
        self.len_of_turn = params['gen']['length_of_turn']
        self.population_size = params['gen']['population_size']
        self.probability_of_mutation = params['gen']['probability_of_mutation']
        self.tournSel_k = params['gen']['tournSel_k']
        self.CXPB = params['gen']['CXPB']
        self.MUTPB = params['gen']['MUTPB']
        self.a_max = params['geom']['a_max']
        self.a_min = params['geom']['a_min']
        self.I = params['geom']['I']
        self.spacing = params['geom']['spacing']
        self.cp = params['geom']['cp']
        self.minimal_gap = params['geom']['minimal_gap']

    def preparation(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("ZeroOrOne", random.randint, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.ZeroOrOne, random.randint(50, 1000))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.pop = self.toolbox.population(n=self.population_size)

    def execution(self):
        def bounds_fn(ind: creator.Individual):
            ind = [0] * (self.len_of_turn - (len(ind) % self.len_of_turn)) + ind

            bounds = [(self.a_min, self.a_max)]
            len_chromosome = len(ind)
            len_chromosome_one_var = self.len_of_turn
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

            precision = (self.a_max - self.a_min) / ((2 ** len_chromosome_one_var) - 1)

            radiuses = [x * precision + self.a_min for x in array_of_chromosomes_one_decimal_sorted]

            for i in range(1, no_of_variables):
                bounds.append((self.a_min, radiuses[i - 1] - i * self.minimal_gap))

            return [sorted_individual, bounds]

        def decode_all_x(individual: creator.Individual):
            len_chromosome_one_var = self.len_of_turn
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

        def objective_fxn(individual: creator.Individual):
            # decoding chromosome to get decoded x in a list
            r_i = decode_all_x(individual)
            Bz = ff.Bz(self.a_max, self.a_min, len(r_i), self.I, self.spacing, self.cp, r_i)

            height = 0.015  # [m]
            COV = ff.COV_circ(Bz, self.a_max, height, self.spacing)

            obj_function_value = COV
            return [obj_function_value]

        def length(ind: creator.Individual):
            l = 2 * math.pi * np.sum(np.array(decode_all_x(ind)))
            return l

        def check_feasibility(ind: creator.Individual):
            if length(ind) > 100:
                return False
            else:
                return True

        def mutate(ind: creator.Individual, Indpb: float):
            p = random.random()
            if p <= 0.5:
                ind += [(i - (i - 1)) * random.randint(0, 1) for i in range(self.len_of_turn)]
            else:
                del ind[len(ind) - self.len_of_turn::]
            ind = tools.mutFlipBit(ind, indpb=Indpb)
            return ind

        # registering objetive function with constraint
        self.toolbox.register("evaluate", objective_fxn)  # privide the objective function here
        self.toolbox.decorate("evaluate",
                              tools.DeltaPenalty(check_feasibility, 1.5))  # , penalty_fxn))  # constraint on our
        # objective function

        # registering basic processes using built-in functions in DEAP
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournSel_k)  # selection strategy
        self.toolbox.register("mate",
                         tools.cxMessyOnePoint)  # strategy for crossover, this classic two point crossover
        self.toolbox.register("mutate", mutate,
                         Indpb=self.probability_of_mutation)  # mutation strategy with probability of mutation

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('Min', np.min)
        stats.register('Max', np.max)
        stats.register('Avg', np.mean)
        stats.register('Std', np.std)

        self.pop, self.logbook = algorithms.eaSimple(self.pop,
                                                     self.toolbox,
                                                     cxpb=self.CXPB,
                                                     mutpb=self.MUTPB,
                                                     ngen=self.no_of_generations,
                                                     stats=stats,
                                                     verbose=True)

    def show(self):
        # using select method in logbook object to extract the argument/key as list
        plt.plot(self.logbook.select('Min'))

        plt.title("Minimum values of f(x,y) Reached Through Generations", fontsize=20, fontweight='bold')
        plt.xlabel("Generations", fontsize=18, fontweight='bold')
        plt.ylabel("Value of Himmelblau's Function", fontsize=18, fontweight='bold')
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')

        hall_of_fame = tools.HallOfFame(1)
        hall_of_fame.update(self.pop)
        print(decode_all_x(hall_of_fame[0]))
        print(execution().objective_fxn(hall_of_fame[0])[0])

        print(f'Total length = {length(hall_of_fame[0])} m.')
        # print(bounds_fn(hall_of_fame[0]))
        plt.show()

        # df = pd.DataFrame(decode_all_x(hall_of_fame[0]))
        # df.to_excel('hall_of_fame.xlsx')

        return objective_fxn(hall_of_fame[0])


if __name__ == '__main__':
    GA = Genetic(parameters)
    GA.preparation()
    GA.execution()
    # GA.show()
