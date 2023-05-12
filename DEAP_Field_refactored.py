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
import tomli
# from scoop import futures
# import multiprocessing

seed = 4090899410329119572
#np.random.Random(seed)
random.seed(seed)

with open('parameters.toml', 'rb') as toml:
    parameters = tomli.load(toml)

toolbox = base.Toolbox()  # create toolbox for genetic algorithm

# toolbox.register("map", futures.map)
# pool = multiprocessing.Pool()
# toolbox.register("map", pool.map)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


class Genetic:
    """
    This class describes the genetic algorithm.
    """

    def __init__(self, params):
        self.hall_of_fame = None
        self.logbook = None
        self.pop = None
        self.no_of_generations = 1  # params['gen']['no_of_generations']
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
        """
        Describes and generates basic genetic algorithm objects:
        an individual is a list of random length consisting of 1s and 0s, that encodes
        a sequence of turn radiuses of a spiral coil;
        the population is a set of individuals.
        @return:
        """
        toolbox.register("ZeroOrOne", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.ZeroOrOne, random.randint(250, 5000)//5)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        self.pop = toolbox.population(n=self.population_size)

    def bounds_fn(self, ind):
        """
        Calculates boundaries of coil turn placement based on the diameter of the wire;
        sorts the turns of an individual (coil) in descending order.
        @param ind: creator.Individual
        @return: list, containing the sorted individual and the boundaries
        """
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

    def decode_all_x(self, individual):
        """
        Decodes the individual from 1-0 code to a sequence of radiuses of turns of the coil.
        @param individual: creator.Individual
        @return: list of radiuses
        """
        len_chromosome_one_var = self.len_of_turn
        bound_index = 0
        x = []

        individual = self.bounds_fn(individual)[0]
        len_chromosome = len(individual)

        bounds = self.bounds_fn(individual)[1]

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

    def objective_fxn(self, individual):
        """
        This is the objective function of the genetic algorithm. It returns the coefficient of variation
        of the magnetic field induced by the coil.
        @param individual: creator.Individual
        @return: list, containing the COV
        """
        r_i = self.decode_all_x(individual)
        Bz = ff.Bz(self.a_max, self.a_min, len(r_i), self.I, self.spacing, self.cp, r_i)

        height = 0.015  # [m]
        COV = ff.COV_circ(Bz, self.a_max, height, self.spacing)

        obj_function_value = COV
        return [obj_function_value]

    def mutate(self, ind, Indpb):
        """
        The mutation algorithm.
        @param ind: creator.Individual
        @param Indpb: float
        @return: creator.Individual
        """
        p = random.random()
        if p <= 0.5:
            ind += [(i - (i - 1)) * random.randint(0, 1) for i in range(self.len_of_turn)]
        else:
            del ind[len(ind) - self.len_of_turn::]
        ind = tools.mutFlipBit(ind, indpb=Indpb)
        return ind

    def length(self, ind):
        """
        Calculates the length  of the coil.
        @param ind: creator.Individual
        @return: float
        """
        l = 2 * math.pi * np.sum(np.array(self.decode_all_x(ind)))
        return l

    def check_feasibility(self, ind):
        """
        Checks the feasibility of an individual: whether the length of the coil is adequate.
        @param ind: creator.Individual
        @return: boolean
        """
        if self.length(ind) > 100:
            return False
        else:
            return True

    def execution(self):
        """
        Executes the genetic algorithm with selection, mating and mutation.
        Also stores the best individual from the perspective of its objective function in @hall_of_fame@.
        @return: list, containing the COV of the best individual
        """
        # registering objective function with constraint
        toolbox.register("evaluate", self.objective_fxn)  # privide the objective function here
        toolbox.decorate("evaluate",
                         tools.DeltaPenalty(self.check_feasibility, 1.5))  # constraint on the objective function

        # registering basic processes using built-in functions in DEAP
        toolbox.register("select", tools.selTournament, tournsize=self.tournSel_k)  # selection strategy
        toolbox.register("mate",
                         tools.cxMessyOnePoint)  # strategy for crossover, this classic two point crossover
        toolbox.register("mutate", self.mutate,
                         Indpb=self.probability_of_mutation)  # mutation strategy with probability of mutation

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('Min', np.min)
        stats.register('Max', np.max)
        stats.register('Avg', np.mean)
        stats.register('Std', np.std)

        self.pop, self.logbook = algorithms.eaSimple(self.pop,
                                                     toolbox,
                                                     cxpb=self.CXPB,
                                                     mutpb=self.MUTPB,
                                                     ngen=self.no_of_generations,
                                                     stats=stats,
                                                     verbose=True)

        self.hall_of_fame = tools.HallOfFame(1)
        self.hall_of_fame.update(self.pop)
        return self.objective_fxn(self.hall_of_fame[0])

    def show(self):
        """
        Displays the results (statistics plot, best COV, total length of the best individual).
        @return:
        """
        # using select method in logbook object to extract the argument/key as list
        plt.plot(self.logbook.select('Min'))

        plt.title("Minimum values of f(x,y) Reached Through Generations", fontsize=20, fontweight='bold')
        plt.xlabel("Generations", fontsize=18, fontweight='bold')
        plt.ylabel("Value of Himmelblau's Function", fontsize=18, fontweight='bold')
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')

        print(self.decode_all_x(self.hall_of_fame[0]))
        print(self.objective_fxn(self.hall_of_fame[0])[0])

        print(f'Total length = {self.length(self.hall_of_fame[0])} m.')
        # plt.show()

        # df = pd.DataFrame(self.decode_all_x(hall_of_fame[0]))
        # df.to_excel('hall_of_fame.xlsx')

GA = Genetic(parameters)
GA.preparation()
GA.execution()
GA.show()
# for i in range(50, 101, 10):
#     no_of_generations = i
#     GA = Genetic(parameters)
#     GA.preparation()
#     GA.execution()
#     GA.show()
# plt.show()
