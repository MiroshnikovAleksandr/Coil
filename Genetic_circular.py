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


class Genetic_circular:
    """
    This class describes the genetic algorithm.
    """

    def __init__(self, params):
        self.hall_of_fame = None
        self.logbook = None
        self.pop = None
        self.no_of_generations = params['gen']['no_of_generations']
        self.len_of_turn = params['gen']['length_of_turn']
        self.population_size = params['gen']['population_size']
        self.probability_of_mutation = params['gen']['probability_of_mutation']
        self.tournSel_k = params['gen']['tournSel_k']
        self.CXPB = params['gen']['CXPB']
        self.MUTPB = params['gen']['MUTPB']

        self.figure = 'Circular'
        self.X_side = params['geom']['X_side']
        self.Y_side = params['geom']['Y_side']
        self.a_max = params['geom']['a_max']
        self.a_min = params['geom']['a_min']
        self.I = params['geom']['I']
        self.spacing = params['geom']['spacing']
        self.cp = params['geom']['cp']
        self.minimal_gap = params['geom']['minimal_gap']
        self.height = params['geom']['height']
        self.coords = params['geom']['coords']
        self.calculation_area = params['geom']['calculation_area']
        self.material = params['geom']['material']
        self.freq = params['geom']['freq']

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
                         toolbox.ZeroOrOne, random.randint(50, 5 * self.a_max // self.minimal_gap))
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

    def determine_Bz(self, individual):
        return Bz_Field.Bz_circular_contour\
                (R=self.decode_all_x(individual),
                 I=self.I,
                 spacing=self.spacing,
                 cp=self.cp,
                 height=self.height)

    def determine_COV(self, bz):
        return COV.COV_circle\
                (Bz=bz,
                 max_coil_r=self.a_max,
                 spacing=self.spacing,
                 P=self.calculation_area)

    def objective_fxn(self, individual):
        """
        This is the objective function of the genetic algorithm. It returns the coefficient of variation
        of the magnetic field induced by the coil.
        @param individual: creator.Individual
        @return: list, containing the COV
        """
        bz = self.determine_Bz(individual)
        cov = self.determine_COV(bz)

        obj_function_value = cov
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
        coil = self.decode_all_x(ind)
        if self.figure == 'Circular':
            l = 2 * math.pi * np.sum(np.array(coil))
            return l
        elif self.figure == 'Rectangle':
            l = 2 * (self.X_side + self.Y_side) * np.sum(np.array(coil)) / max(coil)
            return l
        elif self.figure == 'Piecewise':
            coords_ = Radii_in_coords(coil, self.coords)
            l = 0
            coords = []
            for i in range(len(coords_)):
                for j in range(len(coords_[i])):
                    coords.append(coords_[i][j])
            for i in range(len(coords)):
                try:
                    l += np.sqrt((coords[i][0] - coords[i + 1][0]) ** 2 + (coords[i][1] - coords[i + 1][1]) ** 2)
                except IndexError:
                    l += np.sqrt((coords[0][0] - coords[i][0]) ** 2 + (coords[0][1] - coords[i][1]) ** 2)
            return l

    def check_feasibility(self, ind):
        """
        Checks the feasibility of an individual: whether the length of the coil is adequate.
        @param ind: creator.Individual
        @return: boolean
        """
        # if self.length(ind) > 100:
        #     return False
        if len(self.decode_all_x(ind)) < 5:
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
        toolbox.register("evaluate", self.objective_fxn)  # provide the objective function here
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
        return self.decode_all_x(self.hall_of_fame[0])

    def show(self):
        """
        Displays the results (statistics plot, best COV, total length of the best individual).
        @return:
        """
        print(self.hall_of_fame[0])
        print(self.decode_all_x(self.hall_of_fame[0]))
