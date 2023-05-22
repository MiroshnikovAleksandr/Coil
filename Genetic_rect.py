from deap import base, algorithms
from deap import creator
from deap import tools
import random
import numpy as np
import Bz_Field
import Bz_Field as Bz
import COV
import Resistance
import tomli
import matplotlib.pyplot as plt

with open('parameters.toml', 'rb') as toml:
    parameters = tomli.load(toml)

toolbox = base.Toolbox()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


class GeneticRectangle:

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

        self.figure = params['geom']['figure']
        self.X_side = params['geom']['X_side']
        self.Y_side = params['geom']['Y_side']
        self.L = min(self.X_side, self.Y_side)
        self.l = self.L / 10
        self.I = params['geom']['I']
        self.spacing = params['geom']['spacing']
        self.cp = params['geom']['cp']
        self.minimal_gap = params['geom']['minimal_gap']
        self.minimal_gap = self.minimal_gap
        self.height = params['geom']['height']
        self.coords = params['geom']['coords']
        self.calculation_area = params['geom']['calculation_area']
        self.material = params['geom']['material']
        self.freq = params['geom']['freq']

    def preparation(self):
        toolbox.register("ZeroOrOne", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.ZeroOrOne, random.randint(50, 5 * self.L // (2 * self.minimal_gap)))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        self.pop = toolbox.population(n=self.population_size)

    def bounds_fn(self, ind):
        ind = [0] * (self.len_of_turn - (len(ind) % self.len_of_turn)) + ind
        Min = self.l / self.L
        Max = self.L / self.L
        bounds = [(Min, Max)]

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

        precision = (Max - Min) / ((2 ** len_chromosome_one_var) - 1)

        coeffs = [x * precision + Min for x in array_of_chromosomes_one_decimal_sorted]

        for i in range(1, len(coeffs)):
            bounds.append((Min, coeffs[i - 1] - 2 * i * self.minimal_gap / self.L))

        return [sorted_individual, bounds]

    def decode_all_x(self, individual):
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
        sides = self.decode_all_x(individual)
        bz = Bz_Field.Bz_square_contour(R=sides, X_side=self.X_side, Y_side=self.Y_side,
                                        I=self.I, spacing=self.spacing, cp=self.cp)
        cov = COV.COV_square(Bz=bz, X_side=self.X_side, Y_side=self.Y_side,
                             height=self.height, spacing=self.spacing, P=self.calculation_area)

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

    def execution(self):
        """
        Executes the genetic algorithm with selection, mating and mutation.
        Also stores the best individual from the perspective of its objective function in @hall_of_fame@.
        @return: list, containing the COV of the best individual
        """
        # registering objective function with constraint
        toolbox.register("evaluate", self.objective_fxn)  # provide the objective function here
        # toolbox.decorate("evaluate",
        #                  tools.DeltaPenalty(self.check_feasibility, 1.5))  # constraint on the objective function

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


GA = GeneticRectangle(parameters)
# GA.decode_all_x([0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1])

GA.preparation()
GA.execution()
GA.show()
