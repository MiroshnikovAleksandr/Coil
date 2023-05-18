import os
os.system('cmd /c "streamlit run website.py"')
# from random import randint
#
# numbers = []
# for i in range(1000):
#     numbers.append(randint(0, 1))
#
#
# class A:
#     def __init__(self):
#         self.len_of_turn = 5
#         self.a_max = 1
#         self.a_min = 0.1
#         self.minimal_gap = 0.002
#
#     def bounds_fn(self, ind):
#         """
#         Calculates boundaries of coil turn placement based on the diameter of the wire;
#         sorts the turns of an individual (coil) in descending order.
#         @param ind: creator.Individual
#         @return: list, containing the sorted individual and the boundaries
#         """
#         ind = [0] * (self.len_of_turn - (len(ind) % self.len_of_turn)) + ind
#
#         bounds = [(self.a_min, self.a_max)]
#         len_chromosome = len(ind)
#         len_chromosome_one_var = self.len_of_turn
#         no_of_variables = len_chromosome // len_chromosome_one_var
#         array_of_chromosomes_one = []
#
#         for i in range(0, len_chromosome, len_chromosome_one_var):
#             array_of_chromosomes_one.append(''.join(str(xi) for xi in ind[i:i + len_chromosome_one_var]))
#
#         array_of_chromosomes_one_decimal_sorted = list(map(lambda x: int(x, 2), array_of_chromosomes_one))
#
#         zipped = zip(array_of_chromosomes_one, array_of_chromosomes_one_decimal_sorted)
#         zipped_sorted = sorted(zipped, key=lambda tup: tup[1], reverse=True)
#
#         array_of_chromosomes_one = [x[0] for x in zipped_sorted]
#         array_of_chromosomes_one_decimal_sorted = [x[1] for x in zipped_sorted]
#
#         sorted_individual = list(''.join(s for s in array_of_chromosomes_one))
#
#         precision = (self.a_max - self.a_min) / ((2 ** len_chromosome_one_var) - 1)
#
#         radiuses = [x * precision + self.a_min for x in array_of_chromosomes_one_decimal_sorted]
#
#         for i in range(1, no_of_variables):
#             bounds.append((self.a_min, radiuses[i - 1] - i * self.minimal_gap))
#
#         return [sorted_individual, bounds]
#
#     def decode_all_x(self, individual):
#         """
#         Decodes the individual from 1-0 code to a sequence of radiuses of turns of the coil.
#         @param individual: creator.Individual
#         @return: list of radiuses
#         """
#         len_chromosome_one_var = self.len_of_turn
#         bound_index = 0
#         x = []
#
#         individual = self.bounds_fn(individual)[0]
#         len_chromosome = len(individual)
#
#         bounds = self.bounds_fn(individual)[1]
#
#         for i in range(0, len_chromosome, len_chromosome_one_var):
#             # converts binary to decimal using 2**place_value
#             chromosome_string = ''.join((str(xi) for xi in individual[i:i + len_chromosome_one_var]))
#             binary_to_decimal = int(chromosome_string, 2)
#
#             lb = bounds[bound_index][0]
#             ub = bounds[bound_index][1]
#             precision = (ub - lb) / ((2 ** len_chromosome_one_var) - 1)
#             decoded = (binary_to_decimal * precision) + lb
#             x.append(decoded)
#             bound_index += 1
#
#         return x
#
#
# ex = A()
# print(numbers)
# print(ex.decode_all_x(numbers))
# for x in ex.decode_all_x(numbers):
#     if x < 0: print('False')
