import random
import sys

a_max = 0.5  # [m] Max coil radius
a_min = 0.05  # [m] Min coil radius
minimal_gap = 0.002
no_of_variables = 10  # Количество переменных (радиусов витков)
ind = [1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1]


def decode_ind(ind):
    x = []
    len_chromosome = len(ind)
    len_chromosome_one_var = int(len_chromosome / no_of_variables)

    for i in range(0, len_chromosome, len_chromosome_one_var):
        # converts binary to decimial using 2**place_value
        chromosome_string = ''.join((str(xi) for xi in ind[i:i + len_chromosome_one_var]))
        binary_to_decimal = int(chromosome_string, 2)
        x.append(binary_to_decimal)

    return x


def bounds(ind):
    bounds = [(a_min, a_max)]
    len_chromosome = len(ind)
    len_chromosome_one_var = int(len_chromosome / no_of_variables)
    array_of_chromosomes_one = []

    for i in range(0, len_chromosome, len_chromosome_one_var):
        array_of_chromosomes_one.append(''.join(str(xi) for xi in ind[i:i + len_chromosome_one_var]))

    array_of_chromosomes_one_decimal_sorted = list(
        reversed(sorted(list(map(lambda x: int(x, 2), array_of_chromosomes_one)))))

    precision = (a_max - a_min) / ((2 ** len_chromosome_one_var) - 1)

    radiuses = [x * precision + a_min for x in array_of_chromosomes_one_decimal_sorted]

    for i in range(1, no_of_variables):
        bounds.append((a_min, radiuses[i - 1] - i*minimal_gap))

    return bounds


def decode(ind):

