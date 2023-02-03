no_of_variables = 10  # Количество переменных (радиусов витков)
a_max = 0.5  # [m] Max coil radius
a_min = 0.05 # [m] Min coil radius

ind = [1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1]

len_chromosome = len(ind)
len_chromosome_one_var = int(len_chromosome / no_of_variables)
array_of_chromosomes_one = []

for i in range(0, len_chromosome, len_chromosome_one_var):
    array_of_chromosomes_one.append(''.join(str(xi) for xi in ind[i:i + len_chromosome_one_var]))

array_of_chromosomes_one_decimal_sorted = list(reversed(sorted(list(map(lambda x: int(x, 2), array_of_chromosomes_one)))))

precision = (a_max - a_min) / ((2 ** len_chromosome_one_var) - 1)

radiuses = [x * precision + a_min for x in array_of_chromosomes_one_decimal_sorted]
print(radiuses)
