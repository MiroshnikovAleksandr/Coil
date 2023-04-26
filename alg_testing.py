import matplotlib.pyplot as plt
from DEAP_Field_refactored import Genetic
import tomli
import numpy as np
import scienceplots

n = 2  # number of points in each data array

plt.style.use(['science'])

with open('parameters.toml', 'rb') as toml:
    parameters = tomli.load(toml)


def stop_criteria(GA):
    res = []

    GA.preparation()
    GA.execution()

    for i in range(GA.no_of_generations + 1):
        if len(set(GA.logbook.select('Min')[i:])) == 1:
            res.append(GA.logbook.select('gen')[i])
            break

    return res[0]


CXPB = np.linspace(0.4, 0.6, n)
MUTPB = np.linspace(0.1, 0.2, n)

matrix = []
for cxpb in CXPB:
    parameters['gen']['CXPB'] = cxpb
    string = []
    for mutpb in MUTPB:
        parameters['gen']['MUTPB'] = mutpb
        string.append(stop_criteria(Genetic(parameters)))
    matrix.append(string)

print(np.array(matrix))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Crossover rate')
ax.set_ylabel('Mutation rate')
ax.set_zlabel('Final epoch')

ax.plot_surface(np.outer(CXPB, np.ones(n)),
                np.outer(MUTPB, np.ones(n)).T,
                np.array(matrix),
                cmap='magma')

plt.show()

# print(stop_criteria(Genetic(parameters)))
