import matplotlib.pyplot as plt
from DEAP_Field_refactored import Genetic
import tomli
import numpy as np
import scienceplots

n = 2  # number of points in each data array

# plt.style.use(['science', 'ieee', 'no-latex'])

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


CXPB = np.linspace(0.2, 1.0, n)
MUTPB = np.linspace(0.02, 1.0, n)

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

print(f"The minimal Final epoch value of {np.min(matrix)} is achieved at indices ({np.argmin(matrix)//n}, "
      f"{np.argmin(matrix)%n}).")
print(f'CXPB is {CXPB[np.argmin(matrix)//n]}, MUTPB is {MUTPB[np.argmin(matrix)%n]}.')
plt.show()
plt.savefig('pic.jpg')
# print(stop_criteria(Genetic(parameters)))
