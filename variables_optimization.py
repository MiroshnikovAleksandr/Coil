import numpy as np
from scipy.optimize import minimize, Bounds
import DEAP_Field_newnew as DF


def main(x):
    return DF.main(round(x[0]))


x0 = np.array([10.0])
bounds = Bounds([1, 225])
res = minimize(main, x0, method='trust-constr', options={'verbose': 1}, bounds=bounds)
#res = minimize(main, x0, method='nelder-mead',
#               options={'xatol': 1e-8, 'disp': True})

print(res)
