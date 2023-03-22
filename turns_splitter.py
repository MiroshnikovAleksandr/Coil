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

turns = [0.49986530221352876, 0.4957422334496335, 0.4913159397152264, 0.48650341844713946, 0.46856754015057495,
         0.4194663585872939, 0.35410272529314335, 0.28729411095991536, 0.15641894155542901, 0.07312009035069601]
turns = [0.5, 0.498, 0.496, 0.494, 0.4349677419354839, 0.3703329864724246, 0.2621789802289282, 0.05]
turns = [0.5, 0.498, 0.496, 0.4653548387096774, 0.409681581685744, 0.33337148803329864, 0.19536108220603537]
turns = [0.4989791366925755, 0.49465199892091494, 0.4893070502223121, 0.4851274611458574, 0.4828771291165515, 0.45175410160866813, 0.3991732570934159, 0.3442020278190352, 0.24223506510855264, 0.07677023841421055]
turns = [0.5, 0.498, 0.496, 0.494, 0.492, 0.49, 0.47387096774193543, 0.4442913631633715, 0.4027138397502601, 0.3636233090530697, 0.31518210197710717, 0.2310863683662851, 0.05]

freq = 6.78      # [MHz]
c = 299_792_458  # [m/s]
U = 1            # [V]
ro = 0.05        # [Ohm/m]
a_max = 0.5      # [m] Max coil radius
a_min = 0.05     # [m] Min coil radius
cp = 30          # Calculation domain points
spacing = 1.5    # spacing for calculation domain
height = 0.015   # [m]


def normal_solution(turns, freq):

    # This code works best if the :turns: array is sorted, which it is.
    freq = freq*1_000_000
    length_of_wave = c/freq
    max_length = length_of_wave / 6
    n = math.ceil((sum(turns)*2*math.pi) / max_length)
    # print(n)

    res = []
    for i in range(n):
        res.append([])

    for turn in turns:
        res[0].append(turn)
        res = sorted(res, key=lambda x: sum(x))

    # for i in range(n):
    #     print(sum(res[i]) * 2 * math.pi)
    # print(sum(turns) * 2 * math.pi)
    # print(res)
    return res


coils = normal_solution(turns, freq)
Bz_total = np.zeros((cp, cp, cp))
for coil in coils:
    R = 2 * math.pi * sum(coil) * ro
    I = U/R
    Bz = ff.Bz(a_max, a_min, len(coil), I, spacing, cp, coil)
    Bz_total += Bz

COV_reavaluated = ff.COV_circ(Bz_total, a_max, height, spacing)

Bz_old = ff.Bz(a_max, a_min, len(turns), 1, spacing, cp, turns)
COV_old = ff.COV_circ(Bz_old, a_max, height, spacing)
print(f'The reavaluated COV is {COV_reavaluated}\nThe original COV is {COV_old}')
