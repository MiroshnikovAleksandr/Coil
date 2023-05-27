from deap import base, algorithms
from deap import creator
from deap import tools

import random
import sys
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# turns = [0.49986530221352876, 0.4957422334496335, 0.4913159397152264, 0.48650341844713946, 0.46856754015057495,
#          0.4194663585872939, 0.35410272529314335, 0.28729411095991536, 0.15641894155542901, 0.07312009035069601]
# turns = [0.5, 0.498, 0.496, 0.494, 0.4349677419354839, 0.3703329864724246, 0.2621789802289282, 0.05]
# turns = [0.5, 0.498, 0.496, 0.4653548387096774, 0.409681581685744, 0.33337148803329864, 0.19536108220603537]
# turns = [0.4989791366925755, 0.49465199892091494, 0.4893070502223121, 0.4851274611458574, 0.4828771291165515,
#          0.45175410160866813, 0.3991732570934159, 0.3442020278190352, 0.24223506510855264, 0.07677023841421055]
# turns = [0.5, 0.498, 0.496, 0.494, 0.492, 0.49, 0.47387096774193543, 0.4442913631633715, 0.4027138397502601,
#          0.3636233090530697, 0.31518210197710717, 0.2310863683662851, 0.05]
# turns = [0.5, 0.498, 0.496, 0.494, 0.492, 0.47580645161290325, 0.4325015608740895, 0.40447242455775234,
#          0.35228511966701354, 0.2951259105098855, 0.19174817898022894, 0.05]
turns = [0.08000000000000002, 0.07700000000000001, 0.07400000000000001, 0.010032258064516129, 0.007687825182101977]

# freq = 6.78  # [MHz]
# U = 1  # [V]
# ro = 0.05  # [Ohm/m]
# a_max = 0.5  # [m] Max coil radius
# a_min = 0.05  # [m] Min coil radius
# cp = 30  # Calculation domain points
# spacing = 1.5  # spacing for calculation domain
# height = 0.015  # [m]
c = 299_792_458  # [m/s]


def split(turns, freq):
    """
    Function that breaks up the original array of turn radiuses into a 2-dimensional array of multiple separate coils,
    that are to be connected as parallel circuit. This is required to match the maximum possible length of the wires
    which is for magnetostatic approximation to work.
    @param turns: initial array of radiuses
    @param freq: frequency of the EM wave, is used to determine the maximum possible wire length
    @return: the broken up array of radiuses
    """
    # This code works best if the :turns: array is sorted, which it is.
    # freq = freq * 1_000_000
    length_of_wave = c / freq
    max_length = length_of_wave / 6
    n = math.ceil((sum(turns) * 2 * math.pi) / max_length)

    res = []
    for i in range(n):
        res.append([])

    for turn in turns:
        res[0].append(turn)
        res = sorted(res, key=lambda x: sum(x))

    return res

# # Magnetic field and COV reevaluation
# coils = normal_solution(turns, freq)
# Bz_total = np.zeros((cp, cp, cp))
# I_total = 0
# for coil in coils:
#     R = 2 * math.pi * sum(coil) * ro
#     I = U / R
#     I_total += I
#     Bz = ff.Bz(a_max, a_min, len(coil), I, spacing, cp, coil)
#     Bz_total += Bz
#
# COV_reevaluated = ff.COV_circ(Bz_total, a_max, height, spacing)
#
# Bz_old = ff.Bz(a_max, a_min, len(turns), I_total, spacing, cp, turns)
# COV_old = ff.COV_circ(Bz_old, a_max, height, spacing)
# print(f'The reevaluated COV is {COV_reevaluated}\nThe original COV is {COV_old}')
