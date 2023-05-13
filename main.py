# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 13:56:41 2021

@author: Pavel
"""
import Bz_Field as Bz
import COV
from DEAP_Field_refactored import Genetic
import numpy as np
import matplotlib.pyplot as plt
import Plot
import Resistance
import tomli
from turns_splitter import split

with open('parameters.toml', 'rb') as toml:
    parameters = tomli.load(toml)

GA = Genetic(parameters)
GA.preparation()

flat_radii_array = GA.execution()
radii_array = split(flat_radii_array, GA.freq)

if GA.figure == 'Circular':
    length = Resistance.length_circular_coils(coils=radii_array)
    Plot.plot_coil(a_max=GA.a_max, spacing=GA.spacing, R=flat_radii_array)
elif GA.figure == 'Rectangle':
    length = Resistance.length_square_coils(coils=radii_array)
    Plot.plot_square_coil(m_max=GA.X_side, n_max=GA.Y_side, spacing=GA.spacing, R=flat_radii_array)
elif GA.figure == 'Piecewise':
    length = Resistance.length_piecewise_linear_coils(coils=radii_array)
    Plot.plot_piecewise_linear_coil(coords_max=GA.coords, spacing=GA.spacing, R=flat_radii_array)

resistance = Resistance.resistance_contour(l=length, material=GA.material, d=GA.minimal_gap, nu=GA.freq)

GA.show()
print(radii_array)
print(resistance)

# # plt.style.use(['science','ieee'])
#
# # Geometrical parameters.toml
# a_max = 1  # [m] Max coil radius
# a_min = 0.1  # [m] Min coil radius
# n = 10  # Number of turns
# I = 1  # [A] Current
# spacing = 1.3  # spacing for calculation domain
# cp = 100  # Calculation domain points
#
# # r_i = [0.19962854349951126, 0.19327272727272726, 0.19582209188660804, 0.1339706744868035, 0.19148973607038122, 0.07749755620723363, 0.18653958944281526, 0.1641466275659824, 0.0261544477028348, 0.1803440860215054]
#
# # r_i = [0.19795698924731184, 0.19218181818181818, 0.1904848484848485, 0.19052003910068427, 0.18740762463343108, 0.18102639296187684, 0.04991006842619746, 0.13516715542521995, 0.1056950146627566, 0.1604731182795699]
# # r_i = [0.19962854349951126, 0.19763636363636364, 0.11932160312805475, 0.194, 0.17669208211143694, 0.07101661779081134, 0.022162267839687194, 0.185841642228739, 0.1821466275659824, 0.15580645161290324]
# # r_i = np.linspace(a_max,a_min,n) # creating turns
# # #r_i = [1]
# # r_i = [0.49208211143695013, 0.49756207233626587, 0.4759452590420332, 0.4471260997067449, 0.4902717497556207, 0.49, 0.39808797653958944, 0.24690322580645163, 0.05381818181818182, 0.33968914956011725]
# r_i = [0.206158357771261, 0.5, 0.49824046920821113, 0.3675953079178885, 0.41686217008797655, 0.48944281524926686,
#        0.4806451612903226, 0.3073313782991202, 0.49032258064516127, 0.48372434017595306]
# Bz = ff.Bz(a_max, a_min, n, I, spacing, cp, r_i)
#
# height = 0.1  # [m]
#
# ff.plot_2d(Bz, height, a_max, spacing, cp)
# ff.plot_3d(Bz, height, a_max, spacing, cp)
# ff.plot_coil(a_max, spacing, r_i)
#
# COV = ff.COV_circ(Bz, a_max, height, spacing)
# print(COV)
# ff.plot_coil
