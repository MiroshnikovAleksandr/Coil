# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:33:36 2024

@author: smirp
"""

import numpy as np
import matplotlib.pyplot as plt
import scienceplots

import Bz_Field
import COV
import Plot

plt.style.use(['science','ieee','no-latex', 'high-vis'])
plt.rcParams.update({'figure.dpi': '300'})
#%% Один виток на разных высотах

a = 1
I = 1
cp = 100
spacing = 1
calc_radius = a*spacing
height = 1
r = [a]

Bz = Bz_Field.Bz_circular_single(a, I, cp, calc_radius, height)

fig, ax = plt.subplots()
#heights = np.arange(0.01,0.5,0.1)*2*a
heights = np.array([0.01,0.05,0.1,0.25,0.5])*2*a
for height in heights:
    Bz = Bz_Field.Bz_circular_single(a, I, cp, calc_radius, height)
    Bz = Bz/np.max(Bz)
    #Plot.plot_2d(Bz/np.max(Bz), height, a, spacing, cp, ax, 1)
    calc_radius = a * spacing  # Calculation domain length
    x = np.linspace(-calc_radius, calc_radius, cp)
    ax.plot(x/a, Bz[cp // 2, :], label=f'$h = {round(height/2/a,2)},D$')
    ax.set_xlabel('$r/D$')
    ax.set_ylabel('Normalized $B_z$')
    ax.set_title('Bz field profile for different heights')
    ax.grid()
    plt.legend(bbox_to_anchor=(1.01, 1))
    
#%% Эквидистантная на разных высотах

a = 1
I = 1
cp = 100
spacing = 1
calc_radius = a*spacing
r = np.linspace(0.1,1,10)


Bz = Bz_Field.Bz_circular_contour(r, I, spacing, cp, height)

fig, ax = plt.subplots()
heights = np.array([0.01,0.05,0.1,0.25,0.5])*2*a
for height in heights:
    Bz = Bz_Field.Bz_circular_contour(r, I, spacing, cp, height)
    Bz = Bz/np.max(Bz)
    #Plot.plot_2d(Bz/np.max(Bz), height, a, spacing, cp, ax, 1)
    calc_radius = a * spacing  # Calculation domain length
    x = np.linspace(-calc_radius, calc_radius, cp)
    ax.plot(x/a, Bz[cp // 2, :], label=f'$h = {round(height/2/a,2)}\,D$')
    ax.set_xlabel('$r/D$')
    ax.set_ylabel('Normalized $B_z$')
    ax.set_title('Bz field profile for different heights')
    ax.grid()
plt.legend(bbox_to_anchor=(1.25, 1))

#%% График для одиночного витка


a = 1
I = 1
cp = 100
spacing = 1
calc_radius = a*spacing
r = [a]

heights = np.linspace(0.01,1,101)*2*a
#ROI = np.arange(0.1,0.5,0.1)
ROI = np.linspace(0.1,0.9,101)

cov = np.zeros([len(heights),len(ROI)])

X, Y = np.meshgrid(heights,ROI)
i = 0
for height in heights:
    j = 0
    for P in ROI:
        Bz = Bz_Field.Bz_circular_single(a, I, cp, calc_radius, height)
        cov[i][j] = COV.COV_circle(Bz, a, spacing, P)
        j += 1
    i += 1
    
fig, ax = plt.subplots()
ax.set_ylabel('$h/D$')
ax.set_xlabel('$S_{ROI}/S_{max}$')
cov = np.transpose(cov)
#fig = ax.imshow(cov,cmap='hot',vmin = 0, vmax = 0.5)
fig = ax.pcolormesh(Y,X/2/a,cov*100, vmin = 0,vmax = 60, cmap = 'hot')

plt.colorbar(fig, label='$COV$, %')

cov_single = cov

#%% График для многовитковой эквидистантной

a = 1
I = 1
cp = 100
spacing = 1
calc_radius = a*spacing
r = np.linspace(0.1,1,10)*a

heights = np.linspace(0.01,1,101)*2*a
#ROI = np.arange(0.1,0.5,0.1)
ROI = np.linspace(0.1,0.9,101)

cov = np.zeros([len(heights),len(ROI)])

X, Y = np.meshgrid(heights,ROI)
i = 0
for height in heights:
    j = 0
    for P in ROI:
        Bz = Bz_Field.Bz_circular_contour(r, I, spacing, cp, height)
        cov[i][j] = COV.COV_circle(Bz, a, spacing, P)
        j += 1
    i += 1

fig, ax = plt.subplots()
ax.set_ylabel('$h/D$')
ax.set_xlabel('$S_{ROI}/S_{max}$')
cov = np.transpose(cov)
#fig = ax.imshow(cov,cmap='hot',vmin = 0, vmax = 0.5)
fig = ax.pcolormesh(Y,X/2/a,cov*100, vmin = 0, vmax = 90, cmap = 'hot')

plt.colorbar(fig, label='$COV$, %')


#%% Оптимизированная на ROI 0.4
# радиус 12.5 см, высота 3 см (0.12)

a = 0.125
I = 1
cp = 100
spacing = 1
calc_radius = a*spacing

file_path = 'par0.4_5.npy'
data = np.load(file_path)
r = data[3::]


fig, ax = plt.subplots()
heights = np.array([0.01,0.05,0.1,0.25,0.5])*2*a
for height in heights:
    Bz = Bz_Field.Bz_circular_contour(r, I, spacing, cp, height)
    Bz = Bz/np.max(Bz)
    #Plot.plot_2d(Bz/np.max(Bz), height, a, spacing, cp, ax, 1)
    calc_radius = a * spacing  # Calculation domain length
    x = np.linspace(-calc_radius, calc_radius, cp)
    ax.plot(x/a, Bz[cp // 2, :], label=f'$h = {round(height/a/2,2)}\,D$')
    ax.set_xlabel('$r/D$')
    ax.set_ylabel('Normalized $B_z$')
    ax.set_title('Bz field profile for different heights')
    ax.grid()
plt.legend(bbox_to_anchor=(1.27, 1), loc='upper right')



#%% Оптимизированная на ROI 0.4
heights = np.linspace(0.01,1,101)*a*2
ROI = np.linspace(0.1,0.9,101)

cov = np.zeros([len(heights),len(ROI)])

X, Y = np.meshgrid(heights,ROI)
i = 0
for height in heights:
    j = 0
    for P in ROI:
        Bz = Bz_Field.Bz_circular_contour(r, I, spacing, cp, height)
        cov[i][j] = COV.COV_circle(Bz, a, spacing, P)
        j += 1
    i += 1

fig, ax = plt.subplots()
ax.set_ylabel('$h/D$')
ax.set_xlabel('$S_{ROI}/S_{max}$')
cov = np.transpose(cov)
#fig = ax.imshow(cov,cmap='hot',vmin = 0, vmax = 0.5)
fig = ax.pcolormesh(Y,X/2/a,cov*100, vmin = 0, cmap = 'hot')

plt.colorbar(fig, label='$COV$, %')

cov_opt = cov
#%% 
fig, ax = plt.subplots()
fig = ax.pcolormesh(Y,X/2/a,cov_opt-cov_single, cmap = 'hot')
plt.colorbar(fig, label='COV difference')

