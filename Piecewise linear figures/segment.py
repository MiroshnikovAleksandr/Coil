import Bz_Field
import Plot
import matplotlib.pyplot as plt
import numpy as np


coords =  [[0.1, 0.1], [-0.1, 0.1]]
I = 1
cp = 30
spacing = 1.5
l = np.sqrt((coords[0][0] - coords[1][0])**2 + (coords[0][1] - coords[1][1])**2)
calc_radius = spacing * l
height = 0.03

Bz = Bz_Field.Bz_segment(start_point=coords[0],
                         end_point=coords[1],
                         I=I,
                         cp=cp,
                         calc_radius=calc_radius)

Plot.plot_3d(Bz=Bz,
             height=height,
             a_max=l,
             spacing=spacing,
             cp=cp)
plt.show()