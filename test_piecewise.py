import COV, Bz_Field, Resistance, Plot, Field_functions
import see_on_arrays as np
import matplotlib.pyplot as plt


Array_radii = [1, 0.6, 0.4, 0.2]
# coords_rectangle = [[0.05, -0.05], [-0.05, -0.05], [-0.05, 0.05], [0.05, 0.05]]
coords_rectangle = [[0.05, -0.15], [-0.05, -0.15], [-0.05, 0.15], [0.05, 0.15]]
cp = 50
spacing = 1.5
height = 0.015
P = 0.9
I = 1
l = []
for i in range(len(coords_rectangle)):
    l.append(np.sqrt((coords_rectangle[i][0])**2 + (coords_rectangle[i][1])**2))

g = max(l)

Bz_single = Bz_Field.Bz_piecewise_linear_contour_single(coords_rectangle, I, spacing, cp, g, False)
Bz_contour = Bz_Field.Bz_piecewise_linear_contour(Array_radii, coords_rectangle, I, spacing, cp, False)

Plot.plot_2d(Bz_single, height, g, spacing, cp)
plt.show()