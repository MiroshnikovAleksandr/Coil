import COV, Bz_Field, Resistance, Plot, Field_functions
import see_on_arrays as np
import matplotlib.pyplot as plt


One_radius = 0.5
Array_radii = [1, 0.6, 0.4, 0.2]
r_max = max(Array_radii)
I = 1
cp = 50
spacing = 1.5
height = 0.015
P = 0.9

Bz_Nikita_array = Bz_Field.Bz_circular_contour(R=Array_radii,
                                         I=I,
                                         spacing=spacing,
                                         cp=cp)
Bz_Nikita_one =  Bz_Field.Bz_circular_single(r_max=One_radius,
                                             a=One_radius,
                                             I=I,
                                             spacing=spacing,
                                             cp=cp)
Bz_Pavel_array = Field_functions.Bz(a_max=r_max,
                                    a_min=min(Array_radii),
                                    n=len(Array_radii),
                                    I=I,
                                    spacing=spacing,
                                    cp=cp,
                                    r_i=Array_radii)
Bz_Pavel_one = Field_functions.Bz_single(a=One_radius,
                                         I=I,
                                         spacing=spacing,
                                         cp=cp)

COV_Nikita = COV.COV_circle(Bz=Bz_Nikita_array,
                            max_coil_r=r_max,
                            height=height,
                            spacing=spacing,
                            P=P)
COV_Pavel = Field_functions.COV_circ(Bz=Bz_Pavel_array,
                                     max_coil_r=r_max,
                                     height=height,
                                     spacing=spacing)

difference = Bz_Pavel_one - Bz_Nikita_one

print(difference)
print('...')
print(COV_Pavel, COV_Nikita)
Array_numpy = np.array([1, 2, 3, 4, 5])
Array_python = [1, 2, 3, 4, 5]
print(np.amax(Array_python), np.amax(Array_numpy))
# if Bz_Pavel_one - Bz_Nikita_one != np.zeros((cp, cp, cp)):
#     print('Не равны!!')

# Plot.plot_2d(Bz=Bz_Nikita_one,
#              height=height,
#              a_max=One_radius,
#              spacing=spacing,
#              cp=cp)
# plt.show()
# Plot.plot_2d(Bz=Bz_Pavel_one,
#              height=height,
#              a_max=One_radius,
#              spacing=spacing,
#              cp=cp)
# plt.show()

