import os

import streamlit as st
import pandas as pd
import numpy as np
import macros
import Field_functions as ff
import tomli
import Resistance
from DEAP_Field_refactored import Genetic
from turns_splitter import split
import Plot
import Bz_Field
from Genetic_rect import GeneticRectangle
from Genetic_piece import Genetic_piecewise

radius = 0.0
min_radius = 0.0
h = 0.0
length = 0.0
width = 0.0
frequency = 0.0
I = 0.0
do = False
max_of_generation = 0.0
population_size = 0.0
probability_of_mutation = 0.0
tournSel_k = 0.0
CXPB = 0.0

st.set_page_config(layout="wide")


def changeSize(var, size):
    if size == 'м' or size == 'А' or size == 'МГц':
        return var
    if size == 'см':
        return var / 100
    if size == 'мм':
        return var / 1000
    else:
        return var


with open('parameters.toml', 'rb') as toml:
    parameters = tomli.load(toml)

cord = parameters['geom']['coords']
start_n = len(cord)

with st.sidebar:
    st.title("Ввод параметров")
    form = st.radio("Форма катушки", ['круглая', 'прямоугольная', 'кусочно-линейная'])
    st.header("Геометрически параметры")
    h = 0
    variable, size = st.columns([3, 1])

    if form == 'круглая':
        with variable:
            radius = st.number_input('Радиус', value=parameters['geom']['a_max'])
            min_radius = st.number_input('Минимальный радиус', value=parameters['geom']['a_min'])
            h = st.number_input('Высота', value=parameters['geom']['height'])
        with size:
            size_r = st.selectbox("", ('м', 'см', 'мм'))
            size_min = st.selectbox(" ", ('м', 'см', 'мм'))
            size_h = st.selectbox("  ", ('м', 'см', 'мм'))

    if form == 'прямоугольная':
        with variable:
            length = st.number_input('Длина, м', value=parameters['geom']['X_side'])
            width = st.number_input('Ширина, м', value=parameters['geom']['Y_side'])
            h = st.number_input('Высота, м', value=parameters['geom']['height'])
        with size:
            size_l = st.selectbox("", ('м', 'см', 'мм'))
            size_w = st.selectbox(" ", ('м', 'см', 'мм'))
            size_h = st.selectbox("  ", ('м', 'см', 'мм'))

    if form == 'кусочно-линейная':
        variable, size = st.columns([3, 1])
        with variable:
            h = st.number_input('Высота, м', value=parameters['geom']['height'])
        with size:
            size_h = st.selectbox("  ", ('м', 'см', 'мм'))

        n = st.number_input('Количество точек', value=start_n)

        col1, col2 = st.columns(2)
        x = [0] * n
        y = [0] * n

        if n - start_n > 0:
            cord = cord + [0] * (n - start_n)
            for i in range(start_n, n):
                cord[i] = [0, 0]

        with col1:
            st.header("x, м")
            for i in range(0, n):
                x[i] = st.number_input("x" + str(i), value=cord[i][0])
        with col2:
            st.header("y, м")
            for i in range(0, n):
                y[i] = st.number_input("y" + str(i), value=cord[i][1])

        cord = [0] * n
        for i in range(0, n):
            cord[i] = [x[i], y[i]]

    st.header("Электрические параметры")

    with variable:
        frequency = st.number_input('Частота, [МГц]', value=(parameters['geom']['freq'] / (10 ** 6)))
        I = st.number_input('Сила тока [A]', value=parameters['geom']['I'])
    with size:
        size_freq = st.selectbox('    ', ('МГц', ' '))
        size_I = st.selectbox('      ', ('А', ' '))

    with st.expander("Параметры генетического алгоритма"):
        max_of_generation = st.number_input("Максимально количество поколений",
                                            value=parameters['gen']['no_of_generations'])
        population_size = st.number_input("Kоличество индивидуумов в популяции",
                                          value=parameters['gen']['population_size'])
        probability_of_mutation = st.number_input("Вероятность мутации индивидуума",
                                                  value=parameters['gen']['probability_of_mutation'])
        tournSel_k = st.number_input("Количество особей, участвующих в соревновании для скрещивания",
                                     value=parameters['gen']['tournSel_k'])
        CXPB = st.number_input("вероятность скрещивания", value=parameters['gen']['CXPB'])

    do = st.button('Посчитать')

if do:
    if frequency > 0.0:
        frequency = changeSize(frequency, size_freq)
        parameters['geom']['freq'] = frequency * (10 ** 6)
    if I > 0.0:
        I = changeSize(I, size_I)
        parameters['geom']['I'] = I
    if radius > 0.0:
        radius = changeSize(radius, size_r)
        parameters['geom']['a_max'] = radius
    if min_radius > 0.0:
        min_radius = changeSize(min_radius, size_min)
        parameters['geom']['a_min'] = min_radius
    if length > 0.0:
        length = changeSize(length, size_l)
        parameters['geom']['X_side'] = length
    if width > 0.0:
        width = changeSize(width, size_w)
        parameters['geom']['Y_side'] = width
    if h > 0.0:
        h = changeSize(h, size)
        parameters['geom']['height'] = h
    if form == 'круглая':
        parameters['geom']['figure'] = 'Circular'
    elif form == 'прямоугольная':
        parameters['geom']['figure'] = 'Rectangle'
    elif form == 'кусочно-линейная':
        parameters['geom']['figure'] = 'Piecewise'
    if max_of_generation > 0.0:
        parameters['gen']['no_of_generations'] = max_of_generation
    if population_size > 0.0:
        parameters['gen']['population_size'] = population_size
    if probability_of_mutation > 0.0:
        parameters['gen']['probability_of_mutation'] > 0.0
    if tournSel_k > 0.0:
        parameters['gen']['tournSel_k'] = tournSel_k
    if CXPB > 0.0:
        parameters['gen']['CXPB'] = CXPB

    parameters['geom']['coords'] = cord

    if form == 'прямоугольная':
        GA = GeneticRectangle(parameters)
    elif form == 'кусочно-линейная':
        GA = Genetic_piecewise(parameters)
    else:
        GA = Genetic(parameters)
    # GA = Genetic(parameters)
    GA.preparation()
    if form == 'кусочно-линейная':
        GA.minimal_side()
    flat_radii_array = GA.execution()
    radii_array = split(flat_radii_array, GA.freq)
    Magnetic_field = GA.determine_Bz(GA.hall_of_fame[0])
    final_COV = GA.determine_COV(Magnetic_field)
    length = GA.length(GA.hall_of_fame[0])
    GA.show()

    if GA.figure == 'Circular':
        lengths = Resistance.length_circular_coils(coils=radii_array)
        coil_pic = Plot.plot_coil(a_max=GA.a_max, spacing=GA.spacing, R=flat_radii_array)
        field_pic_3d = Plot.plot_3d(Bz=Magnetic_field,
                                    height=GA.height, a_max=GA.a_max,
                                    spacing=GA.spacing, cp=GA.cp)
        macros = macros.create_circular_macros(radii_array)
    elif GA.figure == 'Rectangle':
        lengths = Resistance.length_square_coils(coils=Bz_Field.Radii_in_sides_square(radii_array,
                                                                                      X_side=GA.X_side,
                                                                                      Y_side=GA.Y_side,
                                                                                      split=True))
        coil_pic = Plot.plot_square_coil(m_max=GA.X_side, n_max=GA.Y_side, spacing=GA.spacing, R=flat_radii_array)
        field_pic_3d = Plot.plot_3d(Bz=Magnetic_field,
                                    height=GA.height, a_max=0.5 * max(GA.X_side, GA.Y_side),
                                    spacing=GA.spacing, cp=GA.cp)
        macros = macros.create_rectangular_macros(radii_array)
    elif GA.figure == 'Piecewise':
        l = []
        for i in range(len(GA.coords)):
            l.append(np.sqrt((GA.coords[i][0]) ** 2 + (GA.coords[i][1]) ** 2))

        lengths = Resistance.length_piecewise_linear_coils(coils=Bz_Field.Radii_in_coords(radii_array,
                                                                                          coords_max=GA.coords,
                                                                                          split=True))
        coil_pic = Plot.plot_piecewise_linear_coil(coords_max=GA.coords, spacing=GA.spacing, R=flat_radii_array)
        field_pic_3d = Plot.plot_3d(Bz=Magnetic_field,
                                    height=GA.height, a_max=max(l),
                                    spacing=GA.spacing, cp=GA.cp)
    # macros = macros.(radii_array)

    if True:  # GA.figure != 'Piecewise':
        resistance = Resistance.resistance_contour(l=lengths, material=GA.material, d=GA.minimal_gap, nu=GA.freq)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.write(f'The COV if the magnetic field generated by your coil is {100 * final_COV:.0f}%.')
        if True:  # GA.figure != 'Piecewise':
            st.download_button(label="Download a VBA macros for your coil",
                               file_name="macros.mcs", data=macros, mime="text/mcs")
            st.write(f'This is the resistance of your coil (Ohm):')
            st.write(resistance)
            st.write(f'This is the total length of your coil (m):')
            st.write(length)
        st.write(f'These are the proportion coefficients (radii):')
        st.write(radii_array)

    with col2:
        st.pyplot(coil_pic, dpi=1000)
        # st.pyplot(field_pic_2d, dpi=1000)
        st.pyplot(field_pic_3d, dpi=1000, orientation='landscape')
