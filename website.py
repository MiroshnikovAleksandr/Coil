import os

import streamlit as st
import pandas as pd
import numpy as np
import Field_functions as ff
import macros

# Настройка боковой панели (Переменные пока от балды назвала)

optimisation = False

with st.sidebar:
    st.title("Ввод параметров")
    form = st.radio("Форма катушки", ['круглая', 'прямоугольная'])
    st.header("Геометрически параметры")
    col1, col2 = st.columns([3, 1])
    h = 0
    if form == 'круглая':
        with col1:
            radius = st.number_input('Радиус, м')
            min_radius = st.nimber_inpur('Минимальный радиус, м')
            h = st.number_input('Высота, м') # две высоты для целевой функции и для графика
            if not optimisation:
                n = st.number_input('Количество витков')
    if form == 'прямоугольная':
        with col1:
            length = st.number_input('Длина, м')
            width = st.number_input('Ширина, м')
            h = st.number_input('Высота, м')
    st.header("Электрические параметры")
    col1, col2 = st.columns([3, 1])
    with col1:
        frequency = st.number_input('Частота')
        I = st.number_input('Сила тока')
    # with col2:
    #     frequency_ = st.selectbox('', ('мГц', 'Гц', 'кГц'))
    #     voltage = st.selectbox('', ('Вт'))

    optimisation = st.checkbox('Оптимизация витков') # deao-fiel new approach'

    with st.expander("Глобальные переменные оптимизации"):
        spacing = st.number_input("spacing")

    do = st.button('Посчитать')


if do:
    if frequency > 0.0 and I > 0.0:
        spacing = 1.3
        cp = 100
        if form == 'круглая':
            if radius > 0.0 and h > 0.0:
                if optimisation:

                else:
                    r_i = np.linspace(radius, min_radius, n)
                    Bz = ff.Bz(radius, 0.1, 10, I, spacing, cp, r_i)
                    st.pyplot(ff.plot_3d(Bz, h, radius, spacing, cp))
            else:
                st.error('Не указан радиус/высота', icon="🚨")
        if form == 'прямоугольная':
            if length > 0 and width > 0 and h > 0:
                m_i = length * np.array([0.206158357771261, 0.5, 0.49824046920821113, 0.3675953079178885,
                                         0.41686217008797655, 0.48944281524926686, 0.4806451612903226,
                                         0.3073313782991202, 0.49032258064516127, 0.48372434017595306])
                n_i = width * np.array([0.206158357771261, 0.5, 0.49824046920821113, 0.3675953079178885,
                                        0.41686217008797655, 0.48944281524926686, 0.4806451612903226,
                                        0.3073313782991202, 0.49032258064516127, 0.48372434017595306])
                Bz = ff.Bz_square(length, width, None, I, spacing, cp, m_i, n_i)
                st.pyplot(ff.plot_square_coil(length, width, spacing, m_i, n_i))
                st.pyplot(ff.plot_3d(Bz, h, max(length, width), spacing, cp))
                # file =
            else:
                st.error('Не указана длина/ширина/высота', icon="🚨")
    else:
        st.error('Не указана частоты/сила тока', icon="🚨")


# вывод коээфициента вариации
#

