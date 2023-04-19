import streamlit as st
import pandas as pd
import numpy as np

# Настройка боковой панели (Переменные пока от балды назвала)
with st.sidebar:
    st.title("Ввод параметров")
    form = st.radio("Форма катушки", ['круглая', 'прямоугольная'])
    st.header("Геометрически параметры")
    col1, col2 = st.columns([3, 1])
    if form == 'круглая':
        with col1:
            diameter = st.number_input('Диаметр, м')
    if form == 'прямоугольная':
        with col1:
            length = st.number_input('Длина, м')
            width = st.number_input('Ширина, м')
    st.header("Электрические параметры")
    col1, col2 = st.columns([3, 1])
    with col1:
        frequency = st.number_input('Частота')
        I = st.number_input('Сила тока')
    # with col2:
    #     frequency_ = st.selectbox('', ('мГц', 'Гц', 'кГц'))
    #     voltage = st.selectbox('', ('Вт'))

    optimisation = st.checkbox('Оптимизация витков')
    do = st.button('Посчитать')

    if (form == 'круглая'):
        if diameter != 0.0 and frequency != 0.0 and voltage != 0.0:
                Bz = ff.Bz(diameter, a_min, n, I, 1.3, cp, r_i)

