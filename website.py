import streamlit as st
import pandas as pd
import numpy as np

# Настройка боковой панели (Переменные пока от балды назвала)
with st.sidebar:
    st.title("Ввод параметров")
    st.radio("Форма катушки", ['круглая', 'прямоугольная'])
    st.header("Геометрически параметры")
    col1, col2 = st.columns([3, 1])
    with col1:
        diameter = st.number_input('Диаметр')
        length = st.number_input('Длина')
        width = st.number_input('Ширина')
    with col2:
        diameter_ = st.selectbox('', ('мм', 'см', 'м'))
        length_ = st.selectbox('  ', ('мм', 'см', 'м'))
        width_ = st.selectbox(' ', ('мм', 'см', 'м'))
    st.header("Электрические параметры")
    col1, col2 = st.columns([3, 1])
    with col1:
        frequency = st.number_input('Частота', value=0.001)
        voltage = st.number_input('Напряжение', value=)
    with col2:
        frequency_ = st.selectbox('', ('мГц', 'Гц', 'кГц'))
        voltage = st.selectbox('', ('Вт'))

    optimisation = st.checkbox('Оптимизация витков')
    do = st.button('Посчитать')


