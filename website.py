import streamlit as st
import pandas as pd
import numpy as np
import tomli
import DEAP_Field_refactored as alg

st.title('Project')

with open('parameters.toml', 'rb') as toml:
    parameters = tomli.load(toml)

minimal_gap = st.number_input('Diameter of your wire, m', value=0.001)
parameters['geom']['minimal_gap'] = minimal_gap

geometry = st.selectbox('Select the geometry of your coil', ('Circular', 'Square', 'Piecewise linear'))

if geometry == 'Circular':
    parameters['geom']['a_max'] = st.number_input('Maximum radius of your coil, m', value=0.5)
    parameters['geom']['a_min'] = st.number_input('Minimal radius of your coil, m', value=0.05)

GA = alg.Genetic(parameters)
GA.preparation()
st.write(GA.execution())
GA.show()

