import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import google.generativeai as genai

# Configuración base de la página
st.set_page_config(page_title="Análisis Estadístico e Inferencia", layout="wide")
st.title("Plataforma de Inferencia Estadística y Pruebas de Hipótesis")

# Módulo de Carga de Datos
st.sidebar.header("Módulo de Carga de Datos")
data_option = st.sidebar.radio("Elige la fuente de datos", ("Generar datos sintéticos", "Subir archivo CSV"))

df = None
if data_option == "Generar datos sintéticos":
    n_samples = st.sidebar.number_input("Tamaño de la muestra (n ≥ 30)", min_value=30, value=100)
    mu_sim = st.sidebar.number_input("Media simulada", value=50.0)
    sigma_sim = st.sidebar.number_input("Desviación estándar simulada", value=10.0)
    
    # Generación de datos
    np.random.seed(42)
    data = np.random.normal(loc=mu_sim, scale=sigma_sim, size=n_samples)
    df = pd.DataFrame({"Variable_Sintetica": data})
    st.sidebar.success("Datos sintéticos generados correctamente.")
else:
    uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Archivo cargado correctamente.")

if df is not None:
    st.subheader("Vista Previa de los Datos")
    st.dataframe(df.head())