import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import google.generativeai as genai

# --- CONFIGURACIÓN BASE ---
st.set_page_config(page_title="Análisis Estadístico e Inferencia", layout="wide")
st.title("Plataforma de Inferencia Estadística y Pruebas de Hipótesis")

# --- MÓDULO DE CARGA DE DATOS ---
st.sidebar.header("Módulo de Carga de Datos")
data_option = st.sidebar.radio("Elige la fuente de datos", ("Generar datos sintéticos", "Subir archivo CSV"))

df = None
if data_option == "Generar datos sintéticos":
    n_samples = st.sidebar.number_input("Tamaño de la muestra (n ≥ 30)", min_value=30, value=100)
    mu_sim = st.sidebar.number_input("Media simulada", value=50.0)
    sigma_sim = st.sidebar.number_input("Desviación estándar simulada", value=10.0)
    
    np.random.seed(42)
    data = np.random.normal(loc=mu_sim, scale=sigma_sim, size=n_samples)
    df = pd.DataFrame({"Variable_Sintetica": data})
    st.sidebar.success("Datos sintéticos generados correctamente.")
else:
    uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Archivo cargado correctamente.")

# --- SECCIÓN DE VISUALIZACIÓN ---
if df is not None:
    st.subheader("Vista Previa de los Datos")
    st.dataframe(df.head())
    
    st.header("Visualización de Distribuciones")
    
    # Filtrar solo columnas numéricas
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if columnas_numericas:
        var_seleccionada = st.selectbox("Selecciona la variable a analizar", columnas_numericas)
        datos_var = df[var_seleccionada].dropna()
        
        # Crear la figura para las gráficas
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histograma y KDE
        sns.histplot(datos_var, kde=True, ax=axes[0], color="skyblue")
        axes[0].set_title(f"Histograma y KDE de {var_seleccionada}")
        
        # Boxplot
        sns.boxplot(x=datos_var, ax=axes[1], color="lightgreen")
        axes[1].set_title(f"Boxplot de {var_seleccionada}")
        
        # Enviar gráficas a la interfaz de Streamlit
        st.pyplot(fig)
        
        # Preguntas para el estudiante
        st.subheader("Análisis Cualitativo")
        st.text_area("1. ¿La distribución parece normal?", key="norm_q")
        st.text_area("2. ¿Hay presencia de sesgo o valores atípicos (outliers)?", key="bias_q")
    else:
        st.warning("El archivo cargado no contiene columnas numéricas.")