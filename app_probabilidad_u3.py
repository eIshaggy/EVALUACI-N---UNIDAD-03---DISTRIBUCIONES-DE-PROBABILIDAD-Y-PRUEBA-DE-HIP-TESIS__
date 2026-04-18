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

# --- SECCIÓN DE VISUALIZACIÓN Y PRUEBA Z ---
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
        
        # --- SECCIÓN DE PRUEBA DE HIPÓTESIS (ETAPA 3) ---
        st.header("Prueba de Hipótesis (Prueba Z)")
        st.write("Configuración para varianza poblacional conocida y muestra grande (n ≥ 30).")
        
        # Interfaz en dos columnas para los parámetros
        col1, col2 = st.columns(2)
        with col1:
            mu_0 = st.number_input("Hipótesis Nula (μ0)", value=0.0)
            sigma_pop = st.number_input("Desviación Estándar Poblacional (σ) conocida", min_value=0.01, value=1.0)
        with col2:
            alpha = st.selectbox("Nivel de Significancia (α)", [0.01, 0.05, 0.10], index=1)
            tipo_prueba = st.selectbox("Tipo de Prueba", ["Bilateral", "Cola Izquierda", "Cola Derecha"])
            
        # Variables base
        n = len(datos_var)
        media_muestral = datos_var.mean()
        
        # Cálculos matemáticos
        error_estandar = sigma_pop / np.sqrt(n)
        z_stat = (media_muestral - mu_0) / error_estandar
        
        # Lógica de decisión según el tipo de prueba
        if tipo_prueba == "Bilateral":
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            z_crit_inf = stats.norm.ppf(alpha / 2)
            z_crit_sup = stats.norm.isf(alpha / 2)
            rechazar = p_value < alpha
        elif tipo_prueba == "Cola Izquierda":
            p_value = stats.norm.cdf(z_stat)
            z_crit_inf = stats.norm.ppf(alpha)
            z_crit_sup = None
            rechazar = p_value < alpha
        else: # Cola Derecha
            p_value = 1 - stats.norm.cdf(z_stat)
            z_crit_inf = None
            z_crit_sup = stats.norm.isf(alpha)
            rechazar = p_value < alpha
            
        # Impresión de resultados
        st.subheader("Resultados Estadísticos")
        st.write(f"**Tamaño de muestra (n):** {n}")
        st.write(f"**Media muestral:** {media_muestral:.4f}")
        st.write(f"**Estadístico Z calculado:** {z_stat:.4f}")
        st.write(f"**Valor p (p-value):** {p_value:.6f}")
        
        # Decisión automática
        decision_texto = "Rechazar la Hipótesis Nula (H0)" if rechazar else "No rechazar la Hipótesis Nula (H0)"
        st.markdown(f"### Decisión Automática: {decision_texto}")
        # --- VISUALIZACIÓN DE LA REGIÓN CRÍTICA (ETAPA 4) ---
        st.subheader("Visualización de la Región Crítica")
        
        # Crear los datos de la curva de la Campana de Gauss (Distribución Normal Estándar)
        x = np.linspace(-4, 4, 1000)
        y = stats.norm.pdf(x, 0, 1)
        
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(x, y, color='black', label='Distribución Normal Estándar')
        
        # Dibujar una línea roja donde cayó nuestro estadístico Z calculado
        ax2.axvline(z_stat, color='red', linestyle='-', linewidth=2, label=f'Z Calculado ({z_stat:.2f})')
        
        # Sombrear la zona de rechazo de color salmón según el tipo de prueba
        if tipo_prueba == "Bilateral":
            ax2.fill_between(x, y, where=(x < z_crit_inf), color='salmon', alpha=0.5, label='Zona de Rechazo')
            ax2.fill_between(x, y, where=(x > z_crit_sup), color='salmon', alpha=0.5)
            ax2.axvline(z_crit_inf, color='blue', linestyle='--', label=f'Z Crítico Inf ({z_crit_inf:.2f})')
            ax2.axvline(z_crit_sup, color='blue', linestyle='--', label=f'Z Crítico Sup ({z_crit_sup:.2f})')
        elif tipo_prueba == "Cola Izquierda":
            ax2.fill_between(x, y, where=(x < z_crit_inf), color='salmon', alpha=0.5, label='Zona de Rechazo')
            ax2.axvline(z_crit_inf, color='blue', linestyle='--', label=f'Z Crítico ({z_crit_inf:.2f})')
        else: # Cola Derecha
            ax2.fill_between(x, y, where=(x > z_crit_sup), color='salmon', alpha=0.5, label='Zona de Rechazo')
            ax2.axvline(z_crit_sup, color='blue', linestyle='--', label=f'Z Crítico ({z_crit_sup:.2f})')
            
        # Enviar la gráfica a Streamlit
        ax2.set_title("Gráfico de la Prueba de Hipótesis")
        ax2.set_xlabel("Valor Z")
        ax2.set_ylabel("Densidad")
        ax2.legend()
        
        # --- NUEVO CANDADO VISUAL ---
        ax2.set_xlim([-5, 5]) # Congela el zoom en la campana
        if abs(z_stat) > 5:
            st.warning(f"Nota visual: El Estadístico Z calculado ({z_stat:.2f}) es tan extremo que cae fuera del área visible de esta gráfica.")
        
        # Enviar la gráfica a Streamlit
        st.pyplot(fig2)
        
    else:
        st.warning("El archivo cargado no contiene columnas numéricas.")