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
st.sidebar.header("Configuración de Gemini")
api_key = st.sidebar.text_input("Ingresa tu API Key de Google Gemini:", type="password")
st.sidebar.markdown("---")

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
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if columnas_numericas:
        var_seleccionada = st.selectbox("Selecciona la variable a analizar", columnas_numericas)
        datos_var = df[var_seleccionada].dropna()
        
        # Gráficas
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(datos_var, kde=True, ax=axes[0], color="skyblue")
        axes[0].set_title(f"Histograma y KDE de {var_seleccionada}")
        sns.boxplot(x=datos_var, ax=axes[1], color="lightgreen")
        axes[1].set_title(f"Boxplot de {var_seleccionada}")
        st.pyplot(fig)
        
        # Preguntas
        st.subheader("Análisis Cualitativo")
        st.text_area("1. ¿La distribución parece normal?", key="norm_q")
        st.text_area("2. ¿Hay presencia de sesgo o valores atípicos (outliers)?", key="bias_q")
        
        # --- SECCIÓN DE PRUEBA DE HIPÓTESIS ---
        st.header("Prueba de Hipótesis (Prueba Z)")
        col1, col2 = st.columns(2)
        with col1:
            mu_0 = st.number_input("Hipótesis Nula (μ0)", value=0.0)
            sigma_pop = st.number_input("Desviación Estándar Poblacional (σ) conocida", min_value=0.01, value=1.0)
        with col2:
            alpha = st.selectbox("Nivel de Significancia (α)", [0.01, 0.05, 0.10], index=1)
            tipo_prueba = st.selectbox("Tipo de Prueba", ["Bilateral", "Cola Izquierda", "Cola Derecha"])
            
        n = len(datos_var)
        media_muestral = datos_var.mean()
        error_estandar = sigma_pop / np.sqrt(n)
        z_stat = (media_muestral - mu_0) / error_estandar
        
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
        else:
            p_value = 1 - stats.norm.cdf(z_stat)
            z_crit_inf = None
            z_crit_sup = stats.norm.isf(alpha)
            rechazar = p_value < alpha
            
        st.subheader("Resultados Estadísticos")
        st.write(f"**Tamaño de muestra (n):** {n}")
        st.write(f"**Media muestral:** {media_muestral:.4f}")
        st.write(f"**Estadístico Z calculado:** {z_stat:.4f}")
        st.write(f"**Valor p (p-value):** {p_value:.6f}")
        
        decision_texto = "Rechazar la Hipótesis Nula (H0)" if rechazar else "No rechazar la Hipótesis Nula (H0)"
        st.markdown(f"### Decisión Automática: {decision_texto}")
        
        # --- VISUALIZACIÓN DE LA REGIÓN CRÍTICA ---
        st.subheader("Visualización de la Región Crítica")
        x = np.linspace(-4, 4, 1000)
        y = stats.norm.pdf(x, 0, 1)
        
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(x, y, color='black', label='Distribución Normal Estándar')
        ax2.axvline(z_stat, color='red', linestyle='-', linewidth=2, label=f'Z Calculado ({z_stat:.2f})')
        
        if tipo_prueba == "Bilateral":
            ax2.fill_between(x, y, where=(x < z_crit_inf), color='salmon', alpha=0.5, label='Zona de Rechazo')
            ax2.fill_between(x, y, where=(x > z_crit_sup), color='salmon', alpha=0.5)
            ax2.axvline(z_crit_inf, color='blue', linestyle='--', label=f'Z Crítico Inf ({z_crit_inf:.2f})')
            ax2.axvline(z_crit_sup, color='blue', linestyle='--', label=f'Z Crítico Sup ({z_crit_sup:.2f})')
        elif tipo_prueba == "Cola Izquierda":
            ax2.fill_between(x, y, where=(x < z_crit_inf), color='salmon', alpha=0.5, label='Zona de Rechazo')
            ax2.axvline(z_crit_inf, color='blue', linestyle='--', label=f'Z Crítico ({z_crit_inf:.2f})')
        else:
            ax2.fill_between(x, y, where=(x > z_crit_sup), color='salmon', alpha=0.5, label='Zona de Rechazo')
            ax2.axvline(z_crit_sup, color='blue', linestyle='--', label=f'Z Crítico ({z_crit_sup:.2f})')
            
        ax2.set_title("Gráfico de la Prueba de Hipótesis")
        ax2.set_xlabel("Valor Z")
        ax2.set_ylabel("Densidad")
        ax2.legend()
        
        ax2.set_xlim([-5, 5]) 
        if abs(z_stat) > 5:
            st.warning(f"Nota visual: El Estadístico Z calculado ({z_stat:.2f}) cae fuera del área visible de esta gráfica.")
            
        st.pyplot(fig2)
        
        # --- MÓDULO DE IA (GEMINI) ---
        st.header("Asistente de IA (Análisis de Resultados)")
        if api_key:
            genai.configure(api_key=api_key)
            
            try:
                # Auto-detectar el modelo correcto para tu API Key
                modelo_disponible = None
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        modelo_disponible = m.name
                        break # Usamos el primero que funcione
                
                if modelo_disponible:
                    modelo = genai.GenerativeModel(modelo_disponible)
                    
                    prompt_ia = f"""
                    Actúa como un profesor de estadística evaluando un ejercicio.
                    Se realizó una prueba Z con los siguientes parámetros:
                    - Variable analizada: {var_seleccionada}
                    - n = {n}
                    - media muestral = {media_muestral:.4f}
                    - media hipotética (H0) = {mu_0}
                    - sigma conocida = {sigma_pop}
                    - alpha = {alpha}
                    - tipo de prueba = {tipo_prueba}
                    
                    El estadístico Z calculado fue {z_stat:.4f} y el p-value fue {p_value:.6f}.
                    
                    Explica detalladamente la decisión estadística de rechazar o no rechazar H0. ¿Qué podemos inferir de los resultados en términos prácticos? ¿Son razonables los supuestos de esta prueba (por ejemplo, el tamaño de la muestra)? Redacta tu respuesta en prosa continua, sin usar viñetas ni formatos de lista, como un texto fluido.
                    """
                    
                    if st.button("Solicitar Análisis a Gemini"):
                        with st.spinner("Gemini está analizando los datos..."):
                            respuesta = modelo.generate_content(prompt_ia)
                            st.info(respuesta.text)
                else:
                    st.error("Tu API Key es válida, pero no tiene modelos de texto habilitados.")
            except Exception as e:
                st.error(f"Error al conectar con la API: {e}")
        else:
            st.warning("Por favor, ingresa tu API Key en el menú lateral izquierdo para activar el asistente de IA.")
            
    else:
        st.warning("El archivo cargado no contiene columnas numéricas.")