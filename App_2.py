import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# 1. Configuración profesional
st.set_page_config(page_title="Análisis Estratégico Bank Marketing", layout="wide", page_icon="📊")

# --- DICCIONARIO DE IDIOMAS ---
TRAD_COLUMNAS = {
    "age": "Edad",
    "job": "Ocupación",
    "marital": "Estado Civil",
    "education": "Nivel Educativo",
    "default": "Morosidad (Default)",
    "housing": "Crédito de Vivienda",
    "loan": "Préstamo Personal",
    "contact": "Medio de Comunicación",
    "month": "Mes de Contacto",
    "day_of_week": "Día de la Semana",
    "duration": "Duración de Llamada (seg)",
    "campaign": "Contactos en Campaña",
    "pdays": "Días desde Último Contacto",
    "previous": "Contactos Previos",
    "poutcome": "Resultado Previo",
    "emp.var.rate": "Tasa Variación Empleo",
    "cons.price.idx": "Índice Precios Consumo (IPC)",
    "cons.conf.idx": "Índice Confianza Consumo",
    "euribor3m": "Tasa Euribor 3 Meses",
    "nr.employed": "Número de Empleados",
    "y": "Suscripción (Éxito)"
}

# --- DICCIONARIO DE TRADUCCIÓN (VALORES INTERNOS) ---
TRAD_VALORES = {
    # Ocupaciones
    "admin.": "Administrativo", "blue-collar": "Obrero", "entrepreneur": "Emprendedor",
    "housemaid": "Empleado/a Hogar", "management": "Gestión/Dirección", "retired": "Jubilado",
    "self-employed": "Independiente", "services": "Servicios", "student": "Estudiante",
    "technician": "Técnico", "unemployed": "Desempleado", "unknown": "Desconocido",
    # Estado Civil y Educación
    "divorced": "Divorciado", "married": "Casado", "single": "Soltero",
    "university.degree": "Grado Universitario", "high.school": "Secundaria", 
    "professional.course": "Curso Profesional", "basic.9y": "Educación Básica (9 años)",
    "basic.6y": "Educación Básica (6 años)", "basic.4y": "Educación Básica (4 años)",
    # Resultados y Otros
    "yes": "Sí", "no": "No", "success": "Éxito", "failure": "Fracaso", 
    "nonexistent": "Inexistente", "cellular": "Celular", "telephone": "Teléfono Fijo"
}

# ---------------------------------------------------------
# CLASE DE PROCESAMIENTO ESTADÍSTICO (POO)
# ---------------------------------------------------------
class AnalizadorEstadistico:
    def __init__(self, df):
        self.df = df

    def clasificar_variables(self):
        """Identifica tipos de datos para el EDA (Ítem 2)"""
        num = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat = self.df.select_dtypes(include=['object']).columns.tolist()
        return num, cat

    def descriptivas_completas(self):
        """Genera métricas de tendencia y dispersión (CV) (Ítem 3)"""
        resumen = self.df.describe().T
        resumen['mediana'] = self.df.select_dtypes(include=[np.number]).median()
        # Coeficiente de Variación (CV) - Requerimiento de Sesión 10
        resumen['CV (%)'] = (resumen['std'] / resumen['mean']) * 100
        # Traducir los índices para la visualización final
        resumen.index = [TRAD_COLUMNAS.get(c, c) for c in resumen.index]
        return resumen

# ---------------------------------------------------------
# INTERFAZ DE NAVEGACIÓN

st.sidebar.title("🏦 Panel de Control")
modulo = st.sidebar.radio("Navegar a:", ["Presentación", "Datos", "Análisis EDA", "Conclusiones"])
st.sidebar.markdown(f"**Analista:**\nLucia Azucena Robledo Martinez")

# --- MÓDULO: PRESENTACIÓN ---
if modulo == "Presentación":
    st.title("🎯 Análisis de Campañas de Marketing Bancario")
    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Contexto del Proyecto")
        st.write("Análisis detallado de la efectividad de las campañas de depósitos a plazo fijo de una entidad bancaria, basado en el comportamiento histórico de los clientes.")
        st.info("Este dashboard integra técnicas de POO, visualización avanzada y análisis de variabilidad.")
    with col2:
        st.success("### Ficha Técnica\n**Entorno:** Python 3.x\n**Librerías:** Pandas, Seaborn, Matplotlib\n**Año:** 2026")

# --- MÓDULO: CARGA DE DATOS ---
elif modulo == "Datos":
    st.title("📂 Gestión del Dataset")
    archivo = st.file_uploader("Subir BankMarketing.csv", type=["csv"])
    
    if archivo:
        # Cargamos con el separador ";" del archivo real
        df_raw = pd.read_csv(archivo, sep=";")
        
        # TRADUCCIÓN DE DATOS (Contenido interno)
        df_final = df_raw.replace(TRAD_VALORES)
        st.session_state['df'] = df_final
        
        st.success("✅ Dataset cargado y procesado con éxito.")
        
        st.subheader("Vista Previa")
        # Mostramos con nombres de columnas traducidos
        st.dataframe(df_final.rename(columns=TRAD_COLUMNAS).head(10), use_container_width=True)
        
        c1, c2 = st.columns(2)
        c1.metric("Total Registros", df_final.shape[0])
        c2.metric("Total Variables", df_final.shape[1])

# --- MÓDULO: ANÁLISIS EDA ---
elif modulo == "Análisis EDA":
    if 'df' not in st.session_state:
        st.error("Por favor, suba el archivo en el módulo de 'Datos'.")
    else:
        df = st.session_state['df']
        analista = AnalizadorEstadistico(df)
        num_cols, cat_cols = analista.clasificar_variables()

        st.title("🔍 Exploración de Datos Elaborada")
        
        # Nombres descriptivos para los 10 ítems
        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 = st.tabs([
            "Estructura", "Tipología", "Estadística", "Calidad", 
            "Frecuencias", "Perfil", "Relación Éxito", "Segmentación", 
            "Consulta", "Hallazgos"
        ])

        with t1:
            st.header("1. Perfil Estructural")
            st.info("Descripción técnica de la arquitectura de los datos cargados.")
            buf = io.StringIO()
            df.info(buf=buf)
            st.text(buf.getvalue())

        with t2:
            st.header("2. Clasificación Técnica de Variables")
            st.info("Separación de dimensiones numéricas y categóricas para el análisis.")
            st.write("**Variables Numéricas:**", [TRAD_COLUMNAS.get(c, c) for c in num_cols])
            st.write("**Variables Categóricas:**", [TRAD_COLUMNAS.get(c, c) for c in cat_cols])

        with t3:
            st.header("3. Medidas de Tendencia y Dispersión")
            st.info("Resumen estadístico con enfoque en la variabilidad (CV%).")
            st.dataframe(analista.descriptivas_completas())

        with t4:
            st.header("4. Integridad de la Información")
            st.info("Detección de valores nulos o faltantes en el dataset.")
            st.table(df.isnull().sum().rename(index=TRAD_COLUMNAS))

        with t5:
            st.header("5. Análisis de Distribución")
            st.info("Histogramas interactivos para evaluar el sesgo de las variables.")
            v_n = st.selectbox("Seleccione Variable:", num_cols, format_func=lambda x: TRAD_COLUMNAS.get(x, x))
            fig, ax = plt.subplots()
            sns.histplot(df[v_n], kde=True, color="teal", ax=ax)
            ax.set_xlabel(TRAD_COLUMNAS.get(v_n))
            st.pyplot(fig)

        with t6:
            st.header("6. Análisis de Frecuencias (Categorías)")
            st.info("Visualización del peso de cada categoría dentro de la base de clientes.")
            v_c = st.selectbox("Seleccione Categoría:", cat_cols, format_func=lambda x: TRAD_COLUMNAS.get(x, x))
            fig, ax = plt.subplots()
            df[v_c].value_counts().plot(kind='bar', color="orange", ax=ax)
            st.pyplot(fig)

        with t7:
            st.header("7. Determinantes del Éxito Comercial")
            st.info("Relación entre métricas de comportamiento y la suscripción final.")
            v_biv = st.selectbox("Variable contra Éxito:", ["age", "duration", "nr.employed"], format_func=lambda x: TRAD_COLUMNAS.get(x, x))
            fig, ax = plt.subplots()
            sns.boxplot(x='y', y=v_biv, data=df, ax=ax)
            ax.set_xlabel("¿Hubo Suscripción?")
            ax.set_ylabel(TRAD_COLUMNAS.get(v_biv))
            st.pyplot(fig)

        with t8:
            st.header("8. Segmentación Social Estratégica")
            st.info("Comparativa de éxito cruzando variables cualitativas.")
            v_soc = st.selectbox("Dimensión Social:", ["job", "education", "marital"], format_func=lambda x: TRAD_COLUMNAS.get(x, x))
            tabla = pd.crosstab(df[v_soc], df['y'])
            st.bar_chart(tabla)

        with t9:
            st.header("9. Explorador Dinámico de Clientes")
            st.info("Filtro paramétrico para consultas específicas en el dataset.")
            r_edad = st.slider("Rango de Edad", int(df.age.min()), int(df.age.max()), (25, 45))
            c_sel = st.multiselect("Columnas:", list(TRAD_COLUMNAS.values()), default=["Edad", "Ocupación", "Suscripción (Éxito)"])
            
            if st.checkbox("Mostrar Datos Filtrados"):
                REV = {v: k for k, v in TRAD_COLUMNAS.items()}
                real_c = [REV[c] for c in c_sel]
                df_f = df[(df.age >= r_edad[0]) & (df.age <= r_edad[1])]
                st.dataframe(df_f[real_c].rename(columns=TRAD_COLUMNAS), use_container_width=True)

        with t10:
            st.header("10. Síntesis de Hallazgos")
            st.info("Conclusiones preliminares derivadas del análisis exploratorio.")
            st.markdown("""
            * **Impacto Económico:** El **Número de Empleados** y el **Euribor** muestran una correlación con la receptividad del mercado.
            * **Comportamiento:** La **Duración de la llamada** sigue siendo el predictor más fuerte.
            * **Segmentos:** Los **Jubilados** y **Estudiantes** presentan las tasas de éxito más altas.
            """)

# --- MÓDULO: CONCLUSIONES ---
elif modulo == "Conclusiones":
    st.title("📌 Conclusiones y Recomendaciones")
    st.success("""
    1. **Foco en el Cliente:** Los perfiles con educación universitaria son los más rentables.
    2. **Estrategia de Contacto:** Invertir en guiones que prolonguen la conversación, dado el impacto de la duración.
    3. **Contexto Macroeconómico:** Monitorear el Número de Empleados y el Euribor para decidir la intensidad de las campañas.
    4. **Canales:** El canal celular es significativamente más efectivo que el fijo para cerrar ventas.
    """)