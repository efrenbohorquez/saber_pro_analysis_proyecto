"""
Aplicación principal del dashboard en Streamlit para visualizar los resultados del análisis de Saber Pro.
"""

# Importaciones básicas
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy import stats
import os
import sys
from pathlib import Path
import joblib  # Para cargar modelos guardados

# Manejo de importación de folium y streamlit_folium con fallback
FOLIUM_AVAILABLE = False
try:
    import folium
    from streamlit_folium import st_folium, folium_static
    FOLIUM_AVAILABLE = True
except ImportError as e:
    st.warning(f"🗺️ Funcionalidades de mapas no disponibles: {e}")
    st.info("💡 Para habilitar mapas interactivos, instale: pip install folium streamlit-folium")
    # Definir funciones dummy para evitar errores
    def st_folium(*args, **kwargs):
        st.error("streamlit_folium no está disponible")
        return None
    def folium_static(*args, **kwargs):
        st.error("streamlit_folium no está disponible")
        return None

# Configuración de la plantilla estilo The Economist
economist_template = go.layout.Template()

economist_template.layout = go.Layout(
    font_family="Arial, sans-serif",
    font_color="#2E2E2E",  # Gris oscuro para texto
    title_font_family="Georgia, serif", # Fuente con serifa para títulos
    title_font_color="#2E2E2E",
    title_x=0.01,  # Título alineado a la izquierda
    title_y=0.95,
    title_yanchor='top',
    title_xanchor='left',
    title_pad_t=20, # Más espacio arriba del título
    title_pad_l=10,
    title_font_size=20,
    plot_bgcolor='white',
    paper_bgcolor='white',
    colorway=['#FF9999', '#757575', '#BDBDBD', '#2E2E2E', '#FFB6C1', '#87CEEB'], # Rojo Economist cambiado a pastel, grises, negro, rosa claro, azul claro
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        linecolor='#757575',
        linewidth=1,
        ticks='outside',
        tickfont_color='#757575',
        title_font_color='#2E2E2E',
        title_standoff=15,
        automargin=True
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='#E0E0E0', # Rejilla horizontal muy sutil
        zeroline=False,
        linecolor='#757575',
        linewidth=1,
        ticks='outside',
        tickfont_color='#757575',
        title_font_color='#2E2E2E',
        title_standoff=15,
        automargin=True
    ),
    legend_orientation="h",
    legend_yanchor="bottom",
    legend_y=1.01,
    legend_xanchor="left", # Leyenda a la izquierda debajo del título
    legend_x=0.01,
    legend_bgcolor='rgba(255,255,255,0.7)',
    legend_bordercolor='#CCCCCC',
    legend_borderwidth=0.5,
    legend_tracegroupgap=10,
    margin=dict(l=80, r=30, t=100, b=80) # Ajustar márgenes (más espacio para título y ejes)
)

# Registrar y establecer la plantilla como predeterminada
pio.templates['economist'] = economist_template
pio.templates.default = 'economist'

# Agregar el directorio raíz al path para importar módulos del proyecto
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config.constants import (
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    SOCIOECONOMIC_VARS,
    ACADEMIC_VARS,
    GEO_VARS,
    STRATA_COLORS
)

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Análisis Saber Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para cargar datos
@st.cache_data
def load_data():
    """Carga los datos procesados."""
    from src.config.constants import RAW_DATA_FILE
    
    try:
        df = pd.read_csv(RAW_DATA_FILE)
        st.success(f"Datos cargados exitosamente desde {RAW_DATA_FILE}")
        return df
    except FileNotFoundError:
        st.error(f"No se encontró el archivo de datos en la ruta especificada: {RAW_DATA_FILE}. Por favor, verifique la ruta.")
        return None
    except Exception as e:
        st.error(f"Ocurrió un error al cargar los datos: {e}")
        return None

@st.cache_data
def load_pca_results():
    """Carga los resultados del análisis PCA."""
    try:
        df = pd.read_csv(PROCESSED_DATA_DIR / "pca_academic_results.csv")
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def load_mca_results():
    """Carga los resultados del análisis MCA."""
    try:
        df = pd.read_csv(PROCESSED_DATA_DIR / "mca_socioeconomic_results.csv")
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def load_clustering_results():
    """Carga los resultados del clustering."""
    try:
        df = pd.read_csv(PROCESSED_DATA_DIR / "clustering_results.csv")
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def load_department_stats():
    """Carga las estadísticas por departamento."""
    try:
        df = pd.read_csv(PROCESSED_DATA_DIR / "department_stats.csv")
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def load_model_comparison():
    """Carga la comparación de modelos predictivos."""
    try:
        df = pd.read_csv(PROCESSED_DATA_DIR / "model_comparison.csv")
        return df
    except FileNotFoundError:
        return None

# Función para mostrar el encabezado
def show_header():
    """Muestra el encabezado del dashboard."""
    st.title("📊 Dashboard Análisis Saber Pro")
    st.markdown("""
    Este dashboard presenta un análisis multivariado de la relación entre el nivel socioeconómico y el rendimiento académico
    en las pruebas Saber Pro en Colombia, utilizando técnicas estadísticas avanzadas.
    """)
    st.markdown("---")

# Función para la página de inicio
def show_home():
    """Muestra la página de inicio del dashboard."""
    st.header("🏠 Inicio")

    # Mostrar la imagen del banner
    banner_path = Path(__file__).parent / "assets" / "484798221_1031777162319148_3372633552707418771_n.jpg"
    if banner_path.exists():
        st.image(str(banner_path), use_container_width=True) # Corregido aquí
    else:
        st.warning(f"No se encontró la imagen del banner en {banner_path}")
    
    st.markdown("""
    ## Análisis Multivariante: Relación entre Nivel Socioeconómico y Rendimiento Académico en las Pruebas Saber Pro
    
    Este proyecto realiza un análisis estadístico exhaustivo de la relación entre el nivel socioeconómico (NSE) y el rendimiento académico (RA) 
    de los estudiantes que presentaron las Pruebas Saber Pro en Colombia, utilizando datos oficiales del ICFES y aplicando técnicas multivariantes avanzadas.
    
    ### Objetivos del Análisis
    
    1. **Describir** las características socioeconómicas y de rendimiento académico de la población estudiantil evaluada.
    2. **Reducir** la dimensionalidad de las variables mediante Análisis de Componentes Principales (ACP) y Análisis Factorial (AF).
    3. **Visualizar** las asociaciones entre variables socioeconómicas y rendimiento académico con Análisis de Correspondencias Múltiples (ACM).
    4. **Identificar** perfiles de estudiantes mediante Análisis de Clúster Jerárquico.
    5. **Explorar** la capacidad predictiva de las variables socioeconómicas sobre el rendimiento académico.
    6. **Analizar** la distribución geográfica de los resultados por estrato y ubicación.
    
    ### Navegación del Dashboard
    
    Utilice el menú lateral para explorar las diferentes secciones del análisis:
    
    - **Exploración de Datos**: Visualización descriptiva de las variables principales.
    - **Análisis de Componentes Principales**: Reducción de dimensionalidad de variables académicas.
    - **Análisis de Correspondencias Múltiples**: Asociaciones entre variables categóricas.
    - **Clustering Jerárquico**: Identificación de perfiles de estudiantes.
    - **Modelos Predictivos**: Predicción del rendimiento académico.
    - **Análisis Geoespacial**: Distribución geográfica de resultados.
    """)
    
    # Mostrar algunas estadísticas generales si los datos están disponibles
    df = load_data()
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Estudiantes", f"{len(df):,}")
        
        with col2:
            if 'FAMI_ESTRATOVIVIENDA' in df.columns:
                strata_counts = df['FAMI_ESTRATOVIVIENDA'].value_counts()
                most_common_strata = strata_counts.index[0]
                st.metric("Estrato más común", most_common_strata)
        
        with col3:
            if 'MOD_RAZONA_CUANTITAT_PUNT' in df.columns:
                avg_score = df['MOD_RAZONA_CUANTITAT_PUNT'].mean()
                st.metric("Puntaje Promedio Razonamiento", f"{avg_score:.1f}")
        
        with col4:
            if 'ESTU_DEPTO_RESIDE' in df.columns:
                dept_counts = df['ESTU_DEPTO_RESIDE'].value_counts()
                most_common_dept = dept_counts.index[0]
                st.metric("Departamento con más estudiantes", most_common_dept)

# Función para la página de exploración de datos
def show_eda():
    """Muestra la página de exploración de datos."""
    st.header("🔍 Exploración de Datos")
    
    df = load_data()
    if df is None:
        st.warning("No se encontraron datos para mostrar.")
        return
    
    # Pestañas para diferentes aspectos del EDA
    tab1, tab2, tab3, tab4 = st.tabs(["Variables Académicas", "Variables Socioeconómicas", "Distribuciones", "Correlaciones"])
    
    with tab1:
        st.subheader("Análisis de Variables Académicas")
        
        # Seleccionar variable académica
        academic_vars = [col for col in df.columns if 'MOD_' in col and 'PUNT' in col]
        if academic_vars:
            selected_var = st.selectbox("Seleccione una variable académica:", academic_vars)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histograma
                fig = px.histogram(
                    df, 
                    x=selected_var,
                    title=f"Distribución de {selected_var}",
                    color_discrete_sequence=['#1f77b4'],
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Estadísticas descriptivas
                stats = df[selected_var].describe()
                st.write("Estadísticas descriptivas:")
                st.dataframe(stats)
                
                # Boxplot por género si está disponible
                if 'ESTU_GENERO' in df.columns:
                    fig = px.box(
                        df, 
                        x="ESTU_GENERO", 
                        y=selected_var,
                        title=f"{selected_var} por Género",
                        color="ESTU_GENERO"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Análisis de Variables Socioeconómicas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribución de estratos
            if 'FAMI_ESTRATOVIVIENDA' in df.columns:
                fig = px.pie(
                    df, 
                    names='FAMI_ESTRATOVIVIENDA',
                    title="Distribución por Estrato Socioeconómico",
                    color='FAMI_ESTRATOVIVIENDA',
                    color_discrete_map=STRATA_COLORS
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Nivel educativo de los padres
            if 'FAMI_EDUCACIONPADRE' in df.columns and 'FAMI_EDUCACIONMADRE' in df.columns:
                # Seleccionar padre o madre
                parent = st.radio("Seleccione:", ["Padre", "Madre"])
                
                if parent == "Padre":
                    edu_col = 'FAMI_EDUCACIONPADRE'
                    title = "Nivel Educativo del Padre"
                else:
                    edu_col = 'FAMI_EDUCACIONMADRE'
                    title = "Nivel Educativo de la Madre"
                
                # Contar y ordenar
                edu_counts = df[edu_col].value_counts().reset_index()
                edu_counts.columns = [edu_col, 'Cantidad']
                
                fig = px.bar(
                    edu_counts,
                    x=edu_col,
                    y='Cantidad',
                    title=title,
                    color_discrete_sequence=['#2ca02c']
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Nuevas visualizaciones socioeconómicas

            st.subheader("Horas de Trabajo Semanales")
            if 'ESTU_HORASSEMANATRABAJA_NUM' in df.columns:
                fig_horas_trabajo = px.histogram(
                    df, 
                    x='ESTU_HORASSEMANATRABAJA_NUM',
                    title="Distribución de Horas de Trabajo Semanales",
                    marginal="box"
                )
                st.plotly_chart(fig_horas_trabajo, use_container_width=True, key="horas_trabajo_hist")
            else:
                st.markdown("_Variable 'ESTU_HORASSEMANATRABAJA_NUM' no encontrada._")

            st.subheader("Valor de Matrícula Universitaria")
            if 'ESTU_VALORMATRICULAUNIVERSIDAD' in df.columns:
                # Podría ser útil transformar esta variable (e.g. log) si está muy sesgada
                # o manejar valores no numéricos si existen.
                # Por ahora, un histograma directo asumiendo que es numérica.
                df_matricula = df[pd.to_numeric(df['ESTU_VALORMATRICULAUNIVERSIDAD'], errors='coerce').notna()]
                fig_valor_matricula = px.histogram(
                    df_matricula, 
                    x='ESTU_VALORMATRICULAUNIVERSIDAD',
                    title="Distribución del Valor de Matrícula Universitaria",
                    marginal="box"
                )
                st.plotly_chart(fig_valor_matricula, use_container_width=True, key="valor_matricula_hist")
            else:
                st.markdown("_Variable 'ESTU_VALORMATRICULAUNIVERSIDAD' no encontrada._")

            st.subheader("Fuentes de Pago de Matrícula")
            payment_vars = [
                'ESTU_PAGOMATRICULABECA',
                'ESTU_PAGOMATRICULACREDITO',
                'ESTU_PAGOMATRICULAPADRES',
                'ESTU_PAGOMATRICULAPROPIO'
            ]
            existing_payment_vars = [var for var in payment_vars if var in df.columns]

            if existing_payment_vars:
                payment_data = pd.melt(
                    df[existing_payment_vars],
                    var_name='Fuente de Pago',
                    value_name='Respuesta'
                )
                

                payment_names = {
                    'ESTU_PAGOMATRICULABECA': 'Beca',
                    'ESTU_PAGOMATRICULACREDITO': 'Crédito',
                    'ESTU_PAGOMATRICULAPADRES': 'Padres/Familiares',
                    'ESTU_PAGOMATRICULAPROPIO': 'Recursos Propios'
                }
                payment_data['Fuente de Pago'] = payment_data['Fuente de Pago'].map(payment_names)
                
                # Contar respuestas (asumiendo que son categóricas, e.g., 'Si', 'No', o niveles)
                # Si son booleanas o 'Si'/'No', podríamos filtrar por 'Si'
                # Por ahora, mostramos todas las respuestas encontradas por fuente
                payment_counts = payment_data.groupby(['Fuente de Pago', 'Respuesta']).size().reset_index(name='Cantidad')
                
                fig_payment = px.bar(
                    payment_counts,
                    x='Fuente de Pago',
                    y='Cantidad',
                    color='Respuesta',
                    title="Fuentes de Pago de Matrícula",
                    barmode='group'
                )
                st.plotly_chart(fig_payment, use_container_width=True, key="fuentes_pago_bar")
            else:
                st.markdown("_No se encontraron variables sobre fuentes de pago de matrícula._")
    
    with tab3:
        st.subheader("Distribuciones")
        
        # Seleccionar variable para distribución
        all_numeric = df.select_dtypes(include=['number']).columns.tolist()
        selected_var = st.selectbox("Seleccione una variable para visualizar su distribución:", all_numeric)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma con KDE
            fig = px.histogram(
                df, 
                x=selected_var,
                title=f"Distribución de {selected_var}",
                color_discrete_sequence=['#1f77b4'],
                marginal="box"
            )
            st.plotly_chart(fig, use_container_width=True, key=f"dist_hist_{selected_var}")
        
        with col2:
            # QQ Plot para normalidad
            from scipy import stats
            
            fig = px.scatter(
                x=np.sort(stats.norm.ppf(np.linspace(0.01, 0.99, len(df)))),
                y=np.sort(df[selected_var].dropna()),
                title=f"QQ Plot de {selected_var}",
                labels={"x": "Cuantiles teóricos", "y": "Cuantiles observados"}
            )
            st.plotly_chart(fig, use_container_width=True, key=f"dist_qq_{selected_var}")
        
        # Prueba de Shapiro-Wilk para normalidad
        st.subheader("Prueba de Normalidad")
        
        if selected_var in df.columns:
            from scipy.stats import shapiro
            
            # Realizar prueba de Shapiro-Wilk
            stat, p_value = shapiro(df[selected_var].dropna())
            
            # Mostrar resultados
            st.write(f"Estadístico de prueba: {stat:.4f}")
            st.write(f"Valor p: {p_value:.4f}")
            
            # Interpretar resultados
            alpha = 0.05
            if p_value > alpha:
                st.success("No se rechaza la hipótesis nula: los datos siguen una distribución normal.")
            else:
                st.error("Se rechaza la hipótesis nula: los datos no siguen una distribución normal.")
    
    with tab4:
        st.subheader("Correlaciones")
        
        # Seleccionar variables para matriz de correlación
        numeric_vars = df.select_dtypes(include=['number']).columns.tolist()
        
        # Filtrar solo variables académicas y socioeconómicas numéricas
        academic_vars = [var for var in numeric_vars if 'MOD_' in var and 'PUNT' in var]
        socioeconomic_vars = [var for var in numeric_vars if var in ['NSE_SCORE', 'ESTRATO_NUM', 'FAMI_EDUCACIONPADRE_NIVEL', 'FAMI_EDUCACIONMADRE_NIVEL']]
        
        # Combinar y eliminar duplicados
        corr_vars = list(set(academic_vars + socioeconomic_vars))
        
        if len(corr_vars) > 1:
            # Calcular matriz de correlación
            corr_matrix = df[corr_vars].corr()
            
            # Crear mapa de calor con Plotly
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Matriz de Correlación"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar correlaciones específicas con rendimiento académico
            st.subheader("Correlaciones con Rendimiento Académico")
            
            # Seleccionar variable de rendimiento
            target_var = st.selectbox("Seleccione una variable de rendimiento:", academic_vars)
            
            # Calcular correlaciones con la variable objetivo
            correlations = df[corr_vars].corr()[target_var].sort_values(ascending=False)
            correlations = correlations.drop(target_var)  # Eliminar autocorrelación
            
            # Mostrar como gráfico de barras
            fig = px.bar(
                x=correlations.index,
                y=correlations.values,
                title=f"Correlaciones con {target_var}",
                labels={"x": "Variable", "y": "Coeficiente de correlación"},
                color=correlations.values,
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig, use_container_width=True)

# Función para la página de PCA
def show_pca():
    """Muestra la página de análisis de componentes principales."""
    st.header("🧩 Análisis de Componentes Principales")
    
    df_pca = load_pca_results()
    if df_pca is None:
        st.warning("No se encontraron resultados de PCA. Por favor ejecute primero el análisis completo.")
        return
    
    # Pestañas para diferentes aspectos del PCA
    tab1, tab2, tab3 = st.tabs(["Varianza Explicada", "Biplot", "Proyección de Datos"])
    
    with tab1:
        st.subheader("Varianza Explicada por Componentes")
        
        # Mostrar gráfico de sedimentación
        scree_path = FIGURES_DIR / 'pca_academic_scree.png'
        if os.path.exists(scree_path):
            st.image(str(scree_path), caption="Gráfico de Sedimentación (Scree Plot)")
        else:
            st.error("No se encontró el gráfico de sedimentación.")
        
        st.markdown("""
        **Interpretación:**
        
        El gráfico de sedimentación muestra la proporción de varianza explicada por cada componente principal.
        - La línea azul representa la varianza explicada por cada componente individual.
        - La línea naranja muestra la varianza acumulada.
        - La línea roja punteada indica el umbral del 80% de varianza explicada.
        
        Este gráfico ayuda a determinar cuántos componentes retener para el análisis.
        """)
    
    with tab2:
        st.subheader("Biplot: Relación entre Variables y Componentes")
        
        # Mostrar biplot
        biplot_path = FIGURES_DIR / 'pca_academic_biplot.png'
        if os.path.exists(biplot_path):
            st.image(str(biplot_path), caption="Biplot de PCA")
        else:
            st.error("No se encontró el biplot.")
        
        st.markdown("""
        **Interpretación:**
        
        El biplot muestra simultáneamente:
        - La proyección de las observaciones (estudiantes) en el espacio de los dos primeros componentes principales.
        - Los vectores que representan las variables originales en este espacio.
        
        Puntos clave para la interpretación:
        - La dirección de los vectores indica qué variables contribuyen más a cada componente.
        - Vectores que apuntan en direcciones similares indican variables correlacionadas positivamente.
        - Vectores que apuntan en direcciones opuestas indican variables correlacionadas negativamente.
        - La longitud de los vectores indica la importancia de la variable en los componentes mostrados.
        """)
        
        # Mostrar mapa de calor de cargas
        loadings_path = FIGURES_DIR / 'pca_academic_loadings.png'
        if os.path.exists(loadings_path):
            st.image(str(loadings_path), caption="Mapa de Calor de Cargas Factoriales")
        else:
            st.error("No se encontró el mapa de calor de cargas factoriales.")
    
    with tab3:
        st.subheader("Proyección de Datos en Componentes Principales")
        
        # Verificar si existen columnas de PCA
        pca_cols = [col for col in df_pca.columns if col.startswith('PCA_ACADEMIC_')]
        
        if pca_cols and len(pca_cols) >= 2:
            # Seleccionar variables para colorear
            color_options = ['FAMI_ESTRATOVIVIENDA', 'NSE_NIVEL', 'RA_NIVEL', 'ESTU_GENERO', 'CLUSTER']
            color_options = [opt for opt in color_options if opt in df_pca.columns]
            
            if color_options:
                color_var = st.selectbox("Colorear por:", color_options)
                
                # Crear scatter plot interactivo
                fig = px.scatter(
                    df_pca,
                    x=pca_cols[0],
                    y=pca_cols[1],
                    color=color_var,
                    title=f"Proyección de Datos en Componentes Principales (coloreado por {color_var})",
                    labels={
                        pca_cols[0]: "Componente Principal 1",
                        pca_cols[1]: "Componente Principal 2"
                    },
                    opacity=0.7
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Añadir selector para componentes 3D si hay suficientes componentes
                if len(pca_cols) >= 3:
                    st.subheader("Visualización 3D")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        x_comp = st.selectbox("Componente X:", pca_cols, index=0)
                    
                    with col2:
                        y_comp = st.selectbox("Componente Y:", pca_cols, index=1)
                    
                    with col3:
                        z_comp = st.selectbox("Componente Z:", pca_cols, index=2)
                    
                    # Crear scatter plot 3D
                    fig = px.scatter_3d(
                        df_pca,
                        x=x_comp,
                        y=y_comp,
                        z=z_comp,
                        color=color_var,
                        title=f"Proyección 3D (coloreado por {color_var})",
                        opacity=0.7
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No se encontraron variables categóricas para colorear el gráfico.")
        else:
            st.error("No se encontraron columnas de PCA en los datos.")

# Función para la página de MCA
def show_mca():
    """Muestra la página de análisis de correspondencias múltiples."""
    st.header("🔄 Análisis de Correspondencias Múltiples")
    
    df_mca = load_mca_results()
    if df_mca is None:
        st.warning("No se encontraron resultados de MCA. Por favor ejecute primero el análisis completo.")
        return
    
    # Pestañas para diferentes aspectos del MCA
    tab1, tab2, tab3 = st.tabs(["Inercia Explicada", "Mapa Factorial", "Proyección de Individuos"])
    
    with tab1:
        st.subheader("Inercia Explicada por Dimensiones")
        
        # Mostrar gráfico de sedimentación
        scree_path = FIGURES_DIR / 'mca_socioeconomic_scree.png'
        if os.path.exists(scree_path):
            st.image(str(scree_path), caption="Gráfico de Sedimentación (Scree Plot) - MCA")
        else:
            st.error("No se encontró el gráfico de sedimentación para MCA.")
        
        st.markdown("""
        **Interpretación:**
        
        El gráfico de sedimentación muestra la proporción de inercia (varianza) explicada por cada dimensión del ACM.
        - La línea azul representa la inercia explicada por cada dimensión individual.
        - La línea naranja muestra la inercia acumulada.
        
        A diferencia del ACP, en el ACM es común que se necesiten más dimensiones para explicar una proporción significativa de la inercia total.
        """)
    
    with tab2:
        st.subheader("Mapa Factorial: Relación entre Categorías")
        
        # Mostrar mapa factorial
        factor_map_path = FIGURES_DIR / 'mca_socioeconomic_factor_map.png'
        if os.path.exists(factor_map_path):
            st.image(str(factor_map_path), caption="Mapa Factorial de Categorías - ACM")
        else:
            st.error("No se encontró el mapa factorial.")
        
        st.markdown("""
        **Interpretación:**
        
        El mapa factorial muestra la posición de las categorías de las variables en el espacio de las dos primeras dimensiones del ACM.
        
        Puntos clave para la interpretación:
        - Categorías cercanas en el mapa tienden a aparecer juntas en los mismos individuos.
        - Categorías alejadas del origen contribuyen más a la definición de las dimensiones.
        - La proximidad entre categorías de diferentes variables sugiere asociación entre ellas.
        - La distribución de las categorías a lo largo de los ejes ayuda a interpretar el significado de cada dimensión.
        """)
    
    with tab3:
        st.subheader("Proyección de Individuos")
        
        # Verificar si existen columnas de MCA
        mca_cols = [col for col in df_mca.columns if col.startswith('MCA_SOCIOECONOMIC_')]
        
        if mca_cols and len(mca_cols) >= 2:
            # Seleccionar variables para colorear
            color_options = ['FAMI_ESTRATOVIVIENDA', 'NSE_NIVEL', 'RA_NIVEL', 'ESTU_GENERO', 'CLUSTER']
            color_options = [opt for opt in color_options if opt in df_mca.columns]
            
            if color_options:
                color_var = st.selectbox("Colorear por:", color_options, key="mca_color")
                
                # Crear scatter plot interactivo
                fig = px.scatter(
                    df_mca,
                    x=mca_cols[0],
                    y=mca_cols[1],
                    color=color_var,
                    title=f"Proyección de Individuos en ACM (coloreado por {color_var})",
                    labels={
                        mca_cols[0]: "Dimensión 1",
                        mca_cols[1]: "Dimensión 2"
                    },
                    opacity=0.7
                )
                
                # Añadir líneas de referencia
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar imágenes específicas por estrato y rendimiento académico
                st.subheader("Visualizaciones Específicas")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Individuos por estrato
                    strata_path = FIGURES_DIR / 'mca_socioeconomic_individuals_by_strata.png'
                    if os.path.exists(strata_path):
                        st.image(str(strata_path), caption="Individuos por Estrato")
                    else:
                        st.error("No se encontró la visualización por estrato.")
                
                with col2:
                    # Individuos por rendimiento académico
                    ra_path = FIGURES_DIR / 'mca_socioeconomic_individuals_by_ra.png'
                    if os.path.exists(ra_path):
                        st.image(str(ra_path), caption="Individuos por Rendimiento Académico")
                    else:
                        st.error("No se encontró la visualización por rendimiento académico.")
            else:
                st.warning("No se encontraron variables categóricas para colorear el gráfico.")
        else:
            st.error("No se encontraron columnas de MCA en los datos.")

# Función para la página de clustering
def show_clustering():
    """Muestra la página de clustering jerárquico."""
    st.header("🔬 Clustering Jerárquico")
    
    df_cluster = load_clustering_results()
    if df_cluster is None:
        st.warning("No se encontraron resultados de clustering. Por favor ejecute primero el análisis completo.")
        return
    
    # Pestañas para diferentes aspectos del clustering
    tab1, tab2, tab3 = st.tabs(["Dendrograma y Validación", "Perfiles de Clusters", "Distribución por Variables"])
    
    with tab1:
        st.subheader("Dendrograma y Validación de Clusters")
        
        # Mostrar dendrograma
        dendrogram_path = FIGURES_DIR / 'cluster_dendrogram.png'
        if os.path.exists(dendrogram_path):
            st.image(str(dendrogram_path), caption="Dendrograma de Clustering Jerárquico")
        else:
            st.error("No se encontró el dendrograma.")
        
        st.markdown("""
        **Interpretación:**
        
        El dendrograma muestra la estructura jerárquica de los clusters:
        - El eje vertical representa la distancia o disimilitud entre clusters.
        - Las líneas horizontales representan fusiones de clusters.
        - La altura de las líneas horizontales indica la distancia a la que se fusionan los clusters.
        - Cortar el dendrograma a diferentes alturas produce diferentes números de clusters.
        """)
        
        # Mostrar gráfico de puntuaciones de silueta
        silhouette_path = FIGURES_DIR / 'cluster_silhouette_scores.png'
        if os.path.exists(silhouette_path):
            st.image(str(silhouette_path), caption="Puntuaciones de Silueta para Diferentes Números de Clusters")
        else:
            st.error("No se encontró el gráfico de puntuaciones de silueta.")
        
        st.markdown("""
        **Interpretación:**
        
        El gráfico de puntuaciones de silueta ayuda a determinar el número óptimo de clusters:
        - El eje horizontal representa diferentes números de clusters.
        - El eje vertical muestra la puntuación de silueta promedio para cada número de clusters.
        - Una mayor puntuación de silueta indica una mejor calidad de clustering.
        - La línea vertical punteada indica el número óptimo de clusters según este criterio.
        """)
    
    with tab2:
        st.subheader("Perfiles de Clusters")
        
        # Mostrar perfiles de clusters
        profiles_path = FIGURES_DIR / 'cluster_profiles.png'
        if os.path.exists(profiles_path):
            st.image(str(profiles_path), caption="Perfiles de Clusters")
        else:
            st.error("No se encontró el gráfico de perfiles de clusters.")
        
        st.markdown("""
        **Interpretación:**
        
        El gráfico de perfiles muestra las características promedio de cada cluster:
        - Cada línea representa un cluster.
        - El eje horizontal representa diferentes variables.
        - El eje vertical muestra los valores estandarizados de cada variable.
        - Valores positivos indican que el cluster tiene valores por encima de la media en esa variable.
        - Valores negativos indican que el cluster tiene valores por debajo de la media en esa variable.
        
        Este gráfico ayuda a interpretar qué características definen a cada cluster.
        """)
        
        # Mostrar estadísticas por cluster si la columna CLUSTER existe
        if 'CLUSTER' in df_cluster.columns:
            st.subheader("Estadísticas por Cluster")
            
            # Seleccionar variables para mostrar estadísticas
            numeric_vars = df_cluster.select_dtypes(include=['number']).columns.tolist()
            academic_vars = [var for var in numeric_vars if 'MOD_' in var and 'PUNT' in var]
            socioeconomic_vars = [var for var in numeric_vars if var in ['NSE_SCORE', 'ESTRATO_NUM', 'FAMI_EDUCACIONPADRE_NIVEL', 'FAMI_EDUCACIONMADRE_NIVEL']]
            
            # Combinar y eliminar duplicados
            selected_vars = list(set(academic_vars + socioeconomic_vars))
            
            if selected_vars:
                # Calcular estadísticas por cluster
                cluster_stats = df_cluster.groupby('CLUSTER')[selected_vars].mean()
                
                # Mostrar como tabla
                st.write("Valores promedio por cluster:")
                st.dataframe(cluster_stats)
                
                # Mostrar como gráfico de radar
                # Preparar datos para gráfico de radar
                fig = go.Figure()
                
                for cluster in cluster_stats.index:
                    fig.add_trace(go.Scatterpolar(
                        r=cluster_stats.loc[cluster].values,
                        theta=cluster_stats.columns,
                        fill='toself',
                        name=f'Cluster {cluster}'
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                        )
                    ),
                    title="Perfiles de Clusters (Gráfico de Radar)",
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Distribución por Variables")
        
        # Mostrar distribución por estrato
        strata_dist_path = FIGURES_DIR / 'cluster_distribution_by_strata.png'
        if os.path.exists(strata_dist_path):
            st.image(strata_dist_path, caption="Distribución de Clusters por Estrato")
        else:
            st.error("No se encontró el gráfico de distribución por estrato.")
        
        st.markdown("""
        **Interpretación:**
        
        El gráfico de distribución muestra cómo se distribuyen los clusters en diferentes estratos:
        - El eje horizontal representa los diferentes estratos.
        - El eje vertical muestra el porcentaje de cada cluster dentro de cada estrato.
        - Las barras apiladas representan la proporción de cada cluster en cada estrato.
        
        Este gráfico ayuda a entender si ciertos clusters están más asociados con determinados estratos socioeconómicos.
        """)
        
        # Visualización interactiva de clusters en el espacio de PCA o MCA
        st.subheader("Visualización de Clusters en Espacio Reducido")
        
        # Verificar si existen columnas de PCA o MCA
        pca_cols = [col for col in df_cluster.columns if col.startswith('PCA_ACADEMIC_')]
        mca_cols = [col for col in df_cluster.columns if col.startswith('MCA_SOCIOECONOMIC_')]
        
        if 'CLUSTER' in df_cluster.columns:
            if pca_cols and len(pca_cols) >= 2:
                st.subheader("Visualización de Clusters en Espacio PCA")
                
                # Crear scatter plot interactivo
                fig = px.scatter(
                    df_cluster,
                    x=pca_cols[0],
                    y=pca_cols[1],
                    color='CLUSTER',
                    title="Clusters en Espacio PCA",
                    labels={
                        pca_cols[0]: "Componente Principal 1",
                        pca_cols[1]: "Componente Principal 2"
                    },
                    opacity=0.7,
                    category_orders={"CLUSTER": sorted(df_cluster['CLUSTER'].unique())}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            if mca_cols and len(mca_cols) >= 2:
                st.subheader("Visualización de Clusters en Espacio MCA")
                
                # Crear scatter plot interactivo
                fig = px.scatter(
                    df_cluster,
                    x=mca_cols[0],
                    y=mca_cols[1],
                    color='CLUSTER',
                    title="Clusters en Espacio MCA",
                    labels={
                        mca_cols[0]: "Dimensión 1",
                        mca_cols[1]: "Dimensión 2"
                    },
                    opacity=0.7,
                    category_orders={"CLUSTER": sorted(df_cluster['CLUSTER'].unique())}
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Función para la página de modelos predictivos
def show_predictive_models():
    """Muestra la página de modelos predictivos."""
    st.header("📈 Modelos Predictivos")
    
    df = load_data()
    model_comparison = load_model_comparison()
    
    if df is None or model_comparison is None:
        st.warning("No se encontraron resultados de modelos predictivos. Por favor ejecute primero el análisis completo.")
        return
    
    # Pestañas para diferentes aspectos de los modelos predictivos
    tab1, tab2, tab3 = st.tabs(["Comparación de Modelos", "Importancia de Variables", "Predicciones vs. Reales"])
    
    with tab1:
        st.subheader("Comparación de Modelos")
        
        # Mostrar comparación de modelos
        comparison_path = FIGURES_DIR / 'model_comparison.png'
        if os.path.exists(comparison_path):
            st.image(str(comparison_path), caption="Comparación de Modelos Predictivos")
        else:
            st.error("No se encontró el gráfico de comparación de modelos.")
        
        # Mostrar tabla de comparación
        if model_comparison is not None:
            st.write("Métricas de evaluación de modelos:")
            st.dataframe(model_comparison)
        
        st.markdown("""
        **Interpretación:**
        
        La comparación de modelos muestra el rendimiento de diferentes algoritmos predictivos:
        - **R²**: Coeficiente de determinación. Mayor es mejor (máximo 1).
        - **RMSE**: Error cuadrático medio. Menor es mejor.
        - **MAE**: Error absoluto medio. Menor es mejor.
        
        El mejor modelo es aquel que maximiza R² y minimiza RMSE y MAE.
        """)
    
    with tab2:
        st.subheader("Importancia de Variables")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Mostrar importancia de características para regresión lineal
            lr_importance_path = FIGURES_DIR / 'lr_feature_importance.png'
            if os.path.exists(lr_importance_path):
                st.image(str(lr_importance_path), caption="Importancia de Variables - Regresión Lineal")
            else:
                st.error("No se encontró el gráfico de importancia para regresión lineal.")
        
        with col2:
            # Mostrar importancia de características para random forest
            rf_importance_path = FIGURES_DIR / 'rf_feature_importance.png'
            if os.path.exists(rf_importance_path):
                st.image(str(rf_importance_path), caption="Importancia de Variables - Random Forest")
            else:
                st.error("No se encontró el gráfico de importancia para Random Forest.")
        
        st.markdown("""
        **Interpretación:**
        
        Los gráficos de importancia de variables muestran qué características tienen mayor influencia en las predicciones:
        - Las barras representan la importancia relativa de cada variable.
        - Variables con mayor importancia (barras más largas) tienen mayor impacto en la predicción.
        - La comparación entre modelos muestra si diferentes algoritmos utilizan las variables de manera distinta.
        
        Las variables socioeconómicas más importantes pueden considerarse como factores claves que influyen en el rendimiento académico.
        """)
    
    with tab3:
        st.subheader("Predicciones vs. Valores Reales")
        
        # Mostrar predicciones vs reales para regresión lineal
        lr_pred_path = FIGURES_DIR / 'lr_prediction_vs_actual.png'
        if os.path.exists(lr_pred_path):
            st.image(str(lr_pred_path), caption="Predicciones vs. Valores Reales - Regresión Lineal")
        else:
            st.error("No se encontró el gráfico de predicciones para regresión lineal.")
        
        # Mostrar residuos para regresión lineal
        lr_residuals_path = FIGURES_DIR / 'lr_residuals.png'
        if os.path.exists(lr_residuals_path):
            st.image(str(lr_residuals_path), caption="Residuos - Regresión Lineal")
        else:
            st.error("No se encontró el gráfico de residuos para regresión lineal.")
        
        # Mostrar predicciones vs reales para Random Forest
        rf_pred_path = FIGURES_DIR / 'rf_prediction_vs_actual.png'
        if os.path.exists(rf_pred_path):
            st.image(str(rf_pred_path), caption="Predicciones vs. Valores Reales - Random Forest")
        else:
            st.error("No se encontró el gráfico de predicciones para Random Forest.")
        
        st.markdown("""
        **Interpretación:**
        
        Los gráficos de predicciones vs. valores reales muestran cuán bien el modelo predice los valores:
        - Puntos cercanos a la línea diagonal indican predicciones precisas.
        - Puntos por encima de la línea indican sobreestimación.
        - Puntos por debajo de la línea indican subestimación.
        - Un patrón aleatorio alrededor de la línea sugiere un buen ajuste del modelo.
        
        El gráfico de residuos muestra las diferencias entre las predicciones y los valores reales:
        - Un buen modelo debe mostrar residuos distribuidos aleatoriamente alrededor de cero.
        - Patrones claros en los residuos sugieren que el modelo no captura alguna estructura importante en los datos.
        """)

# Función para la página de análisis geoespacial
def show_geospatial():
    """Muestra la página de análisis geoespacial."""
    st.header("🗺️ Análisis Geoespacial")
    
    # Verificar si folium está disponible
    if not FOLIUM_AVAILABLE:
        st.warning("🗺️ Las funcionalidades de mapas interactivos no están disponibles en este entorno.")
        st.info("💡 Los mapas requieren las librerías folium y streamlit-folium.")
        
        # Mostrar solo los gráficos estáticos disponibles
        st.subheader("Gráficos de Análisis Geoespacial Disponibles")
        
        # Mostrar gráfico de rendimiento por departamento si existe
        performance_path = FIGURES_DIR / 'performance_by_department.png'
        if os.path.exists(performance_path):
            st.image(str(performance_path), caption="Rendimiento Académico por Departamento")
        
        # Mostrar distribución de estratos por departamento si existe
        strata_dist_path = FIGURES_DIR / 'strata_by_department.png'
        if os.path.exists(strata_dist_path):
            st.image(str(strata_dist_path), caption="Distribución de Estratos por Departamento")
        
        if not os.path.exists(performance_path) and not os.path.exists(strata_dist_path):
            st.info("📊 Para generar gráficos geoespaciales, ejecute primero el análisis completo en un entorno local.")
        
        return
    
    df_geo = load_department_stats()
    if df_geo is None:
        st.warning("No se encontraron resultados geoespaciales. Por favor ejecute primero el análisis completo.")
        return
    
    # Pestañas para diferentes aspectos del análisis geoespacial
    tab1, tab2, tab3 = st.tabs(["Mapas de Rendimiento", "Distribución por Departamento", "Comparación Regional"])
    
    with tab1:
        st.subheader("Mapas de Colombia")
        
        # Seleccionar variable para mapa
        map_options = ["Cantidad de Estudiantes", "Razonamiento Cuantitativo", "Lectura Crítica", "Estrato"]
        selected_map = st.selectbox("Seleccione el mapa a visualizar:", map_options)
        
        # Mostrar mapa correspondiente
        if selected_map == "Cantidad de Estudiantes":
            map_path = FIGURES_DIR / 'map_cantidad_estudiantes.html'
        elif selected_map == "Razonamiento Cuantitativo":
            map_path = FIGURES_DIR / 'map_razonamiento_cuantitativo.html'
        elif selected_map == "Lectura Crítica":
            map_path = FIGURES_DIR / 'map_lectura_critica.html'
        else:  # Estrato
            map_path = FIGURES_DIR / 'cluster_map_estrato.html'
        
        # Mostrar el mapa con folium_static
        if os.path.exists(map_path):
            st.components.v1.html(open(map_path, 'r', encoding='utf-8').read(), height=500)
        else:
            st.error(f"No se encontró el mapa {selected_map}.")
        
        st.markdown("""
        **Interpretación:**
        
        Los mapas coropletas muestran la distribución geográfica de diferentes variables:
        - **Cantidad de Estudiantes**: Visualiza el número de estudiantes por departamento.
        - **Razonamiento Cuantitativo**: Muestra el puntaje promedio en razonamiento cuantitativo por departamento.
        - **Lectura Crítica**: Muestra el puntaje promedio en lectura crítica por departamento.
        - **Estrato**: Visualiza el estrato socioeconómico predominante por departamento.
        
        Los colores más oscuros generalmente indican valores más altos. Esta visualización permite identificar patrones regionales y desigualdades geográficas.
        """)
    
    with tab2:
        st.subheader("Distribución por Departamento")
        
        # Mostrar heatmap de rendimiento por departamento
        heatmap_path = FIGURES_DIR / 'heatmap_rendimiento.html'
        if os.path.exists(heatmap_path):
            st.components.v1.html(open(heatmap_path, 'r', encoding='utf-8').read(), height=500)
        else:
            st.error("No se encontró el mapa de calor de rendimiento por departamento.")
        
        # Mostrar distribución de estratos por departamento
        strata_dist_path = FIGURES_DIR / 'strata_by_department.png'
        if os.path.exists(strata_dist_path):
            st.image(str(strata_dist_path), caption="Distribución de Estratos por Departamento")
        else:
            st.error("No se encontró el gráfico de distribución de estratos por departamento.")
        
        st.markdown("""
        **Interpretación:**
        
        El mapa de calor muestra la relación entre departamentos y rendimiento académico:
        - Colores más cálidos (rojos) indican mejor rendimiento.
        - Colores más fríos (azules) indican peor rendimiento.
        - Este mapa permite identificar rápidamente qué departamentos tienen mejor o peor desempeño en las diferentes pruebas.
        
        La distribución de estratos por departamento muestra la composición socioeconómica de cada región:
        - Las barras apiladas muestran la proporción de cada estrato dentro de cada departamento.
        - Esto permite identificar desigualdades socioeconómicas entre diferentes regiones del país.
        """)
    
    with tab3:
        st.subheader("Comparación Regional")
        
        # Mostrar gráfico de rendimiento por departamento
        performance_path = FIGURES_DIR / 'performance_by_department.png'
        if os.path.exists(performance_path):
            st.image(str(performance_path), caption="Rendimiento Académico por Departamento")
        else:
            st.error("No se encontró el gráfico de rendimiento por departamento.")
        
        # Mostrar visualización interactiva si los datos están disponibles
        if df_geo is not None and 'DEPARTMENT' in df_geo.columns:
            # Seleccionar variable para visualizar
            columns = [col for col in df_geo.columns if col not in ['DEPARTMENT', 'geometry']]
            
            if columns:
                selected_var = st.selectbox(
                    "Seleccione una variable para comparar entre departamentos:",
                    columns,
                    key="geo_var_select"
                )
                
                # Ordenar por la variable seleccionada
                df_sorted = df_geo.sort_values(by=selected_var)
                
                # Crear gráfico de barras
                fig = px.bar(
                    df_sorted, 
                    x='DEPARTMENT',
                    y=selected_var,
                    title=f"{selected_var} por Departamento",
                    color=selected_var,
                    color_continuous_scale='Viridis',
                )
                
                # Actualizar layout para mejor visualización
                fig.update_layout(
                    xaxis_tickangle=-45,
                    xaxis_title="Departamento",
                    yaxis_title=selected_var
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar estadísticas básicas
                st.subheader(f"Estadísticas de {selected_var}")
                stats = df_geo[selected_var].describe()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Promedio", f"{stats['mean']:.2f}")
                
                with col2:
                    st.metric("Mediana", f"{stats['50%']:.2f}")
                
                with col3:
                    st.metric("Mínimo", f"{stats['min']:.2f}")
                
                with col4:
                    st.metric("Máximo", f"{stats['max']:.2f}")
        
        st.markdown("""
        **Interpretación:**
        
        Estas visualizaciones permiten comparar el rendimiento académico entre departamentos:
        - Se puede identificar qué departamentos tienen mejor o peor desempeño en las pruebas.
        - Se pueden explorar posibles relaciones entre el nivel socioeconómico de los departamentos y su rendimiento académico.
        - Los departamentos con valores atípicos (muy por encima o muy por debajo del promedio) pueden ser objeto de análisis más detallado.
        """)

# Función principal para mostrar la sección seleccionada
def show_selected_section():
    """Muestra la sección seleccionada en la barra lateral."""
    # Diccionario de páginas disponibles
    pages = {
        "🏠 Inicio": show_home,
        "🔍 Exploración de Datos": show_eda,
        "🧩 Análisis de Componentes Principales": show_pca,
        "🔄 Análisis de Correspondencias Múltiples": show_mca,
        "🔬 Clustering Jerárquico": show_clustering,
        "📈 Modelos Predictivos": show_predictive_models,
    }
    
    # Agregar análisis geoespacial solo si folium está disponible
    if FOLIUM_AVAILABLE:
        pages["🗺️ Análisis Geoespacial"] = show_geospatial
    else:
        pages["📊 Análisis Geoespacial (Solo Gráficos)"] = show_geospatial
    
    # Selección de página
    selection = st.sidebar.radio("Ir a:", list(pages.keys()))
    
    # Ejecutar función de la página seleccionada
    pages[selection]()

# Función principal
def main():
    """Función principal que controla el flujo del dashboard."""
    # Mostrar encabezado
    show_header()
    
    # Configuración del sidebar
    st.sidebar.title("Navegación")
    
    # Mostrar sección seleccionada
    show_selected_section()
    
    # Información adicional en el sidebar
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Este dashboard fue creado para analizar la relación entre el nivel socioeconómico "
        "y el rendimiento académico en las pruebas Saber Pro en Colombia."
    )
    
    # Botón para ejecutar análisis completo
    st.sidebar.markdown("---")
    st.sidebar.subheader("Ejecutar Análisis")
    
    if st.sidebar.button("Ejecutar Análisis Completo"):
        st.sidebar.warning("Ejecutando análisis completo. Esto puede tardar varios minutos...")
        
        # Importar y ejecutar análisis
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from src.main import run_all_analyses
        
        with st.spinner("Ejecutando análisis..."):
            run_all_analyses(force_reload=True)
        
        st.sidebar.success("¡Análisis completado! Actualiza las visualizaciones.")
        st.experimental_rerun()

if __name__ == "__main__":
    main()
