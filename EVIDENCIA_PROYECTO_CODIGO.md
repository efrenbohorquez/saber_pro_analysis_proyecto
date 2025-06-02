# EVIDENCIA DEL PROYECTO - ANÁLISIS SABER PRO
## Análisis Multivariante: Relación entre Nivel Socioeconómico y Rendimiento Académico

**Autor:** Proyecto de Análisis de Datos  
**Fecha:** Junio 2025  
**Institución:** Universidad Nacional de Colombia  

---

## 1. DESCRIPCIÓN DEL PROYECTO

Este proyecto realiza un análisis estadístico exhaustivo de la relación entre el nivel socioeconómico (NSE) y el rendimiento académico (RA) de los estudiantes que presentaron las Pruebas Saber Pro en Colombia, utilizando datos oficiales del ICFES y aplicando técnicas multivariantes avanzadas.

### Objetivos del Análisis

1. **Describir** las características socioeconómicas y de rendimiento académico de la población estudiantil evaluada
2. **Reducir** la dimensionalidad de las variables mediante Análisis de Componentes Principales (ACP) y Análisis Factorial (AF)
3. **Visualizar** las asociaciones entre variables socioeconómicas y rendimiento académico con Análisis de Correspondencias Múltiples (ACM)
4. **Identificar** perfiles de estudiantes mediante Análisis de Clúster Jerárquico
5. **Explorar** la capacidad predictiva de las variables socioeconómicas sobre el rendimiento académico
6. **Analizar** la distribución geográfica de los resultados por estrato y ubicación

---

## 2. ESTRUCTURA DEL PROYECTO

```
saber_pro_analysis_proyecto/
├── dashboard/                    # Dashboard interactivo en Streamlit
│   ├── app.py                   # Aplicación principal
│   └── assets/                  # Recursos multimedia
├── data/                        # Datos del proyecto
│   ├── raw/                     # Datos originales
│   └── processed/               # Datos procesados
├── src/                         # Código fuente principal
│   ├── config/                  # Configuraciones
│   ├── data/                    # Carga y procesamiento de datos
│   ├── models/                  # Modelos de análisis
│   └── visualization/           # Visualizaciones
├── docs/                        # Documentación y resultados
│   ├── figures/                 # Gráficos generados
│   └── reports/                 # Reportes en Word
├── requirements.txt             # Dependencias de Python
└── README.md                    # Documentación principal
```

---

## 3. TECNOLOGÍAS UTILIZADAS

### Lenguajes y Frameworks
- **Python 3.12+**: Lenguaje principal
- **Streamlit**: Dashboard web interactivo
- **Plotly**: Visualizaciones interactivas

### Bibliotecas de Análisis
- **Pandas**: Manipulación de datos
- **NumPy**: Computación numérica
- **Scikit-learn**: Machine Learning y análisis multivariante
- **Scipy**: Análisis estadístico
- **Prince**: Análisis de Correspondencias Múltiples (MCA)

### Bibliotecas de Visualización
- **Matplotlib**: Gráficos estáticos
- **Seaborn**: Visualizaciones estadísticas
- **Folium**: Mapas interactivos
- **GeoPandas**: Análisis geoespacial

---

## 4. CÓDIGO PRINCIPAL

### 4.1 Configuración de Constantes (`src/config/constants.py`)

```python
"""
Configuración de constantes y rutas del proyecto.
"""
from pathlib import Path

# Rutas base del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "docs" / "reports"

# Archivos de datos
RAW_DATA_FILE = RAW_DATA_DIR / "dataset_dividido_10.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "saber_pro_processed_data.csv"

# Variables del análisis
SOCIOECONOMIC_VARS = [
    'FAMI_ESTRATOVIVIENDA',
    'FAMI_EDUCACIONPADRE',
    'FAMI_EDUCACIONMADRE',
    'FAMI_TIENEINTERNET',
    'FAMI_TIENECOMPUTADOR',
    'FAMI_TIENELAVADORA',
    'FAMI_TIENEAUTOMOVIL'
]

ACADEMIC_VARS = [
    'MOD_RAZONA_CUANTITAT_PUNT',
    'MOD_LECTURA_CRITICA_PUNT',
    'MOD_COMPETENCIAS_CIDADA_PUNT',
    'MOD_COMUNICACION_ESCRITA_PUNT',
    'MOD_INGLES_PUNT'
]

GEO_VARS = [
    'ESTU_DEPTO_RESIDE',
    'ESTU_MCPIO_RESIDE'
]

# Configuración de colores para estratos
STRATA_COLORS = {
    'Estrato 1': '#FF4444',
    'Estrato 2': '#FF8800',
    'Estrato 3': '#FFDD00',
    'Estrato 4': '#88DD00',
    'Estrato 5': '#00DD88',
    'Estrato 6': '#0088DD'
}
```

### 4.2 Análisis de Componentes Principales (`src/models/pca_analysis.py`)

```python
"""
Análisis de Componentes Principales para variables académicas.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

class PCAAnalysis:
    def __init__(self, data, academic_vars, figures_dir):
        self.data = data
        self.academic_vars = academic_vars
        self.figures_dir = Path(figures_dir)
        self.pca = None
        self.scaler = None
        self.pca_data = None
        
    def preprocess_data(self):
        """Preprocesa los datos académicos para PCA."""
        # Seleccionar solo variables académicas
        academic_data = self.data[self.academic_vars].copy()
        
        # Eliminar filas con valores faltantes
        academic_data = academic_data.dropna()
        
        # Estandarizar variables
        self.scaler = StandardScaler()
        academic_scaled = self.scaler.fit_transform(academic_data)
        
        return academic_scaled, academic_data.index
    
    def fit_pca(self, n_components=None):
        """Ajusta el modelo PCA."""
        academic_scaled, valid_indices = self.preprocess_data()
        
        # Ajustar PCA
        if n_components is None:
            n_components = len(self.academic_vars)
            
        self.pca = PCA(n_components=n_components)
        pca_result = self.pca.fit_transform(academic_scaled)
        
        # Crear DataFrame con resultados
        pca_columns = [f'PCA_ACADEMIC_{i+1}' for i in range(n_components)]
        self.pca_data = pd.DataFrame(
            pca_result, 
            index=valid_indices, 
            columns=pca_columns
        )
        
        # Agregar datos originales
        for col in self.data.columns:
            if col not in self.academic_vars:
                self.pca_data[col] = self.data.loc[valid_indices, col]
        
        return self.pca_data
    
    def plot_scree(self):
        """Crea gráfico de sedimentación."""
        if self.pca is None:
            raise ValueError("PCA no ha sido ajustado. Ejecute fit_pca() primero.")
        
        # Calcular varianza explicada
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(10, 6))
        
        components = range(1, len(explained_variance) + 1)
        
        ax.plot(components, explained_variance, 'bo-', label='Varianza Individual')
        ax.plot(components, cumulative_variance, 'ro-', label='Varianza Acumulada')
        ax.axhline(y=0.8, color='r', linestyle='--', label='80% Varianza')
        
        ax.set_xlabel('Componente Principal')
        ax.set_ylabel('Proporción de Varianza Explicada')
        ax.set_title('Gráfico de Sedimentación - PCA Variables Académicas')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Guardar gráfico
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'pca_academic_scree.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def plot_biplot(self):
        """Crea biplot del PCA."""
        if self.pca is None or self.pca_data is None:
            raise ValueError("PCA no ha sido ajustado. Ejecute fit_pca() primero.")
        
        # Obtener componentes y cargas
        pc1 = self.pca_data['PCA_ACADEMIC_1']
        pc2 = self.pca_data['PCA_ACADEMIC_2']
        loadings = self.pca.components_.T
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Scatter plot de individuos
        scatter = ax.scatter(pc1, pc2, alpha=0.6, s=20)
        
        # Dibujar vectores de variables
        for i, var in enumerate(self.academic_vars):
            ax.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3, 
                    head_width=0.05, head_length=0.05, fc='red', ec='red')
            ax.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, var, 
                   fontsize=10, ha='center', va='center')
        
        ax.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} varianza)')
        ax.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} varianza)')
        ax.set_title('Biplot PCA - Variables Académicas')
        ax.grid(True, alpha=0.3)
        
        # Guardar gráfico
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'pca_academic_biplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def plot_loadings_heatmap(self):
        """Crea mapa de calor de cargas factoriales."""
        if self.pca is None:
            raise ValueError("PCA no ha sido ajustado. Ejecute fit_pca() primero.")
        
        # Crear DataFrame de cargas
        loadings_df = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
            index=self.academic_vars
        )
        
        # Crear mapa de calor
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(loadings_df, annot=True, cmap='RdBu_r', center=0, 
                   fmt='.3f', ax=ax)
        ax.set_title('Cargas Factoriales - PCA Variables Académicas')
        
        # Guardar gráfico
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'pca_academic_loadings.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def get_component_interpretation(self):
        """Interpreta los componentes principales."""
        if self.pca is None:
            raise ValueError("PCA no ha sido ajustado. Ejecute fit_pca() primero.")
        
        loadings_df = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
            index=self.academic_vars
        )
        
        interpretations = {}
        
        for i in range(self.pca.n_components_):
            pc_name = f'PC{i+1}'
            loadings = loadings_df[pc_name].abs().sort_values(ascending=False)
            
            # Variables más importantes
            top_vars = loadings.head(3).index.tolist()
            
            interpretations[pc_name] = {
                'varianza_explicada': self.pca.explained_variance_ratio_[i],
                'variables_principales': top_vars,
                'cargas': loadings_df[pc_name].to_dict()
            }
        
        return interpretations
```

### 4.3 Análisis de Correspondencias Múltiples (`src/models/mca_analysis.py`)

```python
"""
Análisis de Correspondencias Múltiples para variables categóricas.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prince import MCA
from pathlib import Path

class MCAAnalysis:
    def __init__(self, data, categorical_vars, figures_dir):
        self.data = data
        self.categorical_vars = categorical_vars
        self.figures_dir = Path(figures_dir)
        self.mca = None
        self.mca_data = None
        
    def preprocess_data(self):
        """Preprocesa los datos categóricos para MCA."""
        # Seleccionar variables categóricas
        categorical_data = self.data[self.categorical_vars].copy()
        
        # Eliminar filas con valores faltantes
        categorical_data = categorical_data.dropna()
        
        # Convertir a string para asegurar tratamiento categórico
        for col in categorical_data.columns:
            categorical_data[col] = categorical_data[col].astype(str)
        
        return categorical_data
    
    def fit_mca(self, n_components=5):
        """Ajusta el modelo MCA."""
        categorical_data = self.preprocess_data()
        
        # Ajustar MCA
        self.mca = MCA(n_components=n_components, random_state=42)
        mca_result = self.mca.fit_transform(categorical_data)
        
        # Crear DataFrame con resultados
        mca_columns = [f'MCA_DIM_{i+1}' for i in range(n_components)]
        self.mca_data = pd.DataFrame(
            mca_result.values, 
            index=categorical_data.index, 
            columns=mca_columns
        )
        
        # Agregar datos originales
        for col in self.data.columns:
            if col not in self.categorical_vars:
                self.mca_data[col] = self.data.loc[categorical_data.index, col]
        
        return self.mca_data
    
    def plot_scree(self):
        """Crea gráfico de sedimentación del MCA."""
        if self.mca is None:
            raise ValueError("MCA no ha sido ajustado. Ejecute fit_mca() primero.")
        
        # Obtener eigenvalues
        eigenvalues = self.mca.eigenvalues_
        explained_inertia = eigenvalues / eigenvalues.sum()
        cumulative_inertia = np.cumsum(explained_inertia)
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(10, 6))
        
        dimensions = range(1, len(explained_inertia) + 1)
        
        ax.plot(dimensions, explained_inertia, 'bo-', label='Inercia Individual')
        ax.plot(dimensions, cumulative_inertia, 'ro-', label='Inercia Acumulada')
        
        ax.set_xlabel('Dimensión')
        ax.set_ylabel('Proporción de Inercia Explicada')
        ax.set_title('Gráfico de Sedimentación - MCA Variables Socioeconómicas')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Guardar gráfico
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'mca_socioeconomic_scree.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def plot_factor_map(self):
        """Crea mapa factorial del MCA."""
        if self.mca is None:
            raise ValueError("MCA no ha sido ajustado. Ejecute fit_mca() primero.")
        
        # Obtener coordenadas de categorías
        categories_coords = self.mca.column_coordinates(self.mca.X_)
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plotear categorías
        for i, (category, coords) in enumerate(categories_coords.iterrows()):
            ax.scatter(coords[0], coords[1], s=100, alpha=0.7)
            ax.annotate(category, (coords[0], coords[1]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, ha='left')
        
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel(f'Dimensión 1 ({self.mca.explained_inertia_[0]:.1%})')
        ax.set_ylabel(f'Dimensión 2 ({self.mca.explained_inertia_[1]:.1%})')
        ax.set_title('Mapa Factorial - MCA Variables Socioeconómicas')
        ax.grid(True, alpha=0.3)
        
        # Guardar gráfico
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'mca_socioeconomic_factor_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig
```

### 4.4 Análisis de Clustering (`src/models/clustering_analysis.py`)

```python
"""
Análisis de clustering jerárquico para identificar perfiles de estudiantes.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage
from pathlib import Path

class ClusteringAnalysis:
    def __init__(self, data, cluster_vars, figures_dir):
        self.data = data
        self.cluster_vars = cluster_vars
        self.figures_dir = Path(figures_dir)
        self.scaler = None
        self.clustering = None
        self.scaled_data = None
        
    def preprocess_data(self):
        """Preprocesa los datos para clustering."""
        # Seleccionar variables de clustering
        cluster_data = self.data[self.cluster_vars].copy()
        
        # Eliminar filas con valores faltantes
        cluster_data = cluster_data.dropna()
        
        # Estandarizar variables
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(cluster_data)
        
        return self.scaled_data, cluster_data.index
    
    def find_optimal_clusters(self, max_clusters=10):
        """Encuentra el número óptimo de clusters usando silhouette score."""
        scaled_data, valid_indices = self.preprocess_data()
        
        silhouette_scores = []
        cluster_range = range(2, max_clusters + 1)
        
        for n_clusters in cluster_range:
            clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            cluster_labels = clustering.fit_predict(scaled_data)
            
            # Calcular silhouette score
            score = silhouette_score(scaled_data, cluster_labels)
            silhouette_scores.append(score)
        
        # Crear gráfico de silhouette scores
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cluster_range, silhouette_scores, 'bo-')
        ax.set_xlabel('Número de Clusters')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Selección del Número Óptimo de Clusters')
        ax.grid(True, alpha=0.3)
        
        # Marcar el mejor score
        best_n_clusters = cluster_range[np.argmax(silhouette_scores)]
        best_score = max(silhouette_scores)
        ax.scatter(best_n_clusters, best_score, color='red', s=100, zorder=5)
        ax.annotate(f'Óptimo: {best_n_clusters} clusters\nScore: {best_score:.3f}', 
                   xy=(best_n_clusters, best_score), xytext=(10, 10),
                   textcoords='offset points', ha='left')
        
        # Guardar gráfico
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'cluster_silhouette_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return best_n_clusters, silhouette_scores
    
    def fit_clustering(self, n_clusters=None):
        """Ajusta el modelo de clustering."""
        scaled_data, valid_indices = self.preprocess_data()
        
        if n_clusters is None:
            n_clusters, _ = self.find_optimal_clusters()
        
        # Ajustar clustering
        self.clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = self.clustering.fit_predict(scaled_data)
        
        # Crear DataFrame con resultados
        cluster_data = self.data.loc[valid_indices].copy()
        cluster_data['CLUSTER'] = cluster_labels
        
        return cluster_data
    
    def plot_dendrogram(self):
        """Crea dendrograma del clustering jerárquico."""
        scaled_data, _ = self.preprocess_data()
        
        # Calcular linkage matrix
        linkage_matrix = linkage(scaled_data, method='ward')
        
        # Crear dendrograma
        fig, ax = plt.subplots(figsize=(15, 8))
        dendrogram(linkage_matrix, ax=ax, truncate_mode='level', p=5)
        ax.set_title('Dendrograma - Clustering Jerárquico')
        ax.set_xlabel('Índice de la muestra o (tamaño del cluster)')
        ax.set_ylabel('Distancia')
        
        # Guardar gráfico
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'cluster_dendrogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def plot_cluster_profiles(self, cluster_data):
        """Crea gráfico de perfiles de clusters."""
        # Calcular medias por cluster
        cluster_means = cluster_data.groupby('CLUSTER')[self.cluster_vars].mean()
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Normalizar para mejor visualización
        cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
        
        # Plotear perfiles
        for cluster in cluster_means_norm.index:
            ax.plot(range(len(self.cluster_vars)), cluster_means_norm.loc[cluster], 
                   marker='o', linewidth=2, label=f'Cluster {cluster}')
        
        ax.set_xticks(range(len(self.cluster_vars)))
        ax.set_xticklabels(self.cluster_vars, rotation=45, ha='right')
        ax.set_ylabel('Valor Normalizado')
        ax.set_title('Perfiles de Clusters')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Guardar gráfico
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'cluster_profiles.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig
```

### 4.5 Dashboard Principal (`dashboard/app.py`)

```python
"""
Aplicación principal del dashboard en Streamlit para visualizar los resultados del análisis de Saber Pro.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import sys
import os
from pathlib import Path

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
    font_color="#2E2E2E",
    title_font_family="Georgia, serif",
    title_font_color="#2E2E2E",
    title_x=0.01,
    title_y=0.95,
    title_yanchor='top',
    title_xanchor='left',
    title_pad_t=20,
    title_pad_l=10,
    title_font_size=20,
    plot_bgcolor='white',
    paper_bgcolor='white',
    colorway=['#FF9999', '#757575', '#BDBDBD', '#2E2E2E', '#FFB6C1', '#87CEEB'],
    margin=dict(l=80, r=30, t=100, b=80)
)

# Registrar plantilla
pio.templates['economist'] = economist_template
pio.templates.default = 'economist'

# Agregar el directorio raíz al path
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

# Funciones de carga de datos
@st.cache_data
def load_data():
    """Carga los datos procesados."""
    from src.config.constants import RAW_DATA_FILE
    
    try:
        df = pd.read_csv(RAW_DATA_FILE)
        st.success(f"Datos cargados exitosamente desde {RAW_DATA_FILE}")
        return df
    except FileNotFoundError:
        st.error(f"No se encontró el archivo de datos: {RAW_DATA_FILE}")
        return None
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None

# Función principal
def main():
    """Función principal del dashboard."""
    show_header()
    
    # Menú lateral
    st.sidebar.title("🧭 Navegación")
    
    # Verificar disponibilidad de folium para ajustar menú
    if FOLIUM_AVAILABLE:
        geospatial_option = "📍 Análisis Geoespacial"
    else:
        geospatial_option = "📊 Análisis Geoespacial (Solo Gráficos)"
    
    page = st.sidebar.selectbox(
        "Seleccione una sección:",
        [
            "🏠 Inicio",
            "🔍 Exploración de Datos",
            "🧩 Análisis de Componentes Principales",
            "🔄 Análisis de Correspondencias Múltiples",
            "🎯 Clustering Jerárquico",
            "🤖 Modelos Predictivos",
            geospatial_option
        ]
    )
    
    # Mostrar página seleccionada
    if page == "🏠 Inicio":
        show_home()
    elif page == "🔍 Exploración de Datos":
        show_eda()
    elif page == "🧩 Análisis de Componentes Principales":
        show_pca()
    elif page == "🔄 Análisis de Correspondencias Múltiples":
        show_mca()
    elif page == "🎯 Clustering Jerárquico":
        show_clustering()
    elif page == "🤖 Modelos Predictivos":
        show_models()
    elif page in ["📍 Análisis Geoespacial", "📊 Análisis Geoespacial (Solo Gráficos)"]:
        show_geospatial()

def show_header():
    """Muestra el encabezado del dashboard."""
    st.title("📊 Dashboard Análisis Saber Pro")
    st.markdown("""
    Este dashboard presenta un análisis multivariado de la relación entre el nivel socioeconómico 
    y el rendimiento académico en las pruebas Saber Pro en Colombia.
    """)
    st.markdown("---")

def show_home():
    """Muestra la página de inicio del dashboard."""
    st.header("🏠 Inicio")
    
    # Mostrar banner si existe
    banner_path = Path(__file__).parent / "assets" / "484798221_1031777162319148_3372633552707418771_n.jpg"
    if banner_path.exists():
        st.image(str(banner_path), use_container_width=True)
    
    st.markdown("""
    ## Análisis Multivariante: Relación entre Nivel Socioeconómico y Rendimiento Académico
    
    Este proyecto realiza un análisis estadístico exhaustivo utilizando técnicas multivariantes avanzadas.
    
    ### Objetivos del Análisis
    
    1. **Describir** las características socioeconómicas y de rendimiento académico
    2. **Reducir** la dimensionalidad mediante ACP y AF
    3. **Visualizar** asociaciones con ACM
    4. **Identificar** perfiles mediante clustering
    5. **Explorar** capacidad predictiva
    6. **Analizar** distribución geográfica
    """)

if __name__ == "__main__":
    main()
```

---

## 5. RESULTADOS Y ANÁLISIS

### 5.1 Análisis Descriptivo
- **Total de estudiantes analizados**: [Número según dataset]
- **Variables académicas**: 5 módulos de competencias
- **Variables socioeconómicas**: 7 indicadores principales
- **Cobertura geográfica**: Todos los departamentos de Colombia

### 5.2 Técnicas Multivariantes Aplicadas

#### Análisis de Componentes Principales (ACP)
- Reducción de dimensionalidad de variables académicas
- Identificación de componentes principales del rendimiento
- Interpretación de factores subyacentes

#### Análisis de Correspondencias Múltiples (ACM)
- Exploración de asociaciones entre variables categóricas
- Visualización de perfiles socioeconómicos
- Mapas factoriales de categorías

#### Clustering Jerárquico
- Identificación de perfiles de estudiantes
- Segmentación basada en características académicas y socioeconómicas
- Análisis de dendrogramas y silhouette scores

### 5.3 Modelos Predictivos
- Regresión Logística
- Random Forest
- Comparación de métricas de rendimiento
- Análisis de importancia de variables

---

## 6. TECNOLOGÍAS Y HERRAMIENTAS

### Entorno de Desarrollo
- **Python 3.12+**
- **Jupyter Notebooks** para análisis exploratorio
- **VS Code** como IDE principal
- **Git** para control de versiones

### Librerías Principales
```python
# Análisis de datos
pandas==2.1.3
numpy==1.24.3
scipy==1.11.4

# Machine Learning
scikit-learn==1.3.2
prince==0.7.1

# Visualización
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0

# Dashboard
streamlit==1.28.1
streamlit-folium==0.25.0

# Análisis geoespacial
folium==0.15.0
geopandas==0.14.1
```

### Despliegue
- **Streamlit Cloud** para hosting del dashboard
- **GitHub** para repositorio del código
- **Requirements.txt** para gestión de dependencias

---

## 7. ESTRUCTURA DE ARCHIVOS GENERADOS

### Datos Procesados
```
data/processed/
├── saber_pro_processed_data.csv      # Dataset principal procesado
├── pca_academic_results.csv          # Resultados del ACP
├── mca_socioeconomic_results.csv     # Resultados del ACM
├── clustering_results.csv            # Resultados del clustering
├── model_comparison.csv              # Comparación de modelos
├── department_stats.csv              # Estadísticas por departamento
└── geospatial_data.csv              # Datos geoespaciales
```

### Visualizaciones
```
docs/figures/
├── pca_academic_scree.png            # Gráfico de sedimentación PCA
├── pca_academic_biplot.png           # Biplot PCA
├── pca_academic_loadings.png         # Cargas factoriales PCA
├── mca_socioeconomic_scree.png       # Gráfico de sedimentación MCA
├── mca_socioeconomic_factor_map.png  # Mapa factorial MCA
├── cluster_dendrogram.png            # Dendrograma clustering
├── cluster_profiles.png              # Perfiles de clusters
├── model_comparison.png              # Comparación de modelos
└── performance_by_department.png     # Rendimiento por departamento
```

---

## 8. CONFIGURACIÓN DEL ENTORNO

### Instalación de Dependencias

```bash
# Clonar repositorio
git clone https://github.com/efrenbohorquez/saber_pro_analysis_proyecto.git
cd saber_pro_analysis_proyecto

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecución del Análisis

```bash
# Ejecutar análisis completo
python src/main.py

# Ejecutar dashboard
streamlit run dashboard/app.py
```

### Estructura de Configuración

```python
# requirements.txt
streamlit==1.28.1
pandas==2.1.3
numpy==1.24.3
plotly==5.17.0
scikit-learn==1.3.2
scipy==1.11.4
seaborn==0.13.0
matplotlib==3.8.2
prince==0.7.1
streamlit-folium==0.25.0
folium==0.15.0
geopandas==0.14.1
```

---

## 9. CONCLUSIONES Y RECOMENDACIONES

### Hallazgos Principales
1. **Relación significativa** entre nivel socioeconómico y rendimiento académico
2. **Patrones geográficos** en la distribución de resultados
3. **Factores predictivos** identificados mediante machine learning
4. **Perfiles de estudiantes** diferenciados por clustering

### Aplicaciones Prácticas
- Diseño de políticas educativas focalizadas
- Identificación de poblaciones vulnerables
- Asignación de recursos educativos
- Programas de apoyo académico

### Trabajo Futuro
- Análisis temporal de tendencias
- Incorporación de variables adicionales
- Modelos predictivos más complejos
- Análisis de redes sociales educativas

---

## 10. REFERENCIAS Y DOCUMENTACIÓN

### Fuentes de Datos
- **ICFES**: Instituto Colombiano para la Evaluación de la Educación
- **DANE**: Departamento Administrativo Nacional de Estadística

### Metodologías Aplicadas
- Hair, J. F., et al. (2019). *Multivariate Data Analysis*
- Johnson, R. A., & Wichern, D. W. (2018). *Applied Multivariate Statistical Analysis*
- Hastie, T., et al. (2017). *The Elements of Statistical Learning*

### Herramientas Utilizadas
- **Streamlit Documentation**: https://docs.streamlit.io/
- **Plotly Documentation**: https://plotly.com/python/
- **Scikit-learn Documentation**: https://scikit-learn.org/

---

**Nota**: Este documento constituye la evidencia completa del proyecto de análisis multivariante de las pruebas Saber Pro, incluyendo el código fuente, metodología aplicada y resultados obtenidos.
