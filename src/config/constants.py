"""
Constantes y configuraciones para el análisis de datos Saber Pro.
"""

# Rutas de archivos
import os
from pathlib import Path

# Directorios principales
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Detectar si estamos en Streamlit Cloud
STREAMLIT_CLOUD = os.getenv('STREAMLIT_SHARING_MODE') == '1' or '/mount/src/' in str(ROOT_DIR)

if STREAMLIT_CLOUD:
    # En Streamlit Cloud, usar rutas relativas desde el directorio de la aplicación
    ROOT_DIR = Path("/mount/src/saber_pro_analysis_proyecto")

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOCS_DIR = ROOT_DIR / "docs"
FIGURES_DIR = DOCS_DIR / "figures"
REPORTS_DIR = DOCS_DIR / "reports"

# Asegurar que los directorios existan
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Archivos de datos
RAW_DATA_FILE = RAW_DATA_DIR / "dataset_dividido_10.csv" # Actualizado para usar el conjunto de datos más pequeño
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "saber_pro_processed_data.csv"
GEOCODED_DATA_FILE = PROCESSED_DATA_DIR / "resultados_geocoded.csv"

# Columnas de interés
# Variables socioeconómicas
SOCIOECONOMIC_VARS = [
    'FAMI_ESTRATOVIVIENDA',
    'FAMI_TIENECOMPUTADOR',
    'FAMI_TIENEINTERNET',
    'FAMI_TIENELAVADORA',
    'FAMI_TIENEAUTOMOVIL',
    'FAMI_EDUCACIONPADRE',
    'FAMI_EDUCACIONMADRE',
    'ESTU_HORASSEMANATRABAJA_NUM',
    'ESTU_VALORMATRICULAUNIVERSIDAD',
    'ESTU_PAGOMATRICULABECA',
    'ESTU_PAGOMATRICULACREDITO',
    'ESTU_PAGOMATRICULAPADRES',
    'ESTU_PAGOMATRICULAPROPIO'
]

# Variables de rendimiento académico
ACADEMIC_VARS = [
    'MOD_RAZONA_CUANTITAT_PUNT',
    'MOD_COMUNI_ESCRITA_PUNT',
    'MOD_COMUNI_ESCRITA_DESEM',
    'MOD_INGLES_DESEM',
    'MOD_LECTURA_CRITICA_PUNT',
    'MOD_INGLES_PUNT',
    'MOD_COMPETEN_\nCIUDADA_PUNT'  # Actualizado para coincidir con el nombre exacto en el archivo CSV
]

# Variables de ubicación geográfica
GEO_VARS = [
    'ESTU_DEPTO_RESIDE',
    'ESTU_MCPIO_RESIDE',
    'ESTU_INST_DEPARTAMENTO',
    'ESTU_INST_MUNICIPIO',
    'ESTU_PRGM_DEPARTAMENTO',
    'ESTU_PRGM_MUNICIPIO',
    'ESTU_DEPTO_PRESENTACION',
    'ESTU_MCPIO_PRESENTACION'
]

# Variables demográficas
DEMOGRAPHIC_VARS = [
    'ESTU_GENERO',
    'ESTU_FECHANACIMIENTO',
    'ESTU_TIPODOCUMENTO'
]

# Variables institucionales
INSTITUTIONAL_VARS = [
    'INST_NOMBRE_INSTITUCION',
    'INST_CARACTER_ACADEMICO',
    'INST_ORIGEN',
    'ESTU_PRGM_ACADEMICO',
    'ESTU_NIVEL_PRGM_ACADEMICO',
    'ESTU_METODO_PRGM',
    'ESTU_NUCLEO_PREGRADO'
]

# Mapeo de niveles educativos para ordenamiento
EDUCATION_LEVELS = {
    'Ninguno': 0,
    'Primaria incompleta': 1,
    'Primaria completa': 2,
    'Secundaria (Bachillerato) incompleta': 3,
    'Secundaria (Bachillerato) completa': 4,
    'Tecnica o tecnologica incompleta': 5,
    'Tecnica o tecnologica completa': 6,
    'Educacionprofesional incompleta': 7,
    'Educacionprofesional completa': 8,
    'Postgrado': 9,
    'No sabe': -1,
    'No Aplica': -1
}

# Mapeo de niveles de inglés
ENGLISH_LEVELS = {
    'A-': 0,
    'A1': 1,
    'A2': 2,
    'B1': 3,
    'B+': 4,
    'B2': 5
}

# Colores para visualizaciones
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'tertiary': '#2ca02c',
    'quaternary': '#d62728',
    'quinary': '#9467bd',
    'senary': '#8c564b',
    'septenary': '#e377c2',
    'octonary': '#7f7f7f',
    'nonary': '#bcbd22',
    'denary': '#17becf'
}

# Paletas de colores para estratos
STRATA_COLORS = {
    'Estrato 1': '#d73027',
    'Estrato 2': '#fc8d59',
    'Estrato 3': '#fee090',
    'Estrato 4': '#e0f3f8',
    'Estrato 5': '#91bfdb',
    'Estrato 6': '#4575b4',
    'Sin Estrato': '#999999'
}

# Configuración para gráficos
FIGURE_SIZE = (12, 8)
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

# Parámetros para análisis
RANDOM_STATE = 42
TEST_SIZE = 0.3
N_COMPONENTS_PCA = 3
N_CLUSTERS_RANGE = range(2, 11)
