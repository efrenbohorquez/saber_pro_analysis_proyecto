# Análisis Multivariante: Relación entre Nivel Socioeconómico y Rendimiento Académico en las Pruebas Saber Pro

Este proyecto realiza un análisis estadístico exhaustivo de la relación entre el nivel socioeconómico (NSE) y el rendimiento académico (RA) de los estudiantes que presentaron las Pruebas Saber Pro en Colombia, utilizando datos oficiales del ICFES y aplicando técnicas multivariantes avanzadas.

## Estructura del Proyecto

```
saber_pro_analysis/
│
├── data/                          # Datos crudos y procesados
│   ├── raw/                       # Datos originales sin procesar
│   └── processed/                 # Datos procesados y listos para análisis
│
├── src/                           # Código fuente del proyecto
│   ├── config/                    # Configuraciones y constantes
│   ├── data/                      # Módulos para procesamiento de datos
│   ├── models/                    # Implementación de modelos estadísticos
│   ├── visualization/             # Funciones para visualización de datos
│   ├── utils/                     # Utilidades y funciones auxiliares
│   ├── notebooks/                 # Jupyter notebooks para análisis exploratorio
│   └── tests/                     # Tests unitarios
│
├── dashboard/                     # Aplicación Streamlit para visualización interactiva
│
└── docs/                          # Documentación y reportes generados
    ├── figures/                   # Figuras y visualizaciones generadas
    └── reports/                   # Reportes en formato docx
```

## Requisitos

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Geopandas
- Folium
- Statsmodels
- Factor-analyzer
- Prince
- Scipy
- Docx

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### Análisis de Datos

Para ejecutar el análisis completo:

```bash
python src/main.py
```

### Dashboard Interactivo

Para iniciar el dashboard de Streamlit:

```bash
streamlit run dashboard/app.py
```

## Características Principales

- Análisis exploratorio de datos (EDA)
- Análisis de Componentes Principales (ACP)
- Análisis de Correspondencias Múltiples (ACM)
- Clustering Jerárquico
- Modelos predictivos (Regresión, Random Forest, etc.)
- Visualizaciones interactivas
- Análisis geoespacial
- Generación automática de reportes

## Autor

[Tu Nombre]

## Licencia

Este proyecto está bajo la Licencia [Especificar Licencia].
