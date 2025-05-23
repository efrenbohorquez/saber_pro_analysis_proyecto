# Análisis Multivariante: Relación entre Nivel Socioeconómico y Rendimiento Académico en las Pruebas Saber Pro

Este proyecto realiza un análisis estadístico exhaustivo de la relación entre el nivel socioeconómico (NSE) y el rendimiento académico (RA) de los estudiantes que presentaron las Pruebas Saber Pro en Colombia, utilizando datos oficiales del ICFES y aplicando técnicas multivariantes avanzadas.

## Actualización Reciente

Hemos realizado varias mejoras para optimizar el rendimiento y corregir errores:

- Ahora utilizamos un conjunto de datos reducido (`dataset_dividido_10.csv`) para mejor rendimiento
- Se han corregido los problemas de carga de datos y manejo de columnas categóricas
- Se configuró un repositorio Git para facilitar la colaboración
- Se actualizó la aplicación Streamlit para mejorar la experiencia de usuario

Ver [MEJORAS.md](MEJORAS.md) para más detalles sobre las correcciones implementadas.

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

### Ejecutar el Dashboard

Para iniciar la aplicación de visualización interactiva:

```bash
cd dashboard
streamlit run app.py
```

### Análisis de Datos

El proyecto utiliza varios algoritmos de análisis multivariante:

- Análisis de Componentes Principales (PCA)
- Análisis de Correspondencias Múltiples (MCA)
- Regresiones multivariantes
- Análisis de Conglomerados (Clustering)
- Visualización geoespacial

## Contribución

1. Haz un Fork del repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Haz commit de tus cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Haz push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crea un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT.

## Agradecimientos

- ICFES por proporcionar los datos de las pruebas Saber Pro
- Contribuidores y asesores del proyecto
