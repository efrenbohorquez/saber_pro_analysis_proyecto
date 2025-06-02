# Configuración para Streamlit Cloud

Este archivo contiene las configuraciones necesarias para el despliegue en Streamlit Cloud.

## Archivos de configuración creados:

### `.streamlit/config.toml`
Configuración específica de Streamlit con optimizaciones para el entorno de producción.

### `packages.txt`
Dependencias del sistema necesarias para las librerías geoespaciales (geopandas, folium).

### `requirements.txt`
Dependencias de Python con versiones específicas para asegurar compatibilidad.

## Soluciones implementadas:

1. **Manejo robusto de importaciones**: La aplicación detecta automáticamente si las librerías de mapas están disponibles y se adapta en consecuencia.

2. **Versión específica de streamlit-folium**: Se especifica la versión 0.25.0 que es conocida por ser estable en Streamlit Cloud.

3. **Configuración de servidor**: Optimizada para el entorno de Streamlit Cloud.

4. **Dependencias del sistema**: Se incluyen las librerías necesarias para geopandas y folium.

## Instrucciones de despliegue:

1. Ve a [Streamlit Cloud](https://share.streamlit.io/)
2. Conecta tu repositorio GitHub: `https://github.com/efrenbohorquez/saber_pro_analysis_proyecto`
3. Configura:
   - **Branch**: `main`
   - **Main file path**: `dashboard/app.py`
4. Haz clic en "Deploy"

La aplicación debería desplegarse sin errores de importación.
