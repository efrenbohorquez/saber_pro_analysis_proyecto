# Configuración para Streamlit Cloud

## Archivos Necesarios para el Despliegue

### Datos Requeridos
- `data/raw/dataset_dividido_10.csv` - Archivo principal de datos (debe estar incluido en el repositorio)
- El archivo está configurado como excepción en `.gitignore` para asegurar que se suba al repositorio

### Archivos de Configuración
- `requirements.txt` - Dependencias de Python
- `packages.txt` - Dependencias del sistema (para geopandas/folium)
- `.streamlit/config.toml` - Configuración específica de Streamlit

### Manejo de Rutas
El código ha sido actualizado para manejar diferentes estructuras de rutas:
- Entorno local: Rutas relativas normales
- Streamlit Cloud: Rutas que empiezan con `/mount/src/`

### Manejo de Dependencias Opcionales
- `folium` y `streamlit-folium`: Si no están disponibles, la aplicación funciona sin mapas interactivos
- Se muestran gráficos estáticos como alternativa

## Verificación del Despliegue

Si el archivo de datos no se encuentra:
1. Verificar que `dataset_dividido_10.csv` está en el repositorio
2. Confirmar que `.gitignore` tiene la excepción para este archivo
3. Asegurar que el archivo se subió correctamente a GitHub
4. Verificar que Streamlit Cloud tiene acceso a todos los archivos del repositorio

## Estructura de Archivos Esperada en Streamlit Cloud

```
/mount/src/saber_pro_analysis_proyecto/
├── data/
│   ├── raw/
│   │   └── dataset_dividido_10.csv
│   └── processed/
├── src/
├── dashboard/
│   └── app.py
├── requirements.txt
├── packages.txt
└── .streamlit/
    └── config.toml
```
