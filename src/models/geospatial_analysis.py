"""
Módulo para análisis geoespacial de los datos de Saber Pro.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, HeatMap
import sys
from pathlib import Path
import os
import requests
import json
from io import BytesIO

# Agregar el directorio raíz al path para importar módulos del proyecto
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config.constants import (
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    FIGURE_SIZE,
    FIGURE_DPI,
    FIGURE_FORMAT,
    COLORS,
    STRATA_COLORS
)

def download_colombia_geojson():
    """
    Descarga el archivo GeoJSON de departamentos de Colombia.
    
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame con la geometría de los departamentos.
    """
    # URL del GeoJSON de departamentos de Colombia
    url = "https://gist.githubusercontent.com/john-guerra/43c7656821069d00dcbc/raw/3aadedf47badbdac823b00dbe259f6bc6d9e1899/colombia.geo.json"
    
    try:
        # Descargar GeoJSON
        response = requests.get(url)
        response.raise_for_status()
        
        # Cargar como GeoDataFrame
        gdf = gpd.read_file(BytesIO(response.content))
        
        # Guardar localmente para uso futuro
        geojson_file = PROCESSED_DATA_DIR / "colombia_departments.geojson"
        gdf.to_file(geojson_file, driver="GeoJSON")
        print(f"GeoJSON guardado en {geojson_file}")
        
        return gdf
    
    except Exception as e:
        print(f"Error al descargar GeoJSON: {e}")
        
        # Intentar cargar desde archivo local si existe
        geojson_file = PROCESSED_DATA_DIR / "colombia_departments.geojson"
        if os.path.exists(geojson_file):
            print(f"Cargando GeoJSON desde archivo local: {geojson_file}")
            return gpd.read_file(geojson_file)
        
        return None

def geocode_departments(df):
    """
    Agrega coordenadas aproximadas para los departamentos de Colombia.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        
    Returns:
        pandas.DataFrame: DataFrame con coordenadas agregadas.
    """
    # Diccionario de coordenadas aproximadas de los departamentos de Colombia
    # (latitud, longitud)
    department_coords = {
        'AMAZONAS': (0.8667, -71.4167),
        'ANTIOQUIA': (7.0000, -75.5000),
        'ARAUCA': (7.0762, -70.7105),
        'ATLANTICO': (10.6966, -74.8741),
        'BOGOTA': (4.6097, -74.0817),
        'BOLIVAR': (9.0000, -74.0000),
        'BOYACA': (5.5000, -72.5000),
        'CALDAS': (5.2983, -75.2479),
        'CAQUETA': (1.0000, -74.0000),
        'CASANARE': (5.7589, -71.5723),
        'CAUCA': (2.5000, -76.5000),
        'CESAR': (9.3373, -73.6536),
        'CHOCO': (6.0000, -77.0000),
        'CORDOBA': (8.0000, -75.5000),
        'CUNDINAMARCA': (5.0000, -74.0000),
        'GUAINIA': (2.5854, -68.5247),
        'GUAVIARE': (2.0412, -72.6313),
        'HUILA': (2.5359, -75.5277),
        'LA GUAJIRA': (11.5444, -72.9072),
        'MAGDALENA': (10.4113, -74.4057),
        'META': (3.5000, -73.0000),
        'NARIÑO': (1.5000, -78.0000),
        'NORTE DE SANTANDER': (8.0000, -73.0000),
        'PUTUMAYO': (0.4356, -76.5277),
        'QUINDIO': (4.4389, -75.6668),
        'RISARALDA': (5.0000, -76.0000),
        'SAN ANDRES Y PROVIDENCIA': (12.5567, -81.7185),
        'SANTANDER': (7.0000, -73.2500),
        'SUCRE': (9.0000, -75.0000),
        'TOLIMA': (4.0000, -75.0000),
        'VALLE': (3.8000, -76.5000),
        'VALLE DEL CAUCA': (3.8000, -76.5000),
        'VAUPES': (0.8554, -70.2314),
        'VICHADA': (5.1667, -69.5000)
    }
    
    # Crear copia del DataFrame
    df_geo = df.copy()
    
    # Normalizar nombres de departamentos
    for col in ['ESTU_DEPTO_RESIDE', 'ESTU_INST_DEPARTAMENTO', 'ESTU_PRGM_DEPARTAMENTO', 'ESTU_DEPTO_PRESENTACION']:
        if col in df_geo.columns:
            # Convertir a mayúsculas y eliminar espacios al final
            df_geo[col] = df_geo[col].str.upper().str.strip()
    
    # Agregar coordenadas para departamento de residencia
    if 'ESTU_DEPTO_RESIDE' in df_geo.columns:
        # Crear columnas para latitud y longitud
        df_geo['LATITUD_RESIDE'] = df_geo['ESTU_DEPTO_RESIDE'].map(lambda x: department_coords.get(x, (np.nan, np.nan))[0])
        df_geo['LONGITUD_RESIDE'] = df_geo['ESTU_DEPTO_RESIDE'].map(lambda x: department_coords.get(x, (np.nan, np.nan))[1])
    
    # Agregar coordenadas para departamento de institución
    if 'ESTU_INST_DEPARTAMENTO' in df_geo.columns:
        df_geo['LATITUD_INST'] = df_geo['ESTU_INST_DEPARTAMENTO'].map(lambda x: department_coords.get(x, (np.nan, np.nan))[0])
        df_geo['LONGITUD_INST'] = df_geo['ESTU_INST_DEPARTAMENTO'].map(lambda x: department_coords.get(x, (np.nan, np.nan))[1])
    
    return df_geo

def create_choropleth_map(df, geo_data, value_column, title, output_file=None):
    """
    Crea un mapa de coropletas para visualizar una variable por departamento.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos agregados por departamento.
        geo_data (geopandas.GeoDataFrame): GeoDataFrame con la geometría de los departamentos.
        value_column (str): Columna con los valores a visualizar.
        title (str): Título del mapa.
        output_file (str, optional): Ruta para guardar el mapa HTML. Si es None, no se guarda.
        
    Returns:
        folium.Map: Objeto mapa de Folium.
    """
    # Crear mapa base centrado en Colombia
    m = folium.Map(
        location=[4.5709, -74.2973],
        zoom_start=6,
        tiles='CartoDB positron'
    )
    
    # Agregar título
    title_html = f'''
        <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Agregar mapa de coropletas
    folium.Choropleth(
        geo_data=geo_data,
        name='choropleth',
        data=df,
        columns=['DEPARTAMENTO', value_column],
        key_on='feature.properties.NOMBRE_DPT',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=value_column
    ).add_to(m)
    
    # Agregar capa de control
    folium.LayerControl().add_to(m)
    
    # Guardar si se especifica ruta
    if output_file:
        m.save(output_file)
        print(f"Mapa guardado en {output_file}")
    
    return m

def create_heatmap(df, lat_column, lon_column, value_column=None, radius=15, title='Mapa de calor', output_file=None):
    """
    Crea un mapa de calor para visualizar la densidad de estudiantes.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        lat_column (str): Columna con latitudes.
        lon_column (str): Columna con longitudes.
        value_column (str, optional): Columna con valores para ponderar el mapa de calor. Si es None, no se pondera.
        radius (int, optional): Radio de los puntos en el mapa de calor.
        title (str): Título del mapa.
        output_file (str, optional): Ruta para guardar el mapa HTML. Si es None, no se guarda.
        
    Returns:
        folium.Map: Objeto mapa de Folium.
    """
    # Filtrar filas con coordenadas válidas
    df_valid = df.dropna(subset=[lat_column, lon_column])
    
    # Crear mapa base centrado en Colombia
    m = folium.Map(
        location=[4.5709, -74.2973],
        zoom_start=6,
        tiles='CartoDB positron'
    )
    
    # Agregar título
    title_html = f'''
        <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Preparar datos para el mapa de calor
    if value_column and value_column in df_valid.columns:
        # Usar valores como pesos
        heat_data = [[row[lat_column], row[lon_column], row[value_column]] 
                     for _, row in df_valid.iterrows()]
    else:
        # Sin ponderación
        heat_data = [[row[lat_column], row[lon_column]] 
                     for _, row in df_valid.iterrows()]
    
    # Agregar mapa de calor
    HeatMap(
        heat_data,
        radius=radius,
        blur=15,
        gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
    ).add_to(m)
    
    # Guardar si se especifica ruta
    if output_file:
        m.save(output_file)
        print(f"Mapa guardado en {output_file}")
    
    return m

def create_cluster_map(df, lat_column, lon_column, popup_columns=None, title='Mapa de clusters', output_file=None):
    """
    Crea un mapa de clusters para visualizar la distribución de estudiantes.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        lat_column (str): Columna con latitudes.
        lon_column (str): Columna con longitudes.
        popup_columns (list, optional): Lista de columnas para mostrar en el popup. Si es None, no se muestra popup.
        title (str): Título del mapa.
        output_file (str, optional): Ruta para guardar el mapa HTML. Si es None, no se guarda.
        
    Returns:
        folium.Map: Objeto mapa de Folium.
    """
    # Filtrar filas con coordenadas válidas
    df_valid = df.dropna(subset=[lat_column, lon_column])
    
    # Crear mapa base centrado en Colombia
    m = folium.Map(
        location=[4.5709, -74.2973],
        zoom_start=6,
        tiles='CartoDB positron'
    )
    
    # Agregar título
    title_html = f'''
        <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Crear cluster de marcadores
    marker_cluster = MarkerCluster().add_to(m)
    
    # Agregar marcadores
    for _, row in df_valid.iterrows():
        # Crear popup si se especifican columnas
        if popup_columns:
            popup_text = '<br>'.join([f"{col}: {row[col]}" for col in popup_columns if col in row])
            popup = folium.Popup(popup_text, max_width=300)
        else:
            popup = None
        
        # Agregar marcador
        folium.Marker(
            location=[row[lat_column], row[lon_column]],
            popup=popup
        ).add_to(marker_cluster)
    
    # Guardar si se especifica ruta
    if output_file:
        m.save(output_file)
        print(f"Mapa guardado en {output_file}")
    
    return m

def plot_performance_by_department(df, output_file=None):
    """
    Genera un gráfico de barras del rendimiento académico por departamento.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        output_file (str, optional): Ruta para guardar el gráfico. Si es None, no se guarda.
        
    Returns:
        matplotlib.figure.Figure: Objeto figura con el gráfico.
    """
    # Verificar columnas necesarias
    if 'ESTU_DEPTO_RESIDE' not in df.columns or not any('PUNT' in col for col in df.columns):
        print("Columnas necesarias no encontradas en el DataFrame")
        return None
    
    # Seleccionar columnas de puntaje
    score_columns = [col for col in df.columns if 'PUNT' in col]
    
    # Calcular promedio de puntajes por departamento
    dept_scores = df.groupby('ESTU_DEPTO_RESIDE')[score_columns].mean().reset_index()
    
    # Calcular puntaje global promedio
    dept_scores['PUNTAJE_GLOBAL'] = dept_scores[score_columns].mean(axis=1)
    
    # Ordenar por puntaje global
    dept_scores = dept_scores.sort_values('PUNTAJE_GLOBAL', ascending=False)
    
    # Limitar a los 15 departamentos con mayor número de estudiantes
    top_depts = df['ESTU_DEPTO_RESIDE'].value_counts().nlargest(15).index
    dept_scores = dept_scores[dept_scores['ESTU_DEPTO_RESIDE'].isin(top_depts)]
    
    # Crear figura
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Graficar barras
    sns.barplot(
        x='ESTU_DEPTO_RESIDE',
        y='PUNTAJE_GLOBAL',
        data=dept_scores,
        ax=ax,
        palette='viridis'
    )
    
    # Etiquetas y título
    ax.set_xlabel('Departamento')
    ax.set_ylabel('Puntaje global promedio')
    ax.set_title('Rendimiento académico por departamento')
    
    # Rotar etiquetas
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if output_file:
        plt.savefig(output_file, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"Gráfico guardado en {output_file}")
    
    return fig

def plot_strata_by_department(df, output_file=None):
    """
    Genera un gráfico de barras apiladas de la distribución de estratos por departamento.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        output_file (str, optional): Ruta para guardar el gráfico. Si es None, no se guarda.
        
    Returns:
        matplotlib.figure.Figure: Objeto figura con el gráfico.
    """
    # Verificar columnas necesarias
    if 'ESTU_DEPTO_RESIDE' not in df.columns or 'FAMI_ESTRATOVIVIENDA' not in df.columns:
        print("Columnas necesarias no encontradas en el DataFrame")
        return None
    
    # Calcular distribución de estratos por departamento
    strata_dist = pd.crosstab(
        df['ESTU_DEPTO_RESIDE'],
        df['FAMI_ESTRATOVIVIENDA'],
        normalize='index'
    ) * 100  # Convertir a porcentaje
    
    # Limitar a los 15 departamentos con mayor número de estudiantes
    top_depts = df['ESTU_DEPTO_RESIDE'].value_counts().nlargest(15).index
    strata_dist = strata_dist.loc[top_depts]
    
    # Crear figura
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Graficar barras apiladas
    strata_dist.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        colormap='viridis'
    )
    
    # Etiquetas y título
    ax.set_xlabel('Departamento')
    ax.set_ylabel('Porcentaje')
    ax.set_title('Distribución de estratos por departamento')
    ax.legend(title='Estrato')
    
    # Rotar etiquetas
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if output_file:
        plt.savefig(output_file, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"Gráfico guardado en {output_file}")
    
    return fig

def analyze_geospatial_data(df):
    """
    Realiza análisis geoespacial de los datos de Saber Pro.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos.
        
    Returns:
        pandas.DataFrame: DataFrame con datos geoespaciales.
    """
    # Crear directorio para figuras
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Geocodificar departamentos
    df_geo = geocode_departments(df)
    
    # Descargar GeoJSON de Colombia
    geo_data = download_colombia_geojson()
    
    # Calcular estadísticas por departamento
    dept_stats = df_geo.groupby('ESTU_DEPTO_RESIDE').agg({
        'MOD_RAZONA_CUANTITAT_PUNT': 'mean',
        'MOD_LECTURA_CRITICA_PUNT': 'mean',
        'MOD_INGLES_PUNT': 'mean',
        'ESTU_CONSECUTIVO': 'count'
    }).reset_index()
    
    dept_stats.rename(columns={
        'ESTU_DEPTO_RESIDE': 'DEPARTAMENTO',
        'ESTU_CONSECUTIVO': 'CANTIDAD_ESTUDIANTES'
    }, inplace=True)
    
    # Crear mapas si se pudo cargar el GeoJSON
    if geo_data is not None:
        # Mapa de rendimiento en razonamiento cuantitativo
        quant_map_file = FIGURES_DIR / 'map_razonamiento_cuantitativo.html'
        create_choropleth_map(
            dept_stats,
            geo_data,
            'MOD_RAZONA_CUANTITAT_PUNT',
            'Rendimiento en Razonamiento Cuantitativo por Departamento',
            output_file=quant_map_file
        )
        
        # Mapa de rendimiento en lectura crítica
        reading_map_file = FIGURES_DIR / 'map_lectura_critica.html'
        create_choropleth_map(
            dept_stats,
            geo_data,
            'MOD_LECTURA_CRITICA_PUNT',
            'Rendimiento en Lectura Crítica por Departamento',
            output_file=reading_map_file
        )
        
        # Mapa de cantidad de estudiantes
        students_map_file = FIGURES_DIR / 'map_cantidad_estudiantes.html'
        create_choropleth_map(
            dept_stats,
            geo_data,
            'CANTIDAD_ESTUDIANTES',
            'Cantidad de Estudiantes por Departamento',
            output_file=students_map_file
        )
    
    # Crear mapa de calor de rendimiento académico
    if 'LATITUD_RESIDE' in df_geo.columns and 'LONGITUD_RESIDE' in df_geo.columns:
        heatmap_file = FIGURES_DIR / 'heatmap_rendimiento.html'
        create_heatmap(
            df_geo,
            'LATITUD_RESIDE',
            'LONGITUD_RESIDE',
            'MOD_RAZONA_CUANTITAT_PUNT',
            title='Mapa de Calor de Rendimiento en Razonamiento Cuantitativo',
            output_file=heatmap_file
        )
    
    # Crear mapa de clusters por estrato
    if 'LATITUD_RESIDE' in df_geo.columns and 'LONGITUD_RESIDE' in df_geo.columns:
        cluster_file = FIGURES_DIR / 'cluster_map_estrato.html'
        create_cluster_map(
            df_geo,
            'LATITUD_RESIDE',
            'LONGITUD_RESIDE',
            popup_columns=['FAMI_ESTRATOVIVIENDA', 'MOD_RAZONA_CUANTITAT_PUNT', 'ESTU_DEPTO_RESIDE'],
            title='Distribución de Estudiantes por Estrato',
            output_file=cluster_file
        )
    
    # Generar gráficos estáticos
    performance_dept_file = FIGURES_DIR / 'performance_by_department.png'
    plot_performance_by_department(df_geo, output_file=performance_dept_file)
    
    strata_dept_file = FIGURES_DIR / 'strata_by_department.png'
    plot_strata_by_department(df_geo, output_file=strata_dept_file)
    
    # Guardar datos geoespaciales
    geo_file = PROCESSED_DATA_DIR / 'geospatial_data.csv'
    df_geo.to_csv(geo_file, index=False)
    print(f"Datos geoespaciales guardados en {geo_file}")
    
    # Guardar estadísticas por departamento
    stats_file = PROCESSED_DATA_DIR / 'department_stats.csv'
    dept_stats.to_csv(stats_file, index=False)
    print(f"Estadísticas por departamento guardadas en {stats_file}")
    
    return df_geo

if __name__ == "__main__":
    # Importar módulo de carga de datos
    from src.data.data_loader import get_data
    
    # Cargar datos
    df = get_data()
    
    # Realizar análisis geoespacial
    if df is not None:
        analyze_geospatial_data(df)
