"""
Módulo para cargar y procesar los datos de las pruebas Saber Pro.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys

# Agregar el directorio raíz al path para importar módulos del proyecto
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config.constants import (
    RAW_DATA_FILE, 
    PROCESSED_DATA_FILE,
    SOCIOECONOMIC_VARS,
    ACADEMIC_VARS,
    GEO_VARS,
    DEMOGRAPHIC_VARS,
    INSTITUTIONAL_VARS,
    EDUCATION_LEVELS,
    ENGLISH_LEVELS
)

def load_raw_data():
    """
    Carga los datos crudos desde el archivo de origen (CSV o Excel).
    
    Returns:
        pandas.DataFrame: DataFrame con los datos crudos.
    """
    try:
        # Determinar la extensión del archivo
        file_extension = str(RAW_DATA_FILE).lower().split('.')[-1]
        
        if file_extension == 'csv':
            df = pd.read_csv(RAW_DATA_FILE)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(RAW_DATA_FILE)
        else:
            print(f"Formato de archivo no soportado: {file_extension}")
            return None
            
        print(f"Datos cargados exitosamente desde {RAW_DATA_FILE}. Dimensiones: {df.shape}")
        return df
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None

def clean_data(df):
    """
    Limpia y preprocesa los datos crudos.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos crudos.
        
    Returns:
        pandas.DataFrame: DataFrame con los datos limpios.
    """
    if df is None:
        return None
    
    # Crear una copia para no modificar el original
    df_clean = df.copy()
    
    # Convertir columnas a tipos de datos apropiados
    # Convertir fechas
    try:
        df_clean['ESTU_FECHANACIMIENTO'] = pd.to_datetime(df_clean['ESTU_FECHANACIMIENTO'], errors='coerce')
    except:
        print("Error al convertir fechas de nacimiento")
    
    # Calcular edad aproximada basada en el periodo de la prueba
    try:
        # Extraer el año del periodo (asumiendo formato YYYYN donde N es el número del periodo)
        df_clean['AÑO_PRUEBA'] = df_clean['PERIODO'].astype(str).str[:4].astype(int)
        # Calcular edad
        df_clean['EDAD'] = df_clean['AÑO_PRUEBA'] - df_clean['ESTU_FECHANACIMIENTO'].dt.year
        # Filtrar edades improbables
        df_clean = df_clean[(df_clean['EDAD'] >= 16) & (df_clean['EDAD'] <= 80)]
    except:
        print("Error al calcular edades")
    
    # Limpiar y estandarizar nombres de departamentos y municipios
    for col in GEO_VARS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].str.strip().str.upper()
    
    # Convertir variables numéricas
    numeric_vars = [var for var in ACADEMIC_VARS if 'PUNT' in var]
    for var in numeric_vars:
        if var in df_clean.columns:
            df_clean[var] = pd.to_numeric(df_clean[var], errors='coerce')
    
    # Manejar valores faltantes en variables académicas
    for var in ACADEMIC_VARS:
        if var in df_clean.columns:
            # Para variables de puntaje, imputar con la media
            if 'PUNT' in var:
                df_clean[var] = df_clean[var].fillna(df_clean[var].mean())
            # Para variables de desempeño categóricas, imputar con la moda
            else:
                df_clean[var] = df_clean[var].fillna(df_clean[var].mode()[0])
    
    # Estandarizar estratos
    if 'FAMI_ESTRATOVIVIENDA' in df_clean.columns:
        # Asegurar formato consistente (Estrato X)
        df_clean['FAMI_ESTRATOVIVIENDA'] = df_clean['FAMI_ESTRATOVIVIENDA'].astype(str)
        df_clean['FAMI_ESTRATOVIVIENDA'] = df_clean['FAMI_ESTRATOVIVIENDA'].apply(
            lambda x: f"Estrato {x}" if x.isdigit() else x
        )
        # Manejar valores faltantes o inválidos
        valid_strata = [f"Estrato {i}" for i in range(1, 7)]
        df_clean.loc[~df_clean['FAMI_ESTRATOVIVIENDA'].isin(valid_strata), 'FAMI_ESTRATOVIVIENDA'] = 'Sin Estrato'
    
    # Convertir variables binarias (Sí/No) a 1/0
    binary_vars = ['FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET', 'FAMI_TIENELAVADORA', 'FAMI_TIENEAUTOMOVIL',
                  'ESTU_PAGOMATRICULABECA', 'ESTU_PAGOMATRICULACREDITO', 'ESTU_PAGOMATRICULAPADRES', 'ESTU_PAGOMATRICULAPROPIO']
    
    for var in binary_vars:
        if var in df_clean.columns:
            df_clean[var] = df_clean[var].map({'Si': 1, 'No': 0, 'si': 1, 'no': 0, 'SI': 1, 'NO': 0})
            df_clean[var] = df_clean[var].fillna(0)
    
    # Ordenar niveles educativos
    for var in ['FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE']:
        if var in df_clean.columns:
            df_clean[f'{var}_NIVEL'] = df_clean[var].map(EDUCATION_LEVELS)
            df_clean[f'{var}_NIVEL'] = df_clean[f'{var}_NIVEL'].fillna(-1)
    
    # Ordenar niveles de inglés
    if 'MOD_INGLES_DESEM' in df_clean.columns:
        df_clean['MOD_INGLES_NIVEL'] = df_clean['MOD_INGLES_DESEM'].map(ENGLISH_LEVELS)
        df_clean['MOD_INGLES_NIVEL'] = df_clean['MOD_INGLES_NIVEL'].fillna(-1)
    
    # Crear variable de NSE (Nivel Socioeconómico) compuesto
    try:
        # Componentes para NSE
        nse_components = []
        
        # Estrato (convertir a numérico)
        if 'FAMI_ESTRATOVIVIENDA' in df_clean.columns:
            df_clean['ESTRATO_NUM'] = df_clean['FAMI_ESTRATOVIVIENDA'].str.extract(r'(\d+)').astype(float)
            df_clean['ESTRATO_NUM'] = df_clean['ESTRATO_NUM'].fillna(0)
            nse_components.append('ESTRATO_NUM')
        
        # Educación de los padres
        for var in ['FAMI_EDUCACIONPADRE_NIVEL', 'FAMI_EDUCACIONMADRE_NIVEL']:
            if var in df_clean.columns:
                nse_components.append(var)
        
        # Bienes en el hogar
        for var in ['FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET', 'FAMI_TIENELAVADORA', 'FAMI_TIENEAUTOMOVIL']:
            if var in df_clean.columns:
                nse_components.append(var)
        
        # Calcular NSE como promedio ponderado normalizado
        if nse_components:
            # Normalizar componentes
            for comp in nse_components:
                max_val = df_clean[comp].max()
                min_val = df_clean[comp].min()
                if max_val > min_val:
                    df_clean[f'{comp}_NORM'] = (df_clean[comp] - min_val) / (max_val - min_val)
                else:
                    df_clean[f'{comp}_NORM'] = 0
            
            # Calcular NSE como promedio de componentes normalizados
            norm_components = [f'{comp}_NORM' for comp in nse_components]
            df_clean['NSE_SCORE'] = df_clean[norm_components].mean(axis=1)
            
            # Categorizar NSE en 5 niveles
            df_clean['NSE_NIVEL'] = pd.qcut(df_clean['NSE_SCORE'], 5, labels=['Muy bajo', 'Bajo', 'Medio', 'Alto', 'Muy alto'])
    except Exception as e:
        print(f"Error al calcular NSE: {e}")
    
    # Crear variable de rendimiento académico global
    try:
        academic_scores = [var for var in ACADEMIC_VARS if 'PUNT' in var]
        if academic_scores:
            # Normalizar puntajes
            for score in academic_scores:
                max_val = df_clean[score].max()
                min_val = df_clean[score].min()
                if max_val > min_val:
                    df_clean[f'{score}_NORM'] = (df_clean[score] - min_val) / (max_val - min_val)
                else:
                    df_clean[f'{score}_NORM'] = 0
            
            # Calcular rendimiento académico como promedio de puntajes normalizados
            norm_scores = [f'{score}_NORM' for score in academic_scores]
            df_clean['RA_SCORE'] = df_clean[norm_scores].mean(axis=1)
            
            # Categorizar RA en 5 niveles
            df_clean['RA_NIVEL'] = pd.qcut(df_clean['RA_SCORE'], 5, labels=['Muy bajo', 'Bajo', 'Medio', 'Alto', 'Muy alto'])
    except Exception as e:
        print(f"Error al calcular RA: {e}")
    
    return df_clean

def save_processed_data(df):
    """
    Guarda los datos procesados en un archivo CSV.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos procesados.
        
    Returns:
        bool: True si se guardó correctamente, False en caso contrario.
    """
    try:
        df.to_csv(PROCESSED_DATA_FILE, index=False)
        print(f"Datos procesados guardados en {PROCESSED_DATA_FILE}")
        return True
    except Exception as e:
        print(f"Error al guardar los datos procesados: {e}")
        return False

def get_data(force_reload=False):
    """
    Obtiene los datos procesados. Si no existen, los crea a partir de los datos crudos.
    
    Args:
        force_reload (bool): Si es True, recarga los datos desde el archivo crudo aunque exista el procesado.
        
    Returns:
        pandas.DataFrame: DataFrame con los datos procesados.
    """
    if os.path.exists(PROCESSED_DATA_FILE) and not force_reload:
        try:
            df = pd.read_csv(PROCESSED_DATA_FILE)
            print(f"Datos procesados cargados desde {PROCESSED_DATA_FILE}")
            return df
        except Exception as e:
            print(f"Error al cargar los datos procesados: {e}")
            return None
    else:
        df_raw = load_raw_data()
        df_clean = clean_data(df_raw)
        if df_clean is not None:
            save_processed_data(df_clean)
        return df_clean

if __name__ == "__main__":
    # Prueba de funcionamiento
    df = get_data(force_reload=True)
    if df is not None:
        print(f"Dimensiones finales: {df.shape}")
        print(f"Columnas: {df.columns.tolist()}")
