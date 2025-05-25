"""
Módulo para cargar y procesar los datos de las pruebas Saber Pro.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
import logging

# Configurar logger
logger = logging.getLogger(__name__)

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
    """Carga los datos crudos desde el archivo de origen (CSV o Excel).

    Determina el tipo de archivo (CSV o Excel) basado en la extensión
    especificada en `RAW_DATA_FILE` y carga los datos en un DataFrame de pandas.

    Returns:
        pd.DataFrame | None: DataFrame con los datos crudos, o None si ocurre un error
                              durante la carga (e.g., archivo no encontrado, formato no soportado,
                              error de parseo).
    """
    logger.info(f"Intentando cargar datos crudos desde: {RAW_DATA_FILE}")
    try:
        file_extension = RAW_DATA_FILE.suffix.lower()
        logger.info(f"Extensión del archivo detectada: {file_extension}")
        
        if file_extension == '.csv':
            df = pd.read_csv(RAW_DATA_FILE)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(RAW_DATA_FILE)
        else:
            logger.warning(f"Formato de archivo no soportado: {file_extension}")
            return None
            
        logger.info(f"Datos cargados exitosamente. Dimensiones: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Archivo no encontrado en la ruta: {RAW_DATA_FILE}", exc_info=True)
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"El archivo {RAW_DATA_FILE} está vacío.", exc_info=True)
        return None
    except pd.errors.ParserError as pe:
        logger.error(f"Error de parseo al leer el archivo {RAW_DATA_FILE}: {pe}", exc_info=True)
        return None
    except ValueError as ve: # Por ejemplo, si read_excel no puede manejar el archivo.
        logger.error(f"Error de valor al leer el archivo {RAW_DATA_FILE} (podría ser un problema con read_excel): {ve}", exc_info=True)
        return None
    except IOError as ioe:
        logger.error(f"Error de E/S al leer el archivo {RAW_DATA_FILE}: {ioe}", exc_info=True)
        return None
    except Exception as e: # Captura general para errores inesperados
        logger.error(f"Error inesperado al cargar los datos desde {RAW_DATA_FILE}: {e}", exc_info=True)
        return None

# Helper functions for clean_data
def _preprocess_dates_and_age(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte fechas de nacimiento y calcula la edad."""
    logger.debug("Iniciando preprocesamiento de fechas y cálculo de edad.")
    if 'ESTU_FECHANACIMIENTO' in df.columns:
        try:
            df['ESTU_FECHANACIMIENTO'] = pd.to_datetime(df['ESTU_FECHANACIMIENTO'], errors='coerce')
            logger.info("Columna 'ESTU_FECHANACIMIENTO' convertida a datetime.")
        except (ValueError, TypeError, OverflowError) as e:
            logger.warning(f"Error al convertir 'ESTU_FECHANACIMIENTO' a datetime: {e}. Se dejarán como NaT donde falle.", exc_info=True)
    else:
        logger.warning("Columna 'ESTU_FECHANACIMIENTO' no encontrada para conversión de fecha.")

    if 'PERIODO' in df.columns and \
       'ESTU_FECHANACIMIENTO' in df.columns and \
       pd.api.types.is_datetime64_any_dtype(df['ESTU_FECHANACIMIENTO']):
        try:
            initial_rows = df.shape[0]
            df['AÑO_PRUEBA'] = df['PERIODO'].astype(str).str[:4].astype(int)
            df['EDAD'] = df['AÑO_PRUEBA'] - df['ESTU_FECHANACIMIENTO'].dt.year
            df = df[(df['EDAD'] >= 16) & (df['EDAD'] <= 80)].copy() # Use .copy() to avoid SettingWithCopyWarning
            logger.info(f"Edad calculada. Filas antes del filtro de edad: {initial_rows}, después: {df.shape[0]}.")
        except (TypeError, AttributeError, ValueError) as e:
            logger.warning(f"Error al calcular la edad: {e}. Este paso se omitirá.", exc_info=True)
    else:
        logger.warning("No se pudo calcular la edad por falta de 'PERIODO' o 'ESTU_FECHANACIMIENTO' (datetime).")
    return df

def _standardize_text_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Limpia y estandariza columnas de texto (strip, upper)."""
    logger.debug(f"Estandarizando columnas de texto: {columns}")
    for col in columns:
        if col in df.columns:
            try:
                df[col] = df[col].astype(str).str.strip().str.upper()
                logger.debug(f"Columna '{col}' limpiada y estandarizada.")
            except AttributeError as e: # Should not happen with .astype(str) but good for safety
                logger.warning(f"Error al procesar columna de texto '{col}': {e}", exc_info=True)
    return df

def _process_academic_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte variables académicas a numéricas e imputa faltantes."""
    logger.debug("Procesando variables académicas.")
    for var in ACADEMIC_VARS:
        if var in df.columns:
            try:
                if 'PUNT' in var: # Variables de puntaje
                    df[var] = pd.to_numeric(df[var], errors='coerce')
                    mean_val = df[var].mean()
                    df[var] = df[var].fillna(mean_val)
                    logger.debug(f"Variable académica '{var}' (puntaje) convertida a numérica e imputada con media ({mean_val:.2f}).")
                else: # Variables de desempeño (categóricas)
                    mode_val = df[var].mode()
                    if not mode_val.empty:
                        df[var] = df[var].fillna(mode_val[0])
                        logger.debug(f"Variable académica '{var}' (desempeño) imputada con moda ('{mode_val[0]}').")
                    else:
                        logger.warning(f"No se pudo calcular la moda para '{var}', NaNs podrían persistir.")
            except (ValueError, TypeError) as e:
                logger.warning(f"Error al procesar variable académica '{var}': {e}", exc_info=True)
    return df

def _standardize_socioeconomic_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Estandariza variables socioeconómicas categóricas (estrato, binarias)."""
    logger.debug("Estandarizando variables socioeconómicas categóricas.")
    # Estandarizar estratos
    if 'FAMI_ESTRATOVIVIENDA' in df.columns:
        try:
            df['FAMI_ESTRATOVIVIENDA'] = df['FAMI_ESTRATOVIVIENDA'].astype(str)
            df['FAMI_ESTRATOVIVIENDA'] = df['FAMI_ESTRATOVIVIENDA'].apply(
                lambda x: f"Estrato {x.split('.')[0]}" if str(x).replace('.0','').isdigit() else str(x)
            )
            valid_strata = [f"Estrato {i}" for i in range(1, 7)] + ['Sin Estrato']
            df.loc[~df['FAMI_ESTRATOVIVIENDA'].isin(valid_strata), 'FAMI_ESTRATOVIVIENDA'] = 'Sin Estrato'
            logger.info("Columna 'FAMI_ESTRATOVIVIENDA' estandarizada.")
        except Exception as e: # Broad exception for .apply()
            logger.warning(f"Error al estandarizar 'FAMI_ESTRATOVIVIENDA': {e}", exc_info=True)

    # Convertir variables binarias
    binary_vars = ['FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET', 'FAMI_TIENELAVADORA', 'FAMI_TIENEAUTOMOVIL',
                   'ESTU_PAGOMATRICULABECA', 'ESTU_PAGOMATRICULACREDITO', 'ESTU_PAGOMATRICULAPADRES', 'ESTU_PAGOMATRICULAPROPIO']
    for var in binary_vars:
        if var in df.columns:
            try:
                df[var] = df[var].map({'Si': 1, 'No': 0, 'si': 1, 'no': 0, 'SI': 1, 'NO': 0}).fillna(0).astype(int)
                logger.debug(f"Variable binaria '{var}' convertida a 0/1.")
            except (TypeError, ValueError) as e:
                 logger.warning(f"Error al convertir variable binaria '{var}': {e}", exc_info=True)
    return df

def _map_education_and_english_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Mapea niveles educativos y de inglés a valores numéricos."""
    logger.debug("Mapeando niveles educativos y de inglés.")
    for var in ['FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE']:
        if var in df.columns:
            try:
                df[f'{var}_NIVEL'] = df[var].map(EDUCATION_LEVELS).fillna(-1).astype(int)
                logger.debug(f"Variable '{var}' mapeada a niveles numéricos.")
            except (TypeError, ValueError) as e:
                logger.warning(f"Error al mapear niveles educativos para '{var}': {e}", exc_info=True)

    if 'MOD_INGLES_DESEM' in df.columns:
        try:
            df['MOD_INGLES_NIVEL'] = df['MOD_INGLES_DESEM'].map(ENGLISH_LEVELS).fillna(-1).astype(int)
            logger.info("Columna 'MOD_INGLES_DESEM' mapeada a niveles numéricos.")
        except (TypeError, ValueError) as e:
            logger.warning(f"Error al mapear niveles de inglés: {e}", exc_info=True)
    return df

def _calculate_nse_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el Nivel Socioeconómico (NSE) compuesto."""
    logger.info("Calculando puntaje NSE.")
    try:
        nse_components = []
        if 'FAMI_ESTRATOVIVIENDA' in df.columns:
            if 'ESTRATO_NUM' not in df.columns or not pd.api.types.is_numeric_dtype(df['ESTRATO_NUM']):
                df['ESTRATO_NUM'] = df['FAMI_ESTRATOVIVIENDA'].str.extract(r'(\d+)').astype(float).fillna(0)
            nse_components.append('ESTRATO_NUM')

        for var_comp in ['FAMI_EDUCACIONPADRE_NIVEL', 'FAMI_EDUCACIONMADRE_NIVEL', 
                         'FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET', 
                         'FAMI_TIENELAVADORA', 'FAMI_TIENEAUTOMOVIL']:
            if var_comp in df.columns: nse_components.append(var_comp)
        
        if nse_components:
            for comp in nse_components:
                if comp in df.columns:
                    max_val, min_val = df[comp].max(), df[comp].min()
                    df[f'{comp}_NORM'] = (df[comp] - min_val) / (max_val - min_val) if max_val > min_val else 0
                else: logger.warning(f"Componente NSE '{comp}' no encontrado en el DataFrame.")
            
            norm_components = [f'{c}_NORM' for c in nse_components if f'{c}_NORM' in df.columns]
            if norm_components:
                df['NSE_SCORE'] = df[norm_components].mean(axis=1)
                df['NSE_NIVEL'] = pd.qcut(df['NSE_SCORE'], 5, labels=['Muy bajo', 'Bajo', 'Medio', 'Alto', 'Muy alto'], duplicates='drop')
                logger.info("Variables NSE_SCORE y NSE_NIVEL creadas.")
            else: logger.warning("No se pudieron calcular componentes normalizados para NSE_SCORE.")
        else: logger.warning("No hay componentes definidos para calcular NSE_SCORE.")
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        logger.warning(f"Error al calcular NSE: {e}", exc_info=True)
    return df

def _calculate_ra_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el Rendimiento Académico (RA) global."""
    logger.info("Calculando puntaje RA.")
    try:
        academic_scores = [var for var in ACADEMIC_VARS if 'PUNT' in var and var in df.columns]
        if academic_scores:
            for score in academic_scores:
                max_val, min_val = df[score].max(), df[score].min()
                df[f'{score}_NORM'] = (df[score] - min_val) / (max_val - min_val) if max_val > min_val else 0
            
            norm_scores = [f'{s}_NORM' for s in academic_scores if f'{s}_NORM' in df.columns]
            if norm_scores:
                df['RA_SCORE'] = df[norm_scores].mean(axis=1)
                df['RA_NIVEL'] = pd.qcut(df['RA_SCORE'], 5, labels=['Muy bajo', 'Bajo', 'Medio', 'Alto', 'Muy alto'], duplicates='drop')
                logger.info("Variables RA_SCORE y RA_NIVEL creadas.")
            else: logger.warning("No se pudieron calcular componentes normalizados para RA_SCORE.")
        else: logger.warning("No hay variables de puntaje académico definidas para calcular RA_SCORE.")
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        logger.warning(f"Error al calcular RA: {e}", exc_info=True)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame | None:
    """Limpia y preprocesa los datos crudos.

    Esta función orquesta una serie de pasos de limpieza y transformación de datos,
    incluyendo manejo de fechas, cálculo de edad, estandarización de texto,
    procesamiento de variables académicas y socioeconómicas, y cálculo
    de puntajes compuestos de NSE y RA.

    Args:
        df: DataFrame con los datos crudos.

    Returns:
        DataFrame con los datos limpios y preprocesados, o None si ocurre un error crítico.
    """
    if df is None:
        logger.warning("DataFrame de entrada para clean_data es None. No se realizará la limpieza.")
        return None
    
    logger.info(f"Iniciando proceso de limpieza de datos. Dimensiones iniciales: {df.shape}")
    df_clean = df.copy()
    
    try:
        df_clean = _preprocess_dates_and_age(df_clean)
        df_clean = _standardize_text_columns(df_clean, GEO_VARS)
        df_clean = _process_academic_variables(df_clean)
        df_clean = _standardize_socioeconomic_categorical(df_clean)
        df_clean = _map_education_and_english_levels(df_clean)
        df_clean = _calculate_nse_score(df_clean)
        df_clean = _calculate_ra_score(df_clean)
            
        logger.info(f"Proceso de limpieza de datos finalizado. Dimensiones finales: {df_clean.shape}")
        return df_clean

    except Exception as e: # Captura general para errores inesperados en la orquestación
        logger.error(f"Error inesperado durante la orquestación de la limpieza de datos: {e}", exc_info=True)
        return None


def save_processed_data(df: pd.DataFrame) -> bool:
    """Guarda los datos procesados en un archivo CSV.

    Args:
        df: DataFrame con los datos procesados.
        
    Returns:
        True si se guardó correctamente, False en caso contrario.
    """
    if df is None:
        logger.warning("DataFrame para guardar es None. No se guardarán los datos.")
        return False
    
    logger.info(f"Intentando guardar datos procesados en: {PROCESSED_DATA_FILE}. Dimensiones: {df.shape}")
    try:
        df.to_csv(PROCESSED_DATA_FILE, index=False)
        logger.info(f"Datos procesados guardados exitosamente en {PROCESSED_DATA_FILE}")
        return True
    except (IOError, OSError) as e:
        logger.error(f"Error de E/S al guardar los datos procesados en {PROCESSED_DATA_FILE}: {e}", exc_info=True)
        return False
    except Exception as e: # Captura general para errores inesperados
        logger.error(f"Error inesperado al guardar los datos procesados: {e}", exc_info=True)
        return False

def get_data(force_reload=False):
    """
    Obtiene los datos procesados. Si no existen, los crea a partir de los datos crudos.
    
    Args:
        force_reload (bool): Si es True, recarga los datos desde el archivo crudo aunque exista el procesado.
        
    Returns:
        pandas.DataFrame: DataFrame con los datos procesados o None si hay error.
    """
    logger.info(f"Iniciando get_data. force_reload={force_reload}")
    if os.path.exists(PROCESSED_DATA_FILE) and not force_reload:
        logger.info(f"Archivo procesado encontrado en {PROCESSED_DATA_FILE}. Intentando cargar.")
        try:
            df = pd.read_csv(PROCESSED_DATA_FILE)
            logger.info(f"Datos procesados cargados exitosamente desde {PROCESSED_DATA_FILE}. Dimensiones: {df.shape}")
            return df
        except FileNotFoundError: # Aunque os.path.exists pasó, podría haber una race condition o un path incorrecto para pd.read_csv
            logger.error(f"FileNotFoundError al intentar cargar {PROCESSED_DATA_FILE} aunque existía. Se intentará recargar desde crudo.", exc_info=True)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as pe:
            logger.error(f"Error de parseo/datos vacíos al cargar {PROCESSED_DATA_FILE}: {pe}. Se intentará recargar desde crudo.", exc_info=True)
        except IOError as ioe:
            logger.error(f"Error de E/S al cargar {PROCESSED_DATA_FILE}: {ioe}. Se intentará recargar desde crudo.", exc_info=True)
        except Exception as e: # Captura general para errores inesperados
             logger.error(f"Error inesperado al cargar datos procesados desde {PROCESSED_DATA_FILE}: {e}. Se intentará recargar desde crudo.", exc_info=True)
    
    logger.info("Procediendo a cargar y procesar datos crudos (o forzado por force_reload).")
    df_raw = load_raw_data()
    if df_raw is None:
        logger.error("Falló la carga de datos crudos. No se pueden procesar los datos.")
        return None
        
    df_clean = clean_data(df_raw)
    if df_clean is None:
        logger.error("Falló la limpieza de datos. No se pueden guardar los datos procesados.")
        return None # Si la limpieza falla, no hay datos válidos para guardar o devolver.
        
    if not save_processed_data(df_clean):
        logger.error("Falló el guardado de datos procesados. Se devolverán los datos limpios en memoria si están disponibles.")
        # Dependiendo de la criticidad, podría devolverse df_clean o None.
        # Por ahora, si falla el guardado, aún devolvemos los datos limpios en memoria.
    
    logger.info("Proceso get_data completado.")
    return df_clean

if __name__ == "__main__":
    # Configuración básica de logging para pruebas
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)] # Asegurar que el log va a stdout
    )
    
    logger.info("Ejecutando data_loader.py como script principal para prueba.")
    df = get_data(force_reload=True) # Forzar recarga para probar todo el pipeline
    
    if df is not None:
        logger.info(f"Prueba finalizada. Dimensiones finales del DataFrame: {df.shape}")
        logger.info(f"Columnas del DataFrame: {df.columns.tolist()}")
    else:
        logger.error("Prueba finalizada. No se pudo obtener el DataFrame.")
