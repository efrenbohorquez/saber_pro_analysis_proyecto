"""
Módulo principal para integrar todos los análisis y ejecutarlos secuencialmente.
"""

import sys
from pathlib import Path
import os
import logging

# Configurar logger
logger = logging.getLogger(__name__)

# Agregar el directorio raíz al path para importar módulos del proyecto
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_loader import get_data
from src.models.pca_analysis import pca_academic_performance
from src.models.mca_analysis import mca_socioeconomic
from src.models.clustering_analysis import cluster_socioeconomic_academic
from src.models.predictive_models import predict_academic_performance
from src.models.geospatial_analysis import analyze_geospatial_data

def run_all_analyses(force_reload=False):
    """Ejecuta todos los análisis definidos en el proyecto de forma secuencial.

    Esta función actúa como el punto de entrada principal para el pipeline de análisis.
    Carga los datos, luego ejecuta secuencialmente PCA, MCA, clustering,
    modelos predictivos y análisis geoespacial. Cada paso está encapsulado en
    un bloque try-except para permitir que el pipeline continúe incluso si un
    análisis individual falla.

    Args:
        force_reload (bool, optional): Si es True, fuerza la recarga y el reprocesamiento
            de los datos crudos, ignorando cualquier archivo procesado existente.
            Defaults to False.
    """
    logger.info("=== Iniciando análisis completo de datos Saber Pro ===")
    
    # Cargar y procesar datos
    logger.info("1. Cargando y procesando datos...")
    try:
        df = get_data(force_reload=force_reload)
        if df is None:
            logger.error("Error crítico: No se pudieron cargar los datos. Abortando análisis.")
            return
        logger.info("Carga y procesamiento de datos completado.")
    except Exception as e:
        logger.error(f"Error crítico durante la carga y procesamiento de datos: {e}", exc_info=True)
        return

    # Análisis de Componentes Principales
    logger.info("\n2. Iniciando Análisis de Componentes Principales (PCA)...")
    try:
        pca_results = pca_academic_performance(df)
        if pca_results: # Asumiendo que la función devuelve algo útil o None/Exception en error
            logger.info("Análisis de Componentes Principales (PCA) completado.")
        else:
            logger.warning("Análisis de Componentes Principales (PCA) no produjo resultados o falló internamente.")
    except Exception as e:
        logger.error(f"Error durante el Análisis de Componentes Principales (PCA): {e}", exc_info=True)
    
    # Análisis de Correspondencias Múltiples
    logger.info("\n3. Iniciando Análisis de Correspondencias Múltiples (MCA)...")
    try:
        mca_results = mca_socioeconomic(df)
        if mca_results:
            logger.info("Análisis de Correspondencias Múltiples (MCA) completado.")
        else:
            logger.warning("Análisis de Correspondencias Múltiples (MCA) no produjo resultados o falló internamente.")
    except Exception as e:
        logger.error(f"Error durante el Análisis de Correspondencias Múltiples (MCA): {e}", exc_info=True)
    
    # Clustering Jerárquico
    logger.info("\n4. Iniciando Clustering Jerárquico...")
    try:
        # Usar parámetros optimizados para reducir el uso de memoria
        cluster_results = cluster_socioeconomic_academic(df, sample_size=3000, max_clusters=6)
        if cluster_results:
            logger.info("Clustering Jerárquico completado.")
        else:
            logger.warning("Clustering Jerárquico no produjo resultados o falló internamente.")
    except Exception as e:
        logger.error(f"Error durante el Clustering Jerárquico: {e}", exc_info=True)
    
    # Modelos Predictivos
    logger.info("\n5. Iniciando entrenamiento de Modelos Predictivos...")
    try:
        predictive_models_results = predict_academic_performance(df)
        if predictive_models_results:
            logger.info("Entrenamiento de Modelos Predictivos completado.")
        else:
            logger.warning("Entrenamiento de Modelos Predictivos no produjo resultados o falló internamente.")
    except Exception as e:
        logger.error(f"Error durante el entrenamiento de Modelos Predictivos: {e}", exc_info=True)
    
    # Análisis Geoespacial
    logger.info("\n6. Iniciando Análisis Geoespacial...")
    try:
        geospatial_results = analyze_geospatial_data(df)
        if geospatial_results is not None: # df_geo es devuelto
            logger.info("Análisis Geoespacial completado.")
        else:
            logger.warning("Análisis Geoespacial no produjo resultados o falló internamente.")
    except Exception as e:
        logger.error(f"Error durante el Análisis Geoespacial: {e}", exc_info=True)
    
    logger.info("\n=== Análisis completo finalizado ===")
    logger.info("Todos los resultados (o los que se pudieron generar) han sido guardados en las carpetas correspondientes.")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("saber_pro_analysis.log"), # Para guardar logs en un archivo
            logging.StreamHandler(sys.stdout) # Para mostrar logs en la consola
        ]
    )
    run_all_analyses(force_reload=False)
