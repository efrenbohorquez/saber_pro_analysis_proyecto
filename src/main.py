"""
Módulo principal para integrar todos los análisis y ejecutarlos secuencialmente.
"""

import sys
from pathlib import Path
import os

# Agregar el directorio raíz al path para importar módulos del proyecto
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_loader import get_data
from src.models.pca_analysis import pca_academic_performance
from src.models.mca_analysis import mca_socioeconomic
from src.models.clustering_analysis import cluster_socioeconomic_academic
from src.models.predictive_models import predict_academic_performance
from src.models.geospatial_analysis import analyze_geospatial_data

def run_all_analyses(force_reload=False):
    """
    Ejecuta todos los análisis en secuencia.
    
    Args:
        force_reload (bool): Si es True, recarga los datos desde el archivo crudo aunque exista el procesado.
    """
    print("=== Iniciando análisis completo de datos Saber Pro ===")
    
    # Cargar y procesar datos
    print("\n1. Cargando y procesando datos...")
    df = get_data(force_reload=force_reload)
    
    if df is None:
        print("Error: No se pudieron cargar los datos. Abortando análisis.")
        return
    
    # Análisis de Componentes Principales
    print("\n2. Realizando Análisis de Componentes Principales...")
    pca, X_pca, explained_variance, loadings, feature_names = pca_academic_performance(df)
    
    # Análisis de Correspondencias Múltiples
    print("\n3. Realizando Análisis de Correspondencias Múltiples...")
    mca, X_mca, explained_inertia = mca_socioeconomic(df)
    
    # Clustering Jerárquico
    print("\n4. Realizando Clustering Jerárquico...")
    # Usar parámetros optimizados para reducir el uso de memoria
    model, labels, X_scaled = cluster_socioeconomic_academic(df, sample_size=3000, max_clusters=6)
    
    # Modelos Predictivos
    print("\n5. Entrenando Modelos Predictivos...")
    models = predict_academic_performance(df)
    
    # Análisis Geoespacial
    print("\n6. Realizando Análisis Geoespacial...")
    df_geo = analyze_geospatial_data(df)
    
    print("\n=== Análisis completo finalizado ===")
    print("Todos los resultados han sido guardados en las carpetas correspondientes.")

if __name__ == "__main__":
    run_all_analyses(force_reload=False)
