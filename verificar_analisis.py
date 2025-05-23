"""
Script para ejecutar y verificar los análisis individuales.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Agregar el directorio raíz al path para importar módulos del proyecto
sys.path.append(str(Path(__file__).resolve().parent))

from src.config.constants import (
    RAW_DATA_FILE,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    ACADEMIC_VARS,
    SOCIOECONOMIC_VARS
)
from src.data.data_loader import load_raw_data, preprocess_data
from src.models.pca_analysis import pca_academic_performance
from src.models.mca_analysis import mca_socioeconomic

def ejecutar_y_verificar_analisis():
    """Ejecuta y verifica cada parte del análisis."""
    
    print("=== Iniciando verificación de análisis ===\n")
    
    # 1. Cargar y preprocesar datos
    print("1. Cargando y procesando datos...")
    try:
        df = load_raw_data()
        if df is None:
            print("Error: No se pudieron cargar los datos.")
            return
        
        df = preprocess_data(df)
        print(f"Datos procesados exitosamente. Dimensiones: {df.shape}")
        
        # Guardar una muestra de los datos procesados
        print("\nMuestra de datos procesados:")
        print(df.head())
        
        # Verificar columnas
        print("\nVerificando columnas importantes:")
        academic_disponibles = [var for var in ACADEMIC_VARS if var in df.columns]
        socioeconomic_disponibles = [var for var in SOCIOECONOMIC_VARS if var in df.columns]
        
        print(f"Variables académicas disponibles: {len(academic_disponibles)}/{len(ACADEMIC_VARS)}")
        print(academic_disponibles)
        
        print(f"Variables socioeconómicas disponibles: {len(socioeconomic_disponibles)}/{len(SOCIOECONOMIC_VARS)}")
        print(socioeconomic_disponibles)
        
    except Exception as e:
        print(f"Error en carga y procesamiento de datos: {e}")
        return
    
    # 2. Ejecutar PCA
    print("\n2. Probando Análisis de Componentes Principales...")
    try:
        pca, X_pca, explained_variance, loadings, feature_names = pca_academic_performance(df)
        print("PCA completado exitosamente.")
        print(f"Dimensiones de X_pca: {X_pca.shape}")
        print(f"Varianza explicada: {explained_variance[:5]}")
        
    except Exception as e:
        print(f"Error en PCA: {e}")
    
    # 3. Ejecutar MCA
    print("\n3. Probando Análisis de Correspondencias Múltiples...")
    try:
        mca, X_mca, explained_inertia = mca_socioeconomic(df)
        print("MCA completado exitosamente.")
        print(f"Dimensiones de X_mca: {X_mca.shape}")
        print(f"Inercia explicada: {explained_inertia[:5]}")
        
    except Exception as e:
        print(f"Error en MCA: {e}")
    
    print("\n=== Verificación completada ===")

if __name__ == "__main__":
    ejecutar_y_verificar_analisis()
