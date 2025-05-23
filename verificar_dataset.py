"""
Script para verificar el dataset cargado y sus columnas.
"""

import pandas as pd
import sys
from pathlib import Path

# Agregar el directorio raíz al path para importar módulos del proyecto
sys.path.append(str(Path(__file__).resolve().parent))

from src.config.constants import (
    RAW_DATA_FILE,
    SOCIOECONOMIC_VARS,
    ACADEMIC_VARS,
    GEO_VARS,
    DEMOGRAPHIC_VARS,
    INSTITUTIONAL_VARS
)

def verificar_dataset():
    """Verificar el dataset y sus columnas."""
    try:
        print(f"Intentando cargar datos desde: {RAW_DATA_FILE}")
        df = pd.read_csv(RAW_DATA_FILE)
        print(f"Dataset cargado exitosamente. Dimensiones: {df.shape}")
        
        print("\nColumnas disponibles en el dataset:")
        print(df.columns.tolist())
        
        print("\nVerificando columnas de interés...")
        
        # Verificar variables socioeconómicas
        print("\nVariables socioeconómicas:")
        for var in SOCIOECONOMIC_VARS:
            presente = var in df.columns
            print(f"  {var}: {'✓' if presente else '✗'} {'(Presente)' if presente else '(No presente)'}")
        
        # Verificar variables académicas
        print("\nVariables académicas:")
        for var in ACADEMIC_VARS:
            presente = var in df.columns
            print(f"  {var}: {'✓' if presente else '✗'} {'(Presente)' if presente else '(No presente)'}")
        
        # Verificar variables geográficas
        print("\nVariables geográficas:")
        for var in GEO_VARS:
            presente = var in df.columns
            print(f"  {var}: {'✓' if presente else '✗'} {'(Presente)' if presente else '(No presente)'}")
        
        # Verificar variables demográficas
        print("\nVariables demográficas:")
        for var in DEMOGRAPHIC_VARS:
            presente = var in df.columns
            print(f"  {var}: {'✓' if presente else '✗'} {'(Presente)' if presente else '(No presente)'}")
        
        # Verificar variables institucionales
        print("\nVariables institucionales:")
        for var in INSTITUTIONAL_VARS:
            presente = var in df.columns
            print(f"  {var}: {'✓' if presente else '✗'} {'(Presente)' if presente else '(No presente)'}")
        
        # Estadísticas básicas de las variables académicas
        print("\nEstadísticas básicas de variables académicas:")
        academic_vars_df = df[ACADEMIC_VARS].copy()
        print(academic_vars_df.describe().T)
        
        # Verificar valores nulos
        print("\nValores nulos por columna:")
        print(df[ACADEMIC_VARS + SOCIOECONOMIC_VARS].isna().sum())
        
    except Exception as e:
        print(f"Error al verificar el dataset: {e}")

if __name__ == "__main__":
    verificar_dataset()
