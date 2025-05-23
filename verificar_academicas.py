import pandas as pd
import sys
from pathlib import Path

# Agregar el directorio raíz al path para importar módulos del proyecto
sys.path.append('d:/Downloads/saber_pro_analysis_proyecto')

from src.config.constants import (
    RAW_DATA_FILE,
    ACADEMIC_VARS
)

try:
    print(f"Intentando cargar datos desde: {RAW_DATA_FILE}")
    df = pd.read_csv(RAW_DATA_FILE)
    print(f"Dataset cargado exitosamente. Dimensiones: {df.shape}")
    
    print("\nVerificando columnas académicas:")
    for var in ACADEMIC_VARS:
        presente = var in df.columns
        print(f"  {var}: {'✓' if presente else '✗'} {'(Presente)' if presente else '(No presente)'}")
        
    print("\nMostrando primeras filas de columnas académicas disponibles:")
    academic_vars_disponibles = [var for var in ACADEMIC_VARS if var in df.columns]
    if academic_vars_disponibles:
        print(df[academic_vars_disponibles].head())
    else:
        print("No hay columnas académicas disponibles en el dataset.")
        
except Exception as e:
    print(f"Error: {e}")
