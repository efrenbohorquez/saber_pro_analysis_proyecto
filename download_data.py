#!/usr/bin/env python3
"""
Script para descargar el dataset desde GitHub Releases.
Este script se ejecutará automáticamente en Streamlit Cloud si el archivo local no existe.
"""

import os
import requests
import zipfile
from pathlib import Path

def download_dataset():
    """Descarga el dataset desde GitHub Releases si no existe localmente."""
    
    # Rutas del archivo
    data_dir = Path("data/raw")
    dataset_file = data_dir / "dataset_dividido_10.csv"
    
    # Si el archivo ya existe, no hacer nada
    if dataset_file.exists():
        print(f"✅ Dataset ya existe: {dataset_file}")
        return True
    
    print("📥 Descargando dataset desde GitHub Releases...")
    
    # URL del release (tendrás que actualizar esta URL después de crear el release)
    release_url = "https://github.com/efrenbohorquez/saber_pro_analysis_proyecto/releases/download/v1.0/dataset_dividido_10.csv"
    
    try:
        # Crear directorio si no existe
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Descargar archivo
        response = requests.get(release_url, stream=True)
        response.raise_for_status()
        
        # Guardar archivo
        with open(dataset_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Dataset descargado exitosamente: {dataset_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error descargando dataset: {e}")
        return False

if __name__ == "__main__":
    download_dataset()
