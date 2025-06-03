#!/usr/bin/env python3
"""
Script optimizado para verificar y manejar los datos en Streamlit Cloud.
Este script implementa múltiples estrategias para asegurar que los datos estén disponibles.
"""

import os
import sys
import pandas as pd
import requests
import zipfile
from pathlib import Path
import tempfile
import shutil

def detect_environment():
    """Detecta si estamos ejecutando en Streamlit Cloud."""
    indicators = [
        os.getenv('STREAMLIT_SHARING_MODE') == '1',
        '/mount/src/' in str(Path.cwd()),
        os.getenv('HOME', '').startswith('/home/appuser'),
        'streamlit.io' in os.getenv('HOSTNAME', ''),
    ]
    return any(indicators)

def create_sample_data():
    """Crea datos de muestra cuando el dataset real no está disponible."""
    print("📊 Creando datos de muestra para demostración...")
    
    # Simular estructura del dataset real
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'ESTU_CONSECUTIVO': range(1, n_samples + 1),
        'MOD_RAZONA_CUANTITAT_PUNT': np.random.normal(150, 30, n_samples),
        'MOD_LECTURA_CRITICA_PUNT': np.random.normal(150, 30, n_samples),
        'MOD_COMPETEN_CIUDADA_PUNT': np.random.normal(150, 30, n_samples),
        'MOD_COMUNI_ESCRITA_PUNT': np.random.normal(150, 30, n_samples),
        'MOD_INGLES_PUNT': np.random.normal(150, 30, n_samples),
        'FAMI_ESTRATOVIVIENDA': np.random.choice([1, 2, 3, 4, 5, 6], n_samples),
        'FAMI_EDUCACIONPADRE': np.random.choice(['No aplica', 'Ninguno', 'Primaria incompleta', 
                                               'Primaria completa', 'Secundaria incompleta',
                                               'Secundaria completa', 'Técnica', 'Tecnológica',
                                               'Universitaria', 'Posgrado'], n_samples),
        'FAMI_EDUCACIONMADRE': np.random.choice(['No aplica', 'Ninguno', 'Primaria incompleta', 
                                               'Primaria completa', 'Secundaria incompleta',
                                               'Secundaria completa', 'Técnica', 'Tecnológica',
                                               'Universitaria', 'Posgrado'], n_samples),
        'ESTU_DEPTO_RESIDE': np.random.choice(['AMAZONAS', 'ANTIOQUIA', 'ARAUCA', 'ATLÁNTICO', 
                                             'BOGOTÁ D.C.', 'BOLÍVAR', 'BOYACÁ', 'CALDAS', 
                                             'CAQUETÁ', 'CASANARE', 'CAUCA', 'CESAR', 'CHOCÓ', 
                                             'CÓRDOBA', 'CUNDINAMARCA', 'GUAINÍA', 'GUAVIARE', 
                                             'HUILA', 'LA GUAJIRA', 'MAGDALENA', 'META', 'NARIÑO', 
                                             'NORTE DE SANTANDER', 'PUTUMAYO', 'QUINDÍO', 'RISARALDA', 
                                             'SAN ANDRÉS', 'SANTANDER', 'SUCRE', 'TOLIMA', 'VALLE', 
                                             'VAUPÉS', 'VICHADA'], n_samples),
    }
    
    df = pd.DataFrame(data)
    return df

def download_from_url(url, destination):
    """Descarga un archivo desde una URL con manejo de errores robusto."""
    try:
        print(f"📥 Descargando desde: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, stream=True, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Crear directorio padre si no existe
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Descargar archivo
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"✅ Descarga completada: {destination}")
        return True
        
    except Exception as e:
        print(f"❌ Error en descarga: {e}")
        return False

def verify_and_prepare_data():
    """Verifica y prepara los datos para la aplicación."""
    
    # Detectar entorno
    is_cloud = detect_environment()
    print(f"🌐 Entorno detectado: {'Streamlit Cloud' if is_cloud else 'Local'}")
    
    # Definir rutas posibles
    possible_paths = [
        Path("data/raw/dataset_dividido_10.csv"),
        Path("./data/raw/dataset_dividido_10.csv"),
        Path("/mount/src/saber_pro_analysis_proyecto/data/raw/dataset_dividido_10.csv"),
    ]
    
    # Verificar si existe algún archivo
    existing_file = None
    for path in possible_paths:
        if path.exists():
            existing_file = path
            print(f"✅ Archivo encontrado: {path}")
            break
    
    if existing_file:
        try:
            # Verificar que el archivo se puede cargar
            df = pd.read_csv(existing_file, nrows=5)  # Solo leer primeras 5 filas para verificar
            print(f"✅ Archivo válido con {len(df.columns)} columnas")
            return str(existing_file)
        except Exception as e:
            print(f"❌ Error al leer archivo: {e}")
    
    # Si no se encontró archivo, intentar descarga en entorno cloud
    if is_cloud:
        print("📥 Intentando descarga automática...")
        
        # URLs de respaldo para descargar el dataset
        download_urls = [
            "https://github.com/efrenbohorquez/saber_pro_analysis_proyecto/releases/download/v1.0/dataset_dividido_10.csv",
            "https://raw.githubusercontent.com/efrenbohorquez/saber_pro_analysis_proyecto/main/data/raw/dataset_dividido_10.csv",
        ]
        
        target_path = Path("data/raw/dataset_dividido_10.csv")
        
        for url in download_urls:
            if download_from_url(url, target_path):
                return str(target_path)
    
    # Como último recurso, crear datos de muestra
    print("🔄 Creando datos de muestra...")
    try:
        import numpy as np
        sample_data = create_sample_data()
        
        # Guardar datos de muestra
        sample_path = Path("data/raw/dataset_sample.csv")
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_data.to_csv(sample_path, index=False)
        
        print(f"✅ Datos de muestra creados: {sample_path}")
        return str(sample_path)
        
    except Exception as e:
        print(f"❌ Error creando datos de muestra: {e}")
        return None

if __name__ == "__main__":
    result = verify_and_prepare_data()
    if result:
        print(f"🎯 Datos preparados exitosamente: {result}")
        sys.exit(0)
    else:
        print("❌ No se pudieron preparar los datos")
        sys.exit(1)
