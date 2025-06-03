#!/usr/bin/env python3
"""
Script de despliegue y configuración para Streamlit Cloud.
Este script se ejecuta automáticamente durante el despliegue.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_streamlit_environment():
    """Configura el entorno para Streamlit Cloud."""
    
    print("🚀 Configurando entorno para Streamlit Cloud...")
    
    # Verificar que estamos en Streamlit Cloud
    if not any([
        os.getenv('STREAMLIT_SHARING_MODE') == '1',
        '/mount/src/' in str(Path.cwd()),
        'streamlit.io' in os.getenv('HOSTNAME', ''),
    ]):
        print("ℹ️ No se detectó entorno de Streamlit Cloud")
        return True
    
    print("✅ Entorno de Streamlit Cloud detectado")
    
    # Crear directorios necesarios
    required_dirs = [
        'data/raw',
        'data/processed', 
        'docs/figures',
        'src/__pycache__',
        'whatever/joblib'
    ]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Directorio creado/verificado: {dir_path}")
    
    # Configurar variables de entorno específicas para Streamlit Cloud
    os.environ['STREAMLIT_SHARING_MODE'] = '1'
    os.environ['MPLBACKEND'] = 'Agg'  # Backend no interactivo para matplotlib
    
    print("✅ Configuración de entorno completada")
    return True

def verify_dependencies():
    """Verifica que las dependencias críticas estén instaladas."""
    
    print("📦 Verificando dependencias críticas...")
    
    critical_packages = [
        'pandas',
        'numpy', 
        'streamlit',
        'plotly',
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in critical_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NO ENCONTRADO")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️ Paquetes faltantes: {missing_packages}")
        return False
    
    print("✅ Todas las dependencias críticas están disponibles")
    return True

def test_data_loading():
    """Prueba que el sistema de carga de datos funcione."""
    
    print("📊 Probando sistema de carga de datos...")
    
    try:
        # Agregar path del proyecto
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        # Intentar importar y ejecutar la función de carga
        from src.data.data_loader import load_raw_data
        
        df = load_raw_data()
        
        if df is not None:
            print(f"✅ Datos cargados exitosamente: {df.shape}")
            return True
        else:
            print("⚠️ No se pudieron cargar datos, pero el sistema funcionará con datos de muestra")
            return True
            
    except Exception as e:
        print(f"❌ Error en prueba de carga de datos: {e}")
        return False

def create_status_file():
    """Crea un archivo de estado para monitorear el despliegue."""
    
    status = {
        'environment_setup': setup_streamlit_environment(),
        'dependencies_ok': verify_dependencies(), 
        'data_loading_ok': test_data_loading(),
    }
    
    # Escribir estado a archivo
    status_file = Path('deployment_status.txt')
    with open(status_file, 'w') as f:
        f.write("ESTADO DEL DESPLIEGUE EN STREAMLIT CLOUD\n")
        f.write("=" * 40 + "\n\n")
        
        for check, result in status.items():
            status_icon = "✅" if result else "❌"
            f.write(f"{status_icon} {check.replace('_', ' ').title()}: {result}\n")
        
        overall_status = all(status.values())
        f.write(f"\n🎯 Estado General: {'EXITOSO' if overall_status else 'CON PROBLEMAS'}\n")
    
    print(f"📄 Estado guardado en: {status_file}")
    return all(status.values())

if __name__ == "__main__":
    print("🌟 Iniciando configuración para Streamlit Cloud...")
    
    success = create_status_file()
    
    if success:
        print("🎉 ¡Configuración completada exitosamente!")
        sys.exit(0)
    else:
        print("⚠️ Configuración completada con advertencias")
        sys.exit(0)  # No fallar el despliegue por advertencias
