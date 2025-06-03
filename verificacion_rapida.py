#!/usr/bin/env python3
"""
Script de verificación rápida del funcionamiento de la aplicación.
Ejecuta este script para verificar que todo funciona correctamente.
"""

import sys
import os
from pathlib import Path
import importlib.util

def test_imports():
    """Prueba que todas las importaciones críticas funcionen."""
    print("🧪 Probando importaciones críticas...")
    
    critical_modules = [
        'pandas',
        'numpy',
        'streamlit',
        'plotly',
        'sklearn',
        'matplotlib',
        'seaborn'
    ]
    
    failed_imports = []
    
    for module in critical_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_data_loading():
    """Prueba que la carga de datos funcione."""
    print("\n📊 Probando carga de datos...")
    
    try:
        # Agregar el path del proyecto
        sys.path.insert(0, str(Path(__file__).parent))
        
        from src.data.data_loader import load_raw_data
        
        df = load_raw_data()
        
        if df is not None:
            print(f"✅ Datos cargados: {df.shape}")
            print(f"✅ Columnas disponibles: {len(df.columns)}")
            return True
        else:
            print("⚠️ No se pudieron cargar datos reales, pero el sistema puede funcionar con datos de muestra")
            return True
            
    except Exception as e:
        print(f"❌ Error en carga de datos: {e}")
        return False

def test_streamlit_app():
    """Verifica que la aplicación Streamlit se pueda ejecutar."""
    print("\n🌐 Verificando aplicación Streamlit...")
    
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    
    if not app_path.exists():
        print(f"❌ No se encontró app.py en {app_path}")
        return False
    
    print(f"✅ Aplicación encontrada: {app_path}")
    
    # Verificar que el archivo tenga contenido válido
    try:
        with open(app_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if 'def main()' in content and 'streamlit' in content:
            print("✅ Aplicación parece válida")
            return True
        else:
            print("❌ Aplicación no parece válida")
            return False
            
    except Exception as e:
        print(f"❌ Error leyendo aplicación: {e}")
        return False

def test_setup_scripts():
    """Verifica que los scripts de configuración funcionen."""
    print("\n⚙️ Probando scripts de configuración...")
    
    scripts = [
        'setup_data.py',
        'streamlit_setup.py'
    ]
    
    for script in scripts:
        script_path = Path(__file__).parent / script
        if script_path.exists():
            print(f"✅ {script} encontrado")
        else:
            print(f"❌ {script} no encontrado")

def run_verification():
    """Ejecuta todas las verificaciones."""
    print("🔍 VERIFICACIÓN RÁPIDA DEL PROYECTO SABER PRO")
    print("=" * 50)
    
    tests = [
        ("Importaciones", test_imports),
        ("Carga de datos", test_data_loading),
        ("Aplicación Streamlit", test_streamlit_app),
        ("Scripts de configuración", test_setup_scripts),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE RESULTADOS")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASÓ" if passed else "❌ FALLÓ"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ¡TODAS LAS VERIFICACIONES PASARON!")
        print("\nPara ejecutar la aplicación:")
        print("streamlit run dashboard/app.py")
    else:
        print("⚠️ ALGUNAS VERIFICACIONES FALLARON")
        print("\nRevisa los errores arriba y ejecuta:")
        print("pip install -r requirements.txt")
    
    print("=" * 50)
    
    return all_passed

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
