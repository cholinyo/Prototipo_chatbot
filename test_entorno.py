#!/usr/bin/env python3
"""
Diagnóstico del entorno Python - TFM Vicente Caruncho
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Verificar versión de Python"""
    print("🐍 INFORMACIÓN DE PYTHON:")
    print(f"   Versión: {sys.version}")
    print(f"   Ejecutable: {sys.executable}")
    print(f"   Plataforma: {sys.platform}")
    return sys.version_info

def check_pip():
    """Verificar pip"""
    print("\n📦 INFORMACIÓN DE PIP:")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True)
        print(f"   {result.stdout.strip()}")
        return True
    except Exception as e:
        print(f"   ❌ Error con pip: {e}")
        return False

def check_installed_packages():
    """Verificar paquetes instalados"""
    print("\n📋 PAQUETES CRÍTICOS:")
    
    critical_packages = ['numpy', 'torch', 'sentence_transformers', 'psutil']
    installed = {}
    
    for package in critical_packages:
        try:
            result = subprocess.run([sys.executable, "-c", f"import {package}; print({package}.__version__)"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"   ✅ {package}: {version}")
                installed[package] = version
            else:
                print(f"   ❌ {package}: No instalado")
                installed[package] = None
        except Exception as e:
            print(f"   ❌ {package}: Error - {e}")
            installed[package] = None
    
    return installed

def check_site_packages():
    """Verificar directorios de site-packages"""
    print("\n📁 DIRECTORIOS SITE-PACKAGES:")
    for path in sys.path:
        if 'site-packages' in path:
            print(f"   📂 {path}")
            if os.path.exists(path):
                print(f"      ✅ Existe")
            else:
                print(f"      ❌ No existe")

def suggest_fixes(installed_packages):
    """Sugerir soluciones"""
    print("\n🔧 SUGERENCIAS DE SOLUCIÓN:")
    
    missing = [pkg for pkg, version in installed_packages.items() if version is None]
    
    if missing:
        print(f"   📋 Paquetes faltantes: {', '.join(missing)}")
        print("\n   💡 Comandos de instalación:")
        
        if 'numpy' in missing:
            print("   pip install \"numpy>=1.26.0\"")
        if 'torch' in missing:
            print("   pip install torch")
        if 'sentence_transformers' in missing:
            print("   pip install sentence-transformers")
        if 'psutil' in missing:
            print("   pip install psutil")
    else:
        print("   ✅ Todos los paquetes críticos están instalados")

def create_minimal_test():
    """Crear test mínimo"""
    print("\n🧪 CREANDO TEST MÍNIMO...")
    
    test_code = """
# Test mínimo de imports
try:
    import numpy as np
    print("✅ NumPy OK")
    
    import torch
    print("✅ PyTorch OK")
    
    from sentence_transformers import SentenceTransformer
    print("✅ SentenceTransformers OK")
    
    import psutil
    print("✅ psutil OK")
    
    print("\\n🎉 ¡Todos los imports funcionan!")
    
except ImportError as e:
    print(f"❌ Error de import: {e}")
except Exception as e:
    print(f"❌ Error inesperado: {e}")
"""
    
    with open("test_minimal_imports.py", "w") as f:
        f.write(test_code)
    
    print("   ✅ Archivo creado: test_minimal_imports.py")
    print("   🚀 Ejecutar con: python test_minimal_imports.py")

def main():
    """Función principal de diagnóstico"""
    print("🔍 DIAGNÓSTICO DEL ENTORNO PYTHON")
    print("=" * 50)
    
    # Verificaciones
    python_version = check_python_version()
    pip_ok = check_pip()
    installed = check_installed_packages()
    check_site_packages()
    suggest_fixes(installed)
    create_minimal_test()
    
    print("\n" + "=" * 50)
    print("📊 RESUMEN:")
    print(f"   Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    print(f"   Pip: {'✅ OK' if pip_ok else '❌ Problema'}")
    
    missing_count = sum(1 for v in installed.values() if v is None)
    print(f"   Paquetes faltantes: {missing_count}/4")
    
    if missing_count == 0:
        print("\n🎉 ¡Entorno listo para ejecutar las pruebas!")
        print("🚀 Ejecutar: python ejecutar_fase1_embeddings.py")
    else:
        print(f"\n⚠️  Instalar {missing_count} paquetes faltantes primero")

if __name__ == "__main__":
    main()