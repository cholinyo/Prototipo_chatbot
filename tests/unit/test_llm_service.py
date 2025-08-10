#!/usr/bin/env python3
"""
Suite de Pruebas Completa para LLM Service - CORREGIDO
TFM Vicente Caruncho - Sistemas Inteligentes UJI
"""

import sys
import os
import time
import json
from pathlib import Path

# SOLUCIÓN AL PROBLEMA DE IMPORTACIÓN
# Añadir el directorio raíz al path de Python
project_root = Path(__file__).parent.parent  # Desde tests/ subir a raíz
sys.path.insert(0, str(project_root))

# También añadir directamente el directorio actual
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

def test_basic_imports():
    """Prueba básica de importaciones sin dependencias del proyecto"""
    print("🔍 Probando importaciones básicas...")
    
    try:
        import requests
        print("   ✅ requests disponible")
    except ImportError:
        print("   ❌ requests no disponible - pip install requests")
        return False
    
    try:
        import flask
        print("   ✅ flask disponible")
    except ImportError:
        print("   ❌ flask no disponible - pip install flask")
        return False
    
    return True

def test_ollama_direct():
    """Prueba directa de Ollama sin usar nuestro código"""
    print("\n🤖 Probando Ollama directamente...")
    
    try:
        import requests
        
        # Test 1: Servidor ejecutándose
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code != 200:
            print("   ❌ Servidor Ollama no responde")
            return False
        
        print("   ✅ Servidor Ollama ejecutándose")
        
        # Test 2: Modelos disponibles
        data = response.json()
        models = [model['name'] for model in data.get('models', [])]
        
        if not models:
            print("   ❌ No hay modelos instalados")
            print("   💡 Ejecuta: ollama pull llama3.2:3b")
            return False
        
        print(f"   ✅ {len(models)} modelos disponibles")
        for model in models[:3]:
            print(f"      - {model}")
        
        # Test 3: Generación simple
        test_model = models[0]
        print(f"   🔄 Probando generación con {test_model}...")
        
        payload = {
            "model": test_model,
            "prompt": "Hola, responde con una palabra",
            "stream": False,
            "options": {"max_tokens": 5}
        }
        
        gen_response = requests.post(
            'http://localhost:11434/api/generate',
            json=payload,
            timeout=20
        )
        
        if gen_response.status_code == 200:
            gen_data = gen_response.json()
            text = gen_data.get('response', '').strip()
            print(f"   ✅ Generación exitosa: '{text}'")
            return True
        else:
            print(f"   ❌ Error en generación: {gen_response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error probando Ollama: {e}")
        return False

def test_project_structure():
    """Verificar estructura del proyecto"""
    print("\n📁 Verificando estructura del proyecto...")
    
    project_root = Path(__file__).parent.parent
    
    required_dirs = [
        'app',
        'app/services',
        'app/models',
        'app/routes'
    ]
    
    required_files = [
        'app/__init__.py',
        'app/services/__init__.py',
        'requirements.txt'
    ]
    
    missing_items = []
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/ - FALTANTE")
            missing_items.append(dir_path)
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - FALTANTE")
            missing_items.append(file_path)
    
    if missing_items:
        print(f"\n   💡 Elementos faltantes:")
        for item in missing_items:
            print(f"      - {item}")
        return False
    
    return True

def test_project_imports():
    """Intentar importar nuestro código del proyecto"""
    print("\n🔧 Probando importaciones del proyecto...")
    
    try:
        # Verificar que podemos importar la app
        print("   🔄 Importando app...")
        import app
        print("   ✅ app importada correctamente")
        
        # Verificar que podemos importar el core
        print("   🔄 Importando app.core...")
        from app.core import config, logger
        print("   ✅ app.core importado correctamente")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Error importando proyecto: {e}")
        print("   💡 Verifica que todos los archivos __init__.py existan")
        print("   💡 Verifica que estés en el directorio correcto")
        return False
    except Exception as e:
        print(f"   ❌ Error inesperado: {e}")
        return False

def main():
    """Función principal de diagnóstico"""
    print("🎓 TFM Vicente Caruncho - Diagnóstico del Sistema")
    print("🏛️ Prototipo Chatbot RAG para Administraciones Locales")
    print("🔍 Objetivo: Identificar y resolver problemas")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Importaciones básicas
    print("\n📦 1. VERIFICANDO DEPENDENCIAS BÁSICAS")
    basic_imports_ok = test_basic_imports()
    results['basic_imports'] = basic_imports_ok
    
    # Test 2: Ollama directo
    print("\n🤖 2. VERIFICANDO OLLAMA")
    ollama_ok = test_ollama_direct()
    results['ollama'] = ollama_ok
    
    # Test 3: Estructura del proyecto
    print("\n📁 3. VERIFICANDO ESTRUCTURA DEL PROYECTO")
    structure_ok = test_project_structure()
    results['structure'] = structure_ok
    
    # Test 4: Importaciones del proyecto (solo si estructura OK)
    if structure_ok:
        print("\n🔧 4. VERIFICANDO IMPORTACIONES DEL PROYECTO")
        project_imports_ok = test_project_imports()
        results['project_imports'] = project_imports_ok
    else:
        print("\n⏭️ 4. SALTANDO IMPORTACIONES (estructura incorrecta)")
        results['project_imports'] = False
    
    # Generar reporte
    print("\n📋 REPORTE DE DIAGNÓSTICO")
    print("=" * 40)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ OK" if passed else "❌ FALLO"
        print(f"   {status} {test_name.replace('_', ' ').title()}")
    
    success_rate = (passed_tests / total_tests) * 100
    print(f"\n📊 RESUMEN:")
    print(f"   Tests: {passed_tests}/{total_tests}")
    print(f"   Éxito: {success_rate:.1f}%")
    
    # Recomendaciones específicas
    print(f"\n💡 RECOMENDACIONES:")
    
    if not results.get('basic_imports', False):
        print("   🔧 Instalar dependencias: pip install requests flask")
    
    if not results.get('ollama', False):
        print("   🤖 Configurar Ollama:")
        print("      1. Descargar de https://ollama.ai/download")
        print("      2. Instalar y ejecutar")
        print("      3. Descargar modelo: ollama pull llama3.2:3b")
    
    if not results.get('structure', False):
        print("   📁 Crear estructura del proyecto:")
        print("      mkdir -p app/services app/models app/routes")
        print("      touch app/__init__.py app/services/__init__.py")
    
    if not results.get('project_imports', False) and results.get('structure', False):
        print("   🔧 Revisar archivos __init__.py y configuración")
    
    if all(results.values()):
        print("\n🎉 ¡SISTEMA LISTO! Puedes proceder con la implementación completa")
        return 0
    else:
        print("\n⚠️ Resolver problemas antes de continuar")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️ Diagnóstico cancelado")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        sys.exit(1)