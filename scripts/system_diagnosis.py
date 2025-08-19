#!/usr/bin/env python3
"""
Sistema de Diagnóstico Completo - TFM Chatbot RAG
Funciones de diagnóstico extraídas de run.py

Autor: Vicente Caruncho Ramos
TFM: Sistemas Inteligentes - UJI
"""
import sys
import time
import traceback
from pathlib import Path

# Añadir el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_basic_imports():
    """Verificar importaciones básicas"""
    print("📦 Verificando dependencias básicas...")
    
    basic_packages = [
        ('flask', 'Flask'),
        ('sentence_transformers', 'SentenceTransformer'),
        ('requests', 'requests'),
        ('yaml', 'PyYAML'),
        ('structlog', 'structlog')
    ]
    
    all_ok = True
    for package, display_name in basic_packages:
        try:
            __import__(package)
            print(f"   ✅ {display_name}")
        except ImportError as e:
            print(f"   ❌ {display_name} - {e}")
            all_ok = False
    
    return all_ok


def test_ollama_direct():
    """Verificar Ollama directamente"""
    print("🤖 Verificando Ollama...")
    
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"   ✅ Ollama disponible - {len(models)} modelos")
            for model in models[:3]:  # Mostrar máximo 3
                print(f"      📦 {model.get('name', 'unknown')}")
            return True
        else:
            print(f"   ❌ Ollama responde pero con error: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("   ❌ Ollama no disponible (ConnectionError)")
    except Exception as e:
        print(f"   ❌ Error verificando Ollama: {e}")
    
    return False


def check_project_structure():
    """Verificar estructura del proyecto"""
    print("📁 Verificando estructura del proyecto...")
    
    required_dirs = [
        'app', 'app/core', 'app/services', 'app/models', 'app/routes',
        'app/templates', 'app/static', 'data', 'logs', 'config'
    ]
    
    required_files = [
        'requirements.txt', 'run.py', 'app/__init__.py',
        'app/core/config.py', 'app/core/logger.py'
    ]
    
    missing_items = []
    
    # Verificar directorios
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/ - FALTANTE")
            missing_items.append(dir_path)
    
    # Verificar archivos
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
    """Intentar importar código del proyecto"""
    print("🔧 Probando importaciones del proyecto...")
    
    try:
        print("   🔄 Importando app...")
        import app
        print("   ✅ app importada correctamente")
        
        print("   🔄 Importando app.core...")
        from app.core import config, logger
        print("   ✅ app.core importado correctamente")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Error importando proyecto: {e}")
        print("   💡 Verifica que todos los archivos __init__.py existan")
        return False
    except Exception as e:
        print(f"   ❌ Error inesperado: {e}")
        return False


def verify_services():
    """Verificar servicios principales"""
    print("🔍 Verificando servicios principales...")
    
    services_status = {}
    
    # Verificar LLM Service
    try:
        from app.services.llm.llm_services import LLMService
        llm_service = LLMService()
        health = llm_service.health_check()
        services_status['llm_service'] = health['status'] == 'healthy'
        print(f"   ✅ LLM Service: {health['status']}")
    except Exception as e:
        services_status['llm_service'] = False
        print(f"   ❌ LLM Service: {e}")
    
    # Verificar RAG Pipeline
    try:
        from app.services.rag.pipeline import get_rag_pipeline
        pipeline = get_rag_pipeline()
        services_status['rag_pipeline'] = pipeline is not None
        print("   ✅ RAG Pipeline: disponible")
    except Exception as e:
        services_status['rag_pipeline'] = False
        print(f"   ❌ RAG Pipeline: {e}")
    
    return services_status


def run_complete_diagnosis():
    """Ejecutar diagnóstico completo del sistema"""
    print("🎓 TFM Vicente Caruncho - Diagnóstico Completo del Sistema")
    print("🏛️ Prototipo Chatbot RAG para Administraciones Locales")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Importaciones básicas
    print("\n📦 1. VERIFICANDO DEPENDENCIAS BÁSICAS")
    results['basic_imports'] = test_basic_imports()
    
    # Test 2: Ollama
    print("\n🤖 2. VERIFICANDO OLLAMA")
    results['ollama'] = test_ollama_direct()
    
    # Test 3: Estructura
    print("\n📁 3. VERIFICANDO ESTRUCTURA DEL PROYECTO")
    results['structure'] = check_project_structure()
    
    # Test 4: Importaciones proyecto
    if results['structure']:
        print("\n🔧 4. VERIFICANDO IMPORTACIONES DEL PROYECTO")
        results['project_imports'] = test_project_imports()
    else:
        print("\n⏭️ 4. SALTANDO IMPORTACIONES (estructura incorrecta)")
        results['project_imports'] = False
    
    # Test 5: Servicios
    if results['project_imports']:
        print("\n🔍 5. VERIFICANDO SERVICIOS")
        services = verify_services()
        results.update(services)
    
    # Generar reporte
    print("\n📋 REPORTE DE DIAGNÓSTICO")
    print("=" * 40)
    
    total_tests = len([k for k in results.keys() if k in ['basic_imports', 'ollama', 'structure', 'project_imports']])
    passed_tests = sum([results.get(k, False) for k in ['basic_imports', 'ollama', 'structure', 'project_imports']])
    
    for test_name, passed in results.items():
        if test_name in ['basic_imports', 'ollama', 'structure', 'project_imports', 'llm_service', 'rag_pipeline']:
            status = "✅ OK" if passed else "❌ FALLO"
            print(f"   {status} {test_name.replace('_', ' ').title()}")
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    print(f"\n📊 RESUMEN:")
    print(f"   Tests principales: {passed_tests}/{total_tests}")
    print(f"   Éxito: {success_rate:.1f}%")
    
    if all([results.get(k, False) for k in ['basic_imports', 'structure', 'project_imports']]):
        print("\n🎉 ¡SISTEMA BÁSICO FUNCIONANDO!")
        return 0
    else:
        print("\n⚠️ Resolver problemas antes de continuar")
        return 1


if __name__ == "__main__":
    try:
        exit_code = run_complete_diagnosis()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️ Diagnóstico cancelado")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        print(f"📋 Traceback: {traceback.format_exc()}")
        sys.exit(1)
