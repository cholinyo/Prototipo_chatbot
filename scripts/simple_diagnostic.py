#!/usr/bin/env python3
"""
Script de Diagnóstico Simplificado
TFM Vicente Caruncho - Prototipo_chatbot
"""

import sys
import os
import time
from pathlib import Path

# Añadir el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def print_section(title):
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def test_import(module_path, description):
    """Test de importación individual"""
    try:
        exec(f"import {module_path}")
        print(f"✅ {description}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {e}")
        return False
    except Exception as e:
        print(f"⚠️ {description}: Error - {e}")
        return False

def test_function_import(import_statement, description):
    """Test de importación de función específica"""
    try:
        exec(import_statement)
        print(f"✅ {description}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {e}")
        return False
    except Exception as e:
        print(f"⚠️ {description}: Error - {e}")
        return False

def check_file_exists(file_path, description):
    """Verificar si un archivo existe"""
    path = project_root / file_path
    if path.exists():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description}: NO EXISTE - {path}")
        return False

def main():
    print_section("DIAGNÓSTICO SIMPLIFICADO DEL SISTEMA")
    print("👨‍🎓 Vicente Caruncho Ramos - Sistemas Inteligentes UJI")
    
    total_tests = 0
    passed_tests = 0
    
    # 1. Verificar archivos clave
    print_section("1. VERIFICANDO ARCHIVOS CLAVE")
    
    files_to_check = [
        ("app/core/config.py", "Archivo de configuración"),
        ("app/services/rag_pipeline.py", "Pipeline RAG principal"),
        ("app/services/rag/embeddings.py", "Servicio de embeddings"),
        ("app/services/rag/faiss_store.py", "FAISS vector store"),
        ("app/services/rag/chromadb_store.py", "ChromaDB vector store"),
        ("app/services/llm/llm_services.py", "Servicio LLM"),
        ("run.py", "Script principal"),
    ]
    
    for file_path, description in files_to_check:
        total_tests += 1
        if check_file_exists(file_path, description):
            passed_tests += 1
    
    # 2. Verificar imports básicos
    print_section("2. VERIFICANDO IMPORTS BÁSICOS")
    
    basic_imports = [
        ("torch", "PyTorch"),
        ("sentence_transformers", "Sentence Transformers"),
        ("faiss", "FAISS"),
        ("chromadb", "ChromaDB"),
        ("flask", "Flask"),
        ("numpy", "NumPy"),
    ]
    
    for module, description in basic_imports:
        total_tests += 1
        if test_import(module, description):
            passed_tests += 1
    
    # 3. Verificar imports específicos del proyecto
    print_section("3. VERIFICANDO IMPORTS DEL PROYECTO")
    
    project_imports = [
        ("from app.core.config import get_app_config", "Configuración de app"),
        ("from app.core.config import get_model_config", "Configuración de modelos"),
        ("from app.core.config import get_embedding_config", "Configuración de embeddings"),
        ("from app.services.rag_pipeline import get_rag_pipeline", "Pipeline RAG"),
    ]
    
    for import_stmt, description in project_imports:
        total_tests += 1
        if test_function_import(import_stmt, description):
            passed_tests += 1
    
    # 4. Verificar servicios específicos
    print_section("4. VERIFICANDO SERVICIOS ESPECÍFICOS")
    
    try:
        from app.services.rag.embeddings import get_embedding_service
        embedding_service = get_embedding_service()
        if embedding_service.is_available():
            print("✅ Embedding Service funcional")
            passed_tests += 1
        else:
            print("❌ Embedding Service no disponible")
        total_tests += 1
    except Exception as e:
        print(f"❌ Error con Embedding Service: {e}")
        total_tests += 1
    
    try:
        from app.services.rag.faiss_store import get_faiss_store
        faiss_store = get_faiss_store()
        if faiss_store.is_available():
            print("✅ FAISS Store funcional")
            passed_tests += 1
        else:
            print("❌ FAISS Store no disponible")
        total_tests += 1
    except Exception as e:
        print(f"❌ Error con FAISS Store: {e}")
        total_tests += 1
    
    try:
        from app.services.rag.chromadb_store import get_chromadb_store
        chromadb_store = get_chromadb_store()
        if chromadb_store.is_available():
            print("✅ ChromaDB Store funcional")
            passed_tests += 1
        else:
            print("❌ ChromaDB Store no disponible")
        total_tests += 1
    except Exception as e:
        print(f"❌ Error con ChromaDB Store: {e}")
        total_tests += 1
    
    try:
        from app.services.llm.llm_services import get_llm_service
        llm_service = get_llm_service()
        health = llm_service.health_check()
        if health['status'] in ['healthy', 'degraded']:
            print(f"✅ LLM Service funcional - Estado: {health['status']}")
            passed_tests += 1
        else:
            print(f"❌ LLM Service con problemas - Estado: {health['status']}")
        total_tests += 1
    except Exception as e:
        print(f"❌ Error con LLM Service: {e}")
        total_tests += 1
    
    # 5. Test del pipeline completo
    print_section("5. VERIFICANDO PIPELINE COMPLETO")
    
    try:
        from app.services.rag_pipeline import get_rag_pipeline
        pipeline = get_rag_pipeline()
        
        if pipeline:
            print("✅ Pipeline RAG importado correctamente")
            
            if hasattr(pipeline, 'health_check'):
                health = pipeline.health_check()
                print(f"✅ Health check disponible - Estado: {health.get('status', 'unknown')}")
                passed_tests += 1
            else:
                print("❌ Pipeline sin método health_check")
            
            if hasattr(pipeline, 'get_stats'):
                stats = pipeline.get_stats()
                print(f"✅ Estadísticas disponibles - Docs: {stats.get('documents_count', 0)}")
                passed_tests += 1
            else:
                print("❌ Pipeline sin método get_stats")
            
            if hasattr(pipeline, 'is_available'):
                available = pipeline.is_available()
                print(f"✅ Pipeline disponible: {available}")
                passed_tests += 1
            else:
                print("❌ Pipeline sin método is_available")
            
            total_tests += 3
        else:
            print("❌ Pipeline RAG no se pudo inicializar")
            total_tests += 1
            
    except Exception as e:
        print(f"❌ Error grave con Pipeline RAG: {e}")
        print(f"   Tipo de error: {type(e).__name__}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        total_tests += 1
    
    # 6. Verificar configuración
    print_section("6. VERIFICANDO CONFIGURACIÓN")
    
    try:
        from app.core.config import validate_configuration
        validation = validate_configuration()
        
        if validation['valid']:
            print("✅ Configuración válida")
            passed_tests += 1
        else:
            print("❌ Configuración con errores:")
            for error in validation['errors']:
                print(f"   - {error}")
        
        if validation['warnings']:
            print("⚠️ Advertencias de configuración:")
            for warning in validation['warnings']:
                print(f"   - {warning}")
        
        total_tests += 1
        
    except Exception as e:
        print(f"❌ Error verificando configuración: {e}")
        total_tests += 1
    
    # Resumen final
    print_section("RESUMEN DIAGNÓSTICO")
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"📊 Tests ejecutados: {total_tests}")
    print(f"✅ Tests exitosos: {passed_tests}")
    print(f"❌ Tests fallidos: {total_tests - passed_tests}")
    print(f"📈 Tasa de éxito: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("\n🎉 SISTEMA EN BUEN ESTADO")
        print("💡 Próximo paso: python run.py")
    elif success_rate >= 60:
        print("\n⚠️ SISTEMA FUNCIONAL CON ADVERTENCIAS")
        print("💡 Revisa los errores específicos arriba")
    else:
        print("\n❌ SISTEMA NECESITA REPARACIONES")
        print("💡 Muchos componentes tienen problemas")
    
    print(f"\n🕒 Diagnóstico completado en {time.time():.1f} segundos")
    
    return 0 if success_rate >= 60 else 1

if __name__ == "__main__":
    sys.exit(main())