"""
Script de prueba para sistemas de ingesta de documentos y web
TFM Vicente Caruncho - Sistemas Inteligentes

Este script valida que tanto la ingesta de documentos como web
funcionen correctamente después de la consolidación.
"""

import os
import sys
import time
from pathlib import Path

# Añadir ruta del proyecto
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Probar que todas las importaciones funcionen"""
    print("=" * 60)
    print("PRUEBA 1: Verificar importaciones")
    print("=" * 60)
    
    try:
        # Importaciones del sistema consolidado
        from app.models.data_sources import (
            ScrapedPage, DocumentSource, WebSource, 
            create_document_source, create_web_source
        )
        from app.models.document import (
            DocumentChunk, create_web_chunks, create_document_chunks
        )
        print("✅ Importaciones de modelos: OK")
        
        # Servicios principales
        from app.services.document_ingestion_service import DocumentIngestionService
        from app.services.web_ingestion_service import WebIngestionService
        from app.services.web_scraper_service import WebScraperService
        print("✅ Importaciones de servicios: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en importaciones: {e}")
        return False


def test_document_ingestion():
    """Probar sistema de ingesta de documentos"""
    print("\n" + "=" * 60)
    print("PRUEBA 2: Sistema de ingesta de documentos")
    print("=" * 60)
    
    try:
        from app.services.document_ingestion_service import DocumentIngestionService
        from app.models.data_sources import create_document_source
        
        # Inicializar servicio
        doc_service = DocumentIngestionService()
        print("✅ Servicio de documentos inicializado")
        
        # Crear fuente de prueba
        test_source = create_document_source(
            name="Test Documentos",
            directories=["data", "docs"],  # Directorios que probablemente existan
            file_extensions=['.txt', '.md', '.pdf']
        )
        print(f"✅ Fuente de documentos creada: {test_source.id}")
        
        # Probar escaneo de directorios
        files = test_source.scan_directories()
        print(f"✅ Escaneo de directorios: {len(files)} archivos encontrados")
        
        # Listar primeros archivos si existen
        if files:
            print("   Archivos encontrados:")
            for i, file in enumerate(files[:3]):
                print(f"   - {file.path} ({file.size} bytes)")
                if i >= 2:
                    break
        
        return True
        
    except Exception as e:
        print(f"❌ Error en ingesta de documentos: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_web_ingestion():
    """Probar sistema de ingesta web"""
    print("\n" + "=" * 60)
    print("PRUEBA 3: Sistema de ingesta web")
    print("=" * 60)
    
    try:
        from app.services.web_ingestion_service import WebIngestionService
        from app.models.data_sources import create_web_source
        
        # Inicializar servicio
        web_service = WebIngestionService()
        print("✅ Servicio web inicializado")
        
        # Crear fuente de prueba con URL simple
        test_source = create_web_source(
            name="Test Web",
            base_urls=["https://httpbin.org/html"],  # URL de prueba simple
            max_depth=1,
            delay_seconds=2.0
        )
        print(f"✅ Fuente web creada: {test_source.id}")
        
        # Probar test de URL
        test_result = web_service.test_url("https://httpbin.org/html")
        print(f"✅ Test de URL: accesible={test_result.get('accessible', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en ingesta web: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_web_scraping():
    """Probar scraping básico"""
    print("\n" + "=" * 60)
    print("PRUEBA 4: Sistema de scraping web")
    print("=" * 60)
    
    try:
        from app.services.web_scraper_service import WebScraperService
        from app.models.data_sources import create_web_source
        
        # Inicializar servicio
        scraper = WebScraperService()
        print("✅ Servicio de scraping inicializado")
        
        # Crear fuente simple
        test_source = create_web_source(
            name="Test Scraping",
            base_urls=["https://httpbin.org/html"],
            max_depth=1
        )
        
        # Test de URL individual
        test_result = scraper.test_url("https://httpbin.org/html")
        print(f"✅ Test de conectividad: {test_result.get('status_code', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en scraping: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chunk_creation():
    """Probar creación de chunks desde web"""
    print("\n" + "=" * 60)
    print("PRUEBA 5: Creación de chunks web")
    print("=" * 60)
    
    try:
        from app.models.data_sources import ScrapedPage
        from app.models.document import create_web_chunks
        
        # Crear página de prueba
        test_page = ScrapedPage.from_response(
            url="https://example.com/test",
            title="Página de Prueba",
            content="Este es contenido de prueba para verificar que el sistema de chunks funciona correctamente. " * 10,
            links=["https://example.com/link1", "https://example.com/link2"],
            source_id="test-source-123"
        )
        print("✅ ScrapedPage de prueba creada")
        
        # Crear chunks
        chunks = create_web_chunks(test_page, chunk_size=100, chunk_overlap=20)
        print(f"✅ Chunks creados: {len(chunks)} chunks")
        
        # Verificar primer chunk
        if chunks:
            first_chunk = chunks[0]
            print(f"   - Primer chunk: {len(first_chunk.content)} caracteres")
            print(f"   - Metadatos: {list(first_chunk.metadata.keys())[:5]}")
            print(f"   - Es contenido web: {first_chunk.is_web_content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en creación de chunks: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_scraper():
    """Probar scraper mejorado"""
    print("\n" + "=" * 60)
    print("PRUEBA 6: Scraper mejorado")
    print("=" * 60)
    
    try:
        from app.services.enhanced_web_scraper import (
            EnhancedWebScraperService, ScrapingConfig, ScrapingMethod
        )
        
        # Inicializar servicio
        enhanced_scraper = EnhancedWebScraperService()
        print("✅ Scraper mejorado inicializado")
        
        # Obtener métodos disponibles
        methods = enhanced_scraper.get_available_methods()
        print(f"✅ Métodos disponibles: {len(methods)}")
        for method in methods:
            print(f"   - {method['name']}: {method['available']}")
        
        # Obtener estadísticas
        stats = enhanced_scraper.get_scraping_stats()
        print(f"✅ Estadísticas obtenidas: {stats['service_name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en scraper mejorado: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_full_test():
    """Ejecutar todas las pruebas"""
    print("INICIANDO PRUEBAS DE SISTEMAS DE INGESTA")
    print("TFM Vicente Caruncho - Sistemas Inteligentes")
    
    tests = [
        ("Importaciones", test_imports),
        ("Ingesta de documentos", test_document_ingestion),
        ("Ingesta web", test_web_ingestion),
        ("Scraping web", test_web_scraping),
        ("Creación de chunks", test_chunk_creation),
        ("Scraper mejorado", test_enhanced_scraper)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Error inesperado en {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN DE PRUEBAS")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResultado: {passed_tests}/{total_tests} pruebas exitosas")
    
    if passed_tests == total_tests:
        print("🎉 TODOS LOS SISTEMAS FUNCIONAN CORRECTAMENTE")
        print("✅ Listo para APIs")
    else:
        print("⚠️  Hay problemas que necesitan atención")
        
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)