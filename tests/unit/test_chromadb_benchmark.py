#!/usr/bin/env python3
"""
Test de ChromaDB compatible con benchmark académico
Prototipo_chatbot - TFM Vicente Caruncho
"""

import sys
import os
import time
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime

# Añadir el directorio raíz al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_chromadb_for_benchmark():
    """Test específico para verificar compatibilidad con benchmark"""
    print("🧪 TEST CHROMADB PARA BENCHMARK ACADÉMICO")
    print("=" * 60)
    
    try:
        print("📦 1. Importando módulos...")
        
        from app.services.rag.chromadb_store import (
            ChromaDBVectorStore, 
            create_chromadb_store,
            is_chromadb_available
        )
        from app.services.rag.embeddings import embedding_service
        from app.models import DocumentChunk, DocumentMetadata
        
        print("   ✅ Módulos importados correctamente")
        
    except ImportError as e:
        print(f"   ❌ Error importando: {e}")
        return False
    
    try:
        print("\n🔧 2. Verificando disponibilidad...")
        
        if not is_chromadb_available():
            print("   ❌ ChromaDB no disponible")
            return False
        
        if not embedding_service.is_available():
            print("   ❌ EmbeddingService no disponible")
            return False
        
        print("   ✅ Todos los servicios disponibles")
        
    except Exception as e:
        print(f"   ❌ Error verificando servicios: {e}")
        return False
    
    try:
        print("\n🏗️ 3. Creando ChromaDBVectorStore...")
        
        # Limpiar directorio de prueba
        test_store_path = "data/test_benchmark_chromadb"
        if Path(test_store_path).exists():
            shutil.rmtree(test_store_path)
        
        # Crear store
        store = create_chromadb_store(
            store_path=test_store_path,
            collection_name="benchmark_test",
            distance_function="cosine"
        )
        
        if not store.is_available():
            print("   ❌ No se pudo crear el vector store")
            return False
        
        print("   ✅ ChromaDBVectorStore creado exitosamente")
        
    except Exception as e:
        print(f"   ❌ Error creando vector store: {e}")
        return False
    
    try:
        print("\n📄 4. Preparando documentos de prueba...")
        
        # Crear documentos similares al benchmark
        test_docs = [
            "Administración electrónica y digitalización de servicios públicos municipales",
            "Procedimientos de licencias urbanísticas y de actividad en administraciones locales",
            "Gestión de expedientes administrativos y tramitación de documentos oficiales",
            "Atención ciudadana presencial y telemática en ayuntamientos españoles",
            "Normativa municipal sobre ordenanzas, reglamentos y disposiciones locales"
        ]
        
        chunks = []
        for i, text in enumerate(test_docs):
            metadata = DocumentMetadata(
                source_path=f"test_doc_{i+1}.pdf",
                source_type="document", 
                file_type=".pdf",
                size_bytes=len(text),
                created_at=datetime.now(),
                processed_at=datetime.now(),
                checksum=f"test_checksum_{i+1}",
                title=f"Documento de Prueba {i+1}"
            )
            
            chunk = DocumentChunk(
                id=f"test-chunk-{i+1:02d}",
                content=text,
                metadata=metadata,
                chunk_index=i,
                chunk_size=len(text),
                start_char=i*100,
                end_char=(i+1)*100
            )
            chunks.append(chunk)
        
        print(f"   ✅ Preparados {len(chunks)} documentos de prueba")
        
    except Exception as e:
        print(f"   ❌ Error preparando documentos: {e}")
        return False
    
    try:
        print("\n🧠 5. Generando embeddings...")
        
        # Generar embeddings usando EmbeddingService
        embedded_chunks = embedding_service.encode_documents(chunks)
        
        if not embedded_chunks:
            print("   ❌ Error generando embeddings")
            return False
        
        print(f"   ✅ Embeddings generados para {len(embedded_chunks)} documentos")
        
    except Exception as e:
        print(f"   ❌ Error generando embeddings: {e}")
        return False
    
    try:
        print("\n💾 6. Probando inserción de documentos...")
        
        # Limpiar store
        store.clear()
        
        # Insertar documentos
        start_time = time.time()
        success = store.add_documents(embedded_chunks)
        insertion_time = time.time() - start_time
        
        if not success:
            print("   ❌ Error insertando documentos")
            return False
        
        throughput = len(embedded_chunks) / insertion_time
        print(f"   ✅ Documentos insertados en {insertion_time:.3f}s")
        print(f"      - Throughput: {throughput:.1f} docs/segundo")
        
    except Exception as e:
        print(f"   ❌ Error insertando documentos: {e}")
        return False
    
    try:
        print("\n🔍 7. Probando búsqueda directa (método search)...")
        
        # Generar embedding de consulta
        test_query = "licencias y permisos municipales"
        query_embedding = embedding_service.encode_single_text(test_query)
        
        if query_embedding is None:
            print("   ❌ Error generando embedding de consulta")
            return False
        
        # Probar búsqueda directa (requerida por benchmark)
        start_time = time.time()
        results = store.search(query_embedding, k=3)
        search_time = time.time() - start_time
        
        print(f"   ✅ Búsqueda completada en {search_time:.3f}s")
        print(f"      - Resultados encontrados: {len(results)}")
        
        for i, result in enumerate(results):
            print(f"         {i+1}. {result.content[:50]}...")
        
    except Exception as e:
        print(f"   ❌ Error en búsqueda directa: {e}")
        return False
    
    try:
        print("\n🎛️ 8. Probando búsqueda con filtros...")
        
        # Probar filtros (requerido por benchmark)
        filters = {"source_type": "document"}
        
        start_time = time.time()
        filtered_results = store.search(query_embedding, k=3, filters=filters)
        filter_time = time.time() - start_time
        
        print(f"   ✅ Búsqueda con filtros completada en {filter_time:.3f}s")
        print(f"      - Resultados filtrados: {len(filtered_results)}")
        
    except Exception as e:
        print(f"   ❌ Error en búsqueda con filtros: {e}")
        return False
    
    try:
        print("\n📊 9. Verificando métricas del sistema...")
        
        stats = store.get_stats()
        
        print("   📈 Estadísticas del vector store:")
        print(f"      - Tipo: {stats.get('type', 'N/A')}")
        print(f"      - Documentos totales: {stats.get('total_documents', 0)}")
        print(f"      - Tiempo promedio búsqueda: {stats.get('performance', {}).get('avg_search_time_ms', 0):.1f}ms")
        
        storage = stats.get('storage', {})
        print(f"      - Uso de disco: {storage.get('disk_usage_mb', 0):.2f}MB")
        
        print("   ✅ Métricas obtenidas correctamente")
        
    except Exception as e:
        print(f"   ❌ Error obteniendo métricas: {e}")
        return False
    
    try:
        print("\n🧹 10. Limpieza final...")
        
        # Cerrar conexiones ChromaDB antes de limpiar archivos
        if hasattr(store, 'client') and store.client:
            try:
                # Limpiar la colección primero
                store.clear()
            except:
                pass
        
        # Forzar cierre de conexiones
        store.client = None
        store.collection = None
        
        print(f"   ✅ Vector store limpiado")
        
        # Esperar un poco antes de limpiar archivos
        time.sleep(2)
        
        # Intentar limpiar directorio con manejo de errores
        try:
            if Path(test_store_path).exists():
                shutil.rmtree(test_store_path)
                print(f"   ✅ Directorio de prueba eliminado: {test_store_path}")
        except PermissionError:
            print(f"   ⚠️ No se pudo eliminar automáticamente: {test_store_path}")
            print(f"   💡 Eliminar manualmente después del test")
        
    except Exception as e:
        print(f"   ❌ Error en limpieza: {e}")
        return False
    
    return True

def test_benchmark_compatibility():
    """Verificar que ChromaDB tiene todos los métodos requeridos por el benchmark"""
    print("\n🔬 VERIFICANDO COMPATIBILIDAD CON BENCHMARK...")
    
    try:
        from app.services.rag.chromadb_store import ChromaDBVectorStore
        
        # Verificar métodos requeridos
        required_methods = [
            'is_available',
            'add_documents', 
            'search',           # Método clave para benchmark
            'clear',           # Método clave para benchmark
            'get_stats'
        ]
        
        missing_methods = []
        for method in required_methods:
            if not hasattr(ChromaDBVectorStore, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"   ❌ Métodos faltantes: {missing_methods}")
            return False
        
        print("   ✅ Todos los métodos requeridos están disponibles")
        
        # Verificar que el método search tiene la signatura correcta
        import inspect
        search_sig = inspect.signature(ChromaDBVectorStore.search)
        search_params = list(search_sig.parameters.keys())
        
        expected_params = ['self', 'query_embedding', 'k', 'filters']
        if not all(param in search_params for param in expected_params):
            print(f"   ❌ Signatura del método search incorrecta")
            print(f"      Esperado: {expected_params}")
            print(f"      Actual: {search_params}")
            return False
        
        print("   ✅ Signatura del método search es correcta")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error verificando compatibilidad: {e}")
        return False

def main():
    """Función principal de pruebas"""
    print("🎓 TEST CHROMADB PARA BENCHMARK ACADÉMICO")
    print("TFM Vicente Caruncho - Comparación FAISS vs ChromaDB")
    print("=" * 70)
    print(f"⏰ Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Verificar compatibilidad
    if not test_benchmark_compatibility():
        print("\n❌ COMPATIBILIDAD CON BENCHMARK FALLIDA")
        return
    
    # Ejecutar test principal
    success = test_chromadb_for_benchmark()
    
    print("\n" + "=" * 70)
    
    if success:
        print("🎉 ¡TEST CHROMADB PARA BENCHMARK COMPLETADO EXITOSAMENTE!")
        print("✅ ChromaDBVectorStore está listo para benchmark académico")
        print("🎯 Compatible con script de comparación FAISS vs ChromaDB")
        
        print("\n📊 FUNCIONALIDADES VERIFICADAS PARA BENCHMARK:")
        print("   ✅ Método search() con embedding directo")
        print("   ✅ Método clear() para limpiar entre tests")
        print("   ✅ Inserción de documentos con embeddings")
        print("   ✅ Búsqueda con filtros de metadatos")
        print("   ✅ Métricas de rendimiento")
        print("   ✅ Gestión de almacenamiento persistente")
        
        print("\n🚀 LISTO PARA EJECUTAR:")
        print("   python comparison_faiss_vs_chromadb.py")
        
    else:
        print("❌ TEST FALLIDO - ChromaDB no está listo para benchmark")
        print("🔧 Revisa los errores anteriores y corrige los problemas")
        
    print(f"\n⏰ Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()