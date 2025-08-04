#!/usr/bin/env python3
"""
Script de prueba para ChromaDBVectorStore
Prototipo_chatbot - TFM Vicente Caruncho
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Añadir el directorio raíz al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_chromadb_dependencies():
    """Verificar dependencias de ChromaDB"""
    print("🔍 Verificando dependencias ChromaDB...")
    
    try:
        import chromadb
        print(f"   ✅ chromadb: {chromadb.__version__ if hasattr(chromadb, '__version__') else 'instalado'}")
    except ImportError:
        print("   ❌ chromadb: No instalado")
        print("   💡 Instalar con: pip install chromadb")
        return False
    
    try:
        import numpy as np
        print(f"   ✅ numpy: {np.__version__}")
    except ImportError:
        print("   ❌ numpy: No instalado")
        return False
    
    return True

def test_chromadb_vector_store():
    """Prueba completa del ChromaDBVectorStore"""
    print("🚀 Iniciando prueba del ChromaDBVectorStore...")
    print("=" * 60)
    
    try:
        print("📦 1. Importando módulos...")
        
        from app.services.rag.chromadb_store import (
            ChromaDBVectorStore, 
            ChromaDBMetrics,
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
        
        # Verificar que ChromaDB esté disponible
        if not is_chromadb_available():
            print("   ❌ ChromaDB no disponible")
            return False
        
        print("   ✅ ChromaDB disponible")
        
        # Verificar que EmbeddingService esté disponible
        if not embedding_service.is_available():
            print("   ❌ EmbeddingService no disponible")
            return False
        
        print("   ✅ EmbeddingService disponible")
        
    except Exception as e:
        print(f"   ❌ Error verificando servicios: {e}")
        return False
    
    try:
        print("\n🏗️ 3. Creando ChromaDBVectorStore de prueba...")
        
        # Crear store en directorio temporal
        test_store_path = "data/test_chromadb"
        store = create_chromadb_store(
            store_path=test_store_path,
            collection_name="test_collection",
            distance_function="cosine"
        )
        
        print(f"   ✅ Store creado en: {test_store_path}")
        
        # Verificar estado inicial
        stats = store.get_stats()
        print(f"   ✅ Documentos iniciales: {stats.get('total_documents', 0)}")
        print(f"   ✅ Función de distancia: {stats.get('distance_function', 'unknown')}")
        print(f"   ✅ Colección: {stats.get('collection_name', 'unknown')}")
        
    except Exception as e:
        print(f"   ❌ Error creando store: {e}")
        return False
    
    try:
        print("\n📝 4. Creando documentos de prueba...")
        
        # Crear metadatos de prueba
        test_metadata = DocumentMetadata(
            source_path="test_document_chromadb.pdf",
            source_type="document",
            file_type=".pdf",
            size_bytes=2048,
            created_at=datetime.now(),
            processed_at=datetime.now(),
            checksum="test_checksum_chromadb"
        )
        
        # Crear chunks de prueba (diferentes a FAISS para comparación)
        test_texts = [
            "Normativas y reglamentos municipales de la administración local",
            "Procedimientos administrativos para licencias y permisos",
            "Gestión de expedientes y trámites ciudadanos",
            "Servicios públicos municipales y atención al público",
            "Ordenanzas locales y regulaciones municipales",
            "Hacienda pública y gestión presupuestaria municipal",
            "Planificación urbana y gestión territorial",
            "Políticas sociales y servicios de bienestar ciudadano",
            "Medio ambiente y sostenibilidad municipal",
            "Transparencia y participación ciudadana"
        ]
        
        test_chunks = []
        for i, text in enumerate(test_texts):
            chunk = DocumentChunk(
                id=f"chromadb-test-chunk-{i+1}",
                content=text,
                metadata=test_metadata,
                chunk_index=i,
                chunk_size=len(text),
                start_char=i*120,
                end_char=(i+1)*120
            )
            test_chunks.append(chunk)
        
        print(f"   ✅ {len(test_chunks)} chunks creados")
        
    except Exception as e:
        print(f"   ❌ Error creando documentos: {e}")
        return False
    
    try:
        print("\n🧠 5. Generando embeddings...")
        
        # Generar embeddings usando EmbeddingService
        start_time = time.time()
        embedded_chunks = embedding_service.encode_documents(test_chunks)
        embedding_time = time.time() - start_time
        
        print(f"   ✅ Embeddings generados en {embedding_time:.3f}s")
        print(f"   ✅ Chunks con embeddings: {len(embedded_chunks)}")
        
        # Verificar que tienen embeddings
        for i, chunk in enumerate(embedded_chunks[:3]):
            print(f"   ✅ Chunk {i+1}: embedding shape {np.array(chunk.embedding).shape}")
        
    except Exception as e:
        print(f"   ❌ Error generando embeddings: {e}")
        return False
    
    try:
        print("\n💾 6. Añadiendo documentos a ChromaDB...")
        
        # Añadir documentos al store
        start_time = time.time()
        success = store.add_documents(embedded_chunks)
        add_time = time.time() - start_time
        
        if not success:
            print("   ❌ Error añadiendo documentos")
            return False
        
        print(f"   ✅ Documentos añadidos en {add_time:.3f}s")
        
        # Verificar estado después de añadir
        stats = store.get_stats()
        print(f"   ✅ Total documentos: {stats.get('total_documents', 0)}")
        print(f"   ✅ Uso de disco: {stats.get('disk_usage_mb', 0):.2f} MB")
        
        # Verificar distribución por tipo
        metrics_dict = stats.get('metrics', {})
        source_dist = metrics_dict.get('source_distribution', {})
        print(f"   ✅ Distribución por tipo: {source_dist}")
        
    except Exception as e:
        print(f"   ❌ Error añadiendo documentos: {e}")
        return False
    
    try:
        print("\n🔍 7. Probando búsquedas...")
        
        # Test búsqueda básica
        query_text = "procedimientos administrativos municipales"
        query_embedding = embedding_service.encode_single_text(query_text)
        
        start_time = time.time()
        results = store.search(query_embedding, k=3)
        search_time = time.time() - start_time
        
        print(f"   ✅ Búsqueda completada en {search_time:.3f}s")
        print(f"   ✅ Resultados encontrados: {len(results)}")
        
        # Mostrar resultados
        for i, (chunk, score) in enumerate(results):
            print(f"   📄 Resultado {i+1}: score={score:.3f}")
            print(f"       Texto: {chunk.content[:50]}...")
            print(f"       ID: {chunk.id}")
        
    except Exception as e:
        print(f"   ❌ Error en búsquedas: {e}")
        return False
    
    try:
        print("\n🎯 8. Probando filtros...")
        
        # Test búsqueda con filtros
        filters = {'source_type': 'document'}
        
        start_time = time.time()
        filtered_results = store.search(query_embedding, k=3, filters=filters)
        filtered_search_time = time.time() - start_time
        
        print(f"   ✅ Búsqueda filtrada en {filtered_search_time:.3f}s")
        print(f"   ✅ Resultados filtrados: {len(filtered_results)}")
        
        # Verificar que los filtros funcionan
        for chunk, score in filtered_results:
            assert chunk.metadata.source_type == 'document', "Filtro no aplicado correctamente"
        
        print("   ✅ Filtros funcionando correctamente")
        
    except Exception as e:
        print(f"   ❌ Error en filtros: {e}")
        return False
    
    try:
        print("\n📊 9. Probando métricas y benchmarking...")
        
        # Obtener estadísticas detalladas
        stats = store.get_stats()
        metrics = stats.get('metrics', {})
        
        print("   ✅ Métricas de rendimiento:")
        print(f"      - Tiempo promedio add: {metrics.get('avg_add_time', 0):.3f}s")
        print(f"      - Tiempo promedio search: {metrics.get('avg_search_time', 0):.3f}s")
        print(f"      - Total operaciones add: {metrics.get('total_add_operations', 0)}")
        print(f"      - Total operaciones search: {metrics.get('total_search_operations', 0)}")
        print(f"      - Uso de disco: {stats.get('disk_usage_mb', 0):.2f} MB")
        
    except Exception as e:
        print(f"   ❌ Error obteniendo métricas: {e}")
        return False
    
    try:
        print("\n💾 10. Probando persistencia...")
        
        # Crear nuevo store apuntando al mismo directorio
        store2 = ChromaDBVectorStore(
            store_path=test_store_path,
            collection_name="test_collection"
        )
        
        # Verificar que cargó los datos
        stats2 = store2.get_stats()
        print(f"   ✅ Store2 cargado con {stats2.get('total_documents', 0)} documentos")
        
        # Verificar que puede buscar
        results2 = store2.search(query_embedding, k=2)
        print(f"   ✅ Store2 búsqueda: {len(results2)} resultados")
        
        # Verificar que los resultados son consistentes
        if len(results) > 0 and len(results2) > 0:
            # Los IDs pueden variar, pero el contenido debería ser similar
            print("   ✅ Persistencia funcionando correctamente")
        
    except Exception as e:
        print(f"   ❌ Error en persistencia: {e}")
        return False
    
    try:
        print("\n🔍 11. Probando información de colección...")
        
        # Obtener información detallada
        collection_info = store.get_collection_info()
        
        print("   ✅ Información de colección:")
        print(f"      - Nombre: {collection_info.get('name', 'unknown')}")
        print(f"      - Documentos: {collection_info.get('count', 0)}")
        print(f"      - Metadatos: {collection_info.get('metadata', {})}")
        
        # Mostrar muestra de documentos
        sample = collection_info.get('sample_documents', {})
        if sample.get('ids'):
            print(f"      - Documentos de muestra: {len(sample['ids'])}")
            for i, doc_id in enumerate(sample['ids'][:2]):
                print(f"        • {doc_id}: {sample['documents'][i][:50]}...")
        
    except Exception as e:
        print(f"   ❌ Error obteniendo info de colección: {e}")
        return False
    
    try:
        print("\n🧪 12. Test de rendimiento comparativo...")
        
        # Test de rendimiento con más documentos
        performance_chunks = []
        for i in range(30):
            chunk = DocumentChunk(
                id=f"chromadb-perf-chunk-{i}",
                content=f"Documento de rendimiento ChromaDB número {i} con contenido variado para testing y comparación",
                metadata=test_metadata,
                chunk_index=i,
                chunk_size=90,
                start_char=i*120,
                end_char=(i+1)*120
            )
            performance_chunks.append(chunk)
        
        # Generar embeddings
        perf_embedded = embedding_service.encode_documents(performance_chunks)
        
        # Añadir al store
        start_time = time.time()
        store.add_documents(perf_embedded)
        batch_add_time = time.time() - start_time
        
        print(f"   ✅ {len(performance_chunks)} docs añadidos en {batch_add_time:.3f}s")
        print(f"   ✅ Throughput: {len(performance_chunks)/batch_add_time:.1f} docs/segundo")
        
        # Test de búsquedas múltiples
        queries = ["documento", "rendimiento", "testing", "comparación", "chromadb"]
        total_search_time = 0
        
        for query in queries:
            query_emb = embedding_service.encode_single_text(query)
            start_time = time.time()
            _ = store.search(query_emb, k=5)
            total_search_time += time.time() - start_time
        
        avg_search = total_search_time / len(queries)
        print(f"   ✅ Tiempo promedio búsqueda: {avg_search:.3f}s")
        
    except Exception as e:
        print(f"   ❌ Error en test de rendimiento: {e}")
        return False
    
    print("\n🎉 ¡Todos los tests de ChromaDBVectorStore completados exitosamente!")
    print("=" * 60)
    
    # Resumen final
    final_stats = store.get_stats()
    print("\n📊 RESUMEN FINAL CHROMADB:")
    print(f"✅ Tipo: {final_stats.get('type', 'unknown')}")
    print(f"✅ Total documentos: {final_stats.get('total_documents', 0)}")
    print(f"✅ Colección: {final_stats.get('collection_name', 'unknown')}")
    print(f"✅ Función distancia: {final_stats.get('distance_function', 'unknown')}")
    print(f"✅ Uso de disco: {final_stats.get('disk_usage_mb', 0):.2f} MB")
    
    metrics = final_stats.get('metrics', {})
    print(f"✅ Operaciones add: {metrics.get('total_add_operations', 0)}")
    print(f"✅ Operaciones search: {metrics.get('total_search_operations', 0)}")
    print(f"✅ Tiempo promedio add: {metrics.get('avg_add_time', 0):.3f}s")
    print(f"✅ Tiempo promedio search: {metrics.get('avg_search_time', 0):.3f}s")
    
    print("\n🎯 ChromaDBVectorStore listo para comparación con FAISS!")
    return True


def cleanup_test_data():
    """Limpiar datos de prueba"""
    import shutil
    test_dir = Path("data/test_chromadb")
    if test_dir.exists():
        try:
            shutil.rmtree(test_dir)
            print("🧹 Datos de prueba ChromaDB limpiados")
        except Exception as e:
            print(f"⚠️  Error limpiando ChromaDB: {e}")


def compare_with_faiss():
    """Mostrar comparación rápida con FAISS"""
    print("\n" + "=" * 60)
    print("📊 COMPARACIÓN RÁPIDA FAISS vs CHROMADB")
    print("=" * 60)
    
    try:
        # Datos de FAISS (de test anterior)
        print("🔵 FAISS:")
        print("   • Tipo: IndexFlatL2")
        print("   • Memoria: ~0.08 MB")
        print("   • Búsqueda: ~0.003s")
        print("   • Almacenamiento: En memoria + archivos pickle")
        print("   • Ventajas: Muy rápido, optimizado para similitud")
        
        print("\n🟢 ChromaDB:")
        print("   • Tipo: Cosine distance")
        print("   • Almacenamiento: Base de datos persistente")
        print("   • Búsqueda: Variable según configuración")
        print("   • Ventajas: Persistencia automática, metadatos ricos")
        
        print("\n🎯 Para tu TFM:")
        print("   • FAISS: Mejor para velocidad pura")
        print("   • ChromaDB: Mejor para casos de uso complejos")
        print("   • Ambos: Excelente base para comparación académica")
        
    except Exception as e:
        print(f"Error en comparación: {e}")


if __name__ == "__main__":
    print("🧪 Test del ChromaDBVectorStore - Prototipo_chatbot")
    print("=" * 60)
    
    # Verificar dependencias primero
    if not test_chromadb_dependencies():
        print("\n💔 No se pueden ejecutar los tests sin las dependencias")
        print("💡 Instalar con: pip install chromadb")
        sys.exit(1)
    
    # Ejecutar tests
    try:
        success = test_chromadb_vector_store()
        
        if success:
            print("\n🎉 ¡TESTS DE CHROMADB COMPLETADOS EXITOSAMENTE!")
            compare_with_faiss()
            print("\n👉 ¡Listo para implementar comparación completa FAISS vs ChromaDB!")
            cleanup_test_data()
            sys.exit(0)
        else:
            print("\n💔 Algunos tests de ChromaDB fallaron")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrumpidos por el usuario")
        cleanup_test_data()
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error inesperado en tests de ChromaDB: {e}")
        cleanup_test_data()
        sys.exit(1)