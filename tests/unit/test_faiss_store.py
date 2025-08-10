#!/usr/bin/env python3
"""
Script de prueba para FaissVectorStore
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

def test_faiss_dependencies():
    """Verificar dependencias de FAISS"""
    print("🔍 Verificando dependencias FAISS...")
    
    try:
        import faiss
        print(f"   ✅ faiss: {faiss.__version__ if hasattr(faiss, '__version__') else 'instalado'}")
    except ImportError:
        print("   ❌ faiss: No instalado")
        print("   💡 Instalar con: pip install faiss-cpu")
        return False
    
    try:
        import numpy as np
        print(f"   ✅ numpy: {np.__version__}")
    except ImportError:
        print("   ❌ numpy: No instalado")
        return False
    
    return True

def test_faiss_vector_store():
    """Prueba completa del FaissVectorStore"""
    print("🚀 Iniciando prueba del FaissVectorStore...")
    print("=" * 60)
    
    try:
        print("📦 1. Importando módulos...")
        
        from app.services.rag.faiss_store import (
            FaissVectorStore, 
            FaissMetrics,
            create_faiss_store,
            is_faiss_available
        )
        from app.services.rag.embeddings import embedding_service
        from app.models import DocumentChunk, DocumentMetadata
        
        print("   ✅ Módulos importados correctamente")
        
    except ImportError as e:
        print(f"   ❌ Error importando: {e}")
        return False
    
    try:
        print("\n🔧 2. Verificando disponibilidad...")
        
        # Verificar que FAISS esté disponible
        if not is_faiss_available():
            print("   ❌ FAISS no disponible")
            return False
        
        print("   ✅ FAISS disponible")
        
        # Verificar que EmbeddingService esté disponible
        if not embedding_service.is_available():
            print("   ❌ EmbeddingService no disponible")
            return False
        
        print("   ✅ EmbeddingService disponible")
        
    except Exception as e:
        print(f"   ❌ Error verificando servicios: {e}")
        return False
    
    try:
        print("\n🏗️ 3. Creando FaissVectorStore de prueba...")
        
        # Crear store en directorio temporal
        test_store_path = "data/test_faiss"
        store = create_faiss_store(
            store_path=test_store_path,
            dimension=384,
            index_type="IndexFlatL2"
        )
        
        print(f"   ✅ Store creado en: {test_store_path}")
        
        # Verificar estado inicial
        stats = store.get_stats()
        print(f"   ✅ Vectores iniciales: {stats.get('total_vectors', 0)}")
        print(f"   ✅ Tipo de índice: {stats.get('index_type', 'unknown')}")
        
    except Exception as e:
        print(f"   ❌ Error creando store: {e}")
        return False
    
    try:
        print("\n📝 4. Creando documentos de prueba...")
        
        # Crear metadatos de prueba
        test_metadata = DocumentMetadata(
            source_path="test_document.pdf",
            source_type="document",
            file_type=".pdf",
            size_bytes=1024,
            created_at=datetime.now(),
            processed_at=datetime.now(),
            checksum="test_checksum"
        )
        
        # Crear chunks de prueba
        test_texts = [
            "Administración local y servicios públicos municipales",
            "Tramitación de expedientes y licencias de actividad",
            "Gestión de padrones y registros administrativos",
            "Atención ciudadana y ventanilla única",
            "Normativa municipal y ordenanzas locales",
            "Presupuestos y hacienda municipal",
            "Urbanismo y planeamiento territorial",
            "Servicios sociales y bienestar ciudadano"
        ]
        
        test_chunks = []
        for i, text in enumerate(test_texts):
            chunk = DocumentChunk(
                id=f"test-chunk-{i+1}",
                content=text,
                metadata=test_metadata,
                chunk_index=i,
                chunk_size=len(text),
                start_char=i*100,
                end_char=(i+1)*100
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
        print("\n💾 6. Añadiendo documentos a FAISS...")
        
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
        print(f"   ✅ Total vectores: {stats.get('total_vectors', 0)}")
        print(f"   ✅ Memoria usada: {stats.get('index_size_mb', 0):.2f} MB")
        
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
        query_text = "licencias de actividad municipal"
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
        print(f"      - Memoria usada: {stats.get('index_size_mb', 0):.2f} MB")
        
    except Exception as e:
        print(f"   ❌ Error obteniendo métricas: {e}")
        return False
    
    try:
        print("\n💾 10. Probando persistencia...")
        
        # Crear nuevo store apuntando al mismo directorio
        store2 = FaissVectorStore(store_path=test_store_path)
        
        # Verificar que cargó los datos
        stats2 = store2.get_stats()
        print(f"   ✅ Store2 cargado con {stats2.get('total_vectors', 0)} vectores")
        
        # Verificar que puede buscar
        results2 = store2.search(query_embedding, k=2)
        print(f"   ✅ Store2 búsqueda: {len(results2)} resultados")
        
        # Verificar que los resultados son consistentes
        if len(results) > 0 and len(results2) > 0:
            assert results[0][0].id == results2[0][0].id, "Resultados inconsistentes"
            print("   ✅ Persistencia funcionando correctamente")
        
    except Exception as e:
        print(f"   ❌ Error en persistencia: {e}")
        return False
    
    try:
        print("\n🧪 11. Test de rendimiento...")
        
        # Test de rendimiento con más documentos
        performance_chunks = []
        for i in range(50):
            chunk = DocumentChunk(
                id=f"perf-chunk-{i}",
                content=f"Documento de rendimiento número {i} con contenido variado para testing",
                metadata=test_metadata,
                chunk_index=i,
                chunk_size=70,
                start_char=i*100,
                end_char=(i+1)*100
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
        queries = ["documento", "rendimiento", "testing", "contenido"]
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
    
    print("\n🎉 ¡Todos los tests de FaissVectorStore completados exitosamente!")
    print("=" * 60)
    
    # Resumen final
    final_stats = store.get_stats()
    print("\n📊 RESUMEN FINAL:")
    print(f"✅ Tipo de índice: {final_stats.get('index_type', 'unknown')}")
    print(f"✅ Total vectores: {final_stats.get('total_vectors', 0)}")
    print(f"✅ Dimensión: {final_stats.get('dimension', 0)}")
    print(f"✅ Memoria usada: {final_stats.get('index_size_mb', 0):.2f} MB")
    
    metrics = final_stats.get('metrics', {})
    print(f"✅ Operaciones add: {metrics.get('total_add_operations', 0)}")
    print(f"✅ Operaciones search: {metrics.get('total_search_operations', 0)}")
    print(f"✅ Tiempo promedio add: {metrics.get('avg_add_time', 0):.3f}s")
    print(f"✅ Tiempo promedio search: {metrics.get('avg_search_time', 0):.3f}s")
    
    print("\n🎯 FaissVectorStore listo para producción!")
    return True


def cleanup_test_data():
    """Limpiar datos de prueba"""
    import shutil
    test_dir = Path("data/test_faiss")
    if test_dir.exists():
        try:
            shutil.rmtree(test_dir)
            print("🧹 Datos de prueba limpiados")
        except Exception as e:
            print(f"⚠️  Error limpiando: {e}")


if __name__ == "__main__":
    print("🧪 Test del FaissVectorStore - Prototipo_chatbot")
    print("=" * 60)
    
    # Verificar dependencias primero
    if not test_faiss_dependencies():
        print("\n💔 No se pueden ejecutar los tests sin las dependencias")
        print("💡 Instalar con: pip install faiss-cpu")
        sys.exit(1)
    
    # Ejecutar tests
    try:
        success = test_faiss_vector_store()
        
        if success:
            print("\n🎉 ¡TESTS DE FAISS COMPLETADOS EXITOSAMENTE!")
            print("👉 Listo para continuar con ChromaDB")
            cleanup_test_data()
            sys.exit(0)
        else:
            print("\n💔 Algunos tests de FAISS fallaron")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrumpidos por el usuario")
        cleanup_test_data()
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error inesperado en tests de FAISS: {e}")
        cleanup_test_data()
        sys.exit(1) 