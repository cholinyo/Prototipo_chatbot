#!/usr/bin/env python3
"""
Script de prueba para EmbeddingService
Prototipo_chatbot - TFM Vicente Caruncho
"""

import sys
import os
import time
from pathlib import Path

# Añadir el directorio raíz al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_embedding_service():
    """Prueba completa del EmbeddingService"""
    print("🚀 Iniciando prueba del EmbeddingService...")
    print("=" * 60)
    
    try:
        print("📦 1. Importando módulos...")
        
        # Test de imports
        from app.services.rag.embeddings import (
            EmbeddingService, 
            EmbeddingCache, 
            EmbeddingMetrics,
            embedding_service,
            encode_text,
            encode_texts,
            is_embedding_service_available
        )
        print("   ✅ Módulos importados correctamente")
        
    except ImportError as e:
        print(f"   ❌ Error importando: {e}")
        print("   💡 Verifica que sentence-transformers esté instalado:")
        print("   pip install sentence-transformers")
        return False
    
    try:
        print("\n🔧 2. Verificando disponibilidad del servicio...")
        
        # Verificar que el servicio esté disponible
        is_available = is_embedding_service_available()
        print(f"   ✅ Servicio disponible: {is_available}")
        
        if not is_available:
            print("   ⚠️  Servicio no disponible, verificando dependencias...")
            return False
        
        # Obtener info del modelo
        model_info = embedding_service.get_model_info()
        print(f"   ✅ Modelo: {model_info.get('model_name', 'unknown')}")
        print(f"   ✅ Dimensión: {model_info.get('dimension', 'unknown')}")
        print(f"   ✅ Dispositivo: {model_info.get('device', 'unknown')}")
        
    except Exception as e:
        print(f"   ❌ Error verificando servicio: {e}")
        return False
    
    try:
        print("\n📝 3. Probando embedding de texto individual...")
        
        # Test con texto simple
        test_text = "Este es un texto de prueba para el sistema RAG."
        
        start_time = time.time()
        embedding = encode_text(test_text)
        encoding_time = time.time() - start_time
        
        print(f"   ✅ Texto procesado: '{test_text[:50]}...'")
        print(f"   ✅ Embedding generado: shape {embedding.shape}")
        print(f"   ✅ Tiempo: {encoding_time:.3f}s")
        print(f"   ✅ Tipo: {type(embedding)}")
        
        # Verificar que el embedding es válido
        assert len(embedding.shape) == 1, "Embedding debe ser 1D"
        assert embedding.shape[0] > 0, "Embedding debe tener dimensión > 0"
        
    except Exception as e:
        print(f"   ❌ Error en embedding individual: {e}")
        return False
    
    try:
        print("\n📚 4. Probando embedding de múltiples textos...")
        
        # Test con múltiples textos
        test_texts = [
            "Administración local y servicios públicos",
            "Tramitación de expedientes municipales",
            "Gestión de licencias y permisos",
            "Atención ciudadana y consultas",
            "Normativa municipal y ordenanzas"
        ]
        
        start_time = time.time()
        embeddings = encode_texts(test_texts)
        batch_time = time.time() - start_time
        
        print(f"   ✅ Textos procesados: {len(test_texts)}")
        print(f"   ✅ Embeddings generados: {len(embeddings)}")
        print(f"   ✅ Tiempo total: {batch_time:.3f}s")
        print(f"   ✅ Tiempo promedio por texto: {batch_time/len(test_texts):.3f}s")
        
        # Verificar consistency
        for i, emb in enumerate(embeddings):
            assert emb.shape == embedding.shape, f"Inconsistencia en embedding {i}"
        
    except Exception as e:
        print(f"   ❌ Error en embedding batch: {e}")
        return False
    
    try:
        print("\n💾 5. Probando sistema de cache...")
        
        # Test del cache con el mismo texto
        cache_test_text = "Texto para probar el sistema de cache de embeddings"
        
        # Primera llamada (debería usar el modelo)
        start_time = time.time()
        embedding1 = encode_text(cache_test_text)
        first_time = time.time() - start_time
        
        # Segunda llamada (debería usar cache)
        start_time = time.time()
        embedding2 = encode_text(cache_test_text)
        second_time = time.time() - start_time
        
        print(f"   ✅ Primera llamada: {first_time:.3f}s")
        print(f"   ✅ Segunda llamada: {second_time:.3f}s")
        print(f"   ✅ Mejora cache: {((first_time - second_time) / first_time * 100):.1f}%")
        
        # Verificar que los embeddings son idénticos
        import numpy as np
        are_equal = np.allclose(embedding1, embedding2)
        print(f"   ✅ Embeddings idénticos: {are_equal}")
        
    except Exception as e:
        print(f"   ❌ Error en test de cache: {e}")
        return False
    
    try:
        print("\n📊 6. Verificando métricas y estadísticas...")
        
        # Obtener estadísticas
        stats = embedding_service.get_stats()
        
        print("   ✅ Estadísticas del servicio:")
        print(f"      - Servicio disponible: {stats.get('service', {}).get('available', False)}")
        print(f"      - Textos procesados: {stats.get('metrics', {}).get('total_texts_processed', 0)}")
        print(f"      - Cache hits: {stats.get('metrics', {}).get('cache_hits', 0)}")
        print(f"      - Cache misses: {stats.get('metrics', {}).get('cache_misses', 0)}")
        print(f"      - Cache hit rate: {stats.get('metrics', {}).get('cache_hit_rate', 0):.2%}")
        
        if 'cache' in stats:
            cache_stats = stats['cache']
            print(f"      - Entradas en cache: {cache_stats.get('total_entries', 0)}")
            print(f"      - Tamaño cache: {cache_stats.get('total_size_mb', 0):.2f} MB")
        
    except Exception as e:
        print(f"   ❌ Error obteniendo estadísticas: {e}")
        return False
    
    try:
        print("\n🔄 7. Probando casos especiales...")
        
        # Test con texto vacío
        try:
            empty_embedding = encode_text("")
            print("   ⚠️  Texto vacío debería lanzar error")
        except ValueError:
            print("   ✅ Texto vacío correctamente rechazado")
        
        # Test con texto muy largo
        long_text = "Este es un texto muy largo que se repite muchas veces. " * 100
        long_embedding = encode_text(long_text)
        print(f"   ✅ Texto largo procesado: {len(long_text)} chars -> {long_embedding.shape}")
        
        # Test con caracteres especiales
        special_text = "Texto con ácentos, ñ, y símbolos: €$@#%^&*()"
        special_embedding = encode_text(special_text)
        print(f"   ✅ Caracteres especiales: {special_embedding.shape}")
        
    except Exception as e:
        print(f"   ❌ Error en casos especiales: {e}")
        return False
    
    try:
        print("\n🧪 8. Probando DocumentChunk integration...")
        
        # Crear chunks de prueba
        from app.models import DocumentChunk, DocumentMetadata
        
        # Crear metadatos de prueba
        from datetime import datetime
        metadata = DocumentMetadata(
            source_path="test_document.pdf",
            source_type="document",
            file_type=".pdf",
            size_bytes=1024,
            created_at=datetime.now(),
            processed_at=datetime.now(),
            checksum="test_checksum"
        )
        
        # Crear chunks
        test_chunks = [
            DocumentChunk(
                id=f"test-chunk-{i}",
                content=f"Contenido del chunk {i+1} para testing",
                metadata=metadata,
                chunk_index=i,
                chunk_size=len(f"Contenido del chunk {i+1} para testing"),
                start_char=i*100,
                end_char=(i+1)*100
            )
            for i in range(3)
        ]
        
        print(f"   ✅ Creados {len(test_chunks)} chunks de prueba")
        
        # Generar embeddings para chunks
        embedded_chunks = embedding_service.encode_documents(test_chunks)
        
        print(f"   ✅ Embeddings generados para {len(embedded_chunks)} chunks")
        
        # Verificar que tienen embeddings
        for i, chunk in enumerate(embedded_chunks):
            assert hasattr(chunk, 'embedding'), f"Chunk {i} sin embedding"
            assert chunk.embedding is not None, f"Chunk {i} embedding es None"
            print(f"   ✅ Chunk {i+1}: embedding shape {chunk.embedding.shape}")
        
    except Exception as e:
        print(f"   ❌ Error en DocumentChunk integration: {e}")
        return False
    
    try:
        print("\n🚀 9. Test de rendimiento...")
        
        # Test de rendimiento con diferentes tamaños
        performance_texts = [f"Texto de rendimiento número {i}" for i in range(50)]
        
        start_time = time.time()
        perf_embeddings = encode_texts(performance_texts)
        perf_time = time.time() - start_time
        
        print(f"   ✅ {len(performance_texts)} textos en {perf_time:.3f}s")
        print(f"   ✅ Throughput: {len(performance_texts)/perf_time:.1f} textos/segundo")
        
        # Verificar memoria
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"   ✅ Uso de memoria: {memory_mb:.1f} MB")
        except ImportError:
            print("   ℹ️  psutil no disponible para verificar memoria")
        
    except Exception as e:
        print(f"   ❌ Error en test de rendimiento: {e}")
        return False
    
    print("\n🎉 ¡Todos los tests del EmbeddingService completados exitosamente!")
    print("=" * 60)
    
    # Resumen final
    final_stats = embedding_service.get_stats()
    metrics = final_stats.get('metrics', {})
    
    print("\n📊 RESUMEN FINAL:")
    print(f"✅ Modelo: {final_stats.get('model', {}).get('model_name', 'unknown')}")
    print(f"✅ Dimensión: {final_stats.get('model', {}).get('dimension', 'unknown')}")
    print(f"✅ Dispositivo: {final_stats.get('model', {}).get('device', 'unknown')}")
    print(f"✅ Textos procesados: {metrics.get('total_texts_processed', 0)}")
    print(f"✅ Tiempo total: {metrics.get('total_processing_time', 0):.3f}s")
    print(f"✅ Tiempo promedio: {metrics.get('average_time_per_text', 0):.3f}s")
    print(f"✅ Cache hit rate: {metrics.get('cache_hit_rate', 0):.2%}")
    
    if 'cache' in final_stats:
        cache_stats = final_stats['cache']
        print(f"✅ Entradas en cache: {cache_stats.get('total_entries', 0)}")
        print(f"✅ Tamaño cache: {cache_stats.get('total_size_mb', 0):.2f} MB")
    
    print("\n🎯 EmbeddingService listo para integración con Vector Stores!")
    return True


def test_requirements():
    """Verificar que están instaladas las dependencias necesarias"""
    print("🔍 Verificando dependencias...")
    
    missing_deps = []
    
    try:
        import sentence_transformers
        print("   ✅ sentence-transformers")
    except ImportError:
        missing_deps.append("sentence-transformers")
        print("   ❌ sentence-transformers")
    
    try:
        import torch
        print("   ✅ torch")
    except ImportError:
        missing_deps.append("torch")
        print("   ❌ torch")
    
    try:
        import numpy
        print("   ✅ numpy")
    except ImportError:
        missing_deps.append("numpy")
        print("   ❌ numpy")
    
    if missing_deps:
        print(f"\n❌ Faltan dependencias: {', '.join(missing_deps)}")
        print("💡 Instala con:")
        print("   pip install sentence-transformers torch numpy")
        return False
    
    print("✅ Todas las dependencias están instaladas")
    return True


if __name__ == "__main__":
    print("🧪 Test del EmbeddingService - Prototipo_chatbot")
    print("=" * 60)
    
    # Verificar dependencias primero
    if not test_requirements():
        print("\n💔 No se pueden ejecutar los tests sin las dependencias")
        sys.exit(1)
    
    # Ejecutar tests
    try:
        success = test_embedding_service()
        
        if success:
            print("\n🎉 ¡TESTS COMPLETADOS EXITOSAMENTE!")
            print("👉 Puedes continuar con la implementación de Vector Stores")
            sys.exit(0)
        else:
            print("\n💔 Algunos tests fallaron")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrumpidos por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        sys.exit(1)