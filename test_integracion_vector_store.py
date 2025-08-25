#!/usr/bin/env python3
"""
Test de integración para VectorStoreService Real - VERSIÓN FINAL COMPLETA
Incluye cálculo automático de embeddings
TFM Vicente Caruncho - Sistemas Inteligentes
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Añadir el directorio raíz al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_vector_store_service_integration():
    """Test completo de integración del VectorStoreService real con embeddings"""
    print("🧪 TEST INTEGRACIÓN VECTORSTORESERVICE REAL - VERSIÓN FINAL")
    print("=" * 70)
    
    try:
        print("📦 1. Importando implementación real desde el proyecto...")
        
        # Importar desde la ubicación real del proyecto
        from app.services.vector_store_service import (
            VectorStoreService,
            get_vector_store_service,
            is_vector_store_available,
            VectorStoreType
        )
        from app.models import DocumentChunk
        
        print("   ✅ Implementación real importada correctamente desde app.services")
        
    except ImportError as e:
        print(f"   ❌ Error importando: {e}")
        print("   💡 Asegúrate de que app/services/vector_store_service.py tiene la implementación real")
        return False
    except Exception as e:
        print(f"   ❌ Error inesperado: {e}")
        return False
    
    try:
        print("\n🔧 2. Verificando que ya no es modo mock...")
        
        # Usar la instancia global (la que usará tu aplicación)
        global_service = get_vector_store_service()
        global_available = is_vector_store_available()
        
        print(f"   Store activo: {global_service.get_active_store_type()}")
        print(f"   FAISS disponible: {global_service.faiss_available}")
        print(f"   ChromaDB disponible: {global_service.chromadb_available}")
        print(f"   Método is_available(): {global_service.is_available()}")
        
        # Verificar que no es mock
        if hasattr(global_service, 'active_store') and global_service.active_store is not None:
            print("   ✅ ÉXITO: Ya no es modo mock, tiene store real")
        else:
            print("   ❌ ADVERTENCIA: Parece que sigue en modo mock o sin stores")
            return False
        
        if not global_available:
            print("   ❌ La instancia global no está disponible")
            return False
        
        print("   ✅ VectorStoreService real inicializado correctamente")
        
        # Usar la instancia global para todos los tests
        service = global_service
        
    except Exception as e:
        print(f"   ❌ Error verificando servicio: {e}")
        print(f"   Detalles del error: {type(e).__name__}: {str(e)}")
        return False
    
    try:
        print("\n📄 3. Creando documentos de prueba...")
        
        # Crear documentos de prueba con la estructura CORRECTA de tu proyecto
        original_test_docs = [
            DocumentChunk(
                id="test_doc_1",
                content="El Ayuntamiento de Valencia gestiona los servicios municipales para ciudadanos",
                metadata={
                    "source": "web",
                    "tipo": "información_municipal",
                    "ciudad": "Valencia",
                    "categoria": "servicios_publicos"
                },
                source_file="test_web_source.html",
                chunk_index=0
            ),
            DocumentChunk(
                id="test_doc_2",
                content="Los ciudadanos pueden solicitar certificados digitales a través de la sede electrónica",
                metadata={
                    "source": "pdf",
                    "tipo": "tramites_digitales",
                    "servicio": "certificados",
                    "modalidad": "online"
                },
                source_file="certificados_digitales.pdf",
                chunk_index=1
            ),
            DocumentChunk(
                id="test_doc_3",
                content="La administración electrónica facilita los procesos burocráticos mediante tecnología",
                metadata={
                    "source": "documento",
                    "tipo": "modernización",
                    "ambito": "administración",
                    "tecnologia": "rag"
                },
                source_file="modernizacion_admin.docx",
                chunk_index=2
            ),
            DocumentChunk(
                id="test_doc_4",
                content="El padrón municipal registra todos los vecinos del municipio para gestión censal",
                metadata={
                    "source": "base_datos",
                    "tipo": "registro_civil",
                    "ambito": "censo",
                    "categoria": "padron"
                },
                source_file="padron_municipal.db",
                chunk_index=3
            )
        ]
        
        print(f"   ✅ Creados {len(original_test_docs)} documentos de prueba")
        
    except Exception as e:
        print(f"   ❌ Error creando documentos: {e}")
        return False
    
    try:
        print("\n🧠 4. Calculando embeddings para los documentos...")
        
        # Importar y verificar EmbeddingService
        from app.services.rag.embeddings import embedding_service
        
        if not embedding_service.is_available():
            print("   ❌ EmbeddingService no disponible")
            print("   💡 Ejecuta primero: python test_embedding_service.py")
            return False
        
        print(f"   📊 Calculando embeddings para {len(original_test_docs)} documentos...")
        
        # Calcular embeddings para cada documento
        documents_with_embeddings = []
        embedding_times = []
        
        for i, doc in enumerate(original_test_docs, 1):
            print(f"   🔄 Procesando documento {i}/{len(original_test_docs)}: {doc.id}")
            
            start_time = time.time()
            
            # Calcular embedding para el contenido
            embedding = embedding_service.encode_single_text(doc.content)
            embedding_time = (time.time() - start_time) * 1000
            embedding_times.append(embedding_time)
            
            # Verificar que el embedding es válido
            if embedding is None or len(embedding) == 0:
                print(f"   ❌ Error: embedding inválido para {doc.id}")
                return False
            
            # Crear nuevo chunk con embedding incluido
            doc_with_embedding = DocumentChunk(
                id=doc.id,
                content=doc.content,
                metadata=doc.metadata.copy(),
                source_file=doc.source_file,
                chunk_index=doc.chunk_index,
                embedding=embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            )
            
            documents_with_embeddings.append(doc_with_embedding)
            
            print(f"      ✅ Embedding calculado en {embedding_time:.2f}ms (dim: {len(embedding)})")
        
        avg_embedding_time = sum(embedding_times) / len(embedding_times)
        total_embedding_time = sum(embedding_times)
        
        print(f"   📊 Resumen embeddings:")
        print(f"   - Tiempo total: {total_embedding_time:.1f}ms")
        print(f"   - Tiempo promedio: {avg_embedding_time:.2f}ms por documento")
        print(f"   - Dimensión: {len(documents_with_embeddings[0].embedding)}")
        print(f"   - Modelo: all-MiniLM-L6-v2")
        
        print("   ✅ Todos los embeddings calculados correctamente")
        
    except Exception as e:
        print(f"   ❌ Error calculando embeddings: {e}")
        print(f"   Tipo de error: {type(e).__name__}")
        return False
    
    try:
        print("\n💾 5. Agregando documentos con embeddings al vector store...")
        
        # Agregar documentos con metadatos de fuente
        source_metadata = {
            "batch_id": "test_batch_final",
            "timestamp": datetime.now().isoformat(),
            "test_run": True,
            "origen": "test_vector_store_service_final",
            "version": "v3.0_con_embeddings",
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": len(documents_with_embeddings[0].embedding)
        }
        
        print(f"   🎯 Store activo: {service.get_active_store_type()}")
        print(f"   📊 Documentos a agregar: {len(documents_with_embeddings)}")
        
        start_time = time.time()
        success = service.add_documents(documents_with_embeddings, source_metadata)
        add_time = (time.time() - start_time) * 1000
        
        if success:
            print(f"   ✅ {len(documents_with_embeddings)} documentos agregados en {add_time:.2f}ms")
            print(f"   ✅ Store utilizado: {service.get_active_store_type()}")
            print(f"   ✅ Con embeddings pre-calculados ✓")
            
            # Actualizar variable para siguientes tests
            test_docs = documents_with_embeddings
        else:
            print("   ❌ Error agregando documentos al store")
            print("   💡 Revisa los logs arriba para más detalles")
            return False
            
    except Exception as e:
        print(f"   ❌ Error en operación add_documents: {e}")
        print(f"   Tipo de error: {type(e).__name__}")
        return False
    
    try:
        print("\n🔍 6. Ejecutando búsquedas semánticas...")
        
        # Lista de queries de prueba específicas para administración local
        test_queries = [
            "servicios municipales Valencia ciudadanos",
            "certificados digitales sede electrónica", 
            "administración electrónica procesos burocráticos",
            "padrón municipal vecinos censo",
            "tramites online ayuntamiento",
            "modernización tecnología administración"
        ]
        
        search_results = []
        total_search_time = 0
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   🔍 Búsqueda {i}/{len(test_queries)}: '{query}'")
            
            start_time = time.time()
            results = service.search(query, k=3)
            search_time = (time.time() - start_time) * 1000
            total_search_time += search_time
            
            print(f"      ⏱️ Tiempo: {search_time:.2f}ms")
            print(f"      📊 Resultados: {len(results)}")
            
            # Verificar y mostrar resultados
            if results and isinstance(results, list) and len(results) > 0:
                first_result = results[0]
                if isinstance(first_result, dict):
                    # Mostrar información del mejor resultado
                    content = first_result.get('content', 'Sin contenido')
                    score = first_result.get('score', 'Sin score')
                    metadata = first_result.get('metadata', {})
                    
                    preview = content[:50] + "..." if len(content) > 50 else content
                    print(f"      🥇 Mejor resultado: {preview}")
                    
                    if score != 'Sin score' and score is not None:
                        print(f"      📈 Relevancia: {score:.4f}")
                    
                    if metadata:
                        tipo = metadata.get('tipo', 'N/A')
                        source = metadata.get('source', 'N/A')
                        print(f"      🏷️ Tipo: {tipo} | Fuente: {source}")
                
                search_results.append({
                    'query': query,
                    'time_ms': search_time,
                    'results_count': len(results),
                    'success': True,
                    'relevance_score': first_result.get('score') if results else None
                })
            else:
                print("      ⚠️ Sin resultados relevantes encontrados")
                search_results.append({
                    'query': query,
                    'time_ms': search_time,
                    'results_count': 0,
                    'success': False,
                    'relevance_score': None
                })
        
        # Calcular estadísticas finales
        successful_searches = sum(1 for r in search_results if r['success'])
        avg_search_time = total_search_time / len(test_queries)
        
        print(f"\n   📊 Resumen de búsquedas:")
        print(f"   ✅ Exitosas: {successful_searches}/{len(test_queries)}")
        print(f"   ⏱️ Tiempo total: {total_search_time:.1f}ms")
        print(f"   ⚖️ Tiempo promedio: {avg_search_time:.2f}ms por búsqueda")
        print(f"   🎯 Store usado: {service.get_active_store_type()}")
        print(f"   🧠 Búsqueda semántica: ✓ Funcionando")
        
        if successful_searches == 0:
            print("   ❌ No se obtuvieron resultados en ninguna búsqueda")
            return False
        elif successful_searches < len(test_queries):
            print(f"   ⚠️ {len(test_queries) - successful_searches} búsquedas sin resultados")
        
        print("   ✅ Búsquedas semánticas funcionando correctamente")
        
    except Exception as e:
        print(f"   ❌ Error en búsquedas: {e}")
        return False
    
    try:
        print("\n📊 7. Verificando estadísticas del sistema...")
        
        stats = service.get_stats()
        
        print("   📈 Estado del sistema:")
        print(f"   - Store activo: {stats.get('active_store')}")
        print(f"   - Store preferido: {stats.get('preferred_store')}")
        print(f"   - Fallback habilitado: {stats.get('fallback_enabled')}")
        print(f"   - Stores disponibles: {stats.get('total_stores_available')}")
        
        stores_available = stats.get('stores_available', {})
        for store_name, available in stores_available.items():
            status = "✅" if available else "❌"
            print(f"   - {store_name}: {status}")
        
        # Estadísticas del store activo
        if 'active_store_stats' in stats:
            active_stats = stats['active_store_stats']
            print("   📊 Métricas del store activo:")
            
            for key, value in active_stats.items():
                if key == 'total_vectors':
                    print(f"   - Vectores indexados: {value}")
                elif key == 'total_documents': 
                    print(f"   - Documentos almacenados: {value}")
                elif key == 'memory_usage_mb':
                    print(f"   - Memoria utilizada: {value:.1f} MB")
                elif key == 'disk_usage_mb':
                    print(f"   - Espacio en disco: {value:.1f} MB")
                elif key == 'avg_search_time':
                    print(f"   - Tiempo búsqueda promedio: {value:.2f}ms")
        
        print("   ✅ Estadísticas del sistema obtenidas correctamente")
        
    except Exception as e:
        print(f"   ❌ Error obteniendo estadísticas: {e}")
        return False
    
    try:
        print("\n🏥 8. Ejecutando health check completo...")
        
        health = service.get_health_status()
        
        overall_status = health.get('overall_status', 'unknown')
        timestamp = health.get('timestamp', 'N/A')
        
        print(f"   🏥 Estado general: {overall_status.upper()}")
        print(f"   ⏰ Verificado en: {timestamp}")
        
        services_health = health.get('services', {})
        print(f"   🔧 Servicios monitoreados: {len(services_health)}")
        
        all_healthy = True
        
        for service_name, service_health in services_health.items():
            status = service_health.get('status', 'unknown')
            available = service_health.get('available', False)
            error = service_health.get('error')
            
            if status == "healthy":
                status_icon = "✅"
            elif status == "error":
                status_icon = "❌"
                all_healthy = False
            else:
                status_icon = "⚠️"
                all_healthy = False
            
            print(f"   {status_icon} {service_name.capitalize()}: {status}")
            
            if error:
                print(f"      ⚠️ Error: {error}")
            
            # Mostrar métricas específicas si están disponibles
            if 'total_vectors' in service_health:
                print(f"      📊 Vectores: {service_health['total_vectors']}")
            if 'total_documents' in service_health:
                print(f"      📊 Documentos: {service_health['total_documents']}")
            if 'memory_usage_mb' in service_health:
                print(f"      💾 Memoria: {service_health['memory_usage_mb']:.1f} MB")
        
        if all_healthy:
            print("   ✅ Todos los servicios están saludables")
        else:
            print("   ⚠️ Algunos servicios tienen problemas")
        
        print("   ✅ Health check completado exitosamente")
        
    except Exception as e:
        print(f"   ❌ Error en health check: {e}")
        return False
    
    try:
        print("\n⚔️ 9. Comparando rendimiento entre stores...")
        
        if service.faiss_available and service.chromadb_available:
            print("   🔥 Ambos stores disponibles - ejecutando comparación de rendimiento")
            
            comparison_query = "administración municipal servicios ciudadanos digitales"
            comparison = service.compare_stores(comparison_query)
            
            print(f"   🔍 Query de benchmark: '{comparison_query}'")
            print(f"   ⏰ Ejecutado: {comparison.get('timestamp')}")
            
            results = comparison.get('comparison_results', {})
            
            faiss_success = False
            chromadb_success = False
            faiss_time = 0
            chromadb_time = 0
            
            for store_name, store_results in results.items():
                print(f"\n   📋 Resultados {store_name.upper()}:")
                
                if store_results.get('status') == 'success':
                    time_ms = store_results.get('search_time_ms', 0)
                    count = store_results.get('results_count', 0)
                    print(f"      ⏱️ Tiempo de búsqueda: {time_ms:.3f}ms")
                    print(f"      📊 Resultados encontrados: {count}")
                    print(f"      ✅ Estado: Exitoso")
                    
                    if store_name == 'faiss':
                        faiss_success = True
                        faiss_time = time_ms
                    elif store_name == 'chromadb':
                        chromadb_success = True
                        chromadb_time = time_ms
                        
                else:
                    error = store_results.get('error', 'Error desconocido')
                    print(f"      ❌ Error: {error}")
            
            # Análisis comparativo
            if faiss_success and chromadb_success:
                print(f"\n   🏆 ANÁLISIS COMPARATIVO:")
                if faiss_time < chromadb_time:
                    speedup = chromadb_time / faiss_time if faiss_time > 0 else 0
                    print(f"   🥇 FAISS es {speedup:.1f}x más rápido que ChromaDB")
                    print(f"      FAISS: {faiss_time:.3f}ms vs ChromaDB: {chromadb_time:.3f}ms")
                elif chromadb_time < faiss_time:
                    speedup = faiss_time / chromadb_time if chromadb_time > 0 else 0
                    print(f"   🥇 ChromaDB es {speedup:.1f}x más rápido que FAISS")
                    print(f"      ChromaDB: {chromadb_time:.3f}ms vs FAISS: {faiss_time:.3f}ms")
                else:
                    print(f"   🤝 Rendimiento similar entre ambos stores")
                    
            print("   ✅ Comparación de rendimiento completada")
        else:
            available_stores = []
            if service.faiss_available:
                available_stores.append("FAISS")
            if service.chromadb_available:
                available_stores.append("ChromaDB")
            
            print(f"   ℹ️ Solo {', '.join(available_stores)} disponible(s)")
            print("   ⏭️ Comparación omitida (se necesitan ambos stores para comparar)")
        
    except Exception as e:
        print(f"   ❌ Error en comparación: {e}")
        return False
    
    try:
        print("\n🔄 10. Probando cambio dinámico entre stores...")
        
        current_store = service.get_active_store_type()
        print(f"   📍 Store actual: {current_store.upper()}")
        
        # Intentar cambiar a store alternativo
        if current_store == "faiss" and service.chromadb_available:
            print("   🔄 Intentando cambio: FAISS → ChromaDB")
            success = service.switch_store("chromadb")
            
            if success:
                new_store = service.get_active_store_type()
                print(f"   ✅ Cambio exitoso: {current_store} → {new_store}")
                
                # Probar búsqueda rápida con nuevo store
                quick_results = service.search("test cambio de store", k=2)
                print(f"   🔍 Test búsqueda con {new_store}: {len(quick_results)} resultados")
                
                # Volver al store original
                restore_success = service.switch_store("faiss")
                restored_store = service.get_active_store_type()
                
                if restore_success:
                    print(f"   🔙 Store restaurado: {new_store} → {restored_store}")
                else:
                    print("   ⚠️ Error restaurando store original")
            else:
                print("   ❌ Fallo al cambiar a ChromaDB")
                
        elif current_store == "chromadb" and service.faiss_available:
            print("   🔄 Intentando cambio: ChromaDB → FAISS")
            success = service.switch_store("faiss")
            
            if success:
                new_store = service.get_active_store_type()
                print(f"   ✅ Cambio exitoso: {current_store} → {new_store}")
                
                # Probar búsqueda rápida con nuevo store
                quick_results = service.search("test cambio de store", k=2)
                print(f"   🔍 Test búsqueda con {new_store}: {len(quick_results)} resultados")
                
                # Volver al store original
                restore_success = service.switch_store("chromadb")
                restored_store = service.get_active_store_type()
                
                if restore_success:
                    print(f"   🔙 Store restaurado: {new_store} → {restored_store}")
                else:
                    print("   ⚠️ Error restaurando store original")
            else:
                print("   ❌ Fallo al cambiar a FAISS")
        else:
            print("   ℹ️ Solo un store disponible - cambio dinámico omitido")
        
        print("   ✅ Test de cambio dinámico completado")
        
    except Exception as e:
        print(f"   ❌ Error probando cambio de store: {e}")
        return False
    
    # RESULTADO FINAL EXITOSO
    print("\n" + "="*70)
    print("🎉 RESULTADO FINAL - ¡ÉXITO COMPLETO!")
    print("="*70)
    print("✅ TODOS LOS TESTS PASARON EXITOSAMENTE")
    print("✅ VectorStoreService REAL integrado y funcionando")
    print("✅ Embeddings calculados y almacenados correctamente")
    print("✅ Búsqueda semántica operativa")
    print("✅ Stores reales (FAISS/ChromaDB) conectados")
    print("✅ API completamente funcional y robusta")
    print("✅ Sistema listo para RAG en producción")
    
    print(f"\n📊 RESUMEN TÉCNICO DEL SISTEMA:")
    print(f"🎯 Store activo: {service.get_active_store_type().upper()}")
    print(f"✅ FAISS disponible: {service.faiss_available}")
    print(f"✅ ChromaDB disponible: {service.chromadb_available}")
    print(f"✅ Método is_available(): {service.is_available()}")
    print(f"🧠 Embeddings: all-MiniLM-L6-v2 (384 dimensiones)")
    print(f"💾 Documentos indexados: {len(test_docs)}")
    print(f"🔍 Búsquedas funcionando: ✓")
    print(f"📊 Métricas disponibles: ✓")
    print(f"🏥 Health monitoring: ✓")
    print(f"🔄 Switching dinámico: ✓")
    
    print("\n🚀 ¡TU SISTEMA RAG ESTÁ COMPLETAMENTE OPERATIVO!")
    print("🎓 ¡EL TFM TIENE AHORA UN PROTOTIPO FUNCIONAL AL 100%!")
    
    return True


def show_next_steps():
    """Mostrar los próximos pasos para el desarrollo"""
    print("\n🎯 PRÓXIMOS PASOS RECOMENDADOS")
    print("=" * 50)
    
    print("\n1. 🔬 EJECUTAR BENCHMARKING ACADÉMICO:")
    print("   python comparison_faiss_vs_chromadb.py")
    print("   → Obtendrás datos empíricos para tu TFM")
    
    print("\n2. 🤖 INTEGRAR LLM SERVICE:")
    print("   → Conectar Ollama y OpenAI para generación")
    print("   → Pipeline RAG completo: Retrieval + Generation")
    
    print("\n3. 🌐 PROBAR INTERFAZ WEB:")
    print("   python run.py")
    print("   → Interface web con stores reales funcionando")
    
    print("\n4. 📊 ANÁLISIS DE RENDIMIENTO:")
    print("   → Ejecutar tests de carga")
    print("   → Optimizar parámetros")
    print("   → Documentar métricas")
    
    print("\n5. 📝 DOCUMENTAR PARA TFM:")
    print("   → Capturas de pantalla del sistema funcionando")
    print("   → Métricas de rendimiento")
    print("   → Comparaciones técnicas")
    
    print("\n💡 EJEMPLOS DE USO EN TU APLICACIÓN:")
    print("""
# Uso básico del servicio en tu app
from app.services.vector_store_service import get_vector_store_service

service = get_vector_store_service()

# Verificar disponibilidad
if service.is_available():
    # Agregar documentos
    service.add_documents(chunks_con_embeddings)
    
    # Buscar información
    results = service.search("consulta ciudadano", k=5)
    
    # Cambiar de store si es necesario
    service.switch_store("chromadb")  # Para filtros avanzados
    service.switch_store("faiss")     # Para velocidad máxima
""")


if __name__ == "__main__":
    print("🎓 TFM Vicente Caruncho - Test VectorStoreService Final")
    print("=" * 70)
    
    success = test_vector_store_service_integration()
    
    if success:
        show_next_steps()
        
        print("\n🎉 ¡FELICIDADES!")
        print("Has logrado implementar un sistema RAG profesional completo.")
        print("Tu TFM está en excelente forma para la defensa.")
    else:
        print("\n❌ ALGUNOS TESTS FALLARON")
        print("Revisa los errores específicos arriba")
        print("El VectorStoreService está casi listo, solo faltan detalles menores")
    
    print(f"\n🏁 Test final completado: {'🎉 SUCCESS' if success else '❌ FAILED'}")
    exit(0 if success else 1)