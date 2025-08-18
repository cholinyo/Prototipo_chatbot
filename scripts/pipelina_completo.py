#!/usr/bin/env python3
"""
Pipeline Completo RAG - Integración End-to-End
TFM Vicente Caruncho - Sistemas Inteligentes

Integra: Procesamiento → Embeddings → FAISS & ChromaDB → Búsquedas

Uso:
    python scripts/pipeline_completo.py --help
    python scripts/pipeline_completo.py --ingest data/documentos
    python scripts/pipeline_completo.py --search "licencias municipales"
    python scripts/pipeline_completo.py --benchmark
"""

import argparse
import sys
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Agregar el directorio del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_services():
    """Verificar que todos los servicios estén disponibles"""
    print("🔍 VERIFICANDO SERVICIOS RAG...")
    
    services_status = {}
    
    # Test Ingestion Service
    try:
        from app.services.ingestion import ingestion_service
        services_status['ingestion'] = ingestion_service.is_available()
        print(f"   📥 Ingestion Service: {'✅' if services_status['ingestion'] else '❌'}")
    except Exception as e:
        services_status['ingestion'] = False
        print(f"   📥 Ingestion Service: ❌ Error: {e}")
    
    # Test Embedding Service
    try:
        from app.services.rag.embeddings import embedding_service
        services_status['embeddings'] = embedding_service.is_available()
        print(f"   🧠 Embedding Service: {'✅' if services_status['embeddings'] else '❌'}")
        if services_status['embeddings']:
            info = embedding_service.get_model_info()
            print(f"      Modelo: {info['model_name']}")
            print(f"      Dimensión: {info['dimension']}")
    except Exception as e:
        services_status['embeddings'] = False
        print(f"   🧠 Embedding Service: ❌ Error: {e}")
    
    # Test FAISS Store
    try:
        from app.services.rag.faiss_store import faiss_store
        services_status['faiss'] = faiss_store.is_available()
        print(f"   🔢 FAISS Store: {'✅' if services_status['faiss'] else '❌'}")
        if services_status['faiss']:
            stats = faiss_store.get_stats()
            print(f"      Vectores: {stats.get('total_vectors', 0)}")
            print(f"      Tipo índice: {stats.get('index_type', 'unknown')}")
    except Exception as e:
        services_status['faiss'] = False
        print(f"   🔢 FAISS Store: ❌ Error: {e}")
    
    # Test ChromaDB Store
    try:
        from app.services.rag.chromadb_store import chromadb_store
        services_status['chromadb'] = chromadb_store.is_available()
        print(f"   🗄️ ChromaDB Store: {'✅' if services_status['chromadb'] else '❌'}")
        if services_status['chromadb']:
            stats = chromadb_store.get_stats()
            print(f"      Documentos: {stats.get('total_documents', 0)}")
            print(f"      Colección: {stats.get('collection_name', 'unknown')}")
    except Exception as e:
        services_status['chromadb'] = False
        print(f"   🗄️ ChromaDB Store: ❌ Error: {e}")
    
    return services_status

def ingest_documents_complete(directory: str) -> Dict[str, Any]:
    """Pipeline completo de ingesta: Documentos → Embeddings → Vector Stores"""
    print(f"\n🚀 INICIANDO PIPELINE COMPLETO DE INGESTA")
    print(f"📂 Directorio: {directory}")
    print("=" * 60)
    
    # Verificar servicios
    services = test_services()
    if not all(services.values()):
        print("❌ No todos los servicios están disponibles")
        return {"success": False, "error": "Servicios no disponibles"}
    
    start_time = time.time()
    results = {
        "success": False,
        "total_files": 0,
        "processed_files": 0,
        "total_chunks": 0,
        "faiss_indexed": 0,
        "chromadb_indexed": 0,
        "processing_time": 0,
        "errors": []
    }
    
    try:
        # Importar servicios
        from app.services.ingestion import ingestion_service
        from app.services.rag.embeddings import embedding_service
        from app.services.rag.faiss_store import faiss_store
        from app.services.rag.chromadb_store import chromadb_store
        
        # Paso 1: Encontrar archivos
        print("\n📋 PASO 1: Escaneando archivos...")
        supported_extensions = ingestion_service.processor.get_supported_extensions()
        files_found = []
        
        for ext in supported_extensions:
            pattern = f"*{ext}"
            files_found.extend(Path(directory).glob(pattern))
        
        results["total_files"] = len(files_found)
        print(f"   📄 Encontrados: {len(files_found)} archivos")
        
        if not files_found:
            print(f"   ❌ No se encontraron archivos soportados")
            print(f"   💡 Extensiones soportadas: {supported_extensions}")
            return results
        
        # Paso 2: Procesar documentos
        print("\n🔄 PASO 2: Procesando documentos...")
        all_chunks = []
        
        for file_path in files_found:
            print(f"   📝 Procesando: {file_path.name}")
            
            # Procesar con ingestion service
            chunks = ingestion_service.process_file(str(file_path))
            
            if chunks:
                all_chunks.extend(chunks)
                results["processed_files"] += 1
                print(f"      ✅ {len(chunks)} chunks generados")
            else:
                results["errors"].append(f"Error procesando {file_path.name}")
                print(f"      ❌ Error procesando")
        
        results["total_chunks"] = len(all_chunks)
        print(f"\n   📊 Total chunks: {len(all_chunks)}")
        
        if not all_chunks:
            print("   ❌ No se generaron chunks")
            return results
        
        # Paso 3: Generar embeddings
        print("\n🧠 PASO 3: Generando embeddings...")
        embedding_start = time.time()
        
        # Extraer textos
        texts = [chunk.content for chunk in all_chunks]
        
        # Generar embeddings en batch
        embeddings = embedding_service.encode_batch(texts)
        
        if not embeddings:
            print("   ❌ Error generando embeddings")
            results["errors"].append("Error generando embeddings")
            return results
        
        # Asignar embeddings a chunks
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk.embedding = embedding
        
        embedding_time = time.time() - embedding_start
        print(f"   ✅ Embeddings generados en {embedding_time:.2f}s")
        print(f"   📏 Dimensión: {embeddings[0].shape if embeddings else 'N/A'}")
        
        # Paso 4: Indexar en FAISS
        print("\n🔢 PASO 4: Indexando en FAISS...")
        faiss_start = time.time()
        
        faiss_success = faiss_store.add_documents(all_chunks)
        
        if faiss_success:
            results["faiss_indexed"] = len(all_chunks)
            faiss_time = time.time() - faiss_start
            print(f"   ✅ FAISS indexado en {faiss_time:.2f}s")
            
            # Mostrar estadísticas FAISS
            faiss_stats = faiss_store.get_stats()
            print(f"   📊 Total vectores FAISS: {faiss_stats.get('total_vectors', 0)}")
        else:
            results["errors"].append("Error indexando en FAISS")
            print("   ❌ Error indexando en FAISS")
        
        # Paso 5: Indexar en ChromaDB
        print("\n🗄️ PASO 5: Indexando en ChromaDB...")
        chromadb_start = time.time()
        
        chromadb_success = chromadb_store.add_documents(all_chunks)
        
        if chromadb_success:
            results["chromadb_indexed"] = len(all_chunks)
            chromadb_time = time.time() - chromadb_start
            print(f"   ✅ ChromaDB indexado en {chromadb_time:.2f}s")
            
            # Mostrar estadísticas ChromaDB
            chromadb_stats = chromadb_store.get_stats()
            print(f"   📊 Total documentos ChromaDB: {chromadb_stats.get('total_documents', 0)}")
        else:
            results["errors"].append("Error indexando en ChromaDB")
            print("   ❌ Error indexando en ChromaDB")
        
        # Calcular tiempo total
        total_time = time.time() - start_time
        results["processing_time"] = total_time
        results["success"] = faiss_success or chromadb_success
        
        # Resumen final
        print(f"\n📊 RESUMEN DEL PIPELINE:")
        print(f"   ⏱️ Tiempo total: {total_time:.2f}s")
        print(f"   📄 Archivos procesados: {results['processed_files']}/{results['total_files']}")
        print(f"   📦 Chunks generados: {results['total_chunks']}")
        print(f"   🔢 FAISS indexados: {results['faiss_indexed']}")
        print(f"   🗄️ ChromaDB indexados: {results['chromadb_indexed']}")
        print(f"   ⚡ Throughput: {results['total_chunks']/total_time:.1f} chunks/s")
        
        if results["errors"]:
            print(f"\n❌ ERRORES ({len(results['errors'])}):")
            for error in results["errors"]:
                print(f"   • {error}")
        
        return results
        
    except Exception as e:
        results["errors"].append(f"Error general: {str(e)}")
        print(f"\n💥 Error en pipeline: {e}")
        return results

def search_documents(query: str, k: int = 5) -> Dict[str, Any]:
    """Buscar documentos usando ambos vector stores"""
    print(f"\n🔍 BÚSQUEDA SEMÁNTICA")
    print(f"🔎 Query: '{query}'")
    print(f"🎯 Resultados: {k}")
    print("=" * 50)
    
    try:
        from app.services.rag.embeddings import embedding_service
        from app.services.rag.faiss_store import faiss_store
        from app.services.rag.chromadb_store import chromadb_store
        
        # Generar embedding de la query
        print("🧠 Generando embedding de consulta...")
        query_embedding = embedding_service.encode_single_text(query)
        
        if query_embedding is None:
            print("❌ Error generando embedding")
            return {"success": False}
        
        results = {"success": True, "query": query, "results": {}}
        
        # Búsqueda en FAISS
        if faiss_store.is_available():
            print("\n🔢 Búsqueda en FAISS...")
            faiss_start = time.time()
            
            faiss_results = faiss_store.search(query_embedding, k=k)
            faiss_time = time.time() - faiss_start
            
            print(f"   ⏱️ Tiempo: {faiss_time:.3f}s")
            print(f"   📊 Resultados: {len(faiss_results)}")
            
            results["results"]["faiss"] = {
                "time": faiss_time,
                "count": len(faiss_results),
                "results": []
            }
            
            for i, (chunk, score) in enumerate(faiss_results):
                result_info = {
                    "rank": i + 1,
                    "score": score,
                    "source": chunk.metadata.source_path,
                    "content_preview": chunk.content[:100] + "...",
                    "chunk_id": chunk.id
                }
                results["results"]["faiss"]["results"].append(result_info)
                
                print(f"   {i+1}. Score: {score:.3f} | {Path(chunk.metadata.source_path).name}")
                print(f"      {chunk.content[:80]}...")
        
        # Búsqueda en ChromaDB
        if chromadb_store.is_available():
            print("\n🗄️ Búsqueda en ChromaDB...")
            chromadb_start = time.time()
            
            chromadb_results = chromadb_store.search(query_embedding, k=k)
            chromadb_time = time.time() - chromadb_start
            
            print(f"   ⏱️ Tiempo: {chromadb_time:.3f}s")
            print(f"   📊 Resultados: {len(chromadb_results)}")
            
            results["results"]["chromadb"] = {
                "time": chromadb_time,
                "count": len(chromadb_results),
                "results": []
            }
            
            for i, chunk in enumerate(chromadb_results):
                result_info = {
                    "rank": i + 1,
                    "source": chunk.metadata.source_path,
                    "content_preview": chunk.content[:100] + "...",
                    "chunk_id": chunk.id
                }
                results["results"]["chromadb"]["results"].append(result_info)
                
                print(f"   {i+1}. {Path(chunk.metadata.source_path).name}")
                print(f"      {chunk.content[:80]}...")
        
        return results
        
    except Exception as e:
        print(f"❌ Error en búsqueda: {e}")
        return {"success": False, "error": str(e)}

def run_benchmark() -> Dict[str, Any]:
    """Ejecutar benchmark comparativo FAISS vs ChromaDB"""
    print("\n🏁 BENCHMARK COMPARATIVO FAISS vs ChromaDB")
    print("=" * 60)
    
    try:
        from app.services.rag.embeddings import embedding_service
        from app.services.rag.faiss_store import faiss_store
        from app.services.rag.chromadb_store import chromadb_store
        
        # Consultas de prueba
        test_queries = [
            "licencias municipales",
            "ordenanza ruidos",
            "servicios sociales",
            "tramitación administrativa",
            "ayudas económicas"
        ]
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "test_queries": test_queries,
            "faiss": {"available": faiss_store.is_available()},
            "chromadb": {"available": chromadb_store.is_available()}
        }
        
        # Stats iniciales
        if faiss_store.is_available():
            faiss_stats = faiss_store.get_stats()
            results["faiss"]["initial_stats"] = faiss_stats
            print(f"🔢 FAISS: {faiss_stats.get('total_vectors', 0)} vectores")
        
        if chromadb_store.is_available():
            chromadb_stats = chromadb_store.get_stats()
            results["chromadb"]["initial_stats"] = chromadb_stats
            print(f"🗄️ ChromaDB: {chromadb_stats.get('total_documents', 0)} documentos")
        
        # Ejecutar búsquedas
        print(f"\n⚡ Ejecutando {len(test_queries)} búsquedas de prueba...")
        
        faiss_times = []
        chromadb_times = []
        
        for i, query in enumerate(test_queries):
            print(f"\n🔎 Query {i+1}: '{query}'")
            
            # Generar embedding
            query_embedding = embedding_service.encode_single_text(query)
            
            # Test FAISS
            if faiss_store.is_available():
                start_time = time.time()
                faiss_results = faiss_store.search(query_embedding, k=3)
                faiss_time = time.time() - start_time
                faiss_times.append(faiss_time)
                print(f"   🔢 FAISS: {faiss_time:.3f}s ({len(faiss_results)} resultados)")
            
            # Test ChromaDB
            if chromadb_store.is_available():
                start_time = time.time()
                chromadb_results = chromadb_store.search(query_embedding, k=3)
                chromadb_time = time.time() - start_time
                chromadb_times.append(chromadb_time)
                print(f"   🗄️ ChromaDB: {chromadb_time:.3f}s ({len(chromadb_results)} resultados)")
        
        # Calcular métricas
        if faiss_times:
            results["faiss"]["performance"] = {
                "avg_search_time": sum(faiss_times) / len(faiss_times),
                "min_search_time": min(faiss_times),
                "max_search_time": max(faiss_times),
                "total_searches": len(faiss_times)
            }
        
        if chromadb_times:
            results["chromadb"]["performance"] = {
                "avg_search_time": sum(chromadb_times) / len(chromadb_times),
                "min_search_time": min(chromadb_times),
                "max_search_time": max(chromadb_times),
                "total_searches": len(chromadb_times)
            }
        
        # Comparación
        print(f"\n📊 RESULTADOS COMPARATIVOS:")
        
        if faiss_times and chromadb_times:
            faiss_avg = sum(faiss_times) / len(faiss_times)
            chromadb_avg = sum(chromadb_times) / len(chromadb_times)
            
            faster_system = "FAISS" if faiss_avg < chromadb_avg else "ChromaDB"
            speed_factor = max(faiss_avg, chromadb_avg) / min(faiss_avg, chromadb_avg)
            
            print(f"   🔢 FAISS promedio: {faiss_avg:.3f}s")
            print(f"   🗄️ ChromaDB promedio: {chromadb_avg:.3f}s")
            print(f"   🏆 Ganador: {faster_system} ({speed_factor:.1f}x más rápido)")
            
            results["comparison"] = {
                "winner": faster_system,
                "speed_factor": speed_factor,
                "faiss_avg": faiss_avg,
                "chromadb_avg": chromadb_avg
            }
        
        # Guardar resultados
        results_file = project_root / "data" / "benchmark_results.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Resultados guardados en: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error en benchmark: {e}")
        return {"success": False, "error": str(e)}

def show_system_status():
    """Mostrar estado completo del sistema"""
    print("\n📊 ESTADO COMPLETO DEL SISTEMA RAG")
    print("=" * 50)
    
    services = test_services()
    
    if not any(services.values()):
        print("❌ Ningún servicio disponible")
        return False
    
    try:
        # Stats detalladas de cada servicio
        if services.get('embeddings'):
            from app.services.rag.embeddings import embedding_service
            embed_stats = embedding_service.get_stats()
            print(f"\n🧠 EMBEDDING SERVICE:")
            print(f"   Modelo: {embed_stats['model_info']['model_name']}")
            print(f"   Dimensión: {embed_stats['model_info']['dimension']}")
            print(f"   Requests: {embed_stats['metrics']['total_requests']}")
            print(f"   Cache hit rate: {embed_stats['metrics']['cache_hit_rate']:.1%}")
        
        if services.get('faiss'):
            from app.services.rag.faiss_store import faiss_store
            faiss_stats = faiss_store.get_stats()
            print(f"\n🔢 FAISS STORE:")
            print(f"   Vectores: {faiss_stats.get('total_vectors', 0)}")
            print(f"   Tipo índice: {faiss_stats.get('index_type', 'unknown')}")
            print(f"   Memoria: {faiss_stats.get('index_size_mb', 0):.1f} MB")
        
        if services.get('chromadb'):
            from app.services.rag.chromadb_store import chromadb_store
            chromadb_stats = chromadb_store.get_stats()
            print(f"\n🗄️ CHROMADB STORE:")
            print(f"   Documentos: {chromadb_stats.get('total_documents', 0)}")
            print(f"   Colección: {chromadb_stats.get('collection_name', 'unknown')}")
            print(f"   Disco: {chromadb_stats.get('storage', {}).get('disk_usage_mb', 0):.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error obteniendo estadísticas: {e}")
        return False

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Pipeline Completo RAG - TFM Vicente Caruncho",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/pipeline_completo.py --status
  python scripts/pipeline_completo.py --ingest data/documentos
  python scripts/pipeline_completo.py --search "licencias municipales"
  python scripts/pipeline_completo.py --benchmark
        """
    )
    
    parser.add_argument('--ingest', type=str,
                       help='Ejecutar pipeline completo en directorio')
    parser.add_argument('--search', type=str,
                       help='Buscar documentos por consulta')
    parser.add_argument('--k', type=int, default=5,
                       help='Número de resultados en búsqueda')
    parser.add_argument('--benchmark', action='store_true',
                       help='Ejecutar benchmark comparativo')
    parser.add_argument('--status', action='store_true',
                       help='Mostrar estado del sistema')
    
    args = parser.parse_args()
    
    print("🤖 PIPELINE COMPLETO RAG")
    print("🎓 TFM Vicente Caruncho - Sistemas Inteligentes")
    print("🔗 Integración: Documentos → Embeddings → FAISS & ChromaDB")
    print("=" * 70)
    
    if args.status:
        success = show_system_status()
        return 0 if success else 1
    
    if args.ingest:
        if not os.path.exists(args.ingest):
            print(f"❌ Directorio no encontrado: {args.ingest}")
            return 1
        
        results = ingest_documents_complete(args.ingest)
        return 0 if results["success"] else 1
    
    if args.search:
        results = search_documents(args.search, k=args.k)
        return 0 if results["success"] else 1
    
    if args.benchmark:
        results = run_benchmark()
        return 0 if results.get("success", True) else 1
    
    # Si no se especifica acción, mostrar ayuda
    parser.print_help()
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️ Proceso cancelado")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        sys.exit(1)