#!/usr/bin/env python3
"""
Script de Testing para Pipeline RAG
TFM Vicente Caruncho - Sistemas Inteligentes

Usa el pipeline RAG integrado en tu aplicación Flask
Sin duplicar funcionalidad - todo centralizado

Uso:
    python scripts/test_rag_pipeline.py --status
    python scripts/test_rag_pipeline.py --ingest data/documentos
    python scripts/test_rag_pipeline.py --search "licencias municipales"
    python scripts/test_rag_pipeline.py --benchmark
"""

import argparse
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Agregar el directorio del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def get_rag_pipeline():
    """Obtener el pipeline RAG de tu aplicación"""
    try:
        from app.services.rag.pipeline import get_rag_pipeline
        return get_rag_pipeline()
    except ImportError as e:
        print(f"❌ Error importando pipeline RAG: {e}")
        print("💡 Asegúrate de actualizar app/services/rag/pipeline.py")
        return None

def show_pipeline_status():
    """Mostrar estado completo del pipeline"""
    print("📊 ESTADO DEL PIPELINE RAG")
    print("=" * 50)
    
    pipeline = get_rag_pipeline()
    if not pipeline:
        return False
    
    try:
        # Health check completo
        health = pipeline.health_check()
        
        print(f"🟢 Estado general: {health['status'].upper()}")
        print(f"⏰ Timestamp: {datetime.fromtimestamp(health['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📈 Disponibilidad: {health['availability_rate']:.1%}")
        
        print(f"\n🔧 COMPONENTES:")
        for component, status in health['components'].items():
            icon = "✅" if status == "available" else "❌"
            print(f"   {icon} {component.title()}: {status}")
        
        # Estadísticas detalladas si están disponibles
        if hasattr(pipeline, 'get_stats'):
            stats = pipeline.get_stats()
            
            if 'services' in stats:
                print(f"\n📊 ESTADÍSTICAS DETALLADAS:")
                
                # Embeddings
                if 'embeddings' in stats['services']:
                    embed_stats = stats['services']['embeddings']
                    print(f"   🧠 Embeddings:")
                    print(f"      Modelo: {embed_stats['model_info']['model_name']}")
                    print(f"      Dimensión: {embed_stats['model_info']['dimension']}")
                    print(f"      Requests: {embed_stats['metrics']['total_requests']}")
                    print(f"      Cache hit rate: {embed_stats['metrics']['cache_hit_rate']:.1%}")
                
                # FAISS
                if 'faiss' in stats['services']:
                    faiss_stats = stats['services']['faiss']
                    print(f"   🔢 FAISS:")
                    print(f"      Vectores: {faiss_stats.get('total_vectors', 0)}")
                    print(f"      Tipo índice: {faiss_stats.get('index_type', 'unknown')}")
                    print(f"      Memoria: {faiss_stats.get('index_size_mb', 0):.1f} MB")
                
                # ChromaDB
                if 'chromadb' in stats['services']:
                    chromadb_stats = stats['services']['chromadb']
                    print(f"   🗄️ ChromaDB:")
                    print(f"      Documentos: {chromadb_stats.get('total_documents', 0)}")
                    print(f"      Colección: {chromadb_stats.get('collection_name', 'unknown')}")
                    print(f"      Disco: {chromadb_stats.get('storage', {}).get('disk_usage_mb', 0):.1f} MB")
        
        return health['status'] in ['healthy', 'partial']
        
    except Exception as e:
        print(f"❌ Error obteniendo estado: {e}")
        return False

def ingest_documents(directory: str):
    """Ingestar documentos usando el pipeline"""
    print(f"📥 INGESTA DE DOCUMENTOS")
    print(f"📂 Directorio: {directory}")
    print("=" * 50)
    
    pipeline = get_rag_pipeline()
    if not pipeline:
        return False
    
    if not os.path.exists(directory):
        print(f"❌ Directorio no encontrado: {directory}")
        return False
    
    try:
        print("🚀 Iniciando ingesta...")
        start_time = time.time()
        
        result = pipeline.ingest_directory(directory)
        
        if result['success']:
            print(f"\n✅ INGESTA COMPLETADA:")
            print(f"   📄 Archivos procesados: {result['files_processed']}")
            print(f"   📦 Chunks generados: {result['chunks_generated']}")
            print(f"   🔢 FAISS indexados: {result['faiss_indexed']}")
            print(f"   🗄️ ChromaDB indexados: {result['chromadb_indexed']}")
            print(f"   ⏱️ Tiempo: {result['processing_time']:.2f}s")
            
            if result['chunks_generated'] > 0:
                throughput = result['chunks_generated'] / result['processing_time']
                print(f"   ⚡ Throughput: {throughput:.1f} chunks/s")
        else:
            print(f"❌ INGESTA FALLÓ")
        
        if result['errors']:
            print(f"\n⚠️ ERRORES ({len(result['errors'])}):")
            for error in result['errors']:
                print(f"   • {error}")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ Error en ingesta: {e}")
        return False

def search_documents(query: str, k: int = 5):
    """Buscar documentos usando el pipeline"""
    print(f"🔍 BÚSQUEDA SEMÁNTICA")
    print(f"🔎 Query: '{query}'")
    print(f"🎯 Resultados: {k}")
    print("=" * 50)
    
    pipeline = get_rag_pipeline()
    if not pipeline:
        return False
    
    try:
        print("🧠 Buscando...")
        start_time = time.time()
        
        results = pipeline.search_documents(query, k=k)
        search_time = time.time() - start_time
        
        print(f"⏱️ Tiempo de búsqueda: {search_time:.3f}s")
        print(f"📊 Resultados encontrados: {len(results)}")
        
        if results:
            print(f"\n📄 RESULTADOS:")
            for i, doc in enumerate(results, 1):
                print(f"\n{i}. {Path(doc['source']).name}")
                if 'score' in doc:
                    print(f"   Score: {doc['score']:.3f}")
                print(f"   Vector Store: {doc.get('vector_store', 'unknown')}")
                print(f"   Contenido: {doc['content'][:100]}...")
        else:
            print("❌ No se encontraron resultados")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Error en búsqueda: {e}")
        return False

def generate_rag_response(query: str, k: int = 5):
    """Generar respuesta RAG completa"""
    print(f"🤖 RESPUESTA RAG COMPLETA")
    print(f"❓ Pregunta: '{query}'")
    print("=" * 50)
    
    pipeline = get_rag_pipeline()
    if not pipeline:
        return False
    
    try:
        print("🔄 Generando respuesta...")
        
        response = pipeline.generate_rag_response(query, k=k)
        
        if response['success']:
            print(f"\n✅ RESPUESTA GENERADA:")
            print(f"   ⏱️ Tiempo: {response['response_time']:.2f}s")
            print(f"   🧠 Modelo: {response['llm_model']}")
            print(f"   📚 Chunks utilizados: {response['context_chunks']}")
            
            print(f"\n💬 RESPUESTA:")
            print(f"   {response['response']}")
            
            print(f"\n📖 FUENTES:")
            for source in response['sources']:
                print(f"   [{source['index']}] {Path(source['source']).name}")
                if source.get('score'):
                    print(f"       Score: {source['score']:.3f}")
        else:
            print(f"❌ Error generando respuesta: {response.get('error', 'Unknown')}")
        
        return response['success']
        
    except Exception as e:
        print(f"❌ Error en respuesta RAG: {e}")
        return False

def run_benchmark():
    """Ejecutar benchmark del pipeline"""
    print("🏁 BENCHMARK DEL PIPELINE RAG")
    print("=" * 50)
    
    pipeline = get_rag_pipeline()
    if not pipeline:
        return False
    
    # Consultas de prueba
    test_queries = [
        "licencias municipales",
        "ordenanza ruidos",
        "servicios sociales",
        "tramitación administrativa",
        "ayudas económicas"
    ]
    
    print(f"⚡ Ejecutando {len(test_queries)} consultas de prueba...")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "test_queries": test_queries,
        "search_times": [],
        "rag_times": [],
        "success_count": 0
    }
    
    try:
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔎 Test {i}: '{query}'")
            
            # Test búsqueda
            start_time = time.time()
            search_results = pipeline.search_documents(query, k=3)
            search_time = time.time() - start_time
            results["search_times"].append(search_time)
            
            print(f"   🔍 Búsqueda: {search_time:.3f}s ({len(search_results)} resultados)")
            
            # Test respuesta RAG completa (si LLM disponible)
            if hasattr(pipeline, 'generate_rag_response'):
                try:
                    start_time = time.time()
                    rag_response = pipeline.generate_rag_response(query, k=3)
                    rag_time = time.time() - start_time
                    results["rag_times"].append(rag_time)
                    
                    if rag_response['success']:
                        results["success_count"] += 1
                        print(f"   🤖 RAG: {rag_time:.3f}s ✅")
                    else:
                        print(f"   🤖 RAG: {rag_time:.3f}s ❌")
                        
                except Exception as e:
                    print(f"   🤖 RAG: Error - {e}")
        
        # Calcular estadísticas
        if results["search_times"]:
            avg_search = sum(results["search_times"]) / len(results["search_times"])
            print(f"\n📊 ESTADÍSTICAS DE BÚSQUEDA:")
            print(f"   Promedio: {avg_search:.3f}s")
            print(f"   Mínimo: {min(results['search_times']):.3f}s")
            print(f"   Máximo: {max(results['search_times']):.3f}s")
        
        if results["rag_times"]:
            avg_rag = sum(results["rag_times"]) / len(results["rag_times"])
            print(f"\n📊 ESTADÍSTICAS RAG:")
            print(f"   Promedio: {avg_rag:.3f}s")
            print(f"   Éxito: {results['success_count']}/{len(test_queries)}")
        
        # Guardar resultados
        results_file = project_root / "data" / "pipeline_benchmark.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Resultados guardados: {results_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en benchmark: {e}")
        return False

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Test Pipeline RAG - TFM Vicente Caruncho",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/test_rag_pipeline.py --status
  python scripts/test_rag_pipeline.py --ingest data/documentos
  python scripts/test_rag_pipeline.py --search "licencias municipales"
  python scripts/test_rag_pipeline.py --rag "¿Cómo tramitar una licencia?"
  python scripts/test_rag_pipeline.py --benchmark
        """
    )
    
    parser.add_argument('--status', action='store_true',
                       help='Mostrar estado del pipeline')
    parser.add_argument('--ingest', type=str,
                       help='Ingestar documentos desde directorio')
    parser.add_argument('--search', type=str,
                       help='Buscar documentos por consulta')
    parser.add_argument('--rag', type=str,
                       help='Generar respuesta RAG completa')
    parser.add_argument('--benchmark', action='store_true',
                       help='Ejecutar benchmark de rendimiento')
    parser.add_argument('--k', type=int, default=5,
                       help='Número de resultados (default: 5)')
    
    args = parser.parse_args()
    
    print("🤖 TESTING PIPELINE RAG")
    print("🎓 TFM Vicente Caruncho - Sistemas Inteligentes")
    print("🔗 Usa el pipeline integrado en tu aplicación Flask")
    print("=" * 60)
    
    success = True
    
    if args.status:
        success = show_pipeline_status()
    
    elif args.ingest:
        success = ingest_documents(args.ingest)
    
    elif args.search:
        success = search_documents(args.search, k=args.k)
    
    elif args.rag:
        success = generate_rag_response(args.rag, k=args.k)
    
    elif args.benchmark:
        success = run_benchmark()
    
    else:
        # Si no se especifica acción, mostrar estado
        print("ℹ️ No se especificó acción, mostrando estado...")
        success = show_pipeline_status()
    
    return 0 if success else 1

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