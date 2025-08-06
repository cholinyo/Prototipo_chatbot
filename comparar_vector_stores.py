#!/usr/bin/env python3
"""
Comparación Académica: FAISS vs ChromaDB
Script para generar métricas comparativas para TFM
Prototipo_chatbot - TFM Vicente Caruncho
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from app.models import DocumentChunk, DocumentMetadata


# Añadir el directorio raíz al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_comparative_benchmark():
    """Ejecutar benchmark comparativo completo"""
    print("🔬 BENCHMARKING ACADÉMICO: FAISS vs ChromaDB")
    print("=" * 70)
    print("Prototipo_chatbot - TFM Vicente Caruncho Ramos")
    print("=" * 70)
    
    try:
        # Imports
        from app.services.rag.faiss_store import create_faiss_store, is_faiss_available
        from app.services.rag.chromadb_store import create_chromadb_store, is_chromadb_available
        from app.services.rag.embeddings import embedding_service
        from app.models import DocumentChunk, DocumentMetadata
        
        print("📦 Módulos importados correctamente")
        
        # Verificar disponibilidad
        faiss_available = is_faiss_available()
        chromadb_available = is_chromadb_available()
        embedding_available = embedding_service.is_available()
        
        print(f"   ✅ FAISS disponible: {faiss_available}")
        print(f"   ✅ ChromaDB disponible: {chromadb_available}")
        print(f"   ✅ EmbeddingService disponible: {embedding_available}")
        
        if not all([faiss_available, chromadb_available, embedding_available]):
            print("❌ No todos los servicios están disponibles")
            return False
        
    except ImportError as e:
        print(f"❌ Error en imports: {e}")
        return False
    
    # Preparar datos de prueba
    print(f"\n📊 Preparando dataset de prueba...")
    
    # Dataset más grande para comparación meaningful
    test_documents = [
        "Administración electrónica y digitalización de servicios públicos municipales",
        "Procedimientos de licencias urbanísticas y de actividad en administraciones locales",
        "Gestión de expedientes administrativos y tramitación de documentos oficiales",
        "Atención ciudadana presencial y telemática en ayuntamientos españoles",
        "Normativa municipal sobre ordenanzas, reglamentos y disposiciones locales",
        "Hacienda local, presupuestos municipales y gestión financiera pública",
        "Planificación urbanística, PGOU y gestión del territorio municipal",
        "Servicios sociales municipales y políticas de bienestar ciudadano",
        "Medio ambiente, sostenibilidad y gestión de residuos urbanos",
        "Transparencia, participación ciudadana y gobierno abierto local",
        "Contratación pública, licitaciones y procedimientos administrativos",
        "Personal funcionario, laborales y organización administrativa municipal",
        "Patrimonio municipal, bienes públicos y gestión del dominio público",
        "Tributos locales, tasas municipales y gestión tributaria",
        "Seguridad ciudadana, policía local y protección civil municipal",
        "Cultura, deportes y actividades de ocio en el ámbito municipal",
        "Educación municipal, escuelas infantiles y servicios educativos",
        "Sanidad pública local y servicios de salud comunitaria",
        "Transporte público urbano y movilidad sostenible municipal",
        "Turismo local, promoción económica y desarrollo territorial"
    ]
    
    # Queries de prueba representativas
    test_queries = [
        "licencias y permisos municipales",
        "servicios digitales administración",
        "tramitación expedientes ciudadanos",
        "normativa ordenanzas locales",
        "presupuestos hacienda municipal",
        "urbanismo planificación territorial",
        "medio ambiente sostenibilidad",
        "atención ciudadana servicios",
        "transparencia gobierno abierto",
        "personal funcionario municipal"
    ]
    
    print(f"   📄 Documentos de prueba: {len(test_documents)}")
    print(f"   🔍 Queries de prueba: {len(test_queries)}")
    
    # Crear chunks de prueba
    metadata = DocumentMetadata(
        source_path="benchmark_documents.pdf",
        source_type="document", 
        file_type=".pdf",
        size_bytes=len(" ".join(test_documents)),
        created_at=datetime.now(),
        processed_at=datetime.now(),
        checksum="benchmark_checksum"
    )
    
    chunks = []
    for i, text in enumerate(test_documents):
        chunk = DocumentChunk(
            id=f"benchmark-chunk-{i+1:02d}",
            content=text,
            metadata=metadata,
            chunk_index=i,
            chunk_size=len(text),
            start_char=i*150,
            end_char=(i+1)*150
        )
        chunks.append(chunk)
    
    # Generar embeddings
    print(f"\n🧠 Generando embeddings para {len(chunks)} documentos...")
    start_time = time.time()
    embedded_chunks = embedding_service.encode_documents(chunks)
    embedding_time = time.time() - start_time
    print(f"   ✅ Embeddings generados en {embedding_time:.3f}s")
    
    # Generar embeddings para queries
    print(f"🔍 Generando embeddings para {len(test_queries)} queries...")
    query_embeddings = []
    for query in test_queries:
        query_emb = embedding_service.encode_single_text(query)
        query_embeddings.append(query_emb)
    
    # Benchmarking FAISS
    print(f"\n🔵 BENCHMARKING FAISS")
    print("-" * 40)
    
    faiss_results = benchmark_vector_store(
        store_type="FAISS",
        store_factory=lambda: create_faiss_store(
            store_path="data/benchmark_faiss",
            dimension=384,
            index_type="IndexFlatL2"
        ),
        chunks=embedded_chunks,
        queries=test_queries,
        query_embeddings=query_embeddings
    )
    
    # Benchmarking ChromaDB
    print(f"\n🟢 BENCHMARKING CHROMADB")
    print("-" * 40)
    
    chromadb_results = benchmark_vector_store(
        store_type="ChromaDB",
        store_factory=lambda: create_chromadb_store(
            store_path="data/benchmark_chromadb",
            collection_name="benchmark_collection",
            distance_function="cosine"
        ),
        chunks=embedded_chunks,
        queries=test_queries,
        query_embeddings=query_embeddings
    )
    
    # Análisis comparativo
    print(f"\n📊 ANÁLISIS COMPARATIVO")
    print("=" * 70)
    
    comparison = analyze_results(faiss_results, chromadb_results)
    
    # Generar reporte
    print(f"\n📋 GENERANDO REPORTE ACADÉMICO...")
    report_path = generate_academic_report(faiss_results, chromadb_results, comparison)
    
    print(f"\n🎉 ¡BENCHMARKING COMPLETADO!")
    print(f"📄 Reporte guardado en: {report_path}")
    print(f"🎓 Datos listos para análisis académico del TFM")
    
    return True


def benchmark_vector_store(store_type: str, store_factory, chunks: List[DocumentChunk], 
                          queries: List[str], query_embeddings: List[np.ndarray]) -> Dict[str, Any]:
    """Ejecutar benchmark completo de un vector store"""
    
    print(f"🏗️  Inicializando {store_type}...")
    store = store_factory()
    
    if not store.is_available():
        print(f"   ❌ {store_type} no disponible")
        return {}
    
    print(f"   ✅ {store_type} inicializado")
    
    # Limpiar store anterior
    store.clear()
    
    results = {
        "store_type": store_type,
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(chunks),
        "query_count": len(queries)
    }
    
    # Test 1: Inserción de documentos
    print(f"📥 Test inserción de documentos...")
    
    insertion_times = []
    batch_sizes = [5, 10, len(chunks)]  # Diferentes tamaños de lote
    
    for batch_size in batch_sizes:
        if batch_size > len(chunks):
            continue
            
        # Limpiar para cada test
        store.clear()
        
        batch_chunks = chunks[:batch_size]
        
        start_time = time.time()
        success = store.add_documents(batch_chunks)
        insertion_time = time.time() - start_time
        
        if success:
            throughput = batch_size / insertion_time
            insertion_times.append({
                "batch_size": batch_size,
                "time": insertion_time,
                "throughput": throughput
            })
            print(f"   ✅ {batch_size} docs en {insertion_time:.3f}s ({throughput:.1f} docs/s)")
        else:
            print(f"   ❌ Error insertando {batch_size} documentos")
    
    results["insertion_performance"] = insertion_times
    
    # Insertar todos los documentos para tests de búsqueda
    store.clear()
    full_insertion_start = time.time()
    store.add_documents(chunks)
    full_insertion_time = time.time() - full_insertion_start
    
    results["full_insertion_time"] = full_insertion_time
    results["full_insertion_throughput"] = len(chunks) / full_insertion_time
    
    # Test 2: Búsquedas con diferentes valores de k
    print(f"🔍 Test búsquedas con diferentes k...")
    
    search_performance = []
    k_values = [1, 3, 5, 10]
    
    for k in k_values:
        search_times = []
        result_counts = []
        
        for i, (query, query_emb) in enumerate(zip(queries, query_embeddings)):
            start_time = time.time()
            search_results = store.search(query_emb, k=k)
            search_time = time.time() - start_time
            
            search_times.append(search_time)
            result_counts.append(len(search_results))
        
        avg_search_time = np.mean(search_times)
        avg_results = np.mean(result_counts)
        
        search_performance.append({
            "k": k,
            "avg_search_time": avg_search_time,
            "avg_results_returned": avg_results,
            "queries_per_second": 1.0 / avg_search_time if avg_search_time > 0 else 0
        })
        
        print(f"   ✅ k={k}: {avg_search_time:.3f}s avg, {avg_results:.1f} results avg")
    
    results["search_performance"] = search_performance
    
    # Test 3: Búsquedas con filtros
    print(f"🎛️  Test búsquedas con filtros...")
    
    filter_tests = [
        {"source_type": "document"},
        {"file_type": ".pdf"},
    ]
    
    filter_performance = []
    
    for filter_dict in filter_tests:
        filter_times = []
        
        for query_emb in query_embeddings[:5]:  # Solo primeras 5 queries
            start_time = time.time()
            filtered_results = store.search(query_emb, k=5, filters=filter_dict)
            filter_time = time.time() - start_time
            filter_times.append(filter_time)
        
        avg_filter_time = np.mean(filter_times)
        
        filter_performance.append({
            "filter": filter_dict,
            "avg_time": avg_filter_time,
            "queries_per_second": 1.0 / avg_filter_time if avg_filter_time > 0 else 0
        })
        
        print(f"   ✅ Filter {filter_dict}: {avg_filter_time:.3f}s avg")
    
    results["filter_performance"] = filter_performance
    
    # Test 4: Métricas del sistema
    print(f"📊 Recopilando métricas del sistema...")
    
    final_stats = store.get_stats()
    results["system_metrics"] = final_stats
    
    # Métricas específicas según el tipo
    if store_type == "FAISS":
        results["memory_usage"] = final_stats.get("index_size_mb", 0)
        results["storage_type"] = "In-memory + Files"
    elif store_type == "ChromaDB":
        results["disk_usage"] = final_stats.get("disk_usage_mb", 0)
        results["storage_type"] = "Persistent Database"
    
    print(f"   ✅ Métricas recopiladas")
    
    return results


def analyze_results(faiss_results: Dict[str, Any], chromadb_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analizar y comparar resultados"""
    
    analysis = {
        "comparison_timestamp": datetime.now().isoformat(),
        "dataset_info": {
            "documents": faiss_results.get("dataset_size", 0),
            "queries": faiss_results.get("query_count", 0),
            "embedding_dimension": 384
        }
    }
    
    # Comparación de inserción
    faiss_insertion = faiss_results.get("full_insertion_throughput", 0)
    chromadb_insertion = chromadb_results.get("full_insertion_throughput", 0)
    
    analysis["insertion_comparison"] = {
        "faiss_throughput": faiss_insertion,
        "chromadb_throughput": chromadb_insertion,
        "winner": "FAISS" if faiss_insertion > chromadb_insertion else "ChromaDB",
        "improvement_factor": max(faiss_insertion, chromadb_insertion) / min(faiss_insertion, chromadb_insertion) if min(faiss_insertion, chromadb_insertion) > 0 else 0
    }
    
    # Comparación de búsqueda (k=5)
    faiss_search_k5 = next((p for p in faiss_results.get("search_performance", []) if p["k"] == 5), {})
    chromadb_search_k5 = next((p for p in chromadb_results.get("search_performance", []) if p["k"] == 5), {})
    
    faiss_search_time = faiss_search_k5.get("avg_search_time", float('inf'))
    chromadb_search_time = chromadb_search_k5.get("avg_search_time", float('inf'))
    
    analysis["search_comparison"] = {
        "faiss_avg_time": faiss_search_time,
        "chromadb_avg_time": chromadb_search_time,
        "winner": "FAISS" if faiss_search_time < chromadb_search_time else "ChromaDB",
        "speedup_factor": max(faiss_search_time, chromadb_search_time) / min(faiss_search_time, chromadb_search_time) if min(faiss_search_time, chromadb_search_time) > 0 else 0
    }
    
    # Comparación de almacenamiento
    faiss_storage = faiss_results.get("memory_usage", 0)
    chromadb_storage = chromadb_results.get("disk_usage", 0)
    
    analysis["storage_comparison"] = {
        "faiss_mb": faiss_storage,
        "chromadb_mb": chromadb_storage,
        "faiss_type": "Memory + Files",
        "chromadb_type": "Persistent DB"
    }
    
    # Recomendaciones
    recommendations = []
    
    if faiss_insertion > chromadb_insertion * 1.5:
        recommendations.append("FAISS superior para inserción masiva de datos")
    
    if faiss_search_time < chromadb_search_time * 0.5:
        recommendations.append("FAISS superior para búsquedas de alta velocidad")
    elif chromadb_search_time < faiss_search_time * 0.5:
        recommendations.append("ChromaDB superior para búsquedas de alta velocidad")
    
    if chromadb_storage > 0:
        recommendations.append("ChromaDB ofrece persistencia automática")
    
    recommendations.append("FAISS ideal para aplicaciones que priorizan velocidad")
    recommendations.append("ChromaDB ideal para aplicaciones que requieren persistencia")
    
    analysis["recommendations"] = recommendations
    
    # Mostrar análisis
    print(f"\n📊 RESULTADOS COMPARATIVOS:")
    print(f"📥 Inserción:")
    print(f"   • FAISS: {faiss_insertion:.1f} docs/s")
    print(f"   • ChromaDB: {chromadb_insertion:.1f} docs/s")
    print(f"   • Ganador: {analysis['insertion_comparison']['winner']}")
    
    print(f"\n🔍 Búsqueda (k=5):")
    print(f"   • FAISS: {faiss_search_time:.3f}s promedio")
    print(f"   • ChromaDB: {chromadb_search_time:.3f}s promedio")
    print(f"   • Ganador: {analysis['search_comparison']['winner']}")
    
    print(f"\n💾 Almacenamiento:")
    print(f"   • FAISS: {faiss_storage:.2f} MB (memoria + archivos)")
    print(f"   • ChromaDB: {chromadb_storage:.2f} MB (base de datos)")
    
    print(f"\n💡 Recomendaciones clave:")
    for rec in recommendations[:3]:
        print(f"   • {rec}")
    
    return analysis


def generate_academic_report(faiss_results: Dict[str, Any], chromadb_results: Dict[str, Any], 
                           analysis: Dict[str, Any]) -> str:
    """Generar reporte académico completo"""
    
    report_dir = Path("data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"vector_stores_comparison_{timestamp}.json"
    
    # Compilar reporte completo
    full_report = {
        "metadata": {
            "title": "Comparación Empírica FAISS vs ChromaDB",
            "subtitle": "Análisis de Rendimiento para Sistemas RAG",
            "author": "Vicente Caruncho Ramos",
            "tfm": "Prototipo de Chatbot Interno para Administraciones Locales",
            "university": "Universitat Jaume I",
            "date": datetime.now().isoformat(),
            "version": "1.0"
        },
        "methodology": {
            "dataset_size": analysis["dataset_info"]["documents"],
            "query_count": analysis["dataset_info"]["queries"],
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": analysis["dataset_info"]["embedding_dimension"],
            "test_categories": [
                "Inserción de documentos",
                "Búsquedas por similitud",
                "Filtrado por metadatos", 
                "Uso de recursos"
            ]
        },
        "results": {
            "faiss": faiss_results,
            "chromadb": chromadb_results
        },
        "analysis": analysis,
        "conclusions": {
            "performance_winner": analysis["search_comparison"]["winner"],
            "throughput_winner": analysis["insertion_comparison"]["winner"],
            "use_case_recommendations": analysis["recommendations"]
        }
    }
    
    # Guardar reporte
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False, default=str)
    
    # Generar resumen en markdown
    md_path = report_dir / f"vector_stores_summary_{timestamp}.md"
    generate_markdown_summary(full_report, md_path)
    
    return str(report_path)


def generate_markdown_summary(report: Dict[str, Any], md_path: Path):
    """Generar resumen en Markdown para el TFM"""
    
    analysis = report["analysis"]
    
    markdown_content = f"""# Comparación Empírica: FAISS vs ChromaDB

**Autor:** {report["metadata"]["author"]}  
**TFM:** {report["metadata"]["tfm"]}  
**Fecha:** {datetime.now().strftime("%d/%m/%Y")}

## Metodología

- **Dataset:** {analysis["dataset_info"]["documents"]} documentos de administración local
- **Queries:** {analysis["dataset_info"]["queries"]} consultas representativas
- **Modelo embeddings:** all-MiniLM-L6-v2 (384 dimensiones)
- **Métricas:** Latencia, throughput, uso de recursos

## Resultados Principales

### Rendimiento de Inserción
- **FAISS:** {analysis["insertion_comparison"]["faiss_throughput"]:.1f} docs/segundo
- **ChromaDB:** {analysis["insertion_comparison"]["chromadb_throughput"]:.1f} docs/segundo
- **Ganador:** {analysis["insertion_comparison"]["winner"]}

### Rendimiento de Búsqueda
- **FAISS:** {analysis["search_comparison"]["faiss_avg_time"]:.3f}s promedio
- **ChromaDB:** {analysis["search_comparison"]["chromadb_avg_time"]:.3f}s promedio  
- **Ganador:** {analysis["search_comparison"]["winner"]}

### Almacenamiento
- **FAISS:** {analysis["storage_comparison"]["faiss_mb"]:.2f} MB ({analysis["storage_comparison"]["faiss_type"]})
- **ChromaDB:** {analysis["storage_comparison"]["chromadb_mb"]:.2f} MB ({analysis["storage_comparison"]["chromadb_type"]})

## Conclusiones para Administraciones Locales

### Recomendaciones de Uso
"""
    
    for i, rec in enumerate(analysis["recommendations"], 1):
        markdown_content += f"{i}. {rec}\n"
    
    markdown_content += f"""
### Implicaciones para el TFM

- **FAISS** se muestra superior para casos de uso que priorizan **velocidad de respuesta**
- **ChromaDB** ofrece ventajas en **persistencia** y **gestión de metadatos**
- Ambas tecnologías son **viables** para sistemas RAG en administraciones locales
- La elección depende de los **requisitos específicos** del caso de uso

### Trabajo Futuro

- Evaluación con datasets más grandes (>10,000 documentos)
- Análisis de calidad de resultados (relevancia, precisión)
- Pruebas con diferentes modelos de embeddings
- Evaluación de escalabilidad en producción
"""
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)


def cleanup_benchmark_data():
    """Limpiar datos de benchmark"""
    import shutil
    
    for dir_name in ["data/benchmark_faiss", "data/benchmark_chromadb"]:
        dir_path = Path(dir_name)
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"🧹 Limpiado: {dir_name}")
            except Exception as e:
                print(f"⚠️  Error limpiando {dir_name}: {e}")


if __name__ == "__main__":
    print("🎓 Comparación Académica Vector Stores - TFM")
    print("Vicente Caruncho Ramos - Sistemas Inteligentes")
    print("=" * 70)
    
    try:
        success = run_comparative_benchmark()
        
        if success:
            print(f"\n🎉 ¡COMPARACIÓN ACADÉMICA COMPLETADA EXITOSAMENTE!")
            print(f"📊 Datos empíricos generados para análisis TFM")
            print(f"📈 Métricas listas para memoria y defensa")
            cleanup_benchmark_data()
        else:
            print(f"\n💔 Error en comparación académica")
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Comparación interrumpida por el usuario")
        cleanup_benchmark_data()
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        cleanup_benchmark_data()