#!/usr/bin/env python3
"""
Test Simplificado - Comparación Vector Stores
Bypass de problemas de compatibilidad con API directa
"""

import sys
import time
import numpy as np
from pathlib import Path

# Añadir el directorio raíz al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_direct_comparison():
    """Test directo usando APIs básicas"""
    print("🚀 TEST SIMPLIFICADO - VECTOR STORES COMPARISON")
    print("=" * 60)
    
    try:
        # Importar embeddings directamente
        from app.services.rag.embeddings import encode_text
        print("✅ EmbeddingService importado")
        
        # Crear datos de prueba
        documents = [
            "Procedimiento para solicitar licencias de obras menores",
            "Requisitos para el empadronamiento municipal", 
            "Horarios de atención al ciudadano",
            "Normativa sobre ruidos y actividades públicas",
            "Proceso de tramitación de ayudas sociales",
            "Información sobre impuestos locales",
            "Servicios de biblioteca municipal",
            "Gestión de residuos urbanos",
            "Procedimientos de participación ciudadana",
            "Información sobre transporte público"
        ]
        
        queries = [
            "¿Cómo solicitar una licencia de obra?",
            "¿Cuáles son los horarios de atención?", 
            "¿Dónde puedo pagar los impuestos?",
            "¿Cómo denunciar ruidos molestos?",
            "¿Qué ayudas sociales están disponibles?"
        ]
        
        print(f"📋 Dataset: {len(documents)} documentos, {len(queries)} queries")
        
        # Test FAISS directo
        print("\n🔍 EVALUANDO FAISS (API directa)...")
        try:
            import faiss
            
            # Generar embeddings
            doc_embeddings = []
            for doc in documents:
                emb = encode_text(doc)
                doc_embeddings.append(emb)
            
            embeddings_matrix = np.array(doc_embeddings).astype('float32')
            
            # Crear índice FAISS
            dimension = embeddings_matrix.shape[1]
            index = faiss.IndexFlatL2(dimension)
            
            # Test inserción
            start_time = time.time()
            index.add(embeddings_matrix)
            insertion_time = time.time() - start_time
            insertion_throughput = len(documents) / insertion_time
            
            print(f"   ✅ Inserción: {insertion_time:.3f}s ({insertion_throughput:.1f} docs/s)")
            
            # Test búsqueda
            search_times = []
            for query in queries:
                query_emb = encode_text(query).reshape(1, -1).astype('float32')
                
                start_time = time.time()
                distances, indices = index.search(query_emb, k=3)
                search_time = time.time() - start_time
                search_times.append(search_time)
            
            avg_search_time = np.mean(search_times)
            queries_per_second = 1 / avg_search_time if avg_search_time > 0 else 0
            
            print(f"   ✅ Búsqueda: {avg_search_time:.4f}s promedio ({queries_per_second:.1f} queries/s)")
            
            faiss_results = {
                "insertion_time": insertion_time,
                "insertion_throughput": insertion_throughput,
                "avg_search_time": avg_search_time,
                "queries_per_second": queries_per_second,
                "total_vectors": index.ntotal
            }
            
        except Exception as e:
            print(f"   ❌ Error FAISS: {e}")
            faiss_results = None
        
        # Test ChromaDB directo
        print("\n🔍 EVALUANDO CHROMADB (API directa)...")
        try:
            import chromadb
            
            # Crear cliente temporal
            client = chromadb.Client()
            collection = client.create_collection(name="test_collection")
            
            # Preparar datos
            ids = [f"doc_{i}" for i in range(len(documents))]
            embeddings = [encode_text(doc).tolist() for doc in documents]
            metadatas = [{"source": f"doc_{i}", "type": "admin"} for i in range(len(documents))]
            
            # Test inserción
            start_time = time.time()
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            insertion_time = time.time() - start_time
            insertion_throughput = len(documents) / insertion_time
            
            print(f"   ✅ Inserción: {insertion_time:.3f}s ({insertion_throughput:.1f} docs/s)")
            
            # Test búsqueda
            search_times = []
            for query in queries:
                query_emb = encode_text(query).tolist()
                
                start_time = time.time()
                results = collection.query(
                    query_embeddings=[query_emb],
                    n_results=3
                )
                search_time = time.time() - start_time
                search_times.append(search_time)
            
            avg_search_time = np.mean(search_times)
            queries_per_second = 1 / avg_search_time if avg_search_time > 0 else 0
            
            print(f"   ✅ Búsqueda: {avg_search_time:.4f}s promedio ({queries_per_second:.1f} queries/s)")
            
            chromadb_results = {
                "insertion_time": insertion_time,
                "insertion_throughput": insertion_throughput,
                "avg_search_time": avg_search_time,
                "queries_per_second": queries_per_second,
                "total_documents": collection.count()
            }
            
        except Exception as e:
            print(f"   ❌ Error ChromaDB: {e}")
            chromadb_results = None
        
        # Comparación y análisis
        if faiss_results and chromadb_results:
            print("\n📊 ANÁLISIS COMPARATIVO:")
            
            # Inserción
            faiss_insert = faiss_results["insertion_throughput"]
            chroma_insert = chromadb_results["insertion_throughput"]
            insert_winner = "FAISS" if faiss_insert > chroma_insert else "ChromaDB"
            insert_factor = max(faiss_insert, chroma_insert) / min(faiss_insert, chroma_insert)
            
            print(f"   📥 Inserción:")
            print(f"      • FAISS: {faiss_insert:.1f} docs/s")
            print(f"      • ChromaDB: {chroma_insert:.1f} docs/s")
            print(f"      • Ganador: {insert_winner} ({insert_factor:.1f}x mejor)")
            
            # Búsqueda
            faiss_search = faiss_results["avg_search_time"]
            chroma_search = chromadb_results["avg_search_time"]
            search_winner = "FAISS" if faiss_search < chroma_search else "ChromaDB"
            search_factor = max(faiss_search, chroma_search) / min(faiss_search, chroma_search)
            
            print(f"   🔍 Búsqueda:")
            print(f"      • FAISS: {faiss_search:.4f}s promedio")
            print(f"      • ChromaDB: {chroma_search:.4f}s promedio")
            print(f"      • Ganador: {search_winner} ({search_factor:.1f}x más rápido)")
            
            # Recomendaciones
            print(f"\n🎯 RECOMENDACIONES:")
            if insert_winner == "FAISS":
                print(f"   • FAISS superior para carga masiva de documentos")
            else:
                print(f"   • ChromaDB superior para carga masiva de documentos")
                
            if search_winner == "FAISS":
                print(f"   • FAISS superior para búsquedas de alta frecuencia")
            else:
                print(f"   • ChromaDB superior para búsquedas de alta frecuencia")
            
            print(f"   • FAISS: Ideal para máximo rendimiento")
            print(f"   • ChromaDB: Ideal para facilidad de uso y metadatos")
            print(f"   • Ambos viables para administraciones locales")
            
            # Generar reporte simple
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_file = f"docs/resultados_pruebas/vector_stores_simple_{timestamp}.txt"
            
            Path("docs/resultados_pruebas").mkdir(parents=True, exist_ok=True)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("COMPARACIÓN VECTOR STORES - TFM VICENTE CARUNCHO\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Dataset: {len(documents)} documentos, {len(queries)} queries\n\n")
                
                f.write("RESULTADOS FAISS:\n")
                f.write(f"- Inserción: {faiss_insert:.1f} docs/s\n")
                f.write(f"- Búsqueda: {faiss_search:.4f}s promedio\n")
                f.write(f"- Queries/segundo: {faiss_results['queries_per_second']:.1f}\n\n")
                
                f.write("RESULTADOS CHROMADB:\n")
                f.write(f"- Inserción: {chroma_insert:.1f} docs/s\n")
                f.write(f"- Búsqueda: {chroma_search:.4f}s promedio\n")
                f.write(f"- Queries/segundo: {chromadb_results['queries_per_second']:.1f}\n\n")
                
                f.write("CONCLUSIONES:\n")
                f.write(f"- Inserción: {insert_winner} ganador ({insert_factor:.1f}x mejor)\n")
                f.write(f"- Búsqueda: {search_winner} ganador ({search_factor:.1f}x mejor)\n")
                f.write("- Ambas tecnologías viables para administraciones locales\n")
            
            print(f"\n📄 Reporte generado: {report_file}")
            print("\n🎉 ¡COMPARACIÓN COMPLETADA EXITOSAMENTE!")
            print("✅ Datos listos para inclusión en TFM")
            
        else:
            print("\n❌ No se pudieron completar todas las evaluaciones")
            if not faiss_results:
                print("   • FAISS falló")
            if not chromadb_results:
                print("   • ChromaDB falló")
        
    except Exception as e:
        print(f"❌ Error general: {e}")

if __name__ == "__main__":
    test_direct_comparison()