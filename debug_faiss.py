#!/usr/bin/env python3
"""
Debug script para FAISS
"""
import sys
from pathlib import Path

print("🔧 DEBUG FAISS - Paso a paso")
print("=" * 50)

try:
    print("1. Verificando ubicación...")
    print(f"   Directorio actual: {Path.cwd()}")
    print(f"   Python path: {sys.executable}")
    
    print("\n2. Verificando dependencias básicas...")
    import numpy as np
    print(f"   ✅ numpy: {np.__version__}")
    
    try:
        import faiss
        print(f"   ✅ faiss: disponible")
    except ImportError as e:
        print(f"   ❌ faiss: {e}")
        print("   💡 Instalar: pip install faiss-cpu")
        sys.exit(1)
    
    print("\n3. Verificando estructura proyecto...")
    required_files = [
        "app/__init__.py",
        "app/services/__init__.py",
        "app/services/rag/__init__.py",
        "app/models/__init__.py"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
    
    print("\n4. Probando imports del proyecto...")
    try:
        sys.path.insert(0, str(Path.cwd()))
        from app.models import DocumentChunk, DocumentMetadata
        print("   ✅ app.models")
    except Exception as e:
        print(f"   ❌ app.models: {e}")
    
    try:
        from app.services.rag.embeddings import embedding_service
        print("   ✅ app.services.rag.embeddings")
        print(f"   ✅ EmbeddingService disponible: {embedding_service.is_available()}")
    except Exception as e:
        print(f"   ❌ app.services.rag.embeddings: {e}")
    
    print("\n5. Probando FAISS básico...")
    try:
        # Test FAISS básico
        import faiss
        index = faiss.IndexFlatL2(384)
        print(f"   ✅ Índice FAISS creado: {index.ntotal} vectores")
        
        # Test con vector dummy
        import numpy as np
        dummy_vector = np.random.random((1, 384)).astype('float32')
        index.add(dummy_vector)
        print(f"   ✅ Vector añadido: {index.ntotal} vectores")
        
        # Test búsqueda
        distances, indices = index.search(dummy_vector, 1)
        print(f"   ✅ Búsqueda OK: distancia={distances[0][0]:.6f}")
        
    except Exception as e:
        print(f"   ❌ FAISS básico: {e}")
    
    print("\n6. Probando importar FaissVectorStore...")
    try:
        from app.services.rag.faiss_store import FaissVectorStore
        print("   ✅ FaissVectorStore importado")
        
        # Crear instancia básica
        store = FaissVectorStore(store_path="data/debug_faiss")
        print(f"   ✅ Instancia creada: disponible={store.is_available()}")
        
        if store.is_available():
            stats = store.get_stats()
            print(f"   ✅ Stats obtenidas: {stats.get('total_vectors', 0)} vectores")
        
    except Exception as e:
        print(f"   ❌ FaissVectorStore: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Debug completado sin errores críticos")
    
except Exception as e:
    print(f"\n💥 Error crítico: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)