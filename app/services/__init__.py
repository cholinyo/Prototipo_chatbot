"""
Paquete de servicios del sistema RAG
TFM Vicente Caruncho - Exportaciones principales
"""

# Importaciones principales de LLM
try:
    from .llm.llm_services import LLMService
except ImportError:
    LLMService = None

# Importaciones de servicios RAG principales
try:
    from .embeddings import embedding_service
except ImportError:
    embedding_service = None

try:
    from .faiss_store import faiss_store
except ImportError:
    faiss_store = None

try:
    from .chromadb_store import chromadb_store
except ImportError:
    chromadb_store = None

# Función search_documents que estaba faltando
def search_documents(query: str, k: int = 5, **kwargs):
    """
    Buscar documentos usando el servicio RAG disponible
    """
    try:
        # Intentar usar FAISS primero
        if faiss_store and faiss_store.is_available():
            return faiss_store.search(query, k=k)
        
        # Fallback a ChromaDB
        if chromadb_store and chromadb_store.is_available():
            return chromadb_store.search(query, k=k)
        
        # Sin vector stores disponibles
        return []
        
    except Exception as e:
        print(f"Error en search_documents: {e}")
        return []

# Exportaciones principales
__all__ = [
    'LLMService',
    'search_documents',
    'embedding_service',
    'faiss_store', 
    'chromadb_store'
]