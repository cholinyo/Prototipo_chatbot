"""
Servicios del sistema RAG - Con Compatibilidad
Adaptado para usar estructura existente del repositorio
"""

# Servicios core existentes
try:
    from .rag.embeddings import embedding_service
except ImportError:
    embedding_service = None

try:
    from .rag.faiss_store import FaissVectorStore
except ImportError:
    FaissVectorStore = None

try:
    from .rag.chromadb_store import ChromaDBVectorStore
except ImportError:
    ChromaDBVectorStore = None

# Servicios con compatibilidad
try:
    from .ingestion.data_ingestion import DataIngestionService
except ImportError:
    try:
        from .compat.data_ingestion import DataIngestionService
    except ImportError:
        DataIngestionService = None

try:
    from .llm.llm_service import LLMService
except ImportError:
    try:
        from .compat.llm_service import LLMService
    except ImportError:
        LLMService = None

__all__ = [
    "embedding_service",
    "FaissVectorStore", 
    "ChromaDBVectorStore",
    "DataIngestionService",
    "LLMService"
]

def check_services_availability():
    """Verificar servicios disponibles"""
    return {
        "embedding_service": embedding_service is not None,
        "FaissVectorStore": FaissVectorStore is not None,
        "ChromaDBVectorStore": ChromaDBVectorStore is not None,
        "DataIngestionService": DataIngestionService is not None,
        "LLMService": LLMService is not None
    }
