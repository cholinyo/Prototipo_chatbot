"""
Servicio RAG (Retrieval-Augmented Generation)
"""

from typing import List, Optional, Dict, Any
import time
from pathlib import Path

from app.core.config import get_rag_config, get_vector_store_config
from app.core.logger import get_logger
from app.models import DocumentChunk, DocumentMetadata

class RAGService:
    """Servicio principal de RAG"""
    
    def __init__(self):
        self.logger = get_logger("prototipo_chatbot.rag_service")
        self.config = get_rag_config()
        self.vector_store = None
        self.embedding_service = None
        self._initialize()
    
    def _initialize(self):
        """Inicializar componentes del servicio"""
        try:
            # Intentar importar embedding service
            from app.services.rag.embeddings import embedding_service
            self.embedding_service = embedding_service
            
            # Intentar importar vector store
            from app.services.rag.faiss_vectorstore import FaissVectorStore
            self.vector_store = FaissVectorStore()
            
            self.logger.info(
                "RAG Service inicializado",
                embedding_available=self.embedding_service.is_available() if self.embedding_service else False,
                vector_store="faiss",
                enabled=self.config.enabled
            )
        except ImportError as e:
            self.logger.warning(f"Componentes RAG no disponibles: {e}")
    
    def is_available(self) -> bool:
        """Verificar disponibilidad del servicio"""
        return (
            self.config.enabled and 
            self.embedding_service is not None and
            self.vector_store is not None
        )
    
    def search(
        self, 
        query: str, 
        k: int = 5,
        threshold: float = 0.0
    ) -> List[DocumentChunk]:
        """Buscar documentos relevantes"""
        if not self.is_available():
            self.logger.warning("RAG Service no disponible")
            return []
        
        try:
            start_time = time.time()
            
            # Generar embedding de la consulta
            query_embedding = self.embedding_service.encode_single_text(query)
            
            if query_embedding is None:
                return []
            
            # Buscar en vector store
            results = self.vector_store.search(
                query_embedding, 
                k=k,
                threshold=threshold
            )
            
            search_time = time.time() - start_time
            
            self.logger.info(
                "Búsqueda RAG completada",
                query_length=len(query),
                results_found=len(results),
                results_filtered=len([r for r in results if r.relevance_score >= threshold]),
                k_requested=k,
                search_time=search_time
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error en búsqueda RAG: {e}")
            return []
    
    def add_document(self, chunk: DocumentChunk) -> bool:
        """Añadir documento al índice"""
        if not self.is_available():
            return False
        
        try:
            # Generar embedding si no existe
            if chunk.embedding is None:
                embedding = self.embedding_service.encode_single_text(chunk.content)
                if embedding is None:
                    return False
                chunk.embedding = embedding.tolist()
            
            # Añadir al vector store
            return self.vector_store.add(chunk)
            
        except Exception as e:
            self.logger.error(f"Error añadiendo documento: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del servicio"""
        stats = {
            'service_available': self.is_available(),
            'rag_enabled': self.config.enabled,
            'total_documents': 0,
            'embedding_service': 'not_available',
            'vector_store': 'not_available'
        }
        
        if self.embedding_service:
            stats['embedding_service'] = self.embedding_service.get_model_info()
        
        if self.vector_store:
            stats['total_documents'] = self.vector_store.get_document_count()
            stats['vector_store'] = self.vector_store.get_stats()
        
        return stats

# Instancia global del servicio
rag_service = RAGService()

# Exportar
__all__ = ['RAGService', 'rag_service']
