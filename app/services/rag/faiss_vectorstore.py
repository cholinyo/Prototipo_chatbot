"""
FAISS Vector Store Implementation
"""

import numpy as np
from typing import List, Optional, Dict, Any
import pickle
from pathlib import Path

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from app.core.config import get_vector_store_config
from app.core.logger import get_logger
from app.models import DocumentChunk

class FaissVectorStore:
    """Implementación de vector store usando FAISS"""
    
    def __init__(self):
        self.logger = get_logger("prototipo_chatbot.faiss_vectorstore")
        self.config = get_vector_store_config()
        self.index = None
        self.documents = []
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        self._initialize()
    
    def _initialize(self):
        """Inicializar índice FAISS"""
        if not HAS_FAISS:
            self.logger.error("FAISS no instalado")
            return
        
        try:
            # Crear índice L2
            self.index = faiss.IndexFlatL2(self.dimension)
            self.logger.info(
                "Nuevo índice FAISS creado",
                dimension=self.dimension,
                index_type="IndexFlatL2"
            )
        except Exception as e:
            self.logger.error(f"Error inicializando FAISS: {e}")
    
    def is_available(self) -> bool:
        """Verificar disponibilidad"""
        return HAS_FAISS and self.index is not None
    
    def add(self, chunk: DocumentChunk) -> bool:
        """Añadir documento al índice"""
        if not self.is_available():
            return False
        
        try:
            if chunk.embedding is None:
                return False
            
            # Convertir a numpy array
            embedding = np.array(chunk.embedding, dtype=np.float32)
            embedding = embedding.reshape(1, -1)
            
            # Añadir al índice
            self.index.add(embedding)
            self.documents.append(chunk)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error añadiendo al índice: {e}")
            return False
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: float = 0.0
    ) -> List[DocumentChunk]:
        """Buscar documentos similares"""
        if not self.is_available() or len(self.documents) == 0:
            self.logger.warning("Índice FAISS vacío")
            return []
        
        try:
            # Preparar embedding
            query_embedding = np.array(query_embedding, dtype=np.float32)
            query_embedding = query_embedding.reshape(1, -1)
            
            # Buscar
            distances, indices = self.index.search(query_embedding, min(k, len(self.documents)))
            
            # Crear resultados
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.documents):
                    chunk = self.documents[idx]
                    chunk.relevance_score = float(1.0 / (1.0 + dist))
                    if chunk.relevance_score >= threshold:
                        results.append(chunk)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error en búsqueda: {e}")
            return []
    
    def get_document_count(self) -> int:
        """Obtener número de documentos"""
        return len(self.documents)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas"""
        return {
            'type': 'faiss',
            'available': self.is_available(),
            'documents': len(self.documents),
            'dimension': self.dimension
        }

__all__ = ['FaissVectorStore']
