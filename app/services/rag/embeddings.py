"""
Servicio de Embeddings para Prototipo_chatbot
TFM Vicente Caruncho - Sistemas Inteligentes UJI
"""

import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger("prototipo_chatbot.embeddings")

class EmbeddingService:
    """Servicio de embeddings usando sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dimension = 384  # Dimensión por defecto del modelo
        self.cache_size = 10000  # ← AÑADIDO: cache_size
        self._load_model()
    
    def _load_model(self):
        """Cargar modelo de embeddings"""
        try:
            logger.info(f"Cargando modelo de embeddings: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Verificar dimensión real
            test_embedding = self.model.encode("test", show_progress_bar=False)
            self.dimension = len(test_embedding)
            
            logger.info(f"Modelo cargado correctamente - Dimensión: {self.dimension}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo de embeddings: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Verificar si el servicio está disponible"""
        return self.model is not None
    
    def encode_single_text(self, text: str) -> Optional[np.ndarray]:
        """Generar embedding para un texto"""
        if not self.is_available():
            return None
        
        try:
            embedding = self.model.encode(text, show_progress_bar=False)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error generando embedding: {e}")
            return None
    
    def encode_texts(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generar embeddings para múltiples textos"""
        if not self.is_available():
            return None
        
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Error generando embeddings: {e}")
            return None
    
    def warm_up(self):
        """Precalentar el modelo"""
        if self.is_available():
            self.encode_single_text("Warming up the embedding model.")
            logger.info("Modelo de embeddings precalentado")
    
    def get_stats(self) -> dict:
        """Obtener estadísticas del servicio"""
        return {
            'available': self.is_available(),
            'model_name': self.model_name,
            'dimension': self.dimension,
            'cache_size': self.cache_size
        }

# Instancia global del servicio
embedding_service = EmbeddingService()

def get_embedding_service() -> EmbeddingService:
    """Obtener instancia del servicio de embeddings"""
    return embedding_service

def encode_batch(self, texts: List[str], batch_size: int = 32) -> Optional[List[np.ndarray]]:
    """Generar embeddings en lotes - alias para encode_texts"""
    embeddings = self.encode_texts(texts)
    if embeddings is not None:
        return [emb for emb in embeddings]  # Convertir a lista de arrays
    return None