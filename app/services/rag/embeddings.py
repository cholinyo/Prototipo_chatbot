"""
Servicio de Embeddings para RAG Pipeline
Prototipo_chatbot - TFM Vicente Caruncho
"""

import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from functools import lru_cache
import hashlib

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

from app.core.config import get_model_config
from app.core.logger import get_logger

class EmbeddingCache:
    """Cache LRU para embeddings"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _hash_text(self, text: str) -> str:
        """Generar hash único para texto"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Obtener embedding del cache"""
        key = self._hash_text(text)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, text: str, embedding: np.ndarray):
        """Guardar embedding en cache"""
        if len(self.cache) >= self.max_size:
            # Eliminar el más antiguo (simple FIFO)
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        
        key = self._hash_text(text)
        self.cache[key] = embedding
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }

class EmbeddingMetrics:
    """Métricas del servicio de embeddings"""
    
    def __init__(self):
        self.total_texts_processed = 0
        self.total_batches_processed = 0
        self.total_processing_time = 0.0
        self.avg_batch_size = 0
        self.last_batch_time = 0.0
    
    def record_batch(self, batch_size: int, processing_time: float):
        """Registrar procesamiento de batch"""
        self.total_texts_processed += batch_size
        self.total_batches_processed += 1
        self.total_processing_time += processing_time
        self.last_batch_time = processing_time
        
        # Actualizar media móvil del tamaño de batch
        alpha = 0.1  # Factor de suavizado
        self.avg_batch_size = (1 - alpha) * self.avg_batch_size + alpha * batch_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas"""
        avg_time_per_text = (
            self.total_processing_time / self.total_texts_processed 
            if self.total_texts_processed > 0 else 0
        )
        
        return {
            'total_texts_processed': self.total_texts_processed,
            'total_batches_processed': self.total_batches_processed,
            'total_processing_time': self.total_processing_time,
            'avg_batch_size': self.avg_batch_size,
            'avg_time_per_text': avg_time_per_text,
            'last_batch_time': self.last_batch_time
        }

class EmbeddingService:
    """Servicio principal de embeddings"""
    
    def __init__(self):
        self.config = get_model_config()
        self.logger = get_logger("prototipo_chatbot.embedding_service")
        self.model = None
        self.cache = EmbeddingCache(max_size=1000)
        self.metrics = EmbeddingMetrics()
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializar modelo de embeddings"""
        if not HAS_SENTENCE_TRANSFORMERS:
            self.logger.error("sentence-transformers no instalado")
            return
        
        try:
            model_name = self.config.embedding_name
            cache_dir = self.config.embedding_cache_dir
            device = self.config.embedding_device
            
            # Crear directorio de cache si no existe
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            
            # Cargar modelo
            self.model = SentenceTransformer(
                model_name,
                cache_folder=cache_dir,
                device=device
            )
            
            # Obtener dimensión del modelo
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            self.dimension = len(test_embedding)
            
            self.logger.info(
                "Modelo de embeddings inicializado",
                model=model_name,
                device=device,
                dimension=self.dimension
            )
            
        except Exception as e:
            self.logger.error(f"Error inicializando modelo: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Verificar disponibilidad del servicio"""
        return self.model is not None
    
    def get_dimension(self) -> int:
        """Obtener dimensión de los embeddings"""
        return getattr(self, 'dimension', self.config.embedding_dimension)
    
    def encode_single_text(self, text: str, use_cache: bool = True) -> Optional[np.ndarray]:
        """Codificar un texto individual"""
        if not self.is_available():
            self.logger.warning("Servicio no disponible")
            return None
        
        try:
            # Verificar cache
            if use_cache:
                cached = self.cache.get(text)
                if cached is not None:
                    return cached
            
            # Generar embedding
            start_time = time.time()
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            processing_time = time.time() - start_time
            
            # Actualizar métricas
            self.metrics.record_batch(1, processing_time)
            
            # Guardar en cache
            if use_cache:
                self.cache.put(text, embedding)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error codificando texto: {e}")
            return None
    
    def encode_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[np.ndarray]:
        """Codificar batch de textos"""
        if not self.is_available():
            self.logger.warning("Servicio no disponible")
            return []
        
        try:
            start_time = time.time()
            
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress
            )
            
            processing_time = time.time() - start_time
            
            # Actualizar métricas
            self.metrics.record_batch(len(texts), processing_time)
            
            self.logger.debug(
                f"Batch procesado: {len(texts)} textos en {processing_time:.2f}s"
            )
            
            return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
            
        except Exception as e:
            self.logger.error(f"Error procesando batch: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo"""
        if not self.is_available():
            return {
                'available': False,
                'model_name': 'not_loaded',
                'dimension': 0,
                'device': 'none'
            }
        
        return {
            'available': True,
            'model_name': self.config.embedding_name,
            'dimension': self.get_dimension(),
            'device': self.config.embedding_device,
            'max_seq_length': getattr(self.model, 'max_seq_length', 512)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas completas del servicio"""
        return {
            'model_info': self.get_model_info(),
            'cache_stats': self.cache.get_stats(),
            'metrics': self.metrics.get_stats()
        }

# Instancia global del servicio
embedding_service = EmbeddingService()

# Funciones de conveniencia para compatibilidad
def encode_text(text: str) -> Optional[np.ndarray]:
    """Función de conveniencia para codificar texto"""
    return embedding_service.encode_single_text(text)

def encode_texts(texts: List[str]) -> List[np.ndarray]:
    """Función de conveniencia para codificar múltiples textos"""
    return embedding_service.encode_batch(texts)

def is_embedding_service_available() -> bool:
    """Verificar si el servicio está disponible"""
    return embedding_service.is_available()

# Exportar todo lo necesario
__all__ = [
    'EmbeddingService',
    'EmbeddingCache', 
    'EmbeddingMetrics',
    'embedding_service',
    'encode_text',
    'encode_texts',
    'is_embedding_service_available'
]