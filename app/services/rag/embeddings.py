"""
Servicio de Embeddings para el sistema RAG
TFM Vicente Caruncho - Sistemas Inteligentes
"""

import numpy as np
from typing import List, Optional, Dict, Any, Union
import time
from pathlib import Path

# Imports principales
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers no disponible. Instalar con: pip install sentence-transformers")

from app.core.config import get_embedding_config
from app.core.logger import get_logger

class EmbeddingCache:
    """Cache simple para embeddings"""
    
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
        self.current_time = 0
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Obtener embedding del cache"""
        if text in self.cache:
            self.access_times[text] = self.current_time
            self.current_time += 1
            return self.cache[text].copy()
        return None
    
    def put(self, text: str, embedding: np.ndarray):
        """Guardar embedding en cache"""
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[text] = embedding.copy()
        self.access_times[text] = self.current_time
        self.current_time += 1
    
    def _evict_oldest(self):
        """Eliminar entrada más antigua"""
        if not self.cache:
            return
        
        oldest_text = min(self.access_times.keys(), key=lambda x: self.access_times[x])
        del self.cache[oldest_text]
        del self.access_times[oldest_text]
    
    def get_stats(self) -> Dict[str, Any]:
        """Estadísticas del cache"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': getattr(self, '_hit_rate', 0.0)
        }

class EmbeddingMetrics:
    """Métricas del servicio de embeddings"""
    
    def __init__(self):
        self.total_requests = 0
        self.total_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
    
    def record_request(self, time_taken: float, cache_hit: bool = False, error: bool = False):
        """Registrar una petición"""
        self.total_requests += 1
        self.total_time += time_taken
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        if error:
            self.errors += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas"""
        avg_time = self.total_time / max(self.total_requests, 1)
        cache_hit_rate = self.cache_hits / max(self.total_requests, 1)
        
        return {
            'total_requests': self.total_requests,
            'average_time': avg_time,
            'cache_hit_rate': cache_hit_rate,
            'total_errors': self.errors
        }

class EmbeddingService:
    """Servicio principal de embeddings con compatibilidad completa"""
    
    def __init__(self, model_name: str = None):
        self.logger = get_logger("prototipo_chatbot.embedding_service")
        self.config = get_embedding_config()
        self.model = None
        self.cache = EmbeddingCache(max_size=self.config.cache_size)
        self.metrics = EmbeddingMetrics()
        
        # Usar modelo de configuración o parámetro
        self.model_name = model_name or self.config.embedding_name
        self._load_model()
    
    def _load_model(self):
        """Cargar modelo de sentence transformers"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.error("sentence-transformers no disponible")
            return
        
        try:
            self.logger.info(f"Cargando modelo: {self.model_name}")
            
            self.model = SentenceTransformer(
                self.model_name,
                device=self.config.embedding_device,
                cache_folder=self.config.cache_dir
            )
            
            self.logger.info(
                "Modelo de embeddings inicializado",
                model=self.model_name,
                device=self.config.embedding_device,
                dimension=self.get_dimension()
            )
            
        except Exception as e:
            self.logger.error(f"Error cargando modelo: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Verificar si el servicio está disponible"""
        return SENTENCE_TRANSFORMERS_AVAILABLE and self.model is not None
    
    def get_dimension(self) -> int:
        """Obtener dimensión de los embeddings"""
        if not self.is_available():
            return 0
        return self.model.get_sentence_embedding_dimension()
    
    # =========================================================================
    # MÉTODOS PRINCIPALES - COMPATIBILIDAD TOTAL
    # =========================================================================
    
    def encode(self, sentences: Union[str, List[str]], **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Método principal 'encode' para compatibilidad total con sentence-transformers
        Acepta tanto strings individuales como listas
        """
        if isinstance(sentences, str):
            return self.encode_single_text(sentences)
        elif isinstance(sentences, list):
            return self.encode_batch(sentences)
        else:
            raise ValueError(f"Tipo no soportado: {type(sentences)}")
    
    def encode_single_text(self, text: str) -> Optional[np.ndarray]:
        """Codificar un texto individual"""
        if not self.is_available():
            self.logger.warning("Servicio de embeddings no disponible")
            return None
        
        if not text or not text.strip():
            self.logger.warning("Texto vacío proporcionado")
            return None
        
        # Verificar cache
        cached = self.cache.get(text)
        if cached is not None:
            self.metrics.record_request(0.0, cache_hit=True)
            return cached
        
        try:
            start_time = time.time()
            
            # Generar embedding usando sentence-transformers
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            processing_time = time.time() - start_time
            
            # Normalizar si está configurado
            if self.config.normalize_embeddings:
                embedding = embedding / np.linalg.norm(embedding)
            
            # Guardar en cache
            self.cache.put(text, embedding)
            
            # Registrar métricas
            self.metrics.record_request(processing_time, cache_hit=False)
            
            self.logger.debug(
                "Embedding generado",
                text_length=len(text),
                embedding_shape=embedding.shape,
                processing_time=processing_time
            )
            
            return embedding
            
        except Exception as e:
            self.metrics.record_request(0.0, cache_hit=False, error=True)
            self.logger.error(f"Error generando embedding: {e}")
            return None
    
    def encode_batch(self, texts: List[str], batch_size: int = None) -> List[np.ndarray]:
        """Codificar múltiples textos en batch"""
        if not self.is_available():
            self.logger.warning("Servicio de embeddings no disponible")
            return []
        
        if not texts:
            return []
        
        batch_size = batch_size or self.config.batch_size
        results = []
        
        try:
            # Procesar en batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Verificar cache para textos del batch
                batch_embeddings = []
                texts_to_process = []
                indices_to_process = []
                
                for j, text in enumerate(batch):
                    cached = self.cache.get(text)
                    if cached is not None:
                        batch_embeddings.append((j, cached))
                        self.metrics.record_request(0.0, cache_hit=True)
                    else:
                        texts_to_process.append(text)
                        indices_to_process.append(j)
                
                # Procesar textos no cacheados
                if texts_to_process:
                    start_time = time.time()
                    
                    new_embeddings = self.model.encode(
                        texts_to_process,
                        convert_to_numpy=True,
                        batch_size=min(len(texts_to_process), batch_size)
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Normalizar si está configurado
                    if self.config.normalize_embeddings:
                        new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
                    
                    # Añadir a cache y resultados
                    for idx, text, embedding in zip(indices_to_process, texts_to_process, new_embeddings):
                        self.cache.put(text, embedding)
                        batch_embeddings.append((idx, embedding))
                        self.metrics.record_request(processing_time / len(texts_to_process), cache_hit=False)
                
                # Ordenar y añadir a resultados
                batch_embeddings.sort(key=lambda x: x[0])
                results.extend([emb for _, emb in batch_embeddings])
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error procesando batch: {e}")
            return []
    
    def encode_documents(self, documents: List[str]) -> List[np.ndarray]:
        """Método específico para documentos (alias de encode_batch)"""
        return self.encode_batch(documents)
    
    # =========================================================================
    # INFORMACIÓN Y ESTADÍSTICAS
    # =========================================================================
    
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
            'model_name': self.model_name,
            'dimension': self.get_dimension(),
            'device': self.config.embedding_device,
            'max_seq_length': getattr(self.model, 'max_seq_length', 512),
            'normalize_embeddings': self.config.normalize_embeddings
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas completas del servicio"""
        return {
            'model_info': self.get_model_info(),
            'cache_stats': self.cache.get_stats(),
            'metrics': self.metrics.get_stats()
        }

# ============================================================================
# INSTANCIA GLOBAL Y FUNCIONES DE CONVENIENCIA
# ============================================================================

# Instancia global del servicio
embedding_service = EmbeddingService()

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
