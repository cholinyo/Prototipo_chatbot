"""
Servicio de Embeddings para Prototipo_chatbot
Maneja la generación de embeddings semánticos usando sentence-transformers
TFM Vicente Caruncho - Sistemas Inteligentes
"""

import os
import time
import pickle
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Imports para embeddings
try:
    from sentence_transformers import SentenceTransformer
    import torch
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# Imports locales
from app.core.config import get_model_config
from app.core.logger import get_logger
from app.models import DocumentChunk


@dataclass
class EmbeddingMetrics:
    """Métricas de rendimiento del servicio de embeddings"""
    total_texts_processed: int = 0
    total_processing_time: float = 0.0
    average_time_per_text: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    model_name: str = ""
    dimension: int = 0
    device: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "total_texts_processed": self.total_texts_processed,
            "total_processing_time": self.total_processing_time,
            "average_time_per_text": self.average_time_per_text,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self.device
        }


class EmbeddingCache:
    """Cache de embeddings para mejorar rendimiento"""
    
    def __init__(self, cache_dir: str, max_size_mb: int = 500):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.logger = get_logger("embedding_cache")
        
        # Archivo de índice para el cache
        self.index_file = self.cache_dir / "cache_index.pkl"
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Cargar índice del cache"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Error cargando índice cache: {e}")
        return {}
    
    def _save_index(self):
        """Guardar índice del cache"""
        try:
            with open(self.index_file, 'wb') as f:
                pickle.dump(self.index, f)
        except Exception as e:
            self.logger.error(f"Error guardando índice cache: {e}")
    
    def _get_text_hash(self, text: str, model_name: str) -> str:
        """Generar hash único para el texto y modelo"""
        combined = f"{model_name}:{text}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Obtener embedding del cache"""
        text_hash = self._get_text_hash(text, model_name)
        
        if text_hash in self.index:
            cache_file = self.cache_dir / f"{text_hash}.npy"
            if cache_file.exists():
                try:
                    embedding = np.load(cache_file)
                    # Actualizar último acceso
                    self.index[text_hash]["last_access"] = time.time()
                    return embedding
                except Exception as e:
                    self.logger.warning(f"Error cargando cache {text_hash}: {e}")
                    # Limpiar entrada corrupta
                    self._remove_entry(text_hash)
        
        return None
    
    def set(self, text: str, model_name: str, embedding: np.ndarray):
        """Guardar embedding en cache"""
        text_hash = self._get_text_hash(text, model_name)
        cache_file = self.cache_dir / f"{text_hash}.npy"
        
        try:
            # Guardar embedding
            np.save(cache_file, embedding)
            
            # Actualizar índice
            self.index[text_hash] = {
                "text_length": len(text),
                "model_name": model_name,
                "created": time.time(),
                "last_access": time.time(),
                "file_size": cache_file.stat().st_size
            }
            
            # Verificar límite de tamaño
            self._cleanup_if_needed()
            
            # Guardar índice
            self._save_index()
            
        except Exception as e:
            self.logger.error(f"Error guardando en cache {text_hash}: {e}")
    
    def _remove_entry(self, text_hash: str):
        """Eliminar entrada del cache"""
        cache_file = self.cache_dir / f"{text_hash}.npy"
        if cache_file.exists():
            cache_file.unlink()
        if text_hash in self.index:
            del self.index[text_hash]
    
    def _cleanup_if_needed(self):
        """Limpiar cache si excede el tamaño máximo"""
        total_size = self.get_cache_size()
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        if total_size > max_size_bytes:
            # Ordenar por último acceso (LRU)
            sorted_entries = sorted(
                self.index.items(),
                key=lambda x: x[1]["last_access"]
            )
            
            # Eliminar entradas más antiguas hasta estar bajo el límite
            for text_hash, info in sorted_entries:
                if total_size <= max_size_bytes * 0.8:  # Dejar 20% de margen
                    break
                
                self._remove_entry(text_hash)
                total_size -= info.get("file_size", 0)
                
                self.logger.debug(f"Cache cleanup: eliminado {text_hash}")
    
    def get_cache_size(self) -> int:
        """Obtener tamaño total del cache en bytes"""
        return sum(info.get("file_size", 0) for info in self.index.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache"""
        return {
            "total_entries": len(self.index),
            "total_size_mb": self.get_cache_size() / (1024 * 1024),
            "max_size_mb": self.max_size_mb,
            "cache_dir": str(self.cache_dir)
        }
    
    def clear(self):
        """Limpiar todo el cache"""
        for cache_file in self.cache_dir.glob("*.npy"):
            cache_file.unlink()
        self.index.clear()
        self._save_index()
        self.logger.info("Cache de embeddings limpiado")


class EmbeddingService:
    """Servicio principal para generar embeddings semánticos"""
    
    def __init__(self):
        self.config = get_model_config()
        self.logger = get_logger("embedding_service")
        
        # Inicializar componentes
        self.model = None
        self.cache = None
        self.metrics = EmbeddingMetrics()
        
        # Estado del servicio
        self.is_initialized = False
        
        # Inicializar servicio
        self._initialize()
    
    def _initialize(self):
        """Inicializar el servicio completo"""
        if not HAS_SENTENCE_TRANSFORMERS:
            self.logger.error(
                "sentence-transformers no está instalado. "
                "Instala con: pip install sentence-transformers"
            )
            return
        
        try:
            # Inicializar modelo
            self._initialize_model()
            
            # Inicializar cache
            self._initialize_cache()
            
            # Marcar como inicializado
            self.is_initialized = True
            
            self.logger.info(
                "EmbeddingService inicializado correctamente",
                model=self.config.embedding_name,
                device=self.config.embedding_device,
                dimension=self.config.embedding_dimension,
                cache_enabled=self.cache is not None
            )
            
        except Exception as e:
            self.logger.error(f"Error inicializando EmbeddingService: {e}")
            self.is_initialized = False
    
    def _initialize_model(self):
        """Inicializar modelo de sentence-transformers"""
        model_name = self.config.embedding_name
        cache_dir = self.config.embedding_cache_dir
        device = self.config.embedding_device
        
        # Crear directorio de cache del modelo
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Verificar dispositivo
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.logger.info(f"Cargando modelo {model_name} en {device}...")
        
        # Cargar modelo
        self.model = SentenceTransformer(
            model_name,
            cache_folder=cache_dir,
            device=device
        )
        
        # Actualizar métricas
        self.metrics.model_name = model_name
        self.metrics.dimension = self.model.get_sentence_embedding_dimension()
        self.metrics.device = device
        
        self.logger.info(
            "Modelo cargado correctamente",
            dimension=self.metrics.dimension,
            device=device
        )
    
    def _initialize_cache(self):
        """Inicializar cache de embeddings"""
        cache_dir = os.path.join(self.config.embedding_cache_dir, "embeddings_cache")
        
        try:
            self.cache = EmbeddingCache(cache_dir, max_size_mb=500)
            self.logger.info(f"Cache de embeddings inicializado en {cache_dir}")
        except Exception as e:
            self.logger.warning(f"No se pudo inicializar cache: {e}")
            self.cache = None
    
    def is_available(self) -> bool:
        """Verificar si el servicio está disponible"""
        return self.is_initialized and self.model is not None
    
    def encode_single_text(self, text: str) -> np.ndarray:
        """Generar embedding para un solo texto"""
        if not self.is_available():
            raise RuntimeError("EmbeddingService no está disponible")
        
        if not text or not text.strip():
            raise ValueError("El texto no puede estar vacío")
        
        text = text.strip()
        
        # Intentar obtener del cache
        if self.cache:
            cached_embedding = self.cache.get(text, self.config.embedding_name)
            if cached_embedding is not None:
                self.metrics.cache_hits += 1
                self._update_cache_hit_rate()
                return cached_embedding
            else:
                self.metrics.cache_misses += 1
        
        # Generar embedding
        start_time = time.time()
        
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            processing_time = time.time() - start_time
            
            # Actualizar métricas
            self.metrics.total_texts_processed += 1
            self.metrics.total_processing_time += processing_time
            self.metrics.average_time_per_text = (
                self.metrics.total_processing_time / self.metrics.total_texts_processed
            )
            self._update_cache_hit_rate()
            
            # Guardar en cache
            if self.cache:
                self.cache.set(text, self.config.embedding_name, embedding)
            
            self.logger.debug(
                "Embedding generado",
                text_length=len(text),
                processing_time=processing_time,
                dimension=len(embedding)
            )
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generando embedding: {e}")
            raise
    
    def encode_batch(self, texts: List[str], batch_size: int = None) -> List[np.ndarray]:
        """Generar embeddings para múltiples textos en lotes"""
        if not self.is_available():
            raise RuntimeError("EmbeddingService no está disponible")
        
        if not texts:
            return []
        
        # Limpiar textos
        clean_texts = [text.strip() for text in texts if text and text.strip()]
        
        if not clean_texts:
            return []
        
        batch_size = batch_size or getattr(self.config, 'embedding_batch_size', 32)
        
        self.logger.info(
            f"Procesando {len(clean_texts)} textos en lotes de {batch_size}"
        )
        
        all_embeddings = []
        
        # Procesar en lotes
        for i in range(0, len(clean_texts), batch_size):
            batch_texts = clean_texts[i:i + batch_size]
            
            # Verificar cache para el lote
            batch_embeddings = []
            texts_to_process = []
            cache_indices = []
            
            for j, text in enumerate(batch_texts):
                if self.cache:
                    cached_embedding = self.cache.get(text, self.config.embedding_name)
                    if cached_embedding is not None:
                        batch_embeddings.append(cached_embedding)
                        self.metrics.cache_hits += 1
                        continue
                    else:
                        self.metrics.cache_misses += 1
                
                texts_to_process.append(text)
                cache_indices.append(len(batch_embeddings))
                batch_embeddings.append(None)  # Placeholder
            
            # Procesar textos que no están en cache
            if texts_to_process:
                start_time = time.time()
                
                try:
                    new_embeddings = self.model.encode(
                        texts_to_process,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=len(texts_to_process) > 10,
                        batch_size=min(batch_size, len(texts_to_process))
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Actualizar métricas
                    self.metrics.total_texts_processed += len(texts_to_process)
                    self.metrics.total_processing_time += processing_time
                    self.metrics.average_time_per_text = (
                        self.metrics.total_processing_time / 
                        self.metrics.total_texts_processed
                    )
                    
                    # Insertar embeddings en las posiciones correctas
                    for k, embedding in enumerate(new_embeddings):
                        idx = cache_indices[k]
                        batch_embeddings[idx] = embedding
                        
                        # Guardar en cache
                        if self.cache and k < len(texts_to_process):
                            self.cache.set(
                                texts_to_process[k], 
                                self.config.embedding_name, 
                                embedding
                            )
                    
                    self.logger.debug(
                        f"Lote procesado: {len(texts_to_process)} textos "
                        f"en {processing_time:.2f}s"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error procesando lote: {e}")
                    # Rellenar con embeddings vacíos para mantener consistencia
                    for idx in cache_indices:
                        if batch_embeddings[idx] is None:
                            batch_embeddings[idx] = np.zeros(self.metrics.dimension)
            
            all_embeddings.extend(batch_embeddings)
        
        self._update_cache_hit_rate()
        
        self.logger.info(
            f"Batch completado: {len(clean_texts)} embeddings generados"
        )
        
        return all_embeddings
    
    def encode_documents(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generar embeddings para chunks de documentos"""
        if not chunks:
            return []
        
        self.logger.info(f"Generando embeddings para {len(chunks)} chunks")
        
        # Extraer textos
        texts = [chunk.content for chunk in chunks]
        
        # Generar embeddings
        embeddings = self.encode_batch(texts)
        
        # Asignar embeddings a chunks
        updated_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            # Crear copia del chunk con embedding
            chunk_dict = chunk.to_dict()
            chunk_dict['embedding'] = embedding
            updated_chunk = DocumentChunk.from_dict(chunk_dict)
            updated_chunks.append(updated_chunk)
        
        self.logger.info(f"Embeddings asignados a {len(updated_chunks)} chunks")
        
        return updated_chunks
    
    def _update_cache_hit_rate(self):
        """Actualizar tasa de aciertos del cache"""
        total_requests = self.metrics.cache_hits + self.metrics.cache_misses
        if total_requests > 0:
            self.metrics.cache_hit_rate = self.metrics.cache_hits / total_requests
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo"""
        if not self.is_available():
            return {"available": False}
        
        return {
            "available": True,
            "model_name": self.config.embedding_name,
            "dimension": self.metrics.dimension,
            "device": self.metrics.device,
            "max_sequence_length": getattr(self.model, 'max_seq_length', 'unknown')
        }
    
    def get_metrics(self) -> EmbeddingMetrics:
        """Obtener métricas del servicio"""
        return self.metrics
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas completas"""
        stats = {
            "service": {
                "available": self.is_available(),
                "initialized": self.is_initialized
            },
            "model": self.get_model_info(),
            "metrics": self.metrics.to_dict()
        }
        
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        
        return stats
    
    def clear_cache(self) -> bool:
        """Limpiar cache de embeddings"""
        if self.cache:
            try:
                self.cache.clear()
                self.logger.info("Cache de embeddings limpiado")
                return True
            except Exception as e:
                self.logger.error(f"Error limpiando cache: {e}")
                return False
        return False
    
    def warm_up(self, sample_texts: List[str] = None):
        """Precalentar el modelo con textos de ejemplo"""
        if not self.is_available():
            return
        
        if sample_texts is None:
            sample_texts = [
                "Ejemplo de texto para precalentar el modelo de embeddings",
                "Este es otro texto de ejemplo más largo para verificar que el modelo funciona correctamente con diferentes longitudes de texto",
                "Texto corto"
            ]
        
        self.logger.info("Precalentando modelo de embeddings...")
        
        try:
            start_time = time.time()
            _ = self.encode_batch(sample_texts)
            warmup_time = time.time() - start_time
            
            self.logger.info(
                f"Precalentamiento completado en {warmup_time:.2f}s"
            )
            
        except Exception as e:
            self.logger.error(f"Error en precalentamiento: {e}")


# Instancia global del servicio
embedding_service = EmbeddingService()

# Funciones de conveniencia
def encode_text(text: str) -> np.ndarray:
    """Función de conveniencia para generar embedding de un texto"""
    return embedding_service.encode_single_text(text)

def encode_texts(texts: List[str]) -> List[np.ndarray]:
    """Función de conveniencia para generar embeddings de múltiples textos"""
    return embedding_service.encode_batch(texts)

def is_embedding_service_available() -> bool:
    """Función de conveniencia para verificar disponibilidad"""
    return embedding_service.is_available()

def get_embedding_stats() -> Dict[str, Any]:
    """Función de conveniencia para obtener estadísticas"""
    return embedding_service.get_stats()