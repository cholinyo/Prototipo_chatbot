"""
FAISS Vector Store Implementation para Prototipo_chatbot
Implementación completa con benchmarking y filtrado avanzado
TFM Vicente Caruncho - Sistemas Inteligentes
"""

import os
import time
import pickle
import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import shutil

# FAISS import
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# Imports locales
from app.core.logger import get_logger
from app.models import DocumentChunk, DocumentMetadata


@dataclass
class FaissMetrics:
    """Métricas específicas de FAISS para benchmarking"""
    total_vectors: int = 0
    index_size_bytes: int = 0
    index_type: str = ""
    dimension: int = 0
    
    # Métricas de rendimiento
    last_add_time: float = 0.0
    last_search_time: float = 0.0
    total_add_operations: int = 0
    total_search_operations: int = 0
    avg_add_time: float = 0.0
    avg_search_time: float = 0.0
    
    # Métricas de memoria
    memory_usage_mb: float = 0.0
    index_build_time: float = 0.0
    
    # Distribución por tipo de fuente
    source_distribution: Dict[str, int] = None
    
    def __post_init__(self):
        if self.source_distribution is None:
            self.source_distribution = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FaissMetrics':
        """Crear desde diccionario"""
        return cls(**data)


class FaissVectorStore:
    """
    Vector Store usando FAISS con arquitectura híbrida:
    - Índice único global para máxima flexibilidad
    - Filtrado virtual por metadatos
    - Benchmarking integrado para comparaciones académicas
    """
    
    def __init__(self, 
                 store_path: str = "data/vectorstore/faiss",
                 dimension: int = 384,
                 index_type: str = "IndexFlatL2",
                 normalize_vectors: bool = True):
        
        self.logger = get_logger("faiss_vector_store")
        self.store_path = Path(store_path)
        self.dimension = dimension
        self.index_type = index_type
        self.normalize_vectors = normalize_vectors
        
        # Crear directorios
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.backup_path = self.store_path / "backup"
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
        # Archivos de persistencia
        self.index_file = self.store_path / "faiss_index.index"
        self.metadata_file = self.store_path / "metadata.pkl"
        self.id_mapping_file = self.store_path / "id_mapping.pkl"
        self.config_file = self.store_path / "index_config.yaml"
        self.metrics_file = self.store_path / "metrics.pkl"
        
        # Estado interno
        self.index = None
        self.metadata = {}          # {faiss_idx: metadata_dict}
        self.id_mapping = {}        # {chunk_id: faiss_idx}
        self.reverse_mapping = {}   # {faiss_idx: chunk_id}
        self.metrics = FaissMetrics(dimension=dimension, index_type=index_type)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Inicializar
        self._initialize()
    
    def _initialize(self):
        """Inicializar el vector store"""
        if not HAS_FAISS:
            self.logger.error(
                "FAISS no está instalado. Instala con: pip install faiss-cpu"
            )
            return
        
        try:
            # Cargar índice existente o crear nuevo
            if self.index_file.exists():
                self._load_from_disk()
            else:
                self._create_new_index()
            
            self.logger.info(
                "FaissVectorStore inicializado",
                vectors=self.metrics.total_vectors,
                dimension=self.dimension,
                index_type=self.index_type,
                store_path=str(self.store_path)
            )
            
        except Exception as e:
            self.logger.error(f"Error inicializando FaissVectorStore: {e}")
            self.index = None
    
    def _create_new_index(self):
        """Crear un nuevo índice FAISS"""
        if self.index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexHNSW":
            # HNSW con parámetros optimizados
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 100
        else:
            # Fallback a IndexFlatL2
            self.logger.warning(f"Tipo de índice desconocido: {self.index_type}, usando IndexFlatL2")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index_type = "IndexFlatL2"
        
        # Inicializar estructuras
        self.metadata = {}
        self.id_mapping = {}
        self.reverse_mapping = {}
        self.metrics = FaissMetrics(dimension=self.dimension, index_type=self.index_type)
        
        self.logger.info(f"Nuevo índice FAISS creado: {self.index_type}")
    
    def _load_from_disk(self):
        """Cargar índice y metadatos desde disco"""
        try:
            # Cargar índice FAISS
            self.index = faiss.read_index(str(self.index_file))
            
            # Cargar metadatos
            if self.metadata_file.exists():
                with open(self.metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
            
            # Cargar mapeos de IDs
            if self.id_mapping_file.exists():
                with open(self.id_mapping_file, 'rb') as f:
                    self.id_mapping = pickle.load(f)
                
                # Reconstruir mapeo inverso
                self.reverse_mapping = {v: k for k, v in self.id_mapping.items()}
            
            # Cargar métricas
            if self.metrics_file.exists():
                with open(self.metrics_file, 'rb') as f:
                    metrics_data = pickle.load(f)
                    self.metrics = FaissMetrics.from_dict(metrics_data)
            
            # Actualizar métricas con estado actual
            self.metrics.total_vectors = self.index.ntotal
            self.metrics.dimension = self.index.d
            
            self.logger.info(
                "Índice FAISS cargado desde disco",
                vectors=self.metrics.total_vectors,
                metadata_entries=len(self.metadata)
            )
            
        except Exception as e:
            self.logger.error(f"Error cargando desde disco: {e}")
            self.logger.info("Creando nuevo índice...")
            self._create_new_index()
    
    def _save_to_disk(self):
        """Guardar índice y metadatos a disco"""
        try:
            with self._lock:
                # Crear backup antes de guardar
                self._create_backup()
                
                # Guardar índice FAISS
                faiss.write_index(self.index, str(self.index_file))
                
                # Guardar metadatos
                with open(self.metadata_file, 'wb') as f:
                    pickle.dump(self.metadata, f)
                
                # Guardar mapeos
                with open(self.id_mapping_file, 'wb') as f:
                    pickle.dump(self.id_mapping, f)
                
                # Guardar métricas
                with open(self.metrics_file, 'wb') as f:
                    pickle.dump(self.metrics.to_dict(), f)
                
                # Guardar configuración
                config = {
                    'index_info': {
                        'type': self.index_type,
                        'dimension': self.dimension,
                        'normalize_vectors': self.normalize_vectors,
                        'created_at': datetime.now().isoformat()
                    },
                    'statistics': {
                        'total_vectors': self.metrics.total_vectors,
                        'source_distribution': self.metrics.source_distribution
                    },
                    'performance': self.metrics.to_dict()
                }
                
                with open(self.config_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                self.logger.debug("Datos guardados a disco exitosamente")
                
        except Exception as e:
            self.logger.error(f"Error guardando a disco: {e}")
    
    def _create_backup(self):
        """Crear backup del estado actual"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.backup_path / f"backup_{timestamp}"
            backup_dir.mkdir(exist_ok=True)
            
            # Copiar archivos existentes
            for file_path in [self.index_file, self.metadata_file, 
                             self.id_mapping_file, self.config_file]:
                if file_path.exists():
                    shutil.copy2(file_path, backup_dir / file_path.name)
            
            # Limpiar backups antiguos (mantener solo los últimos 5)
            backups = sorted(self.backup_path.glob("backup_*"))
            while len(backups) > 5:
                oldest = backups.pop(0)
                shutil.rmtree(oldest)
                
        except Exception as e:
            self.logger.warning(f"Error creando backup: {e}")
    
    def is_available(self) -> bool:
        """Verificar si el vector store está disponible"""
        return HAS_FAISS and self.index is not None
    
    def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Añadir documentos al índice"""
        if not self.is_available():
            self.logger.error("FaissVectorStore no disponible")
            return False
        
        if not chunks:
            return True
        
        start_time = time.time()
        
        try:
            with self._lock:
                # Extraer embeddings
                embeddings = []
                valid_chunks = []
                
                for chunk in chunks:
                    if hasattr(chunk, 'embedding') and chunk.embedding is not None:
                        embeddings.append(np.array(chunk.embedding, dtype=np.float32))
                        valid_chunks.append(chunk)
                    else:
                        self.logger.warning(f"Chunk {chunk.id} sin embedding, saltando")
                
                if not embeddings:
                    self.logger.warning("No se encontraron embeddings válidos")
                    return False
                
                # Convertir a matriz numpy
                embedding_matrix = np.vstack(embeddings)
                
                # Normalizar si está habilitado
                if self.normalize_vectors:
                    faiss.normalize_L2(embedding_matrix)
                
                # Obtener índices de inicio para los nuevos vectores
                start_idx = self.index.ntotal
                
                # Añadir al índice FAISS
                self.index.add(embedding_matrix)
                
                # Actualizar metadatos y mapeos
                for i, chunk in enumerate(valid_chunks):
                    faiss_idx = start_idx + i
                    
                    # Guardar metadatos completos
                    self.metadata[faiss_idx] = {
                        'chunk_id': chunk.id,
                        'content': chunk.content,
                        'source_type': chunk.metadata.source_type,
                        'file_type': chunk.metadata.file_type,
                        'source_path': chunk.metadata.source_path,
                        'created_at': chunk.metadata.created_at.isoformat() if chunk.metadata.created_at else None,
                        'processed_at': chunk.metadata.processed_at.isoformat() if chunk.metadata.processed_at else None,
                        'chunk_index': chunk.chunk_index,
                        'chunk_size': chunk.chunk_size,
                        'start_char': chunk.start_char,
                        'end_char': chunk.end_char,
                        'section_title': getattr(chunk, 'section_title', None),
                        'page_number': getattr(chunk, 'page_number', None),
                        'word_count': len(chunk.content.split()) if chunk.content else 0
                    }
                    
                    # Mapeos bidireccionales
                    self.id_mapping[chunk.id] = faiss_idx
                    self.reverse_mapping[faiss_idx] = chunk.id
                    
                    # Actualizar distribución por tipo
                    source_type = chunk.metadata.source_type
                    self.metrics.source_distribution[source_type] = (
                        self.metrics.source_distribution.get(source_type, 0) + 1
                    )
                
                # Actualizar métricas
                add_time = time.time() - start_time
                self.metrics.total_vectors = self.index.ntotal
                self.metrics.last_add_time = add_time
                self.metrics.total_add_operations += 1
                self.metrics.avg_add_time = (
                    (self.metrics.avg_add_time * (self.metrics.total_add_operations - 1) + add_time) /
                    self.metrics.total_add_operations
                )
                
                # Guardar a disco
                self._save_to_disk()
                
                self.logger.info(
                    "Documentos añadidos a FAISS",
                    chunks_added=len(valid_chunks),
                    total_vectors=self.metrics.total_vectors,
                    add_time=add_time
                )
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error añadiendo documentos: {e}")
            return False
    
    def search(self, 
               query_embedding: np.ndarray, 
               k: int = 5,
               filters: Dict[str, Any] = None) -> List[Tuple[DocumentChunk, float]]:
        """Buscar documentos similares"""
        if not self.is_available():
            return []
        
        start_time = time.time()
        
        try:
            with self._lock:
                # Preparar query embedding
                query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
                
                if self.normalize_vectors:
                    faiss.normalize_L2(query_vector)
                
                # Determinar k_search (buscar más si hay filtros)
                k_search = k * 3 if filters else k
                k_search = min(k_search, self.index.ntotal)
                
                if k_search == 0:
                    return []
                
                # Búsqueda en FAISS
                distances, indices = self.index.search(query_vector, k_search)
                
                # Procesar resultados
                results = []
                for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx == -1:  # FAISS devuelve -1 para slots vacíos
                        continue
                    
                    # Obtener metadatos
                    if idx not in self.metadata:
                        continue
                    
                    metadata_dict = self.metadata[idx]
                    
                    # Aplicar filtros si existen
                    if filters and not self._apply_filters(metadata_dict, filters):
                        continue
                    
                    # Reconstruir DocumentChunk
                    chunk = self._reconstruct_chunk(metadata_dict)
                    
                    # Convertir distancia a score (1 / (1 + distance))
                    score = float(1.0 / (1.0 + dist))
                    
                    results.append((chunk, score))
                    
                    # Parar si tenemos suficientes resultados
                    if len(results) >= k:
                        break
                
                # Actualizar métricas
                search_time = time.time() - start_time
                self.metrics.last_search_time = search_time
                self.metrics.total_search_operations += 1
                self.metrics.avg_search_time = (
                    (self.metrics.avg_search_time * (self.metrics.total_search_operations - 1) + search_time) /
                    self.metrics.total_search_operations
                )
                
                self.logger.debug(
                    "Búsqueda FAISS completada",
                    k_requested=k,
                    results_found=len(results),
                    search_time=search_time,
                    filters_applied=filters is not None
                )
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error en búsqueda: {e}")
            return []
    
    def _apply_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Aplicar filtros por metadatos"""
        for filter_key, filter_value in filters.items():
            if filter_key not in metadata:
                continue
            
            metadata_value = metadata[filter_key]
            
            if isinstance(filter_value, list):
                if metadata_value not in filter_value:
                    return False
            elif isinstance(filter_value, dict):
                # Filtros de rango (ej: date_range)
                if filter_key == 'date_range' and 'created_at' in metadata:
                    try:
                        from datetime import datetime
                        doc_date = datetime.fromisoformat(metadata['created_at'])
                        start_date = datetime.fromisoformat(filter_value['start'])
                        end_date = datetime.fromisoformat(filter_value['end'])
                        if not (start_date <= doc_date <= end_date):
                            return False
                    except:
                        return False
            else:
                if metadata_value != filter_value:
                    return False
        
        return True
    
    def _reconstruct_chunk(self, metadata: Dict[str, Any]) -> DocumentChunk:
        """Reconstruir DocumentChunk desde metadatos"""
        # Crear DocumentMetadata
        doc_metadata = DocumentMetadata(
            source_path=metadata.get('source_path', ''),
            source_type=metadata.get('source_type', 'unknown'),
            file_type=metadata.get('file_type', ''),
            size_bytes=metadata.get('chunk_size', 0),
            created_at=datetime.fromisoformat(metadata['created_at']) if metadata.get('created_at') else datetime.now(),
            processed_at=datetime.fromisoformat(metadata['processed_at']) if metadata.get('processed_at') else datetime.now(),
            checksum=metadata.get('checksum', '')
        )
        
        # Crear DocumentChunk
        chunk = DocumentChunk(
            id=metadata.get('chunk_id', ''),
            content=metadata.get('content', ''),
            metadata=doc_metadata,
            chunk_index=metadata.get('chunk_index', 0),
            chunk_size=metadata.get('chunk_size', 0),
            start_char=metadata.get('start_char', 0),
            end_char=metadata.get('end_char', 0),
            section_title=metadata.get('section_title'),
            page_number=metadata.get('page_number')
        )
        
        return chunk
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del vector store"""
        if not self.is_available():
            return {"available": False}
        
        # Calcular uso de memoria aproximado
        if self.index:
            # Estimación: vectores * dimensión * 4 bytes (float32) + overhead
            vector_memory = self.index.ntotal * self.dimension * 4
            self.metrics.memory_usage_mb = vector_memory / (1024 * 1024)
        
        return {
            "available": True,
            "type": "FAISS",
            "index_type": self.index_type,
            "total_vectors": self.metrics.total_vectors,
            "dimension": self.dimension,
            "normalize_vectors": self.normalize_vectors,
            "metrics": self.metrics.to_dict(),
            "store_path": str(self.store_path),
            "index_size_mb": self.metrics.memory_usage_mb
        }
    
    def clear(self):
        """Limpiar todo el índice"""
        try:
            with self._lock:
                # Crear backup antes de limpiar
                self._create_backup()
                
                # Recrear índice vacío
                self._create_new_index()
                
                # Limpiar archivos
                for file_path in [self.index_file, self.metadata_file, 
                                 self.id_mapping_file, self.metrics_file]:
                    if file_path.exists():
                        file_path.unlink()
                
                self.logger.info("Índice FAISS limpiado")
                
        except Exception as e:
            self.logger.error(f"Error limpiando índice: {e}")
    
    def optimize_index(self):
        """Optimizar índice para búsquedas más rápidas"""
        if not self.is_available() or self.index.ntotal == 0:
            return
        
        try:
            with self._lock:
                self.logger.info("Optimizando índice FAISS...")
                start_time = time.time()
                
                # Para índices pequeños, mantener IndexFlatL2
                if self.index.ntotal < 10000:
                    self.logger.info("Índice pequeño, no requiere optimización")
                    return
                
                # Para índices grandes, considerar migrar a IndexIVF
                if self.index_type == "IndexFlatL2" and self.index.ntotal > 50000:
                    self.logger.info("Migrando a IndexIVF para mejor rendimiento...")
                    
                    # Crear nuevo índice IVF
                    nlist = min(int(np.sqrt(self.index.ntotal)), 1000)
                    quantizer = faiss.IndexFlatL2(self.dimension)
                    new_index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                    
                    # Entrenar con todos los vectores
                    all_vectors = np.array([self.index.reconstruct(i) for i in range(self.index.ntotal)])
                    new_index.train(all_vectors)
                    new_index.add(all_vectors)
                    
                    # Reemplazar índice
                    self.index = new_index
                    self.index_type = "IndexIVFFlat"
                    
                    # Guardar cambios
                    self._save_to_disk()
                
                optimization_time = time.time() - start_time
                self.logger.info(f"Optimización completada en {optimization_time:.2f}s")
                
        except Exception as e:
            self.logger.error(f"Error optimizando índice: {e}")


# Funciones de conveniencia
def create_faiss_store(store_path: str = "data/vectorstore/faiss",
                      dimension: int = 384,
                      index_type: str = "IndexFlatL2") -> FaissVectorStore:
    """Crear nueva instancia de FaissVectorStore"""
    return FaissVectorStore(
        store_path=store_path,
        dimension=dimension,
        index_type=index_type
    )


# Instancia global por defecto
faiss_store = FaissVectorStore()


def get_faiss_store() -> FaissVectorStore:
    """Obtener instancia global del store"""
    return faiss_store


def is_faiss_available() -> bool:
    """Verificar si FAISS está disponible"""
    return HAS_FAISS and faiss_store.is_available()