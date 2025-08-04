"""
ChromaDB Vector Store Implementation para Prototipo_chatbot
Implementación completa con benchmarking y filtrado avanzado
TFM Vicente Caruncho - Sistemas Inteligentes
"""

import os
import time
import json
import uuid
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import threading

# ChromaDB import
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

# Imports locales
from app.core.logger import get_logger
from app.models import DocumentChunk, DocumentMetadata


@dataclass
class ChromaDBMetrics:
    """Métricas específicas de ChromaDB para benchmarking"""
    total_documents: int = 0
    collection_size: int = 0
    database_path: str = ""
    
    # Métricas de rendimiento
    last_add_time: float = 0.0
    last_search_time: float = 0.0
    total_add_operations: int = 0
    total_search_operations: int = 0
    avg_add_time: float = 0.0
    avg_search_time: float = 0.0
    
    # Métricas de memoria y almacenamiento
    memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0
    
    # Distribución por tipo de fuente
    source_distribution: Dict[str, int] = None
    
    # Configuración ChromaDB
    distance_function: str = "cosine"
    embedding_function: str = "custom"
    
    def __post_init__(self):
        if self.source_distribution is None:
            self.source_distribution = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChromaDBMetrics':
        """Crear desde diccionario"""
        return cls(**data)


class ChromaDBVectorStore:
    """
    Vector Store usando ChromaDB con arquitectura híbrida:
    - Base de datos persistente con colecciones
    - Filtrado avanzado por metadatos
    - Benchmarking integrado para comparaciones académicas con FAISS
    """
    
    def __init__(self, 
                 store_path: str = "data/vectorstore/chromadb",
                 collection_name: str = "prototipo_documents",
                 distance_function: str = "cosine"):
        
        self.logger = get_logger("chromadb_vector_store")
        self.store_path = Path(store_path)
        self.collection_name = collection_name
        self.distance_function = distance_function
        
        # Crear directorios
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Estado interno
        self.client = None
        self.collection = None
        self.metrics = ChromaDBMetrics(
            database_path=str(self.store_path),
            distance_function=distance_function
        )
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Inicializar
        self._initialize()
    
    def _initialize(self):
        """Inicializar el vector store"""
        if not HAS_CHROMADB:
            self.logger.error(
                "ChromaDB no está instalado. Instala con: pip install chromadb"
            )
            return
        
        try:
            # Configurar cliente ChromaDB con persistencia
            settings = Settings(
                persist_directory=str(self.store_path),
                anonymized_telemetry=False
            )
            
            self.client = chromadb.PersistentClient(
                path=str(self.store_path),
                settings=settings
            )
            
            # Crear o obtener colección
            self._create_or_get_collection()
            
            # Cargar métricas existentes
            self._load_metrics()
            
            self.logger.info(
                "ChromaDBVectorStore inicializado",
                collection=self.collection_name,
                documents=self.metrics.total_documents,
                store_path=str(self.store_path)
            )
            
        except Exception as e:
            self.logger.error(f"Error inicializando ChromaDBVectorStore: {e}")
            self.client = None
            self.collection = None
    
    def _create_or_get_collection(self):
        """Crear o obtener colección existente"""
        try:
            # Intentar obtener colección existente
            self.collection = self.client.get_collection(
                name=self.collection_name
            )
            
            # Actualizar métricas con datos existentes
            self.metrics.total_documents = self.collection.count()
            
            self.logger.info(
                f"Colección existente cargada: {self.collection_name}",
                documents=self.metrics.total_documents
            )
            
        except Exception:
            # Crear nueva colección si no existe
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": self.distance_function,
                    "description": "Prototipo_chatbot document collection",
                    "created_at": datetime.now().isoformat()
                }
            )
            
            self.metrics.total_documents = 0
            
            self.logger.info(f"Nueva colección creada: {self.collection_name}")
    
    def _load_metrics(self):
        """Cargar métricas desde archivo"""
        metrics_file = self.store_path / "chromadb_metrics.json"
        
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    self.metrics = ChromaDBMetrics.from_dict(metrics_data)
                    
                self.logger.debug("Métricas ChromaDB cargadas desde disco")
                
            except Exception as e:
                self.logger.warning(f"Error cargando métricas: {e}")
    
    def _save_metrics(self):
        """Guardar métricas a archivo"""
        metrics_file = self.store_path / "chromadb_metrics.json"
        
        try:
            # Calcular uso de disco
            self._calculate_disk_usage()
            
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics.to_dict(), f, indent=2, default=str)
                
            self.logger.debug("Métricas ChromaDB guardadas a disco")
            
        except Exception as e:
            self.logger.warning(f"Error guardando métricas: {e}")
    
    def _calculate_disk_usage(self):
        """Calcular uso de disco de la base de datos"""
        try:
            total_size = 0
            for path in self.store_path.rglob("*"):
                if path.is_file():
                    total_size += path.stat().st_size
            
            self.metrics.disk_usage_mb = total_size / (1024 * 1024)
            
        except Exception as e:
            self.logger.warning(f"Error calculando uso de disco: {e}")
    
    def is_available(self) -> bool:
        """Verificar si el vector store está disponible"""
        return HAS_CHROMADB and self.client is not None and self.collection is not None
    
    def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Añadir documentos al índice"""
        if not self.is_available():
            self.logger.error("ChromaDBVectorStore no disponible")
            return False
        
        if not chunks:
            return True
        
        start_time = time.time()
        
        try:
            with self._lock:
                # Preparar datos para ChromaDB
                documents = []
                embeddings = []
                metadatas = []
                ids = []
                
                valid_chunks = []
                
                for chunk in chunks:
                    if hasattr(chunk, 'embedding') and chunk.embedding is not None:
                        # Documentos (contenido de texto)
                        documents.append(chunk.content)
                        
                        # Embeddings (vectores)
                        embeddings.append(chunk.embedding)
                        
                        # Metadatos (información estructurada)
                        metadata = {
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
                        metadatas.append(metadata)
                        
                        # IDs únicos
                        ids.append(chunk.id)
                        
                        valid_chunks.append(chunk)
                    else:
                        self.logger.warning(f"Chunk {chunk.id} sin embedding, saltando")
                
                if not documents:
                    self.logger.warning("No se encontraron documentos válidos")
                    return False
                
                # Añadir a ChromaDB
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                
                # Actualizar métricas
                add_time = time.time() - start_time
                self.metrics.total_documents = self.collection.count()
                self.metrics.last_add_time = add_time
                self.metrics.total_add_operations += 1
                self.metrics.avg_add_time = (
                    (self.metrics.avg_add_time * (self.metrics.total_add_operations - 1) + add_time) /
                    self.metrics.total_add_operations
                )
                
                # Actualizar distribución por tipo
                for chunk in valid_chunks:
                    source_type = chunk.metadata.source_type
                    self.metrics.source_distribution[source_type] = (
                        self.metrics.source_distribution.get(source_type, 0) + 1
                    )
                
                # Guardar métricas
                self._save_metrics()
                
                self.logger.info(
                    "Documentos añadidos a ChromaDB",
                    chunks_added=len(valid_chunks),
                    total_documents=self.metrics.total_documents,
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
                # Preparar query
                query_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
                
                # Construir where clause para filtros
                where_clause = None
                if filters:
                    where_clause = self._build_where_clause(filters)
                
                # Realizar búsqueda en ChromaDB
                results = self.collection.query(
                    query_embeddings=[query_list],
                    n_results=k,
                    where=where_clause,
                    include=['documents', 'metadatas', 'distances']
                )
                
                # Procesar resultados
                search_results = []
                
                if results['ids'] and results['ids'][0]:
                    for i in range(len(results['ids'][0])):
                        # Extraer datos
                        doc_id = results['ids'][0][i]
                        content = results['documents'][0][i]
                        metadata_dict = results['metadatas'][0][i]
                        distance = results['distances'][0][i] if 'distances' in results else 0.0
                        
                        # Reconstruir DocumentChunk
                        chunk = self._reconstruct_chunk(doc_id, content, metadata_dict)
                        
                        # Convertir distancia a score
                        # Para cosine distance: score = 1 - distance
                        score = max(0.0, 1.0 - distance) if self.distance_function == "cosine" else 1.0 / (1.0 + distance)
                        
                        search_results.append((chunk, float(score)))
                
                # Actualizar métricas
                search_time = time.time() - start_time
                self.metrics.last_search_time = search_time
                self.metrics.total_search_operations += 1
                self.metrics.avg_search_time = (
                    (self.metrics.avg_search_time * (self.metrics.total_search_operations - 1) + search_time) /
                    self.metrics.total_search_operations
                )
                
                self.logger.debug(
                    "Búsqueda ChromaDB completada",
                    k_requested=k,
                    results_found=len(search_results),
                    search_time=search_time,
                    filters_applied=filters is not None
                )
                
                return search_results
                
        except Exception as e:
            self.logger.error(f"Error en búsqueda: {e}")
            return []
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Construir cláusula WHERE para ChromaDB"""
        where_clause = {}
        
        for filter_key, filter_value in filters.items():
            if isinstance(filter_value, list):
                where_clause[filter_key] = {"$in": filter_value}
            elif isinstance(filter_value, dict):
                # Filtros de rango
                if filter_key == 'date_range' and 'start' in filter_value and 'end' in filter_value:
                    where_clause['created_at'] = {
                        "$gte": filter_value['start'],
                        "$lte": filter_value['end']
                    }
            else:
                where_clause[filter_key] = {"$eq": filter_value}
        
        return where_clause
    
    def _reconstruct_chunk(self, doc_id: str, content: str, metadata_dict: Dict[str, Any]) -> DocumentChunk:
        """Reconstruir DocumentChunk desde datos de ChromaDB"""
        # Crear DocumentMetadata
        doc_metadata = DocumentMetadata(
            source_path=metadata_dict.get('source_path', ''),
            source_type=metadata_dict.get('source_type', 'unknown'),
            file_type=metadata_dict.get('file_type', ''),
            size_bytes=metadata_dict.get('chunk_size', 0),
            created_at=datetime.fromisoformat(metadata_dict['created_at']) if metadata_dict.get('created_at') else datetime.now(),
            processed_at=datetime.fromisoformat(metadata_dict['processed_at']) if metadata_dict.get('processed_at') else datetime.now(),
            checksum=metadata_dict.get('checksum', '')
        )
        
        # Crear DocumentChunk
        chunk = DocumentChunk(
            id=doc_id,
            content=content,
            metadata=doc_metadata,
            chunk_index=metadata_dict.get('chunk_index', 0),
            chunk_size=metadata_dict.get('chunk_size', 0),
            start_char=metadata_dict.get('start_char', 0),
            end_char=metadata_dict.get('end_char', 0),
            section_title=metadata_dict.get('section_title'),
            page_number=metadata_dict.get('page_number')
        )
        
        return chunk
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del vector store"""
        if not self.is_available():
            return {"available": False}
        
        # Actualizar conteo actual
        try:
            self.metrics.total_documents = self.collection.count()
            self.metrics.collection_size = self.metrics.total_documents
        except:
            pass
        
        # Calcular uso de disco
        self._calculate_disk_usage()
        
        return {
            "available": True,
            "type": "ChromaDB",
            "collection_name": self.collection_name,
            "total_documents": self.metrics.total_documents,
            "distance_function": self.distance_function,
            "metrics": self.metrics.to_dict(),
            "store_path": str(self.store_path),
            "disk_usage_mb": self.metrics.disk_usage_mb
        }
    
    def clear(self):
        """Limpiar toda la colección"""
        try:
            with self._lock:
                if self.collection:
                    # Obtener todos los IDs
                    results = self.collection.get()
                    if results['ids']:
                        # Eliminar todos los documentos
                        self.collection.delete(ids=results['ids'])
                
                # Resetear métricas
                self.metrics.total_documents = 0
                self.metrics.source_distribution = {}
                self._save_metrics()
                
                self.logger.info("Colección ChromaDB limpiada")
                
        except Exception as e:
            self.logger.error(f"Error limpiando colección: {e}")
    
    def optimize_collection(self):
        """Optimizar colección (ChromaDB maneja esto automáticamente)"""
        if not self.is_available():
            return
        
        try:
            # ChromaDB optimiza automáticamente, pero podemos hacer housekeeping
            start_time = time.time()
            
            # Actualizar métricas
            self.metrics.total_documents = self.collection.count()
            self._calculate_disk_usage()
            self._save_metrics()
            
            optimization_time = time.time() - start_time
            
            self.logger.info(
                f"Optimización ChromaDB completada en {optimization_time:.2f}s",
                documents=self.metrics.total_documents,
                disk_usage_mb=self.metrics.disk_usage_mb
            )
            
        except Exception as e:
            self.logger.error(f"Error optimizando colección: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Obtener información detallada de la colección"""
        if not self.is_available():
            return {}
        
        try:
            # Información básica
            collection_info = {
                "name": self.collection_name,
                "count": self.collection.count(),
                "metadata": self.collection.metadata
            }
            
            # Ejemplo de documentos (primeros 3)
            sample = self.collection.peek(limit=3)
            collection_info["sample_documents"] = {
                "ids": sample.get('ids', []),
                "documents": [doc[:100] + "..." if len(doc) > 100 else doc 
                            for doc in sample.get('documents', [])],
                "metadatas": sample.get('metadatas', [])
            }
            
            return collection_info
            
        except Exception as e:
            self.logger.error(f"Error obteniendo info de colección: {e}")
            return {}


# Funciones de conveniencia
def create_chromadb_store(store_path: str = "data/vectorstore/chromadb",
                         collection_name: str = "prototipo_documents",
                         distance_function: str = "cosine") -> ChromaDBVectorStore:
    """Crear nueva instancia de ChromaDBVectorStore"""
    return ChromaDBVectorStore(
        store_path=store_path,
        collection_name=collection_name,
        distance_function=distance_function
    )


# Instancia global por defecto
chromadb_store = ChromaDBVectorStore()


def get_chromadb_store() -> ChromaDBVectorStore:
    """Obtener instancia global del store"""
    return chromadb_store


def is_chromadb_available() -> bool:
    """Verificar si ChromaDB está disponible"""
    return HAS_CHROMADB and chromadb_store.is_available()