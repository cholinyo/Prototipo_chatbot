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
        """
        Inicializar ChromaDB Vector Store
        
        Args:
            store_path: Ruta donde almacenar la base de datos
            collection_name: Nombre de la colección
            distance_function: Función de distancia (cosine, l2, ip)
        """
        self.store_path = Path(store_path)
        self.collection_name = collection_name
        self.distance_function = distance_function
        
        self.logger = get_logger(__name__)
        self.client = None
        self.collection = None
        self.metrics = ChromaDBMetrics()
        self._lock = threading.Lock()
        
        # Crear directorio si no existe
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Inicializar ChromaDB
        self._initialize_chromadb()
    
    def _initialize_chromadb(self):
        """Inicializar cliente y colección ChromaDB"""
        if not HAS_CHROMADB:
            self.logger.error(
                "ChromaDB no disponible. "
                "Instala con: pip install chromadb"
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
        metrics_file = self.store_path / "metrics.json"
        
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.metrics = ChromaDBMetrics.from_dict(data)
                    
                self.logger.debug("Métricas cargadas desde archivo")
                
            except Exception as e:
                self.logger.warning(f"Error cargando métricas: {e}")
        
        # Actualizar métricas básicas
        if self.collection:
            self.metrics.total_documents = self.collection.count()
            self.metrics.collection_size = self.metrics.total_documents
            self.metrics.database_path = str(self.store_path)
            self.metrics.distance_function = self.distance_function
    
    def _save_metrics(self):
        """Guardar métricas en archivo"""
        metrics_file = self.store_path / "metrics.json"
        
        try:
            # Actualizar métricas calculadas
            self._update_computed_metrics()
            
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics.to_dict(), f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.warning(f"Error guardando métricas: {e}")
    
    def _update_computed_metrics(self):
        """Actualizar métricas calculadas"""
        # Calcular promedios
        if self.metrics.total_add_operations > 0:
            # Evitar dividir por cero
            total_time = (self.metrics.total_add_operations * self.metrics.avg_add_time + 
                         self.metrics.last_add_time)
            self.metrics.avg_add_time = total_time / (self.metrics.total_add_operations + 1)
        
        if self.metrics.total_search_operations > 0:
            total_time = (self.metrics.total_search_operations * self.metrics.avg_search_time + 
                         self.metrics.last_search_time)
            self.metrics.avg_search_time = total_time / (self.metrics.total_search_operations + 1)
        
        # Calcular uso de disco
        try:
            total_size = sum(
                f.stat().st_size for f in self.store_path.rglob('*') if f.is_file()
            )
            self.metrics.disk_usage_mb = total_size / (1024 * 1024)
        except Exception:
            pass
    
    def is_available(self) -> bool:
        """Verificar si ChromaDB está disponible y funcional"""
        return (HAS_CHROMADB and 
                self.client is not None and 
                self.collection is not None)
    
    def add_documents(self, documents: List[DocumentChunk]) -> bool:
        """
        Añadir documentos al vector store
        
        Args:
            documents: Lista de DocumentChunk a añadir
            
        Returns:
            True si se añadieron correctamente
        """
        if not self.is_available():
            self.logger.error("ChromaDB no disponible")
            return False
        
        if not documents:
            return True
        
        start_time = time.time()
        
        try:
            with self._lock:
                # Importar EmbeddingService aquí para evitar importación circular
                from app.services.rag.embeddings import embedding_service
                
                # Generar embeddings para todos los documentos
                texts = [doc.content for doc in documents]
                embeddings = embedding_service.encode_batch(texts)
                
                if not embeddings:
                    self.logger.error("Error generando embeddings")
                    return False
                
                # Convertir a lista de arrays numpy si es necesario
                if isinstance(embeddings[0], np.ndarray):
                    embeddings_list = [emb.tolist() for emb in embeddings]
                else:
                    embeddings_list = embeddings
                
                # Preparar metadatos para ChromaDB
                ids = []
                metadatas = []
                documents_text = []
                
                for i, doc in enumerate(documents):
                    # Generar ID único si no existe
                    doc_id = doc.id if doc.id else str(uuid.uuid4())
                    ids.append(doc_id)
                    documents_text.append(doc.content)
                    
                    # Preparar metadatos completos para búsquedas
                    metadata = {
                        # Metadatos de DocumentMetadata
                        "source_path": doc.metadata.source_path,
                        "source_type": doc.metadata.source_type,
                        "file_type": doc.metadata.file_type,
                        "size_bytes": doc.metadata.size_bytes,
                        "created_at": doc.metadata.created_at.isoformat(),
                        "processed_at": doc.metadata.processed_at.isoformat(),
                        "checksum": doc.metadata.checksum,
                        "title": doc.metadata.title or "",
                        
                        # Metadatos de DocumentChunk
                        "chunk_id": doc.id,
                        "chunk_index": doc.chunk_index,
                        "chunk_size": doc.chunk_size,
                        "start_char": doc.start_char,
                        "end_char": doc.end_char,
                        
                        # Metadatos adicionales
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Añadir metadatos opcionales si existen
                    if hasattr(doc.metadata, 'language') and doc.metadata.language:
                        metadata["language"] = doc.metadata.language
                    if hasattr(doc.metadata, 'url') and doc.metadata.url:
                        metadata["url"] = doc.metadata.url
                    
                    metadatas.append(metadata)
                
                # Añadir a ChromaDB
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings_list,
                    documents=documents_text,
                    metadatas=metadatas
                )
                
                # Actualizar métricas
                self.metrics.last_add_time = time.time() - start_time
                self.metrics.total_add_operations += 1
                self.metrics.total_documents = self.collection.count()
                
                # Actualizar distribución por fuente
                for metadata in metadatas:
                    source = metadata.get("source_path", "unknown")
                    self.metrics.source_distribution[source] = (
                        self.metrics.source_distribution.get(source, 0) + 1
                    )
                
                # Guardar métricas
                self._save_metrics()
                
                self.logger.info(
                    f"Añadidos {len(documents)} documentos a ChromaDB",
                    time_ms=int(self.metrics.last_add_time * 1000),
                    total_documents=self.metrics.total_documents
                )
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error añadiendo documentos: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, k: int = 5, 
               filters: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        Búsqueda directa con embedding (para benchmark)
        
        Args:
            query_embedding: Vector de embedding de la consulta
            k: Número de resultados
            filters: Filtros opcionales
            
        Returns:
            Lista de DocumentChunk más similares
        """
        if not self.is_available():
            return []
        
        start_time = time.time()
        
        try:
            # Realizar búsqueda en ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=filters,
                include=["documents", "metadatas", "distances"]
            )
            
            # Actualizar métricas
            self.metrics.last_search_time = time.time() - start_time
            self.metrics.total_search_operations += 1
            
            # Convertir resultados a DocumentChunk
            chunks = []
            
            if (results['documents'] and 
                len(results['documents']) > 0 and 
                len(results['documents'][0]) > 0):
                
                documents = results['documents'][0]
                metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(documents)
                distances = results['distances'][0] if results['distances'] else [0.0] * len(documents)
                
                for i, (doc_text, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                    # Crear DocumentMetadata a partir de metadatos almacenados
                    doc_metadata = DocumentMetadata(
                        source_path=metadata.get("source_path", "unknown"),
                        source_type=metadata.get("source_type", "document"),
                        file_type=metadata.get("file_type", ".txt"),
                        size_bytes=metadata.get("size_bytes", len(doc_text)),
                        created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat())),
                        processed_at=datetime.fromisoformat(metadata.get("processed_at", datetime.now().isoformat())),
                        checksum=metadata.get("checksum", ""),
                        title=metadata.get("title", "")
                    )
                    
                    # Crear DocumentChunk
                    chunk = DocumentChunk(
                        id=metadata.get("chunk_id", f"chunk-{i}"),
                        content=doc_text,
                        metadata=doc_metadata,
                        chunk_index=metadata.get("chunk_index", 0),
                        chunk_size=metadata.get("chunk_size", len(doc_text)),
                        start_char=metadata.get("start_char", 0),
                        end_char=metadata.get("end_char", len(doc_text))
                    )
                    
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error en búsqueda directa: {e}")
            return []

    def similarity_search(self, 
                         query: str, 
                         k: int = 5,
                         filters: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        Búsqueda por similitud
        
        Args:
            query: Texto de consulta
            k: Número de resultados a devolver
            filters: Filtros de metadatos (opcional)
            
        Returns:
            Lista de DocumentChunk más similares
        """
        if not self.is_available():
            self.logger.warning("ChromaDB no disponible")
            return []
        
        start_time = time.time()
        
        try:
            # Importar EmbeddingService aquí para evitar importación circular
            from app.services.rag.embeddings import embedding_service
            
            # Generar embedding de la consulta
            query_embedding = embedding_service.encode_single_text(query)
            
            if query_embedding is None:
                self.logger.error("Error generando embedding de consulta")
                return []
            
            # Realizar búsqueda en ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=filters,
                include=["documents", "metadatas", "distances"]
            )
            
            # Actualizar métricas
            self.metrics.last_search_time = time.time() - start_time
            self.metrics.total_search_operations += 1
            
            # Convertir resultados a DocumentChunk
            chunks = []
            
            if (results['documents'] and 
                len(results['documents']) > 0 and 
                len(results['documents'][0]) > 0):
                
                documents = results['documents'][0]
                metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(documents)
                distances = results['distances'][0] if results['distances'] else [0.0] * len(documents)
                
                for i, (doc_text, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                    # Crear DocumentMetadata
                    doc_metadata = DocumentMetadata(
                        source=metadata.get("source", "unknown"),
                        title=metadata.get("title", ""),
                        extra={
                            "similarity_score": 1 - distance,  # Convertir distancia a score
                            "distance": distance,
                            "rank": i + 1,
                            **{k: v for k, v in metadata.items() 
                               if k not in ["source", "title"]}
                        }
                    )
                    
                    # Crear DocumentChunk
                    chunk = DocumentChunk(
                        content=doc_text,
                        metadata=doc_metadata,
                        chunk_index=metadata.get("chunk_index", 0),
                        total_chunks=metadata.get("total_chunks", 1)
                    )
                    
                    chunks.append(chunk)
            
            self.logger.info(
                f"Búsqueda completada: {len(chunks)} resultados",
                query_length=len(query),
                time_ms=int(self.metrics.last_search_time * 1000)
            )
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error en búsqueda: {e}")
            return []
    
    def delete_documents(self, filter_dict: Dict[str, Any]) -> bool:
        """
        Eliminar documentos basado en filtros
        
        Args:
            filter_dict: Diccionario de filtros para eliminar
            
        Returns:
            True si se eliminaron correctamente
        """
        if not self.is_available():
            return False
        
        try:
            with self._lock:
                # Obtener documentos que coinciden con el filtro
                results = self.collection.get(
                    where=filter_dict,
                    include=["metadatas"]
                )
                
                if results['ids']:
                    # Eliminar documentos
                    self.collection.delete(
                        ids=results['ids']
                    )
                    
                    # Actualizar métricas
                    self.metrics.total_documents = self.collection.count()
                    
                    # Actualizar distribución por fuente
                    for metadata in results['metadatas']:
                        source = metadata.get("source_path", "unknown")
                        if source in self.metrics.source_distribution:
                            self.metrics.source_distribution[source] = max(
                                0, self.metrics.source_distribution[source] - 1
                            )
                    
                    self._save_metrics()
                    
                    self.logger.info(
                        f"Eliminados {len(results['ids'])} documentos",
                        filter=filter_dict,
                        remaining=self.metrics.total_documents
                    )
                    
                    return True
                else:
                    self.logger.info("No se encontraron documentos para eliminar", filter=filter_dict)
                    return True
                    
        except Exception as e:
            self.logger.error(f"Error eliminando documentos: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del vector store"""
        if not self.is_available():
            return {}
        
        # Actualizar métricas
        self._update_computed_metrics()
        
        stats = {
            "type": "ChromaDB",
            "available": True,
            "total_documents": self.metrics.total_documents,
            "collection_name": self.collection_name,
            "store_path": str(self.store_path),
            "distance_function": self.metrics.distance_function,
            
            # Métricas de rendimiento
            "performance": {
                "avg_add_time_ms": round(self.metrics.avg_add_time * 1000, 2),
                "avg_search_time_ms": round(self.metrics.avg_search_time * 1000, 2),
                "total_operations": {
                    "add": self.metrics.total_add_operations,
                    "search": self.metrics.total_search_operations
                }
            },
            
            # Métricas de almacenamiento
            "storage": {
                "disk_usage_mb": round(self.metrics.disk_usage_mb, 2),
                "memory_usage_mb": round(self.metrics.memory_usage_mb, 2)
            },
            
            # Distribución por fuente
            "source_distribution": dict(self.metrics.source_distribution),
            
            # Configuración
            "config": {
                "collection_name": self.collection_name,
                "distance_function": self.distance_function,
                "store_path": str(self.store_path)
            }
        }
        
        return stats
    
    def clear(self) -> bool:
        """Limpiar todos los documentos (alias para clear_all para compatibilidad)"""
        return self.clear_all()

    def clear_all(self) -> bool:
        """Limpiar todos los documentos"""
        if not self.is_available():
            return False
        
        try:
            with self._lock:
                # Eliminar colección y recrear
                self.client.delete_collection(name=self.collection_name)
                self._create_or_get_collection()
                
                # Resetear métricas
                self.metrics = ChromaDBMetrics()
                self.metrics.database_path = str(self.store_path)
                self.metrics.distance_function = self.distance_function
                
                self._save_metrics()
                
                self.logger.info("Todos los documentos eliminados de ChromaDB")
                return True
                
        except Exception as e:
            self.logger.error(f"Error limpiando vector store: {e}")
            return False
    
    def optimize(self):
        """Optimizar el vector store"""
        if not self.is_available():
            return
        
        try:
            start_time = time.time()
            
            self.logger.info("Iniciando optimización ChromaDB...")
            
            # ChromaDB se optimiza automáticamente
            # Pero podemos forzar persistencia
            if hasattr(self.collection, 'persist'):
                self.collection.persist()
            
            # Actualizar métricas
            self._update_computed_metrics()
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