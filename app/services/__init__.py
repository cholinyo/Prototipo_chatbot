"""
Servicio RAG (Retrieval-Augmented Generation) para Prototipo_chatbot
Maneja embeddings, búsqueda vectorial y recuperación de contexto
TFM Vicente Caruncho - Sistemas Inteligentes
"""

import os
import pickle
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import json

# Imports para embeddings y vectores
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

# Imports locales
from app.core.config import get_rag_config, get_vector_store_config, get_model_config
from app.core.logger import get_logger
from app.models import DocumentChunk, DocumentMetadata, create_document_chunk

class EmbeddingService:
    """Servicio para generar embeddings semánticos"""
    
    def __init__(self):
        self.config = get_model_config()
        self.logger = get_logger("embedding_service")
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializar modelo de embeddings"""
        if not HAS_SENTENCE_TRANSFORMERS:
            self.logger.error("sentence-transformers no instalado. Instala con: pip install sentence-transformers")
            return
        
        try:
            model_name = self.config.embedding_name
            cache_dir = self.config.embedding_cache_dir
            device = self.config.embedding_device
            
            # Crear directorio de cache
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            
            # Cargar modelo
            self.model = SentenceTransformer(
                model_name,
                cache_folder=cache_dir,
                device=device
            )
            
            self.logger.info("Modelo de embeddings inicializado",
                           model=model_name,
                           device=device,
                           dimension=self.config.embedding_dimension)
            
        except Exception as e:
            self.logger.error("Error inicializando modelo embeddings", error=str(e))
            self.model = None
    
    def is_available(self) -> bool:
        """Verificar si el servicio está disponible"""
        return self.model is not None
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Generar embeddings para lista de textos"""
        if not self.is_available():
            raise RuntimeError("Servicio de embeddings no disponible")
        
        if not texts:
            return np.array([])
        
        try:
            start_time = time.time()
            
            # Generar embeddings
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10
            )
            
            encode_time = time.time() - start_time
            
            self.logger.debug("Embeddings generados",
                            texts_count=len(texts),
                            embedding_shape=embeddings.shape,
                            encode_time=encode_time)
            
            return embeddings
            
        except Exception as e:
            self.logger.error("Error generando embeddings", error=str(e))
            raise
    
    def encode_single_text(self, text: str) -> np.ndarray:
        """Generar embedding para un solo texto"""
        return self.encode_texts([text])[0]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo"""
        if not self.is_available():
            return {"available": False}
        
        return {
            "available": True,
            "model_name": self.config.embedding_name,
            "dimension": self.config.embedding_dimension,
            "device": self.config.embedding_device,
            "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown')
        }

class FAISSVectorStore:
    """Vector store usando FAISS para búsqueda eficiente"""
    
    def __init__(self, dimension: int = 384):
        self.config = get_vector_store_config()
        self.logger = get_logger("faiss_vectorstore")
        self.dimension = dimension
        self.index = None
        self.documents = []  # Lista de DocumentChunk
        self.index_path = Path(self.config.faiss_path)
        
        if not HAS_FAISS:
            self.logger.error("FAISS no instalado. Instala con: pip install faiss-cpu")
            return
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Inicializar índice FAISS"""
        try:
            # Crear directorio si no existe
            self.index_path.mkdir(parents=True, exist_ok=True)
            
            # Intentar cargar índice existente
            if self._load_existing_index():
                self.logger.info("Índice FAISS cargado desde disco")
                return
            
            # Crear nuevo índice
            if self.config.faiss_index_type == "IndexFlatL2":
                self.index = faiss.IndexFlatL2(self.dimension)
            elif self.config.faiss_index_type == "IndexFlatIP":
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                # Fallback a L2
                self.index = faiss.IndexFlatL2(self.dimension)
            
            self.logger.info("Nuevo índice FAISS creado",
                           index_type=self.config.faiss_index_type,
                           dimension=self.dimension)
            
        except Exception as e:
            self.logger.error("Error inicializando FAISS", error=str(e))
            self.index = None
    
    def _load_existing_index(self) -> bool:
        """Cargar índice existente desde disco"""
        index_file = self.index_path / "index.faiss"
        metadata_file = self.index_path / "metadata.pkl"
        
        if not (index_file.exists() and metadata_file.exists()):
            return False
        
        try:
            # Cargar índice FAISS
            self.index = faiss.read_index(str(index_file))
            
            # Cargar metadatos
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                self.documents = metadata.get('documents', [])
            
            self.logger.info("Índice FAISS cargado",
                           total_vectors=self.index.ntotal,
                           total_documents=len(self.documents))
            
            return True
            
        except Exception as e:
            self.logger.warning("Error cargando índice existente", error=str(e))
            return False
    
    def _save_index(self):
        """Guardar índice en disco"""
        try:
            index_file = self.index_path / "index.faiss"
            metadata_file = self.index_path / "metadata.pkl"
            
            # Guardar índice FAISS
            faiss.write_index(self.index, str(index_file))
            
            # Guardar metadatos
            metadata = {
                'documents': self.documents,
                'dimension': self.dimension,
                'created_at': time.time()
            }
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            self.logger.debug("Índice FAISS guardado", path=str(self.index_path))
            
        except Exception as e:
            self.logger.error("Error guardando índice", error=str(e))
    
    def is_available(self) -> bool:
        """Verificar si el vector store está disponible"""
        return self.index is not None
    
    def add_documents(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Añadir documentos al índice"""
        if not self.is_available():
            raise RuntimeError("Vector store FAISS no disponible")
        
        if len(chunks) != len(embeddings):
            raise ValueError("Número de chunks y embeddings no coincide")
        
        try:
            # Normalizar embeddings si está configurado
            if self.config.faiss_normalize_vectors:
                faiss.normalize_L2(embeddings)
            
            # Añadir al índice
            self.index.add(embeddings.astype(np.float32))
            
            # Añadir chunks con embeddings
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding.tolist()
                self.documents.append(chunk)
            
            # Guardar cambios
            self._save_index()
            
            self.logger.info("Documentos añadidos al índice FAISS",
                           new_documents=len(chunks),
                           total_documents=len(self.documents),
                           total_vectors=self.index.ntotal)
            
        except Exception as e:
            self.logger.error("Error añadiendo documentos", error=str(e))
            raise
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Buscar documentos similares"""
        if not self.is_available():
            raise RuntimeError("Vector store FAISS no disponible")
        
        if self.index.ntotal == 0:
            self.logger.warning("Índice FAISS vacío")
            return []
        
        try:
            # Preparar query
            query_vector = query_embedding.reshape(1, -1).astype(np.float32)
            
            if self.config.faiss_normalize_vectors:
                faiss.normalize_L2(query_vector)
            
            # Buscar
            k = min(k, self.index.ntotal)  # No buscar más de lo disponible
            distances, indices = self.index.search(query_vector, k)
            
            # Preparar resultados
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):  # Verificar índice válido
                    chunk = self.documents[idx]
                    
                    # Convertir distancia a score de similitud
                    if self.config.faiss_index_type == "IndexFlatIP":
                        score = float(distance)  # Inner product, mayor es mejor
                    else:
                        score = float(1.0 / (1.0 + distance))  # L2, menor distancia = mayor similitud
                    
                    results.append((chunk, score))
            
            self.logger.debug("Búsqueda FAISS completada",
                            k_requested=k,
                            results_found=len(results))
            
            return results
            
        except Exception as e:
            self.logger.error("Error en búsqueda FAISS", error=str(e))
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del vector store"""
        return {
            "available": self.is_available(),
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_documents": len(self.documents),
            "dimension": self.dimension,
            "index_type": self.config.faiss_index_type,
            "path": str(self.index_path)
        }
    
    def clear(self):
        """Limpiar todo el índice"""
        if self.is_available():
            self.index.reset()
            self.documents.clear()
            self._save_index()
            self.logger.info("Índice FAISS limpiado")

class ChromaDBVectorStore:
    """Vector store usando ChromaDB como alternativa"""
    
    def __init__(self):
        self.config = get_vector_store_config()
        self.logger = get_logger("chromadb_vectorstore")
        self.client = None
        self.collection = None
        
        if not HAS_CHROMADB:
            self.logger.error("ChromaDB no instalado. Instala con: pip install chromadb")
            return
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Inicializar cliente ChromaDB"""
        try:
            # Crear directorio
            db_path = Path(self.config.chromadb_path)
            db_path.mkdir(parents=True, exist_ok=True)
            
            # Inicializar cliente
            self.client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Obtener o crear colección
            self.collection = self.client.get_or_create_collection(
                name=self.config.chromadb_collection,
                metadata={"hnsw:space": self.config.chromadb_distance_function}
            )
            
            self.logger.info("ChromaDB inicializado",
                           collection=self.config.chromadb_collection,
                           path=str(db_path))
            
        except Exception as e:
            self.logger.error("Error inicializando ChromaDB", error=str(e))
            self.client = None
            self.collection = None
    
    def is_available(self) -> bool:
        """Verificar si está disponible"""
        return self.collection is not None
    
    def add_documents(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Añadir documentos"""
        if not self.is_available():
            raise RuntimeError("ChromaDB no disponible")
        
        try:
            # Preparar datos para ChromaDB
            ids = [chunk.id for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata.to_dict() for chunk in chunks]
            embeddings_list = embeddings.tolist()
            
            # Añadir a la colección
            self.collection.add(
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info("Documentos añadidos a ChromaDB",
                           count=len(chunks))
            
        except Exception as e:
            self.logger.error("Error añadiendo a ChromaDB", error=str(e))
            raise
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Buscar documentos similares"""
        if not self.is_available():
            raise RuntimeError("ChromaDB no disponible")
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k
            )
            
            # Convertir resultados
            search_results = []
            
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    content = results['documents'][0][i]
                    metadata_dict = results['metadatas'][0][i]
                    distance = results['distances'][0][i] if 'distances' in results else 0
                    
                    # Reconstruir chunk
                    metadata = DocumentMetadata.from_dict(metadata_dict)
                    chunk = DocumentChunk(
                        id=doc_id,
                        content=content,
                        metadata=metadata,
                        chunk_index=0,  # Estos valores se pueden mejorar
                        chunk_size=len(content),
                        start_char=0,
                        end_char=len(content)
                    )
                    
                    # Convertir distancia a score
                    score = float(1.0 / (1.0 + distance))
                    search_results.append((chunk, score))
            
            return search_results
            
        except Exception as e:
            self.logger.error("Error búsqueda ChromaDB", error=str(e))
            return []

class RAGService:
    """Servicio principal de RAG (Retrieval-Augmented Generation)"""
    
    def __init__(self):
        self.logger = get_logger("rag_service")
        self.rag_config = get_rag_config()
        self.vector_config = get_vector_store_config()
        
        # Inicializar componentes
        self.embedding_service = EmbeddingService()
        self.vector_store = None
        
        self._initialize_vector_store()
        
        self.logger.info("RAG Service inicializado",
                        enabled=self.rag_config.enabled,
                        vector_store=self.vector_config.default,
                        embedding_available=self.embedding_service.is_available())
    
    def _initialize_vector_store(self):
        """Inicializar vector store según configuración"""
        if self.vector_config.default == "faiss":
            self.vector_store = FAISSVectorStore(
                dimension=self.embedding_service.config.embedding_dimension
            )
        elif self.vector_config.default == "chromadb":
            self.vector_store = ChromaDBVectorStore()
        else:
            self.logger.error("Vector store no soportado", store=self.vector_config.default)
    
    def is_available(self) -> bool:
        """Verificar si RAG está disponible"""
        return (
            self.rag_config.enabled and
            self.embedding_service.is_available() and
            self.vector_store and
            self.vector_store.is_available()
        )
    
    def add_documents_to_index(self, chunks: List[DocumentChunk]) -> bool:
        """Añadir documentos al índice RAG"""
        if not self.is_available():
            self.logger.error("RAG Service no disponible")
            return False
        
        if not chunks:
            self.logger.warning("No hay chunks para indexar")
            return False
        
        try:
            start_time = time.time()
            
            # Extraer textos para generar embeddings
            texts = [chunk.content for chunk in chunks]
            
            # Generar embeddings
            self.logger.info("Generando embeddings para documentos", count=len(chunks))
            embeddings = self.embedding_service.encode_texts(texts)
            
            # Añadir al vector store
            self.vector_store.add_documents(chunks, embeddings)
            
            processing_time = time.time() - start_time
            
            self.logger.info("Documentos indexados exitosamente",
                           chunks_added=len(chunks),
                           processing_time=processing_time)
            
            return True
            
        except Exception as e:
            self.logger.error("Error indexando documentos", error=str(e))
            return False
    
    def search_documents(self, query: str, k: int = None, threshold: float = None) -> List[DocumentChunk]:
        """Buscar documentos relevantes para una consulta"""
        if not self.is_available():
            self.logger.warning("RAG Service no disponible")
            return []
        
        # Usar valores por defecto si no se especifican
        k = k or self.rag_config.k_default
        threshold = threshold or self.rag_config.similarity_threshold
        
        # Limitar k al máximo configurado
        k = min(k, self.rag_config.k_max)
        
        try:
            start_time = time.time()
            
            # Generar embedding de la consulta
            query_embedding = self.embedding_service.encode_single_text(query)
            
            # Buscar en vector store
            results = self.vector_store.search(query_embedding, k=k)
            
            # Filtrar por threshold si se especifica
            filtered_results = []
            for chunk, score in results:
                if score >= threshold:
                    filtered_results.append(chunk)
            
            search_time = time.time() - start_time
            
            self.logger.info("Búsqueda RAG completada",
                           query_length=len(query),
                           k_requested=k,
                           results_found=len(results),
                           results_filtered=len(filtered_results),
                           search_time=search_time)
            
            return filtered_results
            
        except Exception as e:
            self.logger.error("Error en búsqueda RAG", error=str(e))
            return []
    
    def search_with_scores(self, query: str, k: int = None, threshold: float = None) -> List[Tuple[DocumentChunk, float]]:
        """Buscar documentos con scores de similitud"""
        if not self.is_available():
            return []
        
        k = k or self.rag_config.k_default
        threshold = threshold or self.rag_config.similarity_threshold
        k = min(k, self.rag_config.k_max)
        
        try:
            query_embedding = self.embedding_service.encode_single_text(query)
            results = self.vector_store.search(query_embedding, k=k)
            
            # Filtrar por threshold
            filtered_results = [(chunk, score) for chunk, score in results if score >= threshold]
            
            return filtered_results
            
        except Exception as e:
            self.logger.error("Error en búsqueda con scores", error=str(e))
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema RAG"""
        base_stats = {
            "enabled": self.rag_config.enabled,
            "available": self.is_available(),
            "embedding_model": self.embedding_service.config.embedding_name,
            "vector_store_type": self.vector_config.default,
            "chunk_size": self.rag_config.chunk_size,
            "k_default": self.rag_config.k_default,
            "similarity_threshold": self.rag_config.similarity_threshold
        }
        
        # Añadir stats específicos del vector store
        if self.vector_store:
            vector_stats = self.vector_store.get_stats()
            base_stats.update({
                "total_documents": vector_stats.get("total_documents", 0),
                "total_vectors": vector_stats.get("total_vectors", 0),
                "vector_dimension": vector_stats.get("dimension", 0)
            })
        
        # Añadir info del modelo de embeddings
        embedding_info = self.embedding_service.get_model_info()
        base_stats.update({
            "embedding_available": embedding_info.get("available", False),
            "embedding_dimension": embedding_info.get("dimension", 0)
        })
        
        return base_stats
    
    def clear_index(self) -> bool:
        """Limpiar todo el índice"""
        if self.vector_store:
            try:
                self.vector_store.clear()
                self.logger.info("Índice RAG limpiado")
                return True
            except Exception as e:
                self.logger.error("Error limpiando índice", error=str(e))
                return False
        return False

# Instancia global del servicio RAG
rag_service = RAGService()

# Funciones de conveniencia
def search_documents(query: str, k: int = None, threshold: float = None) -> List[DocumentChunk]:
    """Función de conveniencia para búsqueda de documentos"""
    return rag_service.search_documents(query, k, threshold)

def add_documents_to_rag(chunks: List[DocumentChunk]) -> bool:
    """Función de conveniencia para añadir documentos"""
    return rag_service.add_documents_to_index(chunks)

def get_rag_stats() -> Dict[str, Any]:
    """Función de conveniencia para estadísticas"""
    return rag_service.get_stats()

def is_rag_available() -> bool:
    """Función de conveniencia para verificar disponibilidad"""
    return rag_service.is_available()