"""
Vector Store basado en FAISS para búsqueda semántica eficiente
TFM Vicente Caruncho - Sistemas Inteligentes
"""

import numpy as np
import pickle
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Imports FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  FAISS no disponible. Instalar con: pip install faiss-cpu")

from app.core.config import get_vector_store_config
from app.core.logger import get_logger
from app.models import DocumentChunk

class FaissVectorStore:
    """Vector Store usando FAISS para búsqueda semántica"""
    
    def __init__(self, dimension: int = 384):
        self.logger = get_logger("prototipo_chatbot.faiss_vector_store")
        self.config = get_vector_store_config()
        self.dimension = dimension
        self.index = None
        self.documents = []  # Lista de DocumentChunk
        self.document_map = {}  # ID -> DocumentChunk
        self.embedding_service = None
        
        self._initialize()
    
    def _initialize(self):
        """Inicializar FAISS y embedding service"""
        if not FAISS_AVAILABLE:
            self.logger.error("FAISS no disponible")
            return
        
        try:
            # Inicializar embedding service
            from app.services.rag.embeddings import embedding_service
            self.embedding_service = embedding_service
            
            if self.embedding_service.is_available():
                self.dimension = self.embedding_service.get_dimension()
            
            # Crear índice FAISS
            self.index = faiss.IndexFlatL2(self.dimension)
            
            # Intentar cargar índice existente
            self._load_existing_index()
            
            self.logger.info(
                "FaissVectorStore inicializado",
                dimension=self.dimension,
                index_type="IndexFlatL2",
                store_path=self.config.faiss_index_path,
                vectors=self.index.ntotal if self.index else 0
            )
            
        except Exception as e:
            self.logger.error(f"Error inicializando FAISS: {e}")
    
    def _load_existing_index(self):
        """Cargar índice existente si existe"""
        try:
            index_path = Path(self.config.faiss_index_path)
            
            if index_path.exists():
                # Cargar índice FAISS
                index_file = index_path / "index.faiss"
                if index_file.exists():
                    self.index = faiss.read_index(str(index_file))
                    self.logger.info(f"Índice FAISS cargado: {self.index.ntotal} vectores")
                
                # Cargar metadatos
                metadata_file = index_path / "metadata.pkl"
                if metadata_file.exists():
                    with open(metadata_file, 'rb') as f:
                        data = pickle.load(f)
                        self.documents = data.get('documents', [])
                        self.document_map = data.get('document_map', {})
                    self.logger.info(f"Metadatos cargados: {len(self.documents)} documentos")
                    
        except Exception as e:
            self.logger.warning(f"No se pudo cargar índice existente: {e}")
    
    def _save_index(self):
        """Guardar índice y metadatos"""
        try:
            index_path = Path(self.config.faiss_index_path)
            index_path.mkdir(parents=True, exist_ok=True)
            
            # Guardar índice FAISS
            if self.index and self.index.ntotal > 0:
                index_file = index_path / "index.faiss"
                faiss.write_index(self.index, str(index_file))
            
            # Guardar metadatos
            metadata_file = index_path / "metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'document_map': self.document_map
                }, f)
            
            self.logger.debug("Índice y metadatos guardados")
            
        except Exception as e:
            self.logger.error(f"Error guardando índice: {e}")
    
    def is_available(self) -> bool:
        """Verificar disponibilidad"""
        return (
            FAISS_AVAILABLE and 
            self.index is not None and 
            self.embedding_service is not None and
            self.embedding_service.is_available()
        )
    
    def add(self, chunk: DocumentChunk) -> bool:
        """Añadir documento al índice"""
        if not self.is_available():
            self.logger.warning("Vector store no disponible")
            return False
        
        try:
            # Generar embedding si no existe
            if chunk.embedding is None:
                embedding = self.embedding_service.encode(chunk.content)
                if embedding is None:
                    self.logger.warning(f"No se pudo generar embedding para chunk {chunk.id}")
                    return False
                chunk.embedding = embedding.tolist()
            
            # Convertir embedding a numpy array
            embedding_array = np.array(chunk.embedding, dtype=np.float32)
            embedding_array = embedding_array.reshape(1, -1)
            
            # Verificar dimensión
            if embedding_array.shape[1] != self.dimension:
                self.logger.error(
                    f"Dimensión incorrecta: esperada {self.dimension}, recibida {embedding_array.shape[1]}"
                )
                return False
            
            # Añadir al índice FAISS
            self.index.add(embedding_array)
            
            # Añadir a estructuras internas
            self.documents.append(chunk)
            self.document_map[chunk.id] = chunk
            
            # Guardar cambios
            self._save_index()
            
            self.logger.debug(
                "Documento añadido al índice",
                chunk_id=chunk.id,
                total_vectors=self.index.ntotal,
                content_preview=chunk.content[:100]
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error añadiendo documento: {e}")
            return False
    
    def add_batch(self, chunks: List[DocumentChunk]) -> int:
        """Añadir múltiples documentos en batch"""
        if not self.is_available():
            return 0
        
        successful = 0
        
        try:
            # Preparar embeddings
            embeddings_to_generate = []
            chunks_to_process = []
            
            for chunk in chunks:
                if chunk.embedding is None:
                    embeddings_to_generate.append(chunk.content)
                    chunks_to_process.append(chunk)
                else:
                    chunks_to_process.append(chunk)
            
            # Generar embeddings en batch si es necesario
            if embeddings_to_generate:
                new_embeddings = self.embedding_service.encode(embeddings_to_generate)
                
                # Asignar embeddings a chunks
                embedding_idx = 0
                for chunk in chunks_to_process:
                    if chunk.embedding is None:
                        if embedding_idx < len(new_embeddings):
                            chunk.embedding = new_embeddings[embedding_idx].tolist()
                            embedding_idx += 1
            
            # Añadir todos los chunks
            embeddings_matrix = []
            valid_chunks = []
            
            for chunk in chunks_to_process:
                if chunk.embedding is not None:
                    embeddings_matrix.append(chunk.embedding)
                    valid_chunks.append(chunk)
            
            if embeddings_matrix:
                # Convertir a numpy array
                embeddings_array = np.array(embeddings_matrix, dtype=np.float32)
                
                # Añadir al índice
                self.index.add(embeddings_array)
                
                # Añadir a estructuras internas
                self.documents.extend(valid_chunks)
                for chunk in valid_chunks:
                    self.document_map[chunk.id] = chunk
                
                successful = len(valid_chunks)
                
                # Guardar cambios
                self._save_index()
                
                self.logger.info(
                    "Batch añadido al índice",
                    chunks_processed=successful,
                    total_vectors=self.index.ntotal
                )
            
        except Exception as e:
            self.logger.error(f"Error en batch insert: {e}")
        
        return successful
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: float = 0.0
    ) -> List[DocumentChunk]:
        """Buscar documentos similares"""
        if not self.is_available() or len(self.documents) == 0:
            self.logger.warning("Índice FAISS vacío o no disponible")
            return []
        
        try:
            # Preparar embedding de consulta
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding, dtype=np.float32)
            
            query_embedding = query_embedding.reshape(1, -1)
            
            # Verificar dimensión
            if query_embedding.shape[1] != self.dimension:
                self.logger.error(f"Dimensión de consulta incorrecta: {query_embedding.shape[1]}")
                return []
            
            # Buscar en FAISS
            k_search = min(k, len(self.documents))
            distances, indices = self.index.search(query_embedding, k_search)
            
            # Crear resultados con puntuaciones
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.documents):
                    chunk = self.documents[idx].copy()  # Crear copia para no modificar original
                    
                    # Convertir distancia L2 a score de similitud
                    similarity_score = float(1.0 / (1.0 + dist))
                    chunk.relevance_score = similarity_score
                    
                    # Filtrar por threshold
                    if similarity_score >= threshold:
                        results.append(chunk)
            
            self.logger.debug(
                "Búsqueda completada",
                results_found=len(results),
                k_requested=k,
                threshold=threshold
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error en búsqueda: {e}")
            return []
    
    def search_by_text(
        self,
        query_text: str,
        k: int = 5,
        threshold: float = 0.0
    ) -> List[DocumentChunk]:
        """Buscar usando texto (genera embedding automáticamente)"""
        if not self.is_available():
            return []
        
        try:
            # Generar embedding de la consulta
            query_embedding = self.embedding_service.encode(query_text)
            if query_embedding is None:
                return []
            
            return self.search(query_embedding, k=k, threshold=threshold)
            
        except Exception as e:
            self.logger.error(f"Error en búsqueda por texto: {e}")
            return []
    
    def get_document_count(self) -> int:
        """Obtener número de documentos indexados"""
        return len(self.documents)
    
    def get_by_id(self, document_id: str) -> Optional[DocumentChunk]:
        """Obtener documento por ID"""
        return self.document_map.get(document_id)
    
    def delete_by_id(self, document_id: str) -> bool:
        """Eliminar documento por ID"""
        # FAISS no soporta eliminación directa, requiere reconstruir índice
        if document_id not in self.document_map:
            return False
        
        try:
            # Eliminar de estructuras internas
            chunk_to_remove = self.document_map[document_id]
            self.documents = [doc for doc in self.documents if doc.id != document_id]
            del self.document_map[document_id]
            
            # Reconstruir índice FAISS
            self._rebuild_index()
            
            self.logger.info(f"Documento eliminado: {document_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error eliminando documento: {e}")
            return False
    
    def _rebuild_index(self):
        """Reconstruir índice FAISS"""
        try:
            # Crear nuevo índice
            self.index = faiss.IndexFlatL2(self.dimension)
            
            if self.documents:
                # Obtener embeddings
                embeddings = []
                for doc in self.documents:
                    if doc.embedding:
                        embeddings.append(doc.embedding)
                
                if embeddings:
                    embeddings_array = np.array(embeddings, dtype=np.float32)
                    self.index.add(embeddings_array)
            
            # Guardar índice reconstruido
            self._save_index()
            
            self.logger.info(f"Índice reconstruido: {self.index.ntotal} vectores")
            
        except Exception as e:
            self.logger.error(f"Error reconstruyendo índice: {e}")
    
    def clear(self):
        """Limpiar todos los documentos"""
        try:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = []
            self.document_map = {}
            self._save_index()
            
            self.logger.info("Vector store limpiado")
            
        except Exception as e:
            self.logger.error(f"Error limpiando vector store: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del vector store"""
        return {
            'type': 'faiss',
            'available': self.is_available(),
            'documents': len(self.documents),
            'vectors_indexed': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'index_type': 'IndexFlatL2',
            'storage_path': str(self.config.faiss_index_path)
        }

# Instancia global
faiss_vector_store = FaissVectorStore()

__all__ = ['FaissVectorStore', 'faiss_vector_store']
