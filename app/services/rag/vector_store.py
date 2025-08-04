"""
Vector Store Implementations - FAISS y ChromaDB
Prototipo_chatbot - TFM Vicente Caruncho
"""

import os
import time
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

# Configurar logging básico si no está disponible
logger = logging.getLogger("vector_store")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# =============================================================================
# CLASES BASE (simplificadas para desarrollo)
# =============================================================================

class DocumentMetadata:
    """Metadatos básicos de documento"""
    def __init__(self, source_path: str, source_type: str, file_type: str, 
                 size_bytes: int, created_at=None, processed_at=None, checksum=""):
        self.source_path = source_path
        self.source_type = source_type
        self.file_type = file_type
        self.size_bytes = size_bytes
        self.created_at = created_at
        self.processed_at = processed_at
        self.checksum = checksum

class DocumentChunk:
    """Fragmento de documento para RAG"""
    def __init__(self, id: str, content: str, metadata: DocumentMetadata,
                 chunk_index: int, chunk_size: int, start_char: int, end_char: int,
                 embedding: Optional[List[float]] = None, section_title: Optional[str] = None):
        self.id = id
        self.content = content
        self.metadata = metadata
        self.chunk_index = chunk_index
        self.chunk_size = chunk_size
        self.start_char = start_char
        self.end_char = end_char
        self.embedding = embedding
        self.section_title = section_title

class SearchResult:
    """Resultado de búsqueda vectorial"""
    def __init__(self, chunk: DocumentChunk, score: float, distance: float, metadata: Dict[str, Any]):
        self.chunk = chunk
        self.score = score
        self.distance = distance
        self.metadata = metadata

# =============================================================================
# INTERFAZ BASE
# =============================================================================

class VectorStoreInterface:
    """Interfaz base para vector stores"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_initialized = False
        self.document_count = 0
        
    def get_store_type(self) -> str:
        raise NotImplementedError
    
    def initialize(self) -> bool:
        raise NotImplementedError
    
    def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        raise NotImplementedError
    
    def search(self, query_embedding: List[float], k: int = 5, threshold: float = 0.7) -> List[SearchResult]:
        raise NotImplementedError
    
    def get_document_count(self) -> int:
        return self.document_count
    
    def get_memory_usage(self) -> float:
        # Estimación básica
        return self.document_count * 0.001  # 1KB por documento estimado

# =============================================================================
# IMPLEMENTACIÓN FAISS
# =============================================================================

class FAISSVectorStore(VectorStoreInterface):
    """Implementación FAISS para vector store"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.index = None
        self.documents = {}  # id -> DocumentChunk
        self.embeddings_dim = config.get('embedding_dimension', 384)
        self.index_path = config.get('faiss_path', 'data/vectorstore/faiss')
        self.normalize = config.get('normalize_vectors', True)
        
    def get_store_type(self) -> str:
        return "faiss"
    
    def initialize(self) -> bool:
        """Inicializar FAISS"""
        try:
            import faiss
            
            # Crear directorio si no existe
            Path(self.index_path).mkdir(parents=True, exist_ok=True)
            
            # Crear índice FAISS
            if self.normalize:
                self.index = faiss.IndexFlatIP(self.embeddings_dim)  # Inner Product (cosine con normalización)
            else:
                self.index = faiss.IndexFlatL2(self.embeddings_dim)  # L2 distance
            
            # Intentar cargar índice existente
            index_file = Path(self.index_path) / "faiss.index"
            docs_file = Path(self.index_path) / "documents.pkl"
            
            if index_file.exists() and docs_file.exists():
                try:
                    self.index = faiss.read_index(str(index_file))
                    with open(docs_file, 'rb') as f:
                        self.documents = pickle.load(f)
                    self.document_count = len(self.documents)
                    logger.info(f"FAISS: Índice cargado con {self.document_count} documentos")
                except Exception as e:
                    logger.warning(f"FAISS: Error cargando índice existente: {e}")
                    # Recrear índice vacío
                    if self.normalize:
                        self.index = faiss.IndexFlatIP(self.embeddings_dim)
                    else:
                        self.index = faiss.IndexFlatL2(self.embeddings_dim)
            
            self.is_initialized = True
            logger.info(f"FAISS inicializado: dimensión={self.embeddings_dim}, normalize={self.normalize}")
            return True
            
        except ImportError:
            logger.error("FAISS no está instalado. Instalar con: pip install faiss-cpu")
            return False
        except Exception as e:
            logger.error(f"Error inicializando FAISS: {e}")
            return False
    
    def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Añadir documentos a FAISS"""
        if not self.is_initialized:
            logger.error("FAISS no está inicializado")
            return False
        
        try:
            import faiss
            
            # Preparar embeddings
            embeddings = []
            valid_chunks = []
            
            for chunk in chunks:
                if chunk.embedding is not None:
                    embeddings.append(chunk.embedding)
                    valid_chunks.append(chunk)
                    self.documents[chunk.id] = chunk
                else:
                    logger.warning(f"Chunk {chunk.id} no tiene embedding, se omite")
            
            if not embeddings:
                logger.warning("No hay embeddings válidos para añadir")
                return False
            
            # Convertir a numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Normalizar si es necesario
            if self.normalize:
                faiss.normalize_L2(embeddings_array)
            
            # Añadir al índice
            self.index.add(embeddings_array)
            self.document_count += len(valid_chunks)
            
            # Guardar índice
            self._save_index()
            
            logger.info(f"FAISS: Añadidos {len(valid_chunks)} documentos. Total: {self.document_count}")
            return True
            
        except Exception as e:
            logger.error(f"Error añadiendo documentos a FAISS: {e}")
            return False
    
    def search(self, query_embedding: List[float], k: int = 5, threshold: float = 0.7) -> List[SearchResult]:
        """Buscar en FAISS"""
        if not self.is_initialized or self.index.ntotal == 0:
            return []
        
        try:
            import faiss
            
            # Preparar query
            query_array = np.array([query_embedding], dtype=np.float32)
            
            if self.normalize:
                faiss.normalize_L2(query_array)
            
            # Buscar
            distances, indices = self.index.search(query_array, min(k, self.index.ntotal))
            
            # Convertir resultados
            results = []
            doc_ids = list(self.documents.keys())
            
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # No hay más resultados
                    break
                
                # Convertir distancia a score
                if self.normalize:
                    score = float(distance)  # Inner product ya es similitud
                else:
                    score = 1.0 / (1.0 + float(distance))  # Convertir L2 a similitud
                
                # Filtrar por threshold
                if score >= threshold:
                    doc_id = doc_ids[idx]
                    chunk = self.documents[doc_id]
                    
                    result = SearchResult(
                        chunk=chunk,
                        score=score,
                        distance=float(distance),
                        metadata={'index': idx, 'rank': i}
                    )
                    results.append(result)
            
            logger.info(f"FAISS: Búsqueda completada, {len(results)} resultados")
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda FAISS: {e}")
            return []
    
    def _save_index(self):
        """Guardar índice FAISS"""
        try:
            import faiss
            
            index_file = Path(self.index_path) / "faiss.index"
            docs_file = Path(self.index_path) / "documents.pkl"
            
            faiss.write_index(self.index, str(index_file))
            
            with open(docs_file, 'wb') as f:
                pickle.dump(self.documents, f)
                
        except Exception as e:
            logger.error(f"Error guardando índice FAISS: {e}")

# =============================================================================
# IMPLEMENTACIÓN CHROMADB
# =============================================================================

class ChromaDBVectorStore(VectorStoreInterface):
    """Implementación ChromaDB para vector store"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = None
        self.collection = None
        self.collection_name = config.get('collection_name', 'prototipo_documents')
        self.persist_path = config.get('chromadb_path', 'data/vectorstore/chromadb')
        self.distance_function = config.get('distance_function', 'cosine')
        
    def get_store_type(self) -> str:
        return "chromadb"
    
    def initialize(self) -> bool:
        """Inicializar ChromaDB"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Crear directorio si no existe
            Path(self.persist_path).mkdir(parents=True, exist_ok=True)
            
            # Crear cliente persistente
            self.client = chromadb.PersistentClient(
                path=self.persist_path,
                settings=Settings(allow_reset=True)
            )
            
            # Crear o obtener colección
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_function}
            )
            
            # Contar documentos existentes
            self.document_count = self.collection.count()
            
            self.is_initialized = True
            logger.info(f"ChromaDB inicializado: colección='{self.collection_name}', documentos={self.document_count}")
            return True
            
        except ImportError:
            logger.error("ChromaDB no está instalado. Instalar con: pip install chromadb")
            return False
        except Exception as e:
            logger.error(f"Error inicializando ChromaDB: {e}")
            return False
    
    def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Añadir documentos a ChromaDB"""
        if not self.is_initialized:
            logger.error("ChromaDB no está inicializado")
            return False
        
        try:
            # Preparar datos
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            
            for chunk in chunks:
                if chunk.embedding is not None:
                    ids.append(chunk.id)
                    documents.append(chunk.content)
                    embeddings.append(chunk.embedding)
                    
                    # Preparar metadatos
                    metadata = {
                        'source_path': chunk.metadata.source_path,
                        'source_type': chunk.metadata.source_type,
                        'file_type': chunk.metadata.file_type,
                        'chunk_index': chunk.chunk_index,
                        'section_title': chunk.section_title or ""
                    }
                    metadatas.append(metadata)
                else:
                    logger.warning(f"Chunk {chunk.id} no tiene embedding, se omite")
            
            if not embeddings:
                logger.warning("No hay embeddings válidos para añadir a ChromaDB")
                return False
            
            # Añadir a la colección
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            self.document_count += len(embeddings)
            
            logger.info(f"ChromaDB: Añadidos {len(embeddings)} documentos. Total: {self.document_count}")
            return True
            
        except Exception as e:
            logger.error(f"Error añadiendo documentos a ChromaDB: {e}")
            return False
    
    def search(self, query_embedding: List[float], k: int = 5, threshold: float = 0.7) -> List[SearchResult]:
        """Buscar en ChromaDB"""
        if not self.is_initialized or self.document_count == 0:
            return []
        
        try:
            # Realizar búsqueda
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self.document_count)
            )
            
            # Convertir resultados
            search_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc_id, document, distance, metadata) in enumerate(zip(
                    results['ids'][0],
                    results['documents'][0], 
                    results['distances'][0],
                    results['metadatas'][0]
                )):
                    # ChromaDB devuelve distancias, convertir a score
                    if self.distance_function == 'cosine':
                        score = 1.0 - float(distance)  # Cosine similarity = 1 - cosine distance
                    else:
                        score = 1.0 / (1.0 + float(distance))  # Para otras distancias
                    
                    # Filtrar por threshold
                    if score >= threshold:
                        # Reconstruir chunk
                        doc_metadata = DocumentMetadata(
                            source_path=metadata.get('source_path', ''),
                            source_type=metadata.get('source_type', ''),
                            file_type=metadata.get('file_type', ''),
                            size_bytes=len(document),
                            checksum=""
                        )
                        
                        chunk = DocumentChunk(
                            id=doc_id,
                            content=document,
                            metadata=doc_metadata,
                            chunk_index=metadata.get('chunk_index', 0),
                            chunk_size=len(document),
                            start_char=0,
                            end_char=len(document),
                            section_title=metadata.get('section_title')
                        )
                        
                        result = SearchResult(
                            chunk=chunk,
                            score=score,
                            distance=float(distance),
                            metadata={'rank': i, 'chroma_metadata': metadata}
                        )
                        search_results.append(result)
            
            logger.info(f"ChromaDB: Búsqueda completada, {len(search_results)} resultados")
            return search_results
            
        except Exception as e:
            logger.error(f"Error en búsqueda ChromaDB: {e}")
            return []

# =============================================================================
# MANAGER PARA COMPARACIÓN
# =============================================================================

class VectorStoreManager:
    """Gestor para comparar múltiples vector stores"""
    
    def __init__(self):
        self.stores = {}
        self.default_store = None
        
    def register_store(self, store: VectorStoreInterface, is_default: bool = False):
        """Registrar un vector store"""
        store_type = store.get_store_type()
        self.stores[store_type] = store
        
        if is_default or self.default_store is None:
            self.default_store = store_type
            
        logger.info(f"Vector store registrado: {store_type} (default: {is_default})")
    
    def initialize_all(self) -> Dict[str, bool]:
        """Inicializar todos los stores"""
        results = {}
        for store_type, store in self.stores.items():
            results[store_type] = store.initialize()
        return results
    
    def compare_search(self, query_embedding: List[float], k: int = 5) -> Dict[str, Any]:
        """Comparar búsqueda en todos los stores"""
        results = {}
        
        for store_type, store in self.stores.items():
            if not store.is_initialized:
                continue
                
            start_time = time.time()
            search_results = store.search(query_embedding, k)
            execution_time = time.time() - start_time
            
            results[store_type] = {
                'results': search_results,
                'execution_time': execution_time,
                'results_count': len(search_results)
            }
        
        return results
    
    def get_store(self, store_type: Optional[str] = None) -> VectorStoreInterface:
        """Obtener store específico"""
        if store_type is None:
            store_type = self.default_store
            
        if store_type not in self.stores:
            raise ValueError(f"Store '{store_type}' no registrado")
        
        return self.stores[store_type]

# Instancia global
vector_store_manager = VectorStoreManager()

# =============================================================================
# FUNCIONES DE PRUEBA
# =============================================================================

def test_vector_stores():
    """Función de prueba para los vector stores"""
    print("🧪 PROBANDO VECTOR STORES...")
    
    # Configuración
    config = {
        'embedding_dimension': 384,
        'faiss_path': 'data/vectorstore/faiss',
        'chromadb_path': 'data/vectorstore/chromadb',
        'collection_name': 'test_collection'
    }
    
    # Crear stores
    faiss_store = FAISSVectorStore(config)
    chroma_store = ChromaDBVectorStore(config)
    
    # Registrar
    vector_store_manager.register_store(faiss_store, is_default=True)
    vector_store_manager.register_store(chroma_store)
    
    # Inicializar
    results = vector_store_manager.initialize_all()
    print(f"Inicialización: {results}")
    
    # Crear datos de prueba
    from datetime import datetime
    
    metadata = DocumentMetadata(
        source_path="test.txt",
        source_type="document", 
        file_type=".txt",
        size_bytes=100,
        created_at=datetime.now(),
        processed_at=datetime.now(),
        checksum="test123"
    )
    
    # Chunk con embedding de prueba (384 dimensiones)
    test_embedding = [0.1] * 384
    
    chunk = DocumentChunk(
        id="test_1",
        content="Este es un documento de prueba para el sistema RAG",
        metadata=metadata,
        chunk_index=0,
        chunk_size=50,
        start_char=0,
        end_char=50,
        embedding=test_embedding
    )
    
    # Añadir documentos
    for store_type, store in vector_store_manager.stores.items():
        if store.is_initialized:
            success = store.add_documents([chunk])
            print(f"{store_type}: Documento añadido = {success}")
    
    # Búsqueda comparativa
    query_embedding = [0.11] * 384  # Ligeramente diferente
    comparison = vector_store_manager.compare_search(query_embedding, k=1)
    
    print("\n📊 RESULTADOS DE COMPARACIÓN:")
    for store_type, result in comparison.items():
        print(f"{store_type}:")
        print(f"  - Tiempo: {result['execution_time']:.4f}s")
        print(f"  - Resultados: {result['results_count']}")
        if result['results']:
            print(f"  - Score: {result['results'][0].score:.4f}")
    
    print("\n✅ Prueba completada!")

if __name__ == "__main__":
    test_vector_stores()