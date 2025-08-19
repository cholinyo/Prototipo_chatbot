"""
RAG Pipeline - Sistema completo integrado
TFM Vicente Caruncho - Sistemas Inteligentes

Pipeline completo: Ingesta → Embeddings → Vector Stores → LLM → Respuesta
"""

# Imports corregidos para usar archivos existentes
try:
    from app.services.rag.embeddings import get_embedding_service
    from app.services.rag.faiss_store import get_faiss_store  
    from app.services.rag.chromadb_store import get_chromadb_store
    from app.services.llm.llm_services import get_llm_service
    from app.services.document_ingestion_service import document_ingestion_service

    SERVICES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Algunos servicios no disponibles: {e}")
    SERVICES_AVAILABLE = False

import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# ✅ LÍNEA CORREGIDA:
from app.services.ingestion import ingestion_service
from app.core.logger import get_logger

class RAGPipeline:
    """Pipeline RAG completo que integra todos los servicios"""
    
    def __init__(self):
        self.logger = get_logger("rag_pipeline")
        
        # Inicializar todos los servicios
        self._initialize_services()
    
    def _initialize_services(self):
        """Inicializar todos los servicios del pipeline"""
        
        # LLM Service (tu implementación original)
        try:
            from app.services.llm.llm_services import get_llm_service
            self.llm_service = get_llm_service()
            self.logger.info("LLM Service inicializado")
        except ImportError:
            self.llm_service = None
            self.logger.warning("LLM Service no disponible")
        
        # Ingestion Service
        try:
            self.ingestion_service = ingestion_service
            self.logger.info("Ingestion Service inicializado")
        except ImportError:
            self.ingestion_service = None
            self.logger.warning("Ingestion Service no disponible")

        
        # Embedding Service
        try:
            from app.services.rag.embeddings import embedding_service
            self.embedding_service = embedding_service
            self.logger.info("Embedding Service inicializado")
        except ImportError:
            self.embedding_service = None
            self.logger.warning("Embedding Service no disponible")
        
        # FAISS Vector Store
        try:
            from app.services.rag.faiss_store import faiss_store
            self.faiss_store = faiss_store
            self.logger.info("FAISS Store inicializado")
        except ImportError:
            self.faiss_store = None
            self.logger.warning("FAISS Store no disponible")
        
        # ChromaDB Vector Store
        try:
            from app.services.rag.chromadb_store import chromadb_store
            self.chromadb_store = chromadb_store
            self.logger.info("ChromaDB Store inicializado")
        except ImportError:
            self.chromadb_store = None
            self.logger.warning("ChromaDB Store no disponible")
    
    def health_check(self) -> Dict[str, Any]:
        """Health check completo del pipeline"""
        components = {}
        
        # Check LLM service (tu código original)
        if self.llm_service:
            try:
                llm_health = self.llm_service.health_check()
                components['llm'] = llm_health.get('status', 'unknown')
            except Exception as e:
                components['llm'] = f'error: {str(e)}'
        else:
            components['llm'] = 'not_available'
        
        # Check servicios adicionales
        components['ingestion'] = 'available' if (self.ingestion_service and self.ingestion_service.is_available()) else 'not_available'
        components['embeddings'] = 'available' if (self.embedding_service and self.embedding_service.is_available()) else 'not_available'
        components['faiss'] = 'available' if (self.faiss_store and self.faiss_store.is_available()) else 'not_available'
        components['chromadb'] = 'available' if (self.chromadb_store and self.chromadb_store.is_available()) else 'not_available'
        
        # Determinar estado general
        available_count = sum(1 for status in components.values() if status == 'available')
        total_services = len(components)
        
        if available_count == 0:
            status = 'critical'
        elif available_count < total_services * 0.5:
            status = 'degraded'
        elif available_count < total_services:
            status = 'partial'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'timestamp': time.time(),
            'components': components,
            'availability_rate': available_count / total_services,
            'pipeline_type': 'complete_rag'
        }
    
    def is_available(self) -> bool:
        """Verificar disponibilidad del pipeline"""
        # Pipeline está disponible si al menos LLM + un vector store funcionan
        llm_ok = self.llm_service and self.llm_service.is_available()
        vector_ok = ((self.faiss_store and self.faiss_store.is_available()) or 
                     (self.chromadb_store and self.chromadb_store.is_available()))
        
        return llm_ok or vector_ok  # Al menos uno debe funcionar
    
    # =========================================================================
    # FUNCIONALIDADES DE INGESTA
    # =========================================================================
    
    def ingest_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Ingestar documentos desde un directorio completo
        Pipeline: Documentos → Chunks → Embeddings → Vector Stores
        """
        if not self.ingestion_service or not self.embedding_service:
            return {
                'success': False,
                'error': 'Servicios de ingesta o embeddings no disponibles'
            }
        
        start_time = time.time()
        result = {
            'success': False,
            'directory': directory_path,
            'files_processed': 0,
            'chunks_generated': 0,
            'faiss_indexed': 0,
            'chromadb_indexed': 0,
            'processing_time': 0,
            'errors': []
        }
        
        try:
            # Paso 1: Encontrar archivos
            directory = Path(directory_path)
            if not directory.exists():
                result['errors'].append(f"Directorio no encontrado: {directory_path}")
                return result
            
            supported_extensions = self.document_processor.get_supported_extensions()
            files_found = []
            for ext in supported_extensions:
                files_found.extend(directory.glob(f"*{ext}"))
            
            if not files_found:
                result['errors'].append("No se encontraron archivos soportados")
                return result
            
            # Paso 2: Procesar archivos
            all_chunks = []
            for file_path in files_found:
                try:
                    chunks = self.document_processor.process_file(str(file_path))
                    if chunks:
                        all_chunks.extend(chunks)
                        result['files_processed'] += 1
                    else:
                        result['errors'].append(f"Error procesando {file_path.name}")
                except Exception as e:
                    result['errors'].append(f"Error en {file_path.name}: {str(e)}")
            
            result['chunks_generated'] = len(all_chunks)
            
            if not all_chunks:
                result['errors'].append("No se generaron chunks")
                return result
            
            # Paso 3: Generar embeddings
            texts = [chunk.content for chunk in all_chunks]
            embeddings = self.embedding_service.encode_batch(texts)
            
            if not embeddings:
                result['errors'].append("Error generando embeddings")
                return result
            
            # Asignar embeddings a chunks
            for chunk, embedding in zip(all_chunks, embeddings):
                chunk.embedding = embedding
            
            # Paso 4: Indexar en vector stores
            if self.faiss_store and self.faiss_store.is_available():
                try:
                    if self.faiss_store.add_documents(all_chunks):
                        result['faiss_indexed'] = len(all_chunks)
                except Exception as e:
                    result['errors'].append(f"Error indexando en FAISS: {str(e)}")
            
            if self.chromadb_store and self.chromadb_store.is_available():
                try:
                    if self.chromadb_store.add_documents(all_chunks):
                        result['chromadb_indexed'] = len(all_chunks)
                except Exception as e:
                    result['errors'].append(f"Error indexando en ChromaDB: {str(e)}")
            
            result['processing_time'] = time.time() - start_time
            result['success'] = result['faiss_indexed'] > 0 or result['chromadb_indexed'] > 0
            
            self.logger.info(
                "Ingesta de directorio completada",
                directory=directory_path,
                files=result['files_processed'],
                chunks=result['chunks_generated'],
                faiss=result['faiss_indexed'],
                chromadb=result['chromadb_indexed'],
                time=result['processing_time']
            )
            
            return result
            
        except Exception as e:
            result['errors'].append(f"Error general: {str(e)}")
            result['processing_time'] = time.time() - start_time
            return result
    
    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """Ingestar un archivo individual"""
        if not self.document_processor:
            return {'success': False, 'error': 'Document processor no disponible'}
        
        try:
            # Procesar archivo
            chunks = self.document_processor.process_file(file_path)
            
            if not chunks:
                return {'success': False, 'error': 'No se generaron chunks'}
            
            # Generar embeddings
            if self.embedding_service:
                texts = [chunk.content for chunk in chunks]
                embeddings = self.embedding_service.encode_batch(texts)
                
                for chunk, embedding in zip(chunks, embeddings):
                    chunk.embedding = embedding
            
            # Indexar
            faiss_success = False
            chromadb_success = False
            
            if self.faiss_store and self.faiss_store.is_available():
                faiss_success = self.faiss_store.add_documents(chunks)
            
            if self.chromadb_store and self.chromadb_store.is_available():
                chromadb_success = self.chromadb_store.add_documents(chunks)
            
            return {
                'success': faiss_success or chromadb_success,
                'chunks_generated': len(chunks),
                'faiss_indexed': faiss_success,
                'chromadb_indexed': chromadb_success
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # =========================================================================
    # FUNCIONALIDADES DE BÚSQUEDA
    # =========================================================================
    
    def search_documents(self, query: str, k: int = 5, use_faiss: bool = True) -> List[Dict[str, Any]]:
        """
        Buscar documentos usando vector stores
        """
        if not self.embedding_service:
            return []
        
        try:
            # Generar embedding de la query
            query_embedding = self.embedding_service.encode_single_text(query)
            
            if query_embedding is None:
                return []
            
            results = []
            
            # Buscar en FAISS si está disponible y se solicita
            if use_faiss and self.faiss_store and self.faiss_store.is_available():
                try:
                    faiss_results = self.faiss_store.search(query_embedding, k=k)
                    
                    for chunk, score in faiss_results:
                        results.append({
                            'content': chunk.content,
                            'source': chunk.metadata.source_path,
                            'score': float(score),
                            'chunk_id': chunk.id,
                            'vector_store': 'faiss'
                        })
                        
                except Exception as e:
                    self.logger.error(f"Error búsqueda FAISS: {e}")
            
            # Buscar en ChromaDB si FAISS no está disponible o no se solicita
            elif self.chromadb_store and self.chromadb_store.is_available():
                try:
                    chromadb_results = self.chromadb_store.search(query_embedding, k=k)
                    
                    for chunk in chromadb_results:
                        results.append({
                            'content': chunk.content,
                            'source': chunk.metadata.source_path,
                            'chunk_id': chunk.id,
                            'vector_store': 'chromadb'
                        })
                        
                except Exception as e:
                    self.logger.error(f"Error búsqueda ChromaDB: {e}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error en búsqueda: {e}")
            return []
    
    def generate_rag_response(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Generar respuesta RAG completa: Búsqueda + LLM
        """
        if not self.llm_service:
            return {
                'success': False,
                'error': 'LLM service no disponible'
            }
        
        start_time = time.time()
        
        try:
            # Paso 1: Buscar documentos relevantes
            relevant_docs = self.search_documents(query, k=k)
            
            if not relevant_docs:
                return {
                    'success': False,
                    'error': 'No se encontraron documentos relevantes'
                }
            
            # Paso 2: Preparar contexto
            context_parts = []
            sources = []
            
            for i, doc in enumerate(relevant_docs):
                context_parts.append(f"[{i+1}] {doc['content']}")
                sources.append({
                    'index': i+1,
                    'source': doc['source'],
                    'chunk_id': doc.get('chunk_id', ''),
                    'score': doc.get('score'),
                    'vector_store': doc.get('vector_store', 'unknown')
                })
            
            context = "\n\n".join(context_parts)
            
            # Paso 3: Generar respuesta con LLM
            llm_response = self.llm_service.generate_response(
                query=query,
                context=context
            )
            
            response_time = time.time() - start_time
            
            return {
                'success': True,
                'query': query,
                'response': llm_response.get('response', ''),
                'sources': sources,
                'context_chunks': len(relevant_docs),
                'response_time': response_time,
                'llm_model': llm_response.get('model', 'unknown')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    # =========================================================================
    # UTILIDADES Y ESTADÍSTICAS
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas completas del pipeline"""
        stats = {
            'pipeline_status': self.health_check(),
            'services': {}
        }
        
        # Stats de cada servicio
        if self.embedding_service and self.embedding_service.is_available():
            stats['services']['embeddings'] = self.embedding_service.get_stats()
        
        if self.faiss_store and self.faiss_store.is_available():
            stats['services']['faiss'] = self.faiss_store.get_stats()
        
        if self.chromadb_store and self.chromadb_store.is_available():
            stats['services']['chromadb'] = self.chromadb_store.get_stats()
        
        if self.ingestion_service and self.ingestion_service.is_available():
            stats['services']['ingestion'] = self.ingestion_service.get_service_stats()
        
        return stats
    
    def clear_vector_stores(self) -> Dict[str, bool]:
        """Limpiar todos los vector stores"""
        results = {}
        
        if self.faiss_store and self.faiss_store.is_available():
            try:
                self.faiss_store.clear()
                results['faiss'] = True
            except Exception as e:
                results['faiss'] = False
                self.logger.error(f"Error limpiando FAISS: {e}")
        
        if self.chromadb_store and self.chromadb_store.is_available():
            try:
                self.chromadb_store.clear()
                results['chromadb'] = True
            except Exception as e:
                results['chromadb'] = False
                self.logger.error(f"Error limpiando ChromaDB: {e}")
        
        return results

# Instancia global (mantener tu código original)
rag_pipeline = RAGPipeline()

def get_rag_pipeline() -> RAGPipeline:
    """Obtener instancia del pipeline"""
    return rag_pipeline

# Funciones de conveniencia adicionales
def ingest_documents(directory_path: str) -> Dict[str, Any]:
    """Función de conveniencia para ingesta"""
    return rag_pipeline.ingest_directory(directory_path)

def search_docs(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Función de conveniencia para búsqueda"""
    return rag_pipeline.search_documents(query, k=k)

def generate_response(query: str, k: int = 5) -> Dict[str, Any]:
    """Función de conveniencia para respuesta RAG completa"""
    return rag_pipeline.generate_rag_response(query, k=k)