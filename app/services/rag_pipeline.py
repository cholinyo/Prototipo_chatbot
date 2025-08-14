"""
Pipeline RAG Completo para TFM Vicente Caruncho
Conexión end-to-end de todos los componentes implementados
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.core.config import get_config
from app.core.logger import get_logger
from app.models import DocumentChunk, ModelResponse
from app.services.rag.embeddings import get_embedding_service
from app.services.llm_service import LLMService


@dataclass
class RAGResponse:
    """Respuesta completa del pipeline RAG"""
    query: str
    response: str
    model_name: str
    provider: str
    response_time: float
    sources: List[str]
    context_chunks: List[DocumentChunk]
    confidence: float
    estimated_cost: Optional[float] = None
    error: Optional[str] = None


class RAGPipeline:
    """Pipeline RAG completo integrado"""
    
    def __init__(self):
        self.logger = get_logger("prototipo_chatbot.rag_pipeline")
        self.config = get_config()
        
        # Inicializar servicios
        self.embedding_service = None
        self.vector_store = None
        self.llm_service = None
        
        self._initialize_services()
    
    def _initialize_services(self):
        """Inicializar todos los servicios del pipeline"""
        try:
            # 1. Embedding Service
            self.embedding_service = get_embedding_service()
            self.logger.info("EmbeddingService inicializado")
            
            # 2. Vector Store (FAISS o ChromaDB según configuración)
            self._initialize_vector_store()
            
            # 3. LLM Service
            self.llm_service = LLMService()
            
            self.logger.info(
                "Pipeline RAG inicializado",
                embedding_model=self.embedding_service.model_name if self.embedding_service else None,
                vector_store=self.vector_store.__class__.__name__ if self.vector_store else None,
                llm_providers=list(self.llm_service.providers.keys()) if self.llm_service else []
            )
            
        except Exception as e:
            self.logger.error(f"Error inicializando pipeline RAG: {e}")
            raise
    
    def _initialize_vector_store(self):
        """Inicializar vector store según configuración"""
        try:
            vector_store_type = self.config.get('DEFAULT_VECTOR_STORE', 'faiss')
            
            if vector_store_type == 'faiss':
                from app.services.rag.faiss_store import FAISSVectorStore
                self.vector_store = FAISSVectorStore()
                
            elif vector_store_type == 'chromadb':
                from app.services.rag.chromadb_store import ChromaDBVectorStore
                self.vector_store = ChromaDBVectorStore()
                
            else:
                raise ValueError(f"Vector store no soportado: {vector_store_type}")
                
            self.logger.info(f"Vector store {vector_store_type} inicializado")
            
        except Exception as e:
            self.logger.error(f"Error inicializando vector store: {e}")
            raise
    
    def is_available(self) -> bool:
        """Verificar disponibilidad del pipeline completo"""
        return all([
            self.embedding_service is not None,
            self.vector_store is not None,
            self.llm_service is not None and self.llm_service.is_available()
        ])
    
    def process_query(
        self,
        query: str,
        provider: str = 'ollama',
        model: Optional[str] = None,
        top_k: int = 5,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> RAGResponse:
        """
        Procesar consulta completa con pipeline RAG
        
        Args:
            query: Consulta del usuario
            provider: Proveedor LLM ('ollama' o 'openai')
            model: Modelo específico (opcional)
            top_k: Número de documentos relevantes a recuperar
            temperature: Temperatura para generación
            max_tokens: Límite de tokens (opcional)
        
        Returns:
            RAGResponse con la respuesta completa y metadatos
        """
        start_time = time.time()
        
        try:
            # 1. Verificar disponibilidad
            if not self.is_available():
                return RAGResponse(
                    query=query,
                    response="Pipeline RAG no disponible",
                    model_name="none",
                    provider="none",
                    response_time=0.0,
                    sources=[],
                    context_chunks=[],
                    confidence=0.0,
                    error="Pipeline not available"
                )
            
            # 2. Generar embedding de la consulta
            self.logger.info(f"Procesando consulta: {query[:100]}...")
            query_embedding = self.embedding_service.encode_single_text(query)
            
            # 3. Buscar documentos relevantes
            context_chunks = self.vector_store.search(
                query_embedding=query_embedding,
                k=top_k
            )
            
            self.logger.info(
                f"Encontrados {len(context_chunks)} documentos relevantes",
                sources=[chunk.metadata.source_path for chunk in context_chunks[:3]]
            )
            
            # 4. Generar respuesta con LLM
            llm_response = self.llm_service.generate_response(
                query=query,
                context=context_chunks,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 5. Extraer fuentes y calcular confianza
            sources = self._extract_sources(context_chunks)
            confidence = self._calculate_confidence(context_chunks, llm_response)
            
            response_time = time.time() - start_time
            
            # 6. Crear respuesta RAG completa
            rag_response = RAGResponse(
                query=query,
                response=llm_response.content,
                model_name=llm_response.model,
                provider=llm_response.provider,
                response_time=response_time,
                sources=sources,
                context_chunks=context_chunks,
                confidence=confidence,
                estimated_cost=getattr(llm_response, 'estimated_cost', None),
                error=llm_response.error
            )
            
            self.logger.info(
                "Consulta procesada exitosamente",
                response_time=f"{response_time:.2f}s",
                model=llm_response.model,
                provider=llm_response.provider,
                sources_count=len(sources)
            )
            
            return rag_response
            
        except Exception as e:
            error_msg = f"Error procesando consulta: {e}"
            self.logger.error(error_msg)
            
            return RAGResponse(
                query=query,
                response="Error interno del sistema",
                model_name="error",
                provider="error",
                response_time=time.time() - start_time,
                sources=[],
                context_chunks=[],
                confidence=0.0,
                error=error_msg
            )
    
    def compare_providers(
        self,
        query: str,
        top_k: int = 5,
        temperature: float = 0.3
    ) -> Dict[str, RAGResponse]:
        """
        Comparar respuestas entre múltiples proveedores
        
        Args:
            query: Consulta del usuario
            top_k: Número de documentos relevantes
            temperature: Temperatura para generación
        
        Returns:
            Diccionario con respuestas de cada proveedor disponible
        """
        results = {}
        
        available_providers = list(self.llm_service.providers.keys())
        self.logger.info(f"Comparando proveedores: {available_providers}")
        
        for provider in available_providers:
            try:
                response = self.process_query(
                    query=query,
                    provider=provider,
                    top_k=top_k,
                    temperature=temperature
                )
                results[provider] = response
                
            except Exception as e:
                self.logger.error(f"Error con proveedor {provider}: {e}")
                results[provider] = RAGResponse(
                    query=query,
                    response=f"Error con {provider}",
                    model_name="error",
                    provider=provider,
                    response_time=0.0,
                    sources=[],
                    context_chunks=[],
                    confidence=0.0,
                    error=str(e)
                )
        
        return results
    
    def _extract_sources(self, chunks: List[DocumentChunk]) -> List[str]:
        """Extraer fuentes únicas de los chunks"""
        sources = []
        seen = set()
        
        for chunk in chunks:
            source = chunk.metadata.source_path
            if source not in seen:
                sources.append(source)
                seen.add(source)
        
        return sources
    
    def _calculate_confidence(
        self,
        chunks: List[DocumentChunk],
        llm_response: ModelResponse
    ) -> float:
        """
        Calcular confianza de la respuesta basada en:
        - Número de fuentes encontradas
        - Calidad de la respuesta del LLM
        - Ausencia de errores
        """
        confidence = 0.5  # Base
        
        # +0.3 si hay documentos relevantes
        if chunks:
            confidence += 0.3 * min(len(chunks) / 3, 1.0)
        
        # +0.2 si no hay errores
        if not llm_response.error:
            confidence += 0.2
        
        # -0.3 si respuesta muy corta (posible error)
        if len(llm_response.content) < 50:
            confidence -= 0.3
        
        return max(0.0, min(1.0, confidence))
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del pipeline"""
        stats = {
            'pipeline_available': self.is_available(),
            'timestamp': time.time()
        }
        
        if self.embedding_service:
            stats['embeddings'] = {
                'model': self.embedding_service.model_name,
                'cache_size': len(getattr(self.embedding_service, '_cache', {})),
                'cache_hits': getattr(self.embedding_service, '_cache_hits', 0)
            }
        
        if self.vector_store:
            try:
                vs_stats = self.vector_store.get_stats()
                stats['vector_store'] = {
                    'type': self.vector_store.__class__.__name__,
                    'documents': vs_stats.get('total_documents', 0)
                }
            except:
                stats['vector_store'] = {'type': 'unknown', 'documents': 0}
        
        if self.llm_service:
            stats['llm'] = {
                'providers': list(self.llm_service.providers.keys()),
                'available': self.llm_service.is_available()
            }
        
        stats['config'] = {
            'default_vector_store': self.config.get('DEFAULT_VECTOR_STORE', 'faiss'),
            'rag_enabled': True
        }
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Verificación de salud completa del pipeline"""
        health = {
            'status': 'healthy',
            'timestamp': time.time(),
            'components': {}
        }
        
        # Test embedding service
        try:
            test_embedding = self.embedding_service.encode_single_text("test")
            health['components']['embeddings'] = 'healthy'
        except Exception as e:
            health['components']['embeddings'] = f'error: {e}'
            health['status'] = 'degraded'
        
        # Test vector store
        try:
            vs_stats = self.vector_store.get_stats()
            health['components']['vector_store'] = 'healthy'
        except Exception as e:
            health['components']['vector_store'] = f'error: {e}'
            health['status'] = 'degraded'
        
        # Test LLM service
        if self.llm_service.is_available():
            health['components']['llm'] = 'healthy'
        else:
            health['components']['llm'] = 'no providers available'
            health['status'] = 'degraded'
        
        return health


# Instancia global del pipeline
rag_pipeline = RAGPipeline()


def get_rag_pipeline() -> RAGPipeline:
    """Obtener instancia global del pipeline RAG"""
    return rag_pipeline


# Funciones de conveniencia para usar en Flask
def process_chat_query(
    query: str,
    provider: str = 'ollama',
    **kwargs
) -> RAGResponse:
    """Función de conveniencia para procesar consultas de chat"""
    return rag_pipeline.process_query(query, provider, **kwargs)


def compare_chat_providers(query: str, **kwargs) -> Dict[str, RAGResponse]:
    """Función de conveniencia para comparar proveedores"""
    return rag_pipeline.compare_providers(query, **kwargs)