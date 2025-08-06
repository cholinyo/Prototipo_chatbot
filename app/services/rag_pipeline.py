"""
Pipeline RAG Principal
Prototipo_chatbot - TFM Vicente Caruncho
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from app.core.config import get_rag_config
from app.core.logger import get_logger
from app.models import (
    DocumentChunk, 
    ModelResponse, 
    ComparisonResult,
    SearchResult
)

# Importar servicios
from app.services.rag import rag_service
from app.services.llm_service import llm_service
from app.services.ingestion import ingestion_service

class RAGPipeline:
    """Pipeline principal que coordina todo el sistema RAG"""
    
    def __init__(self):
        self.logger = get_logger("prototipo_chatbot.rag_pipeline")
        self.config = get_rag_config()
        
        # Configuración del pipeline
        self.chunk_size = self.config.chunk_size
        self.chunk_overlap = self.config.chunk_overlap
        self.default_k = self.config.similarity_top_k
        self.enabled = self.config.enabled
        
        self.logger.info(
            "RAG Pipeline inicializado",
            rag_enabled=self.enabled,
            chunk_size=self.chunk_size,
            default_k=self.default_k
        )
    
    def is_available(self) -> bool:
        """Verificar disponibilidad del pipeline"""
        return (
            self.enabled and 
            rag_service.is_available() and
            llm_service.is_available()
        )
    
    def process_query(
        self,
        query: str,
        k: int = None,
        use_rag: bool = True,
        provider: str = 'ollama',
        model: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Procesar consulta con RAG opcional"""
        
        k = k or self.default_k
        
        self.logger.info(
            "Procesando consulta RAG",
            query_length=len(query),
            use_rag=use_rag,
            k=k,
            provider=provider
        )
        
        context = None
        
        try:
            # Recuperar contexto si RAG está habilitado
            if use_rag and rag_service.is_available():
                start_retrieval = time.time()
                
                # Buscar documentos relevantes
                search_results = rag_service.search(query, k=k)
                context = search_results if search_results else None
                
                retrieval_time = time.time() - start_retrieval
                
                self.logger.info(
                    "Contexto RAG recuperado",
                    chunks_found=len(context) if context else 0,
                    retrieval_time=retrieval_time,
                    sources_count=len(set(c.metadata.source_path for c in context if c.metadata)) if context else 0
                )
            
            # Generar respuesta con LLM
            response = llm_service.generate_response(
                query=query,
                context=context,
                provider=provider,
                model=model,
                **kwargs
            )
            
            return response
            
        except Exception as e:
            self.logger.error(
                "Error procesando consulta RAG",
                error=str(e),
                query=query[:100]
            )
            
            return ModelResponse(
                content="Lo siento, ocurrió un error procesando tu consulta.",
                model=model or "unknown",
                provider=provider,
                error=str(e)
            )
    
    def compare_models(
        self,
        query: str,
        k: int = None,
        local_model: Optional[str] = None,
        openai_model: Optional[str] = None,
        use_rag: bool = True,
        **kwargs
    ) -> ComparisonResult:
        """Comparar respuestas de modelo local y OpenAI"""
        
        k = k or self.default_k
        
        self.logger.info(
            "Iniciando comparación RAG",
            query_length=len(query),
            k=k,
            local_model=local_model,
            openai_model=openai_model
        )
        
        start_time = time.time()
        context = None
        
        try:
            # Recuperar contexto compartido
            if use_rag and rag_service.is_available():
                context = rag_service.search(query, k=k)
                
                self.logger.info(
                    "Contexto compartido recuperado",
                    chunks_count=len(context) if context else 0,
                    retrieval_time=time.time() - start_time
                )
            
            # Comparar modelos
            comparison = llm_service.compare_models(
                query=query,
                context=context,
                local_model=local_model,
                openai_model=openai_model,
                **kwargs
            )
            
            comparison.context_used = context
            comparison.total_time = time.time() - start_time
            
            return comparison
            
        except Exception as e:
            self.logger.error(
                "Error en comparación RAG",
                error=str(e)
            )
            
            return ComparisonResult(
                query=query,
                context_used=context,
                total_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def ingest_document(
        self,
        file_path: str,
        source_type: str = 'document',
        **kwargs
    ) -> Dict[str, Any]:
        """Ingestar documento en el sistema RAG"""
        
        self.logger.info(
            "Iniciando ingesta de documento",
            file_path=file_path,
            source_type=source_type
        )
        
        start_time = time.time()
        
        try:
            # Verificar que el archivo existe
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
            
            # Procesar documento con servicio de ingesta
            try:
                from app.services.ingestion.document_processor import document_processor
                
                chunks = document_processor.process(
                    file_path=file_path,
                    source_type=source_type,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    **kwargs
                )
            except ImportError:
                # Si no está disponible el procesador completo, usar versión simple
                chunks = self._simple_text_processing(file_path, source_type)
            
            processing_time = time.time() - start_time
            
            if not chunks:
                self.logger.warning(
                    "No se crearon chunks del documento",
                    file_path=file_path
                )
                return {
                    'success': False,
                    'file_path': file_path,
                    'chunks_created': 0,
                    'processing_time': processing_time,
                    'error': 'No se pudieron extraer chunks del documento'
                }
            
            # Indexar chunks en el vector store
            indexed_count = 0
            if rag_service.is_available():
                for chunk in chunks:
                    success = rag_service.add_document(chunk)
                    if success:
                        indexed_count += 1
            
            self.logger.info(
                "Documento ingestado exitosamente",
                file_path=file_path,
                chunks_created=len(chunks),
                chunks_indexed=indexed_count,
                processing_time=processing_time
            )
            
            return {
                'success': True,
                'file_path': file_path,
                'chunks_created': len(chunks),
                'chunks_indexed': indexed_count,
                'processing_time': processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            self.logger.error(
                "Error en ingesta de documento",
                file_path=file_path,
                error=str(e),
                processing_time=processing_time
            )
            
            return {
                'success': False,
                'file_path': file_path,
                'chunks_created': 0,
                'chunks_indexed': 0,
                'processing_time': processing_time,
                'error': str(e)
            }
    
    def _simple_text_processing(
        self, 
        file_path: str,
        source_type: str
    ) -> List[DocumentChunk]:
        """Procesamiento simple de texto como fallback"""
        from app.models import create_document_chunk, DocumentMetadata
        import os
        
        chunks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Crear metadata
            metadata = DocumentMetadata(
                source_path=str(file_path),
                source_type=source_type,
                file_type='text',
                file_size=os.path.getsize(file_path),
                created_at=time.time()
            )
            
            # Dividir en chunks simples
            chunk_size = self.chunk_size
            for i in range(0, len(content), chunk_size - self.chunk_overlap):
                chunk_content = content[i:i + chunk_size]
                if chunk_content.strip():
                    chunk = create_document_chunk(
                        content=chunk_content,
                        metadata=metadata,
                        chunk_index=i // (chunk_size - self.chunk_overlap)
                    )
                    chunks.append(chunk)
            
        except Exception as e:
            self.logger.error(f"Error en procesamiento simple: {e}")
        
        return chunks
    
    def search_documents(
        self,
        query: str,
        k: int = None,
        threshold: float = 0.0
    ) -> SearchResult:
        """Buscar documentos relevantes"""
        
        k = k or self.default_k
        start_time = time.time()
        
        try:
            chunks = rag_service.search(
                query=query,
                k=k,
                threshold=threshold
            )
            
            search_time = time.time() - start_time
            
            return SearchResult(
                chunks=chunks,
                query=query,
                search_time=search_time,
                total_results=len(chunks)
            )
            
        except Exception as e:
            self.logger.error(f"Error en búsqueda: {e}")
            
            return SearchResult(
                chunks=[],
                query=query,
                search_time=time.time() - start_time,
                total_results=0,
                metadata={'error': str(e)}
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del pipeline"""
        stats = {
            'pipeline_enabled': self.enabled,
            'pipeline_available': self.is_available(),
            'configuration': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'default_k': self.default_k
            }
        }
        
        # Agregar estadísticas de servicios
        if rag_service:
            stats['rag_service'] = rag_service.get_stats()
        
        if llm_service:
            stats['llm_service'] = llm_service.get_service_stats()
        
        return stats

# Instancia global del pipeline
rag_pipeline = RAGPipeline()

# Exportar
__all__ = ['RAGPipeline', 'rag_pipeline']