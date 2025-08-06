#!/usr/bin/env python3
"""
Script para corregir la estructura completa del proyecto
TFM Vicente Caruncho - Sistemas Inteligentes
"""

import os
from pathlib import Path
import shutil

def create_service_init_files():
    """Crear archivos __init__.py correctos para los servicios"""
    
    print("📝 Creando archivos __init__.py de servicios...")
    
    # app/services/__init__.py
    services_init = '''"""
Servicios principales del sistema
"""

# Importar servicios cuando estén disponibles
__all__ = ['rag', 'ingestion', 'llm']
'''
    
    # app/services/rag/__init__.py
    rag_init = '''"""
Servicio RAG (Retrieval-Augmented Generation)
"""

from typing import List, Optional, Dict, Any
import time
from pathlib import Path

from app.core.config import get_rag_config, get_vector_store_config
from app.core.logger import get_logger
from app.models import DocumentChunk, DocumentMetadata

class RAGService:
    """Servicio principal de RAG"""
    
    def __init__(self):
        self.logger = get_logger("prototipo_chatbot.rag_service")
        self.config = get_rag_config()
        self.vector_store = None
        self.embedding_service = None
        self._initialize()
    
    def _initialize(self):
        """Inicializar componentes del servicio"""
        try:
            # Intentar importar embedding service
            from app.services.rag.embeddings import embedding_service
            self.embedding_service = embedding_service
            
            # Intentar importar vector store
            from app.services.rag.faiss_vectorstore import FaissVectorStore
            self.vector_store = FaissVectorStore()
            
            self.logger.info(
                "RAG Service inicializado",
                embedding_available=self.embedding_service.is_available() if self.embedding_service else False,
                vector_store="faiss",
                enabled=self.config.enabled
            )
        except ImportError as e:
            self.logger.warning(f"Componentes RAG no disponibles: {e}")
    
    def is_available(self) -> bool:
        """Verificar disponibilidad del servicio"""
        return (
            self.config.enabled and 
            self.embedding_service is not None and
            self.vector_store is not None
        )
    
    def search(
        self, 
        query: str, 
        k: int = 5,
        threshold: float = 0.0
    ) -> List[DocumentChunk]:
        """Buscar documentos relevantes"""
        if not self.is_available():
            self.logger.warning("RAG Service no disponible")
            return []
        
        try:
            start_time = time.time()
            
            # Generar embedding de la consulta
            query_embedding = self.embedding_service.encode_single_text(query)
            
            if query_embedding is None:
                return []
            
            # Buscar en vector store
            results = self.vector_store.search(
                query_embedding, 
                k=k,
                threshold=threshold
            )
            
            search_time = time.time() - start_time
            
            self.logger.info(
                "Búsqueda RAG completada",
                query_length=len(query),
                results_found=len(results),
                results_filtered=len([r for r in results if r.relevance_score >= threshold]),
                k_requested=k,
                search_time=search_time
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error en búsqueda RAG: {e}")
            return []
    
    def add_document(self, chunk: DocumentChunk) -> bool:
        """Añadir documento al índice"""
        if not self.is_available():
            return False
        
        try:
            # Generar embedding si no existe
            if chunk.embedding is None:
                embedding = self.embedding_service.encode_single_text(chunk.content)
                if embedding is None:
                    return False
                chunk.embedding = embedding.tolist()
            
            # Añadir al vector store
            return self.vector_store.add(chunk)
            
        except Exception as e:
            self.logger.error(f"Error añadiendo documento: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del servicio"""
        stats = {
            'service_available': self.is_available(),
            'rag_enabled': self.config.enabled,
            'total_documents': 0,
            'embedding_service': 'not_available',
            'vector_store': 'not_available'
        }
        
        if self.embedding_service:
            stats['embedding_service'] = self.embedding_service.get_model_info()
        
        if self.vector_store:
            stats['total_documents'] = self.vector_store.get_document_count()
            stats['vector_store'] = self.vector_store.get_stats()
        
        return stats

# Instancia global del servicio
rag_service = RAGService()

# Exportar
__all__ = ['RAGService', 'rag_service']
'''
    
    # app/services/ingestion/__init__.py
    ingestion_init = '''"""
Servicio de Ingesta de Documentos
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import time

from app.core.logger import get_logger
from app.models import IngestionJob, DocumentChunk

class IngestionService:
    """Servicio principal de ingesta"""
    
    def __init__(self):
        self.logger = get_logger("prototipo_chatbot.ingestion_service")
        self.active_jobs = []
        self.processor = None
        self._initialize()
    
    def _initialize(self):
        """Inicializar procesadores"""
        try:
            from app.services.ingestion.document_processor import document_processor
            self.processor = document_processor
            self.logger.info(
                "Servicio de ingesta inicializado",
                processors=len(self.processor.processors) if self.processor else 0
            )
        except ImportError as e:
            self.logger.warning(f"Procesador de documentos no disponible: {e}")
    
    def is_available(self) -> bool:
        """Verificar disponibilidad"""
        return self.processor is not None
    
    def process_file(
        self,
        file_path: str,
        source_type: str = 'document'
    ) -> List[DocumentChunk]:
        """Procesar archivo"""
        if not self.is_available():
            return []
        
        try:
            return self.processor.process(file_path, source_type=source_type)
        except Exception as e:
            self.logger.error(f"Error procesando archivo: {e}")
            return []
    
    def get_active_jobs(self) -> List[IngestionJob]:
        """Obtener trabajos activos"""
        return self.active_jobs
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas"""
        return {
            'service_available': self.is_available(),
            'active_jobs': len(self.active_jobs),
            'supported_extensions': self.processor.get_supported_extensions() if self.processor else []
        }

# Instancia global
ingestion_service = IngestionService()

__all__ = ['IngestionService', 'ingestion_service']
'''
    
    # app/services/llm/__init__.py
    llm_init = '''"""
Servicio de LLM
"""
'''
    
    # Escribir archivos
    files_to_create = {
        "app/services/__init__.py": services_init,
        "app/services/rag/__init__.py": rag_init,
        "app/services/ingestion/__init__.py": ingestion_init,
        "app/services/llm/__init__.py": llm_init
    }
    
    for file_path, content in files_to_create.items():
        full_path = Path(file_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding='utf-8')
        print(f"   ✅ {file_path}")

def create_llm_service():
    """Crear servicio LLM completo"""
    
    print("\n📝 Creando LLM Service...")
    
    llm_service_content = '''"""
Servicio de Modelos de Lenguaje (LLM)
TFM Vicente Caruncho
"""

import time
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.core.config import get_model_config
from app.core.logger import get_logger
from app.models import ModelResponse, ComparisonResult, DocumentChunk

class LLMService:
    """Servicio principal para gestión de LLMs"""
    
    def __init__(self):
        self.logger = get_logger("prototipo_chatbot.llm_service")
        self.config = get_model_config()
        self.providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Inicializar proveedores disponibles"""
        
        # Intentar cargar Ollama
        try:
            from app.services.llm.ollama_provider import ollama_provider
            if ollama_provider.is_available():
                self.providers['ollama'] = ollama_provider
                self.logger.info("Proveedor Ollama disponible")
        except ImportError as e:
            self.logger.warning(f"Ollama no disponible: {e}")
        
        # Intentar cargar OpenAI
        try:
            from app.services.llm.openai_provider import openai_provider
            if openai_provider.is_available():
                self.providers['openai'] = openai_provider
                self.logger.info("Proveedor OpenAI disponible")
        except ImportError as e:
            self.logger.warning(f"OpenAI no disponible: {e}")
        
        self.logger.info(
            "LLM Service inicializado",
            providers=list(self.providers.keys())
        )
    
    def is_available(self) -> bool:
        """Verificar si hay algún proveedor disponible"""
        return len(self.providers) > 0
    
    def generate_response(
        self,
        query: str,
        context: Optional[List[DocumentChunk]] = None,
        provider: str = 'ollama',
        model: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Generar respuesta usando un proveedor específico"""
        
        if provider not in self.providers:
            available = list(self.providers.keys())
            if available:
                provider = available[0]
                self.logger.warning(
                    f"Proveedor {provider} no disponible, usando {provider}"
                )
            else:
                return ModelResponse(
                    content="No hay proveedores de LLM disponibles",
                    model="none",
                    provider="none",
                    error="No providers available"
                )
        
        try:
            return self.providers[provider].generate(
                prompt=query,
                model=model,
                context=context,
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"Error generando respuesta: {e}")
            return ModelResponse(
                content="Error generando respuesta",
                model=model or "unknown",
                provider=provider,
                error=str(e)
            )
    
    def compare_models(
        self,
        query: str,
        context: Optional[List[DocumentChunk]] = None,
        local_model: Optional[str] = None,
        openai_model: Optional[str] = None,
        **kwargs
    ) -> ComparisonResult:
        """Comparar respuestas de diferentes modelos"""
        
        self.logger.info(
            "Iniciando comparación de modelos",
            local_model=local_model,
            openai_model=openai_model
        )
        
        comparison = ComparisonResult(query=query)
        
        # Generar respuesta local
        if 'ollama' in self.providers:
            try:
                comparison.local_response = self.generate_response(
                    query=query,
                    context=context,
                    provider='ollama',
                    model=local_model,
                    **kwargs
                )
            except Exception as e:
                self.logger.error(f"Error en modelo local: {e}")
                comparison.local_response = ModelResponse(
                    content="",
                    model=local_model or "unknown",
                    provider="ollama",
                    error=str(e)
                )
        
        # Generar respuesta OpenAI
        if 'openai' in self.providers:
            try:
                comparison.openai_response = self.generate_response(
                    query=query,
                    context=context,
                    provider='openai',
                    model=openai_model,
                    **kwargs
                )
            except Exception as e:
                self.logger.error(f"Error en OpenAI: {e}")
                comparison.openai_response = ModelResponse(
                    content="",
                    model=openai_model or "unknown",
                    provider="openai",
                    error=str(e)
                )
        
        return comparison
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del servicio"""
        stats = {
            'service_available': self.is_available(),
            'providers_available': {},
            'providers_total': len(self.providers)
        }
        
        for name, provider in self.providers.items():
            stats['providers_available'][name] = provider.is_available()
        
        return stats

# Instancia global
llm_service = LLMService()

__all__ = ['LLMService', 'llm_service']
'''
    
    llm_service_path = Path("app/services/llm_service.py")
    llm_service_path.write_text(llm_service_content, encoding='utf-8')
    print(f"   ✅ app/services/llm_service.py")

def create_faiss_vectorstore():
    """Crear vector store FAISS"""
    
    print("\n📝 Creando FAISS Vector Store...")
    
    faiss_content = '''"""
FAISS Vector Store Implementation
"""

import numpy as np
from typing import List, Optional, Dict, Any
import pickle
from pathlib import Path

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from app.core.config import get_vector_store_config
from app.core.logger import get_logger
from app.models import DocumentChunk

class FaissVectorStore:
    """Implementación de vector store usando FAISS"""
    
    def __init__(self):
        self.logger = get_logger("prototipo_chatbot.faiss_vectorstore")
        self.config = get_vector_store_config()
        self.index = None
        self.documents = []
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        self._initialize()
    
    def _initialize(self):
        """Inicializar índice FAISS"""
        if not HAS_FAISS:
            self.logger.error("FAISS no instalado")
            return
        
        try:
            # Crear índice L2
            self.index = faiss.IndexFlatL2(self.dimension)
            self.logger.info(
                "Nuevo índice FAISS creado",
                dimension=self.dimension,
                index_type="IndexFlatL2"
            )
        except Exception as e:
            self.logger.error(f"Error inicializando FAISS: {e}")
    
    def is_available(self) -> bool:
        """Verificar disponibilidad"""
        return HAS_FAISS and self.index is not None
    
    def add(self, chunk: DocumentChunk) -> bool:
        """Añadir documento al índice"""
        if not self.is_available():
            return False
        
        try:
            if chunk.embedding is None:
                return False
            
            # Convertir a numpy array
            embedding = np.array(chunk.embedding, dtype=np.float32)
            embedding = embedding.reshape(1, -1)
            
            # Añadir al índice
            self.index.add(embedding)
            self.documents.append(chunk)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error añadiendo al índice: {e}")
            return False
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: float = 0.0
    ) -> List[DocumentChunk]:
        """Buscar documentos similares"""
        if not self.is_available() or len(self.documents) == 0:
            self.logger.warning("Índice FAISS vacío")
            return []
        
        try:
            # Preparar embedding
            query_embedding = np.array(query_embedding, dtype=np.float32)
            query_embedding = query_embedding.reshape(1, -1)
            
            # Buscar
            distances, indices = self.index.search(query_embedding, min(k, len(self.documents)))
            
            # Crear resultados
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.documents):
                    chunk = self.documents[idx]
                    chunk.relevance_score = float(1.0 / (1.0 + dist))
                    if chunk.relevance_score >= threshold:
                        results.append(chunk)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error en búsqueda: {e}")
            return []
    
    def get_document_count(self) -> int:
        """Obtener número de documentos"""
        return len(self.documents)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas"""
        return {
            'type': 'faiss',
            'available': self.is_available(),
            'documents': len(self.documents),
            'dimension': self.dimension
        }

__all__ = ['FaissVectorStore']
'''
    
    faiss_path = Path("app/services/rag/faiss_vectorstore.py")
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss_path.write_text(faiss_content, encoding='utf-8')
    print(f"   ✅ app/services/rag/faiss_vectorstore.py")

def main():
    """Función principal"""
    
    print("🔧 CORRECCIÓN COMPLETA DE ESTRUCTURA DEL PROYECTO")
    print("=" * 60)
    
    # Crear archivos de servicios
    create_service_init_files()
    create_llm_service()
    create_faiss_vectorstore()
    
    print("\n✅ Estructura corregida exitosamente")
    print("\n📋 Próximos pasos:")
    print("   1. Instalar dependencias: pip install -r requirements.txt")
    print("   2. Configurar .env con tus API keys")
    print("   3. Iniciar Ollama: ollama serve")
    print("   4. Descargar modelo: ollama pull llama3.1:8b")
    print("   5. Ejecutar test: python test_rag_pipeline.py")

if __name__ == "__main__":
    main()