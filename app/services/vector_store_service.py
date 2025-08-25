"""
VectorStoreService - Implementación Real
TFM Vicente Caruncho - Sistemas Inteligentes

Servicio unificado que integra FAISS y ChromaDB existentes
"""

import os
from typing import List, Dict, Any, Optional
from enum import Enum

from app.core.logger import get_logger
from app.services.ingestion.document_processor import DocumentChunk

# Importar los stores reales existentes
from app.services.rag.faiss_store import (
    FaissVectorStore, 
    get_faiss_store, 
    is_faiss_available
)
from app.services.rag.chromadb_store import (
    ChromaDBVectorStore, 
    get_chromadb_store, 
    is_chromadb_available
)


class VectorStoreType(Enum):
    """Tipos de vector stores disponibles"""
    FAISS = "faiss"
    CHROMADB = "chromadb"
    AUTO = "auto"


class VectorStoreService:
    """
    Servicio unificado para gestionar vector stores
    Integra con las implementaciones reales de FAISS y ChromaDB
    """
    
    def __init__(self, 
                 preferred_store: str = None,
                 enable_fallback: bool = True):
        """
        Inicializar servicio de vector stores
        
        Args:
            preferred_store: Store preferido ('faiss', 'chromadb', 'auto')
            enable_fallback: Si usar fallback automático entre stores
        """
        self.logger = get_logger("vector_store_service")
        self.enable_fallback = enable_fallback
        
        # Configurar store preferido desde env o parámetro
        self.preferred_store = (
            preferred_store or 
            os.getenv("DEFAULT_VECTOR_STORE", "auto")
        ).lower()
        
        # Inicializar stores
        self._initialize_stores()
        
        # Seleccionar store activo
        self.active_store = self._select_active_store()
        
        self.logger.info(
            "VectorStoreService inicializado",
            preferred_store=self.preferred_store,
            active_store=self.active_store.__class__.__name__ if self.active_store else None,
            fallback_enabled=self.enable_fallback,
            faiss_available=self.faiss_available,
            chromadb_available=self.chromadb_available
        )
    
    def _initialize_stores(self):
        """Inicializar conexiones a los stores"""
        # FAISS Store
        try:
            self.faiss_store = get_faiss_store()
            self.faiss_available = is_faiss_available()
            if self.faiss_available:
                self.logger.info("✅ FAISS Store disponible")
            else:
                self.logger.warning("⚠️ FAISS Store no disponible")
        except Exception as e:
            self.logger.error(f"Error inicializando FAISS: {e}")
            self.faiss_store = None
            self.faiss_available = False
        
        # ChromaDB Store
        try:
            self.chromadb_store = get_chromadb_store()
            self.chromadb_available = is_chromadb_available()
            if self.chromadb_available:
                self.logger.info("✅ ChromaDB Store disponible")
            else:
                self.logger.warning("⚠️ ChromaDB Store no disponible")
        except Exception as e:
            self.logger.error(f"Error inicializando ChromaDB: {e}")
            self.chromadb_store = None
            self.chromadb_available = False
    
    def _select_active_store(self):
        """Seleccionar el store activo basado en configuración y disponibilidad"""
        if self.preferred_store == "faiss" and self.faiss_available:
            return self.faiss_store
        elif self.preferred_store == "chromadb" and self.chromadb_available:
            return self.chromadb_store
        elif self.preferred_store == "auto":
            # Auto: preferir FAISS por rendimiento, fallback a ChromaDB
            if self.faiss_available:
                return self.faiss_store
            elif self.chromadb_available:
                return self.chromadb_store
        
        # Fallback si el preferido no está disponible
        if self.enable_fallback:
            if self.faiss_available:
                self.logger.warning(
                    f"Store preferido '{self.preferred_store}' no disponible, usando FAISS"
                )
                return self.faiss_store
            elif self.chromadb_available:
                self.logger.warning(
                    f"Store preferido '{self.preferred_store}' no disponible, usando ChromaDB"
                )
                return self.chromadb_store
        
        # Ningún store disponible
        self.logger.error("Ningún vector store disponible")
        return None
    
    def is_available(self) -> bool:
        """Verificar si hay al menos un vector store disponible"""
        return self.active_store is not None
    
    def get_active_store_type(self) -> Optional[str]:
        """Obtener tipo del store actualmente activo"""
        if not self.active_store:
            return None
        
        if isinstance(self.active_store, FaissVectorStore):
            return "faiss"
        elif isinstance(self.active_store, ChromaDBVectorStore):
            return "chromadb"
        else:
            return "unknown"
    
    def switch_store(self, store_type: str) -> bool:
        """
        Cambiar el store activo
        
        Args:
            store_type: 'faiss' o 'chromadb'
            
        Returns:
            bool: Si el cambio fue exitoso
        """
        store_type = store_type.lower()
        
        if store_type == "faiss" and self.faiss_available:
            self.active_store = self.faiss_store
            self.logger.info("Cambiado a FAISS Store")
            return True
        elif store_type == "chromadb" and self.chromadb_available:
            self.active_store = self.chromadb_store
            self.logger.info("Cambiado a ChromaDB Store")
            return True
        else:
            self.logger.error(f"No se puede cambiar a store '{store_type}' (no disponible)")
            return False
    
    def add_documents(self, chunks: List[DocumentChunk], source_metadata: Dict[str, Any] = None) -> bool:
        """
        Agregar documentos al vector store activo
        
        Args:
            chunks: Lista de chunks de documentos
            source_metadata: Metadatos adicionales de la fuente
            
        Returns:
            bool: Si la operación fue exitosa
        """
        if not self.is_available():
            self.logger.error("Ningún vector store disponible para agregar documentos")
            return False
        
        try:
            # Enriquecer chunks con metadatos de fuente si se proporcionan
            if source_metadata:
                for chunk in chunks:
                    if chunk.metadata:
                        chunk.metadata.update(source_metadata)
                    else:
                        chunk.metadata = source_metadata.copy()
            
            # Usar el store activo
            success = self.active_store.add_documents(chunks)
            
            if success:
                self.logger.info(
                    f"Agregados {len(chunks)} chunks usando {self.get_active_store_type()}",
                    chunks_count=len(chunks),
                    store_type=self.get_active_store_type()
                )
            else:
                self.logger.error(f"Error agregando documentos en {self.get_active_store_type()}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error agregando documentos: {e}")
            
            # Intentar fallback si está habilitado
            if self.enable_fallback:
                return self._try_fallback_add(chunks, source_metadata)
            
            return False
    
    def _try_fallback_add(self, chunks: List[DocumentChunk], source_metadata: Dict[str, Any] = None) -> bool:
        """Intentar agregar documentos usando store alternativo"""
        current_type = self.get_active_store_type()
        
        if current_type == "faiss" and self.chromadb_available:
            self.logger.warning("Intentando fallback a ChromaDB para agregar documentos")
            try:
                return self.chromadb_store.add_documents(chunks)
            except Exception as e:
                self.logger.error(f"Fallback a ChromaDB falló: {e}")
        elif current_type == "chromadb" and self.faiss_available:
            self.logger.warning("Intentando fallback a FAISS para agregar documentos")
            try:
                return self.faiss_store.add_documents(chunks)
            except Exception as e:
                self.logger.error(f"Fallback a FAISS falló: {e}")
        
        return False
    
    def remove_document(self, document_id: str) -> bool:
        """
        Eliminar documento del vector store activo
        
        Args:
            document_id: ID del documento a eliminar
            
        Returns:
            bool: Si la eliminación fue exitosa
        """
        if not self.is_available():
            self.logger.error("Ningún vector store disponible para eliminar documento")
            return False
        
        try:
            # Solo ChromaDB soporta eliminación por ID directamente
            if isinstance(self.active_store, ChromaDBVectorStore):
                return self.active_store.remove_document(document_id)
            else:
                # Para FAISS, necesitaríamos implementar eliminación personalizada
                self.logger.warning(
                    "Eliminación de documentos individuales no soportada en FAISS. "
                    "Considera usar ChromaDB o reindexar completamente."
                )
                return False
                
        except Exception as e:
            self.logger.error(f"Error eliminando documento {document_id}: {e}")
            return False
    
    def search(self, query: str, k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Buscar documentos similares en el vector store activo
        
        Args:
            query: Consulta de búsqueda
            k: Número máximo de resultados
            filters: Filtros de metadatos (opcional)
            
        Returns:
            List[Dict]: Lista de resultados encontrados
        """
        if not self.is_available():
            self.logger.error("Ningún vector store disponible para búsqueda")
            return []
        
        try:
            # Búsqueda usando el store activo
            results = self.active_store.search(query, k=k, filters=filters)
            
            self.logger.info(
                f"Búsqueda completada usando {self.get_active_store_type()}",
                query=query[:50] + "..." if len(query) > 50 else query,
                results_count=len(results),
                k=k,
                store_type=self.get_active_store_type()
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error en búsqueda: {e}")
            
            # Intentar fallback si está habilitado
            if self.enable_fallback:
                return self._try_fallback_search(query, k, filters)
            
            return []
    
    def _try_fallback_search(self, query: str, k: int, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Intentar búsqueda usando store alternativo"""
        current_type = self.get_active_store_type()
        
        if current_type == "faiss" and self.chromadb_available:
            self.logger.warning("Intentando fallback a ChromaDB para búsqueda")
            try:
                return self.chromadb_store.search(query, k=k, filters=filters)
            except Exception as e:
                self.logger.error(f"Fallback a ChromaDB falló: {e}")
        elif current_type == "chromadb" and self.faiss_available:
            self.logger.warning("Intentando fallback a FAISS para búsqueda")
            try:
                return self.faiss_store.search(query, k=k, filters=filters)
            except Exception as e:
                self.logger.error(f"Fallback a FAISS falló: {e}")
        
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del sistema de vector stores
        
        Returns:
            Dict: Estadísticas completas del sistema
        """
        stats = {
            'active_store': self.get_active_store_type(),
            'preferred_store': self.preferred_store,
            'fallback_enabled': self.enable_fallback,
            'stores_available': {
                'faiss': self.faiss_available,
                'chromadb': self.chromadb_available
            },
            'total_stores_available': sum([
                self.faiss_available, 
                self.chromadb_available
            ])
        }
        
        # Estadísticas del store activo
        if self.active_store:
            if hasattr(self.active_store, 'get_stats'):
                active_stats = self.active_store.get_stats()
                stats['active_store_stats'] = active_stats
            elif hasattr(self.active_store, 'get_metrics'):
                active_metrics = self.active_store.get_metrics()
                stats['active_store_stats'] = active_metrics.to_dict()
        
        # Estadísticas de todos los stores disponibles
        all_store_stats = {}
        
        if self.faiss_available and hasattr(self.faiss_store, 'get_metrics'):
            faiss_metrics = self.faiss_store.get_metrics()
            all_store_stats['faiss'] = faiss_metrics.to_dict()
        
        if self.chromadb_available and hasattr(self.chromadb_store, 'get_metrics'):
            chromadb_metrics = self.chromadb_store.get_metrics()
            all_store_stats['chromadb'] = chromadb_metrics.to_dict()
        
        if all_store_stats:
            stats['all_stores_stats'] = all_store_stats
        
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Obtener estado de salud del sistema
        
        Returns:
            Dict: Estado de salud completo
        """
        health = {
            'overall_status': 'healthy' if self.is_available() else 'unhealthy',
            'timestamp': str(datetime.now().isoformat()),
            'services': {}
        }
        
        # Estado de FAISS
        if self.faiss_store:
            try:
                faiss_healthy = self.faiss_store.is_available()
                health['services']['faiss'] = {
                    'status': 'healthy' if faiss_healthy else 'unhealthy',
                    'available': self.faiss_available,
                    'error': None
                }
                
                if faiss_healthy and hasattr(self.faiss_store, 'get_metrics'):
                    metrics = self.faiss_store.get_metrics()
                    health['services']['faiss']['total_vectors'] = metrics.total_vectors
                    health['services']['faiss']['memory_usage_mb'] = metrics.memory_usage_mb
                    
            except Exception as e:
                health['services']['faiss'] = {
                    'status': 'error',
                    'available': False,
                    'error': str(e)
                }
        
        # Estado de ChromaDB
        if self.chromadb_store:
            try:
                chromadb_healthy = self.chromadb_store.is_available()
                health['services']['chromadb'] = {
                    'status': 'healthy' if chromadb_healthy else 'unhealthy',
                    'available': self.chromadb_available,
                    'error': None
                }
                
                if chromadb_healthy and hasattr(self.chromadb_store, 'get_metrics'):
                    metrics = self.chromadb_store.get_metrics()
                    health['services']['chromadb']['total_documents'] = metrics.total_documents
                    health['services']['chromadb']['disk_usage_mb'] = metrics.disk_usage_mb
                    
            except Exception as e:
                health['services']['chromadb'] = {
                    'status': 'error',
                    'available': False,
                    'error': str(e)
                }
        
        return health
    
    def compare_stores(self, test_query: str = "consulta de prueba") -> Dict[str, Any]:
        """
        Comparar rendimiento básico entre stores disponibles
        
        Args:
            test_query: Query de prueba para benchmark básico
            
        Returns:
            Dict: Resultados de comparación
        """
        comparison = {
            'test_query': test_query,
            'timestamp': datetime.now().isoformat(),
            'stores_tested': [],
            'comparison_results': {}
        }
        
        # Probar FAISS si está disponible
        if self.faiss_available:
            try:
                import time
                start_time = time.time()
                faiss_results = self.faiss_store.search(test_query, k=5)
                search_time = time.time() - start_time
                
                comparison['stores_tested'].append('faiss')
                comparison['comparison_results']['faiss'] = {
                    'search_time_ms': search_time * 1000,
                    'results_count': len(faiss_results),
                    'status': 'success'
                }
            except Exception as e:
                comparison['comparison_results']['faiss'] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Probar ChromaDB si está disponible
        if self.chromadb_available:
            try:
                import time
                start_time = time.time()
                chromadb_results = self.chromadb_store.search(test_query, k=5)
                search_time = time.time() - start_time
                
                comparison['stores_tested'].append('chromadb')
                comparison['comparison_results']['chromadb'] = {
                    'search_time_ms': search_time * 1000,
                    'results_count': len(chromadb_results),
                    'status': 'success'
                }
            except Exception as e:
                comparison['comparison_results']['chromadb'] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return comparison


# Instancia global del servicio
vector_store_service = VectorStoreService()


def get_vector_store_service() -> VectorStoreService:
    """Obtener instancia global del servicio de vector stores"""
    return vector_store_service


def is_vector_store_available() -> bool:
    """Verificar si hay algún vector store disponible"""
    return vector_store_service.is_available()