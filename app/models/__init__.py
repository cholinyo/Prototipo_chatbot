"""Package initialization"""

# =============================================================================
# IMPORTS PRINCIPALES - MODELOS DE DOCUMENTOS
# =============================================================================

from .document import DocumentChunk, DocumentMetadata, create_chunk, create_chunks_from_text

# =============================================================================
# SISTEMA DE ESTADÍSTICAS
# =============================================================================

from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class SystemStats:
    """Estadísticas del sistema en tiempo real"""
    
    # Contadores principales
    documents_indexed: int = 0
    queries_processed: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    
    # Tiempos y rendimiento
    total_processing_time: float = 0.0
    average_response_time: float = 0.0
    uptime_seconds: float = 0.0
    
    # Memoria y recursos
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    last_query_at: Optional[datetime] = None
    last_indexing_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Post-inicialización"""
        import psutil
        import os
        
        # Obtener uso de memoria actual
        process = psutil.Process(os.getpid())
        self.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        self.cpu_usage_percent = process.cpu_percent()
    
    def update_query_stats(self, processing_time: float, success: bool = True):
        """Actualizar estadísticas de consultas"""
        self.queries_processed += 1
        self.last_query_at = datetime.now()
        self.total_processing_time += processing_time
        
        if success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1
        
        # Calcular tiempo promedio
        if self.queries_processed > 0:
            self.average_response_time = self.total_processing_time / self.queries_processed
        
        self._update_system_metrics()
    
    def update_indexing_stats(self, documents_added: int, processing_time: float):
        """Actualizar estadísticas de indexación"""
        self.documents_indexed += documents_added
        self.last_indexing_at = datetime.now()
        self.total_processing_time += processing_time
        
        self._update_system_metrics()
    
    def _update_system_metrics(self):
        """Actualizar métricas del sistema"""
        import psutil
        import os
        
        try:
            process = psutil.Process(os.getpid())
            self.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            self.cpu_usage_percent = process.cpu_percent()
            self.last_updated = datetime.now()
            
            # Calcular uptime
            self.uptime_seconds = (self.last_updated - self.created_at).total_seconds()
            
        except Exception:
            # Si psutil no está disponible, usar valores por defecto
            pass
    
    def get_uptime_hours(self) -> float:
        """Obtener uptime en horas"""
        return self.uptime_seconds / 3600
    
    def get_success_rate(self) -> float:
        """Obtener tasa de éxito de consultas"""
        if self.queries_processed == 0:
            return 0.0
        return (self.successful_queries / self.queries_processed) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para JSON"""
        return {
            'documents_indexed': self.documents_indexed,
            'queries_processed': self.queries_processed,
            'successful_queries': self.successful_queries,
            'failed_queries': self.failed_queries,
            'success_rate': self.get_success_rate(),
            'average_response_time': self.average_response_time,
            'uptime_hours': self.get_uptime_hours(),
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'last_query_at': self.last_query_at.isoformat() if self.last_query_at else None,
            'last_indexing_at': self.last_indexing_at.isoformat() if self.last_indexing_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemStats':
        """Crear desde diccionario"""
        # Convertir timestamps string a datetime
        for field in ['last_updated', 'last_query_at', 'last_indexing_at']:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)
    
    def reset_stats(self):
        """Resetear todas las estadísticas"""
        self.documents_indexed = 0
        self.queries_processed = 0
        self.successful_queries = 0
        self.failed_queries = 0
        self.total_processing_time = 0.0
        self.average_response_time = 0.0
        self.last_query_at = None
        self.last_indexing_at = None
        self.created_at = datetime.now()
        self._update_system_metrics()


# =============================================================================
# MODELOS ADICIONALES PARA EL SISTEMA
# =============================================================================

@dataclass
class QueryMetrics:
    """Métricas de una consulta individual"""
    query: str
    response_time: float
    tokens_used: int = 0
    cost_usd: float = 0.0
    model_used: str = ""
    sources_found: int = 0
    success: bool = True
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DocumentMetrics:
    """Métricas de un documento indexado"""
    filename: str
    file_size_bytes: int
    chunks_created: int
    processing_time: float
    success: bool = True
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ServiceHealth:
    """Estado de salud de un servicio"""
    service_name: str
    status: str  # healthy, warning, error, unknown
    last_check: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    uptime_percent: float = 100.0

@dataclass
class IngestionJob:
    """Job de ingesta de documentos"""
    id: str
    file_path: str
    source_type: str
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

# =============================================================================
# EXPORTACIONES
# =============================================================================

__all__ = [
    # Modelos de documentos
    'DocumentChunk',
    'DocumentMetadata', 
    'create_chunk',
    'create_chunks_from_text',
    
    # Estadísticas del sistema
    'SystemStats',
    'QueryMetrics',
    'DocumentMetrics',
    'ServiceHealth',
    'IngestionJob',
]