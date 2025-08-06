"""
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
