"""
Servicio de Ingesta de Documentos
TFM Vicente Caruncho - Sistemas Inteligentes
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import os

from app.core.logger import get_logger
from app.models import IngestionJob, DocumentChunk

class MockDocumentProcessor:
    """Procesador básico de documentos para validación"""
    
    def __init__(self):
        self.supported_extensions = ['.pdf', '.docx', '.txt', '.md', '.html']
        self.logger = get_logger("prototipo_chatbot.document_processor")
    
    def process(self, file_path: str, source_type: str = 'document') -> List[DocumentChunk]:
        """Procesar archivo (implementación básica)"""
        if not os.path.exists(file_path):
            return []
        
        # Crear chunk mock para testing
        mock_chunk = DocumentChunk(
            id=f"chunk_{int(time.time())}",
            content=f"Contenido procesado de {Path(file_path).name}",
            metadata={
                'source_path': file_path,
                'source_type': source_type,
                'title': Path(file_path).stem
            },
            embedding=None
        )
        
        self.logger.info(f"Archivo procesado: {file_path}")
        return [mock_chunk]
    
    def get_supported_extensions(self) -> List[str]:
        """Obtener extensiones soportadas"""
        return self.supported_extensions

class IngestionService:
    """Servicio principal de ingesta"""
    
    def __init__(self):
        self.logger = get_logger("prototipo_chatbot.ingestion_service")
        self.active_jobs = []
        self.processor = MockDocumentProcessor()  # Usar procesador básico
        self.job_counter = 0
        self._initialize()
    
    def _initialize(self):
        """Inicializar procesadores"""
        try:
            # Intentar cargar procesador real si existe
            try:
                from app.services.ingestion.document_processor import document_processor
                self.processor = document_processor
                self.logger.info("Procesador de documentos real cargado")
            except ImportError:
                self.logger.info("Usando procesador de documentos básico")
            
            self.logger.info(
                "Servicio de ingesta inicializado",
                processors=len(self.processor.supported_extensions) if self.processor else 0
            )
        except Exception as e:
            self.logger.warning(f"Error inicializando procesador: {e}")
    
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
    
    def create_job(self, file_path: str, source_type: str = 'document') -> IngestionJob:
        """Crear nuevo trabajo de ingesta"""
        self.job_counter += 1
        job = IngestionJob(
            id=f"job_{self.job_counter}",
            file_path=file_path,
            source_type=source_type,
            status='pending'
        )
        self.active_jobs.append(job)
        return job
    
    def get_job_status(self, job_id: str) -> Optional[IngestionJob]:
        """Obtener estado de trabajo"""
        for job in self.active_jobs:
            if job.id == job_id:
                return job
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancelar trabajo"""
        for job in self.active_jobs:
            if job.id == job_id:
                job.status = 'cancelled'
                return True
        return False
    
    def get_active_jobs(self) -> List[IngestionJob]:
        """Obtener trabajos activos"""
        return [job for job in self.active_jobs if job.status in ['pending', 'processing']]
    
    def get_supported_formats(self) -> Dict[str, Any]:
        """Obtener formatos soportados"""
        return {
            'documents': ['pdf', 'docx', 'txt', 'md'],
            'data': ['csv', 'xlsx', 'json'],
            'web': ['html', 'xml'],
            'max_file_size': '50MB'
        }
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas"""
        return {
            'service_available': self.is_available(),
            'active_jobs': len(self.get_active_jobs()),
            'total_jobs': len(self.active_jobs),
            'supported_extensions': self.processor.get_supported_extensions() if self.processor else [],
            'processors_available': 1 if self.processor else 0
        }

# Instancia global
ingestion_service = IngestionService()

# Exportar correctamente
__all__ = ['IngestionService', 'ingestion_service']
