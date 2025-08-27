"""
Servicio de Ingesta de Documentos - VERSIÓN LIMPIA SIN MOCKS
TFM Vicente Caruncho - Sistemas Inteligentes

CAMBIOS APLICADOS:
- Eliminado MockDocumentProcessor completo
- Integrado DocumentProcessor real
- Eliminados todos los contenidos mock
- Añadida gestión real de errores y logging
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time
import os
import uuid
from datetime import datetime

from app.core.logger import get_logger
from app.models import IngestionJob, DocumentChunk
from app.services.ingestion.document_processor import DocumentProcessor


class IngestionService:
    """Servicio principal de ingesta - IMPLEMENTACIÓN REAL"""
    
    def __init__(self):
        self.logger = get_logger("prototipo_chatbot.ingestion_service")
        self.active_jobs: Dict[str, IngestionJob] = {}
        
        # USAR PROCESADOR REAL - NO MOCK
        self.processor = DocumentProcessor()
        
        # Directorio de trabajo
        self.work_dir = Path("data/ingestion")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("IngestionService inicializado con DocumentProcessor real")
    
    def create_job(self, source_path: str, job_type: str = 'document') -> str:
        """Crear nuevo trabajo de ingesta"""
        job_id = str(uuid.uuid4())
        
        job = IngestionJob(
            id=job_id,
            source_path=source_path,
            job_type=job_type,
            status='pending',
            created_at=datetime.now(),
            chunks_processed=0,
            total_chunks=0,
            error_message=None
        )
        
        self.active_jobs[job_id] = job
        self.logger.info(f"Trabajo de ingesta creado: {job_id} para {source_path}")
        
        return job_id
    
    def process_file(self, file_path: str, source_type: str = 'document') -> List[DocumentChunk]:
        """Procesar archivo usando DocumentProcessor real"""
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        try:
            self.logger.info(f"Iniciando procesamiento real de: {file_path}")
            
            # USAR PROCESADOR REAL - ELIMINADO MOCK COMPLETAMENTE
            chunks = self.processor.process_file(file_path)
            
            if not chunks:
                self.logger.warning(f"No se generaron chunks para: {file_path}")
                return []
            
            # Enriquecer metadatos
            for chunk in chunks:
                chunk.metadata.update({
                    'source_path': file_path,
                    'source_type': source_type,
                    'processed_at': datetime.now().isoformat(),
                    'processor': 'DocumentProcessor_v1.0'
                })
            
            self.logger.info(f"Procesado exitoso: {file_path} - {len(chunks)} chunks generados")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error procesando {file_path}: {e}")
            raise
    
    def process_directory(self, directory_path: str) -> Dict[str, Any]:
        """Procesar directorio completo"""
        
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {directory_path}")
        
        job_id = self.create_job(str(directory), 'directory')
        
        try:
            # Buscar archivos soportados
            supported_extensions = self.processor.get_supported_extensions()
            all_files = []
            
            for ext in supported_extensions:
                all_files.extend(directory.glob(f"*{ext}"))
                all_files.extend(directory.glob(f"**/*{ext}"))  # Recursivo
            
            if not all_files:
                return {
                    'job_id': job_id,
                    'status': 'completed',
                    'files_processed': 0,
                    'chunks_generated': 0,
                    'message': 'No se encontraron archivos soportados'
                }
            
            # Actualizar job con total de archivos
            self.active_jobs[job_id].total_chunks = len(all_files)
            self.active_jobs[job_id].status = 'processing'
            
            all_chunks = []
            processed_files = 0
            
            for file_path in all_files:
                try:
                    file_chunks = self.process_file(str(file_path), 'directory_file')
                    all_chunks.extend(file_chunks)
                    processed_files += 1
                    
                    # Actualizar progreso
                    self.active_jobs[job_id].chunks_processed = processed_files
                    
                except Exception as e:
                    self.logger.error(f"Error procesando {file_path}: {e}")
                    continue
            
            # Finalizar job
            self.active_jobs[job_id].status = 'completed'
            self.active_jobs[job_id].finished_at = datetime.now()
            
            return {
                'job_id': job_id,
                'status': 'completed',
                'files_processed': processed_files,
                'chunks_generated': len(all_chunks),
                'chunks': all_chunks
            }
            
        except Exception as e:
            self.active_jobs[job_id].status = 'failed'
            self.active_jobs[job_id].error_message = str(e)
            self.logger.error(f"Error procesando directorio {directory_path}: {e}")
            raise
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de trabajo"""
        job = self.active_jobs.get(job_id)
        if not job:
            return None
        
        return {
            'id': job.id,
            'status': job.status,
            'source_path': job.source_path,
            'job_type': job.job_type,
            'chunks_processed': job.chunks_processed,
            'total_chunks': job.total_chunks,
            'created_at': job.created_at.isoformat() if job.created_at else None,
            'finished_at': job.finished_at.isoformat() if job.finished_at else None,
            'error_message': job.error_message
        }
    
    def get_supported_extensions(self) -> List[str]:
        """Obtener extensiones soportadas del procesador real"""
        return self.processor.get_supported_extensions()
    
    def cleanup_completed_jobs(self, max_age_hours: int = 24):
        """Limpiar trabajos completados antiguos"""
        current_time = datetime.now()
        jobs_to_remove = []
        
        for job_id, job in self.active_jobs.items():
            if job.status in ['completed', 'failed'] and job.finished_at:
                age_hours = (current_time - job.finished_at).total_seconds() / 3600
                if age_hours > max_age_hours:
                    jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.active_jobs[job_id]
        
        if jobs_to_remove:
            self.logger.info(f"Limpiados {len(jobs_to_remove)} trabajos antiguos")


# INSTANCIA GLOBAL DEL SERVICIO
ingestion_service = IngestionService()

# FUNCIONES DE CONVENIENCIA PARA IMPORTACIÓN DIRECTA
def process_file(file_path: str, source_type: str = 'document') -> List[DocumentChunk]:
    """Función de conveniencia para procesar un archivo"""
    return ingestion_service.process_file(file_path, source_type)

def process_directory(directory_path: str) -> Dict[str, Any]:
    """Función de conveniencia para procesar un directorio"""
    return ingestion_service.process_directory(directory_path)

def get_supported_extensions() -> List[str]:
    """Función de conveniencia para obtener extensiones soportadas"""
    return ingestion_service.get_supported_extensions()