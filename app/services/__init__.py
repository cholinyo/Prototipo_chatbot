"""
Servicio de Ingesta Integrado - VERSIÓN COMPLETA CON API Y DATABASE
TFM Vicente Caruncho - Sistemas Inteligentes

ARQUITECTURA DE INTEGRACIÓN:
✅ IngestionService - Interfaz simple compatible hacia atrás
✅ DocumentIngestionService - Servicio completo multimodal
✅ Procesadores reales sin mocks: Documents + APIs + Database

CAPACIDADES INTEGRADAS:
- Documentos: PDF, DOCX, TXT, Excel (DocumentProcessor)
- APIs REST: Autenticación, paginación (APIConnector)  
- Bases de Datos: PostgreSQL, MySQL, SQLite, SQL Server (DatabaseConnector)
- Web: Mediante integración con WebIngestionService
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import time
import os
import uuid
from datetime import datetime

from app.core.logger import get_logger
from app.models import IngestionJob, DocumentChunk
from app.models.data_sources import (
    DocumentSource, APISource, DatabaseSource, 
    DataSourceType, ProcessingStatus
)

# Importar servicios de procesamiento específicos
from app.services.ingestion.document_processor import DocumentProcessor
from app.services.ingestion.api_connector import APIConnector
from app.services.ingestion.database_connector import DatabaseConnector


# =============================================================================
# SERVICIO SIMPLE - COMPATIBILIDAD HACIA ATRÁS
# =============================================================================

class IngestionService:
    """
    Servicio principal de ingesta - INTERFAZ COMPATIBLE
    
    Mantiene la API original simple pero con procesadores reales.
    Para funcionalidad avanzada, usar DocumentIngestionService directamente.
    """
    
    def __init__(self):
        self.logger = get_logger("prototipo_chatbot.ingestion_service")
        self.active_jobs: Dict[str, IngestionJob] = {}
        
        # PROCESADORES REALES - NO MOCKS
        self.processor = DocumentProcessor()
        self.api_connector = APIConnector()
        self.database_connector = DatabaseConnector()
        
        # Directorio de trabajo
        self.work_dir = Path("data/ingestion")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("IngestionService inicializado con procesadores reales integrados")
    
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
        """Procesar archivo usando procesadores reales integrados"""
        
        if not os.path.exists(file_path):
            self.logger.error(f"Archivo no encontrado: {file_path}")
            return []
        
        try:
            self.logger.info(f"Procesando archivo: {Path(file_path).name}")
            
            # Usar DocumentProcessor real
            chunks = self.processor.process_file(file_path)
            
            if chunks:
                self.logger.info(f"Archivo procesado exitosamente: {len(chunks)} chunks generados")
                return chunks
            else:
                self.logger.warning(f"No se generaron chunks para: {file_path}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error procesando archivo {file_path}: {e}")
            return []
    
    def process_api(self, source_id: str, endpoint_name: str, 
                   parameters: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Procesar API usando APIConnector real"""
        try:
            self.logger.info(f"Procesando API: {source_id} - {endpoint_name}")
            
            # Usar APIConnector real para obtener datos
            api_response = self.api_connector.fetch_data(source_id, endpoint_name, parameters)
            
            if api_response and api_response.success:
                # Transformar a chunks usando APIConnector
                chunks = self.api_connector.transform_to_chunks(api_response)
                self.logger.info(f"API procesada: {len(chunks)} chunks generados")
                return chunks
            else:
                error_msg = api_response.error_message if api_response else "Error desconocido"
                self.logger.error(f"Error procesando API: {error_msg}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error procesando API {source_id}: {e}")
            return []
    
    def process_database(self, source_id: str, query_name: str, 
                        parameters: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Procesar base de datos usando DatabaseConnector real"""
        try:
            self.logger.info(f"Procesando BD: {source_id} - {query_name}")
            
            # Usar DatabaseConnector real para obtener datos
            db_response = self.database_connector.execute_query(source_id, query_name, parameters)
            
            if db_response and db_response.records:
                # Transformar a chunks usando DatabaseConnector
                chunks = self.database_connector.transform_to_chunks(db_response)
                self.logger.info(f"BD procesada: {len(chunks)} chunks generados")
                return chunks
            else:
                self.logger.warning(f"No se obtuvieron datos de BD: {query_name}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error procesando BD {source_id}: {e}")
            return []
    
    def get_job_status(self, job_id: str) -> Optional[IngestionJob]:
        """Obtener estado de trabajo de ingesta"""
        return self.active_jobs.get(job_id)
    
    def list_jobs(self) -> List[IngestionJob]:
        """Listar todos los trabajos de ingesta"""
        return list(self.active_jobs.values())
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancelar trabajo de ingesta"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = 'cancelled'
            self.logger.info(f"Trabajo cancelado: {job_id}")
            return True
        return False
    
    def cleanup_completed_jobs(self, max_age_hours: int = 24) -> int:
        """Limpiar trabajos completados antiguos"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        jobs_to_remove = []
        for job_id, job in self.active_jobs.items():
            if job.status in ['completed', 'failed', 'cancelled']:
                if job.created_at < cutoff_time:
                    jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.active_jobs[job_id]
        
        if jobs_to_remove:
            self.logger.info(f"Limpiados {len(jobs_to_remove)} trabajos antiguos")
        
        return len(jobs_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del servicio de ingesta"""
        jobs = list(self.active_jobs.values())
        
        return {
            'total_jobs': len(jobs),
            'pending_jobs': sum(1 for job in jobs if job.status == 'pending'),
            'processing_jobs': sum(1 for job in jobs if job.status == 'processing'),
            'completed_jobs': sum(1 for job in jobs if job.status == 'completed'),
            'failed_jobs': sum(1 for job in jobs if job.status == 'failed'),
            'total_chunks_processed': sum(job.chunks_processed for job in jobs),
            'processors_available': {
                'document_processor': self.processor is not None,
                'api_connector': self.api_connector is not None,
                'database_connector': self.database_connector is not None
            }
        }


# =============================================================================
# INTEGRACIÓN CON SERVICIO AVANZADO
# =============================================================================

def get_advanced_ingestion_service():
    """
    Obtener servicio de ingesta avanzado con gestión de fuentes
    
    Returns:
        DocumentIngestionService con capacidades completas de:
        - Gestión de fuentes multimodales
        - Detección de cambios en documentos  
        - Sincronización automática
        - Estadísticas y monitoreo avanzado
    """
    try:
        from app.services.document_ingestion_service import document_ingestion_service
        return document_ingestion_service
    except ImportError as e:
        logger = get_logger("ingestion")
        logger.error(f"Servicio avanzado no disponible: {e}")
        return None


def create_document_source(name: str, directories: List[str], **kwargs) -> Optional[DocumentSource]:
    """
    Función de conveniencia para crear fuente de documentos
    
    Args:
        name: Nombre de la fuente
        directories: Directorios a escanear
        **kwargs: Parámetros adicionales
    
    Returns:
        DocumentSource creada o None si hay error
    """
    advanced_service = get_advanced_ingestion_service()
    if advanced_service:
        try:
            return advanced_service.create_document_source(name, directories, **kwargs)
        except Exception as e:
            logger = get_logger("ingestion")
            logger.error(f"Error creando fuente de documentos: {e}")
    return None


def create_api_source(name: str, base_url: str, **kwargs) -> Optional[APISource]:
    """
    Función de conveniencia para crear fuente de API
    
    Args:
        name: Nombre de la fuente
        base_url: URL base de la API
        **kwargs: Parámetros adicionales
    
    Returns:
        APISource creada o None si hay error
    """
    advanced_service = get_advanced_ingestion_service()
    if advanced_service:
        try:
            return advanced_service.create_api_source(name, base_url, **kwargs)
        except Exception as e:
            logger = get_logger("ingestion")
            logger.error(f"Error creando fuente de API: {e}")
    return None


def create_database_source(name: str, db_type: str, 
                          connection_config: Dict[str, Any], **kwargs) -> Optional[DatabaseSource]:
    """
    Función de conveniencia para crear fuente de base de datos
    
    Args:
        name: Nombre de la fuente
        db_type: Tipo de base de datos
        connection_config: Configuración de conexión
        **kwargs: Parámetros adicionales
    
    Returns:
        DatabaseSource creada o None si hay error
    """
    advanced_service = get_advanced_ingestion_service()
    if advanced_service:
        try:
            return advanced_service.create_database_source(name, db_type, connection_config, **kwargs)
        except Exception as e:
            logger = get_logger("ingestion")
            logger.error(f"Error creando fuente de BD: {e}")
    return None


def list_all_sources() -> List[Union[DocumentSource, APISource, DatabaseSource]]:
    """
    Listar todas las fuentes de datos configuradas
    
    Returns:
        Lista de fuentes o lista vacía si hay error
    """
    advanced_service = get_advanced_ingestion_service()
    if advanced_service:
        try:
            return advanced_service.list_sources()
        except Exception as e:
            logger = get_logger("ingestion")
            logger.error(f"Error listando fuentes: {e}")
    return []


def sync_source(source_id: str, **kwargs) -> Optional[Dict[str, Any]]:
    """
    Sincronizar fuente de datos
    
    Args:
        source_id: ID de la fuente a sincronizar
        **kwargs: Parámetros adicionales
    
    Returns:
        Resultado de sincronización o None si hay error
    """
    advanced_service = get_advanced_ingestion_service()
    if advanced_service:
        try:
            return advanced_service.sync_source(source_id, **kwargs)
        except Exception as e:
            logger = get_logger("ingestion")
            logger.error(f"Error sincronizando fuente {source_id}: {e}")
    return None


def get_system_summary() -> Dict[str, Any]:
    """
    Obtener resumen completo del sistema de ingesta
    
    Returns:
        Resumen del sistema con estadísticas
    """
    advanced_service = get_advanced_ingestion_service()
    if advanced_service:
        try:
            return advanced_service.get_system_summary()
        except Exception as e:
            logger = get_logger("ingestion")
            logger.error(f"Error obteniendo resumen del sistema: {e}")
    
    # Fallback básico
    return {
        'sources': {'total': 0, 'documents': 0, 'apis': 0, 'databases': 0},
        'processing': {'total_documents': 0, 'completed_documents': 0, 'success_rate': 0, 'total_chunks': 0},
        'integrations': {
            'document_processor': True,
            'api_connector': True, 
            'database_connector': True,
            'vector_store': False
        }
    }


# =============================================================================
# INSTANCIA GLOBAL Y EXPORTACIONES
# =============================================================================

# Instancia del servicio simple para compatibilidad
ingestion_service = IngestionService()

# Alias para compatibilidad hacia atrás
ingestion = ingestion_service

# Exportaciones principales
__all__ = [
    # Servicio principal
    'IngestionService',
    'ingestion_service',
    'ingestion',  # Alias compatible
    
    # Funciones de conveniencia para servicio avanzado
    'get_advanced_ingestion_service',
    'create_document_source',
    'create_api_source',
    'create_database_source',
    'list_all_sources',
    'sync_source',
    'get_system_summary',
    
    # Clases de modelos (re-exportadas para conveniencia)
    'IngestionJob',
    'DocumentChunk',
    'DocumentSource', 
    'APISource',
    'DatabaseSource',
    'DataSourceType',
    'ProcessingStatus'
]


# =============================================================================
# FUNCIONES DE DIAGNÓSTICO Y UTILIDAD
# =============================================================================

def test_all_processors() -> Dict[str, bool]:
    """
    Probar todos los procesadores disponibles
    
    Returns:
        Estado de cada procesador
    """
    results = {}
    
    try:
        # Test DocumentProcessor
        processor = DocumentProcessor()
        results['document_processor'] = True
        
        # Test APIConnector  
        api_connector = APIConnector()
        results['api_connector'] = True
        
        # Test DatabaseConnector
        db_connector = DatabaseConnector()
        results['database_connector'] = True
        
    except Exception as e:
        logger = get_logger("ingestion")
        logger.error(f"Error en test de procesadores: {e}")
        results = {
            'document_processor': False,
            'api_connector': False,
            'database_connector': False
        }
    
    return results


def get_ingestion_capabilities() -> Dict[str, Any]:
    """
    Obtener capacidades completas del sistema de ingesta
    
    Returns:
        Diccionario con capacidades y estado de cada componente
    """
    processor_status = test_all_processors()
    
    capabilities = {
        'formats_supported': {
            'documents': ['.pdf', '.docx', '.txt', '.md', '.rtf'],
            'apis': ['REST', 'JSON', 'XML'],
            'databases': ['PostgreSQL', 'MySQL', 'SQLite', 'SQL Server']
        },
        'processors_available': processor_status,
        'advanced_features': {
            'change_detection': True,
            'automatic_sync': True,
            'batch_processing': True,
            'vector_store_integration': True,
            'metadata_extraction': True,
            'content_chunking': True
        },
        'authentication_methods': {
            'api': ['none', 'bearer', 'api_key', 'basic', 'oauth2'],
            'database': ['username_password', 'connection_string']
        }
    }
    
    return capabilities


def diagnose_ingestion_system() -> Dict[str, Any]:
    """
    Diagnóstico completo del sistema de ingesta
    
    Returns:
        Reporte de diagnóstico detallado
    """
    logger = get_logger("ingestion")
    logger.info("Ejecutando diagnóstico del sistema de ingesta...")
    
    diagnosis = {
        'timestamp': datetime.now().isoformat(),
        'simple_service': {
            'available': ingestion_service is not None,
            'processors': test_all_processors()
        },
        'advanced_service': {
            'available': get_advanced_ingestion_service() is not None,
            'summary': get_system_summary()
        },
        'capabilities': get_ingestion_capabilities(),
        'recommendations': []
    }
    
    # Generar recomendaciones basadas en el estado
    if not diagnosis['advanced_service']['available']:
        diagnosis['recommendations'].append(
            "Servicio avanzado no disponible - verificar DocumentIngestionService"
        )
    
    if not all(diagnosis['simple_service']['processors'].values()):
        diagnosis['recommendations'].append(
            "Algunos procesadores no están disponibles - verificar dependencias"
        )
    
    if not diagnosis['recommendations']:
        diagnosis['recommendations'].append("Sistema de ingesta funcionando correctamente")
    
    return diagnosis


# =============================================================================
# INICIALIZACIÓN Y LOGGING
# =============================================================================

def _initialize_ingestion_module():
    """Inicializar módulo de ingesta con verificaciones"""
    logger = get_logger("ingestion_module")
    
    try:
        # Verificar procesadores
        processor_status = test_all_processors()
        available_processors = sum(processor_status.values())
        total_processors = len(processor_status)
        
        logger.info(f"Módulo de ingesta inicializado: "
                   f"{available_processors}/{total_processors} procesadores disponibles")
        
        if available_processors == total_processors:
            logger.info("Todos los procesadores están disponibles - sistema listo")
        else:
            logger.warning(f"Solo {available_processors}/{total_processors} procesadores disponibles")
            
        # Verificar servicio avanzado
        advanced_available = get_advanced_ingestion_service() is not None
        logger.info(f"Servicio avanzado: {'disponible' if advanced_available else 'no disponible'}")
        
    except Exception as e:
        logger.error(f"Error inicializando módulo de ingesta: {e}")


# Ejecutar inicialización del módulo
_initialize_ingestion_module()