"""
Servicio de ingesta de documentos con detección de cambios
TFM Vicente Caruncho - Sistemas Inteligentes
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.core.logger import get_logger
from app.models.data_sources import (
    DocumentSource, ProcessedDocument, FileInfo, FileChange, 
    FileChangeType, ProcessingStatus, IngestionStats
)
from app.services.ingestion.document_processor import DocumentProcessor
from app.services.vector_store_service import VectorStoreService


class DocumentIngestionService:
    """Servicio para ingesta y monitoreo de documentos"""
    
    def __init__(self):
        self.logger = get_logger("document_ingestion")
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStoreService()
        
        # Storage para persistir estado (en producción sería base de datos)
        self.storage_file = Path("data/ingestion/document_sources.json")
        self.documents_file = Path("data/ingestion/processed_documents.json") 
        
        # Asegurar que existen los directorios
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Cache en memoria
        self._sources: Dict[str, DocumentSource] = {}
        self._documents: Dict[str, ProcessedDocument] = {}
        
        # Cargar datos persistentes
        self._load_data()
    
    def _load_data(self):
        """Cargar datos desde archivos JSON"""
        try:
            # Cargar fuentes de datos
            if self.storage_file.exists():
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    sources_data = json.load(f)
                    self._sources = {
                        sid: DocumentSource.from_dict(data) 
                        for sid, data in sources_data.items()
                    }
            
            # Cargar documentos procesados
            if self.documents_file.exists():
                with open(self.documents_file, 'r', encoding='utf-8') as f:
                    docs_data = json.load(f)
                    self._documents = {
                        did: ProcessedDocument.from_dict(data)
                        for did, data in docs_data.items()
                    }
                    
            self.logger.info(f"Cargados {len(self._sources)} fuentes y {len(self._documents)} documentos")
            
        except Exception as e:
            self.logger.error(f"Error cargando datos persistentes: {e}")
            self._sources = {}
            self._documents = {}
    
    def _save_data(self):
        """Guardar datos en archivos JSON"""
        try:
            # Guardar fuentes
            sources_data = {
                sid: source.to_dict() 
                for sid, source in self._sources.items()
            }
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(sources_data, f, indent=2, ensure_ascii=False)
            
            # Guardar documentos
            docs_data = {
                did: doc.to_dict()
                for did, doc in self._documents.items()
            }
            with open(self.documents_file, 'w', encoding='utf-8') as f:
                json.dump(docs_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Error guardando datos: {e}")
    
    def create_source(
        self, 
        name: str, 
        directories: List[str],
        **kwargs
    ) -> DocumentSource:
        """Crear nueva fuente de documentos"""
        source = DocumentSource(
            id=str(uuid.uuid4()),
            name=name,
            directories=directories,
            **kwargs
        )
        
        # Validar directorios
        valid_dirs = []
        for directory in directories:
            dir_path = Path(directory)
            if dir_path.exists() and dir_path.is_dir():
                valid_dirs.append(str(dir_path.absolute()))
            else:
                self.logger.warning(f"Directorio no válido: {directory}")
        
        if not valid_dirs:
            raise ValueError("No se encontraron directorios válidos")
        
        source.directories = valid_dirs
        self._sources[source.id] = source
        self._save_data()
        
        self.logger.info(f"Fuente creada: {source.name} ({source.id})")
        return source
    
    def get_source(self, source_id: str) -> Optional[DocumentSource]:
        """Obtener fuente por ID"""
        return self._sources.get(source_id)
    
    def list_sources(self) -> List[DocumentSource]:
        """Listar todas las fuentes"""
        return list(self._sources.values())
    
    def update_source(self, source_id: str, updates: Dict[str, Any]) -> bool:
        """Actualizar configuración de fuente"""
        if source_id not in self._sources:
            return False
        
        source = self._sources[source_id]
        
        # Actualizar campos permitidos
        updatable_fields = ['name', 'directories', 'file_extensions', 
                           'recursive', 'exclude_patterns', 'max_file_size']
        
        for field, value in updates.items():
            if field in updatable_fields:
                setattr(source, field, value)
        
        self._save_data()
        self.logger.info(f"Fuente actualizada: {source.name}")
        return True
    
    def delete_source(self, source_id: str) -> bool:
        """Eliminar fuente y sus documentos"""
        if source_id not in self._sources:
            return False
        
        # Eliminar documentos asociados
        docs_to_remove = [
            doc_id for doc_id, doc in self._documents.items()
            if doc.source_id == source_id
        ]
        
        for doc_id in docs_to_remove:
            del self._documents[doc_id]
        
        # Eliminar fuente
        source_name = self._sources[source_id].name
        del self._sources[source_id]
        
        self._save_data()
        self.logger.info(f"Fuente eliminada: {source_name} ({len(docs_to_remove)} documentos)")
        return True
    
    def scan_source(self, source_id: str) -> List[FileInfo]:
        """Escanear archivos en fuente de datos"""
        source = self.get_source(source_id)
        if not source:
            raise ValueError(f"Fuente no encontrada: {source_id}")
        
        self.logger.info(f"Escaneando fuente: {source.name}")
        files = source.scan_directories()
        
        self.logger.info(f"Encontrados {len(files)} archivos en {source.name}")
        return files
    
    def detect_changes(self, source_id: str) -> List[FileChange]:
        """Detectar cambios en archivos de una fuente"""
        source = self.get_source(source_id)
        if not source:
            raise ValueError(f"Fuente no encontrada: {source_id}")
        
        # Escanear archivos actuales
        current_files = self.scan_source(source_id)
        current_by_path = {f.path: f for f in current_files}
        
        # Obtener documentos procesados de esta fuente
        processed_docs = {
            doc.file_path: doc for doc in self._documents.values()
            if doc.source_id == source_id
        }
        
        changes = []
        
        # Detectar archivos nuevos y modificados
        for file_info in current_files:
            existing_doc = processed_docs.get(file_info.path)
            
            if not existing_doc:
                # Archivo nuevo
                changes.append(FileChange(
                    type=FileChangeType.NEW,
                    file_info=file_info
                ))
            elif existing_doc.file_hash != file_info.hash:
                # Archivo modificado
                changes.append(FileChange(
                    type=FileChangeType.MODIFIED,
                    file_info=file_info,
                    previous_info=existing_doc
                ))
        
        # Detectar archivos eliminados
        for doc_path, doc in processed_docs.items():
            if doc_path not in current_by_path:
                # Crear FileInfo dummy para archivo eliminado
                deleted_file = FileInfo(
                    path=doc.file_path,
                    size=doc.file_size,
                    modified_time=doc.modified_time,
                    hash=doc.file_hash,
                    extension=Path(doc.file_path).suffix
                )
                changes.append(FileChange(
                    type=FileChangeType.DELETED,
                    file_info=deleted_file,
                    previous_info=doc
                ))
        
        self.logger.info(f"Detectados {len(changes)} cambios en {source.name}")
        return changes
    
    def process_document(
        self, 
        source_id: str, 
        file_info: FileInfo,
        update_existing: bool = True
    ) -> ProcessedDocument:
        """Procesar un documento individual"""
        
        # Crear registro de documento procesado
        doc_id = str(uuid.uuid4())
        processed_doc = ProcessedDocument(
            id=doc_id,
            source_id=source_id,
            file_path=file_info.path,
            file_hash=file_info.hash,
            file_size=file_info.size,
            modified_time=file_info.modified_time,
            status=ProcessingStatus.PROCESSING
        )
        
        try:
            self.logger.info(f"Procesando: {Path(file_info.path).name}")
            
            # Procesar documento y generar chunks
            chunks = self.document_processor.process_file(file_info.path)
            
            if chunks:
                # Actualizar vector store
                self.vector_store.add_documents(chunks, source_metadata={
                    'source_id': source_id,
                    'document_id': doc_id,
                    'file_path': file_info.path
                })
                
                # Actualizar estado exitoso
                processed_doc.chunks_count = len(chunks)
                processed_doc.status = ProcessingStatus.COMPLETED
                processed_doc.processed_at = datetime.now()
                
                self.logger.info(f"Procesado exitosamente: {len(chunks)} chunks")
            else:
                processed_doc.status = ProcessingStatus.SKIPPED
                processed_doc.error_message = "No se generaron chunks"
                
        except Exception as e:
            self.logger.error(f"Error procesando {file_info.path}: {e}")
            processed_doc.status = ProcessingStatus.ERROR
            processed_doc.error_message = str(e)
        
        # Guardar en memoria y persistir
        self._documents[doc_id] = processed_doc
        self._save_data()
        
        return processed_doc
    
    def process_changes(
        self, 
        source_id: str, 
        changes: List[FileChange],
        max_workers: int = 3
    ) -> Dict[str, Any]:
        """Procesar lista de cambios de archivos"""
        
        source = self.get_source(source_id)
        if not source:
            raise ValueError(f"Fuente no encontrada: {source_id}")
        
        results = {
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'deleted': 0,
            'details': []
        }
        
        start_time = time.time()
        
        # Procesar eliminaciones primero
        for change in changes:
            if change.type == FileChangeType.DELETED:
                if change.previous_info:
                    # Eliminar de vector store y memoria
                    self._remove_document(change.previous_info.id)
                    results['deleted'] += 1
                    results['details'].append({
                        'file': change.file_info.path,
                        'action': 'deleted',
                        'status': 'success'
                    })
        
        # Procesar nuevos y modificados en paralelo
        process_changes = [c for c in changes if c.type in [FileChangeType.NEW, FileChangeType.MODIFIED]]
        
        if process_changes:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Enviar tareas
                future_to_change = {
                    executor.submit(self.process_document, source_id, change.file_info): change
                    for change in process_changes
                }
                
                # Procesar resultados
                for future in as_completed(future_to_change):
                    change = future_to_change[future]
                    try:
                        processed_doc = future.result()
                        
                        if processed_doc.status == ProcessingStatus.COMPLETED:
                            results['processed'] += 1
                            status = 'success'
                        elif processed_doc.status == ProcessingStatus.SKIPPED:
                            results['skipped'] += 1
                            status = 'skipped'
                        else:
                            results['failed'] += 1
                            status = 'failed'
                        
                        results['details'].append({
                            'file': change.file_info.path,
                            'action': change.type.value,
                            'status': status,
                            'chunks': processed_doc.chunks_count,
                            'error': processed_doc.error_message if processed_doc.error_message else None
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Error procesando cambio: {e}")
                        results['failed'] += 1
                        results['details'].append({
                            'file': change.file_info.path,
                            'action': change.type.value,
                            'status': 'failed',
                            'error': str(e)
                        })
        
        # Actualizar timestamp de última sincronización
        source.last_sync = datetime.now()
        self._save_data()
        
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        
        self.logger.info(
            f"Procesamiento completado en {processing_time:.2f}s: "
            f"{results['processed']} procesados, {results['failed']} fallidos, "
            f"{results['skipped']} omitidos, {results['deleted']} eliminados"
        )
        
        return results
    
    def sync_source(self, source_id: str, max_workers: int = 3) -> Dict[str, Any]:
        """Sincronizar fuente completa (detectar cambios y procesarlos)"""
        changes = self.detect_changes(source_id)
        
        if not changes:
            return {
                'processed': 0,
                'failed': 0,
                'skipped': 0,
                'deleted': 0,
                'message': 'No hay cambios detectados',
                'processing_time': 0
            }
        
        return self.process_changes(source_id, changes, max_workers)
    
    def _remove_document(self, document_id: str):
        """Eliminar documento del sistema"""
        if document_id in self._documents:
            doc = self._documents[document_id]
            
            # Eliminar del vector store
            try:
                self.vector_store.remove_document(document_id)
            except Exception as e:
                self.logger.warning(f"Error eliminando del vector store: {e}")
            
            # Eliminar de memoria
            del self._documents[document_id]
            
            self.logger.info(f"Documento eliminado: {Path(doc.file_path).name}")
    
    def get_source_stats(self, source_id: str) -> IngestionStats:
        """Obtener estadísticas de una fuente"""
        source = self.get_source(source_id)
        if not source:
            raise ValueError(f"Fuente no encontrada: {source_id}")
        
        # Filtrar documentos de esta fuente
        source_docs = [
            doc for doc in self._documents.values()
            if doc.source_id == source_id
        ]
        
        stats = IngestionStats(source_id=source_id)
        
        for doc in source_docs:
            stats.total_files += 1
            stats.total_size_bytes += doc.file_size
            stats.total_chunks += doc.chunks_count
            
            if doc.status == ProcessingStatus.COMPLETED:
                stats.processed_files += 1
            elif doc.status == ProcessingStatus.ERROR:
                stats.failed_files += 1
        
        stats.last_scan = source.last_sync
        
        return stats
    
    def get_all_stats(self) -> List[IngestionStats]:
        """Obtener estadísticas de todas las fuentes"""
        return [
            self.get_source_stats(source_id)
            for source_id in self._sources.keys()
        ]
    
    def get_source_documents(self, source_id: str) -> List[ProcessedDocument]:
        """Obtener documentos procesados de una fuente"""
        return [
            doc for doc in self._documents.values()
            if doc.source_id == source_id
        ]
    
    def get_processing_logs(
        self, 
        source_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Obtener logs de procesamiento recientes"""
        
        # Filtrar documentos
        docs = self._documents.values()
        if source_id:
            docs = [doc for doc in docs if doc.source_id == source_id]
        
        # Ordenar por fecha de procesamiento (más recientes primero)
        sorted_docs = sorted(
            docs,
            key=lambda d: d.processed_at or d.modified_time,
            reverse=True
        )[:limit]
        
        logs = []
        for doc in sorted_docs:
            source = self.get_source(doc.source_id)
            logs.append({
                'timestamp': doc.processed_at or doc.modified_time,
                'source_name': source.name if source else 'Unknown',
                'file_name': Path(doc.file_path).name,
                'file_path': doc.file_path,
                'status': doc.status.value,
                'chunks_count': doc.chunks_count,
                'file_size': doc.file_size,
                'error_message': doc.error_message
            })
        
        return logs
    
    def cleanup_deleted_files(self, source_id: str) -> int:
        """Limpiar referencias a archivos que ya no existen"""
        removed_count = 0
        docs_to_remove = []
        
        source_docs = self.get_source_documents(source_id)
        
        for doc in source_docs:
            if not Path(doc.file_path).exists():
                docs_to_remove.append(doc.id)
        
        for doc_id in docs_to_remove:
            self._remove_document(doc_id)
            removed_count += 1
        
        if removed_count > 0:
            self._save_data()
            self.logger.info(f"Limpiados {removed_count} documentos eliminados")
        
        return removed_count


# Instancia singleton para uso global
document_ingestion_service = DocumentIngestionService()

# Exportar para importación
__all__ = ['DocumentIngestionService', 'document_ingestion_service']