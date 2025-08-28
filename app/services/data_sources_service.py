"""
REFACTORIZACIÓN: Servicio de fuentes de datos - TODA la lógica operacional
TFM Vicente Caruncho - Sistemas Inteligentes

RESPONSABILIDADES DEL SERVICIO:
1. Toda la lógica de negocio que ANTES estaba en los modelos
2. Gestión de persistencia y cache
3. Coordinación entre procesadores específicos
4. Validaciones operacionales
5. Detección de cambios y sincronización
6. Estadísticas y monitoreo

MODELOS: Solo estructuras de datos + serialización
SERVICIO: Toda la lógica operacional
"""

import json
import time
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.core.logger import get_logger
from app.models.data_sources import (
    # SOLO importar estructuras de datos limpias
    DocumentSource, WebSource, APISource, DatabaseSource,
    ProcessedDocument, ScrapedPage, FileInfo, FileChange, 
    FileChangeType, ProcessingStatus, IngestionStats,
    DataSourceType, DataSourceStatus,
    create_document_source, create_web_source, 
    create_api_source, create_database_source
)

# Importar procesadores específicos
from app.services.ingestion.document_processor import DocumentProcessor
from app.services.ingestion.api_connector import APIConnector  
from app.services.ingestion.database_connector import DatabaseConnector
from app.services.vector_store_service import VectorStoreService

# Importar web scraping si está disponible
try:
    from app.services.web_ingestion_service import WebIngestionService
    WEB_INGESTION_AVAILABLE = True
except ImportError:
    WebIngestionService = None
    WEB_INGESTION_AVAILABLE = False


class DataSourcesService:
    """
    SERVICIO UNIFICADO para gestión de fuentes de datos
    
    CONTIENE TODA LA LÓGICA que antes estaba dispersa entre modelos y servicios:
    - Lógica de validación
    - Operaciones de escaneo y detección de cambios
    - Procesamiento y sincronización  
    - Persistencia y cache
    - Coordinación entre procesadores
    """
    
    def __init__(self):
        self.logger = get_logger("data_sources_service")
        
        # Procesadores específicos
        self.document_processor = DocumentProcessor()
        self.api_connector = APIConnector() 
        self.database_connector = DatabaseConnector()
        self.vector_store = VectorStoreService()
        
        # Web scraping si está disponible
        if WEB_INGESTION_AVAILABLE:
            self.web_ingestion = WebIngestionService()
        else:
            self.web_ingestion = None
        
        # Archivos de persistencia
        self.storage_files = {
            'documents': Path("data/ingestion/document_sources.json"),
            'api': Path("data/ingestion/api_sources.json"),
            'database': Path("data/ingestion/database_sources.json"),
            'web': Path("data/ingestion/web_sources.json")
        }
        
        self.documents_file = Path("data/ingestion/processed_documents.json")
        self.scraped_pages_file = Path("data/ingestion/scraped_pages.json")
        
        # Crear directorios
        for storage_file in self.storage_files.values():
            storage_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Cache en memoria
        self._sources: Dict[str, Union[DocumentSource, WebSource, APISource, DatabaseSource]] = {}
        self._documents: Dict[str, ProcessedDocument] = {}
        self._scraped_pages: Dict[str, ScrapedPage] = {}
        
        # Cargar datos
        self._load_data()
        
        self.logger.info(f"DataSourcesService inicializado - Web: {'✓' if WEB_INGESTION_AVAILABLE else '✗'}")
    
    # =========================================================================
    # LÓGICA DE VALIDACIÓN - Movida desde los modelos
    # =========================================================================
    
    def create_file_info(self, file_path: str) -> FileInfo:
        """
        Crear FileInfo desde ruta - MOVIDO desde FileInfo.from_path()
        """
        path = Path(file_path)
        stat = path.stat()
        
        # Calcular hash MD5
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        except (IOError, OSError):
            hash_md5.update(b"")
        
        return FileInfo(
            path=str(path.absolute()),
            size=stat.st_size,
            modified_time=datetime.fromtimestamp(stat.st_mtime),
            hash=hash_md5.hexdigest(),
            extension=path.suffix.lower()
        )
    
    def is_file_supported(self, source: DocumentSource, file_path: str) -> bool:
        """
        Verificar si archivo es soportado - MOVIDO desde DocumentSource.is_file_supported()
        """
        path = Path(file_path)
        
        # Verificar extensión
        if path.suffix.lower() not in source.file_extensions:
            return False
        
        # Verificar patrones de exclusión
        for pattern in source.exclude_patterns:
            if pattern in str(path):
                return False
        
        # Verificar tamaño
        try:
            if path.stat().st_size > source.max_file_size:
                return False
        except (OSError, IOError):
            return False
        
        return True
    
    def is_url_allowed(self, source: WebSource, url: str) -> bool:
        """
        Verificar si URL es permitida - MOVIDO desde WebSource.is_url_allowed()
        """
        domain = urlparse(url).netloc
        
        # Verificar dominio
        if source.allowed_domains and domain not in source.allowed_domains:
            return False
        
        # Verificar patrones de exclusión
        for pattern in source.exclude_patterns:
            if pattern in url:
                return False
        
        # Verificar extensiones excluidas
        for ext in source.exclude_file_extensions:
            if url.lower().endswith(ext):
                return False
        
        # Verificar patrones de inclusión
        if source.include_patterns:
            return any(pattern in url for pattern in source.include_patterns)
        
        return True
    
    def get_auth_headers(self, source: APISource) -> Dict[str, str]:
        """
        Generar headers de autenticación - MOVIDO desde APISource.get_auth_headers()
        """
        headers = source.default_headers.copy()
        
        if source.auth_type == "bearer" and "token" in source.auth_credentials:
            headers["Authorization"] = f"Bearer {source.auth_credentials['token']}"
        elif source.auth_type == "api_key":
            key_name = source.auth_credentials.get("key_name", "X-API-Key")
            headers[key_name] = source.auth_credentials.get("key_value", "")
        elif source.auth_type == "basic":
            import base64
            user = source.auth_credentials.get("username", "")
            password = source.auth_credentials.get("password", "")
            credentials = base64.b64encode(f"{user}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials}"
        
        return headers
    
    def is_api_response_valid(self, source: APISource, response_data: Any) -> bool:
        """
        Validar respuesta API - MOVIDO desde APISource.is_response_valid()
        """
        if not response_data:
            return False
        
        if isinstance(response_data, str):
            return len(response_data.strip()) >= source.min_content_length
        
        if isinstance(response_data, (dict, list)):
            content_str = json.dumps(response_data)
            return len(content_str) >= source.min_content_length
        
        return True
    
    def get_db_connection_string(self, source: DatabaseSource) -> str:
        """
        Generar string de conexión - MOVIDO desde DatabaseSource.get_connection_string()
        """
        config = source.connection_config
        
        if source.db_type == "postgresql":
            return (f"postgresql://{config['user']}:{config['password']}@"
                   f"{config['host']}:{config['port']}/{config['database']}")
        elif source.db_type == "mysql":
            return (f"mysql://{config['user']}:{config['password']}@"
                   f"{config['host']}:{config['port']}/{config['database']}")
        elif source.db_type == "sqlite":
            return f"sqlite:///{config['database']}"
        elif source.db_type == "mssql":
            return (f"mssql://{config['user']}:{config['password']}@"
                   f"{config['host']}:{config['port']}/{config['database']}")
        
        raise ValueError(f"Tipo de BD no soportado: {source.db_type}")
    
    def is_db_record_valid(self, source: DatabaseSource, record: Dict[str, Any]) -> bool:
        """
        Validar registro de BD - MOVIDO desde DatabaseSource.is_record_valid()
        """
        if not record:
            return False
        
        # Verificar campos de contenido específicos
        if source.content_fields:
            for field in source.content_fields:
                if field in record and record[field]:
                    value = str(record[field]).strip()
                    if len(value) >= source.min_content_length:
                        return True
            return False
        
        # Verificar contenido general
        total_content = ' '.join(str(v) for v in record.values() if v is not None)
        return len(total_content.strip()) >= source.min_content_length
    
    def update_scraped_page_status(self, page: ScrapedPage, status: ProcessingStatus, 
                                  chunks_count: int = 0, error_message: str = ""):
        """
        Actualizar estado de página - MOVIDO desde ScrapedPage.update_processing_status()
        """
        page.processing_status = status
        page.chunks_count = chunks_count
        page.error_message = error_message
        
        if status == ProcessingStatus.COMPLETED:
            page.processed_at = datetime.now()
    
    # =========================================================================
    # OPERACIONES DE ESCANEO - Toda la lógica operacional
    # =========================================================================
    
    def scan_document_directories(self, source: DocumentSource) -> List[FileInfo]:
        """
        Escanear directorios - MOVIDO desde DocumentSource.scan_directories()
        """
        files = []
        
        for directory in source.directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                self.logger.warning(f"Directorio no existe: {directory}")
                continue
            
            pattern = "**/*" if source.recursive else "*"
            
            for file_path in dir_path.glob(pattern):
                if file_path.is_file() and self.is_file_supported(source, str(file_path)):
                    try:
                        file_info = self.create_file_info(str(file_path))
                        files.append(file_info)
                    except (OSError, IOError) as e:
                        self.logger.warning(f"Error accediendo a archivo {file_path}: {e}")
                        continue
        
        self.logger.info(f"Escaneados {len(files)} archivos en fuente {source.name}")
        return files
    
    def test_web_source_connectivity(self, source: WebSource) -> Dict[str, Any]:
        """
        Test de conectividad web - NUEVA funcionalidad específica del servicio
        """
        if not WEB_INGESTION_AVAILABLE or not self.web_ingestion:
            return {'accessible': False, 'error': 'WebIngestionService no disponible'}
        
        test_results = []
        for url in source.base_urls[:3]:  # Test solo primeras 3 URLs
            try:
                # Usar el servicio web para test
                from app.services.web_scraper_service import web_scraper_service
                result = web_scraper_service.test_url(url)
                test_results.append(result)
            except Exception as e:
                test_results.append({'accessible': False, 'url': url, 'error': str(e)})
        
        return {
            'total_urls': len(source.base_urls),
            'tested_urls': len(test_results),
            'accessible_count': sum(1 for r in test_results if r.get('accessible', False)),
            'results': test_results
        }
    
    def test_api_source_connectivity(self, source: APISource) -> Dict[str, Any]:
        """
        Test de conectividad API - NUEVA funcionalidad específica del servicio
        """
        try:
            return self.api_connector.test_connection(source.id, source)
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_database_source_connectivity(self, source: DatabaseSource) -> Dict[str, Any]:
        """
        Test de conectividad BD - NUEVA funcionalidad específica del servicio
        """
        try:
            return self.database_connector.test_connection(source.id, source)
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # =========================================================================
    # DETECCIÓN DE CAMBIOS - Lógica específica del servicio
    # =========================================================================
    
    def detect_document_changes(self, source_id: str) -> List[FileChange]:
        """
        Detectar cambios en documentos - LÓGICA ESPECÍFICA DEL SERVICIO
        """
        source = self.get_source(source_id)
        if not source or not isinstance(source, DocumentSource):
            raise ValueError("Detección de cambios solo para fuentes de documentos")
        
        # Escanear archivos actuales
        current_files = self.scan_document_directories(source)
        current_by_path = {f.path: f for f in current_files}
        
        # Obtener documentos procesados
        processed_docs = {
            doc.file_path: doc for doc in self._documents.values()
            if doc.source_id == source_id
        }
        
        changes = []
        
        # Detectar nuevos y modificados
        for file_info in current_files:
            existing_doc = processed_docs.get(file_info.path)
            
            if not existing_doc:
                changes.append(FileChange(
                    type=FileChangeType.NEW,
                    file_info=file_info
                ))
            elif existing_doc.file_hash != file_info.hash:
                changes.append(FileChange(
                    type=FileChangeType.MODIFIED,
                    file_info=file_info,
                    previous_info=existing_doc
                ))
        
        # Detectar eliminados
        for doc_path, doc in processed_docs.items():
            if doc_path not in current_by_path:
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
    
    # =========================================================================
    # GESTIÓN DE FUENTES - Lógica CRUD completa
    # =========================================================================
    
    def create_document_source(self, name: str, directories: List[str], **kwargs) -> DocumentSource:
        """Crear fuente de documentos con validación"""
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
        
        # Crear usando factory function limpia
        source = create_document_source(
            name=name,
            directories=valid_dirs,
            **kwargs
        )
        
        # Guardar
        self._sources[source.id] = source
        self._save_data()
        
        self.logger.info(f"Fuente de documentos creada: {source.name} ({source.id})")
        return source
    
    def create_web_source(self, name: str, base_urls: List[str], **kwargs) -> WebSource:
        """Crear fuente web con validación"""
        if not WEB_INGESTION_AVAILABLE:
            raise ValueError("WebIngestionService no está disponible")
        
        # Crear usando factory function limpia
        source = create_web_source(
            name=name,
            base_urls=base_urls,
            **kwargs
        )
        
        # Test de conectividad
        connectivity = self.test_web_source_connectivity(source)
        if connectivity['accessible_count'] > 0:
            source.status = DataSourceStatus.ACTIVE
        else:
            source.status = DataSourceStatus.PENDING
            self.logger.warning(f"Fuente web no accesible: {source.name}")
        
        # Guardar
        self._sources[source.id] = source
        self._save_data()
        
        self.logger.info(f"Fuente web creada: {source.name} ({source.id}) - {len(base_urls)} URLs")
        return source
    
    def create_api_source(self, name: str, base_url: str, **kwargs) -> APISource:
        """Crear fuente API con validación"""
        # Crear usando factory function limpia
        source = create_api_source(
            name=name,
            base_url=base_url,
            **kwargs
        )
        
        # Test de conectividad
        connectivity = self.test_api_source_connectivity(source)
        if connectivity.get('success'):
            source.status = DataSourceStatus.ACTIVE
        else:
            source.status = DataSourceStatus.PENDING
            self.logger.warning(f"API fuente no accesible: {connectivity.get('error')}")
        
        # Guardar
        self._sources[source.id] = source
        self._save_data()
        
        self.logger.info(f"Fuente API creada: {source.name} ({source.id})")
        return source
    
    def create_database_source(self, name: str, db_type: str, 
                             connection_config: Dict[str, Any], **kwargs) -> DatabaseSource:
        """Crear fuente BD con validación"""
        # Crear usando factory function limpia
        source = create_database_source(
            name=name,
            db_type=db_type,
            connection_config=connection_config,
            **kwargs
        )
        
        # Test de conectividad
        connectivity = self.test_database_source_connectivity(source)
        if connectivity.get('success'):
            source.status = DataSourceStatus.ACTIVE
        else:
            source.status = DataSourceStatus.PENDING
            self.logger.warning(f"BD fuente no accesible: {connectivity.get('error')}")
        
        # Guardar
        self._sources[source.id] = source
        self._save_data()
        
        self.logger.info(f"Fuente BD creada: {source.name} ({source.id}) - {db_type}")
        return source
    
    def get_source(self, source_id: str) -> Optional[Union[DocumentSource, WebSource, APISource, DatabaseSource]]:
        """Obtener fuente por ID"""
        return self._sources.get(source_id)
    
    def list_sources(self) -> List[Union[DocumentSource, WebSource, APISource, DatabaseSource]]:
        """Listar todas las fuentes"""
        return list(self._sources.values())
    
    def list_sources_by_type(self, source_type: DataSourceType) -> List[Union[DocumentSource, WebSource, APISource, DatabaseSource]]:
        """Listar fuentes por tipo"""
        return [
            source for source in self._sources.values()
            if source.type == source_type
        ]
    
    def delete_source(self, source_id: str) -> bool:
        """Eliminar fuente y contenido asociado"""
        if source_id not in self._sources:
            return False
        
        # Eliminar documentos asociados
        docs_to_remove = [
            doc_id for doc_id, doc in self._documents.items()
            if doc.source_id == source_id
        ]
        
        for doc_id in docs_to_remove:
            self._remove_document(doc_id)
        
        # Eliminar páginas asociadas
        pages_to_remove = [
            page_id for page_id, page in self._scraped_pages.items()
            if page.source_id == source_id
        ]
        
        for page_id in pages_to_remove:
            del self._scraped_pages[page_id]
        
        # Eliminar fuente
        source = self._sources[source_id]
        del self._sources[source_id]
        
        self._save_data()
        
        self.logger.info(f"Fuente eliminada: {source.name} "
                        f"({len(docs_to_remove)} documentos, {len(pages_to_remove)} páginas)")
        return True
    
    # =========================================================================
    # PROCESAMIENTO - Coordinación entre procesadores
    # =========================================================================
    
    def process_document_source(self, source_id: str, file_info: FileInfo) -> ProcessedDocument:
        """Procesar documento individual"""
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
            self.logger.info(f"Procesando documento: {Path(file_info.path).name}")
            
            # Usar procesador específico
            chunks = self.document_processor.process_file(file_info.path)
            
            if chunks:
                # Actualizar vector store
                self.vector_store.add_documents(chunks, source_metadata={
                    'source_id': source_id,
                    'document_id': doc_id,
                    'file_path': file_info.path
                })
                
                processed_doc.chunks_count = len(chunks)
                processed_doc.status = ProcessingStatus.COMPLETED
                processed_doc.processed_at = datetime.now()
                
                self.logger.info(f"Documento procesado: {len(chunks)} chunks")
            else:
                processed_doc.status = ProcessingStatus.SKIPPED
                processed_doc.error_message = "No se generaron chunks"
                
        except Exception as e:
            self.logger.error(f"Error procesando {file_info.path}: {e}")
            processed_doc.status = ProcessingStatus.ERROR
            processed_doc.error_message = str(e)
        
        # Guardar
        self._documents[doc_id] = processed_doc
        self._save_data()
        
        return processed_doc
    
    def process_changes(self, source_id: str, changes: List[FileChange],
                       max_workers: int = 3) -> Dict[str, Any]:
        """Procesar cambios detectados"""
        results = {
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'deleted': 0,
            'details': []
        }
        
        start_time = time.time()
        
        # Procesar eliminaciones
        for change in changes:
            if change.type == FileChangeType.DELETED:
                if change.previous_info:
                    self._remove_document(change.previous_info.id)
                    results['deleted'] += 1
        
        # Procesar nuevos y modificados en paralelo
        process_changes = [c for c in changes if c.type in [FileChangeType.NEW, FileChangeType.MODIFIED]]
        
        if process_changes:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_change = {
                    executor.submit(self.process_document_source, source_id, change.file_info): change
                    for change in process_changes
                }
                
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
                        results['failed'] += 1
                        results['details'].append({
                            'file': change.file_info.path,
                            'action': change.type.value,
                            'status': 'failed',
                            'error': str(e)
                        })
        
        # Actualizar timestamp
        source = self.get_source(source_id)
        if source:
            source.last_sync = datetime.now()
            self._save_data()
        
        results['processing_time'] = time.time() - start_time
        return results
    
    def sync_source(self, source_id: str, **kwargs) -> Dict[str, Any]:
        """Sincronizar fuente según su tipo"""
        source = self.get_source(source_id)
        if not source:
            raise ValueError(f"Fuente no encontrada: {source_id}")
        
        start_time = time.time()
        
        if isinstance(source, DocumentSource):
            # Documentos: detectar cambios y procesar
            changes = self.detect_document_changes(source_id)
            
            if not changes:
                return {
                    'processed': 0,
                    'message': 'No hay cambios detectados',
                    'processing_time': time.time() - start_time
                }
            
            return self.process_changes(source_id, changes, kwargs.get('max_workers', 3))
        
        elif isinstance(source, APISource):
            # APIs: procesar todos los endpoints
            results = {'processed': 0, 'failed': 0, 'details': []}
            
            for endpoint in source.endpoints:
                try:
                    endpoint_name = endpoint.get('name', 'unnamed')
                    processed_docs = self.api_connector.process_endpoint(source_id, endpoint_name)
                    
                    if processed_docs:
                        results['processed'] += len(processed_docs)
                        results['details'].append({
                            'endpoint': endpoint_name,
                            'status': 'success',
                            'docs': len(processed_docs)
                        })
                    else:
                        results['failed'] += 1
                        results['details'].append({
                            'endpoint': endpoint_name,
                            'status': 'failed'
                        })
                        
                except Exception as e:
                    results['failed'] += 1
                    results['details'].append({
                        'endpoint': endpoint.get('name', 'unnamed'),
                        'status': 'failed',
                        'error': str(e)
                    })
            
            results['processing_time'] = time.time() - start_time
            source.last_sync = datetime.now()
            self._save_data()
            return results
        
        elif isinstance(source, WebSource):
            # Web: realizar scraping
            if not WEB_INGESTION_AVAILABLE:
                raise ValueError("WebIngestionService no disponible")
            
            results = {'processed': 0, 'failed': 0, 'details': []}
            
            try:
                max_pages = kwargs.get('max_pages', 50)
                scraped_data = self.web_ingestion.scrape_source(source_id, max_pages)
                
                if scraped_data and scraped_data.get('success'):
                    pages = scraped_data.get('pages', [])
                    results['processed'] = len(pages)
                    results['details'] = [{'pages_scraped': len(pages), 'status': 'success'}]
                    
                    # Guardar páginas scrapeadas
                    for page_data in pages:
                        if isinstance(page_data, dict):
                            page = ScrapedPage.from_dict(page_data)
                        else:
                            page = page_data
                        self._scraped_pages[page.id] = page
                else:
                    results['failed'] = 1
                    results['details'] = [{'status': 'failed', 'error': 'No pages scraped'}]
                    
            except Exception as e:
                results['failed'] = 1
                results['details'] = [{'status': 'failed', 'error': str(e)}]
            
            results['processing_time'] = time.time() - start_time
            source.last_sync = datetime.now()
            self._save_data()
            return results
        
        elif isinstance(source, DatabaseSource):
            # BD: ejecutar queries
            results = {'processed': 0, 'failed': 0, 'details': []}
            
            for query_config in source.queries:
                try:
                    query_name = query_config.get('name', 'unnamed')
                    processed_docs = self.database_connector.execute_query(source_id, query_name)
                    
                    if processed_docs:
                        results['processed'] += len(processed_docs)
                        results['details'].append({
                            'query': query_name,
                            'status': 'success',
                            'records': len(processed_docs)
                        })
                    else:
                        results['failed'] += 1
                        results['details'].append({
                            'query': query_name,
                            'status': 'failed'
                        })
                        
                except Exception as e:
                    results['failed'] += 1
                    results['details'].append({
                        'query': query_config.get('name', 'unnamed'),
                        'status': 'failed',
                        'error': str(e)
                    })
            
            results['processing_time'] = time.time() - start_time
            source.last_sync = datetime.now()
            self._save_data()
            return results
        
        else:
            raise ValueError(f"Tipo de fuente no soportado: {type(source)}")
    
    # =========================================================================
    # ESTADÍSTICAS Y MONITOREO
    # =========================================================================
    
    def get_source_stats(self, source_id: str) -> IngestionStats:
        """Estadísticas de una fuente"""
        source = self.get_source(source_id)
        if not source:
            raise ValueError(f"Fuente no encontrada: {source_id}")
        
        # Obtener documentos de la fuente
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
        """Estadísticas de todas las fuentes"""
        return [
            self.get_source_stats(source_id)
            for source_id in self._sources.keys()
        ]
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Estadísticas globales del servicio"""
        total_sources = len(self._sources)
        source_counts = {}
        
        for source_type in DataSourceType:
            source_counts[source_type.value] = len(self.list_sources_by_type(source_type))
        
        return {
            'total_sources': total_sources,
            'sources_by_type': source_counts,
            'total_documents': len(self._documents),
            'total_scraped_pages': len(self._scraped_pages),
            'processors_available': {
                'documents': self.document_processor is not None,
                'api': self.api_connector is not None,
                'database': self.database_connector is not None,
                'web': WEB_INGESTION_AVAILABLE and self.web_ingestion is not None
            }
        }
    
    # =========================================================================
    # PERSISTENCIA Y CACHE
    # =========================================================================
    
    def _load_data(self):
        """Cargar datos desde archivos"""
        try:
            # Cargar fuentes por tipo
            for source_type, storage_file in self.storage_files.items():
                if storage_file.exists():
                    with open(storage_file, 'r', encoding='utf-8') as f:
                        sources_data = json.load(f)
                        for sid, data in sources_data.items():
                            if source_type == 'documents':
                                self._sources[sid] = DocumentSource.from_dict(data)
                            elif source_type == 'web':
                                self._sources[sid] = WebSource.from_dict(data)
                            elif source_type == 'api':
                                self._sources[sid] = APISource.from_dict(data)
                            elif source_type == 'database':
                                self._sources[sid] = DatabaseSource.from_dict(data)
            
            # Cargar documentos
            if self.documents_file.exists():
                with open(self.documents_file, 'r', encoding='utf-8') as f:
                    docs_data = json.load(f)
                    self._documents = {
                        did: ProcessedDocument.from_dict(data)
                        for did, data in docs_data.items()
                    }
            
            # Cargar páginas scrapeadas
            if self.scraped_pages_file.exists():
                with open(self.scraped_pages_file, 'r', encoding='utf-8') as f:
                    pages_data = json.load(f)
                    self._scraped_pages = {
                        pid: ScrapedPage.from_dict(data)
                        for pid, data in pages_data.items()
                    }
            
            self.logger.info(f"Cargados {len(self._sources)} fuentes, "
                           f"{len(self._documents)} documentos, "
                           f"{len(self._scraped_pages)} páginas")
            
        except Exception as e:
            self.logger.error(f"Error cargando datos: {e}")
            self._sources = {}
            self._documents = {}
            self._scraped_pages = {}
    
    def _save_data(self):
        """Guardar datos en archivos"""
        try:
            # Separar y guardar fuentes por tipo
            sources_by_type = {
                'documents': {},
                'web': {},
                'api': {},
                'database': {}
            }
            
            for sid, source in self._sources.items():
                if isinstance(source, DocumentSource):
                    sources_by_type['documents'][sid] = source.to_dict()
                elif isinstance(source, WebSource):
                    sources_by_type['web'][sid] = source.to_dict()
                elif isinstance(source, APISource):
                    sources_by_type['api'][sid] = source.to_dict()
                elif isinstance(source, DatabaseSource):
                    sources_by_type['database'][sid] = source.to_dict()
            
            # Guardar cada tipo en su archivo
            for source_type, sources_data in sources_by_type.items():
                if sources_data:  # Solo guardar si hay datos
                    with open(self.storage_files[source_type], 'w', encoding='utf-8') as f:
                        json.dump(sources_data, f, indent=2, ensure_ascii=False)
            
            # Guardar documentos
            if self._documents:
                docs_data = {
                    did: doc.to_dict()
                    for did, doc in self._documents.items()
                }
                with open(self.documents_file, 'w', encoding='utf-8') as f:
                    json.dump(docs_data, f, indent=2, ensure_ascii=False)
            
            # Guardar páginas
            if self._scraped_pages:
                pages_data = {
                    pid: page.to_dict()
                    for pid, page in self._scraped_pages.items()
                }
                with open(self.scraped_pages_file, 'w', encoding='utf-8') as f:
                    json.dump(pages_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Error guardando datos: {e}")
    
    def _remove_document(self, document_id: str):
        """Eliminar documento del sistema"""
        if document_id in self._documents:
            doc = self._documents[document_id]
            
            # Eliminar del vector store
            try:
                self.vector_store.remove_document(document_id)
            except Exception as e:
                self.logger.warning(f"Error eliminando del vector store: {e}")
            
            del self._documents[document_id]
            self.logger.info(f"Documento eliminado: {Path(doc.file_path).name}")


# Instancia singleton
data_sources_service = DataSourcesService()

# Exportaciones
__all__ = ['DataSourcesService', 'data_sources_service']