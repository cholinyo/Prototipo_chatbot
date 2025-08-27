"""
Servicio de ingesta de documentos con detección de cambios
TFM Vicente Caruncho - Sistemas Inteligentes

INTEGRACIÓN COMPLETA: Documentos + APIs + Bases de Datos + Web
✅ DocumentProcessor - Archivos PDF, DOCX, TXT
✅ APIConnector - APIs REST con autenticación
✅ DatabaseConnector - Bases de datos SQL
✅ WebIngestionService - Scraping web inteligente
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.core.logger import get_logger
from app.models.data_sources import (
    DocumentSource, ProcessedDocument, FileInfo, FileChange, 
    FileChangeType, ProcessingStatus, IngestionStats,
    APISource, DatabaseSource, WebSource, ScrapedPage,  # ✅ AÑADIDO WebSource, ScrapedPage
    DataSourceType, DataSourceStatus
)

# Importaciones de procesadores
from app.services.ingestion.document_processor import DocumentProcessor
from app.services.ingestion.api_connector import APIConnector
from app.services.ingestion.database_connector import DatabaseConnector
from app.services.vector_store_service import VectorStoreService

# ✅ NUEVA IMPORTACIÓN: Servicio de ingesta web
try:
    from app.services.web_ingestion_service import WebIngestionService
    WEB_INGESTION_AVAILABLE = True
except ImportError:
    WebIngestionService = None
    WEB_INGESTION_AVAILABLE = False


class DocumentIngestionService:
    """
    Servicio integrado para ingesta multimodal COMPLETA
    
    ✅ CAPACIDADES INTEGRADAS:
    - Documentos: PDF, DOCX, TXT, Excel
    - APIs REST: Con autenticación y paginación  
    - Bases de Datos: PostgreSQL, MySQL, SQLite, SQL Server
    - Web: Scraping inteligente con rate limiting y JavaScript
    """
    
    def __init__(self):
        self.logger = get_logger("document_ingestion")
        
        # ✅ PROCESADORES INTEGRADOS - TODOS REALES, NO MOCKS
        self.document_processor = DocumentProcessor()
        self.api_connector = APIConnector()
        self.database_connector = DatabaseConnector()
        self.vector_store = VectorStoreService()
        
        # ✅ NUEVO: Servicio de ingesta web
        if WEB_INGESTION_AVAILABLE:
            self.web_ingestion = WebIngestionService()
            self.logger.info("WebIngestionService integrado correctamente")
        else:
            self.web_ingestion = None
            self.logger.warning("WebIngestionService no disponible")
        
        # Storage para persistir estado (en producción sería base de datos)
        self.storage_file = Path("data/ingestion/document_sources.json")
        self.documents_file = Path("data/ingestion/processed_documents.json") 
        self.api_sources_file = Path("data/ingestion/api_sources.json")
        self.db_sources_file = Path("data/ingestion/database_sources.json")
        self.web_sources_file = Path("data/ingestion/web_sources.json")  # ✅ NUEVO
        self.scraped_pages_file = Path("data/ingestion/scraped_pages.json")  # ✅ NUEVO
        
        # Asegurar que existen los directorios
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Cache en memoria - AMPLIADO PARA INCLUIR WEB
        self._sources: Dict[str, Union[DocumentSource, APISource, DatabaseSource, WebSource]] = {}
        self._documents: Dict[str, ProcessedDocument] = {}
        self._scraped_pages: Dict[str, ScrapedPage] = {}  # ✅ NUEVO
        
        # Cargar datos persistentes
        self._load_data()
        
        integrations = ["Documents", "APIs", "Database"]
        if WEB_INGESTION_AVAILABLE:
            integrations.append("Web")
        
        self.logger.info(f"DocumentIngestionService inicializado con integración completa: "
                        f"{', '.join(integrations)}")
    
    # =========================================================================
    # GESTIÓN DE CARGA Y PERSISTENCIA - AMPLIADA PARA WEB
    # =========================================================================
    
    def _load_data(self):
        """Cargar datos desde archivos JSON - AMPLIADO PARA WEB"""
        try:
            # Cargar fuentes de documentos
            if self.storage_file.exists():
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    sources_data = json.load(f)
                    for sid, data in sources_data.items():
                        self._sources[sid] = DocumentSource.from_dict(data)
            
            # Cargar fuentes de APIs
            if self.api_sources_file.exists():
                with open(self.api_sources_file, 'r', encoding='utf-8') as f:
                    api_data = json.load(f)
                    for sid, data in api_data.items():
                        self._sources[sid] = APISource.from_dict(data)
            
            # Cargar fuentes de bases de datos
            if self.db_sources_file.exists():
                with open(self.db_sources_file, 'r', encoding='utf-8') as f:
                    db_data = json.load(f)
                    for sid, data in db_data.items():
                        self._sources[sid] = DatabaseSource.from_dict(data)
            
            # ✅ NUEVO: Cargar fuentes web
            if self.web_sources_file.exists():
                with open(self.web_sources_file, 'r', encoding='utf-8') as f:
                    web_data = json.load(f)
                    for sid, data in web_data.items():
                        self._sources[sid] = WebSource.from_dict(data)
            
            # Cargar documentos procesados (universal para todos los tipos)
            if self.documents_file.exists():
                with open(self.documents_file, 'r', encoding='utf-8') as f:
                    docs_data = json.load(f)
                    self._documents = {
                        did: ProcessedDocument.from_dict(data)
                        for did, data in docs_data.items()
                    }
            
            # ✅ NUEVO: Cargar páginas web scrapeadas
            if self.scraped_pages_file.exists():
                with open(self.scraped_pages_file, 'r', encoding='utf-8') as f:
                    pages_data = json.load(f)
                    self._scraped_pages = {
                        pid: ScrapedPage.from_dict(data)
                        for pid, data in pages_data.items()
                    }
            
            total_sources = len(self._sources)
            doc_sources = sum(1 for s in self._sources.values() if isinstance(s, DocumentSource))
            api_sources = sum(1 for s in self._sources.values() if isinstance(s, APISource))
            db_sources = sum(1 for s in self._sources.values() if isinstance(s, DatabaseSource))
            web_sources = sum(1 for s in self._sources.values() if isinstance(s, WebSource))
            
            self.logger.info(f"Cargadas {total_sources} fuentes: "
                           f"{doc_sources} documentos, {api_sources} APIs, "
                           f"{db_sources} BBDD, {web_sources} webs. "
                           f"Documentos procesados: {len(self._documents)}, "
                           f"Páginas scrapeadas: {len(self._scraped_pages)}")
            
        except Exception as e:
            self.logger.error(f"Error cargando datos persistentes: {e}")
            self._sources = {}
            self._documents = {}
            self._scraped_pages = {}
    
    def _save_data(self):
        """Guardar datos en archivos JSON - AMPLIADO PARA WEB"""
        try:
            # Separar fuentes por tipo y guardar en archivos específicos
            document_sources = {}
            api_sources = {}
            db_sources = {}
            web_sources = {}
            
            for sid, source in self._sources.items():
                if isinstance(source, DocumentSource):
                    document_sources[sid] = source.to_dict()
                elif isinstance(source, APISource):
                    api_sources[sid] = source.to_dict()
                elif isinstance(source, DatabaseSource):
                    db_sources[sid] = source.to_dict()
                elif isinstance(source, WebSource):
                    web_sources[sid] = source.to_dict()
            
            # Guardar fuentes de documentos
            if document_sources:
                with open(self.storage_file, 'w', encoding='utf-8') as f:
                    json.dump(document_sources, f, indent=2, ensure_ascii=False)
            
            # Guardar fuentes de APIs
            if api_sources:
                with open(self.api_sources_file, 'w', encoding='utf-8') as f:
                    json.dump(api_sources, f, indent=2, ensure_ascii=False)
            
            # Guardar fuentes de bases de datos
            if db_sources:
                with open(self.db_sources_file, 'w', encoding='utf-8') as f:
                    json.dump(db_sources, f, indent=2, ensure_ascii=False)
            
            # ✅ NUEVO: Guardar fuentes web
            if web_sources:
                with open(self.web_sources_file, 'w', encoding='utf-8') as f:
                    json.dump(web_sources, f, indent=2, ensure_ascii=False)
            
            # Guardar documentos procesados (universal)
            docs_data = {
                did: doc.to_dict()
                for did, doc in self._documents.items()
            }
            with open(self.documents_file, 'w', encoding='utf-8') as f:
                json.dump(docs_data, f, indent=2, ensure_ascii=False)
            
            # ✅ NUEVO: Guardar páginas scrapeadas
            if self._scraped_pages:
                pages_data = {
                    pid: page.to_dict()
                    for pid, page in self._scraped_pages.items()
                }
                with open(self.scraped_pages_file, 'w', encoding='utf-8') as f:
                    json.dump(pages_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Error guardando datos: {e}")
    
    # =========================================================================
    # GESTIÓN DE FUENTES - UNIVERSAL PARA TODOS LOS TIPOS
    # =========================================================================
    
    def create_document_source(self, name: str, directories: List[str], **kwargs) -> DocumentSource:
        """Crear fuente de documentos locales"""
        from app.models.data_sources import create_document_source

        source = create_document_source(
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
        
        self.logger.info(f"Fuente de documentos creada: {source.name} ({source.id})")
        return source
    
    def create_api_source(self, name: str, base_url: str, **kwargs) -> APISource:
        """Crear fuente de API REST"""
        from app.models.data_sources import create_api_source
        
        source = create_api_source(
            name=name,
            base_url=base_url,
            **kwargs
        )
        
        # Validar conexión API si es posible
        try:
            test_result = self.api_connector.test_connection(source.id, source)
            if test_result.get('success'):
                source.status = DataSourceStatus.ACTIVE
                self.logger.info(f"API fuente validada: {source.name}")
            else:
                source.status = DataSourceStatus.PENDING
                self.logger.warning(f"API fuente no validada: {test_result.get('error')}")
        except Exception as e:
            source.status = DataSourceStatus.PENDING
            self.logger.warning(f"No se pudo validar API fuente: {e}")
        
        self._sources[source.id] = source
        self._save_data()
        
        self.logger.info(f"Fuente de API creada: {source.name} ({source.id})")
        return source
    
    def create_database_source(self, name: str, db_type: str, 
                             connection_config: Dict[str, Any], **kwargs) -> DatabaseSource:
        """Crear fuente de base de datos"""
        from app.models.data_sources import create_database_source
        
        source = create_database_source(
            name=name,
            db_type=db_type,
            connection_config=connection_config,
            **kwargs
        )
        
        # Validar conexión BD si es posible
        try:
            test_result = self.database_connector.test_connection(source.id, source)
            if test_result.get('success'):
                source.status = DataSourceStatus.ACTIVE
                self.logger.info(f"BD fuente validada: {source.name}")
            else:
                source.status = DataSourceStatus.PENDING
                self.logger.warning(f"BD fuente no validada: {test_result.get('error')}")
        except Exception as e:
            source.status = DataSourceStatus.PENDING
            self.logger.warning(f"No se pudo validar BD fuente: {e}")
        
        self._sources[source.id] = source
        self._save_data()
        
        self.logger.info(f"Fuente de BD creada: {source.name} ({source.id}) - {db_type}")
        return source
    
    def create_web_source(self, name: str, base_urls: List[str], **kwargs) -> WebSource:
        """✅ NUEVO: Crear fuente web para scraping"""
        from app.models.data_sources import create_web_source
        
        if not WEB_INGESTION_AVAILABLE:
            raise ValueError("WebIngestionService no está disponible")
        
        source = create_web_source(
            name=name,
            base_urls=base_urls,
            **kwargs
        )
        
        # Validar URLs si es posible
        try:
            if self.web_ingestion:
                # Test básico de accesibilidad de URLs
                test_results = []
                for url in base_urls[:3]:  # Solo test primeras 3 URLs
                    try:
                        from app.services.web_scraper_service import web_scraper_service
                        test_result = web_scraper_service.test_url(url)
                        test_results.append(test_result.get('accessible', False))
                    except Exception:
                        test_results.append(False)
                
                if any(test_results):
                    source.status = DataSourceStatus.ACTIVE
                    self.logger.info(f"Web fuente validada: {source.name}")
                else:
                    source.status = DataSourceStatus.PENDING
                    self.logger.warning(f"Web fuente no accesible: {source.name}")
            else:
                source.status = DataSourceStatus.PENDING
                
        except Exception as e:
            source.status = DataSourceStatus.PENDING
            self.logger.warning(f"No se pudo validar web fuente: {e}")
        
        self._sources[source.id] = source
        self._save_data()
        
        self.logger.info(f"Fuente web creada: {source.name} ({source.id}) - {len(base_urls)} URLs")
        return source
    
    def get_source(self, source_id: str) -> Optional[Union[DocumentSource, APISource, DatabaseSource, WebSource]]:
        """Obtener fuente por ID - UNIVERSAL"""
        return self._sources.get(source_id)
    
    def list_sources(self) -> List[Union[DocumentSource, APISource, DatabaseSource, WebSource]]:
        """Listar todas las fuentes - UNIVERSAL"""
        return list(self._sources.values())
    
    def list_sources_by_type(self, source_type: DataSourceType) -> List[Union[DocumentSource, APISource, DatabaseSource, WebSource]]:
        """Listar fuentes por tipo específico"""
        return [
            source for source in self._sources.values()
            if source.type == source_type
        ]
    
    def delete_source(self, source_id: str) -> bool:
        """Eliminar fuente y sus documentos - UNIVERSAL"""
        if source_id not in self._sources:
            return False
        
        # Eliminar documentos asociados
        docs_to_remove = [
            doc_id for doc_id, doc in self._documents.items()
            if doc.source_id == source_id
        ]
        
        for doc_id in docs_to_remove:
            del self._documents[doc_id]
        
        # ✅ NUEVO: Eliminar páginas scrapeadas asociadas
        pages_to_remove = [
            page_id for page_id, page in self._scraped_pages.items()
            if page.source_id == source_id
        ]
        
        for page_id in pages_to_remove:
            del self._scraped_pages[page_id]
        
        # Eliminar fuente
        source = self._sources[source_id]
        source_name = source.name
        source_type = source.type.value
        del self._sources[source_id]
        
        self._save_data()
        self.logger.info(f"Fuente {source_type} eliminada: {source_name} "
                        f"({len(docs_to_remove)} documentos, {len(pages_to_remove)} páginas)")
        return True
    
    # =========================================================================
    # PROCESAMIENTO ESPECÍFICO POR TIPO DE FUENTE
    # =========================================================================
    
    def process_document_source(self, source_id: str, file_info: FileInfo) -> ProcessedDocument:
        """Procesar documento individual usando DocumentProcessor"""
        
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
            self.logger.info(f"Procesando documento: {Path(file_info.path).name}")
            
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
                
                self.logger.info(f"Documento procesado exitosamente: {len(chunks)} chunks")
            else:
                processed_doc.status = ProcessingStatus.SKIPPED
                processed_doc.error_message = "No se generaron chunks"
                
        except Exception as e:
            self.logger.error(f"Error procesando documento {file_info.path}: {e}")
            processed_doc.status = ProcessingStatus.ERROR
            processed_doc.error_message = str(e)
        
        # Guardar en memoria y persistir
        self._documents[doc_id] = processed_doc
        self._save_data()
        
        return processed_doc
    
    def process_api_source(self, source_id: str, endpoint_name: str, 
                          parameters: Dict[str, Any] = None) -> List[ProcessedDocument]:
        """Procesar fuente API usando APIConnector"""
        
        source = self.get_source(source_id)
        if not source or not isinstance(source, APISource):
            raise ValueError(f"Fuente API no encontrada: {source_id}")
        
        processed_docs = []
        
        try:
            self.logger.info(f"Procesando API fuente: {source.name} - endpoint: {endpoint_name}")
            
            # Obtener datos de la API usando APIConnector real
            api_response = self.api_connector.fetch_data(source_id, endpoint_name, parameters)
            
            if api_response and api_response.success:
                # Transformar respuesta API a chunks usando APIConnector
                chunks = self.api_connector.transform_to_chunks(
                    api_response, 
                    content_fields=source.content_fields,
                    metadata_fields=source.metadata_fields
                )
                
                if chunks:
                    # Crear registro de documento procesado para la respuesta API
                    doc_id = str(uuid.uuid4())
                    processed_doc = ProcessedDocument(
                        id=doc_id,
                        source_id=source_id,
                        file_path=f"api://{source.base_url}/{endpoint_name}",
                        file_hash=f"api_{endpoint_name}_{int(time.time())}",
                        file_size=len(str(api_response.data)),
                        modified_time=api_response.fetched_at,
                        processed_at=datetime.now(),
                        chunks_count=len(chunks),
                        status=ProcessingStatus.COMPLETED
                    )
                    
                    # Añadir chunks al vector store
                    self.vector_store.add_documents(chunks, source_metadata={
                        'source_id': source_id,
                        'document_id': doc_id,
                        'api_endpoint': endpoint_name,
                        'api_response_time': api_response.response_time_ms
                    })
                    
                    processed_docs.append(processed_doc)
                    self._documents[doc_id] = processed_doc
                    
                    self.logger.info(f"API procesada exitosamente: {len(chunks)} chunks desde {endpoint_name}")
                
                else:
                    self.logger.warning(f"No se generaron chunks desde API endpoint: {endpoint_name}")
            
            else:
                error_msg = api_response.error_message if api_response else "Error desconocido en API"
                self.logger.error(f"Error obteniendo datos de API {endpoint_name}: {error_msg}")
        
        except Exception as e:
            self.logger.error(f"Error procesando API fuente {source_id}: {e}")
        
        self._save_data()
        return processed_docs
    
    def process_database_source(self, source_id: str, query_name: str,
                              parameters: Dict[str, Any] = None) -> List[ProcessedDocument]:
        """Procesar fuente de BD usando DatabaseConnector"""
        
        source = self.get_source(source_id)
        if not source or not isinstance(source, DatabaseSource):
            raise ValueError(f"Fuente BD no encontrada: {source_id}")
        
        processed_docs = []
        
        try:
            self.logger.info(f"Procesando BD fuente: {source.name} - query: {query_name}")
            
            # Ejecutar consulta usando DatabaseConnector real
            db_response = self.database_connector.execute_query(source_id, query_name, parameters)
            
            if db_response and db_response.records:
                # Transformar registros BD a chunks usando DatabaseConnector
                chunks = self.database_connector.transform_to_chunks(
                    db_response,
                    content_fields=source.content_fields,
                    metadata_fields=source.metadata_fields
                )
                
                if chunks:
                    # Crear registro de documento procesado para la respuesta BD
                    doc_id = str(uuid.uuid4())
                    processed_doc = ProcessedDocument(
                        id=doc_id,
                        source_id=source_id,
                        file_path=f"db://{source.db_type}/{query_name}",
                        file_hash=f"db_{query_name}_{int(time.time())}",
                        file_size=len(str(db_response.records)),
                        modified_time=db_response.executed_at,
                        processed_at=datetime.now(),
                        chunks_count=len(chunks),
                        status=ProcessingStatus.COMPLETED
                    )
                    
                    # Añadir chunks al vector store
                    self.vector_store.add_documents(chunks, source_metadata={
                        'source_id': source_id,
                        'document_id': doc_id,
                        'db_query': query_name,
                        'db_execution_time': db_response.execution_time_ms,
                        'total_records': db_response.total_records
                    })
                    
                    processed_docs.append(processed_doc)
                    self._documents[doc_id] = processed_doc
                    
                    self.logger.info(f"BD procesada exitosamente: {len(chunks)} chunks desde {query_name}")
                
                else:
                    self.logger.warning(f"No se generaron chunks desde BD query: {query_name}")
            
            else:
                self.logger.warning(f"No se obtuvieron registros de BD query: {query_name}")
        
        except Exception as e:
            self.logger.error(f"Error procesando BD fuente {source_id}: {e}")
        
        self._save_data()
        return processed_docs
    
    def process_web_source(self, source_id: str, max_pages: int = 50) -> List[ProcessedDocument]:
        """✅ NUEVO: Procesar fuente web usando WebIngestionService"""
        
        source = self.get_source(source_id)
        if not source or not isinstance(source, WebSource):
            raise ValueError(f"Fuente web no encontrada: {source_id}")
        
        if not WEB_INGESTION_AVAILABLE or not self.web_ingestion:
            raise ValueError("WebIngestionService no está disponible")
        
        processed_docs = []
        
        try:
            self.logger.info(f"Procesando web fuente: {source.name} - max {max_pages} páginas")
            
            # Usar WebIngestionService para scraping
            scraping_result = self.web_ingestion.scrape_source(source)
            
            if scraping_result and scraping_result.get('success'):
                scraped_pages = scraping_result.get('pages', [])
                
                # Procesar cada página scrapeada
                for page_data in scraped_pages[:max_pages]:
                    try:
                        # Crear ScrapedPage si no existe
                        if isinstance(page_data, dict):
                            scraped_page = ScrapedPage.from_dict(page_data)
                        else:
                            scraped_page = page_data
                        
                        # Almacenar página scrapeada
                        self._scraped_pages[scraped_page.id] = scraped_page
                        
                        # Generar chunks desde el contenido web usando modelos existentes
                        from app.models.document import create_web_chunks
                        chunks = create_web_chunks(
                            scraped_pages=[scraped_page],
                            chunk_size=source.config.get('chunk_size', 500),
                            chunk_overlap=source.config.get('chunk_overlap', 50)
                        )
                        
                        if chunks:
                            # Crear registro de documento procesado para la página web
                            doc_id = str(uuid.uuid4())
                            processed_doc = ProcessedDocument(
                                id=doc_id,
                                source_id=source_id,
                                file_path=f"web://{scraped_page.url}",
                                file_hash=scraped_page.content_hash,
                                file_size=scraped_page.content_length,
                                modified_time=scraped_page.scraped_at,
                                processed_at=datetime.now(),
                                chunks_count=len(chunks),
                                status=ProcessingStatus.COMPLETED
                            )
                            
                            # Añadir chunks al vector store
                            self.vector_store.add_documents(chunks, source_metadata={
                                'source_id': source_id,
                                'document_id': doc_id,
                                'web_url': scraped_page.url,
                                'web_title': scraped_page.title,
                                'scraped_at': scraped_page.scraped_at.isoformat()
                            })
                            
                            processed_docs.append(processed_doc)
                            self._documents[doc_id] = processed_doc
                            
                            self.logger.debug(f"Página web procesada: {scraped_page.url} - {len(chunks)} chunks")
                        
                        else:
                            self.logger.debug(f"No se generaron chunks para: {scraped_page.url}")
                    
                    except Exception as page_error:
                        self.logger.error(f"Error procesando página individual: {page_error}")
                        continue
                
                self.logger.info(f"Web fuente procesada: {len(processed_docs)} páginas, "
                               f"total chunks: {sum(doc.chunks_count for doc in processed_docs)}")
            
            else:
                error_msg = scraping_result.get('error') if scraping_result else "Error desconocido en scraping"
                self.logger.error(f"Error en scraping de web fuente: {error_msg}")
        
        except Exception as e:
            self.logger.error(f"Error procesando web fuente {source_id}: {e}")
        
        self._save_data()
        return processed_docs
    
    # =========================================================================
    # MÉTODOS DE SINCRONIZACIÓN Y DETECCIÓN DE CAMBIOS
    # =========================================================================
    
    def scan_source(self, source_id: str) -> Union[List[FileInfo], Dict[str, Any]]:
        """Escanear fuente según su tipo - UNIVERSAL CON WEB"""
        source = self.get_source(source_id)
        if not source:
            raise ValueError(f"Fuente no encontrada: {source_id}")
        
        if isinstance(source, DocumentSource):
            self.logger.info(f"Escaneando fuente de documentos: {source.name}")
            files = source.scan_directories()
            self.logger.info(f"Encontrados {len(files)} archivos en {source.name}")
            return files
            
        elif isinstance(source, APISource):
            self.logger.info(f"Validando fuente API: {source.name}")
            # Para APIs, hacer test de conectividad
            test_result = self.api_connector.test_connection(source_id)
            return {
                'type': 'api',
                'available': test_result.get('success', False),
                'endpoints': len(source.endpoints),
                'test_result': test_result
            }
            
        elif isinstance(source, DatabaseSource):
            self.logger.info(f"Validando fuente BD: {source.name}")
            # Para BD, hacer test de conectividad
            test_result = self.database_connector.test_connection(source_id)
            return {
                'type': 'database',
                'available': test_result.get('success', False),
                'queries': len(source.queries),
                'test_result': test_result
            }
        
        elif isinstance(source, WebSource):
            self.logger.info(f"Validando fuente web: {source.name}")
            # ✅ NUEVO: Para web, hacer test de accesibilidad
            if WEB_INGESTION_AVAILABLE and self.web_ingestion:
                test_results = []
                for url in source.base_urls[:3]:  # Test solo primeras 3 URLs
                    try:
                        from app.services.web_scraper_service import web_scraper_service
                        test_result = web_scraper_service.test_url(url)
                        test_results.append(test_result)
                    except Exception as e:
                        test_results.append({'accessible': False, 'error': str(e)})
                
                return {
                    'type': 'web',
                    'available': any(r.get('accessible', False) for r in test_results),
                    'base_urls': len(source.base_urls),
                    'test_results': test_results
                }
            else:
                return {
                    'type': 'web',
                    'available': False,
                    'error': 'WebIngestionService no disponible'
                }
        
        else:
            raise ValueError(f"Tipo de fuente no soportado: {type(source)}")
    
    def detect_changes(self, source_id: str) -> List[FileChange]:
        """Detectar cambios en archivos de una fuente - SOLO PARA DOCUMENTOS"""
        source = self.get_source(source_id)
        if not source or not isinstance(source, DocumentSource):
            raise ValueError(f"Detección de cambios solo soportada para fuentes de documentos")
        
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
    
    def sync_source(self, source_id: str, **kwargs) -> Dict[str, Any]:
        """Sincronizar fuente según su tipo - INCLUYENDO WEB"""
        
        source = self.get_source(source_id)
        if not source:
            raise ValueError(f"Fuente no encontrada: {source_id}")
        
        start_time = time.time()
        
        if isinstance(source, DocumentSource):
            # Sincronización de documentos con detección de cambios
            changes = self.detect_changes(source_id)
            
            if not changes:
                return {
                    'processed': 0,
                    'failed': 0,
                    'skipped': 0,
                    'deleted': 0,
                    'message': 'No hay cambios detectados',
                    'processing_time': time.time() - start_time
                }
            
            return self.process_changes(source_id, changes, kwargs.get('max_workers', 3))
        
        elif isinstance(source, APISource):
            # Sincronización de API - procesar todos los endpoints
            results = {
                'processed': 0,
                'failed': 0,
                'total_endpoints': len(source.endpoints),
                'details': []
            }
            
            for endpoint in source.endpoints:
                try:
                    endpoint_name = endpoint.get('name', 'unnamed')
                    processed_docs = self.process_api_source(source_id, endpoint_name)
                    
                    if processed_docs:
                        results['processed'] += len(processed_docs)
                        results['details'].append({
                            'endpoint': endpoint_name,
                            'status': 'success',
                            'chunks': sum(doc.chunks_count for doc in processed_docs)
                        })
                    else:
                        results['failed'] += 1
                        results['details'].append({
                            'endpoint': endpoint_name,
                            'status': 'failed',
                            'error': 'No se generaron chunks'
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
        
        elif isinstance(source, DatabaseSource):
            # Sincronización de BD - ejecutar todas las consultas
            results = {
                'processed': 0,
                'failed': 0,
                'total_queries': len(source.queries),
                'details': []
            }
            
            for query_config in source.queries:
                try:
                    query_name = query_config.get('name', 'unnamed')
                    processed_docs = self.process_database_source(source_id, query_name)
                    
                    if processed_docs:
                        results['processed'] += len(processed_docs)
                        results['details'].append({
                            'query': query_name,
                            'status': 'success',
                            'chunks': sum(doc.chunks_count for doc in processed_docs)
                        })
                    else:
                        results['failed'] += 1
                        results['details'].append({
                            'query': query_name,
                            'status': 'failed',
                            'error': 'No se generaron chunks'
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
        
        elif isinstance(source, WebSource):
            # ✅ NUEVO: Sincronización de web - scraping completo
            results = {
                'processed': 0,
                'failed': 0,
                'total_urls': len(source.base_urls),
                'details': []
            }
            
            try:
                max_pages = kwargs.get('max_pages', 50)
                processed_docs = self.process_web_source(source_id, max_pages)
                
                if processed_docs:
                    results['processed'] = len(processed_docs)
                    results['details'] = [{
                        'pages_scraped': len(processed_docs),
                        'status': 'success',
                        'chunks': sum(doc.chunks_count for doc in processed_docs)
                    }]
                else:
                    results['failed'] = 1
                    results['details'] = [{
                        'status': 'failed',
                        'error': 'No se scrapearon páginas'
                    }]
                    
            except Exception as e:
                results['failed'] = 1
                results['details'] = [{
                    'status': 'failed',
                    'error': str(e)
                }]
            
            results['processing_time'] = time.time() - start_time
            source.last_sync = datetime.now()
            self._save_data()
            
            return results
        
        else:
            raise ValueError(f"Tipo de fuente no soportado para sincronización: {type(source)}")
    
    # =========================================================================
    # MÉTODOS HEREDADOS - MANTIENEN COMPATIBILIDAD
    # =========================================================================
    
    def process_document(self, source_id: str, file_info: FileInfo, 
                        update_existing: bool = True) -> ProcessedDocument:
        """Mantener compatibilidad - delegar a process_document_source"""
        return self.process_document_source(source_id, file_info)
    
    def process_changes(self, source_id: str, changes: List[FileChange],
                       max_workers: int = 3) -> Dict[str, Any]:
        """Procesar lista de cambios de archivos - SOLO DOCUMENTOS"""
        
        source = self.get_source(source_id)
        if not source or not isinstance(source, DocumentSource):
            raise ValueError(f"Procesamiento de cambios solo soportado para documentos")
        
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
                    executor.submit(self.process_document_source, source_id, change.file_info): change
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
    
    # =========================================================================
    # ESTADÍSTICAS Y MONITOREO
    # =========================================================================
    
    def get_source_stats(self, source_id: str) -> IngestionStats:
        """Obtener estadísticas de una fuente - UNIVERSAL"""
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
    
    def get_scraped_pages(self, source_id: str) -> List[ScrapedPage]:
        """✅ NUEVO: Obtener páginas scrapeadas de una fuente web"""
        return [
            page for page in self._scraped_pages.values()
            if page.source_id == source_id
        ]
    
    def get_processing_logs(self, source_id: Optional[str] = None,
                           limit: int = 50) -> List[Dict[str, Any]]:
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
                'source_type': source.type.value if source else 'unknown',
                'file_name': Path(doc.file_path).name,
                'file_path': doc.file_path,
                'status': doc.status.value,
                'chunks_count': doc.chunks_count,
                'file_size': doc.file_size,
                'error_message': doc.error_message
            })
        
        return logs
    
    def cleanup_deleted_files(self, source_id: str) -> int:
        """Limpiar referencias a archivos que ya no existen - SOLO DOCUMENTOS"""
        source = self.get_source(source_id)
        if not source or not isinstance(source, DocumentSource):
            self.logger.warning(f"Cleanup solo soportado para fuentes de documentos")
            return 0
        
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
    
    # =========================================================================
    # MÉTODOS DE CONVENIENCIA Y DIAGNÓSTICO
    # =========================================================================
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Obtener resumen completo del sistema - INCLUYENDO WEB"""
        
        total_sources = len(self._sources)
        doc_sources = len(self.list_sources_by_type(DataSourceType.DOCUMENTS))
        api_sources = len(self.list_sources_by_type(DataSourceType.API))
        db_sources = len(self.list_sources_by_type(DataSourceType.DATABASE))
        web_sources = len(self.list_sources_by_type(DataSourceType.WEB))
        
        total_docs = len(self._documents)
        completed_docs = sum(1 for doc in self._documents.values() 
                           if doc.status == ProcessingStatus.COMPLETED)
        total_chunks = sum(doc.chunks_count for doc in self._documents.values())
        total_pages = len(self._scraped_pages)
        
        return {
            'sources': {
                'total': total_sources,
                'documents': doc_sources,
                'apis': api_sources,
                'databases': db_sources,
                'web': web_sources  # ✅ NUEVO
            },
            'processing': {
                'total_documents': total_docs,
                'completed_documents': completed_docs,
                'success_rate': (completed_docs / total_docs * 100) if total_docs > 0 else 0,
                'total_chunks': total_chunks,
                'scraped_pages': total_pages  # ✅ NUEVO
            },
            'integrations': {
                'document_processor': self.document_processor is not None,
                'api_connector': self.api_connector is not None,
                'database_connector': self.database_connector is not None,
                'vector_store': self.vector_store is not None,
                'web_ingestion': WEB_INGESTION_AVAILABLE and self.web_ingestion is not None  # ✅ NUEVO
            }
        }


# Instancia singleton para uso global
document_ingestion_service = DocumentIngestionService()

# Exportar para importación
__all__ = ['DocumentIngestionService', 'document_ingestion_service']