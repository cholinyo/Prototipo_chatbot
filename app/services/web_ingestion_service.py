"""
Servicio de ingesta de sitios web con scraping inteligente
TFM Vicente Caruncho - Sistemas Inteligentes

Este servicio maneja:
- Creación y gestión de fuentes web
- Scraping de sitios web con configuraciones avanzadas
- Indexación de contenido en vector store
- Persistencia de datos y estadísticas
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
    WebSource, DataSourceType, ProcessingStatus, IngestionStats,
    create_web_source, ScrapedPage  # <-- AÑADIDO ScrapedPage aquí
)
from app.services.web_scraper_service import WebScraperService
from app.services.vector_store_service import VectorStoreService


# =============================================================================
# SERVICIO PRINCIPAL DE INGESTA WEB
# =============================================================================

class WebIngestionService:
    """
    Servicio principal para ingesta y scraping de sitios web
    
    Características:
    - Gestión completa de fuentes web
    - Scraping paralelo con configuraciones avanzadas
    - Indexación automática en vector store
    - Persistencia de datos y estadísticas
    - Detección de duplicados y actualizaciones
    """
    
    def __init__(self):
        """Inicializar servicio con componentes necesarios"""
        self.logger = get_logger("web_ingestion")
        self.web_scraper = WebScraperService()
        self.vector_store = VectorStoreService()
        
        # Configuración de archivos de persistencia
        self.storage_file = Path("data/ingestion/web_sources.json")
        self.pages_file = Path("data/ingestion/scraped_pages.json")
        
        # Crear directorios necesarios
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Cache en memoria para rendimiento
        self._sources: Dict[str, WebSource] = {}
        self._pages: Dict[str, ScrapedPage] = {}
        
        # Cargar datos existentes
        self._load_data()
    
    # =========================================================================
    # GESTIÓN DE PERSISTENCIA
    # =========================================================================
    
    def _load_data(self):
        """Cargar fuentes y páginas desde archivos JSON"""
        try:
            # Cargar fuentes web
            if self.storage_file.exists():
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    sources_data = json.load(f)
                    self._sources = {
                        sid: WebSource.from_dict(data) 
                        for sid, data in sources_data.items()
                    }
            
            # Cargar páginas scrapeadas
            if self.pages_file.exists():
                with open(self.pages_file, 'r', encoding='utf-8') as f:
                    pages_data = json.load(f)
                    self._pages = {
                        pid: ScrapedPage.from_dict(data)
                        for pid, data in pages_data.items()
                    }
            
            self.logger.info(f"Cargados {len(self._sources)} fuentes web y {len(self._pages)} páginas")
            
        except Exception as e:
            self.logger.error(f"Error cargando datos persistentes: {e}")
            # Inicializar con datos vacíos en caso de error
            self._sources = {}
            self._pages = {}
    
    def _save_data(self):
        """Guardar fuentes y páginas en archivos JSON"""
        try:
            # Guardar fuentes web
            sources_data = {
                sid: source.to_dict() 
                for sid, source in self._sources.items()
            }
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(sources_data, f, indent=2, ensure_ascii=False)
            
            # Guardar páginas scrapeadas
            pages_data = {
                pid: page.to_dict()
                for pid, page in self._pages.items()
            }
            with open(self.pages_file, 'w', encoding='utf-8') as f:
                json.dump(pages_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Error guardando datos: {e}")
    
    # =========================================================================
    # GESTIÓN DE FUENTES WEB
    # =========================================================================
    
    def create_source(self, name: str, base_urls: List[str], **kwargs) -> WebSource:
        """
        Crear nueva fuente web con validación de URLs
        
        Args:
            name: Nombre descriptivo de la fuente
            base_urls: Lista de URLs base para scraping
            **kwargs: Configuraciones adicionales (max_depth, delay_seconds, etc.)
        
        Returns:
            WebSource: Fuente web creada y persistida
        """
        try:
            # Validar y limpiar URLs
            valid_urls = []
            for url in base_urls:
                if url.startswith(('http://', 'https://')):
                    # Normalizar URL (remover trailing slash si existe)
                    url = url.rstrip('/')
                    valid_urls.append(url)
                else:
                    self.logger.warning(f"URL inválida ignorada: {url}")
            
            if not valid_urls:
                raise ValueError("No se encontraron URLs válidas")
            
            # Crear fuente usando factory function
            source = create_web_source(
                name=name,
                base_urls=valid_urls,
                **kwargs
            )
            
            # Persistir en memoria y disco
            self._sources[source.id] = source
            self._save_data()
            
            self.logger.info(f"Fuente web creada: {source.name} ({source.id})")
            return source
            
        except Exception as e:
            self.logger.error(f"Error creando fuente web: {e}")
            raise
    
    def get_source(self, source_id: str) -> Optional[WebSource]:
        """Obtener fuente por ID"""
        return self._sources.get(source_id)
    
    def list_sources(self) -> List[WebSource]:
        """Listar todas las fuentes web disponibles"""
        return list(self._sources.values())
    
    def update_source(self, source_id: str, updates: Dict[str, Any]) -> bool:
        """
        Actualizar configuración de fuente web
        
        Args:
            source_id: ID de la fuente a actualizar
            updates: Diccionario con campos a actualizar
        
        Returns:
            bool: True si la actualización fue exitosa
        """
        if source_id not in self._sources:
            return False
        
        source = self._sources[source_id]
        
        # Campos permitidos para actualización
        updatable_fields = [
            'name', 'base_urls', 'max_depth', 'delay_seconds', 
            'follow_links', 'respect_robots_txt', 'content_selectors',
            'exclude_selectors', 'include_patterns', 'exclude_patterns',
            'min_content_length', 'user_agent'
        ]
        
        # Aplicar actualizaciones
        for field, value in updates.items():
            if field in updatable_fields:
                setattr(source, field, value)
        
        # Actualizar configuración en config dict (para serialización)
        source.__post_init__()
        
        self._save_data()
        self.logger.info(f"Fuente web actualizada: {source.name}")
        return True
    
    def delete_source(self, source_id: str) -> bool:
        """
        Eliminar fuente web y todas sus páginas asociadas
        
        Args:
            source_id: ID de la fuente a eliminar
        
        Returns:
            bool: True si la eliminación fue exitosa
        """
        if source_id not in self._sources:
            return False
        
        # Identificar páginas asociadas
        pages_to_remove = [
            page_id for page_id, page in self._pages.items()
            if page.source_id == source_id
        ]
        
        # Eliminar páginas del vector store
        for page_id in pages_to_remove:
            try:
                # Intentar eliminar del vector store si tiene el método
                if hasattr(self.vector_store, 'remove_document'):
                    self.vector_store.remove_document(page_id)
            except Exception as e:
                self.logger.warning(f"Error eliminando página {page_id} del vector store: {e}")
            
            # Eliminar de memoria
            del self._pages[page_id]
        
        # Eliminar fuente
        source_name = self._sources[source_id].name
        del self._sources[source_id]
        
        self._save_data()
        self.logger.info(f"Fuente web eliminada: {source_name} ({len(pages_to_remove)} páginas)")
        return True
    
    # =========================================================================
    # SCRAPING Y PROCESAMIENTO
    # =========================================================================
    
    def scrape_source(self, source_id: str, max_workers: int = 3) -> Dict[str, Any]:
        """
        Realizar scraping completo de una fuente web
        
        Args:
            source_id: ID de la fuente a procesar
            max_workers: Número máximo de workers paralelos
        
        Returns:
            Dict: Resultados del scraping con estadísticas
        """
        source = self.get_source(source_id)
        if not source:
            raise ValueError(f"Fuente no encontrada: {source_id}")
        
        start_time = time.time()
        self.logger.info(f"Iniciando scraping de: {source.name}")
        
        try:
            # Ejecutar scraping usando WebScraperService
            scraped_pages = self.web_scraper.scrape_source(source)
            
            # Contadores para estadísticas
            new_pages = 0
            updated_pages = 0
            indexed_pages = 0
            
            # Procesar cada página scrapeada
            for scraped_page in scraped_pages:
                # Crear ScrapedPage desde resultado del scraper
                # Nota: scraped_page viene del web_scraper_service con estructura diferente
                page = ScrapedPage.from_response(
                    url=scraped_page.url,
                    title=scraped_page.title,
                    content=scraped_page.content,
                    links=scraped_page.links_found,
                    source_id=source_id,
                    response=None  # No tenemos response object aquí
                )
                
                # Marcar como completado
                page.update_processing_status(ProcessingStatus.COMPLETED)
                
                # Verificar si la página ya existe (por URL)
                existing_page_id = self._find_existing_page(source_id, page.url)
                
                if existing_page_id:
                    # Actualizar página existente
                    old_page = self._pages[existing_page_id]
                    page.id = existing_page_id  # Mantener ID original
                    self._pages[existing_page_id] = page
                    updated_pages += 1
                else:
                    # Nueva página
                    self._pages[page.id] = page
                    new_pages += 1
                
                # Indexar en vector store usando el nuevo sistema
                if self._index_page_in_vector_store(page, source_id):
                    indexed_pages += 1
            
            # Actualizar timestamp de última sincronización
            source.last_sync = datetime.now()
            
            # Persistir todos los cambios
            self._save_data()
            
            # Calcular estadísticas finales
            processing_time = time.time() - start_time
            
            results = {
                'source_id': source_id,
                'source_name': source.name,
                'new_pages': new_pages,
                'updated_pages': updated_pages,
                'indexed_pages': indexed_pages,
                'total_pages': len(scraped_pages),
                'processing_time': processing_time,
                'status': 'completed'
            }
            
            self.logger.info(
                f"Scraping completado: {source.name} - "
                f"{new_pages} nuevas, {updated_pages} actualizadas, "
                f"{indexed_pages} indexadas en {processing_time:.2f}s"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error en scraping de {source.name}: {e}")
            return {
                'source_id': source_id,
                'source_name': source.name,
                'error': str(e),
                'status': 'failed',
                'processing_time': time.time() - start_time
            }
    
    def _find_existing_page(self, source_id: str, url: str) -> Optional[str]:
        """
        Buscar página existente por URL en la misma fuente
        
        Args:
            source_id: ID de la fuente
            url: URL a buscar
        
        Returns:
            Optional[str]: ID de la página existente o None
        """
        for page_id, page in self._pages.items():
            if page.source_id == source_id and page.url == url:
                return page_id
        return None
    
    def _index_page_in_vector_store(self, page: ScrapedPage, source_id: str) -> bool:
        """
        Indexar página en el vector store usando el nuevo sistema de chunks
        
        Args:
            page: Página a indexar
            source_id: ID de la fuente
        
        Returns:
            bool: True si la indexación fue exitosa
        """
        try:
            # Usar el nuevo sistema de chunks desde document.py
            from app.models.document import create_web_chunks
            
            # Crear chunks desde la página scrapeada
            chunks = create_web_chunks(
                scraped_page=page,
                chunk_size=500,
                chunk_overlap=50
            )
            
            if not chunks:
                self.logger.warning(f"No se generaron chunks para: {page.url}")
                return False
            
            # Metadatos adicionales para el vector store
            source_metadata = {
                'source_id': source_id,
                'source_type': 'web',
                'source_name': self._sources[source_id].name,
                'total_chunks': len(chunks)
            }
            
            # Indexar chunks en el vector store
            success = self.vector_store.add_documents(chunks, source_metadata)
            
            if success:
                # Actualizar conteo de chunks en la página
                page.update_processing_status(
                    ProcessingStatus.COMPLETED, 
                    chunks_count=len(chunks)
                )
                self.logger.debug(f"Indexados {len(chunks)} chunks para: {page.url}")
            else:
                self.logger.warning(f"Vector store reportó fallo indexando: {page.url}")
            
            return success
            
        except Exception as e:
            self.logger.warning(f"Error indexando página {page.url}: {e}")
            return False
    
    # =========================================================================
    # CONSULTAS Y ESTADÍSTICAS
    # =========================================================================
    
    def get_source_pages(self, source_id: str) -> List[ScrapedPage]:
        """Obtener todas las páginas scrapeadas de una fuente"""
        return [
            page for page in self._pages.values()
            if page.source_id == source_id
        ]
    
    def get_source_stats(self, source_id: str) -> IngestionStats:
        """
        Obtener estadísticas detalladas de una fuente web
        
        Args:
            source_id: ID de la fuente
        
        Returns:
            IngestionStats: Estadísticas completas de la fuente
        """
        source = self.get_source(source_id)
        if not source:
            raise ValueError(f"Fuente no encontrada: {source_id}")
        
        # Obtener páginas de la fuente
        pages = self.get_source_pages(source_id)
        
        # Calcular estadísticas
        total_content_length = sum(page.content_length for page in pages)
        completed_pages = [
            p for p in pages 
            if p.processing_status == ProcessingStatus.COMPLETED
        ]
        failed_pages = [
            p for p in pages 
            if p.processing_status == ProcessingStatus.ERROR
        ]
        
        # Sumar total de chunks de todas las páginas
        total_chunks = sum(page.chunks_count for page in pages)
        
        return IngestionStats(
            source_id=source_id,
            total_files=len(pages),
            processed_files=len(completed_pages),
            failed_files=len(failed_pages),
            total_chunks=total_chunks,
            total_size_bytes=total_content_length,
            last_scan=source.last_sync or source.created_at,
            processing_time_seconds=0.0  # Se calcula durante el scraping
        )
    
    def get_all_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas globales del servicio
        
        Returns:
            Dict: Estadísticas generales del sistema de web scraping
        """
        total_sources = len(self._sources)
        total_pages = len(self._pages)
        
        # Estadísticas por estado de procesamiento
        completed = len([
            p for p in self._pages.values() 
            if p.processing_status == ProcessingStatus.COMPLETED
        ])
        failed = len([
            p for p in self._pages.values() 
            if p.processing_status == ProcessingStatus.ERROR
        ])
        
        # Tamaño total de contenido
        total_content_size = sum(
            page.content_length for page in self._pages.values()
        )
        
        # Total de chunks generados
        total_chunks = sum(
            page.chunks_count for page in self._pages.values()
        )
        
        # Fuentes activas (con scraping en los últimos 7 días)
        now = datetime.now()
        active_sources = len([
            s for s in self._sources.values() 
            if s.last_sync and (now - s.last_sync).days < 7
        ])
        
        # Última actualización global
        last_updates = [
            s.last_sync for s in self._sources.values() 
            if s.last_sync
        ]
        last_updated = max(last_updates) if last_updates else None
        
        return {
            'total_sources': total_sources,
            'active_sources': active_sources,
            'total_pages': total_pages,
            'total_chunks': total_chunks,
            'pages_completed': completed,
            'pages_failed': failed,
            'success_rate': (completed / total_pages * 100) if total_pages > 0 else 0,
            'total_content_size_mb': total_content_size / (1024 * 1024),
            'last_updated': last_updated
        }
    
    # =========================================================================
    # UTILIDADES Y TESTING
    # =========================================================================
    
    def test_url(self, url: str) -> Dict[str, Any]:
        """
        Probar accesibilidad de una URL específica
        
        Args:
            url: URL a probar
        
        Returns:
            Dict: Resultados del test de conectividad
        """
        try:
            # Crear fuente temporal para testing
            test_source = create_web_source(
                name="Test URL",
                base_urls=[url],
                max_depth=1  # Solo nivel superficial para testing
            )
            
            # Intentar scraping de prueba
            pages = self.web_scraper.scrape_source(test_source)
            
            if pages and len(pages) > 0:
                first_page = pages[0]
                return {
                    'accessible': True,
                    'title': first_page.title,
                    'content_length': len(first_page.content),
                    'links_found': len(first_page.links_found),
                    'pages_discovered': len(pages),
                    'status': 'success'
                }
            else:
                return {
                    'accessible': False,
                    'error': 'No se pudo extraer contenido de la URL',
                    'status': 'no_content'
                }
                
        except Exception as e:
            return {
                'accessible': False,
                'error': str(e),
                'status': 'error'
            }


# =============================================================================
# INSTANCIA GLOBAL DEL SERVICIO
# =============================================================================

# Crear instancia global siguiendo el patrón del document_ingestion_service
web_ingestion_service = WebIngestionService()