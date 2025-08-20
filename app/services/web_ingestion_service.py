"""
Servicio de ingesta de sitios web con scraping inteligente
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
    WebSource, DataSourceType, ProcessingStatus, IngestionStats,
    create_web_source
)
from app.services.web_scraper_service import WebScraperService
from app.services.vector_store_service import VectorStoreService


# Modelo para páginas scrapeadas
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ScrapedPage:
    """Página web scrapeada y procesada"""
    id: str
    source_id: str
    url: str
    title: str
    content: str
    links_found: List[str] = field(default_factory=list)
    scraped_at: datetime = field(default_factory=datetime.now)
    content_length: int = 0
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.content_length:
            self.content_length = len(self.content)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'source_id': self.source_id,
            'url': self.url,
            'title': self.title,
            'content': self.content,
            'links_found': self.links_found,
            'scraped_at': self.scraped_at.isoformat(),
            'content_length': self.content_length,
            'processing_status': self.processing_status.value,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScrapedPage':
        return cls(
            id=data['id'],
            source_id=data['source_id'],
            url=data['url'],
            title=data['title'],
            content=data['content'],
            links_found=data.get('links_found', []),
            scraped_at=datetime.fromisoformat(data['scraped_at']),
            content_length=data.get('content_length', 0),
            processing_status=ProcessingStatus(data.get('processing_status', 'pending')),
            metadata=data.get('metadata', {})
        )


class WebIngestionService:
    """Servicio para ingesta y scraping de sitios web"""
    
    def __init__(self):
        self.logger = get_logger("web_ingestion")
        self.web_scraper = WebScraperService()
        self.vector_store = VectorStoreService()
        
        # Storage para persistir estado (siguiendo patrón de document_ingestion_service)
        self.storage_file = Path("data/ingestion/web_sources.json")
        self.pages_file = Path("data/ingestion/scraped_pages.json")
        
        # Asegurar que existen los directorios
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Cache en memoria
        self._sources: Dict[str, WebSource] = {}
        self._pages: Dict[str, ScrapedPage] = {}
        
        # Cargar datos persistentes
        self._load_data()
    
    def _load_data(self):
        """Cargar datos desde archivos JSON"""
        try:
            # Cargar fuentes web
            if self.storage_file.exists():
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    sources_data = json.load(f)
                    self._sources = {
                        sid: WebSource(**data) 
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
            self._sources = {}
            self._pages = {}
    
    def _save_data(self):
        """Guardar datos en archivos JSON"""
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
    
    def create_source(self, name: str, base_urls: List[str], **kwargs) -> WebSource:
        """Crear nueva fuente web"""
        try:
            # Validar URLs básicamente
            valid_urls = []
            for url in base_urls:
                if url.startswith(('http://', 'https://')):
                    valid_urls.append(url)
                else:
                    self.logger.warning(f"URL inválida ignorada: {url}")
            
            if not valid_urls:
                raise ValueError("No se encontraron URLs válidas")
            
            # Crear fuente web usando factory function
            source = create_web_source(
                name=name,
                base_urls=valid_urls,
                **kwargs
            )
            
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
        """Listar todas las fuentes web"""
        return list(self._sources.values())
    
    def update_source(self, source_id: str, updates: Dict[str, Any]) -> bool:
        """Actualizar configuración de fuente web"""
        if source_id not in self._sources:
            return False
        
        source = self._sources[source_id]
        
        # Actualizar campos permitidos
        updatable_fields = [
            'name', 'base_urls', 'max_depth', 'delay_seconds', 
            'follow_links', 'respect_robots_txt', 'content_selectors',
            'exclude_selectors', 'include_patterns', 'exclude_patterns'
        ]
        
        for field, value in updates.items():
            if field in updatable_fields:
                setattr(source, field, value)
        
        self._save_data()
        self.logger.info(f"Fuente web actualizada: {source.name}")
        return True
    
    def delete_source(self, source_id: str) -> bool:
        """Eliminar fuente web y sus páginas"""
        if source_id not in self._sources:
            return False
        
        # Eliminar páginas asociadas
        pages_to_remove = [
            page_id for page_id, page in self._pages.items()
            if page.source_id == source_id
        ]
        
        for page_id in pages_to_remove:
            del self._pages[page_id]
        
        # Eliminar fuente
        source_name = self._sources[source_id].name
        del self._sources[source_id]
        
        self._save_data()
        self.logger.info(f"Fuente web eliminada: {source_name} ({len(pages_to_remove)} páginas)")
        return True
    
    def scrape_source(self, source_id: str, max_workers: int = 3) -> Dict[str, Any]:
        """Realizar scraping completo de una fuente web"""
        source = self.get_source(source_id)
        if not source:
            raise ValueError(f"Fuente no encontrada: {source_id}")
        
        start_time = time.time()
        self.logger.info(f"Iniciando scraping de: {source.name}")
        
        try:
            # Usar el WebScraperService para hacer el scraping
            scraped_pages = self.web_scraper.scrape_source(source)
            
            # Procesar y guardar páginas
            new_pages = 0
            updated_pages = 0
            
            for scraped_page in scraped_pages:
                # Crear ScrapedPage object
                page = ScrapedPage(
                    id=str(uuid.uuid4()),
                    source_id=source_id,
                    url=scraped_page.url,
                    title=scraped_page.title,
                    content=scraped_page.content,
                    links_found=scraped_page.links_found,
                    scraped_at=datetime.now(),
                    processing_status=ProcessingStatus.COMPLETED
                )
                
                # Verificar si ya existe (por URL)
                existing_page = None
                for existing_id, existing in self._pages.items():
                    if existing.url == page.url and existing.source_id == source_id:
                        existing_page = existing_id
                        break
                
                if existing_page:
                    self._pages[existing_page] = page
                    updated_pages += 1
                else:
                    self._pages[page.id] = page
                    new_pages += 1
                
                # Indexar en vector store
                try:
                    self.vector_store.add_document(
                        doc_id=page.id,
                        content=page.content,
                        metadata={
                            'source_id': source_id,
                            'source_type': 'web',
                            'url': page.url,
                            'title': page.title,
                            'scraped_at': page.scraped_at.isoformat()
                        }
                    )
                except Exception as e:
                    self.logger.warning(f"Error indexando página {page.url}: {e}")
            
            # Actualizar timestamp de la fuente
            source.last_sync = datetime.now()
            
            # Guardar datos
            self._save_data()
            
            processing_time = time.time() - start_time
            
            results = {
                'source_id': source_id,
                'source_name': source.name,
                'new_pages': new_pages,
                'updated_pages': updated_pages,
                'total_pages': len(scraped_pages),
                'processing_time': processing_time,
                'status': 'completed'
            }
            
            self.logger.info(f"Scraping completado: {source.name} - "
                           f"{new_pages} nuevas, {updated_pages} actualizadas en {processing_time:.2f}s")
            
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
    
    def get_source_pages(self, source_id: str) -> List[ScrapedPage]:
        """Obtener páginas scrapeadas de una fuente"""
        return [
            page for page in self._pages.values()
            if page.source_id == source_id
        ]
    
    def get_source_stats(self, source_id: str) -> IngestionStats:
        """Obtener estadísticas de una fuente web"""
        source = self.get_source(source_id)
        if not source:
            raise ValueError(f"Fuente no encontrada: {source_id}")
        
        pages = self.get_source_pages(source_id)
        
        total_content_length = sum(page.content_length for page in pages)
        completed_pages = [p for p in pages if p.processing_status == 
        ProcessingStatus.COMPLETED]
        failed_pages = [p for p in pages if p.processing_status == ProcessingStatus.FAILED]

        
        return IngestionStats(
            source_id=source_id,
            total_files=len(pages),                     
            processed_files=len(completed_pages),      
            failed_files=len(failed_pages),           
            total_chunks=len(pages),                   
            total_size_bytes=total_content_length,     
            last_scan=source.last_sync or source.created_at,  
            processing_time_seconds=0.0                
        )
    
    def test_url(self, url: str) -> Dict[str, Any]:
        """Probar accesibilidad de una URL"""
        try:
            # Crear fuente temporal para testing
            test_source = create_web_source(
                name="Test",
                base_urls=[url],
                max_depth=1
            )
            
            # Hacer scraping de prueba
            pages = self.web_scraper.scrape_source(test_source)
            
            if pages:
                first_page = pages[0]
                return {
                    'accessible': True,
                    'title': first_page.title,
                    'content_length': len(first_page.content),
                    'links_found': len(first_page.links_found),
                    'pages_discovered': len(pages)
                }
            else:
                return {
                    'accessible': False,
                    'error': 'No se pudo acceder al contenido'
                }
                
        except Exception as e:
            return {
                'accessible': False,
                'error': str(e)
            }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas globales del servicio"""
        total_sources = len(self._sources)
        total_pages = len(self._pages)
        
        # Páginas por estado
        completed = len([p for p in self._pages.values() if p.processing_status == ProcessingStatus.COMPLETED])
        failed = len([p for p in self._pages.values() if p.processing_status == ProcessingStatus.FAILED])
        
        # Contenido total
        total_content_size = sum(page.content_length for page in self._pages.values())
        
        # Fuentes activas (con scraping reciente)
        now = datetime.now()
        active_sources = len([
            s for s in self._sources.values() 
            if s.last_sync and (now - s.last_sync).days < 7
        ])
        
        return {
            'total_sources': total_sources,
            'active_sources': active_sources,
            'total_pages': total_pages,
            'pages_completed': completed,
            'pages_failed': failed,
            'success_rate': (completed / total_pages * 100) if total_pages > 0 else 0,
            'total_content_size_mb': total_content_size / (1024 * 1024),
            'last_updated': max([s.last_sync for s in self._sources.values() if s.last_sync], default=None)
        }


# Instancia global del servicio (siguiendo patrón de document_ingestion_service)
web_ingestion_service = WebIngestionService()