"""
Servicio de Web Scraping Mejorado para TFM
Vicente Caruncho - Sistemas Inteligentes
"""

import requests
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
from enum import Enum

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

from app.core.logger import get_logger
from app.models.data_sources import WebSource


class ScrapingMethod(Enum):
    REQUESTS = "requests"
    SELENIUM = "selenium"
    PLAYWRIGHT = "playwright"


class CrawlFrequency(Enum):
    MANUAL = "manual"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class ScrapingConfig:
    method: ScrapingMethod
    max_depth: int = 2
    delay_seconds: float = 1.0
    crawl_frequency: CrawlFrequency = CrawlFrequency.WEEKLY
    content_filters: List[str] = None
    max_pages: int = 100
    
    def __post_init__(self):
        if self.content_filters is None:
            self.content_filters = []


@dataclass
class ScrapedPage:
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    scraped_at: datetime
    content_hash: str
    links_found: List[str]
    status_code: int
    scraping_method: ScrapingMethod
    processing_time: float


class EnhancedWebScraperService:
    def __init__(self):
        self.logger = get_logger("enhanced_web_scraper")
        self.session = requests.Session()
        
        # Configurar headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.logger.info("EnhancedWebScraperService inicializado")
    
    def get_available_methods(self) -> List[Dict[str, Any]]:
        """Obtener métodos de scraping disponibles"""
        methods = [
            {
                'id': 'requests',
                'name': 'Requests + BeautifulSoup',
                'description': 'Método rápido para sitios estáticos',
                'available': BS4_AVAILABLE,
                'pros': ['Muy rápido', 'Bajo consumo'],
                'cons': ['Sin JavaScript'],
                'use_cases': ['Portales institucionales', 'Sitios estáticos']
            },
            {
                'id': 'selenium',
                'name': 'Selenium WebDriver',
                'description': 'Navegador automatizado',
                'available': SELENIUM_AVAILABLE,
                'pros': ['Ejecuta JavaScript', 'Interacciones complejas'],
                'cons': ['Más lento', 'Mayor consumo'],
                'use_cases': ['SPAs', 'Sitios dinámicos']
            }
        ]
        
        return [m for m in methods if m['available']]
    
    def scrape_source_sync(self, source: WebSource, config: ScrapingConfig) -> List[ScrapedPage]:
        """Realizar scraping de una fuente web"""
        if not BS4_AVAILABLE:
            self.logger.error("BeautifulSoup4 no disponible")
            return []
        
        self.logger.info(f"Iniciando scraping: {source.name}")
        
        pages = []
        visited_urls = set()
        urls_to_visit = list(source.base_urls)
        
        for url in urls_to_visit[:config.max_pages]:
            if url in visited_urls:
                continue
                
            try:
                start_time = time.time()
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extraer contenido
                title = soup.title.string if soup.title else ""
                content = soup.get_text(separator=' ', strip=True)
                
                # Extraer enlaces
                links = []
                for link in soup.find_all('a', href=True):
                    full_link = urljoin(url, link['href'])
                    if urlparse(full_link).netloc == urlparse(url).netloc:
                        links.append(full_link)
                
                processing_time = time.time() - start_time
                content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                
                page = ScrapedPage(
                    url=url,
                    title=title,
                    content=content,
                    metadata={'domain': urlparse(url).netloc},
                    scraped_at=datetime.now(),
                    content_hash=content_hash,
                    links_found=links,
                    status_code=response.status_code,
                    scraping_method=config.method,
                    processing_time=processing_time
                )
                
                pages.append(page)
                visited_urls.add(url)
                
                self.logger.info(f"Página procesada: {url}")
                time.sleep(config.delay_seconds)
                
            except Exception as e:
                self.logger.error(f"Error procesando {url}: {e}")
                continue
        
        self.logger.info(f"Scraping completado: {len(pages)} páginas")
        return pages
    
    def get_scraping_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del servicio"""
        return {
            'available_methods': len(self.get_available_methods()),
            'capabilities': {
                'requests': BS4_AVAILABLE,
                'selenium': SELENIUM_AVAILABLE
            }
        }


# Instancia global
enhanced_scraper_service = EnhancedWebScraperService()
