"""
Servicio de Web Scraping Mejorado para TFM
Vicente Caruncho - Sistemas Inteligentes

Este servicio proporciona funcionalidades avanzadas de scraping:
- Múltiples métodos de extracción (requests, selenium)
- Configuraciones flexibles de crawling
- Estadísticas y monitoreo de rendimiento
- Integración con el sistema consolidado de ScrapedPage
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
from app.models.data_sources import WebSource, ScrapedPage  # <-- AÑADIDO ScrapedPage desde data_sources


class ScrapingMethod(Enum):
    """Métodos de scraping disponibles"""
    REQUESTS = "requests"
    SELENIUM = "selenium"
    PLAYWRIGHT = "playwright"


class CrawlFrequency(Enum):
    """Frecuencias de crawling configurables"""
    MANUAL = "manual"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class ScrapingConfig:
    """
    Configuración avanzada para scraping
    
    Permite personalizar el comportamiento del scraper con opciones
    específicas para diferentes tipos de sitios web.
    """
    method: ScrapingMethod
    max_depth: int = 2
    delay_seconds: float = 1.0
    crawl_frequency: CrawlFrequency = CrawlFrequency.WEEKLY
    content_filters: List[str] = None
    max_pages: int = 100
    
    def __post_init__(self):
        """Inicialización automática de campos opcionales"""
        if self.content_filters is None:
            self.content_filters = []


class EnhancedWebScraperService:
    """
    Servicio mejorado de web scraping con capacidades avanzadas
    
    Características:
    - Múltiples métodos de extracción
    - Configuraciones flexibles por fuente
    - Monitoreo de rendimiento
    - Integración con el sistema unificado de ScrapedPage
    """
    
    def __init__(self):
        """Inicializar servicio con configuración por defecto"""
        self.logger = get_logger("enhanced_web_scraper")
        self.session = requests.Session()
        
        # Configurar headers realistas para evitar detección
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        self.logger.info("EnhancedWebScraperService inicializado")
    
    def get_available_methods(self) -> List[Dict[str, Any]]:
        """
        Obtener métodos de scraping disponibles con sus características
        
        Returns:
            Lista de métodos disponibles con descripciones y capacidades
        """
        methods = [
            {
                'id': 'requests',
                'name': 'Requests + BeautifulSoup',
                'description': 'Método rápido y eficiente para sitios estáticos',
                'available': BS4_AVAILABLE,
                'pros': ['Muy rápido', 'Bajo consumo de recursos', 'Estable'],
                'cons': ['Sin soporte JavaScript', 'No interacciones complejas'],
                'use_cases': [
                    'Portales institucionales',
                    'Sitios estáticos',
                    'Documentación web',
                    'Páginas de información'
                ],
                'recommended_for': ['Ayuntamientos', 'Sitios gubernamentales', 'Portales de servicios']
            },
            {
                'id': 'selenium',
                'name': 'Selenium WebDriver',
                'description': 'Navegador automatizado para sitios dinámicos',
                'available': SELENIUM_AVAILABLE,
                'pros': ['Ejecuta JavaScript', 'Interacciones complejas', 'Contenido dinámico'],
                'cons': ['Más lento', 'Mayor consumo', 'Requiere más recursos'],
                'use_cases': [
                    'Aplicaciones SPA',
                    'Sitios con JavaScript',
                    'Contenido cargado dinámicamente',
                    'Formularios complejos'
                ],
                'recommended_for': ['Portales modernos', 'Aplicaciones web', 'Sistemas interactivos']
            }
        ]
        
        return [method for method in methods if method['available']]
    
    def scrape_source_sync(self, source: WebSource, config: ScrapingConfig) -> List[ScrapedPage]:
        """
        Realizar scraping sincronizado de una fuente web
        
        Args:
            source: Fuente web configurada
            config: Configuración específica de scraping
            
        Returns:
            Lista de páginas scrapeadas usando el sistema unificado
        """
        if not BS4_AVAILABLE:
            self.logger.error("BeautifulSoup4 no disponible. Instalar con: pip install beautifulsoup4")
            return []
        
        self.logger.info(f"Iniciando scraping mejorado: {source.name}")
        self.logger.info(f"Configuración: método={config.method.value}, max_pages={config.max_pages}")
        
        pages = []
        visited_urls = set()
        urls_to_visit = list(source.base_urls)
        
        # Procesar URLs hasta el límite configurado
        for i, url in enumerate(urls_to_visit[:config.max_pages]):
            if url in visited_urls:
                continue
                
            try:
                start_time = time.time()
                
                # Ejecutar scraping con método configurado
                if config.method == ScrapingMethod.SELENIUM and SELENIUM_AVAILABLE:
                    page_data = self._scrape_with_selenium(url, source)
                else:
                    page_data = self._scrape_with_requests(url, source)
                
                if not page_data:
                    continue
                
                processing_time = time.time() - start_time
                
                # Crear ScrapedPage usando el sistema unificado
                page = ScrapedPage.from_response(
                    url=url,
                    title=page_data['title'],
                    content=page_data['content'],
                    links=page_data['links'],
                    source_id=source.id,
                    response=page_data.get('response')  # Puede ser None para Selenium
                )
                
                # Añadir metadatos específicos del scraper mejorado
                page.metadata.update({
                    'enhanced_scraper': True,
                    'scraping_method': config.method.value,
                    'processing_time': processing_time,
                    'crawl_frequency': config.crawl_frequency.value,
                    'page_index': i + 1,
                    'total_pages_limit': config.max_pages,
                    'content_filters_applied': len(config.content_filters) > 0
                })
                
                # Aplicar filtros de contenido si están configurados
                if config.content_filters:
                    page.content = self._apply_content_filters(page.content, config.content_filters)
                
                pages.append(page)
                visited_urls.add(url)
                
                self.logger.info(f"Página procesada [{i+1}/{config.max_pages}]: {url} ({processing_time:.2f}s)")
                
                # Rate limiting configurado
                time.sleep(config.delay_seconds)
                
            except Exception as e:
                self.logger.error(f"Error procesando {url}: {e}")
                continue
        
        self.logger.info(f"Scraping mejorado completado: {len(pages)} páginas de {len(visited_urls)} URLs visitadas")
        return pages
    
    def _scrape_with_requests(self, url: str, source: WebSource) -> Optional[Dict[str, Any]]:
        """
        Scraping usando requests + BeautifulSoup
        
        Args:
            url: URL a procesar
            source: Fuente web con configuraciones
            
        Returns:
            Diccionario con datos extraídos o None si falla
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extraer título con fallbacks
            title = ""
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            elif soup.find('h1'):
                title = soup.find('h1').get_text(strip=True)
            else:
                title = "Sin título"
            
            # Extraer contenido principal
            content = soup.get_text(separator=' ', strip=True)
            
            # Extraer enlaces válidos del mismo dominio
            links = []
            base_domain = urlparse(url).netloc
            for link in soup.find_all('a', href=True):
                full_link = urljoin(url, link['href'])
                if urlparse(full_link).netloc == base_domain:
                    links.append(full_link)
            
            return {
                'title': title,
                'content': content,
                'links': links,
                'response': response,
                'method': 'requests'
            }
            
        except Exception as e:
            self.logger.error(f"Error en scraping requests de {url}: {e}")
            return None
    
    def _scrape_with_selenium(self, url: str, source: WebSource) -> Optional[Dict[str, Any]]:
        """
        Scraping usando Selenium para contenido dinámico
        
        Args:
            url: URL a procesar
            source: Fuente web con configuraciones
            
        Returns:
            Diccionario con datos extraídos o None si falla
        """
        if not SELENIUM_AVAILABLE:
            self.logger.warning("Selenium no disponible, usando requests")
            return self._scrape_with_requests(url, source)
        
        driver = None
        try:
            # Configurar Chrome headless optimizado
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            
            driver = webdriver.Chrome(options=options)
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            driver.get(url)
            
            # Esperar carga completa
            driver.implicitly_wait(10)
            time.sleep(3)  # Tiempo adicional para contenido dinámico
            
            # Extraer datos del HTML renderizado
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            title = driver.title or "Sin título"
            content = soup.get_text(separator=' ', strip=True)
            
            # Extraer enlaces
            links = []
            base_domain = urlparse(url).netloc
            for link in soup.find_all('a', href=True):
                full_link = urljoin(url, link['href'])
                if urlparse(full_link).netloc == base_domain:
                    links.append(full_link)
            
            return {
                'title': title,
                'content': content,
                'links': links,
                'response': None,  # Selenium no proporciona response object
                'method': 'selenium'
            }
            
        except Exception as e:
            self.logger.error(f"Error en scraping Selenium de {url}: {e}")
            return None
        finally:
            if driver:
                driver.quit()
    
    def _apply_content_filters(self, content: str, filters: List[str]) -> str:
        """
        Aplicar filtros de contenido configurados
        
        Args:
            content: Contenido original
            filters: Lista de filtros a aplicar
            
        Returns:
            Contenido filtrado
        """
        filtered_content = content
        
        for filter_term in filters:
            if filter_term.startswith('remove:'):
                # Remover texto específico
                term_to_remove = filter_term[7:]  # Quitar "remove:"
                filtered_content = filtered_content.replace(term_to_remove, '')
            elif filter_term.startswith('extract:'):
                # Extraer solo secciones que contengan el término
                term_to_extract = filter_term[8:]  # Quitar "extract:"
                sentences = filtered_content.split('.')
                relevant_sentences = [s for s in sentences if term_to_extract.lower() in s.lower()]
                filtered_content = '. '.join(relevant_sentences)
        
        return filtered_content.strip()
    
    def get_scraping_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas y capacidades del servicio
        
        Returns:
            Diccionario con estadísticas completas del servicio
        """
        available_methods = self.get_available_methods()
        
        return {
            'service_name': 'EnhancedWebScraperService',
            'version': '1.0.0',
            'available_methods': len(available_methods),
            'methods_detail': available_methods,
            'capabilities': {
                'requests': BS4_AVAILABLE,
                'selenium': SELENIUM_AVAILABLE,
                'playwright': False  # No implementado aún
            },
            'features': [
                'Múltiples métodos de scraping',
                'Configuraciones flexibles',
                'Filtros de contenido',
                'Rate limiting configurable',
                'Integración con sistema unificado',
                'Monitoreo de rendimiento'
            ],
            'supported_frequencies': [freq.value for freq in CrawlFrequency],
            'default_config': {
                'max_depth': 2,
                'delay_seconds': 1.0,
                'max_pages': 100,
                'crawl_frequency': CrawlFrequency.WEEKLY.value
            }
        }
    
    def test_configuration(self, source: WebSource, config: ScrapingConfig) -> Dict[str, Any]:
        """
        Probar configuración de scraping sin ejecutar crawling completo
        
        Args:
            source: Fuente web a probar
            config: Configuración a validar
            
        Returns:
            Resultados del test de configuración
        """
        test_results = {
            'source_valid': False,
            'method_available': False,
            'first_url_accessible': False,
            'estimated_pages': 0,
            'recommendations': [],
            'warnings': []
        }
        
        # Validar fuente
        if source.base_urls and len(source.base_urls) > 0:
            test_results['source_valid'] = True
        else:
            test_results['warnings'].append("No hay URLs base configuradas")
        
        # Validar método
        available_methods = [m['id'] for m in self.get_available_methods()]
        if config.method.value in available_methods:
            test_results['method_available'] = True
        else:
            test_results['warnings'].append(f"Método {config.method.value} no disponible")
        
        # Probar primera URL
        if source.base_urls:
            try:
                test_url = source.base_urls[0]
                response = self.session.head(test_url, timeout=10)
                if response.status_code == 200:
                    test_results['first_url_accessible'] = True
                    test_results['estimated_pages'] = min(config.max_pages, len(source.base_urls))
                else:
                    test_results['warnings'].append(f"URL base no accesible: {response.status_code}")
            except Exception as e:
                test_results['warnings'].append(f"Error accediendo URL base: {str(e)}")
        
        # Generar recomendaciones
        if config.delay_seconds < 0.5:
            test_results['recommendations'].append("Considerar aumentar delay_seconds para evitar rate limiting")
        
        if config.max_pages > 500:
            test_results['recommendations'].append("max_pages alto puede tardar mucho tiempo")
        
        if config.method == ScrapingMethod.SELENIUM and config.max_pages > 50:
            test_results['recommendations'].append("Selenium es lento, considerar reducir max_pages")
        
        return test_results


# =============================================================================
# INSTANCIA GLOBAL DEL SERVICIO
# =============================================================================

# Instancia global para uso en toda la aplicación
enhanced_scraper_service = EnhancedWebScraperService()

# Exportaciones públicas del módulo
__all__ = [
    'EnhancedWebScraperService',
    'ScrapingConfig', 
    'ScrapingMethod',
    'CrawlFrequency',
    'enhanced_scraper_service'
]