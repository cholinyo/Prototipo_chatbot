"""
Servicio de Web Scraping para ingesta de sitios web
TFM Vicente Caruncho - Sistemas Inteligentes
"""

import requests
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from dataclasses import dataclass

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

from app.core.logger import get_logger
from app.models.data_sources import WebSource, ProcessingStatus


@dataclass
class ScrapedPage:
    """Página web procesada"""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    scraped_at: datetime
    content_hash: str
    links_found: List[str]
    status_code: int
    
    @classmethod
    def from_response(cls, url: str, response: requests.Response, soup: BeautifulSoup, 
                     content: str, title: str, links: List[str]) -> 'ScrapedPage':
        """Crear ScrapedPage desde respuesta HTTP"""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        metadata = {
            'content_type': response.headers.get('content-type', ''),
            'last_modified': response.headers.get('last-modified', ''),
            'server': response.headers.get('server', ''),
            'content_length': len(content),
            'links_count': len(links),
            'domain': urlparse(url).netloc
        }
        
        return cls(
            url=url,
            title=title,
            content=content,
            metadata=metadata,
            scraped_at=datetime.now(),
            content_hash=content_hash,
            links_found=links,
            status_code=response.status_code
        )


class WebScraperService:
    """Servicio principal de web scraping"""
    
    def __init__(self):
        self.logger = get_logger("web_scraper")
        self.session = requests.Session()
        self.robots_cache: Dict[str, RobotFileParser] = {}
        
        # Configurar session
        self.session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        self.logger.info("WebScraperService inicializado")
        
    def scrape_source(self, source: WebSource) -> List[ScrapedPage]:
        """Procesar una fuente web completa"""
        if not BS4_AVAILABLE:
            raise ImportError("BeautifulSoup4 no está disponible. Instalar con: pip install beautifulsoup4")
        
        self.logger.info(f"Iniciando scraping de fuente: {source.name}")
        
        # Configurar user agent
        self.session.headers['User-Agent'] = source.user_agent
        
        # Añadir headers personalizados
        if source.custom_headers:
            self.session.headers.update(source.custom_headers)
        
        all_pages = []
        visited_urls: Set[str] = set()
        urls_to_visit = list(source.base_urls)
        current_depth = 0
        
        while urls_to_visit and current_depth <= source.max_depth:
            current_level_urls = urls_to_visit.copy()
            urls_to_visit.clear()
            
            for url in current_level_urls:
                if url in visited_urls:
                    continue
                
                try:
                    # Verificar robots.txt
                    if source.respect_robots_txt and not self._is_allowed_by_robots(url, source.user_agent):
                        self.logger.info(f"URL bloqueada por robots.txt: {url}")
                        continue
                    
                    # Verificar si la URL está permitida
                    if not source.is_url_allowed(url):
                        self.logger.debug(f"URL no permitida: {url}")
                        continue
                    
                    # Scraping de la página
                    page = self._scrape_page(url, source)
                    if page:
                        all_pages.append(page)
                        visited_urls.add(url)
                        
                        # Añadir enlaces encontrados para siguiente nivel
                        if source.follow_links and current_depth < source.max_depth:
                            for link in page.links_found:
                                if link not in visited_urls and source.is_url_allowed(link):
                                    urls_to_visit.append(link)
                    
                    # Rate limiting
                    time.sleep(source.delay_seconds)
                    
                except Exception as e:
                    self.logger.error(f"Error procesando {url}: {e}")
                    continue
            
            current_depth += 1
        
        self.logger.info(f"Scraping completado: {len(all_pages)} páginas procesadas")
        return all_pages
    
    def _scrape_page(self, url: str, source: WebSource) -> Optional[ScrapedPage]:
        """Scraping de una página individual"""
        try:
            self.logger.debug(f"Procesando URL: {url}")
            
            # Usar Selenium si se requiere JavaScript
            if source.use_javascript and SELENIUM_AVAILABLE:
                return self._scrape_with_selenium(url, source)
            else:
                return self._scrape_with_requests(url, source)
                
        except Exception as e:
            self.logger.error(f"Error en scraping de {url}: {e}")
            return None
    
    def _scrape_with_requests(self, url: str, source: WebSource) -> Optional[ScrapedPage]:
        """Scraping usando requests + BeautifulSoup"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Parsear HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extraer contenido
            title = self._extract_title(soup, source.title_selectors)
            content = self._extract_content(soup, source)
            links = self._extract_links(soup, url, source)
            
            # Validar contenido mínimo
            if len(content) < source.min_content_length:
                self.logger.debug(f"Contenido insuficiente en {url}: {len(content)} caracteres")
                return None
            
            return ScrapedPage.from_response(url, response, soup, content, title, links)
            
        except Exception as e:
            self.logger.error(f"Error en requests scraping de {url}: {e}")
            return None
    
    def _scrape_with_selenium(self, url: str, source: WebSource) -> Optional[ScrapedPage]:
        """Scraping usando Selenium (para JavaScript)"""
        if not SELENIUM_AVAILABLE:
            self.logger.warning("Selenium no disponible, usando requests")
            return self._scrape_with_requests(url, source)
        
        driver = None
        try:
            # Configurar Chrome headless
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument(f'--user-agent={source.user_agent}')
            
            driver = webdriver.Chrome(options=options)
            driver.get(url)
            
            # Esperar a que el contenido se cargue
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Obtener HTML procesado
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extraer contenido
            title = self._extract_title(soup, source.title_selectors)
            content = self._extract_content(soup, source)
            links = self._extract_links(soup, url, source)
            
            if len(content) < source.min_content_length:
                return None
            
            # Crear respuesta mock para compatibilidad
            class MockResponse:
                def __init__(self):
                    self.status_code = 200
                    self.headers = {'content-type': 'text/html'}
            
            return ScrapedPage.from_response(url, MockResponse(), soup, content, title, links)
            
        except Exception as e:
            self.logger.error(f"Error en Selenium scraping de {url}: {e}")
            return None
        finally:
            if driver:
                driver.quit()
    
    def _extract_title(self, soup: BeautifulSoup, selectors: List[str]) -> str:
        """Extraer título de la página"""
        for selector in selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    return element.get_text(strip=True)
            except Exception:
                continue
        
        # Fallback al title tag
        title_tag = soup.find('title')
        return title_tag.get_text(strip=True) if title_tag else "Sin título"
    
    def _extract_content(self, soup: BeautifulSoup, source: WebSource) -> str:
        """Extraer contenido principal de la página"""
        # Eliminar elementos no deseados
        for selector in source.exclude_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Extraer contenido de selectores específicos
        content_parts = []
        for selector in source.content_selectors:
            try:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(separator=' ', strip=True)
                    if text:
                        content_parts.append(text)
            except Exception as e:
                self.logger.debug(f"Error con selector {selector}: {e}")
                continue
        
        # Si no se encuentra contenido específico, usar body
        if not content_parts:
            body = soup.find('body')
            if body:
                content_parts.append(body.get_text(separator=' ', strip=True))
        
        # Limpiar y unir contenido
        content = ' '.join(content_parts)
        return self._clean_text(content)
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str, source: WebSource) -> List[str]:
        """Extraer enlaces de la página"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            # Normalizar URL
            parsed = urlparse(full_url)
            normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            
            if normalized_url not in links and source.is_url_allowed(normalized_url):
                links.append(normalized_url)
        
        return links
    
    def _clean_text(self, text: str) -> str:
        """Limpiar texto extraído"""
        import re
        
        # Normalizar espacios en blanco
        text = re.sub(r'\s+', ' ', text)
        
        # Eliminar caracteres especiales problemáticos
        text = re.sub(r'[^\w\s\.,;:!?¿¡()-]', '', text)
        
        return text.strip()
    
    def _is_allowed_by_robots(self, url: str, user_agent: str) -> bool:
        """Verificar si la URL está permitida por robots.txt"""
        try:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            
            if robots_url not in self.robots_cache:
                rp = RobotFileParser()
                rp.set_url(robots_url)
                try:
                    rp.read()
                    self.robots_cache[robots_url] = rp
                except Exception:
                    # Si no se puede leer robots.txt, permitir acceso
                    return True
            
            return self.robots_cache[robots_url].can_fetch(user_agent, url)
            
        except Exception:
            # En caso de error, permitir acceso
            return True


# Instancia singleton
web_scraper_service = WebScraperService()

__all__ = ['WebScraperService', 'ScrapedPage', 'web_scraper_service']