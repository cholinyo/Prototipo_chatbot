"""
Servicio de Web Scraping para ingesta de sitios web
TFM Vicente Caruncho - Sistemas Inteligentes
"""

import requests
import time
import hashlib
import json
import re
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
        self.robots_cache: Dict[str, Optional[RobotFileParser]] = {}
        self._content_hashes: Dict[str, str] = {}
        
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
            
            # Verificar si el contenido cambió
            if not self.detect_content_changes(url, content):
                self.logger.debug(f"Contenido sin cambios en {url}")
                return None
            
            # Crear página con metadatos enriquecidos
            page = ScrapedPage.from_response(url, response, soup, content, title, links)
            
            # Añadir metadatos adicionales
            enhanced_metadata = self.extract_metadata_enhanced(soup, url)
            page.metadata.update(enhanced_metadata)
            
            return page
            
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
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            options.add_argument(f'--user-agent={source.user_agent}')
            
            driver = webdriver.Chrome(options=options)
            driver.get(url)
            
            # Esperar a que el contenido se cargue
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Esperar un poco más para contenido dinámico
            time.sleep(2)
            
            # Obtener HTML procesado
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extraer contenido
            title = self._extract_title(soup, source.title_selectors)
            content = self._extract_content(soup, source)
            links = self._extract_links(soup, url, source)
            
            if len(content) < source.min_content_length:
                return None
            
            # Verificar cambios de contenido
            if not self.detect_content_changes(url, content):
                return None
            
            # Crear respuesta mock para compatibilidad
            class MockResponse:
                def __init__(self):
                    self.status_code = 200
                    self.headers = {'content-type': 'text/html'}
            
            page = ScrapedPage.from_response(url, MockResponse(), soup, content, title, links)
            
            # Metadatos enriquecidos
            enhanced_metadata = self.extract_metadata_enhanced(soup, url)
            page.metadata.update(enhanced_metadata)
            page.metadata['scraped_with'] = 'selenium'
            
            return page
            
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
        
        # Si no se encuentra contenido específico, usar técnicas de readability
        if not content_parts:
            content = self.extract_content_with_readability(soup, source.base_urls[0] if source.base_urls else "")
        else:
            content = ' '.join(content_parts)
        
        # Limpiar y unir contenido
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
        
        # Añadir enlaces de paginación si están habilitados
        pagination_links = self.handle_pagination(soup, base_url)
        for link in pagination_links:
            if link not in links and source.is_url_allowed(link):
                links.append(link)
        
        return links
    
    def _clean_text(self, text: str) -> str:
        """Limpiar texto extraído"""
        # Normalizar espacios en blanco
        text = re.sub(r'\s+', ' ', text)
        
        # Eliminar caracteres especiales problemáticos pero mantener acentos y ñ
        text = re.sub(r'[^\w\s\.,;:!¿?¡ñÑáéíóúÁÉÍÓÚüÜ\(\)\-\[\]\"\'\$\%\+\=\/\\\<\>]', '', text)
        
        # Eliminar líneas con muy pocas palabras (probablemente navegación)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            words = line.strip().split()
            # Mantener líneas con al menos 3 palabras o que sean títulos importantes
            if len(words) >= 3 or any(keyword in line.lower() for keyword in 
                                    ['ayuntamiento', 'municipio', 'procedimiento', 'trámite', 'servicio']):
                cleaned_lines.append(line.strip())
        
        # Eliminar URLs completas del texto
        text = re.sub(r'https?://\S+', '', ' '.join(cleaned_lines))
        
        # Eliminar emails del texto
        text = re.sub(r'\S+@\S+', '', text)
        
        # Normalizar espacios finales
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _is_allowed_by_robots(self, url: str, user_agent: str) -> bool:
        """Verificar si robots.txt permite el acceso"""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            domain = parsed_url.netloc
            if domain not in self.robots_cache:
                rp = RobotFileParser()
                rp.set_url(robots_url)
                try:
                    rp.read()
                    self.robots_cache[domain] = rp
                    self.logger.debug(f"Robots.txt cargado para {domain}")
                except Exception as e:
                    self.logger.warning(f"No se pudo cargar robots.txt de {domain}: {e}")
                    # Si no se puede cargar, asumir que está permitido
                    self.robots_cache[domain] = None
                    return True
            
            rp = self.robots_cache[domain]
            if rp is None:
                return True
            
            return rp.can_fetch(user_agent, url)
            
        except Exception as e:
            self.logger.warning(f"Error verificando robots.txt para {url}: {e}")
            # En caso de error, permitir el acceso por defecto
            return True
    
    def extract_content_with_readability(self, soup: BeautifulSoup, base_url: str) -> str:
        """Extraer contenido principal usando técnicas de readability"""
        # Algoritmo simple de extracción de contenido principal
        
        # 1. Buscar contenedores principales comunes
        main_selectors = [
            'main', '[role="main"]', '.main-content', '#main-content',
            '.content', '#content', '.article-content', '.post-content',
            '.entry-content', '.page-content', 'article', '.article'
        ]
        
        content_parts = []
        
        for selector in main_selectors:
            elements = soup.select(selector)
            for element in elements:
                if element and element.get_text(strip=True):
                    content_parts.append(element.get_text(separator=' ', strip=True))
                    break  # Solo tomar el primero que tenga contenido
            if content_parts:
                break
        
        # 2. Si no se encuentra contenido principal, usar heurísticas
        if not content_parts:
            # Buscar párrafos largos (probablemente contenido)
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 50:  # Párrafos con contenido sustancial
                    content_parts.append(text)
            
            # Buscar divs con mucho texto
            if not content_parts:
                divs = soup.find_all('div')
                for div in divs:
                    text = div.get_text(strip=True)
                    if len(text) > 100 and len(text.split()) > 20:
                        content_parts.append(text)
        
        # 3. Fallback: todo el body limpio
        if not content_parts:
            body = soup.find('body')
            if body:
                content_parts.append(body.get_text(separator=' ', strip=True))
        
        content = ' '.join(content_parts)
        return content
    
    def handle_pagination(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Detectar y manejar paginación"""
        pagination_urls = []
        
        # Selectores comunes para paginación
        pagination_selectors = [
            'a[rel="next"]',
            '.pagination a',
            '.pager a',
            'a:contains("Siguiente")',
            'a:contains("Next")',
            'a[href*="page"]',
            'a[href*="pagina"]'
        ]
        
        for selector in pagination_selectors:
            try:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href:
                        full_url = urljoin(base_url, href)
                        if full_url not in pagination_urls:
                            pagination_urls.append(full_url)
            except Exception as e:
                self.logger.debug(f"Error procesando selector de paginación {selector}: {e}")
        
        return pagination_urls
    
    def detect_content_changes(self, url: str, new_content: str) -> bool:
        """Detectar si el contenido ha cambiado desde la última vez"""
        # Generar hash del contenido
        content_hash = hashlib.md5(new_content.encode('utf-8')).hexdigest()
        
        # Verificar si el contenido cambió
        if url in self._content_hashes:
            if self._content_hashes[url] == content_hash:
                self.logger.debug(f"Contenido sin cambios en {url}")
                return False
        
        self._content_hashes[url] = content_hash
        return True
    
    def extract_metadata_enhanced(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extraer metadatos enriquecidos de la página"""
        metadata = {}
        
        # Meta tags básicos
        meta_tags = {
            'description': soup.find('meta', attrs={'name': 'description'}),
            'keywords': soup.find('meta', attrs={'name': 'keywords'}),
            'author': soup.find('meta', attrs={'name': 'author'}),
            'robots': soup.find('meta', attrs={'name': 'robots'}),
        }
        
        for key, tag in meta_tags.items():
            if tag and tag.get('content'):
                metadata[key] = tag.get('content')
        
        # Open Graph tags
        og_tags = soup.find_all('meta', attrs={'property': lambda x: x and x.startswith('og:')})
        for tag in og_tags:
            property_name = tag.get('property', '').replace('og:', '')
            content = tag.get('content')
            if property_name and content:
                metadata[f'og_{property_name}'] = content
        
        # Datos estructurados JSON-LD
        json_ld_scripts = soup.find_all('script', attrs={'type': 'application/ld+json'})
        if json_ld_scripts:
            structured_data = []
            for script in json_ld_scripts:
                try:
                    if script.string:
                        data = json.loads(script.string)
                        structured_data.append(data)
                except Exception as e:
                    self.logger.debug(f"Error procesando JSON-LD: {e}")
                    pass
            if structured_data:
                metadata['structured_data'] = structured_data
        
        # Información de la página
        parsed_url = urlparse(url)
        lang_attr = soup.find('html', attrs={'lang': True})
        metadata.update({
            'domain': parsed_url.netloc,
            'path': parsed_url.path,
            'lang': lang_attr.get('lang') if lang_attr else None,
            'charset': 'utf-8'  # Asumir UTF-8 por defecto
        })
        
        # Estadísticas del contenido
        text_content = soup.get_text()
        metadata.update({
            'word_count': len(text_content.split()),
            'char_count': len(text_content),
            'paragraph_count': len(soup.find_all('p')),
            'heading_count': len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),
            'link_count': len(soup.find_all('a', href=True)),
            'image_count': len(soup.find_all('img'))
        })
        
        return metadata
    
    def test_url(self, url: str, timeout: int = 10) -> Dict[str, Any]:
        """Probar conectividad y características de una URL"""
        result = {
            'url': url,
            'accessible': False,
            'status_code': None,
            'content_type': None,
            'size': None,
            'title': None,
            'error': None,
            'robots_allowed': True,
            'javascript_required': False
        }
        
        try:
            # Test básico con requests
            response = self.session.get(url, timeout=timeout)
            result['accessible'] = True
            result['status_code'] = response.status_code
            result['content_type'] = response.headers.get('content-type', '')
            result['size'] = len(response.content)
            
            # Parsear contenido básico
            if 'text/html' in result['content_type']:
                soup = BeautifulSoup(response.content, 'html.parser')
                title_tag = soup.find('title')
                result['title'] = title_tag.get_text(strip=True) if title_tag else "Sin título"
                
                # Detectar si requiere JavaScript (heurística básica)
                body_text = soup.get_text(strip=True)
                if len(body_text) < 100 or 'javascript' in body_text.lower():
                    result['javascript_required'] = True
            
            # Verificar robots.txt
            result['robots_allowed'] = self._is_allowed_by_robots(url, self.session.headers.get('User-Agent', '*'))
            
        except Exception as e:
            result['error'] = str(e)
            self.logger.warning(f"Error probando URL {url}: {e}")
        
        return result


# Instancia singleton
web_scraper_service = WebScraperService()

__all__ = ['WebScraperService', 'ScrapedPage', 'web_scraper_service']