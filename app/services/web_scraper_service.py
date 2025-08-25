"""
Servicio de Web Scraping para ingesta de sitios web
TFM Vicente Caruncho - Sistemas Inteligentes

Este servicio maneja:
- Extracción de contenido de sitios web
- Navegación inteligente siguiendo enlaces
- Respeto a robots.txt y rate limiting
- Soporte para JavaScript con Selenium
- Extracción de metadatos enriquecidos
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
from app.models.data_sources import WebSource, ProcessingStatus, ScrapedPage  # <-- AÑADIDO ScrapedPage


class WebScraperService:
    """
    Servicio principal de web scraping
    
    Características:
    - Extracción robusta de contenido con múltiples estrategias
    - Soporte para JavaScript usando Selenium
    - Respeto a robots.txt y rate limiting configurable
    - Detección de cambios de contenido
    - Extracción de metadatos enriquecidos
    - Manejo de paginación automática
    """
    
    def __init__(self):
        """Inicializar servicio con configuración por defecto"""
        self.logger = get_logger("web_scraper")
        self.session = requests.Session()
        self.robots_cache: Dict[str, Optional[RobotFileParser]] = {}
        self._content_hashes: Dict[str, str] = {}
        
        # Configurar session con headers realistas
        self.session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        self.logger.info("WebScraperService inicializado")
        
    def scrape_source(self, source: WebSource) -> List[ScrapedPage]:
        """
        Procesar una fuente web completa navegando por múltiples niveles
        
        Args:
            source: Configuración de la fuente web
            
        Returns:
            Lista de páginas scrapeadas exitosamente
        """
        if not BS4_AVAILABLE:
            raise ImportError("BeautifulSoup4 no está disponible. Instalar con: pip install beautifulsoup4")
        
        self.logger.info(f"Iniciando scraping de fuente: {source.name}")
        
        # Configurar headers específicos de la fuente
        self.session.headers['User-Agent'] = source.user_agent
        
        # Añadir headers personalizados
        if source.custom_headers:
            self.session.headers.update(source.custom_headers)
        
        all_pages = []
        visited_urls: Set[str] = set()
        urls_to_visit = list(source.base_urls)
        current_depth = 0
        
        # Navegación por niveles de profundidad
        while urls_to_visit and current_depth <= source.max_depth:
            current_level_urls = urls_to_visit.copy()
            urls_to_visit.clear()
            
            self.logger.info(f"Procesando nivel {current_depth}: {len(current_level_urls)} URLs")
            
            for url in current_level_urls:
                if url in visited_urls:
                    continue
                
                try:
                    # Verificar robots.txt
                    if source.respect_robots_txt and not self._is_allowed_by_robots(url, source.user_agent):
                        self.logger.info(f"URL bloqueada por robots.txt: {url}")
                        continue
                    
                    # Verificar si la URL está permitida según configuración
                    if not source.is_url_allowed(url):
                        self.logger.debug(f"URL no permitida por configuración: {url}")
                        continue
                    
                    # Scraping de la página individual
                    page = self._scrape_page(url, source)
                    if page:
                        all_pages.append(page)
                        visited_urls.add(url)
                        
                        # Añadir enlaces encontrados para siguiente nivel de profundidad
                        if source.follow_links and current_depth < source.max_depth:
                            for link in page.links_found:
                                if link not in visited_urls and source.is_url_allowed(link):
                                    urls_to_visit.append(link)
                    
                    # Rate limiting configurable
                    time.sleep(source.delay_seconds)
                    
                except Exception as e:
                    self.logger.error(f"Error procesando {url}: {e}")
                    continue
            
            current_depth += 1
        
        self.logger.info(f"Scraping completado: {len(all_pages)} páginas procesadas de {len(visited_urls)} URLs visitadas")
        return all_pages
    
    def _scrape_page(self, url: str, source: WebSource) -> Optional[ScrapedPage]:
        """
        Scraping de una página individual con detección automática de JavaScript
        
        Args:
            url: URL a procesar
            source: Configuración de la fuente
            
        Returns:
            ScrapedPage o None si falla el procesamiento
        """
        try:
            self.logger.debug(f"Procesando URL: {url}")
            
            # Decidir método de scraping según configuración
            if source.use_javascript and SELENIUM_AVAILABLE:
                return self._scrape_with_selenium(url, source)
            else:
                return self._scrape_with_requests(url, source)
                
        except Exception as e:
            self.logger.error(f"Error en scraping de {url}: {e}")
            return None
    
    def _scrape_with_requests(self, url: str, source: WebSource) -> Optional[ScrapedPage]:
        """
        Scraping usando requests + BeautifulSoup (método estándar)
        
        Args:
            url: URL a procesar
            source: Configuración de la fuente
            
        Returns:
            ScrapedPage o None si falla
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Parsear HTML con BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extraer información de la página
            title = self._extract_title(soup, source.title_selectors)
            content = self._extract_content(soup, source)
            links = self._extract_links(soup, url, source)
            
            # Validar contenido mínimo según configuración
            if len(content) < source.min_content_length:
                self.logger.debug(f"Contenido insuficiente en {url}: {len(content)} caracteres")
                return None
            
            # Verificar si el contenido cambió desde la última vez
            if not self.detect_content_changes(url, content):
                self.logger.debug(f"Contenido sin cambios en {url}")
                return None
            
            # Crear página usando el método consolidado de data_sources
            page = ScrapedPage.from_response(
                url=url,
                title=title,
                content=content,
                links=links,
                source_id=source.id,
                response=response
            )
            
            # Añadir metadatos enriquecidos específicos del scraper
            enhanced_metadata = self.extract_metadata_enhanced(soup, url)
            page.metadata.update(enhanced_metadata)
            page.metadata['scraper_method'] = 'requests'
            
            return page
            
        except Exception as e:
            self.logger.error(f"Error en requests scraping de {url}: {e}")
            return None
    
    def _scrape_with_selenium(self, url: str, source: WebSource) -> Optional[ScrapedPage]:
        """
        Scraping usando Selenium para contenido que requiere JavaScript
        
        Args:
            url: URL a procesar
            source: Configuración de la fuente
            
        Returns:
            ScrapedPage o None si falla
        """
        if not SELENIUM_AVAILABLE:
            self.logger.warning("Selenium no disponible, fallback a requests")
            return self._scrape_with_requests(url, source)
        
        driver = None
        try:
            # Configurar Chrome en modo headless
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            options.add_argument(f'--user-agent={source.user_agent}')
            options.add_argument('--disable-blink-features=AutomationControlled')
            
            driver = webdriver.Chrome(options=options)
            driver.get(url)
            
            # Esperar a que el contenido se cargue completamente
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Esperar tiempo adicional para contenido dinámico
            time.sleep(2)
            
            # Obtener HTML procesado por JavaScript
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extraer información
            title = self._extract_title(soup, source.title_selectors)
            content = self._extract_content(soup, source)
            links = self._extract_links(soup, url, source)
            
            # Validaciones
            if len(content) < source.min_content_length:
                self.logger.debug(f"Contenido insuficiente en {url}: {len(content)} caracteres")
                return None
            
            if not self.detect_content_changes(url, content):
                self.logger.debug(f"Contenido sin cambios en {url}")
                return None
            
            # Crear página con método consolidado (sin response real de HTTP)
            page = ScrapedPage.from_response(
                url=url,
                title=title,
                content=content,
                links=links,
                source_id=source.id,
                response=None  # Selenium no proporciona response object
            )
            
            # Metadatos enriquecidos
            enhanced_metadata = self.extract_metadata_enhanced(soup, url)
            page.metadata.update(enhanced_metadata)
            page.metadata['scraper_method'] = 'selenium'
            page.metadata['javascript_rendered'] = True
            
            return page
            
        except Exception as e:
            self.logger.error(f"Error en Selenium scraping de {url}: {e}")
            return None
        finally:
            if driver:
                driver.quit()
    
    def _extract_title(self, soup: BeautifulSoup, selectors: List[str]) -> str:
        """
        Extraer título de la página usando selectores configurados
        
        Args:
            soup: BeautifulSoup parseado
            selectors: Lista de selectores CSS para el título
            
        Returns:
            Título extraído o "Sin título" si no se encuentra
        """
        for selector in selectors:
            try:
                element = soup.select_one(selector)
                if element and element.get_text(strip=True):
                    return element.get_text(strip=True)
            except Exception as e:
                self.logger.debug(f"Error con selector de título {selector}: {e}")
                continue
        
        # Fallback al title tag estándar
        title_tag = soup.find('title')
        return title_tag.get_text(strip=True) if title_tag else "Sin título"
    
    def _extract_content(self, soup: BeautifulSoup, source: WebSource) -> str:
        """
        Extraer contenido principal usando selectores y técnicas de readability
        
        Args:
            soup: BeautifulSoup parseado
            source: Configuración de la fuente con selectores
            
        Returns:
            Contenido principal extraído y limpiado
        """
        # Eliminar elementos no deseados según configuración
        for selector in source.exclude_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Extraer contenido de selectores específicos configurados
        content_parts = []
        for selector in source.content_selectors:
            try:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(separator=' ', strip=True)
                    if text:
                        content_parts.append(text)
            except Exception as e:
                self.logger.debug(f"Error con selector de contenido {selector}: {e}")
                continue
        
        # Si no se encuentra contenido específico, usar técnicas de readability
        if not content_parts:
            content = self.extract_content_with_readability(soup, source.base_urls[0] if source.base_urls else "")
        else:
            content = ' '.join(content_parts)
        
        # Limpiar y normalizar el texto extraído
        return self._clean_text(content)
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str, source: WebSource) -> List[str]:
        """
        Extraer enlaces válidos de la página incluyendo paginación
        
        Args:
            soup: BeautifulSoup parseado
            base_url: URL base para resolver enlaces relativos
            source: Configuración de la fuente
            
        Returns:
            Lista de URLs válidas encontradas
        """
        links = set()  # Usar set para evitar duplicados
        
        # Extraer enlaces estándar
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            # Normalizar URL (remover fragmentos y parámetros innecesarios)
            parsed = urlparse(full_url)
            normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            
            if source.is_url_allowed(normalized_url):
                links.add(normalized_url)
        
        # Añadir enlaces de paginación si están disponibles
        pagination_links = self.handle_pagination(soup, base_url)
        for link in pagination_links:
            if source.is_url_allowed(link):
                links.add(link)
        
        return list(links)
    
    def _clean_text(self, text: str) -> str:
        """
        Limpiar y normalizar texto extraído manteniendo contenido relevante
        
        Args:
            text: Texto bruto extraído
            
        Returns:
            Texto limpio y normalizado
        """
        if not text:
            return ""
            
        # Normalizar espacios en blanco múltiples
        text = re.sub(r'\s+', ' ', text)
        
        # Eliminar caracteres especiales problemáticos pero mantener acentos y ñ
        text = re.sub(r'[^\w\s\.,;:!¿?¡ñÑáéíóúÁÉÍÓÚüÜ\(\)\-\[\]\"\'\$\%\+\=\/\\\<\>]', '', text)
        
        # Procesar líneas individualmente para filtrar navegación
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            words = line.strip().split()
            # Mantener líneas con contenido sustancial o términos importantes para administraciones
            if len(words) >= 3 or any(keyword in line.lower() for keyword in 
                                    ['ayuntamiento', 'municipio', 'procedimiento', 'trámite', 'servicio',
                                     'normativa', 'ordenanza', 'decreto', 'resolución', 'convocatoria']):
                cleaned_lines.append(line.strip())
        
        # Unir líneas filtradas
        text = ' '.join(cleaned_lines)
        
        # Eliminar URLs completas del texto (no aportan al contenido semántico)
        text = re.sub(r'https?://\S+', '', text)
        
        # Eliminar direcciones de email del texto
        text = re.sub(r'\S+@\S+', '', text)
        
        # Normalización final de espacios
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _is_allowed_by_robots(self, url: str, user_agent: str) -> bool:
        """
        Verificar si robots.txt permite el acceso a la URL
        
        Args:
            url: URL a verificar
            user_agent: User agent a usar para la verificación
            
        Returns:
            True si está permitido, False si está bloqueado
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Cache por dominio para eficiencia
            if domain not in self.robots_cache:
                robots_url = f"{parsed_url.scheme}://{domain}/robots.txt"
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
        """
        Extraer contenido principal usando técnicas de readability automática
        
        Args:
            soup: BeautifulSoup parseado
            base_url: URL base de la página
            
        Returns:
            Contenido principal extraído
        """
        # Algoritmo de extracción de contenido principal
        
        # 1. Buscar contenedores principales comunes en sitios web
        main_selectors = [
            'main', '[role="main"]', '.main-content', '#main-content',
            '.content', '#content', '.article-content', '.post-content',
            '.entry-content', '.page-content', 'article', '.article',
            '.container', '#container', '.wrapper', '.main-wrapper'
        ]
        
        content_parts = []
        
        for selector in main_selectors:
            elements = soup.select(selector)
            for element in elements:
                if element and element.get_text(strip=True):
                    content_parts.append(element.get_text(separator=' ', strip=True))
                    break  # Solo tomar el primero que tenga contenido sustancial
            if content_parts:
                break
        
        # 2. Si no se encuentra contenido principal, usar heurísticas
        if not content_parts:
            # Buscar párrafos largos (probablemente contenido real)
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 50:  # Párrafos con contenido sustancial
                    content_parts.append(text)
            
            # Buscar divs con mucho texto (posibles contenedores de contenido)
            if not content_parts:
                divs = soup.find_all('div')
                for div in divs:
                    text = div.get_text(strip=True)
                    if len(text) > 100 and len(text.split()) > 20:
                        content_parts.append(text)
                        break  # Solo el primero que cumpla criterios
        
        # 3. Fallback: todo el body limpio (último recurso)
        if not content_parts:
            body = soup.find('body')
            if body:
                content_parts.append(body.get_text(separator=' ', strip=True))
        
        return ' '.join(content_parts)
    
    def handle_pagination(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Detectar y extraer enlaces de paginación automáticamente
        
        Args:
            soup: BeautifulSoup parseado
            base_url: URL base para resolver enlaces relativos
            
        Returns:
            Lista de URLs de paginación encontradas
        """
        pagination_urls = set()
        
        # Selectores comunes para paginación en sitios web españoles/internacionales
        pagination_selectors = [
            'a[rel="next"]',
            '.pagination a', '.pager a', '.paginator a',
            'a:contains("Siguiente")', 'a:contains("Next")',
            'a:contains("»")', 'a:contains(">")',
            'a[href*="page"]', 'a[href*="pagina"]',
            'a[href*="pag"]', 'a[href*="p="]'
        ]
        
        for selector in pagination_selectors:
            try:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href:
                        full_url = urljoin(base_url, href)
                        # Normalizar y validar URL
                        parsed = urlparse(full_url)
                        if parsed.scheme and parsed.netloc:
                            normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                            pagination_urls.add(normalized_url)
            except Exception as e:
                self.logger.debug(f"Error procesando selector de paginación {selector}: {e}")
        
        return list(pagination_urls)
    
    def detect_content_changes(self, url: str, new_content: str) -> bool:
        """
        Detectar si el contenido de una URL ha cambiado desde la última extracción
        
        Args:
            url: URL a verificar
            new_content: Nuevo contenido extraído
            
        Returns:
            True si el contenido cambió, False si es igual
        """
        # Generar hash MD5 del contenido para comparación eficiente
        content_hash = hashlib.md5(new_content.encode('utf-8')).hexdigest()
        
        # Verificar si el contenido cambió desde la última vez
        if url in self._content_hashes:
            if self._content_hashes[url] == content_hash:
                return False  # Contenido sin cambios
        
        # Actualizar hash para futura comparación
        self._content_hashes[url] = content_hash
        return True  # Contenido nuevo o modificado
    
    def extract_metadata_enhanced(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Extraer metadatos enriquecidos para análisis avanzado
        
        Args:
            soup: BeautifulSoup parseado
            url: URL de la página
            
        Returns:
            Diccionario con metadatos enriquecidos
        """
        metadata = {}
        
        # Meta tags estándar HTML
        meta_tags = {
            'description': soup.find('meta', attrs={'name': 'description'}),
            'keywords': soup.find('meta', attrs={'name': 'keywords'}),
            'author': soup.find('meta', attrs={'name': 'author'}),
            'robots': soup.find('meta', attrs={'name': 'robots'}),
            'viewport': soup.find('meta', attrs={'name': 'viewport'}),
        }
        
        for key, tag in meta_tags.items():
            if tag and tag.get('content'):
                metadata[key] = tag.get('content')
        
        # Open Graph tags para redes sociales
        og_tags = soup.find_all('meta', attrs={'property': lambda x: x and x.startswith('og:')})
        for tag in og_tags:
            property_name = tag.get('property', '').replace('og:', '')
            content = tag.get('content')
            if property_name and content:
                metadata[f'og_{property_name}'] = content
        
        # Twitter Card tags
        twitter_tags = soup.find_all('meta', attrs={'name': lambda x: x and x.startswith('twitter:')})
        for tag in twitter_tags:
            name = tag.get('name', '').replace('twitter:', '')
            content = tag.get('content')
            if name and content:
                metadata[f'twitter_{name}'] = content
        
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
        
        # Información técnica de la página
        parsed_url = urlparse(url)
        lang_attr = soup.find('html', attrs={'lang': True})
        charset_meta = soup.find('meta', attrs={'charset': True})
        
        metadata.update({
            'domain': parsed_url.netloc,
            'path': parsed_url.path,
            'lang': lang_attr.get('lang') if lang_attr else None,
            'charset': charset_meta.get('charset') if charset_meta else 'utf-8',
            'extraction_timestamp': datetime.now().isoformat()
        })
        
        # Estadísticas del contenido para análisis
        text_content = soup.get_text()
        metadata.update({
            'content_stats': {
                'word_count': len(text_content.split()),
                'char_count': len(text_content),
                'paragraph_count': len(soup.find_all('p')),
                'heading_count': len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),
                'link_count': len(soup.find_all('a', href=True)),
                'image_count': len(soup.find_all('img')),
                'list_count': len(soup.find_all(['ul', 'ol'])),
                'table_count': len(soup.find_all('table'))
            }
        })
        
        return metadata
    
    def test_url(self, url: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Probar conectividad y características de una URL específica
        
        Args:
            url: URL a probar
            timeout: Timeout en segundos
            
        Returns:
            Diccionario con resultados del test
        """
        result = {
            'url': url,
            'accessible': False,
            'status_code': None,
            'content_type': None,
            'size': None,
            'title': None,
            'error': None,
            'robots_allowed': True,
            'javascript_required': False,
            'response_time': None
        }
        
        start_time = time.time()
        
        try:
            # Test de conectividad básica
            response = self.session.get(url, timeout=timeout)
            result['accessible'] = True
            result['status_code'] = response.status_code
            result['content_type'] = response.headers.get('content-type', '')
            result['size'] = len(response.content)
            result['response_time'] = time.time() - start_time
            
            # Análisis de contenido si es HTML
            if 'text/html' in result['content_type']:
                soup = BeautifulSoup(response.content, 'html.parser')
                title_tag = soup.find('title')
                result['title'] = title_tag.get_text(strip=True) if title_tag else "Sin título"
                
                # Detectar si requiere JavaScript (heurística básica)
                body_text = soup.get_text(strip=True)
                if len(body_text) < 100 or 'javascript' in body_text.lower() or 'loading' in body_text.lower():
                    result['javascript_required'] = True
            
            # Verificar robots.txt
            result['robots_allowed'] = self._is_allowed_by_robots(url, self.session.headers.get('User-Agent', '*'))
            
        except Exception as e:
            result['error'] = str(e)
            result['response_time'] = time.time() - start_time
            self.logger.warning(f"Error probando URL {url}: {e}")
        
        return result


# =============================================================================
# INSTANCIA GLOBAL DEL SERVICIO
# =============================================================================

# Instancia singleton para uso en toda la aplicación
web_scraper_service = WebScraperService()

# Exportaciones públicas del módulo
__all__ = ['WebScraperService', 'web_scraper_service']  # <-- QUITADO ScrapedPage