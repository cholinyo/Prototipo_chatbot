"""
API Routes para gestión de fuentes web - LIMPIO Y CORREGIDO
TFM Vicente Caruncho - Sistemas Inteligentes

CAMBIOS APLICADOS:
- Eliminadas referencias a enhanced_web_scraper
- Estructura de datos consistente con modelo WebSource
- Endpoints simplificados y funcionles
- Compatibilidad completa con web_scraper_service y web_ingestion_service
"""

import json
import os
import time
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from flask import Blueprint, request, jsonify
import requests
from bs4 import BeautifulSoup

# Selenium imports con manejo robusto de errores
SELENIUM_AVAILABLE = False
SELENIUM_ERROR = "No configurado"

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        WEBDRIVER_MANAGER_AVAILABLE = True
    except ImportError:
        WEBDRIVER_MANAGER_AVAILABLE = False
    
    SELENIUM_AVAILABLE = True
    SELENIUM_ERROR = None
    
except ImportError as e:
    SELENIUM_AVAILABLE = False
    SELENIUM_ERROR = f"Selenium no instalado: {e}"

# IMPORTACIONES LIMPIAS - Solo modelo de datos
from app.models.data_sources import (
    WebSource, 
    ScrapedPage, 
    DataSourceStatus,
    ProcessingStatus,
    create_web_source
)
from app.core.logger import get_logger

# Blueprint
web_sources_api = Blueprint('web_sources_api', __name__, url_prefix='/api/web-sources')
logger = get_logger("web_sources_api")

# Configuración
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
SCRAPED_CONTENT_DIR = DATA_DIR / "scraped_content"
SCRAPED_CONTENT_DIR.mkdir(exist_ok=True)

WEB_SOURCES_FILE = DATA_DIR / "web_sources.json"
SCRAPING_TASKS_FILE = DATA_DIR / "scraping_tasks.json"

# Variables globales para tareas activas
active_tasks = {}
task_lock = threading.Lock()


class WebScrapingService:
    """Servicio de web scraping simplificado y funcional"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.selenium_working = None
        
    def check_selenium_status(self):
        """Verificar si Selenium funciona correctamente"""
        if self.selenium_working is not None:
            return self.selenium_working
        
        if not SELENIUM_AVAILABLE:
            self.selenium_working = False
            logger.warning(f"Selenium no disponible: {SELENIUM_ERROR}")
            return False
        
        try:
            driver = self.get_driver()
            if driver:
                driver.quit()
                self.selenium_working = True
                logger.info("Selenium funcionando correctamente")
                return True
        except Exception as e:
            self.selenium_working = False
            logger.error(f"Selenium no funciona: {e}")
            return False
        
        self.selenium_working = False
        return False
        
    def get_driver(self):
        """Crear driver de Selenium con configuración optimizada"""
        if not SELENIUM_AVAILABLE:
            raise Exception(f"Selenium no disponible: {SELENIUM_ERROR}")
        
        try:
            chrome_options = ChromeOptions()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            
            if WEBDRIVER_MANAGER_AVAILABLE:
                service = webdriver.chrome.service.Service(ChromeDriverManager().install())
            else:
                service = None
            
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.set_page_load_timeout(30)
            driver.implicitly_wait(10)
            
            return driver
            
        except Exception as e:
            raise Exception(f"Error creando driver: {e}")

    def test_url_connectivity(self, url: str) -> Dict[str, Any]:
        """Test básico de conectividad"""
        try:
            response = self.session.head(url, timeout=10, allow_redirects=True)
            return {
                'success': True,
                'status_code': response.status_code,
                'accessible': response.status_code < 400,
                'content_type': response.headers.get('content-type', ''),
                'server': response.headers.get('server', '')
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'accessible': False
            }

    def scrape_with_requests(self, url: str, min_content_length: int = 50) -> Dict[str, Any]:
        """Scraping con requests + BeautifulSoup"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Limpiar elementos no deseados
            for script in soup(["script", "style", "nav", "footer", ".sidebar", "#sidebar"]):
                script.decompose()
            
            title = soup.find('title').get_text(strip=True) if soup.find('title') else url
            content = soup.get_text(separator=' ', strip=True)
            
            # Extraer enlaces
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith(('http://', 'https://')):
                    links.append(href)
            
            return {
                'success': True,
                'method': 'requests',
                'title': title,
                'content': content,
                'links': links[:20],  # Limitar enlaces
                'content_length': len(content),
                'sufficient_content': len(content) >= min_content_length,
                'status_code': response.status_code
            }
            
        except Exception as e:
            return {
                'success': False,
                'method': 'requests',
                'error': str(e),
                'content_length': 0,
                'sufficient_content': False
            }

    def scrape_with_selenium(self, url: str, min_content_length: int = 50) -> Dict[str, Any]:
        """Scraping con Selenium para contenido dinámico"""
        if not self.check_selenium_status():
            return {
                'success': False,
                'method': 'selenium',
                'error': 'Selenium no disponible',
                'content_length': 0,
                'sufficient_content': False
            }
        
        driver = None
        try:
            driver = self.get_driver()
            logger.info(f"Navegando a {url} con Selenium")
            
            driver.get(url)
            
            # Esperar carga completa
            WebDriverWait(driver, 15).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            time.sleep(3)  # Tiempo adicional para JavaScript
            
            title = driver.title or url
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Limpiar elementos
            for element in soup(["script", "style", "nav", "footer", ".sidebar"]):
                element.decompose()
            
            content = soup.get_text(separator=' ', strip=True)
            
            # Extraer enlaces
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith(('http://', 'https://')):
                    links.append(href)
            
            return {
                'success': True,
                'method': 'selenium',
                'title': title,
                'content': content,
                'links': links[:20],
                'content_length': len(content),
                'sufficient_content': len(content) >= min_content_length,
                'javascript_rendered': True
            }
            
        except Exception as e:
            logger.error(f"Error Selenium: {e}")
            return {
                'success': False,
                'method': 'selenium',
                'error': str(e),
                'content_length': 0,
                'sufficient_content': False
            }
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass

    def scrape_auto(self, url: str, min_content_length: int = 50) -> Dict[str, Any]:
        """Scraping automático con fallback inteligente"""
        logger.info(f"Scraping automático de {url}")
        
        # Primero intentar con requests
        requests_result = self.scrape_with_requests(url, min_content_length)
        
        if requests_result['success'] and requests_result['sufficient_content']:
            logger.info(f"Requests exitoso: {requests_result['content_length']} chars")
            return requests_result
        
        # Fallback a Selenium si está disponible
        if self.check_selenium_status():
            logger.info("Fallback a Selenium...")
            selenium_result = self.scrape_with_selenium(url, min_content_length)
            
            if selenium_result['success'] and selenium_result['sufficient_content']:
                logger.info(f"Selenium exitoso: {selenium_result['content_length']} chars")
                return selenium_result
        
        # Devolver mejor resultado disponible
        return requests_result if requests_result.get('content_length', 0) > 0 else {
            'success': False,
            'method': 'auto',
            'error': 'Todos los métodos fallaron',
            'content_length': 0,
            'sufficient_content': False
        }


# Instancia global del servicio
scraping_service = WebScrapingService()


# ===== FUNCIONES DE PERSISTENCIA =====

def load_web_sources() -> Dict[str, WebSource]:
    """Cargar fuentes web desde archivo JSON"""
    if not WEB_SOURCES_FILE.exists():
        return {}
    
    try:
        with open(WEB_SOURCES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sources = {}
        for source_id, source_data in data.items():
            try:
                # Crear WebSource desde diccionario
                web_source = WebSource.from_dict(source_data)
                sources[source_id] = web_source
                logger.debug(f"Fuente cargada: {web_source.name}")
                
            except Exception as e:
                logger.error(f"Error cargando fuente {source_id}: {e}")
                continue
        
        logger.info(f"Cargadas {len(sources)} fuentes web")
        return sources
        
    except Exception as e:
        logger.error(f"Error cargando fuentes web: {e}")
        return {}


def save_web_sources(sources: Dict[str, WebSource]):
    """Guardar fuentes web usando modelo WebSource"""
    try:
        # Convertir todas las fuentes a diccionarios
        data = {}
        for source_id, web_source in sources.items():
            data[source_id] = web_source.to_dict()
        
        with open(WEB_SOURCES_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Guardadas {len(sources)} fuentes web")
        
    except Exception as e:
        logger.error(f"Error guardando fuentes web: {e}")


def save_scraped_content(source_id: str, url: str, result: Dict[str, Any]) -> Optional[str]:
    """Guardar contenido scrapeado usando modelo ScrapedPage"""
    if not result.get('success') or not result.get('sufficient_content'):
        return None
    
    try:
        # Crear ScrapedPage usando el modelo correcto
        scraped_page = ScrapedPage.from_response(
            url=url,
            title=result.get('title', ''),
            content=result.get('content', ''),
            links=result.get('links', []),
            source_id=source_id
        )
        
        # Actualizar estado de procesamiento
        scraped_page.update_processing_status(ProcessingStatus.COMPLETED)
        
        # Crear directorio para la fuente
        source_dir = SCRAPED_CONTENT_DIR / source_id
        source_dir.mkdir(exist_ok=True)
        
        # Generar nombre de archivo único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{scraped_page.id}.json"
        content_file = source_dir / filename
        
        # Guardar usando to_dict()
        with open(content_file, 'w', encoding='utf-8') as f:
            json.dump(scraped_page.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Contenido guardado: {content_file}")
        return str(content_file)
        
    except Exception as e:
        logger.error(f"Error guardando contenido: {e}")
        return None


# ===== ENDPOINTS DE LA API =====

@web_sources_api.route('/methods', methods=['GET'])
def get_scraping_methods():
    """Obtener métodos de scraping disponibles - SIMPLIFICADO"""
    try:
        selenium_status = scraping_service.check_selenium_status()
        
        methods = [
            {
                'id': 'requests',
                'name': 'Requests + BeautifulSoup',
                'description': 'Método rápido para sitios estáticos',
                'available': True,
                'pros': ['Muy rápido', 'Bajo consumo de recursos'],
                'cons': ['Sin soporte JavaScript'],
                'use_cases': ['Portales institucionales', 'Sitios estáticos', 'Páginas simples']
            },
            {
                'id': 'selenium',
                'name': 'Selenium WebDriver',
                'description': 'Navegador automatizado para contenido dinámico',
                'available': selenium_status,
                'pros': ['Ejecuta JavaScript', 'Contenido dinámico'],
                'cons': ['Más lento', 'Mayor consumo de recursos'],
                'use_cases': ['SPAs', 'Sitios con JavaScript', 'Contenido dinámico'],
                'error': None if selenium_status else SELENIUM_ERROR
            }
        ]
        
        return jsonify({
            'success': True,
            'methods': methods,
            'selenium_available': selenium_status
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo métodos: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@web_sources_api.route('', methods=['GET'])
def list_web_sources():
    """Listar todas las fuentes web configuradas"""
    try:
        sources = load_web_sources()
        
        # Convertir a formato de respuesta
        sources_list = []
        for source_id, web_source in sources.items():
            source_dict = web_source.to_dict()
            
            # Añadir campos para compatibilidad con frontend
            source_dict.update({
                'scraping_method': web_source.config.get('scraping_method', 'requests'),
                'pages_found': web_source.metadata.get('total_pages_processed', 0),
                'success_rate': web_source.metadata.get('success_rate', 0.0),
                'last_activity': web_source.last_sync,
                'is_active': web_source.metadata.get('scraping_active', False),
                'url': web_source.base_urls[0] if web_source.base_urls else '',
                'method': web_source.config.get('scraping_method', 'requests')
            })
            
            sources_list.append(source_dict)
        
        return jsonify({
            'success': True,
            'sources': sources_list,
            'total': len(sources),
            'active_count': sum(1 for s in sources_list if s.get('is_active', False))
        })
        
    except Exception as e:
        logger.error(f"Error listando fuentes: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@web_sources_api.route('', methods=['POST'])
def create_web_source_endpoint():
    """Crear nueva fuente web con estructura WebSource correcta"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        logger.info(f"Datos recibidos del frontend: {data}")
        
        # Validaciones
        required_fields = ['name', 'type', 'base_urls']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'success': False, 
                'error': f'Campos requeridos faltantes: {missing_fields}'
            }), 400
        
        if not data['base_urls'] or len(data['base_urls']) == 0:
            return jsonify({'success': False, 'error': 'base_urls no puede estar vacío'}), 400
        
        if data['type'] != 'web':
            return jsonify({'success': False, 'error': 'type debe ser "web"'}), 400
        
        # Crear WebSource desde datos estructurados
        try:
            web_source = WebSource.from_dict(data)
            logger.info(f"WebSource creado: {web_source.name}")
        except Exception as e:
            logger.error(f"Error creando WebSource: {e}")
            return jsonify({'success': False, 'error': f'Error en estructura de datos: {e}'}), 400
        
        # Establecer valores por defecto
        web_source.status = DataSourceStatus.ACTIVE
        if not web_source.created_at:
            web_source.created_at = datetime.now()
        
        # Cargar, añadir y guardar
        sources = load_web_sources()
        sources[web_source.id] = web_source
        save_web_sources(sources)
        
        logger.info(f"Fuente web creada: {web_source.name} (ID: {web_source.id})")
        
        return jsonify({
            'success': True,
            'source': web_source.to_dict(),
            'message': f'Fuente creada: {web_source.name}',
            'source_id': web_source.id
        }), 201
        
    except Exception as e:
        logger.error(f"Error creando fuente: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@web_sources_api.route('/test-url', methods=['POST'])
def test_url():
    """Test de conectividad y scraping de URL"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'success': False, 'error': 'URL requerida'}), 400
        
        url = data['url']
        method = data.get('method', 'auto')
        min_content_length = data.get('min_content_length', 50)
        
        logger.info(f"Testing: {url} con método {method}")
        
        # Test conectividad
        connectivity = scraping_service.test_url_connectivity(url)
        
        # Test scraping según método
        if method == 'requests':
            scraping_result = scraping_service.scrape_with_requests(url, min_content_length)
        elif method == 'selenium':
            scraping_result = scraping_service.scrape_with_selenium(url, min_content_length)
        else:  # auto
            scraping_result = scraping_service.scrape_auto(url, min_content_length)
        
        return jsonify({
            'success': True,
            'url': url,
            'connectivity': connectivity,
            'scraping': scraping_result,
            'recommendation': scraping_result.get('method', 'requests')
        })
        
    except Exception as e:
        logger.error(f"Error test URL: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@web_sources_api.route('/scraping/start/<source_id>', methods=['POST'])
def start_scraping(source_id: str):
    """Iniciar proceso de scraping para una fuente"""
    try:
        with task_lock:
            if source_id in active_tasks:
                return jsonify({
                    'success': False,
                    'error': 'Scraping ya en progreso para esta fuente'
                }), 409
        
        # Cargar fuente
        sources = load_web_sources()
        if source_id not in sources:
            return jsonify({'success': False, 'error': 'Fuente no encontrada'}), 404
        
        web_source = sources[source_id]
        task_id = f"{source_id}_{int(time.time())}"
        
        # Inicializar tarea
        with task_lock:
            active_tasks[source_id] = {
                'task_id': task_id,
                'status': 'starting',
                'started_at': datetime.now().isoformat(),
                'total_urls': len(web_source.base_urls),
                'processed_urls': 0,
                'successful_pages': 0,
                'failed_pages': 0
            }
        
        def run_scraping():
            """Función de scraping en hilo separado"""
            try:
                with task_lock:
                    active_tasks[source_id]['status'] = 'running'
                
                # Actualizar estado de la fuente
                web_source.status = DataSourceStatus.PROCESSING
                web_source.last_sync = datetime.now()
                web_source.metadata['scraping_active'] = True
                
                # Obtener configuración
                min_content_length = web_source.config.get('min_content_length', 100)
                delay_seconds = web_source.config.get('delay_seconds', 1.0)
                use_javascript = web_source.config.get('use_javascript', False)
                
                # Procesar cada URL
                for url in web_source.base_urls:
                    try:
                        logger.info(f"Scrapeando: {url}")
                        
                        # Determinar método según configuración
                        if use_javascript:
                            result = scraping_service.scrape_with_selenium(url, min_content_length)
                        else:
                            result = scraping_service.scrape_auto(url, min_content_length)
                        
                        with task_lock:
                            active_tasks[source_id]['processed_urls'] += 1
                            
                            if result.get('success') and result.get('sufficient_content'):
                                active_tasks[source_id]['successful_pages'] += 1
                                save_scraped_content(source_id, url, result)
                            else:
                                active_tasks[source_id]['failed_pages'] += 1
                        
                        # Respetar delay
                        time.sleep(delay_seconds)
                        
                    except Exception as e:
                        logger.error(f"Error scrapeando {url}: {e}")
                        with task_lock:
                            active_tasks[source_id]['failed_pages'] += 1
                
                # Finalizar
                with task_lock:
                    task_info = active_tasks[source_id]
                    task_info['status'] = 'completed'
                    task_info['completed_at'] = datetime.now().isoformat()
                
                # Actualizar metadata de la fuente
                web_source.metadata.update({
                    'total_pages_processed': task_info['processed_urls'],
                    'successful_pages': task_info['successful_pages'],
                    'failed_pages': task_info['failed_pages'],
                    'success_rate': (task_info['successful_pages'] / max(task_info['processed_urls'], 1)) * 100,
                    'scraping_active': False
                })
                
                web_source.status = DataSourceStatus.ACTIVE
                
                # Guardar cambios
                sources[source_id] = web_source
                save_web_sources(sources)
                
                logger.info(f"Scraping completado: {source_id}")
                
            except Exception as e:
                logger.error(f"Error en scraping: {e}")
                with task_lock:
                    active_tasks[source_id]['status'] = 'failed'
                    active_tasks[source_id]['error'] = str(e)
                
                # Limpiar estado activo
                web_source.metadata['scraping_active'] = False
                web_source.status = DataSourceStatus.ERROR
                sources[source_id] = web_source
                save_web_sources(sources)
        
        # Ejecutar en hilo separado
        thread = threading.Thread(target=run_scraping, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': f'Scraping iniciado para {web_source.name}'
        })
        
    except Exception as e:
        logger.error(f"Error iniciando scraping: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@web_sources_api.route('/scraping/status/<source_id>', methods=['GET'])
def get_scraping_status(source_id: str):
    """Obtener estado del scraping de una fuente"""
    try:
        with task_lock:
            task_info = active_tasks.get(source_id)
        
        if not task_info:
            return jsonify({
                'success': True,
                'status': 'not_running',
                'message': 'No hay scraping activo para esta fuente'
            })
        
        return jsonify({
            'success': True,
            'task_info': task_info
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo estado: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@web_sources_api.route('/<source_id>', methods=['DELETE'])
def delete_web_source(source_id: str):
    """Eliminar fuente web y su contenido asociado"""
    try:
        sources = load_web_sources()
        
        if source_id not in sources:
            return jsonify({'success': False, 'error': 'Fuente no encontrada'}), 404
        
        source_name = sources[source_id].name
        
        # Eliminar fuente
        del sources[source_id]
        save_web_sources(sources)
        
        # Eliminar contenido scrapeado asociado
        source_dir = SCRAPED_CONTENT_DIR / source_id
        if source_dir.exists():
            try:
                import shutil
                shutil.rmtree(source_dir)
                logger.info(f"Contenido eliminado: {source_dir}")
            except Exception as e:
                logger.warning(f"Error eliminando contenido: {e}")
        
        # Limpiar tareas activas
        with task_lock:
            if source_id in active_tasks:
                del active_tasks[source_id]
        
        return jsonify({
            'success': True,
            'message': f'Fuente eliminada: {source_name}'
        })
        
    except Exception as e:
        logger.error(f"Error eliminando fuente: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@web_sources_api.route('/system/status', methods=['GET'])
def get_system_status():
    """Estado general del sistema de scraping"""
    try:
        sources = load_web_sources()
        selenium_status = scraping_service.check_selenium_status()
        
        return jsonify({
            'success': True,
            'system_info': {
                'selenium_available': selenium_status,
                'selenium_error': SELENIUM_ERROR if not selenium_status else None,
                'active_tasks': len(active_tasks),
                'total_sources': len(sources),
                'data_directory': str(DATA_DIR),
                'scraped_content_directory': str(SCRAPED_CONTENT_DIR)
            }
        })
        
    except Exception as e:
        logger.error(f"Error estado sistema: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500