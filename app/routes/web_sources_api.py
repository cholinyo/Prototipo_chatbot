"""
API Routes para gestión de fuentes web - ARREGLADO PARA CONSISTENCIA
TFM Vicente Caruncho - Sistemas Inteligentes

CAMBIOS PRINCIPALES:
1. Backend recibe estructura WebSource correcta del frontend
2. Manejo automático de migración legacy
3. Validaciones robustas de datos
4. Compatibilidad con modelo WebSource consolidado
"""

import json
import os
import time
import threading
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

# CRÍTICO: Importar modelo de datos correcto
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

# Variables globales
active_tasks = {}
task_lock = threading.Lock()

class WebScrapingService:
    """Servicio de web scraping híbrido"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.selenium_working = None
        
    def check_selenium_status(self):
        """Verificar si Selenium funciona"""
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
                logger.info("Selenium funcionando")
                return True
        except Exception as e:
            self.selenium_working = False
            logger.error(f"Selenium no funciona: {e}")
            return False
        
        self.selenium_working = False
        return False
        
    def get_driver(self):
        """Crear driver de Selenium"""
        if not SELENIUM_AVAILABLE:
            raise Exception(f"Selenium no disponible: {SELENIUM_ERROR}")
        
        try:
            chrome_options = ChromeOptions()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            
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
        """Test de conectividad"""
        try:
            response = self.session.head(url, timeout=10, allow_redirects=True)
            return {
                'success': True,
                'status_code': response.status_code,
                'accessible': response.status_code < 400
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'accessible': False
            }

    def scrape_with_requests(self, url: str, min_content_length: int = 50) -> Dict[str, Any]:
        """Scraping con requests"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Limpiar
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()
            
            title = soup.find('title').get_text(strip=True) if soup.find('title') else url
            content = soup.get_text(separator=' ', strip=True)
            
            return {
                'success': True,
                'method': 'requests',
                'title': title,
                'content': content,
                'content_length': len(content),
                'sufficient_content': len(content) >= min_content_length
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
        """Scraping con Selenium"""
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
            logger.info(f"Navegando a {url}")
            
            driver.get(url)
            
            # Esperar carga
            WebDriverWait(driver, 15).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            time.sleep(3)
            
            title = driver.title or url
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Limpiar
            for element in soup(["script", "style", "nav", "footer"]):
                element.decompose()
            
            content = soup.get_text(separator=' ', strip=True)
            
            return {
                'success': True,
                'method': 'selenium',
                'title': title,
                'content': content,
                'content_length': len(content),
                'sufficient_content': len(content) >= min_content_length
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
        """Scraping automático con fallback"""
        logger.info(f"Scraping automático de {url}")
        
        # Primero requests
        requests_result = self.scrape_with_requests(url, min_content_length)
        
        if requests_result['success'] and requests_result['sufficient_content']:
            logger.info(f"Requests exitoso: {requests_result['content_length']} chars")
            return requests_result
        
        # Fallback a Selenium
        if self.check_selenium_status():
            logger.info("Intentando con Selenium...")
            selenium_result = self.scrape_with_selenium(url, min_content_length)
            
            if selenium_result['success'] and selenium_result['sufficient_content']:
                logger.info(f"Selenium exitoso: {selenium_result['content_length']} chars")
                return selenium_result
        
        # Devolver mejor resultado
        return requests_result if requests_result.get('content_length', 0) > 0 else {
            'success': False,
            'method': 'auto',
            'error': 'Todos los métodos fallaron',
            'content_length': 0,
            'sufficient_content': False
        }

# Instancia global
scraping_service = WebScrapingService()

# ===== FUNCIONES DE PERSISTENCIA CORREGIDAS =====

def load_web_sources() -> Dict[str, WebSource]:
    """Cargar fuentes web desde archivo JSON con migración automática legacy"""
    if not WEB_SOURCES_FILE.exists():
        return {}
    
    try:
        with open(WEB_SOURCES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sources = {}
        migrated_any = False
        
        for source_id, source_data in data.items():
            try:
                # ✅ DETECCIÓN Y MIGRACIÓN AUTOMÁTICA
                if 'type' not in source_data or 'config' not in source_data:
                    logger.info(f"Migrando fuente legacy: {source_id}")
                    source_data = migrate_legacy_source_data(source_data)
                    migrated_any = True
                
                # ✅ CREAR WebSource USANDO MODELO CORRECTO
                web_source = WebSource.from_dict(source_data)
                sources[source_id] = web_source
                logger.debug(f"Fuente cargada: {web_source.name}")
                
            except Exception as e:
                logger.error(f"Error cargando fuente {source_id}: {e}")
                continue
        
        # ✅ GUARDAR AUTOMÁTICAMENTE SI SE MIGRÓ ALGO
        if migrated_any:
            logger.info("Guardando fuentes migradas automáticamente")
            save_web_sources(sources)
        
        logger.info(f"Cargadas {len(sources)} fuentes web")
        return sources
        
    except Exception as e:
        logger.error(f"Error cargando fuentes web: {e}")
        return {}

def migrate_legacy_source_data(legacy_data: Dict[str, Any]) -> Dict[str, Any]:
    """✅ FUNCIÓN DE MIGRACIÓN MEJORADA - Convierte datos legacy al modelo WebSource"""
    
    # Obtener base_urls - puede estar en config o en nivel raíz
    base_urls = []
    if 'config' in legacy_data and 'base_urls' in legacy_data['config']:
        base_urls = legacy_data['config']['base_urls']
    elif 'base_urls' in legacy_data:
        base_urls = legacy_data['base_urls']
    elif 'url' in legacy_data:
        base_urls = [legacy_data['url']]
    
    # Migrar metadatos del nivel raíz si existen
    metadata = legacy_data.get('metadata', {})
    
    # ✅ ESTRUCTURA CORRECTA SEGÚN MODELO WebSource
    migrated = {
        'id': legacy_data.get('id', ''),
        'name': legacy_data.get('name', ''),
        'type': 'web',
        'status': legacy_data.get('status', 'active'),
        'config': {
            # URLs
            'base_urls': base_urls,
            
            # Configuraciones de scraping (migrar desde nivel raíz)
            'max_depth': legacy_data.get('max_depth', 2),
            'delay_seconds': legacy_data.get('delay_seconds', 1.0),
            'follow_links': legacy_data.get('follow_links', True),
            'respect_robots_txt': legacy_data.get('respect_robots_txt', True),
            'min_content_length': legacy_data.get('min_content_length', 100),
            'user_agent': legacy_data.get('user_agent', 'Mozilla/5.0 (Prototipo_chatbot TFM UJI)'),
            
            # Configuraciones avanzadas del Enhanced Scraper
            'scraping_method': metadata.get('scraping_method', 'requests'),
            'max_pages': metadata.get('max_pages', 50),
            'crawl_frequency': metadata.get('crawl_frequency', 'manual'),
            
            # Selectores por defecto
            'content_selectors': ['main', 'article', '.content'],
            'title_selectors': ['h1', 'title'],
            'exclude_selectors': ['nav', 'footer', '.sidebar'],
            'exclude_file_extensions': ['.pdf', '.doc', '.jpg'],
            'include_patterns': [],
            'exclude_patterns': ['/admin', '/login'],
            'custom_headers': {},
            'use_javascript': metadata.get('scraping_method') == 'selenium'
        },
        'created_at': legacy_data.get('created_at', datetime.now().isoformat()),
        'last_sync': legacy_data.get('last_sync'),
        'metadata': metadata
    }
    
    logger.debug(f"Migración completada para: {migrated['name']}")
    return migrated

def save_web_sources(sources: Dict[str, WebSource]):
    """Guardar fuentes web usando modelo WebSource"""
    try:
        # Convertir todas las fuentes a diccionarios usando to_dict()
        data = {}
        for source_id, web_source in sources.items():
            data[source_id] = web_source.to_dict()
        
        with open(WEB_SOURCES_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Guardadas {len(sources)} fuentes web")
        
    except Exception as e:
        logger.error(f"Error guardando fuentes web: {e}")

def save_scraped_content(source_id: str, url: str, result: Dict[str, Any]) -> Optional[str]:
    """Guardar contenido usando modelo ScrapedPage"""
    if not result.get('success') or not result.get('sufficient_content'):
        return None
    
    try:
        # Crear ScrapedPage usando el modelo
        scraped_page = ScrapedPage.from_response(
            url=url,
            title=result.get('title', ''),
            content=result.get('content', ''),
            links=[],  # TODO: Extraer links en scraping
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

# ===== ENDPOINTS DE LA API CORREGIDOS =====

@web_sources_api.route('', methods=['GET'])
def list_web_sources():
    """Listar fuentes web con manejo de ambas estructuras"""
    try:
        sources = load_web_sources()
        
        # Convertir a formato de respuesta con compatibilidad legacy
        sources_list = []
        for source_id, web_source in sources.items():
            source_dict = web_source.to_dict()
            
            # ✅ AÑADIR CAMPOS PARA COMPATIBILIDAD CON FRONTEND
            source_dict['scraping_method'] = web_source.config.get('scraping_method', 'requests')
            source_dict['pages_found'] = web_source.metadata.get('total_pages_processed', 0)
            source_dict['success_rate'] = web_source.metadata.get('success_rate', 0.0)
            source_dict['last_activity'] = web_source.last_sync
            source_dict['is_active'] = web_source.metadata.get('scraping_active', False)
            
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
    """✅ ENDPOINT CORREGIDO - Recibe estructura WebSource del frontend"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        logger.info(f"Datos recibidos del frontend: {data}")
        
        # ✅ VALIDACIONES MEJORADAS
        required_fields = ['name', 'type', 'base_urls', 'config']
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
        
        # ✅ CREAR WebSource DIRECTAMENTE DESDE DATOS ESTRUCTURADOS
        try:
            web_source = WebSource.from_dict(data)
            logger.info(f"WebSource creado correctamente: {web_source.name}")
        except Exception as e:
            logger.error(f"Error creando WebSource desde datos: {e}")
            return jsonify({'success': False, 'error': f'Error en estructura de datos: {e}'}), 400
        
        # ✅ ESTABLECER VALORES POR DEFECTO SI NO ESTÁN PRESENTES
        web_source.status = DataSourceStatus.ACTIVE
        
        if not web_source.created_at:
            web_source.created_at = datetime.now()
        
        # ✅ CARGAR, AÑADIR Y GUARDAR
        sources = load_web_sources()
        sources[web_source.id] = web_source
        save_web_sources(sources)
        
        logger.info(f"Fuente web creada y guardada: {web_source.name} (ID: {web_source.id})")
        
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
    """Test de URL"""
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'success': False, 'error': 'URL requerida'}), 400
        
        url = data['url']
        method = data.get('method', 'auto')
        min_content_length = data.get('min_content_length', 50)
        
        logger.info(f"Testing: {url} con {method}")
        
        # Test conectividad
        connectivity = scraping_service.test_url_connectivity(url)
        
        # Test scraping
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
            'recommendation': scraping_result.get('method', 'auto')
        })
        
    except Exception as e:
        logger.error(f"Error test URL: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@web_sources_api.route('/scraping/start/<source_id>', methods=['POST'])
def start_scraping(source_id: str):
    """Iniciar scraping usando modelo WebSource"""
    try:
        with task_lock:
            if source_id in active_tasks:
                return jsonify({
                    'success': False,
                    'error': 'Scraping ya en progreso'
                }), 409
        
        # Cargar fuente usando modelo
        sources = load_web_sources()
        if source_id not in sources:
            return jsonify({'success': False, 'error': 'Fuente no encontrada'}), 404
        
        web_source = sources[source_id]
        
        task_id = f"{source_id}_{int(time.time())}"
        
        with task_lock:
            active_tasks[source_id] = {
                'task_id': task_id,
                'status': 'starting',
                'started_at': datetime.now().isoformat(),
                'total_urls': len(web_source.config.get('base_urls', [])),
                'processed_urls': 0,
                'successful_pages': 0,
                'failed_pages': 0
            }
        
        def run_scraping():
            try:
                with task_lock:
                    active_tasks[source_id]['status'] = 'running'
                
                # Actualizar estado de la fuente
                web_source.status = DataSourceStatus.PROCESSING
                web_source.last_sync = datetime.now()
                web_source.metadata['scraping_active'] = True
                
                base_urls = web_source.config.get('base_urls', [])
                min_content_length = web_source.config.get('min_content_length', 100)
                delay_seconds = web_source.config.get('delay_seconds', 1.0)
                use_javascript = web_source.config.get('use_javascript', False)
                
                for url in base_urls:
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
                        
                        time.sleep(delay_seconds)
                        
                    except Exception as e:
                        logger.error(f"Error scrapeando {url}: {e}")
                        with task_lock:
                            active_tasks[source_id]['failed_pages'] += 1
                
                # Finalizar y actualizar metadata
                with task_lock:
                    task_info = active_tasks[source_id]
                    task_info['status'] = 'completed'
                    task_info['completed_at'] = datetime.now().isoformat()
                
                # Actualizar metadata de la fuente
                web_source.metadata.update({
                    'total_pages_processed': task_info['processed_urls'],
                    'failed_pages': task_info['failed_pages'],
                    'success_rate': (task_info['successful_pages'] / max(task_info['processed_urls'], 1)) * 100,
                    'last_scraping_duration': time.time() - time.mktime(datetime.fromisoformat(task_info['started_at']).timetuple()),
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
                sources[source_id] = web_source
                save_web_sources(sources)
        
        thread = threading.Thread(target=run_scraping, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'Scraping iniciado'
        })
        
    except Exception as e:
        logger.error(f"Error iniciando scraping: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@web_sources_api.route('/scraping/status/<source_id>', methods=['GET'])
def get_scraping_status(source_id: str):
    """Estado del scraping"""
    try:
        with task_lock:
            task_info = active_tasks.get(source_id)
        
        if not task_info:
            return jsonify({
                'success': True,
                'status': 'not_running'
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
    """Eliminar fuente"""
    try:
        sources = load_web_sources()
        if source_id in sources:
            del sources[source_id]
            save_web_sources(sources)
        
        return jsonify({
            'success': True,
            'message': 'Fuente eliminada'
        })
        
    except Exception as e:
        logger.error(f"Error eliminando fuente: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@web_sources_api.route('/system/status', methods=['GET'])
def get_system_status():
    """Estado del sistema"""
    try:
        sources = load_web_sources()
        selenium_status = scraping_service.check_selenium_status()
        
        return jsonify({
            'success': True,
            'system_info': {
                'selenium_available': selenium_status,
                'selenium_error': SELENIUM_ERROR if not selenium_status else None,
                'active_tasks': len(active_tasks),
                'total_sources': len(sources)
            }
        })
    except Exception as e:
        logger.error(f"Error estado sistema: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500