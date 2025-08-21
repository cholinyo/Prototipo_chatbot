"""
API Routes para gestión de fuentes web - SISTEMA COMPLETO INTEGRADO
TFM Vicente Caruncho - Sistemas Inteligentes
Universitat Jaume I - Curso 2024-2025

INTEGRACIÓN COMPLETA CON MODELOS DE DATOS:
- Usa las clases WebSource y create_web_source del módulo data_sources
- Mantiene compatibilidad con fallbacks para desarrollo
- Sistema de persistencia en memoria con serialización JSON
- Simulación realista de procesos de scraping

FUNCIONALIDAD COMPLETA:
- Gestión de métodos de scraping disponibles (requests, selenium, playwright)
- CRUD completo de fuentes web con modelo de datos robusto
- Control de procesos de scraping (iniciar, detener, monitorear)
- Estadísticas y monitoreo en tiempo real
- Sistema de simulación para testing y desarrollo
- Manejo robusto de errores y fallbacks inteligentes

RUTAS API DISPONIBLES:
- GET  /api/scraping-methods     - Listar métodos de scraping disponibles
- GET  /api/web-sources          - Listar todas las fuentes configuradas
- POST /api/web-sources          - Crear nueva fuente web
- DELETE /api/web-sources/<id>   - Eliminar fuente específica
- POST /api/scraping/start/<id>  - Iniciar scraping individual
- POST /api/scraping/bulk-start  - Iniciar scraping masivo
- GET  /api/scraping/status      - Estado de procesos activos
- POST /api/scraping/cancel/<id> - Cancelar proceso específico
- GET  /api/stats                - Estadísticas generales del sistema
- GET  /api/debug/routes         - Debug: mostrar rutas registradas
"""

# =============================================================================
# IMPORTACIONES Y CONFIGURACIÓN INICIAL
# =============================================================================

from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
import uuid
import threading
import time
import random
from typing import Dict, Any, List, Optional

# Importaciones locales del core
from app.core.logger import get_logger

# Importaciones de modelos de datos con fallback
try:
    from app.models.data_sources import (
        WebSource, create_web_source, DataSourceType, 
        DataSourceStatus, IngestionStats
    )
    DATA_SOURCES_AVAILABLE = True
    print("✅ Modelos de datos importados correctamente")
except ImportError as e:
    DATA_SOURCES_AVAILABLE = False
    print(f"⚠️ Modelos de datos no disponibles: {e}")

# Importaciones de servicios avanzados con fallback
try:
    from app.services.enhanced_web_scraper import (
        enhanced_scraper_service, ScrapingMethod, CrawlFrequency, ScrapingConfig
    )
    ENHANCED_SCRAPER_AVAILABLE = True
    print("✅ Servicio de scraping avanzado disponible")
except ImportError:
    ENHANCED_SCRAPER_AVAILABLE = False
    print("⚠️ Servicio de scraping avanzado no disponible, usando fallback")

# =============================================================================
# CONFIGURACIÓN DEL BLUEPRINT Y STORAGE
# =============================================================================

# Crear blueprint principal - SIN url_prefix para evitar conflictos
web_sources_api = Blueprint('web_sources_api', __name__)
logger = get_logger("web_sources_api")

# Sistema de almacenamiento en memoria para desarrollo
# En producción esto se reemplazaría por una base de datos
web_sources_store: Dict[str, WebSource] = {}  # ID -> WebSource object
active_scraping_tasks: Dict[str, Dict[str, Any]] = {}  # ID -> task info
scraping_history: List[Dict[str, Any]] = []  # Historial de operaciones

# Configuración global del sistema
SYSTEM_CONFIG = {
    'max_concurrent_tasks': 5,
    'default_timeout': 300,  # 5 minutos
    'max_sources_per_user': 50,
    'simulation_mode': True  # Para desarrollo
}

# =============================================================================
# FUNCIONES DE UTILIDAD - MÉTODOS DE SCRAPING
# =============================================================================

def get_available_scraping_methods() -> List[Dict[str, Any]]:
    """
    Obtener métodos de scraping disponibles en el sistema
    
    Verifica la disponibilidad real de cada método y retorna información
    detallada incluyendo ventajas, limitaciones y casos de uso.
    
    Returns:
        List[Dict]: Lista de métodos con metadatos completos
    """
    if ENHANCED_SCRAPER_AVAILABLE:
        # Usar servicio avanzado si está disponible
        try:
            return enhanced_scraper_service.get_available_methods()
        except Exception as e:
            logger.warning(f"Error usando servicio avanzado: {e}")
    
    # Métodos de fallback con detección automática de disponibilidad
    methods = []
    
    # Método 1: Requests + BeautifulSoup (siempre disponible)
    try:
        import requests
        import bs4
        methods.append({
            'id': 'requests',
            'name': 'Requests + BeautifulSoup',
            'description': 'Método rápido y eficiente para sitios web estáticos',
            'available': True,
            'performance': 'Muy alta',
            'resource_usage': 'Bajo',
            'pros': [
                'Muy rápido y eficiente',
                'Bajo consumo de recursos',
                'Altamente estable',
                'Compatible con la mayoría de sitios'
            ],
            'cons': [
                'No ejecuta JavaScript',
                'Limitado con aplicaciones SPA',
                'No maneja interacciones dinámicas'
            ],
            'use_cases': [
                'Portales institucionales',
                'Sitios web estáticos',
                'Blogs y noticias',
                'Páginas de documentación'
            ],
            'requirements': ['requests', 'beautifulsoup4']
        })
    except ImportError:
        methods.append({
            'id': 'requests',
            'name': 'Requests + BeautifulSoup (No disponible)',
            'description': 'Requiere instalación de dependencias',
            'available': False,
            'error': 'Módulos requests o beautifulsoup4 no instalados'
        })
    
    # Método 2: Selenium WebDriver
    try:
        import selenium
        from selenium import webdriver
        methods.append({
            'id': 'selenium',
            'name': 'Selenium WebDriver',
            'description': 'Navegador automatizado para sitios con JavaScript',
            'available': True,
            'performance': 'Media',
            'resource_usage': 'Alto',
            'pros': [
                'Ejecuta JavaScript completamente',
                'Interacciones complejas posibles',
                'Soporte para múltiples navegadores',
                'Maduro y bien documentado'
            ],
            'cons': [
                'Mayor consumo de recursos',
                'Más lento que requests',
                'Requiere ChromeDriver/GeckoDriver',
                'Puede ser detectado por anti-bot'
            ],
            'use_cases': [
                'Aplicaciones de página única (SPA)',
                'Sitios con autenticación',
                'JavaScript intensivo',
                'Formularios dinámicos'
            ],
            'requirements': ['selenium', 'chromedriver']
        })
    except ImportError:
        methods.append({
            'id': 'selenium',
            'name': 'Selenium WebDriver (No instalado)',
            'description': 'Navegador automatizado - requiere instalación',
            'available': False,
            'installation': 'pip install selenium && playwright install chromium',
            'pros': ['Ejecuta JavaScript', 'Interacciones complejas'],
            'cons': ['Requiere instalación adicional'],
            'use_cases': ['SPAs', 'Sitios dinámicos']
        })
    
    # Método 3: Playwright (más moderno)
    try:
        import playwright
        methods.append({
            'id': 'playwright',
            'name': 'Playwright (Moderno)',
            'description': 'Framework moderno de automatización web',
            'available': True,
            'performance': 'Alta',
            'resource_usage': 'Medio',
            'pros': [
                'Muy rápido y eficiente',
                'Soporte multi-navegador nativo',
                'API moderna y limpia',
                'Mejor manejo de recursos'
            ],
            'cons': [
                'Más complejo de configurar',
                'Menos documentación comunidad',
                'Relativamente nuevo'
            ],
            'use_cases': [
                'Scraping masivo',
                'Testing de múltiples navegadores',
                'Aplicaciones de alto rendimiento'
            ],
            'requirements': ['playwright']
        })
    except ImportError:
        methods.append({
            'id': 'playwright',
            'name': 'Playwright (No instalado)',
            'description': 'Framework moderno - requiere instalación',
            'available': False,
            'installation': 'pip install playwright && playwright install',
            'pros': ['Muy rápido', 'Multi-navegador', 'Moderno'],
            'cons': ['Requiere instalación'],
            'use_cases': ['Scraping avanzado', 'Alto rendimiento']
        })
    
    return methods

# =============================================================================
# FUNCIONES DE UTILIDAD - GESTIÓN DE DATOS
# =============================================================================

def create_websource_from_request(data: Dict[str, Any]) -> WebSource:
    """
    Crear objeto WebSource desde datos de request HTTP
    
    Utiliza la función factory del módulo data_sources para crear
    un objeto WebSource válido con todas las validaciones.
    
    Args:
        data: Diccionario con datos del request JSON
        
    Returns:
        WebSource: Objeto WebSource validado
        
    Raises:
        ValueError: Si los datos son inválidos
    """
    if not DATA_SOURCES_AVAILABLE:
        raise ImportError("Modelos de datos no disponibles")
    
    # Validar campos requeridos
    required_fields = ['name', 'url']
    for field in required_fields:
        if field not in data or not data[field].strip():
            raise ValueError(f"Campo requerido: {field}")
    
    # Preparar URLs base
    url = data['url'].strip()
    if not url.startswith(('http://', 'https://')):
        raise ValueError("URL debe comenzar con http:// o https://")
    
    base_urls = [url]
    
    # Usar factory function del módulo data_sources
    return create_web_source(
        name=data['name'].strip(),
        base_urls=base_urls,
        max_depth=min(int(data.get('max_depth', 2)), 5),
        delay_seconds=max(float(data.get('delay_seconds', 1.0)), 0.5),
        follow_links=data.get('follow_links', True),
        respect_robots_txt=data.get('respect_robots_txt', True),
        user_agent=data.get('user_agent', 'Prototipo_chatbot TFM UJI'),
        min_content_length=max(int(data.get('min_content_length', 100)), 50),
        content_filters=data.get('content_filters', []),
        custom_headers=data.get('custom_headers', {}),
        use_javascript=data.get('method') == 'selenium',
        metadata={
            'created_by': 'web_api',
            'scraping_method': data.get('method', 'requests'),
            'max_pages': min(int(data.get('max_pages', 50)), 200),
            'crawl_frequency': data.get('crawl_frequency', 'manual')
        }
    )

def websource_to_api_format(source: WebSource) -> Dict[str, Any]:
    """
    Convertir WebSource a formato API con información enriquecida
    
    Args:
        source: Objeto WebSource
        
    Returns:
        Dict: Datos formateados para respuesta API
    """
    # Usar método to_dict() del modelo
    base_data = source.to_dict()
    
    # Enriquecer con información de estado
    source_id = source.id
    enriched_data = {
        **base_data,
        'status': 'active' if source_id in active_scraping_tasks else 'inactive',
        'last_activity': base_data.get('last_sync') or 'Nunca',
        'pages_found': source.metadata.get('pages_found', 0),
        'success_rate': source.metadata.get('success_rate', 0.0),
        'last_error': source.metadata.get('last_error'),
        'scraping_method': source.metadata.get('scraping_method', 'requests'),
        'is_active': source_id in active_scraping_tasks
    }
    
    return enriched_data

# =============================================================================
# RUTAS API - MÉTODOS DE SCRAPING
# =============================================================================

@web_sources_api.route('/scraping-methods', methods=['GET'])
def get_scraping_methods():
    """
    GET /api/scraping-methods
    
    Obtener lista completa de métodos de scraping disponibles en el sistema.
    Incluye detección automática de dependencias y información detallada
    de cada método (ventajas, limitaciones, casos de uso).
    
    Returns:
        JSON: {
            "success": bool,
            "methods": [
                {
                    "id": str,
                    "name": str,
                    "description": str,
                    "available": bool,
                    "pros": [str],
                    "cons": [str],
                    "use_cases": [str]
                }
            ],
            "total": int
        }
    """
    try:
        methods = get_available_scraping_methods()
        
        logger.info(f"Métodos de scraping solicitados: {len(methods)} disponibles")
        
        return jsonify({
            'success': True,
            'methods': methods,
            'total': len(methods),
            'available_count': len([m for m in methods if m.get('available', False)]),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo métodos de scraping: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'methods': []
        }), 500

# =============================================================================
# RUTAS API - GESTIÓN DE FUENTES WEB
# =============================================================================

@web_sources_api.route('/web-sources', methods=['GET'])
def get_web_sources():
    """
    GET /api/web-sources
    
    Obtener lista completa de fuentes web configuradas en el sistema.
    Incluye información de estado, estadísticas y metadatos enriquecidos.
    
    Query parameters:
        - status: Filtrar por estado (active, inactive, all)
        - limit: Número máximo de resultados
        
    Returns:
        JSON: {
            "success": bool,
            "sources": [WebSource],
            "total": int,
            "active_count": int,
            "timestamp": str
        }
    """
    try:
        # Obtener parámetros de query opcionales
        status_filter = request.args.get('status', 'all')
        limit = int(request.args.get('limit', 100))
        
        # Convertir fuentes almacenadas a formato API
        sources = []
        for source in web_sources_store.values():
            source_data = websource_to_api_format(source)
            
            # Aplicar filtro de estado si se especifica
            if status_filter != 'all':
                if status_filter == 'active' and source_data['status'] != 'active':
                    continue
                elif status_filter == 'inactive' and source_data['status'] == 'active':
                    continue
            
            sources.append(source_data)
        
        # Aplicar límite
        sources = sources[:limit]
        
        # Calcular estadísticas
        active_count = len([s for s in sources if s['status'] == 'active'])
        
        logger.info(f"Listando {len(sources)} fuentes web (filtro: {status_filter})")
        
        return jsonify({
            'success': True,
            'sources': sources,
            'total': len(sources),
            'active_count': active_count,
            'total_in_system': len(web_sources_store),
            'filter_applied': status_filter,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo fuentes web: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'sources': []
        }), 500

@web_sources_api.route('/web-sources', methods=['POST'])
def add_web_source():
    """
    POST /api/web-sources
    
    Crear nueva fuente web en el sistema con validación completa.
    Utiliza el modelo WebSource para asegurar consistencia de datos.
    
    Expected JSON payload:
    {
        "name": "Nombre descriptivo de la fuente",
        "url": "https://ejemplo.com",
        "method": "requests|selenium|playwright",
        "max_depth": 2,
        "max_pages": 100,
        "crawl_frequency": "manual|daily|weekly|monthly",
        "content_filters": ["filtro1", "filtro2"],
        "delay_seconds": 1.0,
        "follow_links": true,
        "respect_robots_txt": true
    }
    
    Returns:
        JSON: {
            "success": bool,
            "source_id": str,
            "message": str,
            "source": WebSource
        }
    """
    try:
        data = request.get_json()
        
        # Validación básica de entrada
        if not data:
            return jsonify({
                'success': False,
                'error': 'No se proporcionaron datos JSON válidos'
            }), 400
        
        # Verificar límite de fuentes
        if len(web_sources_store) >= SYSTEM_CONFIG['max_sources_per_user']:
            return jsonify({
                'success': False,
                'error': f'Límite máximo de fuentes alcanzado: {SYSTEM_CONFIG["max_sources_per_user"]}'
            }), 400
        
        # Validar método de scraping si se especifica
        if 'method' in data:
            available_methods = get_available_scraping_methods()
            available_ids = [m['id'] for m in available_methods if m.get('available', False)]
            
            if data['method'] not in available_ids:
                return jsonify({
                    'success': False,
                    'error': f'Método no disponible: {data["method"]}. Métodos válidos: {", ".join(available_ids)}'
                }), 400
        
        # Crear objeto WebSource usando factory function
        try:
            web_source = create_websource_from_request(data)
        except ValueError as ve:
            return jsonify({
                'success': False,
                'error': f'Datos inválidos: {str(ve)}'
            }), 400
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'Sistema de modelos de datos no disponible'
            }), 500
        
        # Verificar que la URL no exista ya
        for existing_source in web_sources_store.values():
            if data['url'] in existing_source.base_urls:
                return jsonify({
                    'success': False,
                    'error': f'La URL ya está configurada en la fuente: {existing_source.name}'
                }), 409
        
        # Almacenar en el store
        web_sources_store[web_source.id] = web_source
        
        # Registrar en historial
        scraping_history.append({
            'action': 'source_created',
            'source_id': web_source.id,
            'source_name': web_source.name,
            'timestamp': datetime.now().isoformat(),
            'method': data.get('method', 'requests')
        })
        
        logger.info(f"Nueva fuente web creada: {web_source.name} ({web_source.id})")
        
        return jsonify({
            'success': True,
            'source_id': web_source.id,
            'message': f'Fuente "{web_source.name}" creada correctamente',
            'source': websource_to_api_format(web_source)
        }), 201
        
    except Exception as e:
        logger.error(f"Error creando fuente web: {e}")
        return jsonify({
            'success': False,
            'error': f'Error interno: {str(e)}'
        }), 500

@web_sources_api.route('/web-sources/<source_id>', methods=['GET'])
def get_web_source_details(source_id):
    """
    GET /api/web-sources/<id>
    
    Obtener detalles completos de una fuente web específica.
    Incluye estadísticas, historial y configuración completa.
    
    Returns:
        JSON: {
            "success": bool,
            "source": WebSource,
            "statistics": IngestionStats,
            "history": [dict]
        }
    """
    try:
        if source_id not in web_sources_store:
            return jsonify({
                'success': False,
                'error': 'Fuente no encontrada'
            }), 404
        
        source = web_sources_store[source_id]
        source_data = websource_to_api_format(source)
        
        # Obtener historial relacionado
        source_history = [
            entry for entry in scraping_history 
            if entry.get('source_id') == source_id
        ]
        
        # Estadísticas simuladas (en producción vendrían de la base de datos)
        stats = {
            'total_pages_scraped': source.metadata.get('pages_found', 0),
            'success_rate': source.metadata.get('success_rate', 0.0),
            'last_scraping_duration': source.metadata.get('last_duration', 0),
            'average_page_size': source.metadata.get('avg_page_size', 0),
            'total_content_size': source.metadata.get('total_size', 0)
        }
        
        return jsonify({
            'success': True,
            'source': source_data,
            'statistics': stats,
            'history': source_history[-10:],  # Últimas 10 entradas
            'is_active': source_id in active_scraping_tasks
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo detalles de fuente {source_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@web_sources_api.route('/web-sources/<source_id>', methods=['DELETE'])
def delete_web_source(source_id):
    """
    DELETE /api/web-sources/<id>
    
    Eliminar fuente web del sistema. Si hay un proceso de scraping
    activo, se cancela automáticamente antes de eliminar.
    
    Returns:
        JSON: {
            "success": bool,
            "message": str
        }
    """
    try:
        if source_id not in web_sources_store:
            return jsonify({
                'success': False,
                'error': 'Fuente no encontrada'
            }), 404
        
        source = web_sources_store[source_id]
        source_name = source.name
        
        # Cancelar scraping activo si existe
        if source_id in active_scraping_tasks:
            del active_scraping_tasks[source_id]
            logger.info(f"Scraping cancelado automáticamente para fuente {source_id}")
        
        # Eliminar del store
        del web_sources_store[source_id]
        
        # Registrar en historial
        scraping_history.append({
            'action': 'source_deleted',
            'source_id': source_id,
            'source_name': source_name,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Fuente web eliminada: {source_name} ({source_id})")
        
        return jsonify({
            'success': True,
            'message': f'Fuente "{source_name}" eliminada correctamente'
        })
        
    except Exception as e:
        logger.error(f"Error eliminando fuente {source_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# =============================================================================
# RUTAS API - CONTROL DE SCRAPING
# =============================================================================

@web_sources_api.route('/scraping/start/<source_id>', methods=['POST'])
def start_individual_scraping(source_id):
    """
    POST /api/scraping/start/<id>
    
    Iniciar proceso de scraping para una fuente específica.
    En modo de desarrollo ejecuta simulación realista.
    
    Returns:
        JSON: {
            "success": bool,
            "message": str,
            "task_id": str,
            "estimated_duration": str
        }
    """
    try:
        if source_id not in web_sources_store:
            return jsonify({
                'success': False,
                'error': 'Fuente no encontrada'
            }), 404
        
        if source_id in active_scraping_tasks:
            return jsonify({
                'success': False,
                'error': 'Scraping ya en progreso para esta fuente'
            }), 409
        
        # Verificar límite de tareas concurrentes
        if len(active_scraping_tasks) >= SYSTEM_CONFIG['max_concurrent_tasks']:
            return jsonify({
                'success': False,
                'error': f'Límite de tareas concurrentes alcanzado: {SYSTEM_CONFIG["max_concurrent_tasks"]}'
            }), 429
        
        source = web_sources_store[source_id]
        timestamp = datetime.now()
        
        # Crear información de tarea activa
        task_info = {
            'started_at': timestamp.isoformat(),
            'status': 'running',
            'progress': 0,
            'pages_processed': 0,
            'current_url': source.base_urls[0] if source.base_urls else '',
            'method': source.metadata.get('scraping_method', 'requests'),
            'estimated_pages': source.metadata.get('max_pages', 50)
        }
        
        active_scraping_tasks[source_id] = task_info
        
        # Actualizar última sincronización de la fuente
        source.last_sync = timestamp
        
        # Registrar en historial
        scraping_history.append({
            'action': 'scraping_started',
            'source_id': source_id,
            'source_name': source.name,
            'timestamp': timestamp.isoformat(),
            'method': task_info['method']
        })
        
        logger.info(f"Scraping iniciado para: {source.name} ({source_id})")
        
        # Iniciar simulación en background (en producción sería scraping real)
        if SYSTEM_CONFIG['simulation_mode']:
            start_scraping_simulation(source_id)
        
        return jsonify({
            'success': True,
            'message': f'Scraping iniciado para "{source.name}"',
            'task_id': source_id,
            'estimated_duration': '2-5 minutos',
            'task_info': task_info
        })
        
    except Exception as e:
        logger.error(f"Error iniciando scraping para {source_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@web_sources_api.route('/scraping/bulk-start', methods=['POST'])
def start_bulk_scraping():
    """
    POST /api/scraping/bulk-start
    
    Iniciar scraping masivo para todas las fuentes configuradas.
    Respeta el límite de tareas concurrentes del sistema.
    
    Returns:
        JSON: {
            "success": bool,
            "message": str,
            "started_tasks": [dict],
            "skipped_tasks": [dict],
            "total_started": int,
            "total_skipped": int
        }
    """
    try:
        if not web_sources_store:
            return jsonify({
                'success': False,
                'error': 'No hay fuentes configuradas en el sistema'
            }), 400
        
        started_tasks = []
        skipped_tasks = []
        max_concurrent = SYSTEM_CONFIG['max_concurrent_tasks']
        
        for source_id, source in web_sources_store.items():
            # Verificar límite de concurrencia
            if len(active_scraping_tasks) >= max_concurrent:
                skipped_tasks.append({
                    'id': source_id,
                    'name': source.name,
                    'reason': 'Límite de concurrencia alcanzado'
                })
                continue
            
            # Verificar si ya está activo
            if source_id in active_scraping_tasks:
                skipped_tasks.append({
                    'id': source_id,
                    'name': source.name,
                    'reason': 'Ya en progreso'
                })
                continue
            
            # Intentar iniciar scraping
            try:
                # Reutilizar función individual para consistencia
                response = start_individual_scraping(source_id)
                if response[1] == 200:  # Status code OK
                    started_tasks.append({
                        'id': source_id,
                        'name': source.name,
                        'method': source.metadata.get('scraping_method', 'requests')
                    })
            except Exception as e:
                skipped_tasks.append({
                    'id': source_id,
                    'name': source.name,
                    'reason': f'Error: {str(e)}'
                })
        
        # Registrar operación masiva
        scraping_history.append({
            'action': 'bulk_scraping_started',
            'timestamp': datetime.now().isoformat(),
            'started_count': len(started_tasks),
            'skipped_count': len(skipped_tasks)
        })
        
        logger.info(f"Scraping masivo: {len(started_tasks)} iniciados, {len(skipped_tasks)} omitidos")
        
        return jsonify({
            'success': True,
            'message': f'Scraping masivo iniciado: {len(started_tasks)} fuentes procesándose',
            'started_tasks': started_tasks,
            'skipped_tasks': skipped_tasks,
            'total_started': len(started_tasks),
            'total_skipped': len(skipped_tasks),
            'concurrent_limit': max_concurrent
        })
        
    except Exception as e:
        logger.error(f"Error en scraping masivo: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@web_sources_api.route('/scraping/status', methods=['GET'])
def get_scraping_status():
    """
    GET /api/scraping/status
    
    Obtener estado actual de todos los procesos de scraping.
    Incluye información detallada de progreso y rendimiento.
    
    Returns:
        JSON: {
            "success": bool,
            "active_tasks": dict,
            "total_active": int,
            "total_sources": int,
            "system_stats": dict
        }
    """
    try:
        # Calcular estadísticas del sistema
        system_stats = {
            'total_sources': len(web_sources_store),
            'total_active_tasks': len(active_scraping_tasks),
            'max_concurrent_tasks': SYSTEM_CONFIG['max_concurrent_tasks'],
            'available_slots': SYSTEM_CONFIG['max_concurrent_tasks'] - len(active_scraping_tasks),
            'simulation_mode': SYSTEM_CONFIG['simulation_mode'],
            'uptime_seconds': int(time.time() - getattr(get_scraping_status, 'start_time', time.time()))
        }
        
        # Enriquecer información de tareas activas
        enriched_tasks = {}
        for source_id, task_info in active_scraping_tasks.items():
            if source_id in web_sources_store:
                source = web_sources_store[source_id]
                enriched_tasks[source_id] = {
                    **task_info,
                    'source_name': source.name,
                    'base_url': source.base_urls[0] if source.base_urls else '',
                    'elapsed_seconds': (
                        datetime.now() - datetime.fromisoformat(task_info['started_at'])
                    ).total_seconds()
                }
        
        return jsonify({
            'success': True,
            'active_tasks': enriched_tasks,
            'total_active': len(enriched_tasks),
            'system_stats': system_stats,
            'last_updated': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo estado de scraping: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@web_sources_api.route('/scraping/cancel/<source_id>', methods=['POST'])
def cancel_scraping(source_id):
    """
    POST /api/scraping/cancel/<id>
    
    Cancelar proceso de scraping activo para una fuente específica.
    
    Returns:
        JSON: {
            "success": bool,
            "message": str,
            "cancelled_task": dict
        }
    """
    try:
        if source_id not in active_scraping_tasks:
            return jsonify({
                'success': False,
                'error': 'No hay scraping activo para esta fuente'
            }), 404
        
        # Obtener información de la tarea antes de cancelar
        task_info = active_scraping_tasks[source_id].copy()
        
        # Eliminar de tareas activas
        del active_scraping_tasks[source_id]
        
        # Actualizar fuente con información de cancelación
        if source_id in web_sources_store:
            source = web_sources_store[source_id]
            source.metadata['last_error'] = 'Cancelado por usuario'
            source.last_sync = datetime.now()
        
        # Registrar en historial
        scraping_history.append({
            'action': 'scraping_cancelled',
            'source_id': source_id,
            'source_name': web_sources_store[source_id].name if source_id in web_sources_store else 'Unknown',
            'timestamp': datetime.now().isoformat(),
            'pages_processed': task_info.get('pages_processed', 0)
        })
        
        logger.info(f"Scraping cancelado para fuente {source_id}")
        
        return jsonify({
            'success': True,
            'message': 'Scraping cancelado correctamente',
            'cancelled_task': task_info
        })
        
    except Exception as e:
        logger.error(f"Error cancelando scraping para {source_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# =============================================================================
# RUTAS API - ESTADÍSTICAS Y MONITOREO
# =============================================================================

@web_sources_api.route('/stats', methods=['GET'])
def get_system_stats():
    """
    GET /api/stats
    
    Obtener estadísticas completas del sistema de web scraping.
    Incluye métricas de rendimiento, uso de recursos y tendencias.
    
    Returns:
        JSON: {
            "success": bool,
            "stats": dict
        }
    """
    try:
        # Calcular métricas agregadas
        total_pages = sum(
            source.metadata.get('pages_found', 0) 
            for source in web_sources_store.values()
        )
        
        success_rates = [
            source.metadata.get('success_rate', 0) 
            for source in web_sources_store.values() 
            if source.metadata.get('success_rate', 0) > 0
        ]
        
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
        
        # Estadísticas por método de scraping
        method_stats = {}
        for source in web_sources_store.values():
            method = source.metadata.get('scraping_method', 'requests')
            if method not in method_stats:
                method_stats[method] = {'count': 0, 'total_pages': 0}
            method_stats[method]['count'] += 1
            method_stats[method]['total_pages'] += source.metadata.get('pages_found', 0)
        
        # Actividad reciente
        recent_activity = [
            entry for entry in scraping_history[-20:]  # Últimas 20 entradas
            if (datetime.now() - datetime.fromisoformat(entry['timestamp'])).days < 7
        ]
        
        # Construir respuesta de estadísticas
        stats = {
            'overview': {
                'total_sources': len(web_sources_store),
                'total_pages_indexed': total_pages,
                'active_scraping_tasks': len(active_scraping_tasks),
                'average_success_rate': round(avg_success_rate, 2),
                'system_status': 'operational'
            },
            'performance': {
                'available_methods': len(get_available_scraping_methods()),
                'method_distribution': method_stats,
                'concurrent_capacity': SYSTEM_CONFIG['max_concurrent_tasks'],
                'utilization_rate': round(
                    (len(active_scraping_tasks) / SYSTEM_CONFIG['max_concurrent_tasks']) * 100, 1
                )
            },
            'activity': {
                'recent_operations': len(recent_activity),
                'total_operations': len(scraping_history),
                'last_activity': scraping_history[-1]['timestamp'] if scraping_history else None
            },
            'system': {
                'simulation_mode': SYSTEM_CONFIG['simulation_mode'],
                'data_sources_available': DATA_SOURCES_AVAILABLE,
                'enhanced_scraper_available': ENHANCED_SCRAPER_AVAILABLE,
                'last_updated': datetime.now().isoformat()
            }
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas del sistema: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# =============================================================================
# RUTAS API - DEBUG Y UTILIDADES
# =============================================================================

@web_sources_api.route('/debug/routes', methods=['GET'])
def debug_routes():
    """
    GET /api/debug/routes
    
    Endpoint de debug para mostrar todas las rutas registradas.
    Útil para diagnosticar problemas de registro de blueprints.
    
    Returns:
        HTML: Lista formateada de todas las rutas
    """
    try:
        output = []
        output.append("=== RUTAS REGISTRADAS EN LA APLICACIÓN ===")
        output.append(f"Timestamp: {datetime.now().isoformat()}")
        output.append("")
        
        # Obtener todas las rutas de la aplicación
        routes_by_blueprint = {}
        
        for rule in current_app.url_map.iter_rules():
            methods = ','.join(sorted(rule.methods - {'HEAD', 'OPTIONS'}))
            endpoint_parts = rule.endpoint.split('.')
            blueprint_name = endpoint_parts[0] if len(endpoint_parts) > 1 else 'main'
            
            if blueprint_name not in routes_by_blueprint:
                routes_by_blueprint[blueprint_name] = []
            
            routes_by_blueprint[blueprint_name].append({
                'rule': rule.rule,
                'endpoint': rule.endpoint,
                'methods': methods
            })
        
        # Formatear salida por blueprint
        for blueprint_name, routes in sorted(routes_by_blueprint.items()):
            output.append(f"Blueprint: {blueprint_name}")
            output.append("-" * 50)
            
            for route in sorted(routes, key=lambda x: x['rule']):
                output.append(f"  {route['endpoint']}: {route['rule']} [{route['methods']}]")
            
            output.append("")
        
        output.append(f"Total rutas: {sum(len(routes) for routes in routes_by_blueprint.values())}")
        output.append(f"Total blueprints: {len(routes_by_blueprint)}")
        
        return "<pre>" + "\n".join(output) + "</pre>"
        
    except Exception as e:
        return f"<pre>Error generando debug de rutas: {str(e)}</pre>", 500

# =============================================================================
# FUNCIONES DE UTILIDAD - SIMULACIÓN DE SCRAPING
# =============================================================================

def start_scraping_simulation(source_id: str):
    """
    Iniciar simulación realista de proceso de scraping
    
    En modo de desarrollo simula el scraping real con:
    - Progreso gradual y realista
    - Actualizaciones de estado periódicas
    - Resultados finales con métricas
    
    Args:
        source_id: ID de la fuente a simular
    """
    def run_simulation():
        """Función que ejecuta la simulación en thread separado"""
        try:
            if source_id not in active_scraping_tasks:
                return
            
            task = active_scraping_tasks[source_id]
            source = web_sources_store.get(source_id)
            
            if not source:
                return
            
            # Parámetros de simulación
            total_pages = random.randint(10, 100)
            processing_time = random.uniform(3, 8)  # 3-8 segundos total
            update_interval = processing_time / 10  # 10 actualizaciones
            
            logger.info(f"Iniciando simulación para {source_id}: {total_pages} páginas estimadas")
            
            # Simular progreso gradual
            for i in range(11):  # 0% a 100% en 10 pasos
                if source_id not in active_scraping_tasks:
                    break  # Cancelado
                
                progress = i * 10  # 0, 10, 20, ..., 100
                pages_processed = int((progress / 100) * total_pages)
                
                # Actualizar información de tarea
                active_scraping_tasks[source_id].update({
                    'progress': progress,
                    'pages_processed': pages_processed,
                    'status': 'running' if progress < 100 else 'completing'
                })
                
                time.sleep(update_interval)
            
            # Completar simulación si no fue cancelada
            if source_id in active_scraping_tasks:
                # Generar resultados finales realistas
                final_pages = random.randint(max(1, total_pages - 10), total_pages + 5)
                success_rate = random.uniform(85, 98)
                avg_page_size = random.randint(2000, 15000)
                
                # Actualizar metadatos de la fuente
                source.metadata.update({
                    'pages_found': final_pages,
                    'success_rate': round(success_rate, 1),
                    'last_duration': round(processing_time, 2),
                    'avg_page_size': avg_page_size,
                    'total_size': final_pages * avg_page_size,
                    'last_error': None
                })
                
                source.last_sync = datetime.now()
                
                # Registrar en historial
                scraping_history.append({
                    'action': 'scraping_completed',
                    'source_id': source_id,
                    'source_name': source.name,
                    'timestamp': datetime.now().isoformat(),
                    'pages_found': final_pages,
                    'duration': round(processing_time, 2),
                    'success_rate': round(success_rate, 1)
                })
                
                # Remover de tareas activas
                del active_scraping_tasks[source_id]
                
                logger.info(f"Simulación completada para {source_id}: {final_pages} páginas, {success_rate:.1f}% éxito")
        
        except Exception as e:
            logger.error(f"Error en simulación de scraping para {source_id}: {e}")
            
            # Manejar error en simulación
            if source_id in active_scraping_tasks:
                del active_scraping_tasks[source_id]
            
            if source_id in web_sources_store:
                web_sources_store[source_id].metadata['last_error'] = f"Error en simulación: {str(e)}"
    
    # Ejecutar simulación en thread separado
    thread = threading.Thread(target=run_simulation, daemon=True)
    thread.start()

# =============================================================================
# INICIALIZACIÓN Y CONFIGURACIÓN
# =============================================================================

# Marcar tiempo de inicio para estadísticas de uptime
get_scraping_status.start_time = time.time()

# Log de inicialización
logger.info(f"Blueprint web_sources_api inicializado correctamente")
logger.info(f"Configuración: {SYSTEM_CONFIG}")
logger.info(f"Modelos de datos disponibles: {DATA_SOURCES_AVAILABLE}")
logger.info(f"Servicio de scraping avanzado disponible: {ENHANCED_SCRAPER_AVAILABLE}")

# =============================================================================
# EXPORTACIONES
# =============================================================================

# Exportar blueprint para registro en la aplicación principal
__all__ = ['web_sources_api']