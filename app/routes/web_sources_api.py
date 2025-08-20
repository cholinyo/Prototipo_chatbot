"""
API Routes para gestión de fuentes web - CORREGIDA
TFM Vicente Caruncho - Sistemas Inteligentes
"""

from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
from typing import Dict, Any, List
import uuid

from app.core.logger import get_logger

# Blueprint para API de fuentes web
web_sources_api = Blueprint('web_sources_api', __name__, url_prefix='/api/web-sources')
logger = get_logger("web_sources_api")


def get_web_ingestion_service():
    """Obtener instancia del servicio con importación tardía"""
    from app.services.web_ingestion_service import web_ingestion_service
    return web_ingestion_service


@web_sources_api.route('', methods=['GET'])
def list_web_sources():
    """Listar todas las fuentes web"""
    try:
        web_ingestion_service = get_web_ingestion_service()
        sources = web_ingestion_service.list_sources()
        
        # Convertir a formato JSON con estadísticas
        sources_data = []
        for source in sources:
            source_dict = source.to_dict()
            
            # Agregar estadísticas
            try:
                stats = web_ingestion_service.get_source_stats(source.id)
                source_dict['stats'] = stats.to_dict()
            except Exception as e:
                logger.warning(f"Error obteniendo estadísticas para {source.id}: {e}")
                source_dict['stats'] = None
            
            sources_data.append(source_dict)
        
        return jsonify({
            'success': True,
            'sources': sources_data,
            'total': len(sources_data)
        })
        
    except Exception as e:
        logger.error(f"Error listando fuentes web: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@web_sources_api.route('', methods=['POST'])
def create_web_source_api():
    """Crear nueva fuente web"""
    try:
        web_ingestion_service = get_web_ingestion_service()
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No se proporcionaron datos'
            }), 400
        
        # Validar campos requeridos
        required_fields = ['name', 'base_urls']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Campo requerido: {field}'
                }), 400
        
        # Validar URLs
        base_urls = data['base_urls']
        if not isinstance(base_urls, list) or not base_urls:
            return jsonify({
                'success': False,
                'error': 'Se debe proporcionar al menos una URL base'
            }), 400
        
        # Crear fuente web usando el servicio
        try:
            web_source = web_ingestion_service.create_source(
                name=data['name'],
                base_urls=base_urls,
                max_depth=data.get('max_depth', 2),
                delay_seconds=data.get('delay_seconds', 1.0),
                user_agent=data.get('user_agent', 'Mozilla/5.0 (Prototipo_chatbot TFM UJI)'),
                follow_links=data.get('follow_links', True),
                respect_robots_txt=data.get('respect_robots_txt', True),
                content_selectors=data.get('content_selectors', ['main', 'article', '.content']),
                exclude_selectors=data.get('exclude_selectors', ['nav', 'footer', '.sidebar']),
                include_patterns=data.get('include_patterns', []),
                exclude_patterns=data.get('exclude_patterns', ['/admin', '/login']),
                min_content_length=data.get('min_content_length', 100),
                custom_headers=data.get('custom_headers', {}),
                use_javascript=data.get('use_javascript', False)
            )
            
            logger.info(f"Fuente web creada: {web_source.name} ({web_source.id})")
            
            return jsonify({
                'success': True,
                'source': web_source.to_dict(),
                'message': f'Fuente web creada exitosamente: {web_source.name}'
            }), 201
            
        except Exception as e:
            logger.error(f"Error creando fuente web: {e}")
            return jsonify({
                'success': False,
                'error': f'Error creando fuente: {str(e)}'
            }), 500
        
    except Exception as e:
        logger.error(f"Error en API create_web_source: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@web_sources_api.route('/<source_id>', methods=['GET'])
def get_web_source(source_id: str):
    """Obtener detalles de una fuente web específica"""
    try:
        web_ingestion_service = get_web_ingestion_service()
        source = web_ingestion_service.get_source(source_id)
        
        if not source:
            return jsonify({
                'success': False,
                'error': 'Fuente no encontrada'
            }), 404
        
        # Incluir estadísticas y páginas
        source_data = source.to_dict()
        source_data['stats'] = web_ingestion_service.get_source_stats(source_id).to_dict()
        source_data['pages'] = [
            page.to_dict() for page in web_ingestion_service.get_source_pages(source_id)
        ]
        
        return jsonify({
            'success': True,
            'source': source_data
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo fuente {source_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@web_sources_api.route('/<source_id>', methods=['PUT'])
def update_web_source(source_id: str):
    """Actualizar configuración de fuente web"""
    try:
        web_ingestion_service = get_web_ingestion_service()
        source = web_ingestion_service.get_source(source_id)
        
        if not source:
            return jsonify({
                'success': False,
                'error': 'Fuente no encontrada'
            }), 404
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No se proporcionaron datos'
            }), 400
        
        # Actualizar fuente
        success = web_ingestion_service.update_source(source_id, data)
        
        if success:
            updated_source = web_ingestion_service.get_source(source_id)
            return jsonify({
                'success': True,
                'source': updated_source.to_dict(),
                'message': 'Fuente actualizada exitosamente'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Error actualizando fuente'
            }), 500
        
    except Exception as e:
        logger.error(f"Error actualizando fuente {source_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@web_sources_api.route('/<source_id>', methods=['DELETE'])
def delete_web_source(source_id: str):
    """Eliminar fuente web"""
    try:
        web_ingestion_service = get_web_ingestion_service()
        success = web_ingestion_service.delete_source(source_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Fuente eliminada exitosamente'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Fuente no encontrada'
            }), 404
        
    except Exception as e:
        logger.error(f"Error eliminando fuente {source_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@web_sources_api.route('/<source_id>/scrape', methods=['POST'])
def scrape_web_source(source_id: str):
    """Ejecutar scraping de una fuente web"""
    try:
        web_ingestion_service = get_web_ingestion_service()
        data = request.get_json() or {}
        max_workers = data.get('max_workers', 3)
        
        # Ejecutar scraping
        results = web_ingestion_service.scrape_source(source_id, max_workers)
        
        return jsonify({
            'success': True,
            'results': results,
            'message': f'Scraping completado: {results.get("total_pages", 0)} páginas procesadas'
        })
        
    except Exception as e:
        logger.error(f"Error en scraping de fuente {source_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@web_sources_api.route('/<source_id>/test', methods=['POST'])
def test_web_source(source_id: str):
    """Probar configuración de una fuente web"""
    try:
        web_ingestion_service = get_web_ingestion_service()
        source = web_ingestion_service.get_source(source_id)
        
        if not source:
            return jsonify({
                'success': False,
                'error': 'Fuente no encontrada'
            }), 404
        
        # Probar primera URL de la fuente
        if source.base_urls:
            test_url = source.base_urls[0]
            test_results = web_ingestion_service.test_url(test_url)
            
            return jsonify({
                'success': True,
                'results': test_results,
                'message': f'Prueba de {test_url} completada'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No hay URLs configuradas en la fuente'
            }), 400
        
    except Exception as e:
        logger.error(f"Error en test de fuente web: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@web_sources_api.route('/validate-url', methods=['POST'])
def validate_web_url():
    """Validar una URL antes de agregar a fuente"""
    try:
        web_ingestion_service = get_web_ingestion_service()
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({
                'success': False,
                'error': 'URL requerida'
            }), 400
        
        # Validación básica de formato
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("URL inválida")
        except Exception:
            return jsonify({
                'success': False,
                'error': 'URL con formato inválido'
            }), 400
        
        # Test de conectividad usando el servicio
        test_results = web_ingestion_service.test_url(url)
        
        return jsonify({
            'success': True,
            'validation': test_results
        })
        
    except Exception as e:
        logger.error(f"Error validando URL: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@web_sources_api.route('/stats', methods=['GET'])
def get_web_stats():
    """Obtener estadísticas globales del servicio web"""
    try:
        web_ingestion_service = get_web_ingestion_service()
        stats = web_ingestion_service.get_all_stats()
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Exportar para registro
__all__ = ['web_sources_api']