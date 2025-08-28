"""
API Routes para gestión de fuentes de datos - ACTUALIZADO
TFM Vicente Caruncho - Sistemas Inteligentes

CAMBIOS APLICADOS:
1. Importar data_sources_service en lugar de document_ingestion_service  
2. Actualizar todas las llamadas a métodos
3. Soporte para todos los tipos de fuentes (documentos, web, API, BD)
4. Validaciones específicas por tipo de fuente
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

from app.core.logger import get_logger
from app.models.data_sources import DataSourceType, DataSourceStatus
# ✅ CAMBIO CRÍTICO: Importar servicio refactorizado
from app.services.data_sources_service import data_sources_service

# Blueprint para API de fuentes de datos
data_sources_api = Blueprint('data_sources_api', __name__, url_prefix='/api/data-sources')
logger = get_logger("data_sources_api")


def validate_directories(directories: List[str]) -> tuple[List[str], List[str]]:
    """Validar directorios para fuentes de documentos"""
    valid = []
    invalid = []
    
    for directory in directories:
        try:
            path = Path(directory).resolve()
            if path.exists() and path.is_dir():
                valid.append(str(path))
            else:
                invalid.append(directory)
        except Exception:
            invalid.append(directory)
    
    return valid, invalid


def validate_web_source_data(data: Dict[str, Any]) -> tuple[bool, str]:
    """Validar datos para fuente web"""
    if 'base_urls' not in data or not data['base_urls']:
        return False, "Se requiere al menos una URL base"
    
    if not isinstance(data['base_urls'], list):
        return False, "base_urls debe ser una lista"
    
    # Validar URLs básicas
    for url in data['base_urls']:
        if not url.startswith(('http://', 'https://')):
            return False, f"URL inválida: {url}"
    
    return True, ""


def validate_api_source_data(data: Dict[str, Any]) -> tuple[bool, str]:
    """Validar datos para fuente API"""
    if 'base_url' not in data or not data['base_url']:
        return False, "Se requiere base_url"
    
    if not data['base_url'].startswith(('http://', 'https://')):
        return False, "base_url debe ser una URL válida"
    
    auth_type = data.get('auth_type', 'none')
    valid_auth_types = ['none', 'bearer', 'api_key', 'basic', 'oauth2']
    if auth_type not in valid_auth_types:
        return False, f"auth_type debe ser uno de: {valid_auth_types}"
    
    return True, ""


def validate_database_source_data(data: Dict[str, Any]) -> tuple[bool, str]:
    """Validar datos para fuente de base de datos"""
    if 'db_type' not in data:
        return False, "Se requiere db_type"
    
    if 'connection_config' not in data:
        return False, "Se requiere connection_config"
    
    db_type = data['db_type']
    valid_db_types = ['postgresql', 'mysql', 'sqlite', 'mssql']
    if db_type not in valid_db_types:
        return False, f"db_type debe ser uno de: {valid_db_types}"
    
    # Validar campos requeridos según tipo
    config = data['connection_config']
    required_fields = {
        'postgresql': ['host', 'port', 'database', 'user', 'password'],
        'mysql': ['host', 'port', 'database', 'user', 'password'], 
        'sqlite': ['database'],
        'mssql': ['host', 'port', 'database', 'user', 'password']
    }
    
    for field in required_fields.get(db_type, []):
        if field not in config or not config[field]:
            return False, f"Campo requerido en connection_config: {field}"
    
    return True, ""


@data_sources_api.route('', methods=['GET'])
def list_sources():
    """Listar todas las fuentes de datos"""
    try:
        # ✅ CAMBIO: Usar servicio refactorizado
        sources = data_sources_service.list_sources()
        
        # Convertir a formato JSON con estadísticas
        sources_data = []
        for source in sources:
            source_dict = source.to_dict()
            
            # Agregar estadísticas
            try:
                stats = data_sources_service.get_source_stats(source.id)
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
        logger.error(f"Error listando fuentes: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_sources_api.route('', methods=['POST'])
def create_source():
    """Crear nueva fuente de datos - TODOS LOS TIPOS"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No se proporcionaron datos'
            }), 400
        
        # Campos requeridos básicos
        if 'name' not in data or not data['name']:
            return jsonify({
                'success': False,
                'error': 'Campo requerido: name'
            }), 400
        
        if 'type' not in data:
            return jsonify({
                'success': False,
                'error': 'Campo requerido: type'
            }), 400
        
        source_type = data['type']
        name = data['name']
        
        # Crear según tipo de fuente
        if source_type == 'documents':
            return _create_document_source(name, data)
        elif source_type == 'web':
            return _create_web_source(name, data)
        elif source_type == 'api':
            return _create_api_source(name, data)
        elif source_type == 'database':
            return _create_database_source(name, data)
        else:
            return jsonify({
                'success': False,
                'error': f'Tipo de fuente no soportado: {source_type}'
            }), 400
        
    except Exception as e:
        logger.error(f"Error creando fuente: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def _create_document_source(name: str, data: Dict[str, Any]):
    """Crear fuente de documentos"""
    # Validar directorios
    if 'directories' not in data or not data['directories']:
        return jsonify({
            'success': False,
            'error': 'Campo requerido para documentos: directories'
        }), 400
    
    directories = data['directories']
    if not isinstance(directories, list):
        return jsonify({
            'success': False,
            'error': 'directories debe ser una lista'
        }), 400
    
    valid_dirs, invalid_dirs = validate_directories(directories)
    
    if not valid_dirs:
        return jsonify({
            'success': False,
            'error': 'No se encontraron directorios válidos',
            'invalid_directories': invalid_dirs
        }), 400
    
    # ✅ CAMBIO: Usar servicio refactorizado
    source = data_sources_service.create_document_source(
        name=name,
        directories=valid_dirs,
        file_extensions=data.get('file_extensions', ['.pdf', '.docx', '.txt']),
        recursive=data.get('recursive', True),
        exclude_patterns=data.get('exclude_patterns', []),
        max_file_size=data.get('max_file_size', 100 * 1024 * 1024)
    )
    
    response_data = {
        'success': True,
        'source': source.to_dict(),
        'message': f'Fuente de documentos creada: {source.name}'
    }
    
    if invalid_dirs:
        response_data['warnings'] = [
            f'Directorios ignorados (no existen): {", ".join(invalid_dirs)}'
        ]
    
    logger.info(f"Fuente de documentos creada: {source.name} ({source.id})")
    return jsonify(response_data), 201


def _create_web_source(name: str, data: Dict[str, Any]):
    """Crear fuente web"""
    # Validar datos web
    is_valid, error = validate_web_source_data(data)
    if not is_valid:
        return jsonify({
            'success': False,
            'error': error
        }), 400
    
    # ✅ CAMBIO: Usar servicio refactorizado
    source = data_sources_service.create_web_source(
        name=name,
        base_urls=data['base_urls'],
        allowed_domains=data.get('allowed_domains', []),
        max_depth=data.get('max_depth', 2),
        follow_links=data.get('follow_links', True),
        delay_seconds=data.get('delay_seconds', 1.0),
        user_agent=data.get('user_agent', "Mozilla/5.0 (Prototipo_chatbot TFM UJI)"),
        content_selectors=data.get('content_selectors', ['main', 'article', '.content']),
        exclude_patterns=data.get('exclude_patterns', ['/admin', '/login']),
        min_content_length=data.get('min_content_length', 100),
        use_javascript=data.get('use_javascript', False)
    )
    
    response_data = {
        'success': True,
        'source': source.to_dict(),
        'message': f'Fuente web creada: {source.name}'
    }
    
    logger.info(f"Fuente web creada: {source.name} ({source.id}) - {len(data['base_urls'])} URLs")
    return jsonify(response_data), 201


def _create_api_source(name: str, data: Dict[str, Any]):
    """Crear fuente API"""
    # Validar datos API
    is_valid, error = validate_api_source_data(data)
    if not is_valid:
        return jsonify({
            'success': False,
            'error': error
        }), 400
    
    # ✅ CAMBIO: Usar servicio refactorizado
    source = data_sources_service.create_api_source(
        name=name,
        base_url=data['base_url'],
        auth_type=data.get('auth_type', 'none'),
        auth_credentials=data.get('auth_credentials', {}),
        endpoints=data.get('endpoints', []),
        default_headers=data.get('default_headers', {}),
        timeout_seconds=data.get('timeout_seconds', 30),
        content_fields=data.get('content_fields', []),
        metadata_fields=data.get('metadata_fields', []),
        min_content_length=data.get('min_content_length', 50)
    )
    
    response_data = {
        'success': True,
        'source': source.to_dict(),
        'message': f'Fuente API creada: {source.name}'
    }
    
    logger.info(f"Fuente API creada: {source.name} ({source.id})")
    return jsonify(response_data), 201


def _create_database_source(name: str, data: Dict[str, Any]):
    """Crear fuente de base de datos"""
    # Validar datos BD
    is_valid, error = validate_database_source_data(data)
    if not is_valid:
        return jsonify({
            'success': False,
            'error': error
        }), 400
    
    # ✅ CAMBIO: Usar servicio refactorizado
    source = data_sources_service.create_database_source(
        name=name,
        db_type=data['db_type'],
        connection_config=data['connection_config'],
        pool_size=data.get('pool_size', 5),
        max_overflow=data.get('max_overflow', 10),
        timeout_seconds=data.get('timeout_seconds', 30),
        queries=data.get('queries', []),
        content_fields=data.get('content_fields', []),
        metadata_fields=data.get('metadata_fields', []),
        batch_size=data.get('batch_size', 1000),
        min_content_length=data.get('min_content_length', 50)
    )
    
    response_data = {
        'success': True,
        'source': source.to_dict(),
        'message': f'Fuente BD creada: {source.name}'
    }
    
    logger.info(f"Fuente BD creada: {source.name} ({source.id}) - {data['db_type']}")
    return jsonify(response_data), 201


@data_sources_api.route('/<source_id>', methods=['GET'])
def get_source(source_id: str):
    """Obtener detalles de una fuente específica"""
    try:
        # ✅ CAMBIO: Usar servicio refactorizado
        source = data_sources_service.get_source(source_id)
        
        if not source:
            return jsonify({
                'success': False,
                'error': 'Fuente no encontrada'
            }), 404
        
        # Incluir estadísticas
        source_data = source.to_dict()
        try:
            stats = data_sources_service.get_source_stats(source_id)
            source_data['stats'] = stats.to_dict()
        except Exception as e:
            logger.warning(f"Error obteniendo estadísticas: {e}")
            source_data['stats'] = None
        
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


@data_sources_api.route('/<source_id>', methods=['DELETE'])
def delete_source(source_id: str):
    """Eliminar fuente de datos"""
    try:
        # ✅ CAMBIO: Usar servicio refactorizado
        source = data_sources_service.get_source(source_id)
        
        if not source:
            return jsonify({
                'success': False,
                'error': 'Fuente no encontrada'
            }), 404
        
        success = data_sources_service.delete_source(source_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Fuente eliminada: {source.name}'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Error eliminando fuente'
            }), 500
            
    except Exception as e:
        logger.error(f"Error eliminando fuente {source_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_sources_api.route('/<source_id>/scan', methods=['POST'])
def scan_source(source_id: str):
    """Escanear fuente según su tipo"""
    try:
        # ✅ CAMBIO: Usar servicio refactorizado  
        source = data_sources_service.get_source(source_id)
        
        if not source:
            return jsonify({
                'success': False,
                'error': 'Fuente no encontrada'
            }), 404
        
        # Escanear según tipo
        from app.models.data_sources import DocumentSource, WebSource, APISource, DatabaseSource
        
        if isinstance(source, DocumentSource):
            # Escanear archivos en directorios
            files = data_sources_service.scan_document_directories(source)
            
            files_data = []
            for file_info in files:
                files_data.append({
                    'path': file_info.path,
                    'name': Path(file_info.path).name,
                    'size': file_info.size,
                    'size_mb': round(file_info.size / (1024 * 1024), 2),
                    'modified_time': file_info.modified_time.isoformat(),
                    'extension': file_info.extension,
                    'hash': file_info.hash
                })
            
            return jsonify({
                'success': True,
                'type': 'documents',
                'files': files_data,
                'total_files': len(files_data),
                'total_size_mb': round(sum(f['size_mb'] for f in files_data), 2)
            })
        
        elif isinstance(source, WebSource):
            # Test de conectividad web
            connectivity = data_sources_service.test_web_source_connectivity(source)
            
            return jsonify({
                'success': True,
                'type': 'web',
                'connectivity': connectivity,
                'total_urls': len(source.base_urls),
                'accessible_urls': connectivity.get('accessible_count', 0)
            })
        
        elif isinstance(source, APISource):
            # Test de conectividad API
            connectivity = data_sources_service.test_api_source_connectivity(source)
            
            return jsonify({
                'success': True,
                'type': 'api',
                'connectivity': connectivity,
                'endpoints_count': len(source.endpoints)
            })
        
        elif isinstance(source, DatabaseSource):
            # Test de conectividad BD
            connectivity = data_sources_service.test_database_source_connectivity(source)
            
            return jsonify({
                'success': True,
                'type': 'database',
                'connectivity': connectivity,
                'queries_count': len(source.queries)
            })
        
        else:
            return jsonify({
                'success': False,
                'error': f'Tipo de fuente no soportado para escaneo: {type(source)}'
            }), 400
        
    except Exception as e:
        logger.error(f"Error escaneando fuente {source_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_sources_api.route('/<source_id>/sync', methods=['POST'])
def sync_source(source_id: str):
    """Sincronizar fuente (detectar y procesar cambios)"""
    try:
        # ✅ CAMBIO: Usar servicio refactorizado
        source = data_sources_service.get_source(source_id)
        
        if not source:
            return jsonify({
                'success': False,
                'error': 'Fuente no encontrada'
            }), 404
        
        # Obtener parámetros opcionales
        data = request.get_json() or {}
        
        # Ejecutar sincronización usando servicio refactorizado
        results = data_sources_service.sync_source(source_id, **data)
        
        return jsonify({
            'success': True,
            'results': results,
            'message': f'Sincronización completada en {results.get("processing_time", 0):.2f}s'
        })
        
    except Exception as e:
        logger.error(f"Error sincronizando fuente {source_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_sources_api.route('/<source_id>/stats', methods=['GET'])
def get_source_stats(source_id: str):
    """Obtener estadísticas detalladas de fuente"""
    try:
        # ✅ CAMBIO: Usar servicio refactorizado
        source = data_sources_service.get_source(source_id)
        
        if not source:
            return jsonify({
                'success': False,
                'error': 'Fuente no encontrada'
            }), 404
        
        stats = data_sources_service.get_source_stats(source_id)
        
        return jsonify({
            'success': True,
            'stats': stats.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas de fuente {source_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Rutas globales
@data_sources_api.route('/stats', methods=['GET'])
def get_all_stats():
    """Obtener estadísticas globales de todas las fuentes"""
    try:
        # ✅ CAMBIO: Usar servicio refactorizado
        all_stats = data_sources_service.get_all_stats()
        
        # Calcular totales globales
        global_stats = {
            'total_sources': len(all_stats),
            'total_files': sum(s.total_files for s in all_stats),
            'processed_files': sum(s.processed_files for s in all_stats),
            'failed_files': sum(s.failed_files for s in all_stats),
            'total_chunks': sum(s.total_chunks for s in all_stats),
            'total_size_mb': sum(s.total_size_mb for s in all_stats),
            'overall_success_rate': 0
        }
        
        if global_stats['total_files'] > 0:
            global_stats['overall_success_rate'] = (global_stats['processed_files'] / global_stats['total_files']) * 100
        
        return jsonify({
            'success': True,
            'global_stats': global_stats,
            'source_stats': [stats.to_dict() for stats in all_stats]
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas globales: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_sources_api.route('/service-info', methods=['GET'])
def get_service_info():
    """Obtener información del servicio de fuentes de datos"""
    try:
        # ✅ CAMBIO: Usar servicio refactorizado
        service_stats = data_sources_service.get_service_stats()
        
        return jsonify({
            'success': True,
            'service_stats': service_stats,
            'supported_types': [t.value for t in DataSourceType],
            'available_processors': service_stats.get('processors_available', {})
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo info del servicio: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500