"""
API Routes para gestión de fuentes de datos
TFM Vicente Caruncho - Sistemas Inteligentes
"""

from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
from typing import Dict, Any, List
import os
from pathlib import Path

from app.core.logger import get_logger
from app.services.document_ingestion_service import document_ingestion_service
from app.models.data_sources import DataSourceType, create_document_source


# Blueprint para API de fuentes de datos
data_sources_api = Blueprint('data_sources_api', __name__, url_prefix='/api/data-sources')
logger = get_logger("data_sources_api")


def validate_directories(directories: List[str]) -> tuple[List[str], List[str]]:
    """Validar directorios y retornar válidos e inválidos"""
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


@data_sources_api.route('', methods=['GET'])
def list_sources():
    """Listar todas las fuentes de datos"""
    try:
        sources = document_ingestion_service.list_sources()
        
        # Convertir a formato JSON con estadísticas
        sources_data = []
        for source in sources:
            source_dict = source.to_dict()
            
            # Agregar estadísticas
            try:
                stats = document_ingestion_service.get_source_stats(source.id)
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
    """Crear nueva fuente de datos"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No se proporcionaron datos'
            }), 400
        
        # Validar campos requeridos
        required_fields = ['name', 'directories']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Campo requerido: {field}'
                }), 400
        
        # Validar directorios
        directories = data['directories']
        if not isinstance(directories, list) or not directories:
            return jsonify({
                'success': False,
                'error': 'Se debe proporcionar al menos un directorio'
            }), 400
        
        valid_dirs, invalid_dirs = validate_directories(directories)
        
        if not valid_dirs:
            return jsonify({
                'success': False,
                'error': 'No se encontraron directorios válidos',
                'invalid_directories': invalid_dirs
            }), 400
        
        # Crear fuente de documentos
        source = document_ingestion_service.create_source(
            name=data['name'],
            directories=valid_dirs,
            file_extensions=data.get('file_extensions', ['.pdf', '.docx', '.txt']),
            recursive=data.get('recursive', True),
            exclude_patterns=data.get('exclude_patterns', []),
            max_file_size=data.get('max_file_size', 100 * 1024 * 1024)
        )
        
        # Respuesta con advertencias si hay directorios inválidos
        response_data = {
            'success': True,
            'source': source.to_dict(),
            'message': f'Fuente creada exitosamente: {source.name}'
        }
        
        if invalid_dirs:
            response_data['warnings'] = [
                f'Directorios ignorados (no existen): {", ".join(invalid_dirs)}'
            ]
        
        logger.info(f"Fuente creada: {source.name} ({source.id})")
        return jsonify(response_data), 201
        
    except Exception as e:
        logger.error(f"Error creando fuente: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_sources_api.route('/<source_id>', methods=['GET'])
def get_source(source_id: str):
    """Obtener detalles de una fuente específica"""
    try:
        source = document_ingestion_service.get_source(source_id)
        
        if not source:
            return jsonify({
                'success': False,
                'error': 'Fuente no encontrada'
            }), 404
        
        # Incluir estadísticas y documentos
        source_data = source.to_dict()
        source_data['stats'] = document_ingestion_service.get_source_stats(source_id).to_dict()
        source_data['documents'] = [
            doc.to_dict() for doc in document_ingestion_service.get_source_documents(source_id)
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


@data_sources_api.route('/<source_id>', methods=['PUT'])
def update_source(source_id: str):
    """Actualizar configuración de fuente"""
    try:
        source = document_ingestion_service.get_source(source_id)
        
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
        
        # Validar directorios si se proporcionan
        if 'directories' in data:
            valid_dirs, invalid_dirs = validate_directories(data['directories'])
            if not valid_dirs:
                return jsonify({
                    'success': False,
                    'error': 'No se encontraron directorios válidos',
                    'invalid_directories': invalid_dirs
                }), 400
            data['directories'] = valid_dirs
        
        # Actualizar fuente
        success = document_ingestion_service.update_source(source_id, data)
        
        if success:
            updated_source = document_ingestion_service.get_source(source_id)
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


@data_sources_api.route('/<source_id>', methods=['DELETE'])
def delete_source(source_id: str):
    """Eliminar fuente de datos"""
    try:
        source = document_ingestion_service.get_source(source_id)
        
        if not source:
            return jsonify({
                'success': False,
                'error': 'Fuente no encontrada'
            }), 404
        
        success = document_ingestion_service.delete_source(source_id)
        
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
    """Escanear archivos en directorio de fuente"""
    try:
        source = document_ingestion_service.get_source(source_id)
        
        if not source:
            return jsonify({
                'success': False,
                'error': 'Fuente no encontrada'
            }), 404
        
        files = document_ingestion_service.scan_source(source_id)
        
        # Formatear información de archivos
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
            'files': files_data,
            'total_files': len(files_data),
            'total_size_mb': round(sum(f['size_mb'] for f in files_data), 2)
        })
        
    except Exception as e:
        logger.error(f"Error escaneando fuente {source_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_sources_api.route('/<source_id>/changes', methods=['POST'])
def detect_changes(source_id: str):
    """Detectar cambios en archivos de fuente"""
    try:
        source = document_ingestion_service.get_source(source_id)
        
        if not source:
            return jsonify({
                'success': False,
                'error': 'Fuente no encontrada'
            }), 404
        
        changes = document_ingestion_service.detect_changes(source_id)
        
        # Formatear cambios
        changes_data = []
        for change in changes:
            change_data = {
                'type': change.type.value,
                'file': {
                    'path': change.file_info.path,
                    'name': Path(change.file_info.path).name,
                    'size': change.file_info.size,
                    'modified_time': change.file_info.modified_time.isoformat(),
                    'hash': change.file_info.hash
                }
            }
            
            if change.previous_info:
                change_data['previous'] = {
                    'hash': change.previous_info.file_hash,
                    'size': change.previous_info.file_size,
                    'processed_at': change.previous_info.processed_at.isoformat() if change.previous_info.processed_at else None
                }
            
            changes_data.append(change_data)
        
        # Estadísticas de cambios
        changes_stats = {
            'new': len([c for c in changes if c.type.value == 'new']),
            'modified': len([c for c in changes if c.type.value == 'modified']),
            'deleted': len([c for c in changes if c.type.value == 'deleted'])
        }
        
        return jsonify({
            'success': True,
            'changes': changes_data,
            'total_changes': len(changes_data),
            'stats': changes_stats
        })
        
    except Exception as e:
        logger.error(f"Error detectando cambios en fuente {source_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_sources_api.route('/<source_id>/sync', methods=['POST'])
def sync_source(source_id: str):
    """Sincronizar fuente (detectar y procesar cambios)"""
    try:
        source = document_ingestion_service.get_source(source_id)
        
        if not source:
            return jsonify({
                'success': False,
                'error': 'Fuente no encontrada'
            }), 404
        
        # Obtener parámetros opcionales
        data = request.get_json() or {}
        max_workers = data.get('max_workers', 3)
        
        # Ejecutar sincronización
        results = document_ingestion_service.sync_source(source_id, max_workers)
        
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
        source = document_ingestion_service.get_source(source_id)
        
        if not source:
            return jsonify({
                'success': False,
                'error': 'Fuente no encontrada'
            }), 404
        
        stats = document_ingestion_service.get_source_stats(source_id)
        
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


@data_sources_api.route('/<source_id>/documents', methods=['GET'])
def get_source_documents(source_id: str):
    """Obtener documentos procesados de fuente"""
    try:
        source = document_ingestion_service.get_source(source_id)
        
        if not source:
            return jsonify({
                'success': False,
                'error': 'Fuente no encontrada'
            }), 404
        
        # Parámetros de paginación
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        status_filter = request.args.get('status')
        
        documents = document_ingestion_service.get_source_documents(source_id)
        
        # Filtrar por estado si se especifica
        if status_filter:
            documents = [doc for doc in documents if doc.status.value == status_filter]
        
        # Ordenar por fecha de procesamiento (más recientes primero)
        documents.sort(key=lambda d: d.processed_at or d.modified_time, reverse=True)
        
        # Paginación
        total = len(documents)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_documents = documents[start_idx:end_idx]
        
        return jsonify({
            'success': True,
            'documents': [doc.to_dict() for doc in page_documents],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page
            }
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo documentos de fuente {source_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_sources_api.route('/<source_id>/logs', methods=['GET'])
def get_processing_logs(source_id: str):
    """Obtener logs de procesamiento de fuente"""
    try:
        source = document_ingestion_service.get_source(source_id)
        
        if not source:
            return jsonify({
                'success': False,
                'error': 'Fuente no encontrada'
            }), 404
        
        limit = request.args.get('limit', 50, type=int)
        logs = document_ingestion_service.get_processing_logs(source_id, limit)
        
        return jsonify({
            'success': True,
            'logs': logs,
            'total': len(logs)
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo logs de fuente {source_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@data_sources_api.route('/<source_id>/cleanup', methods=['POST'])
def cleanup_deleted_files(source_id: str):
    """Limpiar referencias a archivos eliminados"""
    try:
        source = document_ingestion_service.get_source(source_id)
        
        if not source:
            return jsonify({
                'success': False,
                'error': 'Fuente no encontrada'
            }), 404
        
        removed_count = document_ingestion_service.cleanup_deleted_files(source_id)
        
        return jsonify({
            'success': True,
            'removed_count': removed_count,
            'message': f'Limpiadas {removed_count} referencias a archivos eliminados'
        })
        
    except Exception as e:
        logger.error(f"Error limpiando archivos eliminados de fuente {source_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Rutas globales para todas las fuentes
@data_sources_api.route('/stats', methods=['GET'])
def get_all_stats():
    """Obtener estadísticas globales de todas las fuentes"""
    try:
        all_stats = document_ingestion_service.get_all_stats()
        
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


@data_sources_api.route('/logs', methods=['GET'])
def get_all_processing_logs():
    """Obtener logs de procesamiento de todas las fuentes"""
    try:
        limit = request.args.get('limit', 100, type=int)
        logs = document_ingestion_service.get_processing_logs(None, limit)
        
        return jsonify({
            'success': True,
            'logs': logs,
            'total': len(logs)
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo logs globales: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500