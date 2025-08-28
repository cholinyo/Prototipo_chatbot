"""
Rutas principales para Prototipo_chatbot - ACTUALIZADO
TFM Vicente Caruncho - Integración Pipeline RAG Dashboard

CAMBIOS APLICADOS:
- Importar data_sources_service en lugar de document_ingestion_service
- Actualizar todas las llamadas a métodos del servicio refactorizado
- Mantener compatibilidad con pipeline RAG
"""

from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from datetime import datetime
import time
from typing import Dict, Any

# Imports locales
from app.core.logger import get_logger
from app.core.config import (
    get_app_config, get_model_config, get_rag_config, 
    get_vector_store_config, get_security_config, validate_configuration
)

# INTEGRACIÓN: Pipeline RAG usando archivos existentes
try:
    from app.services.rag_pipeline import get_rag_pipeline
    from app.services.rag_pipeline import rag_pipeline
    from app.services.llm.llm_services import llm_service
    # ✅ CAMBIO CRÍTICO: Usar servicio refactorizado
    from app.services.data_sources_service import data_sources_service
    RAG_PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Servicios RAG no disponibles: {e}")
    get_rag_pipeline = None
    rag_pipeline = None
    llm_service = None
    data_sources_service = None
    RAG_PIPELINE_AVAILABLE = False

# CORRECCIÓN: Importar servicio web sin variables problemáticas
try:
    from app.services.web_ingestion_service import web_ingestion_service
    WEB_INGESTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Servicio de web ingestion no disponible: {e}")
    web_ingestion_service = None
    WEB_INGESTION_AVAILABLE = False

from app.models import SystemStats

# Crear blueprint
main_bp = Blueprint('main', __name__)
logger = get_logger("main_routes")

# Estadísticas globales del sistema (en producción usar base de datos)
system_stats = SystemStats()

# [RESTO DEL CÓDIGO DE LAS RUTAS PRINCIPALES SIN CAMBIOS...]
# Las rutas index(), dashboard(), about(), docs(), settings() permanecen iguales

@main_bp.route('/')
def index():
    """Página principal del sistema MEJORADA"""
    try:
        # Obtener configuraciones
        app_config = get_app_config()
        
        # NUEVA LÓGICA: Usar pipeline RAG mejorado
        if RAG_PIPELINE_AVAILABLE and get_rag_pipeline:
            pipeline = get_rag_pipeline()
            
            # Obtener estadísticas del pipeline integrado
            pipeline_stats = pipeline.get_stats()
            health_status = pipeline.health_check()
            
            # Actualizar estadísticas del sistema con datos reales del pipeline
            system_stats.documents_indexed = pipeline_stats.get('documents_count', 0)
            system_stats.update_indexing_stats(0, 0)  # Solo actualizar timestamp
            
            # Determinar estado general del sistema
            system_health = {
                'overall': health_status['status'],
                'services': {
                    'rag': 'healthy' if pipeline_stats.get('documents_count', 0) > 0 else 'warning',
                    'llm': health_status['components'].get('llm', 'unknown'),
                    'vector_store': health_status['components'].get('vector_store', 'unknown'),
                    'embeddings': health_status['components'].get('embeddings', 'unknown')
                }
            }
            
            # Estadísticas combinadas
            rag_stats = {
                'total_documents': pipeline_stats.get('documents_count', 0),
                'vector_store_type': pipeline_stats.get('vector_store_type', 'Unknown'),
                'embedding_model': pipeline_stats.get('embedding_model', 'all-MiniLM-L6-v2'),
                'memory_usage_mb': pipeline_stats.get('memory_usage_mb', 0),
                'chunk_size': pipeline_stats.get('chunk_size', 500),
                'last_update': pipeline_stats.get('last_update', 'Never')
            }
            
            # Estadísticas LLM del pipeline
            llm_stats = {
                'providers_available': pipeline_stats.get('llm_providers', {}),
                'models_available': pipeline_stats.get('available_models', {}),
                'total_requests': pipeline_stats.get('total_requests', 0),
                'avg_response_time': pipeline_stats.get('avg_response_time', 0)
            }
            
            # Estadísticas de ingesta del pipeline
            ingestion_stats = {
                'last_ingestion': pipeline_stats.get('last_ingestion', 'Never'),
                'supported_formats': pipeline_stats.get('supported_formats', []),
                'ingestion_queue': pipeline_stats.get('ingestion_queue', 0)
            }
            
        else:
            # Fallback a servicios existentes si no hay pipeline
            logger.warning("Pipeline RAG no disponible, usando servicios individuales")
            
            try:
                rag_stats = rag_pipeline.get_stats() if rag_pipeline else {}
                llm_stats = llm_service.get_service_stats() if llm_service else {}
                # ✅ CAMBIO: Usar servicio refactorizado  
                ingestion_stats = data_sources_service.get_service_stats() if data_sources_service else {}
                
                # Actualizar estadísticas del sistema
                system_stats.documents_indexed = rag_stats.get('total_documents', 0)
                system_stats.update_indexing_stats(0, 0)
                
                # Determinar estado general del sistema
                providers_available = llm_stats.get('providers_available', {})
                system_health = {
                    'overall': 'healthy' if any(providers_available.values()) else 'warning',
                    'services': {
                        'rag': 'healthy' if rag_stats.get('total_documents', 0) > 0 else 'warning',
                        'llm': 'healthy' if any(providers_available.values()) else 'error',
                        'ingestion': 'healthy'
                    }
                }
                
            except Exception as fallback_error:
                logger.error(f"Error accediendo a servicios individuales: {fallback_error}")
                
                # Valores por defecto si todo falla
                rag_stats = {'total_documents': 0, 'error': 'Service unavailable'}
                llm_stats = {'providers_available': {}, 'error': 'Service unavailable'}
                ingestion_stats = {'error': 'Service unavailable'}
                
                system_health = {
                    'overall': 'error',
                    'services': {
                        'rag': 'error',
                        'llm': 'error',
                        'ingestion': 'error'
                    }
                }
        
        # Contexto para el template
        context = {
            'app_config': app_config,
            'system_health': system_health,
            'system_stats': system_stats.to_dict(),
            'rag_stats': rag_stats,
            'llm_stats': llm_stats,
            'ingestion_stats': ingestion_stats,
            'providers_available': llm_stats.get('providers_available', {}),
            'pipeline_available': RAG_PIPELINE_AVAILABLE,
            'quick_actions': _get_quick_actions(),
            'recent_activity': _get_recent_activity()
        }
        
        logger.info("Página principal cargada",
                   documents_indexed=system_stats.documents_indexed,
                   providers_available=list(llm_stats.get('providers_available', {}).keys()),
                   overall_health=system_health['overall'],
                   pipeline_available=RAG_PIPELINE_AVAILABLE)
        
        return render_template('index.html', **context)
        
    except Exception as e:
        logger.error("Error cargando página principal", error=str(e))
        flash('Error cargando la página principal', 'error')
        return render_template('errors/500.html'), 500


@main_bp.route('/fuentes-datos')
def fuentes_datos():
    """Interfaz central de gestión de fuentes de datos - ACTUALIZADA"""
    try:
        # Obtener configuraciones
        app_config = get_app_config()
        
        # ✅ CAMBIO: Usar servicio refactorizado
        # Estadísticas globales de todas las fuentes
        try:
            # Obtener todas las fuentes usando servicio refactorizado
            sources = data_sources_service.list_sources() if data_sources_service else []
            all_stats = data_sources_service.get_all_stats() if data_sources_service else []
            
            # Calcular estadísticas globales
            global_stats = {
                'total_sources': len(sources),
                'total_documents': sum(s.processed_files for s in all_stats),
                'total_size_mb': sum(s.total_size_mb for s in all_stats),
                'total_chunks': sum(s.total_chunks for s in all_stats),
                'success_rate': 0
            }
            
            # Calcular tasa de éxito global
            total_files = sum(s.total_files for s in all_stats)
            if total_files > 0:
                global_stats['success_rate'] = (global_stats['total_documents'] / total_files) * 100
            
            # ✅ CAMBIO: Estadísticas por tipo usando servicio refactorizado
            from app.models.data_sources import DataSourceType
            sources_by_type = {
                'documents': len(data_sources_service.list_sources_by_type(DataSourceType.DOCUMENTS)),
                'web': len(data_sources_service.list_sources_by_type(DataSourceType.WEB)),
                'api': len(data_sources_service.list_sources_by_type(DataSourceType.API)),
                'database': len(data_sources_service.list_sources_by_type(DataSourceType.DATABASE))
            } if data_sources_service else {
                'documents': 0,
                'web': 0,
                'api': 0,
                'database': 0
            }
            
        except Exception as e:
            logger.warning(f"Error obteniendo estadísticas globales: {e}")
            global_stats = {
                'total_sources': 0,
                'total_documents': 0,
                'total_size_mb': 0,
                'total_chunks': 0,
                'success_rate': 0
            }
            sources_by_type = {
                'documents': 0,
                'web': 0,
                'api': 0,
                'database': 0
            }
        
        # Información del procesador
        try:
            from app.services.ingestion.document_processor import DocumentProcessor
            processor = DocumentProcessor()
            processor_info = processor.get_processor_info()
        except Exception as e:
            logger.warning(f"Error obteniendo info del procesador: {e}")
            processor_info = {
                'supported_extensions': ['.pdf', '.docx', '.txt'],
                'processors_available': {
                    'pdf': False,
                    'docx': False,
                    'pandas': False
                }
            }
        
        # Contexto para el template
        context = {
            'app_config': app_config,
            'global_stats': global_stats,
            'sources_by_type': sources_by_type,
            'processor_info': processor_info,
            'page_title': 'Gestión de Fuentes de Datos'
        }
        
        logger.info("Página de fuentes de datos accedida")
        return render_template('fuentes_datos.html', **context)
        
    except Exception as e:
        logger.error(f"Error cargando página de fuentes de datos: {e}")
        flash(f"Error cargando la página: {str(e)}", "error")
        return redirect(url_for('main.index'))


@main_bp.route('/data-sources')
def data_sources():
    """Página de gestión de fuentes de datos - ACTUALIZADA"""
    try:
        # Obtener configuraciones
        app_config = get_app_config()
        
        # Información del procesador de documentos
        try:
            from app.services.ingestion.document_processor import DocumentProcessor
            processor = DocumentProcessor()
            processor_info = processor.get_processor_info()
        except Exception as e:
            logger.warning(f"Error obteniendo info del procesador: {e}")
            processor_info = {
                'supported_extensions': ['.pdf', '.docx', '.txt'],
                'processors_available': {
                    'pdf': False,
                    'docx': False,
                    'pandas': False
                }
            }
        
        # ✅ CAMBIO: Usar servicio refactorizado
        # Estadísticas globales básicas
        try:
            global_stats = {
                'total_sources': len(data_sources_service.list_sources()) if data_sources_service else 0,
                'system_ready': True
            }
        except Exception as e:
            logger.warning(f"Error obteniendo estadísticas: {e}")
            global_stats = {
                'total_sources': 0,
                'system_ready': False
            }
        
        return render_template('data_sources.html',
                             app_config=app_config,
                             processor_info=processor_info,
                             global_stats=global_stats,
                             page_title="Fuentes de Datos")
                             
    except Exception as e:
        logger.error(f"Error en página de fuentes de datos: {e}")
        flash(f"Error cargando página: {str(e)}", 'error')
        return redirect(url_for('main.index'))


@main_bp.route('/webs')
def webs():
    """Página de gestión de sitios web para web scraping - ACTUALIZADA"""
    try:
        app_config = get_app_config()
        
        # ✅ CAMBIO: Usar servicio refactorizado para estadísticas web
        if WEB_INGESTION_AVAILABLE and web_ingestion_service and data_sources_service:
            try:
                # Obtener fuentes web usando servicio refactorizado
                from app.models.data_sources import DataSourceType
                web_sources = data_sources_service.list_sources_by_type(DataSourceType.WEB)
                
                web_stats = {
                    'total_sources': len(web_sources),
                    'total_pages': 0,  # TODO: Sumar páginas de todas las fuentes web
                    'active_scraping': 0,  # TODO: Implementar conteo de tareas activas
                    'last_update': max([s.last_sync for s in web_sources if s.last_sync], default=datetime.min) if web_sources else None
                }
                
                logger.info("Estadísticas de web scraping obtenidas del servicio refactorizado")
                
            except Exception as e:
                logger.warning(f"Error obteniendo estadísticas de web scraping: {e}")
                
                # Valores por defecto si el servicio no está disponible
                web_stats = {
                    'total_sources': 0,
                    'total_pages': 0,
                    'active_scraping': 0,
                    'last_update': None
                }
        else:
            logger.warning("Servicio de web ingestion no disponible")
            web_stats = {
                'total_sources': 0,
                'total_pages': 0,
                'active_scraping': 0,
                'last_update': None
            }
        
        context = {
            'app_config': app_config,
            'web_stats': web_stats,
            'page_title': 'Gestión de Sitios Web',
            'web_ingestion_available': WEB_INGESTION_AVAILABLE
        }
        
        logger.info("Página de gestión de sitios web accedida")
        return render_template('webs.html', **context)
        
    except Exception as e:
        logger.error(f"Error cargando página de gestión de webs: {e}")
        flash('Error cargando la página de gestión de sitios web', 'error')
        return redirect(url_for('main.fuentes_datos'))


# [RESTO DE LAS RUTAS PERMANECEN IGUALES - dashboard(), about(), docs(), settings(), etc.]
# Solo cambian las que hacen referencia directa a document_ingestion_service

# Las funciones auxiliares también necesitan actualización donde corresponda
def _get_recent_activity():
    """Obtener actividad reciente del sistema - ACTUALIZADA"""
    activities = [
        {
            'timestamp': datetime.now(),
            'type': 'system',
            'message': 'Sistema iniciado correctamente',
            'icon': 'fa-check-circle',
            'color': 'success'
        }
    ]
    
    # ✅ CAMBIO: Usar servicio refactorizado para estadísticas
    if data_sources_service:
        try:
            service_stats = data_sources_service.get_service_stats()
            
            activities.append({
                'timestamp': datetime.now(),
                'type': 'ingestion',
                'message': f'Servicio de fuentes activo con {service_stats.get("total_sources", 0)} fuentes',
                'icon': 'fa-database',
                'color': 'info'
            })
            
            sources_by_type = service_stats.get('sources_by_type', {})
            for source_type, count in sources_by_type.items():
                if count > 0:
                    activities.append({
                        'timestamp': datetime.now(),
                        'type': 'config',
                        'message': f'Fuentes {source_type}: {count}',
                        'icon': 'fa-folder',
                        'color': 'primary'
                    })
        except Exception as e:
            activities.append({
                'timestamp': datetime.now(),
                'type': 'warning',
                'message': f'Servicio de fuentes con advertencias: {str(e)[:50]}...',
                'icon': 'fa-exclamation-triangle',
                'color': 'warning'
            })
    
    # Añadir actividad del pipeline si está disponible
    if RAG_PIPELINE_AVAILABLE and get_rag_pipeline:
        try:
            pipeline = get_rag_pipeline()
            stats = pipeline.get_stats()
            
            activities.extend([
                {
                    'timestamp': datetime.now(),
                    'type': 'pipeline',
                    'message': f'Pipeline RAG activo con {stats.get("documents_count", 0)} documentos',
                    'icon': 'fa-search',
                    'color': 'info'
                },
                {
                    'timestamp': datetime.now(),
                    'type': 'config',
                    'message': f'Vector store: {stats.get("vector_store_type", "Unknown")}',
                    'icon': 'fa-database',
                    'color': 'primary'
                }
            ])
        except Exception as e:
            activities.append({
                'timestamp': datetime.now(),
                'type': 'warning',
                'message': f'Pipeline RAG con advertencias: {str(e)[:50]}...',
                'icon': 'fa-exclamation-triangle',
                'color': 'warning'
            })
    else:
        activities.append({
            'timestamp': datetime.now(),
            'type': 'info',
            'message': 'Pipeline RAG no cargado - usando servicios básicos',
            'icon': 'fa-info-circle',
            'color': 'secondary'
        })
    
    return activities

# [RESTO DE FUNCIONES AUXILIARES PERMANECEN IGUALES...]

# TODAS LAS DEMÁS RUTAS (dashboard, about, docs, settings, pipeline_status, etc.) 
# PERMANECEN EXACTAMENTE IGUALES - Solo cambian las que usan document_ingestion_service