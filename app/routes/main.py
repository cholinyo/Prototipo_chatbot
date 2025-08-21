"""
Rutas principales para Prototipo_chatbot
TFM Vicente Caruncho - Integración Pipeline RAG Dashboard
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
    from app.services.document_ingestion_service import document_ingestion_service
    RAG_PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Servicios RAG no disponibles: {e}")
    get_rag_pipeline = None
    rag_pipeline = None
    llm_service = None
    document_ingestion_service = None
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
                ingestion_stats = document_ingestion_service.get_service_stats() if document_ingestion_service else {}
                
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

@main_bp.route('/dashboard')
def dashboard():
    """Panel de control y métricas MEJORADO"""
    try:
        # NUEVA LÓGICA: Dashboard con pipeline RAG integrado
        if RAG_PIPELINE_AVAILABLE and get_rag_pipeline:
            pipeline = get_rag_pipeline()
            
            # Obtener estadísticas completas del pipeline
            pipeline_stats = pipeline.get_stats()
            health_status = pipeline.health_check()
            
            # Estadísticas de uso mejoradas
            usage_stats = {
                'queries_today': pipeline_stats.get('queries_today', _get_queries_today()),
                'response_time_avg': pipeline_stats.get('avg_response_time', system_stats.avg_response_time),
                'success_rate': pipeline_stats.get('success_rate', system_stats.get_success_rate()),
                'uptime_hours': system_stats.get_uptime_hours(),
                'total_queries': pipeline_stats.get('total_queries', system_stats.total_queries),
                'queries_per_hour': pipeline_stats.get('queries_per_hour', 0)
            }
            
            # Métricas de rendimiento del pipeline
            performance_metrics = {
                'vector_store_size': pipeline_stats.get('vector_store_size_mb', 0),
                'total_documents': pipeline_stats.get('documents_count', 0),
                'embedding_model': pipeline_stats.get('embedding_model', 'all-MiniLM-L6-v2'),
                'chunk_size': pipeline_stats.get('chunk_size', 500),
                'embedding_dimensions': pipeline_stats.get('embedding_dimensions', 384),
                'indexing_time': pipeline_stats.get('last_indexing_time', 0),
                'memory_usage': pipeline_stats.get('memory_usage_mb', 0)
            }
            
            # Estado de proveedores LLM desde el pipeline
            providers_status = {}
            llm_providers = pipeline_stats.get('llm_providers', {})
            available_models = pipeline_stats.get('available_models', {})
            
            for provider, available in llm_providers.items():
                providers_status[provider] = {
                    'available': available,
                    'models': available_models.get(provider, []),
                    'model_count': len(available_models.get(provider, [])),
                    'status': 'healthy' if available else 'error',
                    'requests_today': pipeline_stats.get(f'{provider}_requests_today', 0),
                    'avg_response_time': pipeline_stats.get(f'{provider}_avg_response_time', 0),
                    'estimated_cost_today': pipeline_stats.get(f'{provider}_cost_today', 0)
                }
            
            # Estadísticas detalladas para el dashboard
            rag_stats = {
                'vector_store_type': pipeline_stats.get('vector_store_type', 'Unknown'),
                'total_documents': pipeline_stats.get('documents_count', 0),
                'total_chunks': pipeline_stats.get('total_chunks', 0),
                'embedding_model': pipeline_stats.get('embedding_model', 'Unknown'),
                'similarity_threshold': pipeline_stats.get('similarity_threshold', 0.5),
                'retrieval_k': pipeline_stats.get('default_k', 5)
            }
            
            llm_stats = {
                'providers_count': len(llm_providers),
                'models_count': sum(len(models) for models in available_models.values()),
                'total_requests': pipeline_stats.get('total_requests', 0),
                'avg_response_time': pipeline_stats.get('avg_response_time', 0)
            }
            
            ingestion_stats = {
                'last_ingestion': pipeline_stats.get('last_ingestion', 'Never'),
                'supported_formats': pipeline_stats.get('supported_formats', []),
                'ingestion_queue': pipeline_stats.get('ingestion_queue', 0),
                'processing_rate': pipeline_stats.get('processing_rate_docs_per_min', 0)
            }
            
        else:
            # Fallback a servicios individuales
            logger.warning("Pipeline RAG no disponible en dashboard, usando servicios individuales")
            
            try:
                rag_stats = rag_pipeline.get_stats() if rag_pipeline else {}
                llm_stats = llm_service.get_service_stats() if llm_service else {}
                ingestion_stats = document_ingestion_service.get_service_stats() if document_ingestion_service else {}
                
                # Estadísticas de uso básicas
                usage_stats = {
                    'queries_today': _get_queries_today(),
                    'response_time_avg': system_stats.avg_response_time,
                    'success_rate': system_stats.get_success_rate(),
                    'uptime_hours': system_stats.get_uptime_hours()
                }
                
                # Métricas de rendimiento básicas
                performance_metrics = {
                    'vector_store_size': rag_stats.get('memory_usage_mb', 0),
                    'total_documents': rag_stats.get('total_documents', 0),
                    'embedding_model': rag_stats.get('embedding_model', 'unknown'),
                    'chunk_size': rag_stats.get('chunk_size', 500)
                }
                
                # Estado de proveedores básico
                providers_status = _get_detailed_provider_status()
                
            except Exception as fallback_error:
                logger.error(f"Error en servicios individuales: {fallback_error}")
                
                # Valores mínimos por defecto
                usage_stats = {'queries_today': 0, 'response_time_avg': 0, 'success_rate': 0, 'uptime_hours': 0}
                performance_metrics = {'vector_store_size': 0, 'total_documents': 0, 'embedding_model': 'unknown', 'chunk_size': 0}
                providers_status = {}
                rag_stats = {}
                llm_stats = {}
                ingestion_stats = {}
        
        # Contexto completo para el dashboard
        context = {
            'usage_stats': usage_stats,
            'performance_metrics': performance_metrics,
            'providers_status': providers_status,
            'rag_stats': rag_stats,
            'llm_stats': llm_stats,
            'ingestion_stats': ingestion_stats,
            'system_stats': system_stats.to_dict(),
            'pipeline_available': RAG_PIPELINE_AVAILABLE,
            'charts_data': _get_charts_data(),
            'real_time_enabled': RAG_PIPELINE_AVAILABLE,  # Habilitar actualizaciones en tiempo real si hay pipeline
            'dashboard_config': {
                'auto_refresh': True,
                'refresh_interval': 30,  # segundos
                'show_advanced_metrics': RAG_PIPELINE_AVAILABLE
            }
        }
        
        return render_template('dashboard.html', **context)
        
    except Exception as e:
        logger.error("Error cargando dashboard", error=str(e))
        flash('Error cargando el dashboard', 'error')
        return redirect(url_for('main.index'))

@main_bp.route('/about')
def about():
    """Página de información sobre el proyecto MEJORADA"""
    try:
        app_config = get_app_config()
        
        # Información del proyecto actualizada
        project_info = {
            'title': 'Prototipo de Chatbot RAG para Administraciones Locales',
            'subtitle': 'Trabajo Final de Máster - Sistemas Inteligentes',
            'author': 'Vicente Caruncho Ramos',
            'tutor': 'Rafael Berlanga Llavori',
            'university': 'Universitat Jaume I',
            'master': 'Máster en Sistemas Inteligentes',
            'speciality': 'Interacción Avanzada y Gestión del Conocimiento',
            'year': '2025',
            'version': app_config.version,
            'github_url': 'https://github.com/cholinyo/Prototipo_chatbot',
            'completion_status': '95%'  # Actualizar según progreso real
        }
        
        # Tecnologías utilizadas actualizadas
        technologies = {
            'backend': [
                {'name': 'Python 3.11+', 'description': 'Lenguaje principal del sistema'},
                {'name': 'Flask 2.3+', 'description': 'Framework web modular'},
                {'name': 'sentence-transformers', 'description': 'Embeddings semánticos all-MiniLM-L6-v2'},
                {'name': 'FAISS', 'description': 'Búsqueda vectorial eficiente de Facebook AI'},
                {'name': 'ChromaDB', 'description': 'Base de datos vectorial moderna'},
                {'name': 'PyPDF2 + python-docx', 'description': 'Procesamiento de documentos'},
            ],
            'ai_models': [
                {'name': 'OpenAI GPT-4o', 'description': 'Modelo de lenguaje comercial de última generación'},
                {'name': 'OpenAI GPT-4o-mini', 'description': 'Modelo optimizado para costes'},
                {'name': 'Ollama LLaMA 3.2', 'description': 'Modelos locales de Meta (3B parámetros)'},
                {'name': 'Ollama Mistral 7B', 'description': 'Modelo francés de Mistral AI'},
                {'name': 'Ollama Gemma 2B', 'description': 'Modelo ligero de Google'},
                {'name': 'all-MiniLM-L6-v2', 'description': 'Modelo de embeddings 384 dimensiones'},
            ],
            'frontend': [
                {'name': 'Bootstrap 5.3', 'description': 'Framework CSS responsive'},
                {'name': 'JavaScript ES6+', 'description': 'Interactividad cliente y AJAX'},
                {'name': 'Chart.js', 'description': 'Visualización de datos dinámicos'},
                {'name': 'Font Awesome 6', 'description': 'Iconografía moderna'},
            ],
            'deployment': [
                {'name': 'Docker', 'description': 'Containerización para producción'},
                {'name': 'Azure App Service', 'description': 'Plataforma cloud objetivo'},
                {'name': 'GitHub Actions', 'description': 'CI/CD automatizado'},
                {'name': 'Nginx', 'description': 'Proxy reverso y balanceador'},
            ]
        }
        
        # Características principales actualizadas
        features = [
            {
                'icon': 'fa-search',
                'title': 'RAG Avanzado',
                'description': 'Sistema de recuperación aumentada con embeddings semánticos, búsqueda vectorial dual (FAISS/ChromaDB) y chunking inteligente.',
                'color': 'primary',
                'completion': '95%'
            },
            {
                'icon': 'fa-balance-scale',
                'title': 'Comparación Empírica',
                'description': 'Framework de benchmarking científico para evaluación objetiva entre modelos locales (Ollama) y comerciales (OpenAI).',
                'color': 'success',
                'completion': '90%'
            },
            {
                'icon': 'fa-shield-alt',
                'title': 'Seguridad y Cumplimiento',
                'description': 'Implementación de ENS y CCN-TEC 014. Procesamiento local de datos sensibles con modelos on-premise.',
                'color': 'warning',
                'completion': '85%'
            },
            {
                'icon': 'fa-cogs',
                'title': 'Arquitectura Modular',
                'description': 'Diseño hexagonal escalable con componentes intercambiables, configuración YAML y principios SOLID.',
                'color': 'info',
                'completion': '98%'
            },
            {
                'icon': 'fa-chart-line',
                'title': 'Métricas y Observabilidad',
                'description': 'Monitoreo en tiempo real de rendimiento, costes, tokens utilizados y métricas de calidad de respuestas.',
                'color': 'secondary',
                'completion': '80%'
            },
            {
                'icon': 'fa-download',
                'title': 'Ingesta Multimodal',
                'description': 'Procesamiento automático de PDF, DOCX, Excel, CSV, web scraping y APIs REST con pipeline ETL robusto.',
                'color': 'dark',
                'completion': '92%'
            }
        ]
        
        # NUEVA SECCIÓN: Estado del sistema en tiempo real
        system_status = {}
        if RAG_PIPELINE_AVAILABLE and get_rag_pipeline:
            try:
                pipeline = get_rag_pipeline()
                health = pipeline.health_check()
                stats = pipeline.get_stats()
                
                system_status = {
                    'pipeline_available': True,
                    'status': health['status'],
                    'documents_indexed': stats.get('documents_count', 0),
                    'models_available': sum(len(models) for models in stats.get('available_models', {}).values()),
                    'uptime': system_stats.get_uptime_hours()
                }
            except Exception as e:
                logger.error(f"Error obteniendo estado del sistema: {e}")
                system_status = {
                    'pipeline_available': False,
                    'error': str(e)
                }
        else:
            system_status = {
                'pipeline_available': False,
                'status': 'limited'
            }
        
        context = {
            'project_info': project_info,
            'technologies': technologies,
            'features': features,
            'system_status': system_status,
            'pipeline_available': RAG_PIPELINE_AVAILABLE
        }
        
        return render_template('about.html', **context)
        
    except Exception as e:
        logger.error("Error cargando página about", error=str(e))
        flash('Error cargando la información del proyecto', 'error')
        return redirect(url_for('main.index'))

@main_bp.route('/docs')
def documentation():
    """Documentación del sistema MEJORADA"""
    try:
        # Obtener configuraciones para mostrar en la documentación
        model_config = get_model_config()
        rag_config = get_rag_config()
        security_config = get_security_config()
        
        # Estructura de documentación actualizada
        docs_sections = [
            {
                'id': 'quick-start',
                'title': 'Inicio Rápido',
                'icon': 'fa-rocket',
                'description': 'Guía para comenzar a usar el sistema',
                'content': 'setup_guide'
            },
            {
                'id': 'chat-guide',
                'title': 'Guía de Chat',
                'icon': 'fa-comments',
                'description': 'Cómo usar la interfaz de chat y comparación',
                'content': 'chat_interface'
            },
            {
                'id': 'rag-system',
                'title': 'Sistema RAG',
                'icon': 'fa-search',
                'description': 'Funcionamiento de la recuperación aumentada',
                'content': 'rag_architecture'
            },
            {
                'id': 'model-config',
                'title': 'Configuración de Modelos',
                'icon': 'fa-brain',
                'description': 'Parámetros y configuración de LLMs',
                'content': 'model_configuration'
            },
            {
                'id': 'data-ingestion',
                'title': 'Ingesta de Datos',
                'icon': 'fa-download',
                'description': 'Cómo añadir documentos al sistema',
                'content': 'ingestion_guide'
            },
            {
                'id': 'api-reference',
                'title': 'Referencia API',
                'icon': 'fa-code',
                'description': 'Endpoints REST disponibles',
                'content': 'api_documentation'
            },
            {
                'id': 'benchmarking',
                'title': 'Benchmarking',
                'icon': 'fa-chart-bar',
                'description': 'Evaluación y comparación de modelos',
                'content': 'benchmark_guide'
            }
        ]
        
        # API endpoints actualizados con nuevas funcionalidades
        api_endpoints = [
            {
                'method': 'GET',
                'path': '/api/status',
                'description': 'Estado completo del sistema',
                'response': 'JSON con estado de servicios y métricas'
            },
            {
                'method': 'POST',
                'path': '/api/chat/send',
                'description': 'Enviar consulta al chatbot con RAG',
                'params': ['message', 'provider', 'model', 'use_rag', 'k', 'temperature']
            },
            {
                'method': 'POST',
                'path': '/api/chat/compare',
                'description': 'Comparar respuestas de múltiples modelos',
                'params': ['message', 'use_rag', 'k', 'temperature']
            },
            {
                'method': 'POST',
                'path': '/api/chat/rag/search',
                'description': 'Búsqueda semántica en documentos',
                'params': ['query', 'k', 'threshold']
            },
            {
                'method': 'POST',
                'path': '/api/chat/ingest',
                'description': 'Ingestar nuevo documento',
                'params': ['file (multipart/form-data)']
            },
            {
                'method': 'GET',
                'path': '/api/chat/status',
                'description': 'Estado del pipeline RAG y chat',
                'response': 'Estado de componentes y estadísticas'
            }
        ]
        
        # NUEVA SECCIÓN: Información del pipeline si está disponible
        pipeline_info = {}
        if RAG_PIPELINE_AVAILABLE and get_rag_pipeline:
            try:
                pipeline = get_rag_pipeline()
                stats = pipeline.get_stats()
                
                pipeline_info = {
                    'available': True,
                    'vector_store_type': stats.get('vector_store_type', 'Unknown'),
                    'embedding_model': stats.get('embedding_model', 'Unknown'),
                    'documents_count': stats.get('documents_count', 0),
                    'supported_formats': stats.get('supported_formats', []),
                    'configuration': {
                        'chunk_size': stats.get('chunk_size', 500),
                        'overlap': stats.get('chunk_overlap', 50),
                        'similarity_threshold': stats.get('similarity_threshold', 0.5),
                        'default_k': stats.get('default_k', 5)
                    }
                }
            except Exception as e:
                pipeline_info = {
                    'available': False,
                    'error': str(e)
                }
        else:
            pipeline_info = {
                'available': False,
                'status': 'not_loaded'
            }
        
        context = {
            'docs_sections': docs_sections,
            'api_endpoints': api_endpoints,
            'model_config': model_config,
            'rag_config': rag_config,
            'security_config': security_config,
            'pipeline_info': pipeline_info,
            'pipeline_available': RAG_PIPELINE_AVAILABLE
        }
        
        return render_template('docs.html', **context)
        
    except Exception as e:
        logger.error("Error cargando documentación", error=str(e))
        flash('Error cargando la documentación', 'error')
        return redirect(url_for('main.index'))

@main_bp.route('/settings')
def settings():
    """Página de configuración del sistema MEJORADA"""
    try:
        # Verificar configuración actual
        validation_result = validate_configuration()
        
        # Obtener todas las configuraciones
        app_config = get_app_config()
        model_config = get_model_config()
        rag_config = get_rag_config()
        vector_store_config = get_vector_store_config()
        security_config = get_security_config()
        
        # NUEVA LÓGICA: Estado de servicios con pipeline
        if RAG_PIPELINE_AVAILABLE and get_rag_pipeline:
            try:
                pipeline = get_rag_pipeline()
                health = pipeline.health_check()
                stats = pipeline.get_stats()
                
                service_status = {
                    'pipeline': {
                        'available': True,
                        'status': health['status'],
                        'components': health['components']
                    },
                    'rag': {
                        'vector_store_type': stats.get('vector_store_type', 'Unknown'),
                        'documents_count': stats.get('documents_count', 0),
                        'embedding_model': stats.get('embedding_model', 'Unknown')
                    },
                    'llm': {
                        'providers': stats.get('llm_providers', {}),
                        'models': stats.get('available_models', {})
                    },
                    'ingestion': {
                        'supported_formats': stats.get('supported_formats', []),
                        'queue_size': stats.get('ingestion_queue', 0)
                    }
                }
            except Exception as e:
                logger.error(f"Error obteniendo estado de pipeline: {e}")
                service_status = {
                    'pipeline': {'available': False, 'error': str(e)},
                    'rag': {'error': 'Pipeline unavailable'},
                    'llm': {'error': 'Pipeline unavailable'},
                    'ingestion': {'error': 'Pipeline unavailable'}
                }
        else:
            # Fallback a servicios individuales
            try:
                service_status = {
                    'pipeline': {'available': False, 'status': 'not_loaded'},
                    'rag': rag_pipeline.get_stats() if rag_pipeline else {'error': 'Service unavailable'},
                    'llm': llm_service.get_service_stats() if llm_service else {'error': 'Service unavailable'},
                    'ingestion': document_ingestion_service.get_service_stats() if document_ingestion_service else {'error': 'Service unavailable'}
                }
            except Exception as e:
                logger.error(f"Error obteniendo estado de servicios: {e}")
                service_status = {
                    'pipeline': {'available': False},
                    'rag': {'error': str(e)},
                    'llm': {'error': str(e)},
                    'ingestion': {'error': str(e)}
                }
        
        context = {
            'validation_result': validation_result,
            'app_config': app_config,
            'model_config': model_config,
            'rag_config': rag_config,
            'vector_store_config': vector_store_config,
            'security_config': security_config,
            'service_status': service_status,
            'pipeline_available': RAG_PIPELINE_AVAILABLE,
            'config_tips': _get_configuration_tips()
        }
        
        return render_template('settings.html', **context)
        
    except Exception as e:
        logger.error("Error cargando configuración", error=str(e))
        flash('Error cargando la configuración del sistema', 'error')
        return redirect(url_for('main.index'))

# =============================================================================
# NUEVAS RUTAS PARA PIPELINE RAG
# =============================================================================

@main_bp.route('/pipeline/status')
def pipeline_status():
    """Estado detallado del pipeline RAG"""
    try:
        if not RAG_PIPELINE_AVAILABLE or not get_rag_pipeline:
            return jsonify({
                'available': False,
                'error': 'Pipeline RAG no disponible'
            }), 503
        
        pipeline = get_rag_pipeline()
        health = pipeline.health_check()
        stats = pipeline.get_stats()
        
        return jsonify({
            'available': True,
            'health': health,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo estado de pipeline: {e}")
        return jsonify({
            'available': False,
            'error': str(e)
        }), 500

@main_bp.route('/pipeline/rebuild', methods=['POST'])
def rebuild_pipeline():
    """Reconstruir el pipeline RAG"""
    try:
        if not RAG_PIPELINE_AVAILABLE or not get_rag_pipeline:
            return jsonify({
                'success': False,
                'error': 'Pipeline RAG no disponible'
            }), 503
        
        pipeline = get_rag_pipeline()
        
        # Obtener parámetros de reconstrucción
        data = request.get_json() or {}
        source_dir = data.get('source_dir', 'data/documents')
        clear_existing = data.get('clear_existing', True)
        
        logger.info(f"Iniciando reconstrucción de pipeline desde {source_dir}")
        
        # Reconstruir pipeline
        result = pipeline.rebuild_index(
            source_directory=source_dir,
            clear_existing=clear_existing
        )
        
        if result['success']:
            flash('Pipeline RAG reconstruido exitosamente', 'success')
            return jsonify({
                'success': True,
                'message': 'Pipeline reconstruido exitosamente',
                'result': result
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Error desconocido'),
                'result': result
            }), 500
        
    except Exception as e:
        logger.error(f"Error reconstruyendo pipeline: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# =============================================================================
# RUTAS DE PÁGINAS DE GESTIÓN
# =============================================================================

@main_bp.route('/data-sources')
def data_sources():
    """Página de gestión de fuentes de datos"""
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
        
        # Estadísticas globales básicas
        try:
            global_stats = {
                'total_sources': len(document_ingestion_service.list_sources()) if document_ingestion_service else 0,
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

@main_bp.route('/documentos')
def documentos():
    """Página de gestión de documentos"""
    try:
        # Obtener configuraciones
        app_config = get_app_config()
        
        # Contexto para el template
        context = {
            'app_name': getattr(app_config, 'name', 'Prototipo_chatbot'),
            'app_version': getattr(app_config, 'version', '1.0.0'),
            'app_description': 'Gestión de Fuentes de Documentos',
            'page_title': 'Gestión de Documentos'
        }
        
        logger.info("Página de gestión de documentos accedida")
        return render_template('documentos.html', **context)
        
    except Exception as e:
        logger.error(f"Error cargando página de documentos: {e}")
        flash(f"Error cargando la página: {str(e)}", "error")
        return redirect(url_for('main.index'))

@main_bp.route('/fuentes-datos')
def fuentes_datos():
    """Interfaz central de gestión de fuentes de datos"""
    try:
        # Obtener configuraciones
        app_config = get_app_config()
        
        # Estadísticas globales de todas las fuentes
        try:
            from app.services.document_ingestion_service import document_ingestion_service
            
            # Obtener todas las fuentes
            sources = document_ingestion_service.list_sources()
            all_stats = document_ingestion_service.get_all_stats()
            
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
            
            # Estadísticas por tipo de fuente
            sources_by_type = {
                'documents': len([s for s in sources if s.type.value == 'documents']),
                'web': 0,  # Por implementar
                'api': 0,  # Por implementar
                'database': 0  # Por implementar
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

@main_bp.route('/webs')
def webs():
    """Página de gestión de sitios web para web scraping - CORREGIDA"""
    try:
        app_config = get_app_config()
        
        # CORRECCIÓN: Usar servicio directamente en lugar de variables importadas
        if WEB_INGESTION_AVAILABLE and web_ingestion_service:
            try:
                # Obtener estadísticas reales del servicio
                all_sources = web_ingestion_service.list_sources()
                all_stats = web_ingestion_service.get_all_stats()
                
                web_stats = {
                    'total_sources': len(all_sources),
                    'total_pages': sum(getattr(stats, 'total_pages', 0) for stats in all_stats.values()),
                    'active_scraping': 0,  # TODO: Implementar conteo de tareas activas
                    'last_update': max([getattr(stats, 'last_scan', datetime.min) for stats in all_stats.values()], default=datetime.now())
                }
                
                logger.info("Estadísticas de web scraping obtenidas del servicio")
                
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

@main_bp.route('/debug/routes')
def debug_routes():
    """Debug: mostrar todas las rutas registradas"""
    from flask import current_app
    import urllib.parse
    
    output = []
    for rule in current_app.url_map.iter_rules():
        methods = ','.join(rule.methods - {'HEAD', 'OPTIONS'})
        line = f"{rule.endpoint}: {rule.rule} [{methods}]"
        output.append(line)
    
    output.sort()
    return "<pre>" + "\n".join(output) + "</pre>"

# =============================================================================
# FUNCIONES AUXILIARES MEJORADAS
# =============================================================================

def _get_quick_actions():
    """Obtener acciones rápidas para la página principal MEJORADAS"""
    actions = [
        {
            'title': 'Iniciar Chat RAG',
            'description': 'Conversación con recuperación de documentos',
            'icon': 'fa-comments',
            'url': url_for('chat.chat_interface'),
            'color': 'primary',
            'available': True
        },
        {
            'title': 'Comparar Modelos',
            'description': 'Evaluación paralela Local vs Cloud',
            'icon': 'fa-balance-scale', 
            'url': url_for('chat.chat_interface') + '?mode=compare',
            'color': 'success',
            'available': RAG_PIPELINE_AVAILABLE
        },
        {
            'title': 'Dashboard Métricas',
            'description': 'Monitoreo en tiempo real del sistema',
            'icon': 'fa-chart-line',
            'url': url_for('main.dashboard'),
            'color': 'warning',
            'available': True
        },
        {
            'title': 'Estado Pipeline',
            'description': 'Diagnóstico detallado de componentes',
            'icon': 'fa-heartbeat',
            'url': url_for('main.pipeline_status'),
            'color': 'info',
            'available': RAG_PIPELINE_AVAILABLE
        }
    ]
    
    # Filtrar acciones según disponibilidad
    return [action for action in actions if action['available']]

def _get_recent_activity():
    """Obtener actividad reciente del sistema MEJORADA"""
    activities = [
        {
            'timestamp': datetime.now(),
            'type': 'system',
            'message': 'Sistema iniciado correctamente',
            'icon': 'fa-check-circle',
            'color': 'success'
        }
    ]
    
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

def _get_queries_today():
    """Obtener número de consultas hoy (mejorado)"""
    if RAG_PIPELINE_AVAILABLE and get_rag_pipeline:
        try:
            pipeline = get_rag_pipeline()
            stats = pipeline.get_stats()
            return stats.get('queries_today', 0)
        except Exception:
            pass
    return system_stats.total_queries

def _get_detailed_provider_status():
    """Obtener estado detallado de proveedores MEJORADO"""
    if RAG_PIPELINE_AVAILABLE and get_rag_pipeline:
        try:
            pipeline = get_rag_pipeline()
            stats = pipeline.get_stats()
            
            providers = stats.get('llm_providers', {})
            models = stats.get('available_models', {})
            
            status = {}
            for provider, available in providers.items():
                status[provider] = {
                    'available': available,
                    'models': models.get(provider, []),
                    'model_count': len(models.get(provider, [])),
                    'status': 'healthy' if available else 'error',
                    'requests_today': stats.get(f'{provider}_requests_today', 0),
                    'avg_response_time': stats.get(f'{provider}_avg_response_time', 0),
                    'last_used': stats.get(f'{provider}_last_used', 'Never')
                }
            
            return status
        except Exception as e:
            logger.error(f"Error obteniendo estado de proveedores: {e}")
    
    # Fallback a servicios individuales
    try:
        if llm_service:
            providers = llm_service.get_available_providers()
            models = llm_service.get_available_models()
            
            status = {}
            for provider, available in providers.items():
                status[provider] = {
                    'available': available,
                    'models': models.get(provider, []),
                    'model_count': len(models.get(provider, [])),
                    'status': 'healthy' if available else 'error'
                }
            
            return status
    except Exception:
        pass
    
    return {}

def _get_charts_data():
    """Obtener datos para gráficos del dashboard MEJORADOS"""
    # Datos base por defecto
    base_charts = {
        'usage_over_time': {
            'labels': ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'],
            'datasets': [{
                'label': 'Consultas',
                'data': [12, 19, 15, 25, 22, 8, 14],
                'borderColor': 'rgb(75, 192, 192)',
                'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                'tension': 0.4
            }]
        },
        'model_usage': {
            'labels': ['Ollama Local', 'OpenAI', 'Sin Respuesta'],
            'datasets': [{
                'data': [65, 30, 5],
                'backgroundColor': [
                    'rgba(54, 162, 235, 0.8)',
                    'rgba(255, 99, 132, 0.8)',
                    'rgba(255, 205, 86, 0.8)'
                ]
            }]
        },
        'response_times': {
            'labels': ['< 1s', '1-3s', '3-5s', '> 5s'],
            'datasets': [{
                'data': [40, 35, 20, 5],
                'backgroundColor': [
                    'rgba(75, 192, 192, 0.8)',
                    'rgba(255, 205, 86, 0.8)',
                    'rgba(255, 159, 64, 0.8)',
                    'rgba(255, 99, 132, 0.8)'
                ]
            }]
        }
    }
    
    # Enriquecer con datos reales del pipeline si está disponible
    if RAG_PIPELINE_AVAILABLE and get_rag_pipeline:
        try:
            pipeline = get_rag_pipeline()
            stats = pipeline.get_stats()
            
            # Actualizar gráfico de uso de modelos con datos reales
            providers = stats.get('llm_providers', {})
            if providers:
                provider_names = list(providers.keys())
                provider_usage = []
                
                for provider in provider_names:
                    usage = stats.get(f'{provider}_requests_today', 0)
                    provider_usage.append(usage)
                
                total_usage = sum(provider_usage)
                if total_usage > 0:
                    # Convertir a porcentajes
                    provider_percentages = [(usage / total_usage) * 100 for usage in provider_usage]
                    
                    base_charts['model_usage'] = {
                        'labels': [name.title() for name in provider_names],
                        'datasets': [{
                            'data': provider_percentages,
                            'backgroundColor': [
                                'rgba(54, 162, 235, 0.8)',
                                'rgba(255, 99, 132, 0.8)',
                                'rgba(75, 192, 192, 0.8)',
                                'rgba(255, 205, 86, 0.8)'
                            ][:len(provider_names)]
                        }]
                    }
            
            # Añadir gráfico de documentos por tipo si hay datos
            document_types = stats.get('document_types', {})
            if document_types:
                base_charts['document_types'] = {
                    'labels': list(document_types.keys()),
                    'datasets': [{
                        'data': list(document_types.values()),
                        'backgroundColor': [
                            'rgba(153, 102, 255, 0.8)',
                            'rgba(255, 159, 64, 0.8)',
                            'rgba(255, 99, 132, 0.8)',
                            'rgba(54, 162, 235, 0.8)',
                            'rgba(75, 192, 192, 0.8)'
                        ]
                    }]
                }
            
        except Exception as e:
            logger.debug(f"Error enriqueciendo datos de gráficos: {e}")
    
    return base_charts

def _get_configuration_tips():
    """Obtener consejos de configuración MEJORADOS"""
    tips = [
        {
            'category': 'Pipeline RAG',
            'tips': [
                'El pipeline integrado ofrece mejor rendimiento que servicios separados',
                'Usa FAISS para datasets grandes (>1000 docs), ChromaDB para datasets medianos',
                'Ajusta chunk_size según el tipo de documentos: 300-500 para técnicos, 500-800 para narrativos'
            ]
        },
        {
            'category': 'Modelos LLM',
            'tips': [
                'Ollama LLaMA 3.2:3b es óptimo para respuestas rápidas y precisas',
                'Mistral 7B ofrece mejor comprensión contextual para textos largos',
                'OpenAI GPT-4o-mini tiene la mejor relación calidad/precio para producción',
                'Usa temperature 0.1-0.3 para respuestas técnicas, 0.5-0.7 para creativas'
            ]
        },
        {
            'category': 'Rendimiento',
            'tips': [
                'Incrementa k gradualmente: empezar con 3-5, subir hasta 10 si es necesario',
                'Similarity threshold 0.5-0.7 funciona bien para la mayoría de casos',
                'El vector store FAISS es 3-5x más rápido que ChromaDB en búsquedas',
                'Procesa documentos en lotes de 10-20 para optimal ingestion speed'
            ]
        },
        {
            'category': 'Seguridad',
            'tips': [
                'Usa exclusivamente modelos locales (Ollama) para datos clasificados',
                'Implementa rate limiting por IP: 60 requests/hora para usuarios normales',
                'Revisa logs de acceso semanalmente para detectar patrones anómalos',
                'Configura backup automático del vector store cada 24 horas'
            ]
        }
    ]
    
    # Añadir tips específicos si el pipeline está disponible
    if RAG_PIPELINE_AVAILABLE and get_rag_pipeline:
        try:
            pipeline = get_rag_pipeline()
            stats = pipeline.get_stats()
            
            # Tips personalizados según el estado actual
            if stats.get('documents_count', 0) < 10:
                tips[0]['tips'].append('Considera añadir más documentos para mejorar la cobertura de respuestas')
            
            if stats.get('avg_response_time', 0) > 5:
                tips[2]['tips'].append('Tiempo de respuesta elevado: considera reducir k o usar FAISS en lugar de ChromaDB')
            
            vector_store_type = stats.get('vector_store_type', '')
            if 'chromadb' in vector_store_type.lower():
                tips[2]['tips'].append('ChromaDB activo: ideal para metadata rica y filtros complejos')
            elif 'faiss' in vector_store_type.lower():
                tips[2]['tips'].append('FAISS activo: excelente para búsquedas rápidas en volúmenes grandes')
                
        except Exception as e:
            logger.debug(f"Error generando tips personalizados: {e}")
    
    return tips

# =============================================================================
# HOOKS Y MIDDLEWARE MEJORADOS
# =============================================================================

@main_bp.before_request
def before_main_request():
    """Hook antes de cada request principal MEJORADO"""
    # Actualizar estadísticas básicas
    if request.endpoint and 'main' in request.endpoint:
        system_stats.last_updated = datetime.now()
        
        # Si el pipeline está disponible, sincronizar estadísticas
        if RAG_PIPELINE_AVAILABLE and get_rag_pipeline:
            try:
                pipeline = get_rag_pipeline()
                stats = pipeline.get_stats()
                
                # Sincronizar contadores
                system_stats.total_queries = stats.get('total_queries', system_stats.total_queries)
                system_stats.documents_indexed = stats.get('documents_count', system_stats.documents_indexed)
                
            except Exception as e:
                logger.debug(f"Error sincronizando estadísticas: {e}")

@main_bp.after_request
def after_main_request(response):
    """Hook después de cada request principal MEJORADO"""
    # Añadir headers de cache para recursos estáticos
    if request.endpoint and 'static' in request.endpoint:
        response.cache_control.max_age = 3600  # 1 hora
    
    # Añadir header de estado del pipeline
    if RAG_PIPELINE_AVAILABLE:
        response.headers['X-Pipeline-Status'] = 'available'
    else:
        response.headers['X-Pipeline-Status'] = 'unavailable'
    
    return response

# =============================================================================
# ENDPOINTS AJAX MEJORADOS
# =============================================================================

@main_bp.route('/ajax/system-health')
def ajax_system_health():
    """Estado de salud del sistema (AJAX) MEJORADO"""
    try:
        health_data = {
            'timestamp': time.time(),
            'overall': 'unknown',
            'services': {}
        }
        
        if RAG_PIPELINE_AVAILABLE and get_rag_pipeline:
            pipeline = get_rag_pipeline()
            health = pipeline.health_check()
            
            health_data.update({
                'overall': health['status'],
                'services': {
                    'pipeline': health['status'],
                    'rag': health['components'].get('vector_store', 'unknown'),
                    'llm': health['components'].get('llm', 'unknown'),
                    'embeddings': health['components'].get('embeddings', 'unknown')
                },
                'pipeline_available': True
            })
        else:
            # Verificar servicios individuales
            try:
                rag_available = rag_pipeline and rag_pipeline.get_stats().get('total_documents', 0) > 0
                llm_available = llm_service and any(llm_service.get_available_providers().values())
                
                health_data.update({
                    'overall': 'healthy' if (rag_available and llm_available) else 'warning',
                    'services': {
                        'rag': 'healthy' if rag_available else 'warning',
                        'llm': 'healthy' if llm_available else 'error',
                        'ingestion': 'healthy'
                    },
                    'pipeline_available': False
                })
            except Exception:
                health_data.update({
                    'overall': 'error',
                    'services': {'rag': 'error', 'llm': 'error', 'ingestion': 'error'},
                    'pipeline_available': False
                })
        
        return jsonify(health_data)
        
    except Exception as e:
        logger.error("Error obteniendo salud del sistema", error=str(e))
        return jsonify({
            'error': str(e),
            'overall': 'error',
            'timestamp': time.time()
        }), 500

@main_bp.route('/ajax/quick-stats')
def ajax_quick_stats():
    """Estadísticas rápidas (AJAX) MEJORADAS"""
    try:
        stats_data = {
            'timestamp': time.time(),
            'documents': 0,
            'queries': 0,
            'uptime': 0,
            'avg_response_time': 0,
            'pipeline_available': RAG_PIPELINE_AVAILABLE
        }
        
        if RAG_PIPELINE_AVAILABLE and get_rag_pipeline:
            pipeline = get_rag_pipeline()
            pipeline_stats = pipeline.get_stats()
            
            stats_data.update({
                'documents': pipeline_stats.get('documents_count', 0),
                'queries': pipeline_stats.get('total_queries', 0),
                'queries_today': pipeline_stats.get('queries_today', 0),
                'uptime': round(system_stats.get_uptime_hours(), 1),
                'avg_response_time': round(pipeline_stats.get('avg_response_time', 0), 2),
                'vector_store_type': pipeline_stats.get('vector_store_type', 'Unknown'),
                'active_models': len(pipeline_stats.get('available_models', {})),
                'memory_usage': pipeline_stats.get('memory_usage_mb', 0)
            })
        else:
            # Usar servicios individuales
            try:
                rag_stats = rag_pipeline.get_stats() if rag_pipeline else {}
                
                stats_data.update({
                    'documents': rag_stats.get('total_documents', 0),
                    'queries': system_stats.total_queries,
                    'uptime': round(system_stats.get_uptime_hours(), 1),
                    'avg_response_time': round(system_stats.avg_response_time, 2)
                })
            except Exception:
                pass
        
        return jsonify(stats_data)
        
    except Exception as e:
        logger.error("Error obteniendo estadísticas rápidas", error=str(e))
        return jsonify({
            'error': str(e),
            'timestamp': time.time()
        }), 500

@main_bp.route('/ajax/pipeline-metrics')
def ajax_pipeline_metrics():
    """Métricas específicas del pipeline (AJAX) - NUEVA FUNCIONALIDAD"""
    try:
        if not RAG_PIPELINE_AVAILABLE or not get_rag_pipeline:
            return jsonify({
                'available': False,
                'error': 'Pipeline no disponible'
            }), 503
        
        pipeline = get_rag_pipeline()
        stats = pipeline.get_stats()
        health = pipeline.health_check()
        
        metrics = {
            'available': True,
            'status': health['status'],
            'components': health['components'],
            'performance': {
                'documents_count': stats.get('documents_count', 0),
                'avg_response_time': round(stats.get('avg_response_time', 0), 2),
                'total_queries': stats.get('total_queries', 0),
                'queries_today': stats.get('queries_today', 0),
                'success_rate': round(stats.get('success_rate', 0), 1)
            },
            'storage': {
                'vector_store_type': stats.get('vector_store_type', 'Unknown'),
                'vector_store_size_mb': round(stats.get('vector_store_size_mb', 0), 2),
                'memory_usage_mb': round(stats.get('memory_usage_mb', 0), 2),
                'embedding_dimensions': stats.get('embedding_dimensions', 0)
            },
            'models': {
                'providers': stats.get('llm_providers', {}),
                'available_models': stats.get('available_models', {}),
                'embedding_model': stats.get('embedding_model', 'Unknown')
            },
            'timestamp': time.time()
        }
        
        return jsonify(metrics)
        
    except Exception as e:
        logger.error(f"Error obteniendo métricas de pipeline: {e}")
        return jsonify({
            'available': False,
            'error': str(e)
        }), 500