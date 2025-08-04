"""
Rutas principales para Prototipo_chatbot
TFM Vicente Caruncho - Sistemas Inteligentes
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
from app.services.rag import rag_service
from app.services.llm_service import llm_service
from app.services.ingestion import ingestion_service
from app.models import SystemStats

# Crear blueprint
main_bp = Blueprint('main', __name__)
logger = get_logger("main_routes")

# Estadísticas globales del sistema (en producción usar base de datos)
system_stats = SystemStats()

@main_bp.route('/')
def index():
    """Página principal del sistema"""
    try:
        # Obtener configuraciones
        app_config = get_app_config()
        
        # Verificar estado de servicios
        rag_stats = rag_service.get_stats()
        llm_stats = llm_service.get_service_stats()
        ingestion_stats = ingestion_service.get_service_stats()
        
        # Actualizar estadísticas del sistema
        system_stats.documents_indexed = rag_stats.get('total_documents', 0)
        system_stats.update_indexing_stats(0, 0)  # Solo actualizar timestamp
        
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
        
        # Contexto para el template
        context = {
            'app_config': app_config,
            'system_health': system_health,
            'system_stats': system_stats.to_dict(),
            'rag_stats': rag_stats,
            'llm_stats': llm_stats,
            'ingestion_stats': ingestion_stats,
            'providers_available': providers_available,
            'quick_actions': _get_quick_actions(),
            'recent_activity': _get_recent_activity()
        }
        
        logger.info("Página principal cargada",
                   documents_indexed=system_stats.documents_indexed,
                   providers_available=list(providers_available.keys()),
                   overall_health=system_health['overall'])
        
        return render_template('index.html', **context)
        
    except Exception as e:
        logger.error("Error cargando página principal", error=str(e))
        flash('Error cargando la página principal', 'error')
        return render_template('errors/500.html'), 500

@main_bp.route('/dashboard')
def dashboard():
    """Panel de control y métricas"""
    try:
        # Obtener estadísticas detalladas
        rag_stats = rag_service.get_stats()
        llm_stats = llm_service.get_service_stats()
        ingestion_stats = ingestion_service.get_service_stats()
        
        # Estadísticas de uso
        usage_stats = {
            'queries_today': _get_queries_today(),
            'response_time_avg': system_stats.avg_response_time,
            'success_rate': system_stats.get_success_rate(),
            'uptime_hours': system_stats.get_uptime_hours()
        }
        
        # Métricas de rendimiento
        performance_metrics = {
            'vector_store_size': rag_stats.get('memory_usage_mb', 0),
            'total_documents': rag_stats.get('total_documents', 0),
            'embedding_model': rag_stats.get('embedding_model', 'unknown'),
            'chunk_size': rag_stats.get('chunk_size', 500)
        }
        
        # Estado de proveedores de LLM
        providers_status = _get_detailed_provider_status()
        
        context = {
            'usage_stats': usage_stats,
            'performance_metrics': performance_metrics,
            'providers_status': providers_status,
            'rag_stats': rag_stats,
            'llm_stats': llm_stats,
            'ingestion_stats': ingestion_stats,
            'system_stats': system_stats.to_dict(),
            'charts_data': _get_charts_data()
        }
        
        return render_template('dashboard.html', **context)
        
    except Exception as e:
        logger.error("Error cargando dashboard", error=str(e))
        flash('Error cargando el dashboard', 'error')
        return redirect(url_for('main.index'))

@main_bp.route('/about')
def about():
    """Página de información sobre el proyecto"""
    try:
        app_config = get_app_config()
        
        # Información del proyecto
        project_info = {
            'title': 'Prototipo de Chatbot RAG para Administraciones Locales',
            'subtitle': 'Trabajo Final de Máster - Sistemas Inteligentes',
            'author': 'Vicente Caruncho Ramos',
            'tutor': 'Rafael Berlanga Llavori',
            'university': 'Universitat Jaume I',
            'master': 'Máster en Sistemas Inteligentes',
            'speciality': 'Interacción Avanzada y Gestión del Conocimiento',
            'year': '2025',
            'version': app_config.version
        }
        
        # Tecnologías utilizadas
        technologies = {
            'backend': [
                {'name': 'Python 3.9+', 'description': 'Lenguaje principal'},
                {'name': 'Flask', 'description': 'Framework web'},
                {'name': 'sentence-transformers', 'description': 'Embeddings semánticos'},
                {'name': 'FAISS', 'description': 'Búsqueda vectorial'},
                {'name': 'ChromaDB', 'description': 'Base de datos vectorial'},
            ],
            'ai_models': [
                {'name': 'OpenAI GPT-4', 'description': 'Modelo de lenguaje en la nube'},
                {'name': 'Ollama', 'description': 'Modelos locales (LLaMA, Mistral, Gemma)'},
                {'name': 'all-MiniLM-L6-v2', 'description': 'Modelo de embeddings'},
            ],
            'frontend': [
                {'name': 'Bootstrap 5', 'description': 'Framework CSS'},
                {'name': 'JavaScript ES6+', 'description': 'Interactividad cliente'},
                {'name': 'Chart.js', 'description': 'Visualización de datos'},
            ]
        }
        
        # Características principales
        features = [
            {
                'icon': 'fa-search',
                'title': 'RAG Avanzado',
                'description': 'Recuperación aumentada desde múltiples fuentes: documentos, APIs, web y bases de datos.',
                'color': 'primary'
            },
            {
                'icon': 'fa-balance-scale',
                'title': 'Comparación de Modelos',
                'description': 'Evaluación directa entre modelos locales (Ollama) y comerciales (OpenAI).',
                'color': 'success'
            },
            {
                'icon': 'fa-shield-alt',
                'title': 'Seguridad Local',
                'description': 'Cumplimiento ENS y CCN-TEC 014. Procesamiento local de datos sensibles.',
                'color': 'warning'
            },
            {
                'icon': 'fa-cogs',
                'title': 'Arquitectura Modular',
                'description': 'Diseño escalable con componentes intercambiables y configuración flexible.',
                'color': 'info'
            },
            {
                'icon': 'fa-chart-line',
                'title': 'Métricas y Monitoreo',
                'description': 'Seguimiento detallado de rendimiento, uso de tokens y tiempos de respuesta.',
                'color': 'secondary'
            },
            {
                'icon': 'fa-download',
                'title': 'Ingesta Multimodal',
                'description': 'Procesamiento automático de PDF, DOCX, Excel, CSV y contenido web.',
                'color': 'dark'
            }
        ]
        
        context = {
            'project_info': project_info,
            'technologies': technologies,
            'features': features
        }
        
        return render_template('about.html', **context)
        
    except Exception as e:
        logger.error("Error cargando página about", error=str(e))
        flash('Error cargando la información del proyecto', 'error')
        return redirect(url_for('main.index'))

@main_bp.route('/docs')
def documentation():
    """Documentación del sistema"""
    try:
        # Obtener configuraciones para mostrar en la documentación
        model_config = get_model_config()
        rag_config = get_rag_config()
        security_config = get_security_config()
        
        # Estructura de documentación
        docs_sections = [
            {
                'id': 'quick-start',
                'title': 'Inicio Rápido',
                'icon': 'fa-rocket',
                'description': 'Guía para comenzar a usar el sistema'
            },
            {
                'id': 'chat-guide',
                'title': 'Guía de Chat',
                'icon': 'fa-comments',
                'description': 'Cómo usar la interfaz de chat y comparación'
            },
            {
                'id': 'rag-system',
                'title': 'Sistema RAG',
                'icon': 'fa-search',
                'description': 'Funcionamiento de la recuperación aumentada'
            },
            {
                'id': 'model-config',
                'title': 'Configuración de Modelos',
                'icon': 'fa-brain',
                'description': 'Parámetros y configuración de LLMs'
            },
            {
                'id': 'data-ingestion',
                'title': 'Ingesta de Datos',
                'icon': 'fa-download',
                'description': 'Cómo añadir documentos al sistema'
            },
            {
                'id': 'api-reference',
                'title': 'Referencia API',
                'icon': 'fa-code',
                'description': 'Endpoints REST disponibles'
            }
        ]
        
        # API endpoints para documentación
        api_endpoints = [
            {
                'method': 'GET',
                'path': '/api/status',
                'description': 'Estado del sistema',
                'response': 'JSON con estado de servicios'
            },
            {
                'method': 'POST',
                'path': '/api/chat/query',
                'description': 'Enviar consulta al chatbot',
                'params': ['query', 'provider', 'use_rag']
            },
            {
                'method': 'POST',
                'path': '/api/chat/compare',
                'description': 'Comparar respuestas de modelos',
                'params': ['query', 'use_rag', 'rag_k']
            },
            {
                'method': 'POST',
                'path': '/api/rag/search',
                'description': 'Búsqueda semántica',
                'params': ['query', 'k', 'threshold']
            }
        ]
        
        context = {
            'docs_sections': docs_sections,
            'api_endpoints': api_endpoints,
            'model_config': model_config,
            'rag_config': rag_config,
            'security_config': security_config
        }
        
        return render_template('docs.html', **context)
        
    except Exception as e:
        logger.error("Error cargando documentación", error=str(e))
        flash('Error cargando la documentación', 'error')
        return redirect(url_for('main.index'))

@main_bp.route('/settings')
def settings():
    """Página de configuración del sistema"""
    try:
        # Verificar configuración actual
        validation_result = validate_configuration()
        
        # Obtener todas las configuraciones
        app_config = get_app_config()
        model_config = get_model_config()
        rag_config = get_rag_config()
        vector_store_config = get_vector_store_config()
        security_config = get_security_config()
        
        # Estado de servicios
        service_status = {
            'rag': rag_service.get_stats(),
            'llm': llm_service.get_service_stats(),
            'ingestion': ingestion_service.get_service_stats()
        }
        
        context = {
            'validation_result': validation_result,
            'app_config': app_config,
            'model_config': model_config,
            'rag_config': rag_config,
            'vector_store_config': vector_store_config,
            'security_config': security_config,
            'service_status': service_status,
            'config_tips': _get_configuration_tips()
        }
        
        return render_template('settings.html', **context)
        
    except Exception as e:
        logger.error("Error cargando configuración", error=str(e))
        flash('Error cargando la configuración del sistema', 'error')
        return redirect(url_for('main.index'))

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def _get_quick_actions():
    """Obtener acciones rápidas para la página principal"""
    return [
        {
            'title': 'Iniciar Chat',
            'description': 'Comenzar conversación con el chatbot',
            'icon': 'fa-comments',
            'url': url_for('chat.chat_index'),
            'color': 'primary'
        },
        {
            'title': 'Comparar Modelos',
            'description': 'Evaluar respuestas de diferentes LLMs',
            'icon': 'fa-balance-scale', 
            'url': url_for('chat.chat_index') + '?mode=compare',
            'color': 'success'
        },
        {
            'title': 'Ingesta de Datos',
            'description': 'Añadir documentos al sistema',
            'icon': 'fa-upload',
            'url': '#',  # TODO: Implementar cuando tengamos la ruta
            'color': 'info'
        },
        {
            'title': 'Ver Dashboard',
            'description': 'Métricas y estadísticas del sistema',
            'icon': 'fa-chart-line',
            'url': url_for('main.dashboard'),
            'color': 'warning'
        }
    ]

def _get_recent_activity():
    """Obtener actividad reciente del sistema"""
    # En producción, esto vendría de base de datos
    return [
        {
            'timestamp': datetime.now(),
            'type': 'system',
            'message': 'Sistema iniciado correctamente',
            'icon': 'fa-check-circle',
            'color': 'success'
        },
        {
            'timestamp': datetime.now(),
            'type': 'config',
            'message': f'Documentos indexados: {system_stats.documents_indexed}',
            'icon': 'fa-database',
            'color': 'info'
        }
    ]

def _get_queries_today():
    """Obtener número de consultas hoy (placeholder)"""
    return system_stats.total_queries

def _get_detailed_provider_status():
    """Obtener estado detallado de proveedores"""
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

def _get_charts_data():
    """Obtener datos para gráficos del dashboard"""
    # En producción, estos datos vendrían de métricas históricas
    return {
        'usage_over_time': {
            'labels': ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'],
            'datasets': [{
                'label': 'Consultas',
                'data': [12, 19, 15, 25, 22, 8, 14],
                'borderColor': 'rgb(75, 192, 192)',
                'backgroundColor': 'rgba(75, 192, 192, 0.2)'
            }]
        },
        'model_usage': {
            'labels': ['Ollama Local', 'OpenAI', 'Otros'],
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

def _get_configuration_tips():
    """Obtener consejos de configuración"""
    return [
        {
            'category': 'Modelos',
            'tips': [
                'Para mejor rendimiento local, usa modelos pequeños como Gemma 2B',
                'OpenAI GPT-4 ofrece mejor calidad pero mayor coste',
                'Ajusta la temperatura según el tipo de respuesta deseada'
            ]
        },
        {
            'category': 'RAG',
            'tips': [
                'Chunks más pequeños mejoran precisión, más grandes el contexto',
                'Aumenta k para obtener más contexto, reduce para respuestas focalizadas',
                'Ajusta similarity_threshold según la calidad de tus documentos'
            ]
        },
        {
            'category': 'Seguridad',
            'tips': [
                'Usa modelos locales para datos sensibles',
                'Limita max_query_length para prevenir ataques',
                'Revisa logs regularmente para detectar uso anómalo'
            ]
        }
    ]

# =============================================================================
# HOOKS Y MIDDLEWARE
# =============================================================================

@main_bp.before_request
def before_main_request():
    """Hook antes de cada request principal"""
    # Actualizar estadísticas básicas
    if request.endpoint and 'main' in request.endpoint:
        system_stats.last_updated = datetime.now()

@main_bp.after_request
def after_main_request(response):
    """Hook después de cada request principal"""
    return response

# =============================================================================
# ENDPOINTS AJAX PARA COMPONENTES DINÁMICOS
# =============================================================================

@main_bp.route('/ajax/system-health')
def ajax_system_health():
    """Estado de salud del sistema (AJAX)"""
    try:
        # Verificar servicios rápidamente
        rag_available = rag_service.get_stats().get('total_documents', 0) > 0
        llm_providers = llm_service.get_available_providers()
        llm_available = any(llm_providers.values())
        
        health = {
            'overall': 'healthy' if (rag_available and llm_available) else 'warning',
            'rag': 'healthy' if rag_available else 'warning',
            'llm': 'healthy' if llm_available else 'error',
            'ingestion': 'healthy',
            'timestamp': time.time()
        }
        
        return jsonify(health)
        
    except Exception as e:
        logger.error("Error obteniendo salud del sistema", error=str(e))
        return jsonify({'error': str(e)}), 500

@main_bp.route('/ajax/quick-stats')
def ajax_quick_stats():
    """Estadísticas rápidas (AJAX)"""
    try:
        rag_stats = rag_service.get_stats()
        
        stats = {
            'documents': rag_stats.get('total_documents', 0),
            'queries': system_stats.total_queries,
            'uptime': round(system_stats.get_uptime_hours(), 1),
            'avg_response_time': round(system_stats.avg_response_time, 2),
            'timestamp': time.time()
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error("Error obteniendo estadísticas rápidas", error=str(e))
        return jsonify({'error': str(e)}), 500