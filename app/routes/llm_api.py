# app/routes/llm_api.py
"""
API Routes para LLM Service
TFM Vicente Caruncho - Sistemas Inteligentes UJI
"""

from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
import time
from typing import Dict, Any

from app.services.llm import llm_service, LLMRequest, LLMResponse, ComparisonResult
from app.core.logger import get_logger
from app.utils.validators import validate_json_request

# Crear blueprint
llm_bp = Blueprint('llm_api', __name__, url_prefix='/api/llm')
logger = get_logger("prototipo_chatbot.llm_api")

# Rate limiting simple (en producción usar Redis)
REQUEST_COUNTS = {}
RATE_LIMIT = 100  # requests per minute
RATE_WINDOW = 60  # seconds

def check_rate_limit(client_ip: str) -> bool:
    """Verificar límite de tasa de solicitudes"""
    now = time.time()
    window_start = now - RATE_WINDOW
    
    if client_ip not in REQUEST_COUNTS:
        REQUEST_COUNTS[client_ip] = []
    
    # Limpiar solicitudes antiguas
    REQUEST_COUNTS[client_ip] = [
        req_time for req_time in REQUEST_COUNTS[client_ip] 
        if req_time > window_start
    ]
    
    # Verificar límite
    if len(REQUEST_COUNTS[client_ip]) >= RATE_LIMIT:
        return False
    
    # Agregar solicitud actual
    REQUEST_COUNTS[client_ip].append(now)
    return True

@llm_bp.before_request
def before_request():
    """Middleware para todas las rutas LLM"""
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
    
    if not check_rate_limit(client_ip):
        logger.warning(f"Rate limit excedido para IP: {client_ip}")
        return jsonify({
            'error': 'Rate limit exceeded',
            'message': f'Máximo {RATE_LIMIT} solicitudes por minuto'
        }), 429

@llm_bp.route('/health', methods=['GET'])
def health_check():
    """Health check del servicio LLM"""
    try:
        health = llm_service.health_check()
        status_code = 200 if health['status'] == 'healthy' else 503
        
        logger.info("Health check realizado", status=health['status'])
        return jsonify(health), status_code
        
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': time.time()
        }), 500

@llm_bp.route('/status', methods=['GET'])
def service_status():
    """Estado detallado del servicio"""
    try:
        stats = llm_service.get_service_stats()
        
        logger.info("Estadísticas del servicio consultadas")
        return jsonify({
            'success': True,
            'data': stats,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error obteniendo estado del servicio: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@llm_bp.route('/models', methods=['GET'])
def get_available_models():
    """Obtener modelos disponibles por proveedor"""
    try:
        provider = request.args.get('provider')
        models = llm_service.get_available_models(provider)
        
        logger.info("Modelos disponibles consultados", provider=provider)
        return jsonify({
            'success': True,
            'data': {
                'models': models,
                'providers': llm_service.get_available_providers()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error obteniendo modelos: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@llm_bp.route('/generate', methods=['POST'])
def generate_response():
    """Generar respuesta con un proveedor específico"""
    try:
        # Validar JSON de entrada
        data = validate_json_request(request)
        
        # Parámetros requeridos
        query = data.get('query', '').strip()
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query es requerido'
            }), 400
        
        # Parámetros opcionales
        provider = data.get('provider', 'ollama')
        model = data.get('model')
        context = data.get('context')
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 1.0)
        max_tokens = data.get('max_tokens', 1000)
        
        # Crear solicitud
        try:
            llm_request = LLMRequest(
                query=query,
                context=context,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f'Parámetros inválidos: {str(e)}'
            }), 400
        
        # Generar respuesta
        start_time = time.time()
        response = llm_service.generate_response(llm_request, provider, model)
        total_time = time.time() - start_time
        
        # Log del resultado
        logger.info(
            "Respuesta generada exitosamente",
            provider=provider,
            model=response.model,
            query_length=len(query),
            response_length=len(response.response),
            total_time=f"{total_time:.2f}s",
            tokens=response.total_tokens,
            cost=response.estimated_cost
        )
        
        return jsonify({
            'success': True,
            'data': response.to_dict(),
            'metadata': {
                'total_api_time': total_time,
                'timestamp': datetime.now().isoformat()
            }
        }), 200
        
    except RuntimeError as e:
        logger.error(f"Error de runtime en generación: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 503
        
    except Exception as e:
        logger.error(f"Error inesperado en generación: {e}")
        return jsonify({
            'success': False,
            'error': 'Error interno del servidor'
        }), 500

@llm_bp.route('/compare', methods=['POST'])
def compare_models():
    """Comparar respuestas entre Ollama y OpenAI"""
    try:
        # Validar JSON de entrada
        data = validate_json_request(request)
        
        # Parámetros requeridos
        query = data.get('query', '').strip()
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query es requerido'
            }), 400
        
        # Parámetros opcionales
        context = data.get('context')
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 1.0)
        max_tokens = data.get('max_tokens', 1000)
        ollama_model = data.get('ollama_model', 'llama3.2:3b')
        openai_model = data.get('openai_model', 'gpt-4o-mini')
        
        # Crear solicitud
        try:
            llm_request = LLMRequest(
                query=query,
                context=context,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f'Parámetros inválidos: {str(e)}'
            }), 400
        
        # Ejecutar comparación
        start_time = time.time()
        comparison = llm_service.compare_models(
            llm_request, 
            ollama_model, 
            openai_model
        )
        total_time = time.time() - start_time
        
        # Log de la comparación
        logger.info(
            "Comparación completada",
            ollama_model=ollama_model,
            openai_model=openai_model,
            query_length=len(query),
            total_time=f"{total_time:.2f}s",
            ollama_success=comparison.ollama_response.success,
            openai_success=comparison.openai_response.success,
            speed_winner=comparison.winner_speed
        )
        
        return jsonify({
            'success': True,
            'data': comparison.to_dict(),
            'metadata': {
                'total_api_time': total_time,
                'timestamp': datetime.now().isoformat()
            }
        }), 200
        
    except RuntimeError as e:
        logger.error(f"Error de runtime en comparación: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 503
        
    except Exception as e:
        logger.error(f"Error inesperado en comparación: {e}")
        return jsonify({
            'success': False,
            'error': 'Error interno del servidor'
        }), 500

@llm_bp.route('/chat', methods=['POST'])
def chat_with_rag():
    """Chat conversacional con contexto RAG"""
    try:
        # Validar JSON de entrada
        data = validate_json_request(request)
        
        # Parámetros requeridos
        message = data.get('message', '').strip()
        if not message:
            return jsonify({
                'success': False,
                'error': 'Message es requerido'
            }), 400
        
        # Parámetros opcionales
        provider = data.get('provider', 'ollama')
        model = data.get('model')
        session_id = data.get('session_id', 'default')
        use_rag = data.get('use_rag', True)
        
        # TODO: Integrar con vector stores para RAG real
        context = None
        if use_rag:
            # Por ahora, contexto de ejemplo
            # En el futuro: context = vector_store.search(message, top_k=3)
            context = data.get('context')
        
        # Crear solicitud
        llm_request = LLMRequest(
            query=message,
            context=context,
            temperature=0.3,  # Más determinista para chat
            max_tokens=800
        )
        
        # Generar respuesta
        response = llm_service.generate_response(llm_request, provider, model)
        
        # TODO: Guardar en historial de chat
        # chat_history.save_interaction(session_id, message, response)
        
        logger.info(
            "Mensaje de chat procesado",
            provider=provider,
            model=response.model,
            session_id=session_id,
            use_rag=use_rag,
            message_length=len(message)
        )
        
        return jsonify({
            'success': True,
            'data': {
                'message': response.response,
                'model_info': {
                    'provider': response.provider,
                    'model': response.model
                },
                'metadata': {
                    'response_time': response.response_time,
                    'tokens': response.total_tokens,
                    'cost': response.estimated_cost,
                    'sources': response.sources
                },
                'session_id': session_id
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error en chat: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@llm_bp.route('/benchmark', methods=['POST'])
def run_benchmark():
    """Ejecutar benchmark de modelos (para análisis TFM)"""
    try:
        data = validate_json_request(request)
        
        # Parámetros del benchmark
        queries = data.get('queries', [])
        if not queries:
            # Usar queries de prueba por defecto
            from app.models.llm_models import generate_tfm_test_queries
            queries = generate_tfm_test_queries()[:5]  # Limitar para demo
        
        providers = data.get('providers', ['ollama', 'openai'])
        iterations = data.get('iterations', 1)
        
        # Ejecutar benchmark
        results = {}
        
        for provider in providers:
            if provider not in llm_service.get_available_providers():
                continue
                
            provider_results = []
            
            for iteration in range(iterations):
                for query in queries:
                    try:
                        llm_request = LLMRequest(query=query, temperature=0.3)
                        response = llm_service.generate_response(llm_request, provider)
                        provider_results.append(response.to_dict())
                    except Exception as e:
                        logger.warning(f"Error en benchmark {provider}: {e}")
            
            results[provider] = provider_results
        
        logger.info(
            "Benchmark completado",
            providers=list(results.keys()),
            total_queries=len(queries),
            iterations=iterations
        )
        
        return jsonify({
            'success': True,
            'data': {
                'results': results,
                'test_info': {
                    'queries': queries,
                    'providers': providers,
                    'iterations': iterations,
                    'timestamp': datetime.now().isoformat()
                }
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error en benchmark: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Error handlers específicos del blueprint
@llm_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint no encontrado',
        'available_endpoints': [
            '/api/llm/health',
            '/api/llm/status', 
            '/api/llm/models',
            '/api/llm/generate',
            '/api/llm/compare',
            '/api/llm/chat',
            '/api/llm/benchmark'
        ]
    }), 404

@llm_bp.errorhandler(500)
def internal_error(error):
    logger.error(f"Error interno en LLM API: {error}")
    return jsonify({
        'success': False,
        'error': 'Error interno del servidor'
    }), 500

# Documentación de la API (endpoint adicional)
@llm_bp.route('/docs', methods=['GET'])
def api_documentation():
    """Documentación de la API LLM"""
    docs = {
        'title': 'LLM Service API - TFM Vicente Caruncho',
        'version': '1.0.0',
        'description': 'API para gestión de modelos de lenguaje locales y cloud',
        'endpoints': {
            'GET /api/llm/health': {
                'description': 'Health check del servicio',
                'response': 'Estado de salud de todos los proveedores'
            },
            'GET /api/llm/status': {
                'description': 'Estado detallado y métricas del servicio',
                'response': 'Estadísticas completas de uso'
            },
            'GET /api/llm/models': {
                'description': 'Modelos disponibles por proveedor',
                'parameters': {
                    'provider': 'Filtrar por proveedor específico (opcional)'
                }
            },
            'POST /api/llm/generate': {
                'description': 'Generar respuesta con un modelo específico',
                'required': ['query'],
                'optional': ['provider', 'model', 'context', 'temperature', 'top_p', 'max_tokens']
            },
            'POST /api/llm/compare': {
                'description': 'Comparar respuestas entre Ollama y OpenAI',
                'required': ['query'],
                'optional': ['context', 'temperature', 'ollama_model', 'openai_model']
            },
            'POST /api/llm/chat': {
                'description': 'Chat conversacional con RAG',
                'required': ['message'],
                'optional': ['provider', 'model', 'session_id', 'use_rag', 'context']
            },
            'POST /api/llm/benchmark': {
                'description': 'Ejecutar benchmark de rendimiento',
                'optional': ['queries', 'providers', 'iterations']
            }
        },
        'examples': {
            'generate_request': {
                'query': '¿Cómo solicitar una licencia de obras?',
                'provider': 'ollama',
                'model': 'llama3.2:3b',
                'temperature': 0.7,
                'max_tokens': 500
            },
            'compare_request': {
                'query': '¿Cuál es el horario de atención?',
                'context': [
                    {
                        'content': 'El horario de atención es de 9:00 a 14:00 horas',
                        'source': 'Información municipal'
                    }
                ],
                'temperature': 0.3,
                'ollama_model': 'llama3.2:3b',
                'openai_model': 'gpt-4o-mini'
            }
        }
    }
    
    return jsonify(docs), 200