"""
API REST endpoints para Prototipo_chatbot
TFM Vicente Caruncho - Sistemas Inteligentes
"""

from flask import Blueprint, request, jsonify, g
from functools import wraps
import time
from typing import Dict, Any, List

# Imports locales
from app.core.logger import get_logger
from app.core.config import get_security_config
from app.services.rag import rag_service, search_documents
from app.services.llm_service import llm_service, generate_llm_response, compare_llm_responses
from app.services.ingestion import ingestion_service
from app.models import ChatMessage, create_chat_message, ModelEncoder

# Crear blueprint
api_bp = Blueprint('api', __name__)
logger = get_logger("api")

# Decorador para rate limiting básico
def rate_limit(max_requests: int = 60):
    """Decorador simple de rate limiting"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Implementación básica - en producción usar Redis
            client_ip = request.remote_addr
            
            # Por simplicidad, solo loggear
            logger.debug("API request", 
                        endpoint=request.endpoint,
                        ip=client_ip,
                        user_agent=request.headers.get('User-Agent'))
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Decorador para validación de entrada
def validate_json(*required_fields):
    """Decorador para validar campos requeridos en JSON"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({
                    'error': 'Content-Type debe ser application/json'
                }), 400
            
            data = request.get_json()
            if not data:
                return jsonify({
                    'error': 'Body JSON requerido'
                }), 400
            
            # Verificar campos requeridos
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return jsonify({
                    'error': f'Campos requeridos faltantes: {", ".join(missing_fields)}'
                }), 400
            
            # Añadir datos validados al objeto g
            g.json_data = data
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# =============================================================================
# ENDPOINTS DE INFORMACIÓN DEL SISTEMA
# =============================================================================

@api_bp.route('/status', methods=['GET'])
@rate_limit()
def system_status():
    """Estado detallado del sistema"""
    try:
        # Verificar servicios
        rag_stats = rag_service.get_stats()
        llm_stats = llm_service.get_service_stats()
        ingestion_stats = ingestion_service.get_service_stats()
        
        # Determinar estado general
        providers_available = llm_stats.get('providers_available', {})
        has_available_provider = any(providers_available.values())
        has_documents = rag_stats.get('total_documents', 0) > 0
        
        overall_status = 'healthy' if has_available_provider else 'warning'
        
        return jsonify({
            'overall_status': overall_status,
            'timestamp': time.time(),
            'services': {
                'rag': {
                    'status': 'healthy' if has_documents else 'warning',
                    'total_documents': rag_stats.get('total_documents', 0),
                    'vector_store': rag_stats.get('vector_store_type', 'unknown'),
                    'embedding_model': rag_stats.get('embedding_model', 'unknown')
                },
                'llm': {
                    'status': 'healthy' if has_available_provider else 'error',
                    'providers_available': providers_available,
                    'total_providers': llm_stats.get('total_providers', 0),
                    'available_providers': llm_stats.get('available_providers', 0)
                },
                'ingestion': {
                    'status': 'healthy',
                    'processors_available': ingestion_stats.get('processors_available', 0),
                    'jobs_active': ingestion_stats.get('jobs_active', 0),
                    'jobs_completed': ingestion_stats.get('jobs_completed', 0)
                }
            }
        })
        
    except Exception as e:
        logger.error("Error obteniendo estado del sistema", error=str(e))
        return jsonify({
            'overall_status': 'error',
            'error': str(e),
            'timestamp': time.time()
        }), 500

@api_bp.route('/models', methods=['GET'])
@rate_limit()
def available_models():
    """Obtener modelos disponibles"""
    try:
        models = llm_service.get_available_models()
        providers = llm_service.get_available_providers()
        
        return jsonify({
            'providers': providers,
            'models': models,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error("Error obteniendo modelos disponibles", error=str(e))
        return jsonify({'error': str(e)}), 500

@api_bp.route('/stats', methods=['GET'])
@rate_limit()
def system_stats():
    """Estadísticas completas del sistema"""
    try:
        return jsonify({
            'rag': rag_service.get_stats(),
            'llm': llm_service.get_service_stats(),
            'ingestion': ingestion_service.get_service_stats(),
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error("Error obteniendo estadísticas", error=str(e))
        return jsonify({'error': str(e)}), 500

# =============================================================================
# ENDPOINTS DE CHAT Y LLM
# =============================================================================

@api_bp.route('/chat/query', methods=['POST'])
@rate_limit()
@validate_json('query')
def chat_query():
    """Procesar consulta de chat con RAG"""
    try:
        data = g.json_data
        query = data['query'].strip()
        
        if not query:
            return jsonify({'error': 'Query no puede estar vacío'}), 400
        
        # Validar longitud de consulta
        security_config = get_security_config()
        if len(query) > security_config.max_query_length:
            return jsonify({
                'error': f'Query demasiado largo (máximo {security_config.max_query_length} caracteres)'
            }), 400
        
        # Parámetros opcionales
        provider = data.get('provider', 'ollama')
        model_name = data.get('model_name')
        use_rag = data.get('use_rag', True)
        rag_k = data.get('rag_k', 5)
        
        # Parámetros de generación
        generation_params = {
            'temperature': data.get('temperature', 0.7),
            'max_tokens': data.get('max_tokens', 1000),
            'top_p': data.get('top_p'),
            'top_k': data.get('top_k')
        }
        
        # Filtrar parámetros None
        generation_params = {k: v for k, v in generation_params.items() if v is not None}
        
        start_time = time.time()
        
        # Búsqueda RAG si está habilitada
        context_chunks = []
        if use_rag:
            context_chunks = search_documents(query, k=rag_k)
            logger.info("Contexto RAG obtenido",
                       query_length=len(query),
                       chunks_found=len(context_chunks))
        
        # Generar respuesta
        response = generate_llm_response(
            query=query,
            provider=provider,
            model_name=model_name,
            context=context_chunks,
            **generation_params
        )
        
        # Preparar respuesta RAG para el cliente
        rag_sources = []
        if context_chunks:
            for chunk in context_chunks:
                rag_sources.append({
                    'id': chunk.id,
                    'content': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    'source_path': chunk.metadata.source_path,
                    'source_type': chunk.metadata.source_type,
                    'chunk_index': chunk.chunk_index,
                    'section_title': chunk.section_title
                })
        
        total_time = time.time() - start_time
        
        result = {
            'success': response.success,
            'content': response.content,
            'model': {
                'name': response.model_name,
                'type': response.model_type,
                'response_time': response.response_time,
                'tokens_used': response.tokens_used,
                'prompt_tokens': response.prompt_tokens,
                'completion_tokens': response.completion_tokens
            },
            'rag': {
                'enabled': use_rag,
                'sources_used': rag_sources,
                'total_sources': len(context_chunks)
            },
            'request': {
                'query': query,
                'provider': provider,
                'generation_params': generation_params,
                'total_time': total_time
            },
            'timestamp': time.time()
        }
        
        if not response.success:
            result['error'] = response.error
            logger.warning("Chat query failed",
                          query_length=len(query),
                          provider=provider,
                          error=response.error)
            return jsonify(result), 500
        
        logger.info("Chat query completado",
                   query_length=len(query),
                   provider=provider,
                   model=response.model_name,
                   success=response.success,
                   total_time=total_time)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error("Error procesando consulta de chat", error=str(e))
        return jsonify({'error': str(e)}), 500

@api_bp.route('/chat/compare', methods=['POST'])
@rate_limit()
@validate_json('query')
def chat_compare():
    """Comparar respuestas de múltiples modelos"""
    try:
        data = g.json_data
        query = data['query'].strip()
        
        if not query:
            return jsonify({'error': 'Query no puede estar vacío'}), 400
        
        # Parámetros
        use_rag = data.get('use_rag', True)
        rag_k = data.get('rag_k', 5)
        generation_params = {
            'temperature': data.get('temperature', 0.7),
            'max_tokens': data.get('max_tokens', 1000),
            'top_p': data.get('top_p'),
            'top_k': data.get('top_k')
        }
        generation_params = {k: v for k, v in generation_params.items() if v is not None}
        
        start_time = time.time()
        
        # Búsqueda RAG
        context_chunks = []
        if use_rag:
            context_chunks = search_documents(query, k=rag_k)
        
        # Generar respuestas comparativas
        responses = compare_llm_responses(
            query=query,
            context=context_chunks,
            **generation_params
        )
        
        # Preparar respuesta
        result = {
            'query': query,
            'responses': {},
            'comparison': {
                'providers_compared': len(responses),
                'total_time': time.time() - start_time
            },
            'rag': {
                'enabled': use_rag,
                'sources_count': len(context_chunks)
            },
            'timestamp': time.time()
        }
        
        # Añadir respuestas de cada proveedor
        for provider, response in responses.items():
            result['responses'][provider] = {
                'success': response.success,
                'content': response.content,
                'model_name': response.model_name,
                'response_time': response.response_time,
                'tokens_used': response.tokens_used,
                'error': response.error if not response.success else None
            }
        
        # Métricas de comparación básicas
        if len(responses) >= 2:
            successful_responses = [r for r in responses.values() if r.success]
            if len(successful_responses) >= 2:
                response_times = [r.response_time for r in successful_responses]
                result['comparison']['fastest_provider'] = min(responses.keys(), 
                    key=lambda k: responses[k].response_time if responses[k].success else float('inf'))
                result['comparison']['time_difference'] = max(response_times) - min(response_times)
        
        logger.info("Comparación de modelos completada",
                   query_length=len(query),
                   providers=list(responses.keys()),
                   successful_responses=len([r for r in responses.values() if r.success]))
        
        return jsonify(result)
        
    except Exception as e:
        logger.error("Error en comparación de modelos", error=str(e))
        return jsonify({'error': str(e)}), 500

# =============================================================================
# ENDPOINTS DE RAG Y BÚSQUEDA
# =============================================================================

@api_bp.route('/rag/search', methods=['POST'])
@rate_limit()
@validate_json('query')
def rag_search():
    """Búsqueda semántica en el sistema RAG"""
    try:
        data = g.json_data
        query = data['query'].strip()
        
        if not query:
            return jsonify({'error': 'Query no puede estar vacío'}), 400
        
        # Parámetros de búsqueda
        k = min(data.get('k', 5), 20)  # Máximo 20 resultados
        threshold = data.get('threshold', 0.7)
        
        start_time = time.time()
        
        # Realizar búsqueda
        results = rag_service.search_with_scores(query, k=k, threshold=threshold)
        
        search_time = time.time() - start_time
        
        # Preparar resultados
        search_results = []
        for chunk, score in results:
            search_results.append({
                'id': chunk.id,
                'content': chunk.content,
                'score': score,
                'metadata': {
                    'source_path': chunk.metadata.source_path,
                    'source_type': chunk.metadata.source_type,
                    'file_type': chunk.metadata.file_type,
                    'chunk_index': chunk.chunk_index,
                    'section_title': chunk.section_title,
                    'page_number': chunk.page_number
                }
            })
        
        return jsonify({
            'query': query,
            'results': search_results,
            'total_results': len(search_results),
            'search_params': {
                'k': k,
                'threshold': threshold
            },
            'search_time': search_time,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error("Error en búsqueda RAG", error=str(e))
        return jsonify({'error': str(e)}), 500

@api_bp.route('/rag/stats', methods=['GET'])
@rate_limit()
def rag_stats():
    """Estadísticas del sistema RAG"""
    try:
        stats = rag_service.get_stats()
        return jsonify(stats)
        
    except Exception as e:
        logger.error("Error obteniendo estadísticas RAG", error=str(e))
        return jsonify({'error': str(e)}), 500

# =============================================================================
# ENDPOINTS DE INGESTA
# =============================================================================

@api_bp.route('/ingestion/jobs', methods=['GET'])
@rate_limit()
def get_ingestion_jobs():
    """Obtener trabajos de ingesta activos"""
    try:
        jobs = ingestion_service.get_active_jobs()
        jobs_data = [job.to_dict() for job in jobs]
        
        return jsonify({
            'jobs': jobs_data,
            'total_jobs': len(jobs_data),
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error("Error obteniendo trabajos de ingesta", error=str(e))
        return jsonify({'error': str(e)}), 500

@api_bp.route('/ingestion/jobs/<job_id>', methods=['GET'])
@rate_limit()
def get_ingestion_job(job_id: str):
    """Obtener estado de un trabajo específico"""
    try:
        job = ingestion_service.get_job_status(job_id)
        
        if not job:
            return jsonify({'error': 'Trabajo no encontrado'}), 404
        
        return jsonify(job.to_dict())
        
    except Exception as e:
        logger.error("Error obteniendo trabajo de ingesta", job_id=job_id, error=str(e))
        return jsonify({'error': str(e)}), 500

@api_bp.route('/ingestion/jobs/<job_id>/cancel', methods=['POST'])
@rate_limit()
def cancel_ingestion_job(job_id: str):
    """Cancelar un trabajo de ingesta"""
    try:
        success = ingestion_service.cancel_job(job_id)
        
        if not success:
            return jsonify({'error': 'No se pudo cancelar el trabajo'}), 400
        
        return jsonify({
            'success': True,
            'message': 'Trabajo cancelado exitosamente',
            'job_id': job_id
        })
        
    except Exception as e:
        logger.error("Error cancelando trabajo de ingesta", job_id=job_id, error=str(e))
        return jsonify({'error': str(e)}), 500

@api_bp.route('/ingestion/supported-formats', methods=['GET'])
@rate_limit()
def supported_formats():
    """Obtener formatos de archivo soportados"""
    try:
        formats = ingestion_service.get_supported_formats()
        return jsonify(formats)
        
    except Exception as e:
        logger.error("Error obteniendo formatos soportados", error=str(e))
        return jsonify({'error': str(e)}), 500

# =============================================================================
# ENDPOINTS DE CONFIGURACIÓN
# =============================================================================

@api_bp.route('/config/models', methods=['GET'])
@rate_limit()
def get_model_config():
    """Obtener configuración de modelos"""
    try:
        from app.core.config import get_model_config
        config = get_model_config()
        
        return jsonify({
            'embedding': {
                'name': config.embedding_name,
                'dimension': config.embedding_dimension,
                'device': config.embedding_device
            },
            'local': {
                'default': config.local_default,
                'available': config.local_available,
                'endpoint': config.local_endpoint,
                'timeout': config.local_timeout
            },
            'openai': {
                'default': config.openai_default,
                'available': config.openai_available,
                'timeout': config.openai_timeout
            }
        })
        
    except Exception as e:
        logger.error("Error obteniendo configuración de modelos", error=str(e))
        return jsonify({'error': str(e)}), 500

@api_bp.route('/config/rag', methods=['GET'])
@rate_limit()
def get_rag_config():
    """Obtener configuración RAG"""
    try:
        from app.core.config import get_rag_config
        config = get_rag_config()
        
        return jsonify({
            'enabled': config.enabled,
            'k_default': config.k_default,
            'k_max': config.k_max,
            'chunk_size': config.chunk_size,
            'chunk_overlap': config.chunk_overlap,
            'similarity_threshold': config.similarity_threshold,
            'retrieval_strategy': config.retrieval_strategy,
            'rerank_enabled': config.rerank_enabled
        })
        
    except Exception as e:
        logger.error("Error obteniendo configuración RAG", error=str(e))
        return jsonify({'error': str(e)}), 500

# =============================================================================
# MANEJO DE ERRORES
# =============================================================================

@api_bp.errorhandler(400)
def bad_request(error):
    """Manejo de errores 400"""
    return jsonify({
        'error': 'Bad Request',
        'message': 'Solicitud malformada',
        'status_code': 400
    }), 400

@api_bp.errorhandler(404)
def not_found(error):
    """Manejo de errores 404"""
    return jsonify({
        'error': 'Not Found',
        'message': 'Endpoint no encontrado',
        'status_code': 404
    }), 404

@api_bp.errorhandler(429)
def too_many_requests(error):
    """Manejo de errores 429"""
    return jsonify({
        'error': 'Too Many Requests',
        'message': 'Límite de rate limit excedido',
        'status_code': 429
    }), 429

@api_bp.errorhandler(500)
def internal_error(error):
    """Manejo de errores 500"""
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'Error interno del servidor',
        'status_code': 500
    }), 500

# Hook para logging de todas las requests API
@api_bp.before_request
def log_api_request():
    """Log de todas las requests API"""
    logger.info("API request iniciado",
               method=request.method,
               endpoint=request.endpoint,
               path=request.path,
               remote_addr=request.remote_addr,
               content_length=request.content_length)

@api_bp.after_request
def log_api_response(response):
    """Log de todas las responses API"""
    if hasattr(g, 'start_time'):
        response_time = time.time() - g.start_time
        logger.info("API request completado",
                   method=request.method,
                   endpoint=request.endpoint,
                   status_code=response.status_code,
                   response_time=response_time)
    
    return response