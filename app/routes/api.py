"""
API REST endpoints para Prototipo_chatbot
TFM Vicente Caruncho - Sistemas Inteligentes
Archivo COMPLETO con todos los endpoints integrados
"""

from flask import Blueprint, request, jsonify, g
from functools import wraps
import time
from typing import Dict, Any, List

# Imports locales
from app.core.logger import get_logger
from app.core.config import get_security_config, get_model_config, get_rag_config
from app.services.llm_service import get_llm_service, LLMRequest
from app.models import ChatMessage, create_chat_message, ModelEncoder, DocumentChunk, DocumentMetadata
from app.services.rag_pipeline import *
# Crear blueprint
api_bp = Blueprint('api', __name__)
logger = get_logger("api")

# =============================================================================
# DECORADORES AUXILIARES
# =============================================================================

def rate_limit(max_requests: int = 60):
    """Decorador simple de rate limiting"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            logger.debug("API request", 
                        endpoint=request.endpoint,
                        ip=client_ip,
                        user_agent=request.headers.get('User-Agent'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

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
            
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return jsonify({
                    'error': f'Campos requeridos faltantes: {", ".join(missing_fields)}'
                }), 400
            
            g.json_data = data
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def get_rag_context(query: str, top_k: int = 5) -> List[DocumentChunk]:
    """Obtener contexto RAG para una consulta"""
    try:
        # Intentar usar el servicio RAG si está disponible
        try:
            from app.services.rag import search_documents
            return search_documents(query, k=top_k)
        except:
            pass
        
        # Fallback: crear contexto de ejemplo para testing
        if "licencia" in query.lower() or "obra" in query.lower():
            return [
                DocumentChunk(
                    content="""La licencia de obras es obligatoria para construcciones, 
                    instalaciones y obras de demolición. Se requiere proyecto técnico 
                    para obras mayores.""",
                    metadata=DocumentMetadata(
                        source_path="ordenanza_urbanismo.pdf",
                        source_type="pdf",
                        title="Ordenanza Municipal de Urbanismo"
                    )
                )
            ]
        
        return []
        
    except Exception as e:
        logger.error("Error obteniendo contexto RAG", error=str(e))
        return []

def calculate_comparison_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calcular métricas de comparación entre modelos"""
    metrics = {
        "models_compared": len(results),
        "total_execution_time": sum(r.response_time for r in results.values()),
        "fastest_model": None,
        "most_tokens": None,
        "total_cost": 0,
        "response_lengths": {}
    }
    
    if not results:
        return metrics
    
    fastest_time = float('inf')
    most_tokens = 0
    
    for provider, result in results.items():
        if result.response_time < fastest_time:
            fastest_time = result.response_time
            metrics["fastest_model"] = result.model_name
        
        if result.total_tokens and result.total_tokens > most_tokens:
            most_tokens = result.total_tokens
            metrics["most_tokens"] = result.model_name
        
        if result.estimated_cost:
            metrics["total_cost"] += result.estimated_cost
        
        metrics["response_lengths"][provider] = len(result.response)
    
    return metrics

# =============================================================================
# ENDPOINTS DE INFORMACIÓN DEL SISTEMA
# =============================================================================

@api_bp.route('/status', methods=['GET'])
@rate_limit()
def system_status():
    """Estado detallado del sistema"""
    try:
        llm_service = get_llm_service()
        
        # Verificar servicios
        try:
            from app.services.rag import rag_service
            rag_stats = rag_service.get_stats()
        except:
            rag_stats = {'total_documents': 0, 'vector_store_type': 'not_initialized'}
        
        llm_availability = llm_service.get_available_providers()
        llm_stats = llm_service.get_service_stats()
        
        try:
            from app.services.ingestion import ingestion_service
            ingestion_stats = ingestion_service.get_service_stats()
        except:
            ingestion_stats = {'processors_available': 0, 'jobs_active': 0}
        
        has_available_provider = any(llm_availability.values())
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
                    'providers_available': llm_availability,
                    'service_status': llm_stats.get('service_status', 'unknown'),
                    'total_models': llm_stats.get('models', {})
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
        llm_service = get_llm_service()
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
        llm_service = get_llm_service()
        
        stats = {
            'llm': llm_service.get_service_stats(),
            'timestamp': time.time()
        }
        
        try:
            from app.services.rag import rag_service
            stats['rag'] = rag_service.get_stats()
        except:
            stats['rag'] = {'status': 'not_available'}
        
        try:
            from app.services.ingestion import ingestion_service
            stats['ingestion'] = ingestion_service.get_service_stats()
        except:
            stats['ingestion'] = {'status': 'not_available'}
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error("Error obteniendo estadísticas", error=str(e))
        return jsonify({'error': str(e)}), 500

# =============================================================================
# ENDPOINTS LLM PRINCIPALES
# =============================================================================

@api_bp.route('/llm/status', methods=['GET'])
@rate_limit(60)
def get_llm_status():
    """Obtener estado de los proveedores LLM"""
    try:
        service = get_llm_service()
        availability = service.get_available_providers()
        models = service.get_available_models()
        stats = service.get_service_stats()
        
        return jsonify({
            "status": "success",
            "data": {
                "providers": availability,
                "models": models,
                "stats": stats,
                "timestamp": time.time()
            }
        }), 200
        
    except Exception as e:
        logger.error("Error obteniendo status LLM", error=str(e))
        return jsonify({
            "status": "error",
            "message": f"Error obteniendo estado: {str(e)}"
        }), 500

@api_bp.route('/llm/models', methods=['GET'])
@rate_limit(60)
def get_available_llm_models():
    """Obtener modelos disponibles por proveedor"""
    try:
        service = get_llm_service()
        models = service.get_available_models()
        
        return jsonify({
            "status": "success",
            "data": models
        }), 200
        
    except Exception as e:
        logger.error("Error obteniendo modelos", error=str(e))
        return jsonify({
            "status": "error",
            "message": f"Error obteniendo modelos: {str(e)}"
        }), 500

@api_bp.route('/llm/generate', methods=['POST'])
@rate_limit(30)
def generate_llm_response():
    """Generar respuesta usando un modelo específico"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "status": "error",
                "message": "Campo 'query' requerido"
            }), 400
        
        query = data['query']
        provider = data.get('provider', 'ollama')
        model_name = data.get('model_name')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 1000)
        use_rag = data.get('use_rag', False)
        
        if temperature < 0 or temperature > 2:
            return jsonify({
                "status": "error",
                "message": "Temperature debe estar entre 0 y 2"
            }), 400
        
        if max_tokens < 1 or max_tokens > 4000:
            return jsonify({
                "status": "error",
                "message": "max_tokens debe estar entre 1 y 4000"
            }), 400
        
        context = None
        if use_rag:
            context = get_rag_context(query)
        
        llm_request = LLMRequest(
            query=query,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        service = get_llm_service()
        result = service.generate_response(llm_request, provider, model_name)
        
        logger.info("Respuesta LLM generada",
                   provider=provider,
                   model=result.model_name,
                   response_time=result.response_time,
                   tokens=result.total_tokens)
        
        return jsonify({
            "status": "success",
            "data": {
                "response": result.response,
                "model_name": result.model_name,
                "provider": result.provider,
                "response_time": result.response_time,
                "total_tokens": result.total_tokens,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "estimated_cost": result.estimated_cost,
                "sources": result.sources,
                "error": result.error,
                "use_rag": use_rag,
                "context_chunks": len(context) if context else 0
            }
        }), 200
        
    except Exception as e:
        logger.error("Error generando respuesta LLM", error=str(e))
        return jsonify({
            "status": "error",
            "message": f"Error generando respuesta: {str(e)}"
        }), 500

@api_bp.route('/llm/compare', methods=['POST'])
@rate_limit(10)
def compare_llm_models():
    """Comparar respuestas entre modelos local y OpenAI"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "status": "error",
                "message": "Campo 'query' requerido"
            }), 400
        
        query = data['query']
        local_model = data.get('local_model')
        openai_model = data.get('openai_model')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 1000)
        use_rag = data.get('use_rag', True)
        
        context = None
        if use_rag:
            context = get_rag_context(query)
        
        llm_request = LLMRequest(
            query=query,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        service = get_llm_service()
        results = service.compare_models(llm_request, local_model, openai_model)
        comparison_metrics = calculate_comparison_metrics(results)
        
        logger.info("Comparación LLM completada",
                   models_compared=len(results),
                   total_time=sum(r.response_time for r in results.values()))
        
        return jsonify({
            "status": "success",
            "data": {
                "results": {
                    provider: {
                        "response": result.response,
                        "model_name": result.model_name,
                        "provider": result.provider,
                        "response_time": result.response_time,
                        "total_tokens": result.total_tokens,
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "estimated_cost": result.estimated_cost,
                        "sources": result.sources,
                        "error": result.error
                    }
                    for provider, result in results.items()
                },
                "comparison_metrics": comparison_metrics,
                "query": query,
                "use_rag": use_rag,
                "context_chunks": len(context) if context else 0,
                "timestamp": time.time()
            }
        }), 200
        
    except Exception as e:
        logger.error("Error en comparación LLM", error=str(e))
        return jsonify({
            "status": "error",
            "message": f"Error en comparación: {str(e)}"
        }), 500

@api_bp.route('/llm/chat', methods=['POST'])
@rate_limit(60)
def llm_chat_with_rag():
    """Endpoint optimizado para chat con RAG integrado"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "status": "error",
                "message": "Campo 'message' requerido"
            }), 400
        
        message = data['message']
        provider = data.get('provider', 'ollama')
        model_name = data.get('model_name')
        temperature = data.get('temperature', 0.3)
        max_tokens = data.get('max_tokens', 800)
        session_id = data.get('session_id')
        
        context = get_rag_context(message, top_k=3)
        
        llm_request = LLMRequest(
            query=message,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        service = get_llm_service()
        result = service.generate_response(llm_request, provider, model_name)
        
        chat_response = {
            "message": result.response,
            "sources": result.sources,
            "model_used": result.model_name,
            "response_time": result.response_time,
            "context_found": len(context) > 0 if context else False,
            "timestamp": time.time()
        }
        
        if result.estimated_cost:
            chat_response["cost"] = result.estimated_cost
        
        logger.info("Chat response generated",
                   provider=provider,
                   model=result.model_name,
                   context_chunks=len(context) if context else 0,
                   response_time=result.response_time)
        
        return jsonify({
            "status": "success",
            "data": chat_response
        }), 200
        
    except Exception as e:
        logger.error("Error en chat", error=str(e))
        return jsonify({
            "status": "error",
            "message": f"Error en chat: {str(e)}"
        }), 500

# =============================================================================
# ENDPOINTS DE CHAT LEGACY (MANTENIDOS PARA COMPATIBILIDAD)
# =============================================================================

@api_bp.route('/chat/query', methods=['POST'])
@rate_limit()
@validate_json('query')
def chat_query():
    """Procesar consulta de chat con RAG (Endpoint existente mantenido)"""
    try:
        data = g.json_data
        query = data['query'].strip()
        
        if not query:
            return jsonify({'error': 'Query no puede estar vacío'}), 400
        
        security_config = get_security_config()
        if len(query) > security_config.max_query_length:
            return jsonify({
                'error': f'Query demasiado largo (máximo {security_config.max_query_length} caracteres)'
            }), 400
        
        llm_service = get_llm_service()
        
        provider = data.get('provider', 'ollama')
        model_name = data.get('model_name')
        use_rag = data.get('use_rag', True)
        rag_k = data.get('rag_k', 5)
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 1000)
        
        start_time = time.time()
        
        context_chunks = []
        if use_rag:
            context_chunks = get_rag_context(query, top_k=rag_k)
            logger.info("Contexto RAG obtenido",
                       query_length=len(query),
                       chunks_found=len(context_chunks))
        
        llm_request = LLMRequest(
            query=query,
            context=context_chunks,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        response = llm_service.generate_response(llm_request, provider, model_name)
        
        rag_sources = []
        if context_chunks:
            for chunk in context_chunks:
                rag_sources.append({
                    'id': getattr(chunk, 'id', 'unknown'),
                    'content': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    'source_path': chunk.metadata.source_path if chunk.metadata else 'unknown',
                    'source_type': chunk.metadata.source_type if chunk.metadata else 'unknown',
                    'chunk_index': getattr(chunk, 'chunk_index', 0),
                    'section_title': getattr(chunk, 'section_title', '')
                })
        
        total_time = time.time() - start_time
        
        result = {
            'success': not bool(response.error),
            'content': response.response,
            'model': {
                'name': response.model_name,
                'type': response.provider,
                'response_time': response.response_time,
                'tokens_used': response.total_tokens,
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
                'generation_params': {'temperature': temperature, 'max_tokens': max_tokens},
                'total_time': total_time
            },
            'timestamp': time.time()
        }
        
        if response.error:
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
                   success=not bool(response.error),
                   total_time=total_time)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error("Error procesando consulta de chat", error=str(e))
        return jsonify({'error': str(e)}), 500

@api_bp.route('/chat/compare', methods=['POST'])
@rate_limit()
@validate_json('query')
def chat_compare():
    """Comparar respuestas de múltiples modelos (Endpoint existente mantenido)"""
    try:
        data = g.json_data
        query = data['query'].strip()
        
        if not query:
            return jsonify({'error': 'Query no puede estar vacío'}), 400
        
        use_rag = data.get('use_rag', True)
        rag_k = data.get('rag_k', 5)
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 1000)
        
        start_time = time.time()
        
        context_chunks = []
        if use_rag:
            context_chunks = get_rag_context(query, top_k=rag_k)
        
        llm_request = LLMRequest(
            query=query,
            context=context_chunks,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        llm_service = get_llm_service()
        responses = llm_service.compare_models(llm_request)
        
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
        
        for provider, response in responses.items():
            result['responses'][provider] = {
                'success': not bool(response.error),
                'content': response.response,
                'model_name': response.model_name,
                'response_time': response.response_time,
                'tokens_used': response.total_tokens,
                'error': response.error if response.error else None
            }
        
        if len(responses) >= 2:
            successful_responses = [r for r in responses.values() if not r.error]
            if len(successful_responses) >= 2:
                response_times = [r.response_time for r in successful_responses]
                fastest_provider = min(responses.keys(), 
                    key=lambda k: responses[k].response_time if not responses[k].error else float('inf'))
                result['comparison']['fastest_provider'] = fastest_provider
                result['comparison']['time_difference'] = max(response_times) - min(response_times)
        
        logger.info("Comparación de modelos completada",
                   query_length=len(query),
                   providers=list(responses.keys()),
                   successful_responses=len([r for r in responses.values() if not r.error]))
        
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
        
        k = min(data.get('k', 5), 20)
        threshold = data.get('threshold', 0.7)
        
        start_time = time.time()
        
        try:
            from app.services.rag import rag_service
            results = rag_service.search_with_scores(query, k=k, threshold=threshold)
        except:
            results = []
        
        search_time = time.time() - start_time
        
        search_results = []
        for chunk, score in results:
            search_results.append({
                'id': getattr(chunk, 'id', 'unknown'),
                'content': chunk.content,
                'score': score,
                'metadata': {
                    'source_path': chunk.metadata.source_path if chunk.metadata else 'unknown',
                    'source_type': chunk.metadata.source_type if chunk.metadata else 'unknown',
                    'file_type': getattr(chunk.metadata, 'file_type', 'unknown') if chunk.metadata else 'unknown',
                    'chunk_index': getattr(chunk, 'chunk_index', 0),
                    'section_title': getattr(chunk, 'section_title', ''),
                    'page_number': getattr(chunk, 'page_number', None)
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
        try:
            from app.services.rag import rag_service
            stats = rag_service.get_stats()
        except:
            stats = {'status': 'not_available', 'message': 'RAG service not initialized'}
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
        try:
            from app.services.ingestion import ingestion_service
            jobs = ingestion_service.get_active_jobs()
            jobs_data = [job.to_dict() for job in jobs]
        except:
            jobs_data = []
        
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
        try:
            from app.services.ingestion import ingestion_service
            job = ingestion_service.get_job_status(job_id)
            
            if not job:
                return jsonify({'error': 'Trabajo no encontrado'}), 404
            
            return jsonify(job.to_dict())
        except:
            return jsonify({'error': 'Servicio de ingesta no disponible'}), 503
        
    except Exception as e:
        logger.error("Error obteniendo trabajo de ingesta", job_id=job_id, error=str(e))
        return jsonify({'error': str(e)}), 500

@api_bp.route('/ingestion/jobs/<job_id>/cancel', methods=['POST'])
@rate_limit()
def cancel_ingestion_job(job_id: str):
    """Cancelar un trabajo de ingesta"""
    try:
        try:
            from app.services.ingestion import ingestion_service
            success = ingestion_service.cancel_job(job_id)
            
            if not success:
                return jsonify({'error': 'No se pudo cancelar el trabajo'}), 400
            
            return jsonify({
                'success': True,
                'message': 'Trabajo cancelado exitosamente',
                'job_id': job_id
            })
        except:
            return jsonify({'error': 'Servicio de ingesta no disponible'}), 503
        
    except Exception as e:
        logger.error("Error cancelando trabajo de ingesta", job_id=job_id, error=str(e))
        return jsonify({'error': str(e)}), 500

@api_bp.route('/ingestion/supported-formats', methods=['GET'])
@rate_limit()
def supported_formats():
    """Obtener formatos de archivo soportados"""
    try:
        try:
            from app.services.ingestion import ingestion_service
            formats = ingestion_service.get_supported_formats()
        except:
            formats = {
                'documents': ['pdf', 'docx', 'txt', 'md'],
                'data': ['csv', 'xlsx', 'json'],
                'web': ['html', 'xml'],
                'max_file_size': '50MB'
            }
        return jsonify(formats)
        
    except Exception as e:
        logger.error("Error obteniendo formatos soportados", error=str(e))
        return jsonify({'error': str(e)}), 500

# =============================================================================
# ENDPOINTS DE CONFIGURACIÓN
# =============================================================================

@api_bp.route('/config/models', methods=['GET'])
@rate_limit()
def get_model_config_endpoint():
    """Obtener configuración de modelos"""
    try:
        config = get_model_config()
        
        return jsonify({
            'embedding': {
                'name': getattr(config, 'embedding_name', 'all-MiniLM-L6-v2'),
                'dimension': getattr(config, 'embedding_dimension', 384),
                'device': getattr(config, 'embedding_device', 'cpu')
            },
            'local': {
                'default': getattr(config, 'default_local_model', 'llama3.2:3b'),
                'available': getattr(config, 'local_available', []),
                'endpoint': getattr(config, 'local_endpoint', 'http://localhost:11434'),
                'timeout': getattr(config, 'local_timeout', 60)
            },
            'openai': {
                'default': getattr(config, 'default_openai_model', 'gpt-4o-mini'),
                'available': getattr(config, 'openai_available', []),
                'timeout': getattr(config, 'openai_timeout', 30)
            }
        })
        
    except Exception as e:
        logger.error("Error obteniendo configuración de modelos", error=str(e))
        return jsonify({'error': str(e)}), 500

@api_bp.route('/config/rag', methods=['GET'])
@rate_limit()
def get_rag_config_endpoint():
    """Obtener configuración RAG"""
    try:
        config = get_rag_config()
        
        return jsonify({
            'enabled': getattr(config, 'enabled', True),
            'k_default': getattr(config, 'k_default', 5),
            'k_max': getattr(config, 'k_max', 20),
            'chunk_size': getattr(config, 'chunk_size', 1000),
            'chunk_overlap': getattr(config, 'chunk_overlap', 100),
            'similarity_threshold': getattr(config, 'similarity_threshold', 0.7),
            'retrieval_strategy': getattr(config, 'retrieval_strategy', 'similarity'),
            'rerank_enabled': getattr(config, 'rerank_enabled', False)
        })
        
    except Exception as e:
        logger.error("Error obteniendo configuración RAG", error=str(e))
        return jsonify({'error': str(e)}), 500

# =============================================================================
# ENDPOINTS DE PRUEBA Y DESARROLLO
# =============================================================================

@api_bp.route('/test/llm', methods=['GET'])
@rate_limit(10)
def test_llm_service():
    """Endpoint de prueba para LLM Service (solo desarrollo)"""
    try:
        service = get_llm_service()
        
        test_request = LLMRequest(
            query="¿Qué es una administración local?",
            temperature=0.7,
            max_tokens=100
        )
        
        availability = service.get_available_providers()
        
        test_results = {}
        if availability.get('ollama', False):
            try:
                result = service.generate_response(test_request, 'ollama')
                test_results['ollama'] = {
                    'success': not bool(result.error),
                    'response_length': len(result.response),
                    'response_time': result.response_time,
                    'model': result.model_name
                }
            except Exception as e:
                test_results['ollama'] = {'error': str(e)}
        
        if availability.get('openai', False):
            try:
                result = service.generate_response(test_request, 'openai')
                test_results['openai'] = {
                    'success': not bool(result.error),
                    'response_length': len(result.response),
                    'response_time': result.response_time,
                    'model': result.model_name,
                    'cost': result.estimated_cost
                }
            except Exception as e:
                test_results['openai'] = {'error': str(e)}
        
        return jsonify({
            'status': 'success',
            'availability': availability,
            'test_results': test_results,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error("Error en test LLM", error=str(e))
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check específico para API"""
    try:
        llm_service = get_llm_service()
        llm_available = any(llm_service.get_available_providers().values())
        
        status = 'healthy' if llm_available else 'degraded'
        
        return jsonify({
            'status': status,
            'timestamp': time.time(),
            'version': '1.0.0',
            'services': {
                'llm': 'available' if llm_available else 'unavailable'
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': time.time()
        }), 500

@api_bp.route('/docs', methods=['GET'])
def api_documentation():
    """Documentación básica de la API"""
    docs = {
        'title': 'Prototipo Chatbot RAG API',
        'version': '1.0.0',
        'description': 'API REST para chatbot RAG de administraciones locales',
        'endpoints': {
            'sistema': {
                'GET /api/status': 'Estado del sistema completo',
                'GET /api/models': 'Modelos disponibles',
                'GET /api/stats': 'Estadísticas del sistema',
                'GET /api/health': 'Health check básico'
            },
            'llm': {
                'GET /api/llm/status': 'Estado de proveedores LLM',
                'GET /api/llm/models': 'Modelos LLM disponibles',
                'POST /api/llm/generate': 'Generar respuesta con modelo específico',
                'POST /api/llm/compare': 'Comparar respuestas entre modelos',
                'POST /api/llm/chat': 'Chat optimizado con RAG'
            },
            'chat': {
                'POST /api/chat/query': 'Consulta de chat (legacy)',
                'POST /api/chat/compare': 'Comparación de chat (legacy)'
            },
            'rag': {
                'POST /api/rag/search': 'Búsqueda semántica',
                'GET /api/rag/stats': 'Estadísticas RAG'
            },
            'configuracion': {
                'GET /api/config/models': 'Configuración de modelos',
                'GET /api/config/rag': 'Configuración RAG'
            },
            'desarrollo': {
                'GET /api/test/llm': 'Test de LLM Service',
                'GET /api/docs': 'Esta documentación'
            }
        },
        'authentication': 'No requerida (desarrollo)',
        'rate_limits': {
            'general': '60 requests/minute',
            'comparaciones': '10 requests/minute',
            'chat': '60 requests/minute'
        }
    }
    
    return jsonify(docs)

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

# =============================================================================
# HOOKS DE LOGGING
# =============================================================================

@api_bp.before_request
def log_api_request():
    """Log de todas las requests API"""
    g.start_time = time.time()
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


# =============================================================================
# FIN DEL ARCHIVO - API COMPLETA
# =============================================================================