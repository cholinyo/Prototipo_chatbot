"""
Routes de Chat con RAG integrado MEJORADO
TFM Vicente Caruncho - Integración Pipeline RAG Completo
"""

from flask import Blueprint, render_template, request, jsonify, session
from datetime import datetime
import uuid
import time

from app.core.logger import get_logger

# NUEVA INTEGRACIÓN: Pipeline RAG mejorado
try:
    from app.services.rag.pipeline import get_rag_pipeline
    RAG_PIPELINE_AVAILABLE = True
except ImportError:
    # Fallback a versión anterior si existe
    try:
        from app.services.rag_pipeline import get_rag_pipeline, process_chat_query, compare_chat_providers
        RAG_PIPELINE_AVAILABLE = True
    except ImportError:
        RAG_PIPELINE_AVAILABLE = False
        get_rag_pipeline = None

# Blueprint para el chat
chat_bp = Blueprint('chat', __name__)
logger = get_logger("prototipo_chatbot.chat_routes")

# Almacenamiento temporal de sesiones de chat (en producción usar Redis/DB)
chat_sessions = {}


@chat_bp.route('/chat')
def chat_interface():
    """Página principal del chat"""
    try:
        # Verificar estado del pipeline RAG
        if RAG_PIPELINE_AVAILABLE and get_rag_pipeline:
            pipeline = get_rag_pipeline()
            pipeline_status = pipeline.health_check()
            
            # Obtener estadísticas del pipeline
            pipeline_stats = pipeline.get_stats()
            
            context = {
                'pipeline_status': pipeline_status,
                'pipeline_stats': pipeline_stats,
                'title': "Chat RAG - Administración Local",
                'rag_available': True,
                'indexed_documents': pipeline_stats.get('documents_count', 0),
                'vector_store_type': pipeline_stats.get('vector_store_type', 'Unknown'),
                'embedding_model': pipeline_stats.get('embedding_model', 'all-MiniLM-L6-v2')
            }
        else:
            # Fallback sin pipeline RAG
            context = {
                'pipeline_status': {'status': 'unavailable'},
                'pipeline_stats': {},
                'title': "Chat RAG - Administración Local",
                'rag_available': False,
                'warning': 'Pipeline RAG no disponible'
            }
        
        return render_template('chat.html', **context)
        
    except Exception as e:
        logger.error(f"Error cargando interfaz de chat: {e}")
        return render_template(
            'error.html',
            error_message="Error interno del sistema",
            error_code="CHAT_001"
        ), 500


@chat_bp.route('/api/chat/send', methods=['POST'])
def send_message():
    """Enviar mensaje y obtener respuesta RAG MEJORADA"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Mensaje requerido',
                'success': False
            }), 400
        
        user_query = data['message'].strip()
        provider = data.get('provider', 'ollama')
        model = data.get('model')
        session_id = data.get('session_id') or str(uuid.uuid4())
        temperature = data.get('temperature', 0.3)
        use_rag = data.get('use_rag', True)
        k = data.get('k', 5)  # Número de documentos relevantes
        
        if not user_query:
            return jsonify({
                'error': 'Mensaje vacío',
                'success': False
            }), 400
        
        logger.info(
            f"Nueva consulta de chat",
            session_id=session_id,
            query=user_query[:100],
            provider=provider,
            use_rag=use_rag
        )
        
        start_time = time.time()
        
        # NUEVA LÓGICA: Usar pipeline RAG mejorado
        if RAG_PIPELINE_AVAILABLE and get_rag_pipeline and use_rag:
            pipeline = get_rag_pipeline()
            
            if pipeline.is_available():
                # Generar respuesta RAG completa
                rag_response = pipeline.generate_rag_response(
                    query=user_query,
                    provider=provider,
                    model=model,
                    temperature=temperature,
                    k=k
                )
                
                response_data = {
                    'success': True,
                    'session_id': session_id,
                    'response': rag_response['response'],
                    'metadata': {
                        'model': rag_response['model'],
                        'provider': provider,
                        'response_time': round(rag_response['response_time'], 2),
                        'confidence': round(rag_response.get('confidence', 0.8), 2),
                        'sources_count': len(rag_response['sources']),
                        'sources': rag_response['sources'][:5],  # Máximo 5 fuentes
                        'estimated_cost': rag_response.get('estimated_cost'),
                        'use_rag': True,
                        'retrieved_docs': len(rag_response.get('context_docs', []))
                    }
                }
                
                if rag_response.get('error'):
                    response_data['warning'] = f"Error parcial: {rag_response['error']}"
                
            else:
                # Pipeline no disponible, usar respuesta básica
                response_data = {
                    'success': True,
                    'session_id': session_id,
                    'response': "Pipeline RAG no disponible. Esta sería una respuesta básica sin contexto documental.",
                    'metadata': {
                        'model': 'fallback',
                        'provider': 'system',
                        'response_time': round(time.time() - start_time, 2),
                        'confidence': 0.5,
                        'sources_count': 0,
                        'sources': [],
                        'use_rag': False,
                        'warning': 'Pipeline RAG no disponible'
                    }
                }
        
        else:
            # Respuesta sin RAG (modo directo)
            try:
                # Intentar usar solo LLM sin contexto documental
                if provider == 'ollama':
                    # Aquí iría la integración directa con Ollama
                    response_text = f"Respuesta directa de {model or 'ollama'} (sin RAG): {user_query}"
                elif provider == 'openai':
                    # Aquí iría la integración directa con OpenAI
                    response_text = f"Respuesta directa de OpenAI (sin RAG): {user_query}"
                else:
                    response_text = "Proveedor de LLM no soportado."
                
                response_data = {
                    'success': True,
                    'session_id': session_id,
                    'response': response_text,
                    'metadata': {
                        'model': model or 'unknown',
                        'provider': provider,
                        'response_time': round(time.time() - start_time, 2),
                        'confidence': 0.7,
                        'sources_count': 0,
                        'sources': [],
                        'use_rag': False
                    }
                }
                
            except Exception as llm_error:
                logger.error(f"Error en LLM directo: {llm_error}")
                return jsonify({
                    'error': 'Error generando respuesta',
                    'success': False,
                    'details': str(llm_error)
                }), 500
        
        # Guardar en sesión
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {
                'messages': [],
                'created_at': datetime.now(),
                'last_activity': datetime.now()
            }
        
        # Añadir mensaje del usuario
        chat_sessions[session_id]['messages'].append({
            'type': 'user',
            'content': user_query,
            'timestamp': datetime.now().isoformat()
        })
        
        # Añadir respuesta del asistente
        chat_sessions[session_id]['messages'].append({
            'type': 'assistant',
            'content': response_data['response'],
            'model': response_data['metadata']['model'],
            'provider': response_data['metadata']['provider'],
            'sources': response_data['metadata']['sources'],
            'response_time': response_data['metadata']['response_time'],
            'confidence': response_data['metadata']['confidence'],
            'estimated_cost': response_data['metadata'].get('estimated_cost'),
            'use_rag': response_data['metadata']['use_rag'],
            'timestamp': datetime.now().isoformat()
        })
        
        chat_sessions[session_id]['last_activity'] = datetime.now()
        
        logger.info(
            f"Consulta procesada exitosamente",
            session_id=session_id,
            response_time=response_data['metadata']['response_time'],
            provider=response_data['metadata']['provider'],
            use_rag=response_data['metadata']['use_rag']
        )
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error procesando mensaje de chat: {e}")
        return jsonify({
            'error': 'Error interno del servidor',
            'success': False,
            'details': str(e) if request.args.get('debug') else None
        }), 500


@chat_bp.route('/api/chat/compare', methods=['POST'])
def compare_responses():
    """Comparar respuestas entre múltiples proveedores MEJORADO"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Mensaje requerido',
                'success': False
            }), 400
        
        user_query = data['message'].strip()
        session_id = data.get('session_id') or str(uuid.uuid4())
        temperature = data.get('temperature', 0.3)
        use_rag = data.get('use_rag', True)
        k = data.get('k', 5)
        
        logger.info(
            f"Comparación de proveedores solicitada",
            session_id=session_id,
            query=user_query[:100],
            use_rag=use_rag
        )
        
        comparison_data = {
            'success': True,
            'session_id': session_id,
            'query': user_query,
            'results': {},
            'summary': {
                'fastest_provider': None,
                'fastest_time': float('inf'),
                'most_confident': None,
                'highest_confidence': 0.0,
                'cheapest_provider': None,
                'lowest_cost': float('inf'),
                'best_rag_match': None,
                'most_sources': 0
            }
        }
        
        # Lista de proveedores a comparar
        providers_to_test = [
            {'name': 'ollama', 'model': 'llama3.2:3b'},
            {'name': 'ollama', 'model': 'mistral:7b'},
            {'name': 'openai', 'model': 'gpt-4o-mini'}
        ]
        
        # NUEVA LÓGICA: Comparación con pipeline RAG mejorado
        if RAG_PIPELINE_AVAILABLE and get_rag_pipeline and use_rag:
            pipeline = get_rag_pipeline()
            
            if pipeline.is_available():
                # Usar método de comparación del pipeline
                try:
                    comparison_results = pipeline.compare_models(
                        query=user_query,
                        temperature=temperature,
                        k=k
                    )
                    
                    for provider_key, result in comparison_results.items():
                        provider_data = {
                            'response': result['response'],
                            'model': result['model'],
                            'response_time': round(result['response_time'], 2),
                            'confidence': round(result.get('confidence', 0.5), 2),
                            'sources': result.get('sources', [])[:3],  # Top 3 fuentes
                            'estimated_cost': result.get('estimated_cost'),
                            'error': result.get('error'),
                            'success': result.get('error') is None,
                            'use_rag': True,
                            'retrieved_docs': len(result.get('context_docs', []))
                        }
                        
                        comparison_data['results'][provider_key] = provider_data
                        
                        # Actualizar métricas de resumen
                        if result['response_time'] < comparison_data['summary']['fastest_time']:
                            comparison_data['summary']['fastest_provider'] = provider_key
                            comparison_data['summary']['fastest_time'] = result['response_time']
                        
                        confidence = result.get('confidence', 0.5)
                        if confidence > comparison_data['summary']['highest_confidence']:
                            comparison_data['summary']['most_confident'] = provider_key
                            comparison_data['summary']['highest_confidence'] = confidence
                        
                        cost = result.get('estimated_cost', 0)
                        if cost and cost < comparison_data['summary']['lowest_cost']:
                            comparison_data['summary']['cheapest_provider'] = provider_key
                            comparison_data['summary']['lowest_cost'] = cost
                        
                        sources_count = len(result.get('sources', []))
                        if sources_count > comparison_data['summary']['most_sources']:
                            comparison_data['summary']['best_rag_match'] = provider_key
                            comparison_data['summary']['most_sources'] = sources_count
                    
                except Exception as pipeline_error:
                    logger.error(f"Error en comparación de pipeline: {pipeline_error}")
                    # Fallback a comparación manual
                    comparison_data['results']['error'] = {
                        'response': f"Error en comparación automática: {pipeline_error}",
                        'model': 'error',
                        'response_time': 0,
                        'confidence': 0,
                        'sources': [],
                        'success': False,
                        'use_rag': False
                    }
            else:
                # Pipeline no disponible
                comparison_data['results']['pipeline_unavailable'] = {
                    'response': "Pipeline RAG no disponible para comparación",
                    'model': 'system',
                    'response_time': 0,
                    'confidence': 0,
                    'sources': [],
                    'success': False,
                    'use_rag': False
                }
        
        else:
            # Comparación básica sin RAG
            for provider_config in providers_to_test:
                provider_name = provider_config['name']
                model_name = provider_config['model']
                
                start_time = time.time()
                
                try:
                    # Simulación de respuesta (reemplazar con integración real)
                    response_text = f"Respuesta simulada de {provider_name}:{model_name} para: {user_query[:50]}..."
                    response_time = time.time() - start_time
                    
                    provider_data = {
                        'response': response_text,
                        'model': model_name,
                        'response_time': round(response_time, 2),
                        'confidence': 0.6,  # Simulado
                        'sources': [],
                        'estimated_cost': 0.001 if provider_name == 'openai' else 0,
                        'error': None,
                        'success': True,
                        'use_rag': False
                    }
                    
                    comparison_data['results'][f"{provider_name}_{model_name}"] = provider_data
                    
                except Exception as provider_error:
                    logger.error(f"Error con proveedor {provider_name}: {provider_error}")
                    comparison_data['results'][f"{provider_name}_{model_name}"] = {
                        'response': f"Error: {provider_error}",
                        'model': model_name,
                        'response_time': 0,
                        'confidence': 0,
                        'sources': [],
                        'error': str(provider_error),
                        'success': False,
                        'use_rag': False
                    }
        
        # Guardar comparación en sesión
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {
                'messages': [],
                'created_at': datetime.now(),
                'last_activity': datetime.now()
            }
        
        chat_sessions[session_id]['messages'].append({
            'type': 'comparison',
            'query': user_query,
            'results': comparison_data['results'],
            'summary': comparison_data['summary'],
            'use_rag': use_rag,
            'timestamp': datetime.now().isoformat()
        })
        
        chat_sessions[session_id]['last_activity'] = datetime.now()
        
        logger.info(
            f"Comparación completada",
            session_id=session_id,
            providers=list(comparison_data['results'].keys()),
            fastest=comparison_data['summary']['fastest_provider'],
            use_rag=use_rag
        )
        
        return jsonify(comparison_data)
        
    except Exception as e:
        logger.error(f"Error en comparación de proveedores: {e}")
        return jsonify({
            'error': 'Error comparando proveedores',
            'success': False,
            'details': str(e) if request.args.get('debug') else None
        }), 500


@chat_bp.route('/api/chat/status')
def chat_status():
    """Estado del sistema de chat y pipeline RAG MEJORADO"""
    try:
        if RAG_PIPELINE_AVAILABLE and get_rag_pipeline:
            pipeline = get_rag_pipeline()
            health_check = pipeline.health_check()
            stats = pipeline.get_stats()
            
            status_data = {
                'success': True,
                'status': health_check['status'],
                'components': health_check['components'],
                'pipeline_stats': stats,
                'active_sessions': len(chat_sessions),
                'rag_available': True,
                'timestamp': datetime.now().isoformat()
            }
        else:
            status_data = {
                'success': True,
                'status': 'limited',
                'components': {
                    'pipeline': 'unavailable',
                    'chat': 'available',
                    'sessions': 'available'
                },
                'pipeline_stats': {},
                'active_sessions': len(chat_sessions),
                'rag_available': False,
                'warning': 'Pipeline RAG no disponible',
                'timestamp': datetime.now().isoformat()
            }
        
        return jsonify(status_data)
        
    except Exception as e:
        logger.error(f"Error obteniendo estado del chat: {e}")
        return jsonify({
            'error': 'Error obteniendo estado',
            'success': False,
            'rag_available': False
        }), 500


# NUEVAS RUTAS PARA FUNCIONALIDAD AVANZADA

@chat_bp.route('/api/chat/rag/search', methods=['POST'])
def search_documents():
    """Buscar documentos relevantes sin generar respuesta"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Query requerido',
                'success': False
            }), 400
        
        query = data['query'].strip()
        k = data.get('k', 5)
        threshold = data.get('threshold', 0.5)
        
        if not RAG_PIPELINE_AVAILABLE or not get_rag_pipeline:
            return jsonify({
                'error': 'Pipeline RAG no disponible',
                'success': False
            }), 503
        
        pipeline = get_rag_pipeline()
        
        if not pipeline.is_available():
            return jsonify({
                'error': 'Pipeline RAG no inicializado',
                'success': False
            }), 503
        
        # Buscar documentos relevantes
        search_results = pipeline.search_documents(
            query=query,
            k=k,
            similarity_threshold=threshold
        )
        
        return jsonify({
            'success': True,
            'query': query,
            'results': search_results,
            'total_found': len(search_results),
            'parameters': {
                'k': k,
                'threshold': threshold
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error en búsqueda de documentos: {e}")
        return jsonify({
            'error': 'Error en búsqueda',
            'success': False,
            'details': str(e)
        }), 500


@chat_bp.route('/api/chat/ingest', methods=['POST'])
def ingest_document():
    """Ingestar nuevo documento al pipeline RAG"""
    try:
        if not RAG_PIPELINE_AVAILABLE or not get_rag_pipeline:
            return jsonify({
                'error': 'Pipeline RAG no disponible',
                'success': False
            }), 503
        
        # Verificar si hay archivo en la request
        if 'file' not in request.files:
            return jsonify({
                'error': 'Archivo requerido',
                'success': False
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'error': 'Nombre de archivo requerido',
                'success': False
            }), 400
        
        # Validar tipo de archivo
        allowed_extensions = ['.pdf', '.docx', '.txt', '.md']
        file_ext = file.filename.lower().split('.')[-1]
        
        if f'.{file_ext}' not in allowed_extensions:
            return jsonify({
                'error': f'Tipo de archivo no soportado. Permitidos: {allowed_extensions}',
                'success': False
            }), 400
        
        pipeline = get_rag_pipeline()
        
        # Ingestar archivo temporal
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp_file:
            file.save(tmp_file.name)
            
            try:
                # Ingestar documento
                result = pipeline.ingest_document(tmp_file.name)
                
                return jsonify({
                    'success': True,
                    'filename': file.filename,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                })
                
            finally:
                # Limpiar archivo temporal
                os.unlink(tmp_file.name)
        
    except Exception as e:
        logger.error(f"Error ingiriendo documento: {e}")
        return jsonify({
            'error': 'Error procesando documento',
            'success': False,
            'details': str(e)
        }), 500


# FUNCIONES AUXILIARES MEJORADAS

@chat_bp.route('/api/chat/history/<session_id>')
def get_chat_history(session_id):
    """Obtener historial de chat de una sesión"""
    try:
        if session_id not in chat_sessions:
            return jsonify({
                'error': 'Sesión no encontrada',
                'success': False
            }), 404
        
        session_data = chat_sessions[session_id]
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'messages': session_data['messages'],
            'created_at': session_data['created_at'].isoformat(),
            'last_activity': session_data['last_activity'].isoformat(),
            'message_count': len(session_data['messages'])
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo historial: {e}")
        return jsonify({
            'error': 'Error obteniendo historial',
            'success': False
        }), 500


@chat_bp.route('/api/chat/sessions')
def list_chat_sessions():
    """Listar sesiones de chat disponibles"""
    try:
        sessions_list = []
        
        for session_id, session_data in chat_sessions.items():
            # Solo incluir sesiones de las últimas 24 horas
            time_diff = datetime.now() - session_data['last_activity']
            if time_diff.total_seconds() < 24 * 3600:
                sessions_list.append({
                    'session_id': session_id,
                    'created_at': session_data['created_at'].isoformat(),
                    'last_activity': session_data['last_activity'].isoformat(),
                    'message_count': len(session_data['messages']),
                    'preview': session_data['messages'][0]['content'][:100] if session_data['messages'] else ""
                })
        
        # Ordenar por actividad más reciente
        sessions_list.sort(key=lambda x: x['last_activity'], reverse=True)
        
        return jsonify({
            'success': True,
            'sessions': sessions_list,
            'total_sessions': len(sessions_list)
        })
        
    except Exception as e:
        logger.error(f"Error listando sesiones: {e}")
        return jsonify({
            'error': 'Error listando sesiones',
            'success': False
        }), 500


@chat_bp.route('/api/chat/clear/<session_id>', methods=['DELETE'])
def clear_chat_session(session_id):
    """Limpiar sesión de chat específica"""
    try:
        if session_id in chat_sessions:
            del chat_sessions[session_id]
            logger.info(f"Sesión de chat eliminada: {session_id}")
            
            return jsonify({
                'success': True,
                'message': 'Sesión eliminada correctamente'
            })
        else:
            return jsonify({
                'error': 'Sesión no encontrada',
                'success': False
            }), 404
            
    except Exception as e:
        logger.error(f"Error eliminando sesión: {e}")
        return jsonify({
            'error': 'Error eliminando sesión',
            'success': False
        }), 500


# Función para limpiar sesiones antiguas (llamar periódicamente)
def cleanup_old_sessions():
    """Limpiar sesiones de chat antiguas (más de 24 horas)"""
    try:
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, session_data in chat_sessions.items():
            time_diff = current_time - session_data['last_activity']
            if time_diff.total_seconds() > 24 * 3600:  # 24 horas
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del chat_sessions[session_id]
        
        if sessions_to_remove:
            logger.info(f"Limpieza de sesiones: {len(sessions_to_remove)} sesiones eliminadas")
        
        return len(sessions_to_remove)
        
    except Exception as e:
        logger.error(f"Error limpiando sesiones antiguas: {e}")
        return 0


# HOOK PARA VERIFICAR ESTADO DEL PIPELINE

@chat_bp.before_app_request
def before_chat_request():
    """Verificar estado del pipeline antes de cada request"""
    if RAG_PIPELINE_AVAILABLE and get_rag_pipeline:
        try:
            # Limpiar sesiones antiguas ocasionalmente
            import random
            if random.random() < 0.1:  # 10% de probabilidad
                cleanup_old_sessions()
        except Exception as e:
            logger.debug(f"Error en limpieza automática: {e}")