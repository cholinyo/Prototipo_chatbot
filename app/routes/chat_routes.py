"""
Routes de Chat con RAG integrado
TFM Vicente Caruncho
"""

from flask import Blueprint, render_template, request, jsonify, session
from datetime import datetime
import uuid

from app.core.logger import get_logger
from app.services.rag_pipeline import get_rag_pipeline, process_chat_query, compare_chat_providers

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
        pipeline = get_rag_pipeline()
        pipeline_status = pipeline.health_check()
        
        return render_template(
            'chat.html',
            pipeline_status=pipeline_status,
            title="Chat RAG - Administración Local"
        )
    except Exception as e:
        logger.error(f"Error cargando interfaz de chat: {e}")
        return render_template(
            'error.html',
            error_message="Error interno del sistema",
            error_code="CHAT_001"
        ), 500


@chat_bp.route('/api/chat/send', methods=['POST'])
def send_message():
    """Enviar mensaje y obtener respuesta RAG"""
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
        
        if not user_query:
            return jsonify({
                'error': 'Mensaje vacío',
                'success': False
            }), 400
        
        logger.info(
            f"Nueva consulta de chat",
            session_id=session_id,
            query=user_query[:100],
            provider=provider
        )
        
        # Procesar consulta con RAG
        rag_response = process_chat_query(
            query=user_query,
            provider=provider,
            model=model,
            temperature=data.get('temperature', 0.3),
            max_tokens=data.get('max_tokens')
        )
        
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
            'content': rag_response.response,
            'model': rag_response.model_name,
            'provider': rag_response.provider,
            'sources': rag_response.sources,
            'response_time': rag_response.response_time,
            'confidence': rag_response.confidence,
            'estimated_cost': rag_response.estimated_cost,
            'timestamp': datetime.now().isoformat(),
            'error': rag_response.error
        })
        
        chat_sessions[session_id]['last_activity'] = datetime.now()
        
        # Preparar respuesta para el frontend
        response_data = {
            'success': True,
            'session_id': session_id,
            'response': rag_response.response,
            'metadata': {
                'model': rag_response.model_name,
                'provider': rag_response.provider,
                'response_time': round(rag_response.response_time, 2),
                'confidence': round(rag_response.confidence, 2),
                'sources_count': len(rag_response.sources),
                'sources': rag_response.sources[:5],  # Máximo 5 fuentes en respuesta
                'estimated_cost': rag_response.estimated_cost
            }
        }
        
        if rag_response.error:
            response_data['warning'] = f"Error parcial: {rag_response.error}"
        
        logger.info(
            f"Consulta procesada exitosamente",
            session_id=session_id,
            response_time=rag_response.response_time,
            provider=rag_response.provider,
            sources=len(rag_response.sources)
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
    """Comparar respuestas entre múltiples proveedores"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Mensaje requerido',
                'success': False
            }), 400
        
        user_query = data['message'].strip()
        session_id = data.get('session_id') or str(uuid.uuid4())
        
        logger.info(
            f"Comparación de proveedores solicitada",
            session_id=session_id,
            query=user_query[:100]
        )
        
        # Comparar respuestas entre proveedores
        comparison_results = compare_chat_providers(
            query=user_query,
            temperature=data.get('temperature', 0.3)
        )
        
        # Preparar respuesta comparativa
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
                'lowest_cost': float('inf')
            }
        }
        
        # Procesar resultados de cada proveedor
        for provider, rag_response in comparison_results.items():
            provider_data = {
                'response': rag_response.response,
                'model': rag_response.model_name,
                'response_time': round(rag_response.response_time, 2),
                'confidence': round(rag_response.confidence, 2),
                'sources': rag_response.sources[:3],  # Top 3 fuentes
                'estimated_cost': rag_response.estimated_cost,
                'error': rag_response.error,
                'success': rag_response.error is None
            }
            
            comparison_data['results'][provider] = provider_data
            
            # Actualizar métricas de resumen
            if rag_response.response_time < comparison_data['summary']['fastest_time']:
                comparison_data['summary']['fastest_provider'] = provider
                comparison_data['summary']['fastest_time'] = rag_response.response_time
            
            if rag_response.confidence > comparison_data['summary']['highest_confidence']:
                comparison_data['summary']['most_confident'] = provider
                comparison_data['summary']['highest_confidence'] = rag_response.confidence
            
            if (rag_response.estimated_cost and 
                rag_response.estimated_cost < comparison_data['summary']['lowest_cost']):
                comparison_data['summary']['cheapest_provider'] = provider
                comparison_data['summary']['lowest_cost'] = rag_response.estimated_cost
        
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
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(
            f"Comparación completada",
            session_id=session_id,
            providers=list(comparison_results.keys()),
            fastest=comparison_data['summary']['fastest_provider']
        )
        
        return jsonify(comparison_data)
        
    except Exception as e:
        logger.error(f"Error en comparación de proveedores: {e}")
        return jsonify({
            'error': 'Error comparando proveedores',
            'success': False,
            'details': str(e) if request.args.get('debug') else None
        }), 500


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


@chat_bp.route('/api/chat/status')
def chat_status():
    """Estado del sistema de chat y pipeline RAG"""
    try:
        pipeline = get_rag_pipeline()
        health_check = pipeline.health_check()
        stats = pipeline.get_pipeline_stats()
        
        return jsonify({
            'success': True,
            'status': health_check['status'],
            'components': health_check['components'],
            'pipeline_stats': stats,
            'active_sessions': len(chat_sessions),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo estado del chat: {e}")
        return jsonify({
            'error': 'Error obteniendo estado',
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