"""
Rutas de chat web para Prototipo_chatbot
TFM Vicente Caruncho - Sistemas Inteligentes
"""

from flask import Blueprint, render_template, request, jsonify, session, flash, redirect, url_for
import uuid
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# Imports locales
from app.core.logger import get_logger
from app.core.config import get_security_config, get_model_config, get_rag_config
from app.services.rag import search_documents
from app.services.llm_service import generate_llm_response, compare_llm_responses, llm_service
from app.models import ChatSession, ChatMessage, create_chat_message

# Crear blueprint
chat_bp = Blueprint('chat', __name__)
logger = get_logger("chat_routes")

# Almacén en memoria para sesiones de chat (en producción usar Redis)
chat_sessions = {}

def get_or_create_session() -> ChatSession:
    """Obtener o crear sesión de chat"""
    session_id = session.get('chat_session_id')
    
    if session_id and session_id in chat_sessions:
        chat_session = chat_sessions[session_id]
        logger.debug("Sesión de chat existente", session_id=session_id)
    else:
        # Crear nueva sesión
        session_id = str(uuid.uuid4())
        chat_session = ChatSession(
            id=session_id,
            user_ip=request.remote_addr,
            user_agent=request.headers.get('User-Agent', '')
        )
        chat_sessions[session_id] = chat_session
        session['chat_session_id'] = session_id
        
        logger.info("Nueva sesión de chat creada", session_id=session_id)
    
    return chat_session

def clean_old_sessions():
    """Limpiar sesiones antiguas (llamar periódicamente)"""
    current_time = datetime.now()
    sessions_to_remove = []
    
    for session_id, chat_session in chat_sessions.items():
        # Remover sesiones inactivas por más de 24 horas
        if (current_time - chat_session.updated_at).total_seconds() > 86400:
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        del chat_sessions[session_id]
        logger.debug("Sesión de chat limpiada", session_id=session_id)

# =============================================================================
# RUTAS PRINCIPALES DE CHAT
# =============================================================================

@chat_bp.route('/')
def chat_index():
    """Página principal de chat"""
    try:
        # Limpiar sesiones antiguas
        clean_old_sessions()
        
        # Obtener configuraciones para el frontend
        model_config = get_model_config()
        rag_config = get_rag_config()
        
        # Verificar disponibilidad de proveedores
        providers = llm_service.get_available_providers()
        models = llm_service.get_available_models()
        
        context = {
            'session_id': session.get('chat_session_id'),
            'providers_available': providers,
            'models_available': models,
            'config': {
                'model': {
                    'local_default': model_config.local_default,
                    'openai_default': model_config.openai_default,
                    'local_available': model_config.local_available,
                    'openai_available': model_config.openai_available
                },
                'rag': {
                    'enabled': rag_config.enabled,
                    'k_default': rag_config.k_default,
                    'k_max': rag_config.k_max,
                    'similarity_threshold': rag_config.similarity_threshold
                }
            }
        }
        
        return render_template('chat/index.html', **context)
        
    except Exception as e:
        logger.error("Error cargando página de chat", error=str(e))
        flash('Error cargando la interfaz de chat', 'error')
        return redirect(url_for('index'))

@chat_bp.route('/session')
def chat_session_info():
    """Información de la sesión actual"""
    try:
        chat_session = get_or_create_session()
        
        return jsonify({
            'session_id': chat_session.id,
            'message_count': chat_session.message_count,
            'total_tokens': chat_session.total_tokens,
            'created_at': chat_session.created_at.isoformat(),
            'updated_at': chat_session.updated_at.isoformat(),
            'model_config': chat_session.model_config,
            'rag_config': chat_session.rag_config
        })
        
    except Exception as e:
        logger.error("Error obteniendo información de sesión", error=str(e))
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/history')
def chat_history():
    """Historial de mensajes de la sesión"""
    try:
        chat_session = get_or_create_session()
        
        # Obtener últimos N mensajes
        max_messages = request.args.get('limit', 50, type=int)
        messages = chat_session.get_context_messages(max_messages)
        
        messages_data = []
        for message in messages:
            messages_data.append({
                'id': message.id,
                'role': message.role,
                'content': message.content,
                'timestamp': message.timestamp.isoformat(),
                'model_name': message.model_name,
                'model_type': message.model_type,
                'response_time': message.response_time,
                'tokens_used': message.tokens_used,
                'rag_sources': message.rag_sources
            })
        
        return jsonify({
            'session_id': chat_session.id,
            'messages': messages_data,
            'total_messages': len(messages_data)
        })
        
    except Exception as e:
        logger.error("Error obteniendo historial de chat", error=str(e))
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/clear', methods=['POST'])
def clear_chat():
    """Limpiar historial de chat"""
    try:
        session_id = session.get('chat_session_id')
        
        if session_id and session_id in chat_sessions:
            # Crear nueva sesión manteniendo el mismo ID
            chat_sessions[session_id] = ChatSession(
                id=session_id,
                user_ip=request.remote_addr,
                user_agent=request.headers.get('User-Agent', '')
            )
            logger.info("Chat limpiado", session_id=session_id)
        
        return jsonify({
            'success': True,
            'message': 'Chat limpiado exitosamente'
        })
        
    except Exception as e:
        logger.error("Error limpiando chat", error=str(e))
        return jsonify({'error': str(e)}), 500

# =============================================================================
# ENDPOINTS DE MENSAJE
# =============================================================================

@chat_bp.route('/message', methods=['POST'])
def send_message():
    """Enviar mensaje y obtener respuesta"""
    try:
        # Validar request
        if not request.is_json:
            return jsonify({'error': 'Content-Type debe ser application/json'}), 400
        
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Campo "message" requerido'}), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({'error': 'Mensaje no puede estar vacío'}), 400
        
        # Validar longitud del mensaje
        security_config = get_security_config()
        if len(user_message) > security_config.max_query_length:
            return jsonify({
                'error': f'Mensaje demasiado largo (máximo {security_config.max_query_length} caracteres)'
            }), 400
        
        # Obtener sesión de chat
        chat_session = get_or_create_session()
        
        # Parámetros de configuración
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
        generation_params = {k: v for k, v in generation_params.items() if v is not None}
        
        # Guardar mensaje del usuario
        user_msg = create_chat_message(
            role='user',
            content=user_message,
            user_ip=request.remote_addr,
            session_id=chat_session.id
        )
        chat_session.add_message(user_msg)
        
        start_time = time.time()
        
        # Búsqueda RAG si está habilitada
        context_chunks = []
        rag_sources = []
        
        if use_rag:
            context_chunks = search_documents(user_message, k=rag_k)
            logger.info("Contexto RAG obtenido para chat",
                       session_id=chat_session.id,
                       query_length=len(user_message),
                       chunks_found=len(context_chunks))
            
            # Preparar fuentes RAG para guardar
            for chunk in context_chunks:
                rag_sources.append({
                    'id': chunk.id,
                    'content': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    'source_path': chunk.metadata.source_path,
                    'source_type': chunk.metadata.source_type,
                    'chunk_index': chunk.chunk_index,
                    'section_title': chunk.section_title
                })
        
        # Generar respuesta
        response = generate_llm_response(
            query=user_message,
            provider=provider,
            model_name=model_name,
            context=context_chunks,
            **generation_params
        )
        
        total_time = time.time() - start_time
        
        # Guardar respuesta del asistente
        assistant_msg = create_chat_message(
            role='assistant',
            content=response.content if response.success else f"Error: {response.error}",
            model_name=response.model_name,
            model_type=response.model_type,
            response_time=response.response_time,
            tokens_used=response.tokens_used,
            rag_sources=rag_sources,
            rag_query=user_message if use_rag else None,
            rag_k=rag_k if use_rag else None,
            session_id=chat_session.id,
            error=response.error if not response.success else None
        )
        chat_session.add_message(assistant_msg)
        
        # Actualizar configuración de la sesión
        chat_session.model_config = {
            'provider': provider,
            'model_name': response.model_name,
            **generation_params
        }
        chat_session.rag_config = {
            'enabled': use_rag,
            'k': rag_k
        }
        
        # Preparar respuesta
        result = {
            'success': response.success,
            'message_id': assistant_msg.id,
            'content': response.content,
            'model': {
                'name': response.model_name,
                'type': response.model_type,
                'provider': provider,
                'response_time': response.response_time,
                'tokens_used': response.tokens_used,
                'prompt_tokens': response.prompt_tokens,
                'completion_tokens': response.completion_tokens
            },
            'rag': {
                'enabled': use_rag,
                'sources': rag_sources,
                'total_sources': len(context_chunks)
            },
            'session': {
                'id': chat_session.id,
                'message_count': chat_session.message_count,
                'total_tokens': chat_session.total_tokens
            },
            'timing': {
                'response_time': response.response_time,
                'total_time': total_time
            },
            'timestamp': time.time()
        }
        
        if not response.success:
            result['error'] = response.error
            logger.warning("Error en mensaje de chat",
                          session_id=chat_session.id,
                          provider=provider,
                          error=response.error)
            return jsonify(result), 500
        
        logger.info("Mensaje de chat procesado exitosamente",
                   session_id=chat_session.id,
                   provider=provider,
                   model=response.model_name,
                   message_length=len(user_message),
                   response_length=len(response.content),
                   total_time=total_time)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error("Error procesando mensaje de chat", error=str(e))
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/compare', methods=['POST'])
def compare_message():
    """Comparar respuestas de múltiples modelos"""
    try:
        # Validar request
        if not request.is_json:
            return jsonify({'error': 'Content-Type debe ser application/json'}), 400
        
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Campo "message" requerido'}), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({'error': 'Mensaje no puede estar vacío'}), 400
        
        # Obtener sesión de chat
        chat_session = get_or_create_session()
        
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
        
        # Guardar mensaje del usuario
        user_msg = create_chat_message(
            role='user',
            content=user_message,
            user_ip=request.remote_addr,
            session_id=chat_session.id
        )
        chat_session.add_message(user_msg)
        
        start_time = time.time()
        
        # Búsqueda RAG
        context_chunks = []
        rag_sources = []
        
        if use_rag:
            context_chunks = search_documents(user_message, k=rag_k)
            
            for chunk in context_chunks:
                rag_sources.append({
                    'id': chunk.id,
                    'content': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    'source_path': chunk.metadata.source_path,
                    'source_type': chunk.metadata.source_type,
                    'chunk_index': chunk.chunk_index,
                    'section_title': chunk.section_title
                })
        
        # Generar respuestas comparativas
        responses = compare_llm_responses(
            query=user_message,
            context=context_chunks,
            **generation_params
        )
        
        total_time = time.time() - start_time
        
        # Guardar respuestas del asistente
        comparison_results = {}
        for provider, response in responses.items():
            assistant_msg = create_chat_message(
                role='assistant',
                content=response.content if response.success else f"Error ({provider}): {response.error}",
                model_name=response.model_name,
                model_type=response.model_type,
                response_time=response.response_time,
                tokens_used=response.tokens_used,
                rag_sources=rag_sources,
                rag_query=user_message if use_rag else None,
                rag_k=rag_k if use_rag else None,
                session_id=chat_session.id,
                error=response.error if not response.success else None
            )
            chat_session.add_message(assistant_msg)
            
            comparison_results[provider] = {
                'message_id': assistant_msg.id,
                'success': response.success,
                'content': response.content,
                'model_name': response.model_name,
                'response_time': response.response_time,
                'tokens_used': response.tokens_used,
                'error': response.error if not response.success else None
            }
        
        # Preparar respuesta de comparación
        result = {
            'user_message': user_message,
            'responses': comparison_results,
            'comparison': {
                'providers_compared': len(responses),
                'total_time': total_time,
                'successful_responses': len([r for r in responses.values() if r.success])
            },
            'rag': {
                'enabled': use_rag,
                'sources': rag_sources,
                'total_sources': len(context_chunks)
            },
            'session': {
                'id': chat_session.id,
                'message_count': chat_session.message_count,
                'total_tokens': chat_session.total_tokens
            },
            'timestamp': time.time()
        }
        
        # Métricas de comparación adicionales
        if len(responses) >= 2:
            successful_responses = [r for r in responses.values() if r.success]
            if len(successful_responses) >= 2:
                response_times = [r.response_time for r in successful_responses]
                result['comparison']['fastest_response_time'] = min(response_times)
                result['comparison']['slowest_response_time'] = max(response_times)
                result['comparison']['time_difference'] = max(response_times) - min(response_times)
        
        logger.info("Comparación de chat completada",
                   session_id=chat_session.id,
                   providers=list(responses.keys()),
                   message_length=len(user_message),
                   successful_responses=result['comparison']['successful_responses'],
                   total_time=total_time)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error("Error en comparación de chat", error=str(e))
        return jsonify({'error': str(e)}), 500

# =============================================================================
# RUTAS DE CONFIGURACIÓN DE CHAT
# =============================================================================

@chat_bp.route('/config')
def chat_config():
    """Página de configuración de chat"""
    try:
        # Obtener configuraciones actuales
        model_config = get_model_config()
        rag_config = get_rag_config()
        security_config = get_security_config()
        
        # Verificar disponibilidad
        providers = llm_service.get_available_providers()
        models = llm_service.get_available_models()
        
        context = {
            'model_config': model_config,
            'rag_config': rag_config,
            'security_config': security_config,
            'providers_available': providers,
            'models_available': models
        }
        
        return render_template('chat/config.html', **context)
        
    except Exception as e:
        logger.error("Error cargando configuración de chat", error=str(e))
        flash('Error cargando la configuración', 'error')
        return redirect(url_for('chat.chat_index'))

@chat_bp.route('/export')
def export_chat():
    """Exportar historial de chat"""
    try:
        session_id = session.get('chat_session_id')
        if not session_id or session_id not in chat_sessions:
            return jsonify({'error': 'No hay sesión activa'}), 404
        
        chat_session = chat_sessions[session_id]
        
        # Exportar como JSON
        export_data = {
            'session_id': chat_session.id,
            'created_at': chat_session.created_at.isoformat(),
            'updated_at': chat_session.updated_at.isoformat(),
            'message_count': chat_session.message_count,
            'total_tokens': chat_session.total_tokens,
            'messages': [msg.to_dict() for msg in chat_session.messages]
        }
        
        logger.info("Chat exportado", session_id=session_id, messages=len(chat_session.messages))
        
        return jsonify(export_data)
        
    except Exception as e:
        logger.error("Error exportando chat", error=str(e))
        return jsonify({'error': str(e)}), 500

# =============================================================================
# ENDPOINTS DE ESTADÍSTICAS
# =============================================================================

@chat_bp.route('/stats')
def chat_stats():
    """Estadísticas de chat"""
    try:
        # Estadísticas globales
        total_sessions = len(chat_sessions)
        active_sessions = len([s for s in chat_sessions.values() 
                              if (datetime.now() - s.updated_at).total_seconds() < 3600])
        
        total_messages = sum(s.message_count for s in chat_sessions.values())
        total_tokens = sum(s.total_tokens for s in chat_sessions.values())
        
        # Estadísticas de modelos
        model_usage = {}
        for chat_session in chat_sessions.values():
            for message in chat_session.messages:
                if message.model_name:
                    model_usage[message.model_name] = model_usage.get(message.model_name, 0) + 1
        
        return jsonify({
            'sessions': {
                'total': total_sessions,
                'active': active_sessions,
                'inactive': total_sessions - active_sessions
            },
            'messages': {
                'total': total_messages,
                'average_per_session': total_messages / total_sessions if total_sessions > 0 else 0
            },
            'tokens': {
                'total': total_tokens,
                'average_per_session': total_tokens / total_sessions if total_sessions > 0 else 0
            },
            'model_usage': model_usage,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error("Error obteniendo estadísticas de chat", error=str(e))
        return jsonify({'error': str(e)}), 500

# =============================================================================
# HOOKS Y UTILIDADES
# =============================================================================

@chat_bp.before_request
def before_chat_request():
    """Hook antes de cada request de chat"""
    # Limpiar sesiones antiguas periódicamente
    if request.endpoint and 'chat' in request.endpoint:
        clean_old_sessions()

@chat_bp.errorhandler(404)
def chat_not_found(error):
    """Manejo de 404 en rutas de chat"""
    logger.warning("Ruta de chat no encontrada", path=request.path)
    flash('Página de chat no encontrada', 'error')
    return redirect(url_for('chat.chat_index'))

@chat_bp.errorhandler(500)
def chat_internal_error(error):
    """Manejo de 500 en rutas de chat"""
    logger.error("Error interno en chat", error=str(error))
    flash('Error interno en el sistema de chat', 'error')
    return redirect(url_for('chat.chat_index'))