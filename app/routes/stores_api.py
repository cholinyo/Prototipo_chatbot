"""
Stores API para Dashboard de Vector Stores
TFM Vicente Caruncho - Sistemas Inteligentes

API endpoints para gestión administrativa de Vector Stores
"""

from flask import Blueprint, request, jsonify
import time
from datetime import datetime
from typing import Dict, Any

from app.core.logger import get_logger
from app.services.vector_store_service import get_vector_store_service
from app.services.rag.embeddings import get_embedding_service

# Crear blueprint para la API
stores_api_bp = Blueprint('stores_api', __name__)
logger = get_logger("stores_api")

# Cache para métricas históricas
metrics_history = []

@stores_api_bp.route('/api/stores/stats')
def get_system_stats():
    """Obtener estadísticas completas del sistema"""
    try:
        vector_service = get_vector_store_service()
        embedding_service = get_embedding_service()
        
        # Estadísticas principales
        stats = vector_service.get_stats()
        health = vector_service.get_health_status()
        
        # Procesar datos de stores
        stores_data = {}
        
        if vector_service.faiss_available:
            faiss_stats = vector_service.faiss_store.get_stats()
            stores_data['faiss'] = {
                'vectors': faiss_stats.get('total_vectors', 0),
                'search_time': round(faiss_stats.get('metrics', {}).get('avg_search_time', 0) * 1000, 2),
                'memory_usage': round(faiss_stats.get('index_size_mb', 0), 1),
                'available': True,
                'status': 'active' if stats['active_store'] == 'faiss' else 'standby'
            }
        
        if vector_service.chromadb_available:
            chromadb_stats = vector_service.chromadb_store.get_stats()
            stores_data['chromadb'] = {
                'documents': chromadb_stats.get('total_documents', 0),
                'search_time': round(chromadb_stats.get('performance', {}).get('avg_search_time_ms', 0), 2),
                'memory_usage': round(chromadb_stats.get('storage', {}).get('memory_usage_mb', 0), 1),
                'available': True,
                'status': 'active' if stats['active_store'] == 'chromadb' else 'standby'
            }
        
        # Calcular métricas agregadas
        total_documents = sum(
            store.get('vectors', store.get('documents', 0)) 
            for store in stores_data.values()
        )
        
        avg_search_time = 0
        if stores_data:
            search_times = [store['search_time'] for store in stores_data.values() if store['search_time'] > 0]
            if search_times:
                avg_search_time = sum(search_times) / len(search_times)
        
        total_memory = sum(store['memory_usage'] for store in stores_data.values())
        
        response_data = {
            'total_documents': total_documents,
            'total_vectors': total_documents,  # En este contexto son equivalentes
            'avg_search_time': round(avg_search_time, 2),
            'memory_usage': round(total_memory, 1),
            'active_store': stats['active_store'],
            'stores': stores_data,
            'system_status': health['overall_status'],
            'embedding_service': {
                'available': embedding_service.is_available(),
                'model': embedding_service.model_name,
                'dimension': embedding_service.dimension
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Guardar en historial para gráficos
        global metrics_history
        metrics_history.append({
            'timestamp': datetime.now(),
            'avg_search_time': avg_search_time,
            'memory_usage': total_memory,
            'active_store': stats['active_store']
        })
        
        # Mantener solo últimos 50 registros
        if len(metrics_history) > 50:
            metrics_history = metrics_history[-50:]
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas: {e}")
        return jsonify({'error': str(e)}), 500

@stores_api_bp.route('/api/stores/health')
def get_health_status():
    """Obtener estado de salud detallado del sistema"""
    try:
        vector_service = get_vector_store_service()
        health = vector_service.get_health_status()
        
        return jsonify(health)
        
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        return jsonify({'error': str(e)}), 500

@stores_api_bp.route('/api/stores/switch/<store_type>', methods=['POST'])
def switch_vector_store(store_type):
    """Cambiar el vector store activo"""
    try:
        vector_service = get_vector_store_service()
        
        if store_type not in ['faiss', 'chromadb']:
            return jsonify({'error': 'Invalid store type'}), 400
        
        success = vector_service.switch_store(store_type)
        
        if success:
            logger.info(f"Store cambiado a {store_type}")
            return jsonify({
                'success': True,
                'active_store': store_type,
                'message': f'Switched to {store_type.upper()}'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to switch to {store_type}'
            }), 400
            
    except Exception as e:
        logger.error(f"Error cambiando store: {e}")
        return jsonify({'error': str(e)}), 500

@stores_api_bp.route('/api/stores/benchmark/<store_type>', methods=['POST'])
def benchmark_store(store_type):
    """Ejecutar benchmark en un store específico"""
    try:
        vector_service = get_vector_store_service()
        
        if store_type not in ['faiss', 'chromadb']:
            return jsonify({'error': 'Invalid store type'}), 400
        
        # Query de prueba
        test_query = request.json.get('query', 'administración municipal servicios ciudadanos') if request.json else 'administración municipal servicios ciudadanos'
        
        start_time = time.time()
        
        # Ejecutar búsqueda según el store
        if store_type == 'faiss' and vector_service.faiss_available:
            from app.services.rag.embeddings import embedding_service
            query_embedding = embedding_service.encode_single_text(test_query)
            if query_embedding is not None:
                results = vector_service.faiss_store.search(query_embedding, k=5)
            else:
                results = []
        elif store_type == 'chromadb' and vector_service.chromadb_available:
            results = vector_service.chromadb_store.similarity_search(test_query, k=5)
        else:
            return jsonify({'error': f'Store {store_type} not available'}), 400
        
        search_time = (time.time() - start_time) * 1000
        
        return jsonify({
            'success': True,
            'store': store_type,
            'search_time': round(search_time, 2),
            'results_count': len(results),
            'query': test_query
        })
        
    except Exception as e:
        logger.error(f"Error en benchmark {store_type}: {e}")
        return jsonify({'error': str(e)}), 500

@stores_api_bp.route('/api/stores/benchmark/full', methods=['POST'])
def run_full_benchmark():
    """Ejecutar benchmark completo comparativo"""
    try:
        vector_service = get_vector_store_service()
        
        test_queries = [
            'servicios municipales administración',
            'certificados digitales ciudadanos',
            'tramites online ayuntamiento',
            'padrón municipal vecinos'
        ]
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'queries': test_queries,
            'stores': {}
        }
        
        # Benchmark FAISS
        if vector_service.faiss_available:
            faiss_times = []
            faiss_results_count = []
            
            for query in test_queries:
                start_time = time.time()
                
                from app.services.rag.embeddings import embedding_service
                query_embedding = embedding_service.encode_single_text(query)
                if query_embedding is not None:
                    search_results = vector_service.faiss_store.search(query_embedding, k=5)
                else:
                    search_results = []
                
                search_time = (time.time() - start_time) * 1000
                
                faiss_times.append(search_time)
                faiss_results_count.append(len(search_results))
            
            results['stores']['faiss'] = {
                'avg_time': round(sum(faiss_times) / len(faiss_times), 2),
                'times': [round(t, 2) for t in faiss_times],
                'avg_results': round(sum(faiss_results_count) / len(faiss_results_count), 1),
                'total_queries': len(test_queries)
            }
        
        # Benchmark ChromaDB
        if vector_service.chromadb_available:
            chromadb_times = []
            chromadb_results_count = []
            
            for query in test_queries:
                start_time = time.time()
                search_results = vector_service.chromadb_store.similarity_search(query, k=5)
                search_time = (time.time() - start_time) * 1000
                
                chromadb_times.append(search_time)
                chromadb_results_count.append(len(search_results))
            
            results['stores']['chromadb'] = {
                'avg_time': round(sum(chromadb_times) / len(chromadb_times), 2),
                'times': [round(t, 2) for t in chromadb_times],
                'avg_results': round(sum(chromadb_results_count) / len(chromadb_results_count), 1),
                'total_queries': len(test_queries)
            }
        
        # Calcular comparación
        if 'faiss' in results['stores'] and 'chromadb' in results['stores']:
            faiss_avg = results['stores']['faiss']['avg_time']
            chromadb_avg = results['stores']['chromadb']['avg_time']
            
            if faiss_avg > 0:
                results['comparison'] = {
                    'faster_store': 'faiss' if faiss_avg < chromadb_avg else 'chromadb',
                    'speedup': round(max(faiss_avg, chromadb_avg) / min(faiss_avg, chromadb_avg), 2)
                }
        
        logger.info("Full benchmark completed", results=results)
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error en benchmark completo: {e}")
        return jsonify({'error': str(e)}), 500

@stores_api_bp.route('/api/stores/test-query', methods=['POST'])
def run_test_query():
    """Ejecutar query de prueba"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        k = int(data.get('k', 5))
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        vector_service = get_vector_store_service()
        
        start_time = time.time()
        results = vector_service.search(query, k=k)
        search_time = (time.time() - start_time) * 1000
        
        # Formatear resultados para el frontend
        formatted_results = []
        for result in results:
            if isinstance(result, dict):
                formatted_results.append({
                    'content': result.get('content', '')[:200],  # Límitar contenido
                    'score': result.get('score', 0.0),
                    'id': result.get('id', ''),
                    'metadata': result.get('metadata', {})
                })
        
        return jsonify({
            'success': True,
            'query': query,
            'search_time': round(search_time, 2),
            'results': formatted_results,
            'store_used': vector_service.get_active_store_type()
        })
        
    except Exception as e:
        logger.error(f"Error en test query: {e}")
        return jsonify({'error': str(e)}), 500

@stores_api_bp.route('/api/stores/configuration', methods=['GET'])
def get_configuration():
    """Obtener configuración actual del sistema"""
    try:
        vector_service = get_vector_store_service()
        embedding_service = get_embedding_service()
        
        config = {
            'vector_store': {
                'active_store': vector_service.get_active_store_type(),
                'preferred_store': vector_service.preferred_store,
                'fallback_enabled': vector_service.enable_fallback,
                'available_stores': {
                    'faiss': vector_service.faiss_available,
                    'chromadb': vector_service.chromadb_available
                }
            },
            'embedding_service': {
                'model_name': embedding_service.model_name,
                'dimension': embedding_service.dimension,
                'cache_size': getattr(embedding_service, 'cache_size', 10000),
                'available': embedding_service.is_available()
            },
            'search': {
                'default_k': 5,
                'max_k': 20
            }
        }
        
        return jsonify(config)
        
    except Exception as e:
        logger.error(f"Error obteniendo configuración: {e}")
        return jsonify({'error': str(e)}), 500

@stores_api_bp.route('/api/stores/configuration', methods=['POST'])
def update_configuration():
    """Actualizar configuración del sistema"""
    try:
        data = request.get_json()
        vector_service = get_vector_store_service()
        
        # Cambiar store activo si se especifica
        if 'primary_store' in data:
            success = vector_service.switch_store(data['primary_store'])
            if not success:
                return jsonify({'error': 'Failed to switch store'}), 400
        
        # Otras configuraciones se pueden implementar aquí
        # Por ejemplo, cambiar parámetros de embedding service
        
        logger.info("Configuration updated", data=data)
        return jsonify({'success': True, 'message': 'Configuration updated'})
        
    except Exception as e:
        logger.error(f"Error actualizando configuración: {e}")
        return jsonify({'error': str(e)}), 500

@stores_api_bp.route('/api/stores/clear', methods=['POST'])
def clear_all_stores():
    """Limpiar todos los vector stores"""
    try:
        vector_service = get_vector_store_service()
        
        cleared_stores = []
        
        if vector_service.faiss_available:
            try:
                vector_service.faiss_store.clear()
                cleared_stores.append('faiss')
            except Exception as e:
                logger.error(f"Error clearing FAISS: {e}")
        
        if vector_service.chromadb_available:
            try:
                vector_service.chromadb_store.clear_all()
                cleared_stores.append('chromadb')
            except Exception as e:
                logger.error(f"Error clearing ChromaDB: {e}")
        
        if cleared_stores:
            logger.warning(f"Stores cleared: {cleared_stores}")
            return jsonify({
                'success': True,
                'cleared_stores': cleared_stores,
                'message': f'Cleared {len(cleared_stores)} stores'
            })
        else:
            return jsonify({'error': 'No stores were cleared'}), 400
            
    except Exception as e:
        logger.error(f"Error clearing stores: {e}")
        return jsonify({'error': str(e)}), 500

@stores_api_bp.route('/api/stores/logs')
def get_system_logs():
    """Obtener logs recientes del sistema"""
    try:
        # En una implementación real, leerías los logs desde archivos
        # Por ahora, devolvemos logs simulados basados en el historial real
        
        logs = []
        
        # Agregar algunos logs basados en métricas recientes
        if metrics_history:
            recent_metrics = metrics_history[-10:]  # Últimos 10 registros
            
            for metric in recent_metrics:
                logs.append({
                    'timestamp': metric['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'level': 'info',
                    'message': f"System metrics updated - Store: {metric['active_store']}, Search time: {metric['avg_search_time']:.2f}ms"
                })
        
        # Agregar logs de estado del sistema
        vector_service = get_vector_store_service()
        health = vector_service.get_health_status()
        
        logs.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'level': 'success' if health['overall_status'] == 'healthy' else 'warning',
            'message': f"System health check: {health['overall_status']}"
        })
        
        # Ordenar por timestamp (más recientes primero)
        logs.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({'logs': logs[:50]})  # Últimos 50 logs
        
    except Exception as e:
        logger.error(f"Error obteniendo logs: {e}")
        return jsonify({'error': str(e)}), 500

@stores_api_bp.route('/api/stores/metrics/history')
def get_metrics_history():
    """Obtener historial de métricas para gráficos"""
    try:
        global metrics_history
        
        if not metrics_history:
            return jsonify({'history': []})
        
        formatted_history = []
        for metric in metrics_history:
            formatted_history.append({
                'timestamp': metric['timestamp'].isoformat(),
                'avg_search_time': metric['avg_search_time'],
                'memory_usage': metric['memory_usage'],
                'active_store': metric['active_store']
            })
        
        return jsonify({'history': formatted_history})
        
    except Exception as e:
        logger.error(f"Error obteniendo historial de métricas: {e}")
        return jsonify({'error': str(e)}), 500