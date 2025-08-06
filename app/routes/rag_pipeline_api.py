"""
Endpoints API adicionales para Pipeline RAG Completo
Añadir a app/routes/api.py o crear como archivo separado
TFM Vicente Caruncho - Sistemas Inteligentes
"""

from flask import Blueprint, request, jsonify, g
from werkzeug.utils import secure_filename
import os
from typing import Dict, Any, List
import time

# Imports del pipeline RAG completo
from app.services.rag_pipeline import (
    rag_pipeline, 
    process_rag_query, 
    compare_rag_models,
    ingest_document,
    get_pipeline_statistics
)
from app.core.logger import get_logger

# Si estás añadiendo a api.py existente, omite esta línea:
# rag_pipeline_bp = Blueprint('rag_pipeline', __name__, url_prefix='/api/rag')

logger = get_logger("rag_pipeline_api")

# =============================================================================
# ENDPOINTS DEL PIPELINE RAG COMPLETO
# =============================================================================

@api_bp.route('/rag/pipeline/query', methods=['POST'])
@rate_limit(30)  # Más restrictivo por ser computacionalmente intensivo
def rag_pipeline_query():
    """Procesar consulta completa a través del pipeline RAG optimizado"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "status": "error",
                "message": "Campo 'query' requerido"
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                "status": "error", 
                "message": "Query no puede estar vacío"
            }), 400
        
        # Parámetros del pipeline
        provider = data.get('provider', 'ollama')
        model_name = data.get('model_name')
        k = data.get('k', 5)
        temperature = data.get('temperature', 0.7)
        use_rag = data.get('use_rag', True)
        
        # Validaciones
        if k < 1 or k > 20:
            return jsonify({
                "status": "error",
                "message": "k debe estar entre 1 y 20"
            }), 400
        
        if temperature < 0 or temperature > 2:
            return jsonify({
                "status": "error",
                "message": "temperature debe estar entre 0 y 2"
            }), 400
        
        # Procesar con pipeline RAG completo
        result = process_rag_query(
            query=query,
            provider=provider,
            model_name=model_name,
            k=k,
            temperature=temperature,
            use_rag=use_rag
        )
        
        logger.info("Pipeline RAG query procesada",
                   query_length=len(query),
                   provider=provider,
                   chunks_found=len(result.context_chunks),
                   pipeline_time=result.pipeline_time,
                   success=not bool(result.llm_response.error))
        
        response_data = result.to_dict()
        
        return jsonify({
            "status": "success",
            "data": response_data
        }), 200
        
    except Exception as e:
        logger.error("Error en pipeline RAG query", error=str(e))
        return jsonify({
            "status": "error",
            "message": f"Error en pipeline RAG: {str(e)}"
        }), 500

@api_bp.route('/rag/pipeline/compare', methods=['POST'])
@rate_limit(5)  # Muy restrictivo por usar recursos intensivos
def rag_pipeline_compare():
    """Comparación completa entre modelos usando pipeline RAG"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "status": "error",
                "message": "Campo 'query' requerido"
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                "status": "error",
                "message": "Query no puede estar vacío"
            }), 400
        
        # Parámetros de comparación
        local_model = data.get('local_model')
        openai_model = data.get('openai_model')
        k = data.get('k', 5)
        temperature = data.get('temperature', 0.7)
        
        # Procesar comparación con pipeline RAG
        result = compare_rag_models(
            query=query,
            local_model=local_model,
            openai_model=openai_model,
            k=k,
            temperature=temperature
        )
        
        logger.info("Comparación pipeline RAG completada",
                   query_length=len(query),
                   chunks_used=len(result.context_chunks),
                   total_time=result.total_time,
                   local_success=not bool(result.local_result.error),
                   openai_success=not bool(result.openai_result.error))
        
        response_data = result.to_dict()
        
        return jsonify({
            "status": "success",
            "data": response_data
        }), 200
        
    except Exception as e:
        logger.error("Error en comparación pipeline RAG", error=str(e))
        return jsonify({
            "status": "error", 
            "message": f"Error en comparación RAG: {str(e)}"
        }), 500

@api_bp.route('/rag/pipeline/stats', methods=['GET'])
@rate_limit(60)
def get_rag_pipeline_stats():
    """Estadísticas completas del pipeline RAG"""
    try:
        stats = get_pipeline_statistics()
        
        return jsonify({
            "status": "success",
            "data": stats
        }), 200
        
    except Exception as e:
        logger.error("Error obteniendo stats pipeline", error=str(e))
        return jsonify({
            "status": "error",
            "message": f"Error obteniendo estadísticas: {str(e)}"
        }), 500

# =============================================================================
# ENDPOINTS DE INGESTA INTEGRADA
# =============================================================================

@api_bp.route('/rag/ingest/file', methods=['POST'])
@rate_limit(10)  # Restrictivo por ser operación pesada
def ingest_file_to_rag():
    """Ingerir un archivo directamente al sistema RAG"""
    try:
        # Verificar que hay archivo
        if 'file' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No se encontró archivo en la petición"
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "status": "error",
                "message": "No se seleccionó archivo"
            }), 400
        
        # Obtener metadatos adicionales
        metadata = {}
        if request.form.get('title'):
            metadata['title'] = request.form.get('title')
        if request.form.get('description'):
            metadata['description'] = request.form.get('description')
        if request.form.get('category'):
            metadata['category'] = request.form.get('category')
        
        source_type = request.form.get('source_type', 'uploaded_file')
        
        # Guardar archivo temporalmente
        filename = secure_filename(file.filename)
        temp_path = os.path.join('/tmp', filename)  # En producción usar directorio apropiado
        file.save(temp_path)
        
        try:
            # Ingerir usando pipeline RAG
            result = ingest_document(
                file_path=temp_path,
                source_type=source_type,
                metadata=metadata
            )
            
            # Limpiar archivo temporal
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            logger.info("Archivo ingerido via API",
                       filename=filename,
                       success=result['success'],
                       chunks_created=result.get('chunks_created', 0),
                       processing_time=result.get('processing_time', 0))
            
            return jsonify({
                "status": "success" if result['success'] else "error",
                "data": result
            }), 200 if result['success'] else 500
            
        except Exception as e:
            # Limpiar archivo temporal en caso de error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
        
    except Exception as e:
        logger.error("Error ingesta archivo via API", error=str(e))
        return jsonify({
            "status": "error",
            "message": f"Error procesando archivo: {str(e)}"
        }), 500

@api_bp.route('/rag/ingest/batch', methods=['POST'])
@rate_limit(2)  # Muy restrictivo por ser operación muy pesada
def batch_ingest_files():
    """Ingesta batch de múltiples archivos"""
    try:
        # Verificar archivos
        if 'files' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No se encontraron archivos en la petición"
            }), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({
                "status": "error",
                "message": "No se seleccionaron archivos válidos"
            }), 400
        
        max_workers = min(int(request.form.get('max_workers', 3)), 5)  # Máximo 5 workers
        
        # Guardar archivos temporalmente
        temp_paths = []
        for file in files:
            if file.filename != '':
                filename = secure_filename(file.filename)
                temp_path = os.path.join('/tmp', f"{int(time.time())}_{filename}")
                file.save(temp_path)
                temp_paths.append(temp_path)
        
        if not temp_paths:
            return jsonify({
                "status": "error", 
                "message": "No se pudieron procesar los archivos"
            }), 400
        
        try:
            # Procesar batch usando pipeline RAG
            result = rag_pipeline.batch_ingest_documents(
                file_paths=temp_paths,
                max_workers=max_workers
            )
            
            # Limpiar archivos temporales
            for temp_path in temp_paths:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            logger.info("Ingesta batch completada",
                       files_processed=result['total_files'],
                       successful=result['successful'],
                       failed=result['failed'],
                       processing_time=result['processing_time'])
            
            return jsonify({
                "status": "success" if result['success'] else "partial_success",
                "data": result
            }), 200
            
        except Exception as e:
            # Limpiar archivos temporales en caso de error
            for temp_path in temp_paths:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            raise e
        
    except Exception as e:
        logger.error("Error ingesta batch via API", error=str(e))
        return jsonify({
            "status": "error",
            "message": f"Error procesando archivos: {str(e)}"
        }), 500

# =============================================================================
# ENDPOINTS DE TESTING Y DEBUG
# =============================================================================

@api_bp.route('/rag/pipeline/test', methods=['GET'])
@rate_limit(5)
def test_rag_pipeline():
    """Test end-to-end del pipeline RAG completo"""
    try:
        from app.services.rag_pipeline import test_pipeline_end_to_end
        
        # Ejecutar test completo
        success = test_pipeline_end_to_end()
        
        # Obtener estadísticas adicionales
        stats = get_pipeline_statistics()
        
        return jsonify({
            "status": "success",
            "data": {
                "test_passed": success,
                "pipeline_stats": stats,
                "timestamp": time.time()
            }
        }), 200
        
    except Exception as e:
        logger.error("Error en test pipeline RAG", error=str(e))
        return jsonify({
            "status": "error",
            "message": f"Error en test: {str(e)}"
        }), 500

@api_bp.route('/rag/pipeline/benchmark', methods=['POST'])
@rate_limit(2)  # Muy restrictivo por ser muy intensivo
def benchmark_rag_pipeline():
    """Benchmark comparativo del pipeline RAG completo"""
    try:
        data = request.get_json()
        
        # Parámetros del benchmark
        queries = data.get('queries', [
            "¿Qué es una licencia de obras?",
            "¿Cuáles son los plazos administrativos?",
            "¿Qué documentación necesito para un permiso?"
        ])
        
        providers = data.get('providers', ['ollama', 'openai'])
        k_values = data.get('k_values', [3, 5])
        
        results = []
        
        for query in queries:
            for provider in providers:
                for k in k_values:
                    try:
                        start_time = time.time()
                        
                        result = process_rag_query(
                            query=query,
                            provider=provider,
                            k=k,
                            temperature=0.7,
                            use_rag=True
                        )
                        
                        benchmark_result = {
                            'query': query,
                            'provider': provider,
                            'k': k,
                            'success': not bool(result.llm_response.error),
                            'pipeline_time': result.pipeline_time,
                            'retrieval_time': result.retrieval_time,
                            'generation_time': result.generation_time,
                            'context_chunks': len(result.context_chunks),
                            'response_length': len(result.llm_response.response),
                            'confidence_score': result.confidence_score,
                            'error': result.llm_response.error
                        }
                        
                        if result.llm_response.estimated_cost:
                            benchmark_result['estimated_cost'] = result.llm_response.estimated_cost
                        
                        results.append(benchmark_result)
                        
                    except Exception as e:
                        logger.error("Error en benchmark individual",
                                   query=query,
                                   provider=provider,
                                   k=k,
                                   error=str(e))
                        
                        results.append({
                            'query': query,
                            'provider': provider, 
                            'k': k,
                            'success': False,
                            'error': str(e)
                        })
        
        # Calcular estadísticas del benchmark
        successful_results = [r for r in results if r['success']]
        
        benchmark_stats = {
            'total_tests': len(results),
            'successful_tests': len(successful_results),
            'success_rate': len(successful_results) / len(results) if results else 0,
            'avg_pipeline_time': sum(r['pipeline_time'] for r in successful_results) / len(successful_results) if successful_results else 0,
            'avg_confidence': sum(r['confidence_score'] for r in successful_results) / len(successful_results) if successful_results else 0
        }
        
        logger.info("Benchmark RAG completado",
                   total_tests=benchmark_stats['total_tests'],
                   success_rate=benchmark_stats['success_rate'],
                   avg_pipeline_time=benchmark_stats['avg_pipeline_time'])
        
        return jsonify({
            "status": "success",
            "data": {
                "results": results,
                "stats": benchmark_stats,
                "timestamp": time.time()
            }
        }), 200
        
    except Exception as e:
        logger.error("Error en benchmark RAG", error=str(e))
        return jsonify({
            "status": "error",
            "message": f"Error en benchmark: {str(e)}"
        }), 500

# =============================================================================
# ENDPOINTS DE GESTIÓN Y MANTENIMIENTO
# =============================================================================

@api_bp.route('/rag/pipeline/clear', methods=['POST'])
@rate_limit(2)  # Muy restrictivo por ser destructivo
def clear_rag_index():
    """Limpiar completamente el índice RAG"""
    try:
        # Verificar autorización (en producción implementar autenticación)
        auth_token = request.headers.get('Authorization')
        if not auth_token or auth_token != 'Bearer admin-token-change-in-production':
            return jsonify({
                "status": "error",
                "message": "No autorizado para esta operación"
            }), 403
        
        # Limpiar índice
        from app.services import rag_service
        success = rag_service.clear_index()
        
        logger.warning("Índice RAG limpiado via API", success=success)
        
        return jsonify({
            "status": "success" if success else "error",
            "message": "Índice RAG limpiado" if success else "Error limpiando índice",
            "timestamp": time.time()
        }), 200 if success else 500
        
    except Exception as e:
        logger.error("Error limpiando índice RAG", error=str(e))
        return jsonify({
            "status": "error",
            "message": f"Error limpiando índice: {str(e)}"
        }), 500

@api_bp.route('/rag/pipeline/rebuild', methods=['POST'])
@rate_limit(1)  # Muy restrictivo por ser operación muy pesada
def rebuild_rag_index():
    """Reconstruir completamente el índice RAG desde archivos"""
    try:
        data = request.get_json() or {}
        
        # Verificar autorización
        auth_token = request.headers.get('Authorization')
        if not auth_token or auth_token != 'Bearer admin-token-change-in-production':
            return jsonify({
                "status": "error",
                "message": "No autorizado para esta operación"
            }), 403
        
        # Directorio fuente para reconstruir
        source_dir = data.get('source_directory', 'data/documents')
        max_workers = min(data.get('max_workers', 2), 3)  # Máximo 3 workers
        
        if not os.path.exists(source_dir):
            return jsonify({
                "status": "error",
                "message": f"Directorio fuente no existe: {source_dir}"
            }), 400
        
        # Encontrar archivos para procesar
        supported_extensions = {'.pdf', '.docx', '.txt', '.md', '.csv', '.xlsx'}
        files_to_process = []
        
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    files_to_process.append(os.path.join(root, file))
        
        if not files_to_process:
            return jsonify({
                "status": "error",
                "message": f"No se encontraron archivos procesables en {source_dir}"
            }), 400
        
        # Limpiar índice existente
        from app.services import rag_service
        rag_service.clear_index()
        
        # Procesar archivos en batch
        result = rag_pipeline.batch_ingest_documents(
            file_paths=files_to_process,
            max_workers=max_workers
        )
        
        logger.warning("Índice RAG reconstruido",
                      source_dir=source_dir,
                      files_processed=result['total_files'],
                      successful=result['successful'],
                      processing_time=result['processing_time'])
        
        return jsonify({
            "status": "success" if result['success'] else "partial_success",
            "message": f"Índice reconstruido desde {source_dir}",
            "data": result
        }), 200
        
    except Exception as e:
        logger.error("Error reconstruyendo índice RAG", error=str(e))
        return jsonify({
            "status": "error",
            "message": f"Error reconstruyendo índice: {str(e)}"
        }), 500

# =============================================================================
# DOCUMENTACIÓN ADICIONAL PARA PIPELINE RAG
# =============================================================================

def get_rag_pipeline_documentation():
    """Documentación específica del pipeline RAG"""
    return {
        'title': 'Pipeline RAG Completo API',
        'version': '1.0.0',
        'description': 'API completa para sistema RAG con comparación de modelos',
        'pipeline_endpoints': {
            'consulta_optimizada': {
                'POST /api/rag/pipeline/query': {
                    'description': 'Consulta completa a través del pipeline RAG optimizado',
                    'body': {
                        'query': 'string (requerido)',
                        'provider': 'ollama|openai (default: ollama)',
                        'model_name': 'string (opcional)',
                        'k': 'integer 1-20 (default: 5)',
                        'temperature': 'float 0-2 (default: 0.7)',
                        'use_rag': 'boolean (default: true)'
                    }
                }
            },
            'comparacion_avanzada': {
                'POST /api/rag/pipeline/compare': {
                    'description': 'Comparación completa entre modelos usando pipeline RAG',
                    'body': {
                        'query': 'string (requerido)',
                        'local_model': 'string (opcional)',
                        'openai_model': 'string (opcional)',
                        'k': 'integer 1-20 (default: 5)',
                        'temperature': 'float 0-2 (default: 0.7)'
                    }
                }
            },
            'ingesta_integrada': {
                'POST /api/rag/ingest/file': 'Ingerir archivo directamente al RAG',
                'POST /api/rag/ingest/batch': 'Ingesta batch de múltiples archivos'
            },
            'testing_benchmark': {
                'GET /api/rag/pipeline/test': 'Test end-to-end del pipeline',
                'POST /api/rag/pipeline/benchmark': 'Benchmark comparativo completo'
            },
            'mantenimiento': {
                'POST /api/rag/pipeline/clear': 'Limpiar índice RAG (requiere auth)',
                'POST /api/rag/pipeline/rebuild': 'Reconstruir índice desde archivos'
            }
        },
        'rate_limits': {
            'queries': '30 requests/minute',
            'comparisons': '5 requests/minute', 
            'ingestion': '10 requests/minute',
            'batch_operations': '2 requests/minute',
            'maintenance': '1-2 requests/minute'
        },
        'authentication': {
            'public_endpoints': 'La mayoría de consultas',
            'protected_endpoints': 'Operaciones de mantenimiento requieren Bearer token'
        }
    }

# Añadir a la documentación principal si existe endpoint /api/docs
@api_bp.route('/docs/rag-pipeline', methods=['GET'])
def rag_pipeline_docs():
    """Documentación específica del pipeline RAG"""
    return jsonify(get_rag_pipeline_documentation())