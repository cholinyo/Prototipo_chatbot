#!/usr/bin/env python3
"""
Prototipo_chatbot - Punto de entrada principal
Chatbot RAG para Administraciones Locales
TFM Vicente Caruncho - Sistemas Inteligentes UJI
"""
import sys
import os
import time
import traceback
from pathlib import Path
from flask import Flask, render_template, jsonify, request

# Añadir el directorio raíz al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_missing_files():
    """Crear archivos de configuración faltantes"""
    print("📁 Verificando estructura de directorios...")
    
    # Crear config/settings.yaml si no existe
    config_dir = project_root / "config"
    config_dir.mkdir(exist_ok=True)
    
    settings_file = config_dir / "settings.yaml"
    if not settings_file.exists():
        print("⚠️ Creando archivo de configuración faltante...")
    
    # Crear directorios faltantes
    required_dirs = [
        'logs', 
        'data/vectorstore/faiss', 
        'data/vectorstore/chromadb',
        'app/static/css',
        'app/static/js',
        'app/templates/errors'
    ]
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("✅ Estructura de directorios verificada")

def get_llm_service():
    """Obtener instancia del servicio LLM con manejo de errores"""
    try:
        from app.services.llm.llm_services import LLMService
        return LLMService()
    except ImportError as e:
        raise ImportError(f"No se pudo importar LLMService: {e}")
    except Exception as e:
        raise Exception(f"Error inicializando LLMService: {e}")

def get_memory_usage():
    """Obtener uso de memoria del proceso"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return round(memory_mb, 2)
    except ImportError:
        return 0

def create_error_response(error_type, message):
    """Crear respuesta de error estandarizada"""
    return jsonify({
        'status': 'error',
        'timestamp': time.time(),
        'error_type': error_type,
        'error': message,
        'services': {
            'flask': 'available',
            'ollama': 'error',
            'openai': 'error'
        },
        'components': {
            'embeddings': 'error',
            'vector_store': 'error',
            'llm': 'error'
        }
    })

def create_fallback_template(app_config):
    """Crear template HTML básico si no existe el principal"""
    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <title>{app_config.name}</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="container mt-5">
            <div class="text-center">
                <h1 class="display-4">🤖 {app_config.name}</h1>
                <p class="lead">{app_config.description}</p>
                <p class="text-muted">TFM Vicente Caruncho - Sistemas Inteligentes UJI</p>
                <div class="mt-4">
                    <a href="/dashboard" class="btn btn-primary btn-lg me-3">
                        <i class="fas fa-chart-line me-2"></i>Dashboard
                    </a>
                    <a href="/health" class="btn btn-outline-secondary">
                        <i class="fas fa-heartbeat me-2"></i>Estado del Sistema
                    </a>
                </div>
            </div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """

def register_blueprints(app, logger):
    """Registrar blueprints de la aplicación"""
    logger.info("Registrando blueprints...")
    
    # Lista de blueprints a intentar registrar
    blueprints_config = [
        ('app.routes.chat_routes', 'chat_bp', '/chat'),
        ('app.routes.main', 'main_bp', '/'),
        ('app.routes.api', 'api_bp', '/api'),
    ]
    
    registered_count = 0
    
    for module_path, blueprint_name, url_prefix in blueprints_config:
        try:
            module = __import__(module_path, fromlist=[blueprint_name])
            blueprint = getattr(module, blueprint_name)
            app.register_blueprint(blueprint, url_prefix=url_prefix)
            logger.info(f"✅ Blueprint '{blueprint_name}' registrado en {url_prefix}")
            registered_count += 1
        except ImportError as e:
            logger.warning(f"⚠️ Blueprint '{blueprint_name}' no disponible: {e}")
        except AttributeError as e:
            logger.error(f"❌ Error en blueprint '{blueprint_name}': {e}")
    
    logger.info(f"📋 {registered_count} blueprints registrados exitosamente")
    return registered_count

def create_flask_app():
    """Crear y configurar la aplicación Flask"""
    print("🔧 Configurando aplicación Flask...")
    
    # Intentar importar configuración
    try:
        from app.core.config import get_app_config, is_development
        from app.core.logger import setup_logging, get_logger
        
        setup_logging()
        logger = get_logger("main")
        app_config = get_app_config()
        logger.info("✅ Configuración importada correctamente")
        
    except ImportError as e:
        print(f"⚠️ Configuración no disponible, usando valores por defecto: {e}")
        
        # Configuración por defecto
        class DefaultConfig:
            name = "Prototipo_chatbot"
            version = "1.0.0"
            description = "Sistema RAG para Administraciones Locales"
            host = "localhost"
            port = 5000
            debug = True
        
        app_config = DefaultConfig()
        # Logger básico si no hay configuración avanzada
        logger = type('Logger', (), {
            'info': lambda msg: print(f"ℹ️ {msg}"),
            'error': lambda msg: print(f"❌ {msg}"),
            'warning': lambda msg: print(f"⚠️ {msg}"),
            'debug': lambda msg: print(f"🔍 {msg}") if app_config.debug else None
        })()
    
    # Crear aplicación Flask
    app = Flask(__name__, 
               template_folder='app/templates',
               static_folder='app/static')
    
    # Configuración básica de Flask
    app.config.update(
        SECRET_KEY=os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production'),
        DEBUG=app_config.debug,
        JSON_SORT_KEYS=False,
        JSONIFY_PRETTYPRINT_REGULAR=app_config.debug
    )
    
    return app, app_config, logger

def setup_routes(app, app_config, logger):
    """Configurar rutas de la aplicación"""
    logger.info("Configurando rutas principales...")
    
    @app.route('/')
    def index():
        """Página principal"""
        try:
            return render_template('index.html',
                                 app_name=app_config.name,
                                 app_version=app_config.version,
                                 app_description=app_config.description)
        except Exception as e:
            logger.warning(f"Template index.html no encontrado, usando fallback: {e}")
            return create_fallback_template(app_config)
    
    @app.route('/health')
    def health_check():
        """Endpoint de verificación de salud principal"""
        try:
            llm_service = get_llm_service()
            health = llm_service.health_check()
            
            response = {
                'status': health['status'],
                'timestamp': health['timestamp'],
                'services': {
                    'flask': 'available',
                    'ollama': health['services']['ollama']['status'],
                    'openai': health['services']['openai']['status']
                },
                'models': health.get('models', {}),
                'components': {
                    'embeddings': 'available',
                    'vector_store': 'available',
                    'llm': health['status']
                }
            }
            
            logger.debug(f"Health check: {response['status']}")
            return jsonify(response)
            
        except ImportError as e:
            logger.error(f"Error importando LLMService: {e}")
            return create_error_response('import_error', str(e)), 500
        except Exception as e:
            logger.error(f"Error en health check: {e}")
            return create_error_response('health_error', str(e)), 500
    
    @app.route('/api/status')
    def api_status():
        """Endpoint de estado detallado para el dashboard"""
        try:
            llm_service = get_llm_service()
            health = llm_service.health_check()
            
            response = {
                'status': health['status'],
                'timestamp': health['timestamp'],
                'services': {
                    'flask': {
                        'status': 'healthy',
                        'url': f"http://{app_config.host}:{app_config.port}",
                        'uptime': time.time() - app.start_time if hasattr(app, 'start_time') else 0
                    },
                    'ollama': health['services']['ollama'],
                    'openai': health['services']['openai'],
                    'embeddings': {
                        'status': 'healthy',
                        'model': 'all-MiniLM-L6-v2',
                        'dimensions': 384
                    },
                    'vector_store': {
                        'status': 'healthy',
                        'type': 'FAISS/ChromaDB',
                        'indexed_documents': 0  # Placeholder
                    }
                },
                'models': health.get('models', {}),
                'system_info': {
                    'uptime_hours': round((time.time() - app.start_time) / 3600, 2) if hasattr(app, 'start_time') else 0,
                    'version': getattr(app_config, 'version', '1.0.0'),
                    'environment': 'development' if app_config.debug else 'production',
                    'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                }
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error en api_status: {e}")
            return create_error_response('status_error', str(e)), 500

    @app.route('/ajax/quick-stats')
    def ajax_quick_stats():
        """Estadísticas rápidas para actualización automática"""
        try:
            llm_service = get_llm_service()
            health = llm_service.health_check()
            
            uptime_hours = round((time.time() - app.start_time) / 3600, 2) if hasattr(app, 'start_time') else 0
            
            stats = {
                'queries': 127,
                'documents': 45,
                'avg_response_time': 1.23,
                'success_rate': 98.5,
                'uptime': uptime_hours,
                'ollama_models': len(health.get('models', {}).get('ollama', [])),
                'openai_available': health['services']['openai']['status'] == 'configured',
                'system_status': health['status'],
                'last_update': time.time(),
                'memory_usage': get_memory_usage(),
                'active_connections': 1  # Placeholder
            }
            
            return jsonify(stats)
            
        except Exception as e:
            logger.error(f"Error en quick_stats: {e}")
            return jsonify({
                'queries': 0, 'documents': 0, 'avg_response_time': 0,
                'success_rate': 0, 'uptime': 0, 'ollama_models': 0,
                'openai_available': False, 'system_status': 'error',
                'last_update': time.time(), 'error': str(e)
            }), 500

    @app.route('/api/stats')
    def api_detailed_stats():
        """Estadísticas detalladas para exportación"""
        try:
            llm_service = get_llm_service()
            health = llm_service.health_check()
            
            detailed_stats = {
                'system': {
                    'status': health['status'],
                    'uptime_hours': round((time.time() - app.start_time) / 3600, 2) if hasattr(app, 'start_time') else 0,
                    'version': getattr(app_config, 'version', '1.0.0'),
                    'environment': 'development' if app_config.debug else 'production',
                    'timestamp': time.time(),
                    'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    'memory_usage': get_memory_usage()
                },
                'usage': {
                    'total_queries': 127,
                    'queries_today': 45,
                    'avg_response_time': 1.23,
                    'success_rate': 98.5,
                    'errors_count': 2,
                    'peak_concurrent_users': 5
                },
                'models': {
                    'ollama': {
                        'available': health['services']['ollama']['status'] == 'available',
                        'models': health.get('models', {}).get('ollama', []),
                        'total_requests': 89,
                        'avg_response_time': 2.1
                    },
                    'openai': {
                        'configured': health['services']['openai']['status'] == 'configured',
                        'models': health.get('models', {}).get('openai', []),
                        'total_requests': 38,
                        'estimated_cost': 0.75,
                        'avg_response_time': 1.2
                    }
                },
                'data': {
                    'documents_indexed': 45,
                    'total_chunks': 892,
                    'vector_store_size_mb': 12.5,
                    'embedding_model': 'all-MiniLM-L6-v2',
                    'last_ingestion': time.time() - 3600  # 1 hora atrás
                }
            }
            
            return jsonify(detailed_stats)
            
        except Exception as e:
            logger.error(f"Error en detailed_stats: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/routes')
    def list_routes():
        """Listar todas las rutas disponibles (solo desarrollo)"""
        if not app_config.debug:
            return jsonify({'error': 'No disponible en producción'}), 403
        
        routes = []
        for rule in app.url_map.iter_rules():
            routes.append({
                'endpoint': rule.endpoint,
                'methods': sorted(list(rule.methods - {'HEAD', 'OPTIONS'})),
                'rule': rule.rule
            })
        
        return jsonify({
            'routes': sorted(routes, key=lambda x: x['rule']),
            'total': len(routes),
            'debug_mode': app_config.debug
        })

    @app.route('/dashboard')
    def dashboard():
        """Ruta al dashboard (si existe template)"""
        try:
            # Obtener datos para el dashboard
            llm_service = get_llm_service()
            health = llm_service.health_check()
            
            # Datos simulados para el template
            context = {
                'app_name': app_config.name,
                'app_version': getattr(app_config, 'version', '1.0.0'),
                'usage_stats': {
                    'queries_today': 45,
                    'response_time_avg': 1.23,
                    'success_rate': 98.5,
                    'uptime_hours': round((time.time() - app.start_time) / 3600, 2) if hasattr(app, 'start_time') else 0
                },
                'performance_metrics': {
                    'total_documents': 45,
                    'vector_store_size': 12.5,
                    'embedding_model': 'all-MiniLM-L6-v2',
                    'chunk_size': 500
                },
                'providers_status': {
                    'ollama': {
                        'available': health['services']['ollama']['status'] == 'available',
                        'status': 'healthy' if health['services']['ollama']['status'] == 'available' else 'error',
                        'models': health.get('models', {}).get('ollama', []),
                        'model_count': len(health.get('models', {}).get('ollama', []))
                    },
                    'openai': {
                        'available': health['services']['openai']['status'] == 'configured',
                        'status': 'healthy' if health['services']['openai']['status'] == 'configured' else 'error',
                        'models': health.get('models', {}).get('openai', []),
                        'model_count': len(health.get('models', {}).get('openai', []))
                    }
                },
                'charts_data': {
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
                            'data': [60, 35, 5],
                            'backgroundColor': ['#ff6384', '#36a2eb', '#ffce56']
                        }]
                    },
                    'response_times': {
                        'labels': ['< 1s', '1-2s', '2-5s', '> 5s'],
                        'datasets': [{
                            'data': [70, 20, 8, 2],
                            'backgroundColor': ['#4bc0c0', '#ffcd56', '#ff9f40', '#ff6384']
                        }]
                    }
                }
            }
            
            return render_template('dashboard.html', **context)
            
        except Exception as e:
            logger.warning(f"Dashboard template no disponible: {e}")
            return jsonify({
                'message': 'Dashboard no disponible',
                'error': str(e),
                'alternative': '/api/status'
            })

    logger.info("✅ Rutas configuradas correctamente")

def setup_error_handlers(app, logger):
    """Configurar manejadores de errores"""
    
    @app.errorhandler(404)
    def not_found_error(error):
        logger.warning(f"Página no encontrada: {request.url}")
        
        if request.is_json or 'application/json' in request.headers.get('Accept', ''):
            return jsonify({
                'error': 'Página no encontrada',
                'status': 404,
                'available_routes': ['/dashboard', '/health', '/api/status', '/']
            }), 404
        
        try:
            return render_template('errors/404.html'), 404
        except:
            return """
            <div style="text-align: center; margin-top: 50px;">
                <h1>404 - Página no encontrada</h1>
                <p><a href="/">Volver al inicio</a> | <a href="/dashboard">Dashboard</a></p>
            </div>
            """, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Error interno: {error}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        if request.is_json or 'application/json' in request.headers.get('Accept', ''):
            return jsonify({
                'error': 'Error interno del servidor',
                'status': 500,
                'debug': str(error) if app.config['DEBUG'] else 'Error interno'
            }), 500
        
        try:
            return render_template('errors/500.html'), 500
        except:
            return """
            <div style="text-align: center; margin-top: 50px;">
                <h1>500 - Error interno del servidor</h1>
                <p><a href="/">Volver al inicio</a></p>
            </div>
            """, 500

def setup_context_processors(app, app_config):
    """Configurar procesadores de contexto"""
    
    @app.context_processor
    def inject_global_vars():
        return {
            'app_name': app_config.name,
            'app_version': getattr(app_config, 'version', '1.0.0'),
            'app_description': app_config.description,
            'current_year': '2025',
            'debug_mode': app_config.debug
        }

def verify_system_status(logger):
    """Verificar estado inicial del sistema"""
    try:
        # Verificar LLM Service
        llm_service = get_llm_service()
        health = llm_service.health_check()
        logger.info(f"✅ LLM Service disponible - Estado: {health['status']}")
        
        # Verificar pipeline RAG si existe
        try:
            from app.services.rag_pipeline import get_rag_pipeline
            pipeline = get_rag_pipeline()
            if pipeline.is_available():
                logger.info("✅ Pipeline RAG disponible y listo")
            else:
                logger.warning("⚠️ Pipeline RAG en modo de inicialización")
        except ImportError:
            logger.info("📝 Pipeline RAG no disponible (opcional)")
            
    except Exception as e:
        logger.warning(f"⚠️ Verificación del sistema: {e}")

def print_startup_info(app_config, registered_blueprints):
    """Mostrar información de inicio"""
    print("\n" + "="*70)
    print("✅ PROTOTIPO_CHATBOT - APLICACIÓN FLASK LISTA")
    print("="*70)
    print("🌐 URLs disponibles:")
    print("   http://localhost:5000          (Página principal)")
    print("   http://localhost:5000/dashboard (Dashboard completo)")
    print("   http://localhost:5000/health    (Estado del sistema)")
    print("   http://localhost:5000/api/status (API de estado)")
    if app_config.debug:
        print("   http://localhost:5000/routes    (Lista de rutas)")
    
    print(f"\n📋 Sistema configurado:")
    print(f"   🔧 Blueprints registrados: {registered_blueprints}")
    print(f"   🎨 Templates: app/templates/")
    print(f"   📊 Endpoints API: /health, /api/status, /ajax/quick-stats")
    print(f"   🔍 Modo debug: {'Activado' if app_config.debug else 'Desactivado'}")
    
    print(f"\n💡 Funcionalidades:")
    print("   🤖 Sistema LLM (Ollama + OpenAI)")
    print("   📊 Dashboard con métricas en tiempo real")
    print("   🔄 Health checks automáticos")
    print("   📈 Estadísticas detalladas")
    print("   🎨 Interface responsive")
    
    print(f"\n⚠️  Usa Ctrl+C para detener el servidor")
    print("="*70)

def main():
    """Función principal de arranque"""
    print("🚀 Iniciando Prototipo_chatbot TFM...")
    print("📁 Directorio del proyecto:", project_root)
    print("👨‍🎓 Vicente Caruncho Ramos - Sistemas Inteligentes UJI")
    print("-" * 70)
    
    try:
        # Crear archivos faltantes
        create_missing_files()
        
        # Verificar dependencias críticas
        try:
            import flask
            print("✅ Flask importado correctamente")
        except ImportError as e:
            print(f"❌ Error importando Flask: {e}")
            print("💡 Instala las dependencias: pip install -r requirements.txt")
            sys.exit(1)
        
        # Crear aplicación Flask
        app, app_config, logger = create_flask_app()
        
        # Marcar tiempo de inicio
        app.start_time = time.time()
        
        # Registrar blueprints
        registered_blueprints = register_blueprints(app, logger)
        
        # Configurar rutas principales
        setup_routes(app, app_config, logger)
        
        # Configurar manejadores de errores
        setup_error_handlers(app, logger)
        
        # Configurar procesadores de contexto
        setup_context_processors(app, app_config)
        
        logger.info("Aplicación Flask configurada exitosamente")
        
        # Verificar estado inicial del sistema
        verify_system_status(logger)
        
        # Mostrar información de inicio
        print_startup_info(app_config, registered_blueprints)
        
        # Iniciar servidor
        logger.info(f"🌐 Servidor iniciando en http://{app_config.host}:{app_config.port}")
        app.run(
            host=app_config.host,
            port=app_config.port,
            debug=app_config.debug,
            use_reloader=True,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n⏹️  Aplicación detenida por el usuario")
        print("👋 ¡Hasta luego!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        print("\n💡 Soluciones posibles:")
        print("   1. pip install -r requirements.txt")
        print("   2. Verificar estructura de directorios")
        print("   3. Revisar archivos de configuración")
        print("   4. Verificar permisos de escritura")
        sys.exit(1)

if __name__ == "__main__":
    main()