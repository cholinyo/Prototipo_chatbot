#!/usr/bin/env python3
"""
Prototipo_chatbot - Punto de entrada principal
Chatbot RAG para Administraciones Locales
TFM Vicente Caruncho - Sistemas Inteligentes UJI
"""
import sys
import os
import time
from pathlib import Path
from flask import Flask, render_template, jsonify, request

# Añadir el directorio raíz al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_missing_files():
    """Crear archivos de configuración faltantes"""
    
    # Crear config/settings.yaml si no existe
    config_dir = project_root / "config"
    config_dir.mkdir(exist_ok=True)
    
    settings_file = config_dir / "settings.yaml"
    if not settings_file.exists():
        print("⚠️ Creando archivo de configuración faltante...")
        # El archivo se crea automáticamente por el ConfigManager si no existe
    
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

def register_blueprints(app):
    """Registrar blueprints de la aplicación"""
    print("📋 Registrando blueprints...")
    
    # Registrar blueprint de chat RAG
    try:
        from app.routes.chat_routes import chat_bp
        app.register_blueprint(chat_bp, url_prefix='/chat')
        print("✅ Blueprint 'chat' registrado en /chat")
    except ImportError as e:
        print(f"⚠️ Blueprint 'chat' no disponible: {e}")
    
    # Registrar otros blueprints si existen
    blueprints_to_try = [
        ('app.routes.main', 'main_bp', '/'),
        ('app.routes.api', 'api_bp', '/api'),
    ]
    
    for module_path, blueprint_name, url_prefix in blueprints_to_try:
        try:
            module = __import__(module_path, fromlist=[blueprint_name])
            blueprint = getattr(module, blueprint_name)
            app.register_blueprint(blueprint, url_prefix=url_prefix)
            print(f"✅ Blueprint '{blueprint_name}' registrado en {url_prefix}")
        except ImportError:
            print(f"⚠️ Blueprint '{blueprint_name}' no disponible (opcional)")
        except AttributeError:
            print(f"❌ Error en blueprint '{blueprint_name}': atributo no encontrado")

def create_flask_app():
    """Crear y configurar la aplicación Flask"""
    
    # Importar configuración
    try:
        from app.core.config import get_app_config, is_development
        from app.core.logger import setup_logging, get_logger
        print("✅ Configuración importada correctamente")
        
        setup_logging()
        logger = get_logger("main")
        app_config = get_app_config()
        
    except ImportError as e:
        print(f"⚠️ Configuración no disponible, usando valores por defecto: {e}")
        
        # Valores por defecto si no hay configuración
        class DefaultConfig:
            name = "Prototipo_chatbot"
            version = "1.0.0"
            description = "Sistema RAG para Administraciones Locales"
            host = "localhost"
            port = 5000
            debug = True
        
        app_config = DefaultConfig()
        logger = type('Logger', (), {'info': print, 'error': print, 'warning': print})()
    
    # Crear aplicación Flask
    app = Flask(__name__, 
               template_folder='app/templates',
               static_folder='app/static')
    
    app.config.update(
        SECRET_KEY=os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production'),
        DEBUG=app_config.debug
    )
    
    return app, app_config, logger

def setup_routes(app, app_config, logger):
    """Configurar rutas de la aplicación"""
    
    @app.route('/')
    def index():
        """Página principal"""
        try:
            return render_template('index.html',
                                 app_name=app_config.name,
                                 app_version=app_config.version,
                                 app_description=app_config.description)
        except Exception as e:
            # Fallback si no hay template
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{app_config.name}</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
            </head>
            <body>
                <div class="container mt-5">
                    <div class="text-center">
                        <h1 class="display-4">🤖 {app_config.name}</h1>
                        <p class="lead">{app_config.description}</p>
                        <p class="text-muted">TFM Vicente Caruncho - Sistemas Inteligentes UJI</p>
                        <div class="mt-4">
                            <a href="/chat" class="btn btn-primary btn-lg me-3">
                                <i class="fas fa-comments me-2"></i>Chat RAG
                            </a>
                            <a href="/health" class="btn btn-outline-secondary">
                                <i class="fas fa-chart-line me-2"></i>Estado del Sistema
                            </a>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
    
    @app.route('/health')
    def health_check():
        """Endpoint de verificación de salud actualizado"""
        try:
            from app.services.llm.llm_services import LLMService
            llm_service = LLMService()
            
            # Obtener estado real
            health = llm_service.health_check()
            
            # Formatear para el frontend
            response = {
                'status': health['status'],
                'timestamp': health['timestamp'],
                'services': {
                    'llm': 'available' if health['status'] in ['healthy', 'degraded'] else 'unavailable',
                    'ollama': health['services']['ollama']['status'],
                    'openai': health['services']['openai']['status']
                },
                'models': health['models'],
                'components': {
                    'embeddings': 'available',
                    'vector_store': 'available',
                    'llm': health['status']
                }
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'timestamp': time.time(),
                'error': str(e),
                'services': {
                    'llm': 'unavailable',
                    'ollama': 'unavailable', 
                    'openai': 'unavailable'
                },
                'components': {}
            }), 500
    
    @app.route('/routes')
    def list_routes():
        """Listar todas las rutas disponibles (solo para desarrollo)"""
        if not app_config.debug:
            return jsonify({'error': 'No disponible en producción'}), 403
        
        routes = []
        for rule in app.url_map.iter_rules():
            routes.append({
                'endpoint': rule.endpoint,
                'methods': list(rule.methods),
                'rule': rule.rule
            })
        
        return jsonify({
            'routes': routes,
            'total': len(routes)
        })

def setup_error_handlers(app, logger):
    """Configurar manejadores de errores"""
    
    @app.errorhandler(404)
    def not_found_error(error):
        logger.warning(f"Página no encontrada: {error}")
        
        if 'application/json' in str(request.headers.get('Accept', '')):
            return jsonify({
                'error': 'Página no encontrada',
                'status': 404,
                'available_routes': ['/chat', '/health', '/']
            }), 404
        
        try:
            return render_template('errors/404.html'), 404
        except:
            return """
            <h1>404 - Página no encontrada</h1>
            <p><a href="/">Volver al inicio</a> | <a href="/chat">Chat RAG</a></p>
            """, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Error interno: {error}")
        
        if 'application/json' in str(request.headers.get('Accept', '')):
            return jsonify({
                'error': 'Error interno del servidor',
                'status': 500
            }), 500
        
        try:
            return render_template('errors/500.html'), 500
        except:
            return """
            <h1>500 - Error interno del servidor</h1>
            <p><a href="/">Volver al inicio</a></p>
            """, 500

def setup_context_processors(app, app_config):
    """Configurar procesadores de contexto"""
    
    @app.context_processor
    def inject_global_vars():
        return {
            'app_name': app_config.name,
            'app_version': app_config.version,
            'app_description': app_config.description,
            'current_year': '2025'
        }

def verify_system_status():
    """Verificar estado inicial del sistema"""
    try:
        from app.services.rag_pipeline import get_rag_pipeline
        pipeline = get_rag_pipeline()
        if pipeline.is_available():
            print("✅ Pipeline RAG disponible y listo")
        else:
            print("⚠️ Pipeline RAG en modo de inicialización")
    except Exception as e:
        print(f"⚠️ Pipeline RAG: {e}")

def print_startup_info(app_config):
    """Mostrar información de inicio"""
    print("\n" + "="*60)
    print("✅ APLICACIÓN FLASK CONFIGURADA")
    print("="*60)
    print("🌐 URLs disponibles:")
    print("   http://localhost:5000      (Página principal)")
    print("   http://localhost:5000/chat (Chat RAG)")
    print("   http://localhost:5000/health (Estado del sistema)")
    if app_config.debug:
        print("   http://localhost:5000/routes (Lista de rutas)")
    print("\n💡 Características activas:")
    print("   🤖 Pipeline RAG integrado")
    print("   💬 Chat con modelos locales y cloud")
    print("   📊 Sistema de estado y métricas")
    print("   🎨 Interface web responsive")
    print("\n⚠️  Usa Ctrl+C para detener el servidor")
    print("="*60)

def main():
    """Función principal de arranque"""
    print("🚀 Iniciando Prototipo_chatbot TFM...")
    print("📁 Directorio del proyecto:", project_root)
    print("👨‍🎓 Vicente Caruncho Ramos - Sistemas Inteligentes UJI")
    print("-" * 60)
    
    try:
        # Crear archivos faltantes
        create_missing_files()
        
        # Verificar que Flask está disponible
        try:
            print("✅ Flask importado correctamente")
        except ImportError as e:
            print(f"❌ Error importando Flask: {e}")
            print("💡 Instala las dependencias con: pip install -r requirements.txt")
            sys.exit(1)
        
        # Crear aplicación Flask
        app, app_config, logger = create_flask_app()
        
        # Registrar blueprints
        register_blueprints(app)
        
        # Configurar rutas
        setup_routes(app, app_config, logger)
        
        # Configurar manejadores de errores
        setup_error_handlers(app, logger)
        
        # Configurar procesadores de contexto
        setup_context_processors(app, app_config)
        
        logger.info("Aplicación Flask creada exitosamente")
        
        # Mostrar información de inicio
        print_startup_info(app_config)
        
        # Verificar estado inicial del sistema
        verify_system_status()
        
        # Iniciar servidor
        app.run(
            host=app_config.host,
            port=app_config.port,
            debug=app_config.debug,
            use_reloader=True
        )
        
    except KeyboardInterrupt:
        print("\n⏹️  Aplicación detenida por el usuario")
        print("👋 ¡Hasta luego!")
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        print("💡 Soluciones posibles:")
        print("   1. pip install -r requirements.txt")
        print("   2. Verificar que estás en el directorio correcto")
        print("   3. Verificar los archivos de configuración")
        sys.exit(1)

if __name__ == "__main__":
    main()