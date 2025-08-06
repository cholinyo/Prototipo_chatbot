"""
Prototipo_chatbot - Aplicación Flask principal mejorada
TFM Vicente Caruncho - Sistemas Inteligentes
"""
from flask import Flask, render_template, request, g, jsonify
import time
import os
from pathlib import Path
from typing import Optional

from app.core.config import get_app_config, get_openai_api_key, is_development
from app.core.logger import get_logger


def create_app(config_override: Optional[dict] = None) -> Flask:
    """Factory de aplicación Flask mejorado
    
    Args:
        config_override: Configuración adicional para testing
    
    Returns:
        Aplicación Flask configurada
    """
    
    # Obtener configuración
    app_config = get_app_config()
    logger = get_logger("flask_app")
    
    # Crear aplicación Flask
    app = Flask(__name__)
    
    # Configurar Flask
    app.config.update({
        'SECRET_KEY': os.getenv('SECRET_KEY', f"{app_config.name}_dev_secret_key"),
        'DEBUG': app_config.debug,
        'TESTING': False,
        'JSON_AS_ASCII': False,
        'JSONIFY_PRETTYPRINT_REGULAR': is_development(),
        'MAX_CONTENT_LENGTH': 50 * 1024 * 1024,  # 50MB max file upload
    })
    
    # Aplicar configuración adicional si se proporciona
    if config_override:
        app.config.update(config_override)
    
    logger.info("Aplicación Flask inicializada",
               name=app_config.name,
               version=app_config.version,
               debug=app_config.debug)
    
    # Registrar hooks de request
    _register_request_hooks(app, logger)
    
    # Registrar blueprints
    _register_blueprints(app, logger)
    
    # Registrar handlers de error
    _register_error_handlers(app, logger)
    
    # Registrar rutas básicas
    _register_basic_routes(app, app_config, logger)
    
    # Registrar context processors
    _register_context_processors(app, app_config)

    from app.routes.api import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    logger.info("Aplicación Flask configurada completamente")
    from app.routes.rag_pipeline_api import rag_pipeline_bp  
    app.register_blueprint(rag_pipeline_bp)
    
    return app

def _register_request_hooks(app: Flask, logger) -> None:
    """Registrar hooks de request para logging y métricas"""
    
    @app.before_request
    def before_request():
        """Hook antes de cada request"""
        g.start_time = time.time()
        g.logger = get_logger("request")
        
        # Log básico de request entrante
        g.logger.debug("Request iniciado",
                      method=request.method,
                      path=request.path,
                      remote_addr=request.remote_addr,
                      user_agent=request.headers.get('User-Agent', ''))
    
    @app.after_request  
    def after_request(response):
        """Hook después de cada request"""
        if hasattr(g, 'start_time'):
            response_time = time.time() - g.start_time
            
            # Log de la request completada
            if hasattr(g, 'logger'):
                log_level = 'warning' if response.status_code >= 400 else 'info'
                g.logger.log(log_level, "Request completado",
                           method=request.method,
                           endpoint=request.endpoint or 'unknown',
                           path=request.path,
                           status_code=response.status_code,
                           response_time_seconds=round(response_time, 3),
                           content_length=response.content_length,
                           remote_addr=request.remote_addr)
        
        # Headers de seguridad
        response.headers.update({
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Referrer-Policy': 'strict-origin-when-cross-origin'
        })
        
        # Header personalizado para identificar la aplicación
        response.headers['X-Powered-By'] = 'Prototipo_chatbot'
        
        return response
    
    @app.teardown_appcontext
    def teardown_appcontext(error):
        """Limpieza al final del contexto de aplicación"""
        if error:
            logger.error("Error en contexto de aplicación", error=str(error))

def _register_blueprints(app: Flask, logger) -> None:
    """Registrar blueprints de la aplicación"""
    
    blueprints_to_register = [
        # Main routes
        ('app.routes.main', 'main_bp', '/'),
        # Chat functionality  
        ('app.routes.chat', 'chat_bp', '/chat'),
        # API endpoints
        ('app.routes.api', 'api_bp', '/api'),
        # Admin panel
        ('app.routes.admin', 'admin_bp', '/admin'),
        # Configuration
        ('app.routes.config', 'config_bp', '/config'),
    ]
    
    registered_count = 0
    
    for module_path, blueprint_name, url_prefix in blueprints_to_register:
        try:
            module = __import__(module_path, fromlist=[blueprint_name])
            blueprint = getattr(module, blueprint_name)
            app.register_blueprint(blueprint, url_prefix=url_prefix)
            logger.info("Blueprint registrado", 
                       module=module_path,
                       blueprint=blueprint_name,
                       url_prefix=url_prefix)
            registered_count += 1
        except ImportError as e:
            logger.warning("Blueprint no disponible (pendiente de implementar)",
                          module=module_path,
                          blueprint=blueprint_name,
                          error=str(e))
        except AttributeError as e:
            logger.error("Blueprint mal configurado",
                        module=module_path, 
                        blueprint=blueprint_name,
                        error=str(e))
    
    logger.info("Blueprints procesados", 
               registered=registered_count,
               total=len(blueprints_to_register))

def _register_error_handlers(app: Flask, logger) -> None:
    """Registrar manejadores de errores"""
    
    @app.errorhandler(404)
    def not_found_error(error):
        logger.warning("Página no encontrada",
                      path=request.path,
                      method=request.method,
                      remote_addr=request.remote_addr,
                      referrer=request.referrer)
        
        if request.path.startswith('/api/'):
            return jsonify({
                'error': 'Not Found',
                'message': 'Endpoint no encontrado',
                'status_code': 404
            }), 404
        
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error("Error interno del servidor",
                    path=request.path,
                    method=request.method,
                    error=str(error),
                    remote_addr=request.remote_addr)
        
        if request.path.startswith('/api/'):
            return jsonify({
                'error': 'Internal Server Error',
                'message': 'Error interno del servidor',
                'status_code': 500
            }), 500
        
        return render_template('errors/500.html'), 500
    
    @app.errorhandler(403)
    def forbidden_error(error):
        logger.warning("Acceso prohibido",
                      path=request.path,
                      method=request.method,
                      remote_addr=request.remote_addr)
        
        if request.path.startswith('/api/'):
            return jsonify({
                'error': 'Forbidden',
                'message': 'Acceso prohibido',
                'status_code': 403
            }), 403
        
        return render_template('errors/403.html'), 403
    
    @app.errorhandler(413)
    def request_entity_too_large_error(error):
        logger.warning("Archivo demasiado grande",
                      path=request.path,
                      content_length=request.content_length,
                      remote_addr=request.remote_addr)
        
        if request.path.startswith('/api/'):
            return jsonify({
                'error': 'Request Entity Too Large',
                'message': 'Archivo demasiado grande (máximo 50MB)',
                'status_code': 413
            }), 413
        
        return render_template('errors/413.html'), 413
    
    logger.info("Manejadores de error registrados")

def _register_basic_routes(app: Flask, app_config, logger) -> None:
    """Registrar rutas básicas del sistema"""
    
    @app.route('/health')
    def health_check():
        """Endpoint de verificación de salud del sistema"""
        return jsonify({
            'status': 'healthy',
            'version': app_config.version,
            'name': app_config.name,
            'timestamp': time.time(),
            'environment': 'development' if is_development() else 'production'
        })
    
    @app.route('/')
    def index():
        """Página principal"""
        try:
            return render_template('index.html')
        except Exception as e:
            logger.error("Error renderizando página principal", error=str(e))
            return render_template('errors/500.html'), 500
    
    @app.route('/status')
    def system_status():
        """Estado detallado del sistema"""
        try:
            # Verificar componentes del sistema
            status_data = {
                'server': 'healthy',
                'vector_store': _check_vector_store_status(),
                'local_models': _check_local_models_status(),
                'openai': _check_openai_status(),
                'timestamp': time.time()
            }
            
            overall_status = 'healthy'
            if any(status == 'error' for status in status_data.values() if isinstance(status, str)):
                overall_status = 'error'
            elif any(status == 'warning' for status in status_data.values() if isinstance(status, str)):
                overall_status = 'warning'
            
            status_data['overall'] = overall_status
            
            return jsonify(status_data)
            
        except Exception as e:
            logger.error("Error verificando estado del sistema", error=str(e))
            return jsonify({
                'overall': 'error',
                'error': str(e),
                'timestamp': time.time()
            }), 500

def _check_vector_store_status() -> str:
    """Verificar estado del vector store"""
    try:
        import faiss
        # Aquí se podría verificar si hay índices cargados
        return 'healthy'
    except ImportError:
        return 'error'
    except Exception:
        return 'warning'

def _check_local_models_status() -> str:
    """Verificar estado de modelos locales"""
    try:
        # Verificar si Ollama está disponible
        import subprocess
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, 
                              timeout=5)
        return 'healthy' if result.returncode == 0 else 'warning'
    except:
        return 'warning'

def _check_openai_status() -> str:
    """Verificar estado de OpenAI API"""
    api_key = get_openai_api_key()
    if not api_key:
        return 'warning'
    
    try:
        import openai
        # Aquí se podría hacer una verificación real de la API
        return 'healthy'
    except ImportError:
        return 'error'
    except Exception:
        return 'warning'

def _register_context_processors(app: Flask, app_config) -> None:
    """Registrar procesadores de contexto para templates"""
    
    @app.context_processor
    def inject_app_context():
        """Inyectar variables globales en todos los templates"""
        return {
            'app_name': app_config.name,
            'app_version': app_config.version,
            'app_description': app_config.description,
            'is_development': is_development(),
            'current_year': time.strftime('%Y')
        }
    
    @app.template_filter('datetime')
    def datetime_filter(timestamp):
        """Filtro para formatear timestamps"""
        if isinstance(timestamp, (int, float)):
            return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        return timestamp
    
    @app.template_filter('filesize')
    def filesize_filter(bytes_size):
        """Filtro para formatear tamaños de archivo"""
        if not isinstance(bytes_size, (int, float)):
            return bytes_size
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} TB"