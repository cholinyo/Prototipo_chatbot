"""
Prototipo_chatbot - Aplicación Flask principal
"""
from flask import Flask, render_template, request, g
import time
from app.core.config import get_app_config
from app.core.logger import get_logger

def create_app() -> Flask:
    """Factory de aplicación Flask"""
    
    # Obtener configuración
    app_config = get_app_config()
    logger = get_logger("flask_app")
    
    # Crear aplicación Flask
    app = Flask(__name__)
    
    # Configurar Flask
    app.config.update(
        SECRET_KEY=app_config.name + "_secret_key_change_in_production",
        DEBUG=app_config.debug,
        TESTING=False
    )
    
    logger.info("Aplicación Flask inicializada",
               name=app_config.name,
               version=app_config.version)
    
    # Registrar hooks de request
    @app.before_request
    def before_request():
        """Hook antes de cada request"""
        g.start_time = time.time()
        g.logger = get_logger("request")
    
    @app.after_request  
    def after_request(response):
        """Hook después de cada request"""
        if hasattr(g, 'start_time'):
            response_time = time.time() - g.start_time
            
            # Log de la request
            if hasattr(g, 'logger'):
                g.logger.info("Request procesado",
                            method=request.method,
                            endpoint=request.endpoint,
                            path=request.path,
                            status_code=response.status_code,
                            response_time_seconds=response_time,
                            user_agent=request.headers.get('User-Agent', ''),
                            remote_addr=request.remote_addr)
        
        # Headers de seguridad básicos
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        return response
    
    # Registrar blueprints
    _register_blueprints(app, logger)
    
    # Registrar handlers de error
    _register_error_handlers(app, logger)
    
    # Ruta de salud básica
    @app.route('/health')
    def health_check():
        """Endpoint de verificación de salud"""
        return {
            'status': 'healthy',
            'version': app_config.version,
            'name': app_config.name
        }
    
    # Ruta principal temporal
    @app.route('/')
    def index():
        """Página principal temporal"""
        return render_template('index.html',
                             app_name=app_config.name,
                             app_version=app_config.version,
                             app_description=app_config.description)
    
    logger.info("Aplicación Flask configurada completamente")
    
    return app

def _register_blueprints(app: Flask, logger) -> None:
    """Registrar blueprints de la aplicación"""
    
    blueprints_to_register = [
        # ('app.routes.main', 'main_bp'),
        # ('app.routes.chat', 'chat_bp'), 
        # ('app.routes.admin', 'admin_bp'),
        # ('app.routes.config', 'config_bp'),
        # ('app.routes.api', 'api_bp'),
    ]
    
    registered_count = 0
    
    for module_path, blueprint_name in blueprints_to_register:
        try:
            module = __import__(module_path, fromlist=[blueprint_name])
            blueprint = getattr(module, blueprint_name)
            app.register_blueprint(blueprint)
            logger.info("Blueprint registrado", 
                       module=module_path,
                       blueprint=blueprint_name)
            registered_count += 1
        except ImportError as e:
            logger.warning("Blueprint no disponible",
                          module=module_path,
                          blueprint=blueprint_name,
                          error=str(e))
        except AttributeError as e:
            logger.error("Blueprint mal configurado",
                        module=module_path, 
                        blueprint=blueprint_name,
                        error=str(e))
    
    logger.info("Blueprints registrados", count=registered_count)

def _register_error_handlers(app: Flask, logger) -> None:
    """Registrar manejadores de errores"""
    
    @app.errorhandler(404)
    def not_found_error(error):
        logger.warning("Página no encontrada",
                      path=request.path,
                      method=request.method,
                      remote_addr=request.remote_addr)
        
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error("Error interno del servidor",
                    path=request.path,
                    method=request.method,
                    error=str(error),
                    remote_addr=request.remote_addr)
        
        return render_template('errors/500.html'), 500
    
    @app.errorhandler(403)
    def forbidden_error(error):
        logger.warning("Acceso prohibido",
                      path=request.path,
                      method=request.method,
                      remote_addr=request.remote_addr)
        
        return render_template('errors/403.html'), 403
    
    logger.info("Manejadores de error registrados")