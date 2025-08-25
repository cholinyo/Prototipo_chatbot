"""
Aplicación Flask - Factory Pattern
TFM Vicente Caruncho - Sistemas Inteligentes
"""

import os
from pathlib import Path
from flask import Flask, jsonify
from datetime import datetime


# Metadatos de la aplicación
__version__ = "3.0"
__author__ = "Vicente Caruncho Ramos"


def create_app(config=None):
    """Factory para crear aplicación Flask"""
    
    # Obtener directorio raíz del proyecto
    project_root = Path(__file__).parent.parent
    
    # Crear aplicación Flask con rutas corregidas
    app = Flask(
        __name__,
        template_folder=str(project_root / 'app' / 'templates'),  # ✅ CORREGIDO
        static_folder=str(project_root / 'app' / 'static')        # ✅ CORREGIDO
    )
    
    # Configuración básica
    app.config.update({
        'SECRET_KEY': os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production'),
        'DEBUG': os.getenv('FLASK_DEBUG', 'True').lower() == 'true',
        'TESTING': False,
        'JSON_AS_ASCII': False,
        'JSON_SORT_KEYS': False,
        'JSONIFY_PRETTYPRINT_REGULAR': True,
        'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB max file upload
    })
    
    # Aplicar configuración personalizada si se proporciona
    if config:
        app.config.update(config)
    
    # Configurar logging básico
    try:
        from app.core.logger import get_logger
        logger = get_logger("app_factory")
        logger.info("Logging configurado correctamente")
        app.logger = logger
    except ImportError as e:
        import logging
        logging.basicConfig(level=logging.INFO)
        app.logger.info(f"Usando logging básico (configuración core no disponible): {e}")
    
    # Contador de blueprints registrados
    blueprints_registered = 0
    
    # Blueprint principal SIEMPRE disponible
    try:
        from app.routes.main import main_bp
        app.register_blueprint(main_bp)
        app.logger.info("✅ Blueprint main registrado")
        blueprints_registered += 1
    except ImportError as e:
        app.logger.warning(f"⚠️ Blueprint main no disponible: {e}")
        # Crear blueprint básico como fallback
        from flask import Blueprint
        main_bp = Blueprint('main', __name__)
        
        @main_bp.route('/')
        def index():
            return jsonify({
                "message": "Prototipo_chatbot TFM - Sistema funcionando",
                "status": "running",
                "author": __author__,
                "version": __version__,
                "mode": "basic_fallback",
                "available_endpoints": ["/", "/health", "/ajax/quick-stats"]
            })
        
        app.register_blueprint(main_bp)
        app.logger.info("✅ Blueprint main básico creado como fallback")
        blueprints_registered += 1
    
    # ✅ ENDPOINTS BÁSICOS SIEMPRE DISPONIBLES
    @app.route('/health')
    def health_check():
        """Health check endpoint - siempre disponible"""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": __version__,
            "author": __author__,
            "mode": "production" if not app.config.get('DEBUG') else "development",
            "blueprints_registered": blueprints_registered,
            "blueprints_available": list(app.blueprints.keys())
        })
    
    @app.route('/ajax/quick-stats')
    def quick_stats():
        """Quick stats endpoint"""
        return jsonify({
            "status": "ok",
            "timestamp": __import__('time').time(),
            "active": True,
            "documents_indexed": 0,  # Por defecto
            "system_status": "operational",
            "blueprints_count": len(app.blueprints)
        })
    
    # Blueprints opcionales - ORDEN CORREGIDO Y VERIFICADO
    optional_blueprints = [
        # APIs principales
        ('app.routes.web_sources_api', 'web_sources_api', '/api/web-sources'),  
        ('app.routes.api', 'api_bp', '/api'),
        
        # Rutas de gestión
        ('app.routes.chat_routes', 'chat_bp', None), 
        ('app.routes.admin', 'admin_bp', '/admin'),
        
        # APIs específicas (si existen)
        ('app.routes.data_sources', 'data_sources_api', '/api/data-sources'),
        ('app.routes.api.comparison', 'comparison_api', '/api/comparison'),
    ]
    
    for module_path, blueprint_name, url_prefix in optional_blueprints:
        try:
            module = __import__(module_path, fromlist=[blueprint_name])
            blueprint = getattr(module, blueprint_name)
            
            if url_prefix:
                app.register_blueprint(blueprint, url_prefix=url_prefix)
                app.logger.info(f"✅ Blueprint {blueprint_name} registrado en {url_prefix}")
            else:
                app.register_blueprint(blueprint)
                app.logger.info(f"✅ Blueprint {blueprint_name} registrado")
            
            blueprints_registered += 1
            
        except (ImportError, AttributeError) as e:
            app.logger.warning(f"⚠️ Blueprint {blueprint_name} no disponible: {e}")
    
    # Configurar manejadores de errores básicos
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({
            'error': 'Página no encontrada',
            'status_code': 404,
            'message': 'El recurso solicitado no existe',
            'available_blueprints': list(app.blueprints.keys())
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"Error interno del servidor: {error}")
        return jsonify({
            'error': 'Error interno del servidor',
            'status_code': 500,
            'message': 'Ha ocurrido un error inesperado'
        }), 500
    
    # Procesador de contexto para templates
    @app.context_processor
    def inject_global_vars():
        return {
            'app_name': 'Prototipo_chatbot',
            'app_version': __version__,
            'author': __author__,
            'university': 'Universitat Jaume I',
            'master': 'Sistemas Inteligentes',
            'current_year': 2025
        }
    
    app.logger.info(f"📋 Aplicación Flask creada exitosamente con {blueprints_registered} blueprints")
    app.logger.info(f"📋 Blueprints disponibles: {list(app.blueprints.keys())}")
    
    return app


# Para compatibilidad
def get_app():
    """Obtener instancia de la aplicación (para compatibilidad)"""
    return create_app()