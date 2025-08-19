"""
Prototipo_chatbot - TFM Vicente Caruncho Ramos
Universitat Jaume I - Sistemas Inteligentes

Sistema RAG para Administraciones Locales
"""

__version__ = "1.0.0"
__author__ = "Vicente Caruncho Ramos"
__university__ = "Universitat Jaume I"
__project__ = "Prototipo de Chatbot RAG para Administraciones Locales"

# Importaciones principales (con manejo de errores)
try:
    from .models.document import DocumentChunk, DocumentMetadata
except ImportError:
    DocumentChunk = None
    DocumentMetadata = None

try:
    from .services.rag.embeddings import embedding_service
except ImportError:
    embedding_service = None

def get_project_info():
    """Informacion del proyecto"""
    return {
        "name": __project__,
        "version": __version__,
        "author": __author__,
        "university": __university__,
        "status": "TFM Development"
    }


# =============================================================================
# APPLICATION FACTORY
# =============================================================================

def create_app(config=None):
    """
    Factory para crear la aplicación Flask
    Añadido por fix rápido para compatibilidad con run.py simplificado
    
    Args:
        config: Configuración opcional a aplicar
    
    Returns:
        Flask: Instancia de la aplicación configurada
    """
    import os
    import sys
    from pathlib import Path
    from flask import Flask
    from flask import jsonify
    
    # Crear instancia Flask
    app = Flask(
        __name__,
        template_folder='templates',
        static_folder='static',
        instance_relative_config=True
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
    except ImportError:
        import logging
        logging.basicConfig(level=logging.INFO)
        app.logger.info("Usando logging básico (configuración core no disponible)")
    
    # Registrar blueprints disponibles
    blueprints_registered = 0
    
    # Blueprint principal SIEMPRE con /health endpoint
    try:
        from app.routes.main import main_bp
        app.register_blueprint(main_bp)
        app.logger.info("✅ Blueprint main registrado")
        blueprints_registered += 1
    except ImportError as e:
        app.logger.warning(f"⚠️ Blueprint main no disponible: {e}")
        # Crear blueprint básico como fallback
        from flask import Blueprint, jsonify
        main_bp = Blueprint('main', __name__)
        
        @main_bp.route('/')
        def index():
            return jsonify({
                "message": "Prototipo_chatbot TFM - Sistema funcionando",
                "status": "running",
                "author": "Vicente Caruncho Ramos",
                "version": "2.0",
                "mode": "basic_fallback"
            })
        
        app.register_blueprint(main_bp)
        app.logger.info("✅ Blueprint main básico creado como fallback")
        blueprints_registered += 1
    
    # ✅ ENDPOINT /health SIEMPRE DISPONIBLE (corrección crítica)
    @app.route('/health')
    def health_check():
        """Health check endpoint - siempre disponible"""
        from datetime import datetime
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": __version__,
            "author": __author__,
            "mode": "production" if not app.config.get('DEBUG') else "development",
            "blueprints_registered": blueprints_registered
        })
    
    # ✅ ENDPOINT AJAX STATS (para evitar errores 404)
    @app.route('/ajax/quick-stats')
    def quick_stats():
        """Quick stats endpoint"""
        return jsonify({
            "status": "ok",
            "timestamp": __import__('time').time(),
            "active": True,
            "documents_indexed": 0,  # Por defecto
            "system_status": "operational"
        })
    
    # Blueprints opcionales CORREGIDOS
    optional_blueprints = [
        ('app.routes.api', 'api_bp', '/api'),
        ('app.routes.api.data_sources', 'data_sources_api', '/api/data-sources'),
        ('app.routes.api.comparison', 'comparison_api', '/api/comparison'),
        ('app.routes.chat_routes', 'chat_bp', None),  # ✅ Corregido nombre del archivo
        ('app.routes.admin', 'admin_bp', '/admin')
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
        from flask import jsonify
        return jsonify({
            'error': 'Página no encontrada',
            'status_code': 404,
            'message': 'El recurso solicitado no existe'
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"Error interno del servidor: {error}")
        from flask import jsonify
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
            'app_version': '2.0',
            'author': 'Vicente Caruncho Ramos',
            'university': 'Universitat Jaume I',
            'master': 'Sistemas Inteligentes',
            'current_year': 2025
        }
    
    app.logger.info(f"📋 Aplicación Flask creada con {blueprints_registered} blueprints")
    return app


# Para compatibilidad
def get_app():
    """Obtener instancia de la aplicación (para compatibilidad)"""
    return create_app()