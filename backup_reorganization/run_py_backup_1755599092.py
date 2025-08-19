#!/usr/bin/env python3
"""
Prototipo_chatbot - Punto de entrada FINAL FUNCIONAL
TFM Vicente Caruncho - Sistemas Inteligentes UJI
Versión: 2.2 (Compatible con configuración existente)
"""
import sys
import os
import time
from pathlib import Path

# Añadir el directorio raíz al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def create_flask_app():
    """Crear aplicación Flask usando la configuración existente"""
    try:
        # Usar tu sistema de configuración existente
        from app.core.config import get_app_config, get_log_config
        from app.core.logger import get_logger
        
        # Obtener configuraciones
        app_config = get_app_config()
        logger = get_logger("main")
        
        logger.info("✅ Configuración cargada exitosamente")
        
        # Intentar usar el factory existente
        try:
            from app import create_app
            app = create_app()
            logger.info("✅ Aplicación creada usando factory existente")
            return app, app_config, logger
            
        except Exception as factory_error:
            logger.warning(f"Factory no disponible: {factory_error}")
            # Crear app básica con tu configuración
            return create_basic_app_with_config(app_config, logger)
            
    except ImportError as config_error:
        print(f"⚠️ Configuración no disponible: {config_error}")
        return create_fallback_app()


def create_basic_app_with_config(app_config, logger):
    """Crear app básica usando tu configuración existente"""
    from flask import Flask, jsonify
    
    app = Flask(
        __name__,
        template_folder='app/templates',
        static_folder='app/static'
    )
    
    # Configuración usando tu AppConfig
    app.config.update({
        'SECRET_KEY': app_config.secret_key,
        'DEBUG': app_config.debug,
        'JSON_AS_ASCII': False,
        'JSONIFY_PRETTYPRINT_REGULAR': True,
    })
    
    # Registrar blueprints disponibles
    blueprints_registered = register_available_blueprints(app, logger)
    
    # Si no hay blueprints, crear rutas básicas
    if blueprints_registered == 0:
        create_basic_routes(app, app_config, logger)
    
    logger.info(f"✅ Aplicación creada con {blueprints_registered} blueprints")
    return app, app_config, logger


def register_available_blueprints(app, logger):
    """Registrar blueprints disponibles sin fallar"""
    blueprints_registered = 0
    
    # Lista de blueprints a intentar
    blueprint_attempts = [
        ('app.routes.main', 'main_bp', None),
        ('app.routes.chat', 'chat_bp', '/chat'),
        ('app.routes.admin', 'admin_bp', '/admin'),
        ('app.routes.api', 'api_bp', '/api'),
        ('app.routes.api.data_sources', 'data_sources_api', '/api/data-sources'),
        ('app.routes.api.chat', 'chat_api', '/api/chat'),
        ('app.routes.api.comparison', 'comparison_api', '/api/comparison')
    ]
    
    for module_path, blueprint_name, url_prefix in blueprint_attempts:
        try:
            module = __import__(module_path, fromlist=[blueprint_name])
            blueprint = getattr(module, blueprint_name)
            
            if url_prefix:
                app.register_blueprint(blueprint, url_prefix=url_prefix)
                logger.info(f"✅ Blueprint {blueprint_name} registrado en {url_prefix}")
            else:
                app.register_blueprint(blueprint)
                logger.info(f"✅ Blueprint {blueprint_name} registrado")
            
            blueprints_registered += 1
            
        except (ImportError, AttributeError) as e:
            logger.debug(f"Blueprint {blueprint_name} no disponible: {e}")
    
    return blueprints_registered


def create_basic_routes(app, app_config, logger):
    """Crear rutas básicas como fallback"""
    from flask import jsonify
    
    @app.route('/')
    def index():
        return jsonify({
            "message": f"🎓 {app_config.name} - Sistema Funcionando",
            "description": app_config.description,
            "status": "running",
            "author": "Vicente Caruncho Ramos",
            "university": "Universitat Jaume I",
            "master": "Sistemas Inteligentes",
            "version": app_config.version,
            "mode": "basic_routes",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "endpoints": {
                "health": "/health",
                "api_status": "/api/status",
                "system_info": "/info"
            }
        })
    
    @app.route('/health')
    def health_check():
        return jsonify({
            "status": "healthy",
            "timestamp": time.time(),
            "version": app_config.version,
            "environment": "development" if app_config.debug else "production",
            "host": app_config.host,
            "port": app_config.port
        })
    
    @app.route('/api/status')
    def api_status():
        return jsonify({
            "api_status": "available",
            "application": app_config.name,
            "version": app_config.version,
            "configuration": "loaded",
            "message": "Sistema funcionando con configuración completa"
        })
    
    @app.route('/info')
    def system_info():
        return jsonify({
            "system": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform,
                "working_directory": str(project_root)
            },
            "application": {
                "name": app_config.name,
                "version": app_config.version,
                "description": app_config.description,
                "debug_mode": app_config.debug
            },
            "tfm": {
                "author": "Vicente Caruncho Ramos",
                "university": "Universitat Jaume I",
                "master": "Sistemas Inteligentes",
                "year": "2024-2025"
            }
        })
    
    logger.info("✅ Rutas básicas creadas como fallback")


def create_fallback_app():
    """Crear app de emergencia si todo falla"""
    from flask import Flask, jsonify
    
    class FallbackConfig:
        name = "Prototipo_chatbot"
        version = "2.2"
        description = "Sistema en modo emergencia"
        host = "localhost"
        port = 5000
        debug = True
        secret_key = "fallback-secret-key"
    
    class FallbackLogger:
        def info(self, msg): print(f"ℹ️ {msg}")
        def error(self, msg): print(f"❌ {msg}")
        def warning(self, msg): print(f"⚠️ {msg}")
        def debug(self, msg): print(f"🔍 {msg}")
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'fallback-secret'
    
    config = FallbackConfig()
    logger = FallbackLogger()
    
    @app.route('/')
    def emergency_index():
        return jsonify({
            "message": "🚨 Prototipo_chatbot - Modo de Emergencia",
            "status": "emergency_mode",
            "note": "Sistema funcionando con configuración mínima",
            "author": "Vicente Caruncho Ramos"
        })
    
    @app.route('/health')
    def emergency_health():
        return jsonify({"status": "emergency", "mode": "fallback"})
    
    logger.info("⚠️ Aplicación de emergencia creada")
    return app, config, logger


def print_startup_info(app_config, blueprints_count=0):
    """Mostrar información de inicio usando tu configuración"""
    print("\n" + "=" * 70)
    print("🎓 TFM Vicente Caruncho - Prototipo Chatbot RAG")
    print("🏛️ Administraciones Locales - Universitat Jaume I")
    print("=" * 70)
    print(f"🚀 Aplicación: {app_config.name}")
    print(f"📝 Descripción: {app_config.description}")
    print(f"🐍 Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"🔧 Blueprints: {blueprints_count} registrados")
    print(f"📦 Versión: {app_config.version}")
    print(f"🏷️ Debug: {'Activado' if app_config.debug else 'Desactivado'}")
    print(f"\n🌐 Servidor: http://{app_config.host}:{app_config.port}")
    print(f"🩺 Health Check: http://{app_config.host}:{app_config.port}/health")
    print(f"📊 API Status: http://{app_config.host}:{app_config.port}/api/status")
    print(f"ℹ️ System Info: http://{app_config.host}:{app_config.port}/info")
    print(f"\n⚠️ Usa Ctrl+C para detener el servidor")
    print("=" * 70)


def main():
    """Función principal final"""
    print("🚀 Iniciando Prototipo_chatbot TFM v2.2...")
    print(f"📁 Directorio: {project_root}")
    print("🎯 Objetivo: Usar configuración existente correctamente")
    
    try:
        # Crear directorios básicos
        (project_root / "logs").mkdir(exist_ok=True)
        
        # Crear aplicación Flask
        app, app_config, logger = create_flask_app()
        
        # Contar blueprints registrados (si es posible)
        blueprints_count = len(app.blueprints) if hasattr(app, 'blueprints') else 0
        
        # Mostrar información de inicio
        print_startup_info(app_config, blueprints_count)
        
        # Marcar tiempo de inicio
        app.start_time = time.time()
        
        # Iniciar servidor usando tu configuración
        logger.info(f"🌐 Servidor iniciando en http://{app_config.host}:{app_config.port}")
        
        app.run(
            host=app_config.host,
            port=app_config.port,
            debug=app_config.debug,
            use_reloader=True,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n⏹️ Aplicación detenida por el usuario")
        print("👋 ¡Hasta luego! Gracias por usar Prototipo_chatbot TFM")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Error crítico iniciando la aplicación:")
        print(f"🔍 Error: {e}")
        print("\n💡 Soluciones:")
        print("   1. Verificar dependencias: pip install -r requirements.txt")
        print("   2. Verificar estructura de directorios")
        print("   3. Revisar permisos de escritura en logs/")
        print("   4. Ejecutar diagnóstico: python scripts/system_diagnosis.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
