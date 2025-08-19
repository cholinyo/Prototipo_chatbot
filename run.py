#!/usr/bin/env python3
"""
Prototipo_chatbot - Usando infrastructure existente REAL
TFM Vicente Caruncho - Sistemas Inteligentes UJI
Versión: 3.0 (Completamente funcional)
"""
import sys
import os
import time
from pathlib import Path

# Añadir el directorio raíz al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def create_flask_app():
    """Crear aplicación Flask usando tu factory y configuración existente"""
    try:
        # Usar TU sistema de configuración real
        from app.core.config import get_app_config
        from app.core.logger import get_logger
        
        app_config = get_app_config()
        logger = get_logger("main")
        
        logger.info("✅ Configuración real cargada exitosamente")
        
        # Usar TU factory existente
        from app import create_app
        app = create_app()
        
        logger.info("✅ App factory existente utilizada")
        return app, app_config, logger
        
    except ImportError as e:
        print(f"❌ Error con factory existente: {e}")
        print("🔄 Creando app básica compatible...")
        return create_compatible_app()


def create_compatible_app():
    """App compatible que usa tus blueprints existentes"""
    from flask import Flask
    
    # Configuración básica
    app = Flask(
        __name__,
        template_folder='app/templates',
        static_folder='app/static'
    )
    
    app.config.update({
        'SECRET_KEY': os.getenv('SECRET_KEY', 'dev-secret-key'),
        'DEBUG': os.getenv('FLASK_DEBUG', 'True').lower() == 'true',
        'TEMPLATES_AUTO_RELOAD': True
    })
    
    # Usar TU configuración real si está disponible
    try:
        from app.core.config import get_app_config
        from app.core.logger import get_logger
        
        app_config = get_app_config()
        logger = get_logger("main")
        
        # Aplicar configuración real
        app.config.update({
            'SECRET_KEY': app_config.secret_key,
            'DEBUG': app_config.debug,
        })
        
        logger.info("✅ Configuración real aplicada a app básica")
        
    except ImportError:
        # Configuración por defecto
        class DefaultConfig:
            name = "Prototipo_chatbot"
            version = "3.0"
            host = "localhost"
            port = 5000
            debug = True
            secret_key = "dev-secret"
        
        class DefaultLogger:
            def info(self, msg): print(f"ℹ️ {msg}")
            def error(self, msg): print(f"❌ {msg}")
            def warning(self, msg): print(f"⚠️ {msg}")
        
        app_config = DefaultConfig()
        logger = DefaultLogger()
        
        logger.info("⚠️ Usando configuración por defecto")
    
    # Registrar TUS blueprints existentes
    blueprints_registered = register_real_blueprints(app, logger)
    
    logger.info(f"✅ App compatible creada con {blueprints_registered} blueprints")
    return app, app_config, logger


def register_real_blueprints(app, logger):
    """Registrar TUS blueprints existentes - NO valores estáticos"""
    blueprints_registered = 0
    
    # TU blueprint principal que ya tiene la lógica real
    try:
        from app.routes.main import main_bp
        app.register_blueprint(main_bp)
        logger.info("✅ Blueprint main_bp registrado (con lógica real)")
        blueprints_registered += 1
    except ImportError as e:
        logger.error(f"❌ No se pudo importar main_bp: {e}")
        create_fallback_main_route(app, logger)
        blueprints_registered += 1
    
    # TUS otros blueprints existentes
    optional_blueprints = [
        ('app.routes.chat', 'chat_bp', '/chat'),
        ('app.routes.admin', 'admin_bp', '/admin'),
        ('app.routes.api', 'api_bp', '/api'),
        ('app.routes.data_sources', 'data_sources_api', None),
        ('app.routes.llm_api', 'llm_api_bp', '/api/llm'),
        ('app.routes.rag_pipeline_api', 'rag_api_bp', '/api/rag')
    ]
    
    for module_path, blueprint_name, url_prefix in optional_blueprints:
        try:
            module = __import__(module_path, fromlist=[blueprint_name])
            blueprint = getattr(module, blueprint_name)
            app.register_blueprint(blueprint, url_prefix=url_prefix)
            logger.info(f"✅ Blueprint {blueprint_name} registrado en {url_prefix}")
            blueprints_registered += 1
        except (ImportError, AttributeError) as e:
            logger.debug(f"Blueprint {blueprint_name} no disponible: {e}")
    
    return blueprints_registered


def create_fallback_main_route(app, logger):
    """Crear ruta principal de fallback si main_bp no está disponible"""
    from flask import jsonify
    
    @app.route('/')
    def fallback_index():
        logger.warning("Usando ruta principal de fallback")
        return jsonify({
            "message": "🎓 Prototipo_chatbot TFM - Blueprint Fallback",
            "status": "fallback_mode",
            "author": "Vicente Caruncho Ramos",
            "university": "Universitat Jaume I",
            "note": "app.routes.main no disponible - verifica la estructura",
            "suggestion": "Implementar app.routes.main.main_bp para funcionalidad completa"
        })
    
    @app.route('/health')
    def fallback_health():
        return jsonify({
            "status": "healthy",
            "mode": "fallback",
            "timestamp": time.time()
        })
    
    @app.route('/api/status')
    def fallback_api_status():
        return jsonify({
            "api_status": "available",
            "mode": "fallback",
            "blueprints": "limited"
        })
    
    logger.info("✅ Rutas de fallback creadas")


def setup_error_handlers(app, logger):
    """Configurar manejadores de errores usando TUS templates"""
    
    @app.errorhandler(404)
    def not_found_error(error):
        try:
            from flask import render_template
            return render_template('errors/404.html'), 404
        except:
            from flask import jsonify
            return jsonify({
                'error': 'Página no encontrada',
                'status_code': 404
            }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Error interno: {error}")
        try:
            from flask import render_template
            return render_template('errors/500.html'), 500
        except:
            from flask import jsonify
            return jsonify({
                'error': 'Error interno del servidor',
                'status_code': 500
            }), 500


def print_startup_info(app_config, blueprints_count):
    """Mostrar información de inicio"""
    print("\n" + "=" * 70)
    print("🎓 TFM Vicente Caruncho - Prototipo Chatbot RAG")
    print("🏛️ VERSIÓN FUNCIONAL - Universitat Jaume I")
    print("=" * 70)
    print(f"🚀 Aplicación: {getattr(app_config, 'name', 'Prototipo_chatbot')}")
    print(f"📝 Versión: {getattr(app_config, 'version', '3.0')}")
    print(f"🔧 Blueprints: {blueprints_count} registrados")
    print(f"🏷️ Debug: {'Activado' if getattr(app_config, 'debug', True) else 'Desactivado'}")
    print(f"\n🌐 Servidor: http://{getattr(app_config, 'host', 'localhost')}:{getattr(app_config, 'port', 5000)}")
    print(f"🎯 Modo: Funcional con infrastructure existente")
    print(f"📊 Templates: app/templates/ (tus templates reales)")
    print(f"🔗 Static: app/static/ (tus assets reales)")
    print("=" * 70)
    
    # Mostrar blueprints disponibles
    print(f"\n📋 BLUEPRINTS ACTIVOS:")
    print(f"   ✅ main_bp: Página principal con datos reales")
    print(f"   📊 Dashboard con métricas del sistema real")
    print(f"   🔧 Configuración YAML completa")
    print(f"   📝 Logging estructurado")
    
    if blueprints_count > 1:
        print(f"   ✅ +{blueprints_count-1} blueprints adicionales disponibles")
    
    print("=" * 70)


def main():
    """Función principal - Usar TU infrastructure"""
    print("🚀 Iniciando Prototipo_chatbot TFM v3.0 (Completamente Funcional)...")
    print(f"📁 Directorio: {project_root}")
    print("🎯 Objetivo: Usar toda tu infrastructure existente")
    
    try:
        # Crear directorios básicos
        (project_root / "logs").mkdir(exist_ok=True)
        
        # Crear aplicación usando TU código
        app, app_config, logger = create_flask_app()
        
        # Configurar manejadores de errores
        setup_error_handlers(app, logger)
        
        # Contar blueprints registrados
        blueprints_count = len(app.blueprints)
        
        # Mostrar información de inicio
        print_startup_info(app_config, blueprints_count)
        
        # Marcar tiempo de inicio
        app.start_time = time.time()
        
        # Log del estado del sistema
        logger.info("Sistema iniciando",
                   blueprints=list(app.blueprints.keys()),
                   config_loaded=hasattr(app_config, 'name'),
                   debug_mode=getattr(app_config, 'debug', True))
        
        # Iniciar servidor
        logger.info(f"🌐 Servidor iniciando en http://{getattr(app_config, 'host', 'localhost')}:{getattr(app_config, 'port', 5000)}")
        
        app.run(
            host=getattr(app_config, 'host', 'localhost'),
            port=getattr(app_config, 'port', 5000),
            debug=getattr(app_config, 'debug', True),
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
        print(f"📋 Traceback:")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Soluciones:")
        print("   1. Verificar que app/routes/main.py existe")
        print("   2. Verificar que app/core/config.py funciona")
        print("   3. Ejecutar: python scripts/system_diagnosis.py")
        print("   4. Revisar imports en app/__init__.py")
        print("   5. Verificar requirements.txt instalado")
        sys.exit(1)


if __name__ == "__main__":
    main()