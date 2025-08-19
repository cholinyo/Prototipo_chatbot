#!/usr/bin/env python3
"""
Prototipo_chatbot - Punto de entrada principal SIMPLIFICADO
Chatbot RAG para Administraciones Locales

TFM Vicente Caruncho - Sistemas Inteligentes UJI
Versión: 2.0 (Simplificada)
"""
import sys
import os
import time
from pathlib import Path

# Añadir el directorio raíz al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def create_flask_app():
    """Crear aplicación Flask con configuración básica"""
    try:
        from app import create_app
        from app.core.config import get_config
        from app.core.logger import get_logger
        
        # Obtener configuración y logger
        config = get_config()
        logger = get_logger()
        
        # Crear aplicación
        app = create_app()
        
        logger.info("Aplicación Flask creada exitosamente")
        return app, config, logger
        
    except Exception as e:
        print(f"❌ Error creando aplicación Flask: {e}")
        raise


def setup_basic_routes(app, config, logger):
    """Configurar rutas básicas de la aplicación"""
    from flask import jsonify
    
    @app.route('/health')
    def health_check():
        """Endpoint de health check básico"""
        return jsonify({
            "status": "healthy",
            "timestamp": time.time(),
            "version": getattr(config, 'version', '2.0'),
            "environment": getattr(config, 'environment', 'development')
        })
    
    @app.route('/api/status')
    def api_status():
        """Status de la API"""
        try:
            # Verificación rápida de servicios críticos
            services_status = {
                "timestamp": time.time(),
                "status": "healthy",
                "services": {}
            }
            
            # Verificar LLM Service si está disponible
            try:
                from app.services.llm.llm_services import LLMService
                llm_service = LLMService()
                health = llm_service.health_check()
                services_status["services"]["llm_service"] = health
            except Exception as e:
                services_status["services"]["llm_service"] = {
                    "status": "error", 
                    "error": str(e)
                }
            
            return jsonify(services_status)
            
        except Exception as e:
            logger.error(f"Error en status check: {e}")
            return jsonify({
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }), 500


def register_blueprints(app, logger):
    """Registrar blueprints de la aplicación"""
    blueprints_registered = 0
    
    try:
        # Blueprint principal (obligatorio)
        from app.routes.main import main_bp
        app.register_blueprint(main_bp)
        logger.info("Blueprint main registrado")
        blueprints_registered += 1
        
        # Blueprints opcionales
        optional_blueprints = [
            ('app.routes.api.data_sources', 'data_sources_api', 'data_sources_api'),
            ('app.routes.api.chat', 'chat_api', 'chat_api'),
            ('app.routes.api.comparison', 'comparison_api', 'comparison_api')
        ]
        
        for module_path, blueprint_name, display_name in optional_blueprints:
            try:
                module = __import__(module_path, fromlist=[blueprint_name])
                blueprint = getattr(module, blueprint_name)
                app.register_blueprint(blueprint)
                logger.info(f"Blueprint {display_name} registrado")
                blueprints_registered += 1
            except (ImportError, AttributeError) as e:
                logger.warning(f"Blueprint {display_name} no disponible: {e}")
                
    except Exception as e:
        logger.error(f"Error registrando blueprints: {e}")
        raise
    
    return blueprints_registered


def print_startup_info(config, blueprints_count):
    """Mostrar información de inicio"""
    print("\n" + "=" * 60)
    print("🎓 TFM Vicente Caruncho - Prototipo Chatbot RAG")
    print("🏛️ Administraciones Locales - UJI")
    print("=" * 60)
    print(f"🚀 Aplicación: {getattr(config, 'project_name', 'Prototipo_chatbot')}")
    print(f"🐍 Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"🔧 Blueprints: {blueprints_count} registrados")
    print(f"📝 Versión: {getattr(config, 'version', '2.0')}")
    print(f"🏷️ Entorno: {getattr(config, 'environment', 'development')}")
    print(f"\n🌐 Servidor: http://{getattr(config, 'host', 'localhost')}:{getattr(config, 'port', 5000)}")
    print(f"🩺 Health Check: http://{getattr(config, 'host', 'localhost')}:{getattr(config, 'port', 5000)}/health")
    print(f"\n⚠️ Usa Ctrl+C para detener el servidor")
    print("=" * 60)


def main():
    """Función principal simplificada"""
    print("🚀 Iniciando Prototipo_chatbot TFM v2.0...")
    print(f"📁 Directorio: {project_root}")
    
    try:
        # Crear aplicación Flask
        app, config, logger = create_flask_app()
        
        # Configurar rutas básicas
        setup_basic_routes(app, config, logger)
        
        # Registrar blueprints
        blueprints_count = register_blueprints(app, logger)
        
        # Mostrar información de inicio
        print_startup_info(config, blueprints_count)
        
        # Marcar tiempo de inicio
        app.start_time = time.time()
        
        # Iniciar servidor
        logger.info(f"Servidor iniciando en http://{getattr(config, 'host', 'localhost')}:{getattr(config, 'port', 5000)}")
        
        app.run(
            host=getattr(config, 'host', 'localhost'),
            port=getattr(config, 'port', 5000),
            debug=getattr(config, 'debug', True),
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
        print("   1. Ejecutar diagnóstico: python scripts/system_diagnosis.py")
        print("   2. Verificar health: python scripts/health_check.py")
        print("   3. Instalar dependencias: pip install -r requirements.txt")
        print("   4. Revisar logs en logs/ para más detalles")
        sys.exit(1)


if __name__ == "__main__":
    main()
