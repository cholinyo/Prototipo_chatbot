#!/usr/bin/env python3
"""
Quick Fix - Añadir create_app al app/__init__.py existente
TFM Vicente Caruncho - Sistemas Inteligentes UJI
"""
import os
from pathlib import Path

def fix_app_init():
    """Añadir función create_app al app/__init__.py existente"""
    project_root = Path(__file__).parent.parent
    app_init_path = project_root / "app" / "__init__.py"
    
    print("🔧 Quick Fix - Añadiendo create_app a app/__init__.py")
    print("=" * 50)
    
    # Leer contenido actual
    if app_init_path.exists():
        current_content = app_init_path.read_text(encoding='utf-8')
        print(f"📄 Archivo existente: {len(current_content.split())} líneas")
    else:
        current_content = '"""Rutas de la aplicacion"""\n'
        print("📄 Archivo no existe, creando nuevo")
    
    # Verificar si ya tiene create_app
    if 'def create_app' in current_content:
        print("✅ create_app ya existe en app/__init__.py")
        return True
    
    # Contenido adicional para añadir
    create_app_function = '''

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
    
    # Blueprint principal
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
        
        @main_bp.route('/health')
        def health():
            return jsonify({
                "status": "healthy",
                "timestamp": __import__('time').time(),
                "mode": "basic"
            })
        
        app.register_blueprint(main_bp)
        app.logger.info("✅ Blueprint main básico creado como fallback")
        blueprints_registered += 1
    
    # Blueprints opcionales
    optional_blueprints = [
        ('app.routes.api.data_sources', 'data_sources_api', '/api/data-sources'),
        ('app.routes.api.chat', 'chat_api', '/api/chat'),  
        ('app.routes.api.comparison', 'comparison_api', '/api/comparison'),
        ('app.routes.chat', 'chat_bp', '/chat'),
        ('app.routes.admin', 'admin_bp', '/admin')
    ]
    
    for module_path, blueprint_name, url_prefix in optional_blueprints:
        try:
            module = __import__(module_path, fromlist=[blueprint_name])
            blueprint = getattr(module, blueprint_name)
            app.register_blueprint(blueprint, url_prefix=url_prefix)
            app.logger.info(f"✅ Blueprint {blueprint_name} registrado en {url_prefix}")
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
'''
    
    # Combinar contenido actual + nueva función
    new_content = current_content + create_app_function
    
    # Crear backup
    backup_path = project_root / "backup_reorganization" / "app_init_original.py"
    backup_path.parent.mkdir(exist_ok=True)
    backup_path.write_text(current_content, encoding='utf-8')
    
    # Escribir nuevo contenido
    app_init_path.write_text(new_content, encoding='utf-8')
    
    print(f"💾 Backup original: {backup_path}")
    print(f"✅ create_app añadida a app/__init__.py")
    print(f"📊 Líneas añadidas: {len(create_app_function.split('\\n'))}")
    
    return True

def main():
    """Función principal del quick fix"""
    print("🎓 TFM Vicente Caruncho - Quick Fix create_app")
    
    try:
        success = fix_app_init()
        
        if success:
            print("\\n🎉 ¡Quick fix completado!")
            print("\\n💡 Próximos pasos:")
            print("   1. Probar aplicación: python run.py")
            print("   2. Verificar funcionamiento en: http://localhost:5000")
            print("   3. Si funciona, continuar con siguientes fases")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ Error en quick fix: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\\n🚀 ¡Listo para probar run.py!")
        import sys
        sys.exit(0)
    else:
        print("\\n⚠️ Quick fix falló.")
        import sys
        sys.exit(1)