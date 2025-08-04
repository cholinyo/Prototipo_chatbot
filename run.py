#!/usr/bin/env python3
"""
Prototipo_chatbot - Punto de entrada principal
Chatbot RAG para Administraciones Locales
"""
import sys
import os
from pathlib import Path

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
    required_dirs = ['logs', 'data/vectorstore/faiss', 'data/vectorstore/chromadb']
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)

def main():
    """Función principal de arranque"""
    print("🚀 Iniciando Prototipo_chatbot...")
    print("📁 Directorio del proyecto:", project_root)
    
    try:
        # Crear archivos faltantes
        create_missing_files()
        
        # Importar después de crear archivos necesarios
        try:
            from flask import Flask, render_template
            from app.core.config import get_app_config, is_development
            from app.core.logger import setup_logging, get_logger
        except ImportError as e:
            print(f"❌ Error importando dependencias: {e}")
            print("💡 Instala las dependencias con: pip install -r requirements.txt")
            sys.exit(1)
        
        # Configurar logging
        setup_logging()
        logger = get_logger("main")
        
        # Obtener configuración
        app_config = get_app_config()
        
        # Crear aplicación Flask básica
        app = Flask(__name__, 
                   template_folder='app/templates',
                   static_folder='app/static')
        
        app.config.update(
            SECRET_KEY=os.getenv('SECRET_KEY', 'dev-secret-key'),
            DEBUG=app_config.debug
        )
        
        # Rutas básicas
        @app.route('/')
        def index():
            """Página principal"""
            return render_template('index.html',
                                 app_name=app_config.name,
                                 app_version=app_config.version,
                                 app_description=app_config.description)
        
        @app.route('/health')
        def health_check():
            """Endpoint de verificación de salud"""
            return {
                'status': 'healthy',
                'version': app_config.version,
                'name': app_config.name
            }
        
        @app.route('/chat')
        def chat_endpoint():
            """Endpoint temporal para chat"""
            return {"message": "Chat endpoint - En desarrollo"}
        
        # Handler de errores
        @app.errorhandler(404)
        def not_found_error(error):
            return render_template('error.html'), 404
        
        @app.errorhandler(500)
        def internal_error(error):
            return render_template('error.html'), 500
        
        logger.info("Aplicación Flask creada exitosamente")
        print("✅ Aplicación Flask configurada")
        print("🌐 Accede a: http://localhost:5000")
        print("⚠️  Usa Ctrl+C para detener el servidor")
        
        # Iniciar servidor
        app.run(
            host=app_config.host,
            port=app_config.port,
            debug=app_config.debug,
            use_reloader=True
        )
        
    except KeyboardInterrupt:
        print("\n⏹️  Aplicación detenida por el usuario")
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        print("💡 Verifica que todas las dependencias estén instaladas:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()