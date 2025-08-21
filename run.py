#!/usr/bin/env python3
"""
Prototipo_chatbot - Punto de entrada principal
TFM Vicente Caruncho - Sistemas Inteligentes UJI
Versión: 3.0
"""
import sys
import os
import time
from pathlib import Path


# Añadir el directorio raíz al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def verify_basic_structure():
    """Verificar estructura básica del proyecto"""
    required_dirs = ['app', 'app/core', 'app/routes', 'app/services']
    required_files = ['app/__init__.py', 'app/core/config.py', 'app/core/logger.py']
    
    missing_dirs = []
    missing_files = []
    
    for dir_path in required_dirs:
        if not (project_root / dir_path).exists():
            missing_dirs.append(dir_path)
    
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)
    
    if missing_dirs or missing_files:
        print("❌ Estructura del proyecto incompleta:")
        if missing_dirs:
            print(f"   📁 Directorios faltantes: {missing_dirs}")
        if missing_files:
            print(f"   📄 Archivos faltantes: {missing_files}")
        return False
    
    return True


def create_essential_directories():
    """Crear directorios esenciales"""
    essential_dirs = [
        'logs',
        'data',
        'data/ingestion',
        'data/vectorstore',
        'app/templates',
        'app/static'
    ]
    
    for dir_name in essential_dirs:
        dir_path = project_root / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("✅ Directorios esenciales verificados/creados")


def test_basic_imports():
    """Probar imports básicos antes del startup"""
    try:
        # Test Flask
        import flask
        print(f"✅ Flask {flask.__version__} disponible")
        
        # Test imports básicos del proyecto
        from app import create_app
        print("✅ Factory create_app importado")
        
        from app.core.config import get_app_config
        print("✅ Configuración importada")
        
        from app.core.logger import get_logger
        print("✅ Logger importado")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False


def main():
    """Función principal - Punto de entrada único"""
    print("🚀 Iniciando Prototipo_chatbot TFM v3.0...")
    print(f"📁 Directorio: {project_root}")
    print(f"🐍 Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Verificaciones previas
    if not verify_basic_structure():
        print("\n💡 Soluciones:")
        print("   1. Verificar que estás en el directorio correcto del proyecto")
        print("   2. Verificar que la estructura de archivos está completa")
        print("   3. Clonar el repositorio completo si falta algo")
        sys.exit(1)
    
    # Crear directorios esenciales
    create_essential_directories()
    
    # Test de imports básicos
    if not test_basic_imports():
        print("\n💡 Soluciones:")
        print("   1. Instalar dependencias: pip install -r requirements.txt")
        print("   2. Verificar que Flask está instalado: pip install flask")
        print("   3. Verificar estructura de módulos Python")
        sys.exit(1)
    
    try:
        # Imports después de verificación
        from app import create_app
        from app.core.config import get_app_config
        from app.core.logger import get_logger
        
        # Obtener configuración
        app_config = get_app_config()
        logger = get_logger("main")
        
        logger.info("Iniciando aplicación", extra={
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "project_root": str(project_root)
        })
        
        # Crear aplicación (usa tu factory en app/__init__.py)
        app = create_app()
        
        # Contar blueprints registrados
        blueprints_count = len(app.blueprints)
        blueprint_names = list(app.blueprints.keys())
        
        # Verificación específica de web_sources_api
        web_scraping_available = 'web_sources_api' in blueprint_names
        
        # Mostrar información de inicio
        print_startup_info(app_config, blueprints_count, blueprint_names, web_scraping_available)
        
        # Marcar tiempo de inicio
        app.start_time = time.time()
        
        # Log del estado del sistema
        logger.info("Sistema listo para arrancar", extra={
            "blueprints": blueprint_names,
            "blueprints_count": blueprints_count,
            "web_scraping_available": web_scraping_available,
            "debug_mode": app_config.debug
        })
        
        # Iniciar servidor
        print(f"\n🌐 Arrancando servidor en http://{app_config.host}:{app_config.port}")
        logger.info(f"Servidor Flask iniciando en {app_config.host}:{app_config.port}")
        
        app.run(
            host=app_config.host,
            port=app_config.port,
            debug=app_config.debug,
            use_reloader=True,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n⏹️ Aplicación detenida por el usuario")
        print("👋 ¡Hasta luego!")
        sys.exit(0)
        
    except ImportError as e:
        print(f"\n❌ Error de importación: {e}")
        print("\n💡 Soluciones:")
        print("   1. Verificar que app/__init__.py existe y es válido")
        print("   2. Verificar que app/core/config.py funciona")
        print("   3. Instalar dependencias: pip install -r requirements.txt")
        print("   4. Verificar PYTHONPATH y estructura de módulos")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        print("\n🔍 Información de debug:")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Soluciones:")
        print("   1. Revisar logs en la carpeta logs/")
        print("   2. Verificar configuración en config/")
        print("   3. Ejecutar diagnóstico: python scripts/system_diagnosis.py")
        print("   4. Comprobar permisos de archivos y directorios")
        sys.exit(1)


def print_startup_info(app_config, blueprints_count, blueprint_names, web_scraping_available):
    """Mostrar información detallada de inicio"""
    print("\n" + "=" * 75)
    print("🎓 TFM Vicente Caruncho - Prototipo Chatbot RAG")
    print("🏛️ Administraciones Locales - Universitat Jaume I")
    print("=" * 75)
    print(f"🚀 Aplicación: {app_config.name}")
    print(f"📝 Versión: {app_config.version}")
    print(f"🏷️ Entorno: {'Desarrollo' if app_config.debug else 'Producción'}")
    print(f"🔧 Blueprints: {blueprints_count} registrados")
    
    print(f"\n🌐 URLs del Sistema:")
    print(f"   🏠 Aplicación: http://{app_config.host}:{app_config.port}")
    print(f"   🩺 Health Check: http://{app_config.host}:{app_config.port}/health")
    print(f"   📊 Quick Stats: http://{app_config.host}:{app_config.port}/ajax/quick-stats")
    
    # Estado del Web Scraping
    print(f"\n🌐 ESTADO WEB SCRAPING:")
    if web_scraping_available:
        print(f"   ✅ web_sources_api: REGISTRADO")
        print(f"   🔗 API: http://{app_config.host}:{app_config.port}/api/web-sources")
        print(f"   🎯 Estado: FUNCIONAL - Scraping listo para usar")
    else:
        print(f"   ❌ web_sources_api: NO REGISTRADO")
        print(f"   ⚠️ Estado: Web scraping NO disponible")
    
    # Mostrar blueprints registrados
    if blueprint_names:
        print(f"\n📋 MÓDULOS ACTIVOS:")
        for i, bp in enumerate(blueprint_names, 1):
            status = "🎯 FUNCIONAL" if bp == 'web_sources_api' else "✅ OK"
            print(f"   {i:2d}. {bp:<20} - {status}")
    
    print("=" * 75)
    print("⚠️  Usa Ctrl+C para detener el servidor")
    print("💡 Si hay problemas, revisar logs/ y usar /health endpoint")
    print("=" * 75)


if __name__ == "__main__":
    main()