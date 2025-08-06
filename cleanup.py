#!/usr/bin/env python3
"""
Script de limpieza y reorganización del proyecto Prototipo_chatbot
TFM Vicente Caruncho - Sistemas Inteligentes
"""

import os
import shutil
from pathlib import Path
import sys

def cleanup_project():
    """Limpiar archivos conflictivos y reorganizar estructura"""
    
    print("🧹 LIMPIEZA Y REORGANIZACIÓN DEL PROYECTO")
    print("=" * 60)
    
    # Directorio actual
    project_root = Path.cwd()
    print(f"📁 Directorio del proyecto: {project_root}")
    
    # Archivos a eliminar (conflictivos o duplicados)
    files_to_remove = [
        # Archivos duplicados en ubicación incorrecta
        "app/services/embedding_service.py",  # Debe estar en app/services/rag/embeddings.py
        "app/services/ingestion_service.py",   # Debe estar en app/services/ingestion/__init__.py
        "app/services/rag_service.py",         # Debe estar en app/services/rag/__init__.py
        
        # Archivos temporales y cache
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.pyd",
        "**/*.so",
        "**/*.egg-info",
        
        # Archivos de log antiguos
        "logs/*.log.*",
        
        # Archivos de prueba temporal
        "test_*.py",
        "temp_*.py",
    ]
    
    print("\n📝 Eliminando archivos conflictivos...")
    for pattern in files_to_remove:
        if "**" in pattern:
            # Patrón con wildcards
            for file in project_root.glob(pattern):
                try:
                    if file.is_dir():
                        shutil.rmtree(file)
                        print(f"   ❌ Eliminado directorio: {file.relative_to(project_root)}")
                    else:
                        file.unlink()
                        print(f"   ❌ Eliminado: {file.relative_to(project_root)}")
                except Exception as e:
                    print(f"   ⚠️  No se pudo eliminar {file}: {e}")
        else:
            # Archivo específico
            file_path = project_root / pattern
            if file_path.exists():
                try:
                    file_path.unlink()
                    print(f"   ❌ Eliminado: {pattern}")
                except Exception as e:
                    print(f"   ⚠️  No se pudo eliminar {pattern}: {e}")
    
    # Crear estructura correcta de directorios
    print("\n📁 Creando estructura correcta de directorios...")
    
    directories = [
        "app",
        "app/core",
        "app/models",
        "app/routes",
        "app/services",
        "app/services/rag",
        "app/services/ingestion",
        "app/services/llm",
        "app/templates",
        "app/templates/errors",
        "app/templates/chat",
        "app/templates/admin",
        "app/static",
        "app/static/css",
        "app/static/js",
        "app/static/images",
        "app/utils",
        "config",
        "data",
        "data/documents",
        "data/vectorstore",
        "data/vectorstore/faiss",
        "data/vectorstore/chromadb",
        "data/temp",
        "data/cache",
        "data/cache/embeddings",
        "logs",
        "tests",
        "tests/unit",
        "tests/integration",
        "tests/fixtures"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    print("\n✅ Limpieza completada")
    return True

def verify_structure():
    """Verificar la estructura después de la limpieza"""
    
    print("\n🔍 VERIFICANDO ESTRUCTURA DEL PROYECTO")
    print("=" * 60)
    
    project_root = Path.cwd()
    
    # Archivos esenciales que deben existir
    essential_files = [
        "app/__init__.py",
        "app/core/__init__.py",
        "app/core/config.py",
        "app/core/logger.py",
        "app/models/__init__.py",
        "app/services/__init__.py",
        "app/services/rag/__init__.py",
        "app/services/ingestion/__init__.py",
        "app/routes/__init__.py",
    ]
    
    missing = []
    for file_path in essential_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - FALTA")
            missing.append(file_path)
    
    if missing:
        print(f"\n⚠️  Faltan {len(missing)} archivos esenciales")
        print("   Ejecuta fix_project.py para crearlos")
    else:
        print("\n✅ Todos los archivos esenciales están presentes")
    
    return len(missing) == 0

if __name__ == "__main__":
    print("🚀 Iniciando limpieza del proyecto...")
    print("⚠️  Este script eliminará archivos conflictivos")
    
    response = input("\n¿Deseas continuar? (s/n): ")
    if response.lower() != 's':
        print("❌ Operación cancelada")
        sys.exit(0)
    
    # Ejecutar limpieza
    if cleanup_project():
        # Verificar estructura
        if verify_structure():
            print("\n✅ Proyecto limpio y listo")
            print("   Ejecuta ahora: python fix_project.py")
        else:
            print("\n⚠️  Proyecto limpio pero faltan archivos")
            print("   Ejecuta: python fix_project.py")
    else:
        print("\n❌ Error durante la limpieza")
        sys.exit(1)