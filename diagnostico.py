#!/usr/bin/env python3
"""
Script de diagnóstico para verificar la estructura del proyecto
Prototipo_chatbot - TFM Vicente Caruncho
"""

import os
import sys
from pathlib import Path

def check_project_structure():
    """Verificar estructura del proyecto"""
    print("🔍 DIAGNÓSTICO DE ESTRUCTURA DEL PROYECTO")
    print("=" * 60)
    
    # Directorio actual
    current_dir = Path.cwd()
    print(f"📁 Directorio actual: {current_dir}")
    
    # Buscar archivos clave
    key_files = [
        "run.py",
        "requirements.txt", 
        ".env.example",
        "app/__init__.py",
        "app/core/__init__.py",
        "app/core/config.py",
        "app/core/logger.py", 
        "app/models/__init__.py",
        "app/services/__init__.py",
        "app/services/rag/__init__.py"
    ]
    
    print("\n📋 Verificando archivos clave:")
    missing_files = []
    
    for file_path in key_files:
        full_path = current_dir / file_path
        if full_path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - ¡FALTA!")
            missing_files.append(file_path)
    
    # Verificar estructura de app/
    print("\n📂 Estructura de app/:")
    app_dir = current_dir / "app"
    if app_dir.exists():
        for item in sorted(app_dir.rglob("*")):
            if item.is_file():
                rel_path = item.relative_to(current_dir)
                print(f"   📄 {rel_path}")
    else:
        print("   ❌ Directorio app/ no encontrado")
    
    # Verificar Python path
    print(f"\n🐍 Python path:")
    for i, path in enumerate(sys.path):
        print(f"   {i}: {path}")
    
    return missing_files

def create_missing_files():
    """Crear archivos faltantes básicos"""
    print("\n🔧 CREANDO ARCHIVOS FALTANTES...")
    
    current_dir = Path.cwd()
    
    # Crear directorios
    directories = [
        "app",
        "app/core", 
        "app/models",
        "app/services",
        "app/services/rag",
        "app/templates",
        "app/static",
        "config",
        "data",
        "logs"
    ]
    
    for dir_path in directories:
        full_path = current_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"   📁 {dir_path}")
    
    # Crear archivos __init__.py
    init_files = [
        "app/__init__.py",
        "app/core/__init__.py",
        "app/models/__init__.py", 
        "app/services/__init__.py",
        "app/services/rag/__init__.py"
    ]
    
    for init_file in init_files:
        full_path = current_dir / init_file
        if not full_path.exists():
            full_path.write_text('"""Módulo de Prototipo_chatbot"""\n')
            print(f"   ✅ Creado: {init_file}")
    
    # Crear app/core/logger.py básico
    logger_file = current_dir / "app/core/logger.py"
    if not logger_file.exists():
        logger_content = '''"""
Sistema de logging básico para Prototipo_chatbot
"""
import logging

def get_logger(name: str):
    """Obtener logger básico"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
'''
        logger_file.write_text(logger_content)
        print(f"   ✅ Creado: app/core/logger.py")
    
    # Crear app/models/__init__.py básico
    models_file = current_dir / "app/models/__init__.py" 
    models_content = '''"""
Modelos básicos para Prototipo_chatbot
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class DocumentMetadata:
    """Metadatos básicos de documento"""
    source_path: str
    source_type: str
    file_type: str
    size_bytes: int
    created_at: datetime
    processed_at: datetime
    checksum: str
    
    # Campos opcionales
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    url: Optional[str] = None
    title: Optional[str] = None

@dataclass  
class DocumentChunk:
    """Fragmento de documento"""
    id: str
    content: str
    metadata: DocumentMetadata
    chunk_index: int
    chunk_size: int
    start_char: int
    end_char: int
    embedding: Optional[List[float]] = None
    section_title: Optional[str] = None
    page_number: Optional[int] = None
'''
    
    if not models_file.exists() or models_file.stat().st_size < 100:
        models_file.write_text(models_content)
        print(f"   ✅ Creado: app/models/__init__.py")

def test_imports():
    """Probar imports básicos"""
    print("\n🧪 PROBANDO IMPORTS...")
    
    try:
        # Añadir directorio actual al path
        current_dir = str(Path.cwd())
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            print(f"   📍 Añadido al path: {current_dir}")
        
        # Test import básico
        try:
            from app.core.logger import get_logger
            logger = get_logger("test")
            print("   ✅ app.core.logger - OK")
        except Exception as e:
            print(f"   ❌ app.core.logger - Error: {e}")
        
        # Test import models
        try:
            from app.models import DocumentMetadata, DocumentChunk
            print("   ✅ app.models - OK")
        except Exception as e:
            print(f"   ❌ app.models - Error: {e}")
        
        print("\n🎯 Listo para crear vector_store.py")
        return True
        
    except Exception as e:
        print(f"   ❌ Error general en imports: {e}")
        return False

def main():
    """Función principal de diagnóstico"""
    print("🤖 Prototipo_chatbot - Diagnóstico de Estructura")
    print(f"📍 Ubicación: {Path.cwd()}")
    
    # Verificar estructura
    missing_files = check_project_structure()
    
    if missing_files:
        print(f"\n⚠️ Faltan {len(missing_files)} archivos clave")
        response = input("¿Crear archivos faltantes? (s/n): ")
        if response.lower() in ['s', 'y', 'si', 'yes']:
            create_missing_files()
    
    # Probar imports
    if test_imports():
        print("\n🎉 ¡ESTRUCTURA VERIFICADA!")
        print("✅ Listo para implementar vector_store.py")
        print("\n📋 Próximos pasos:")
        print("1. Copiar vector_store.py a app/services/rag/")
        print("2. Ejecutar test_interface.py")
        print("3. Implementar FAISS y ChromaDB")
    else:
        print("\n❌ Hay problemas con los imports")
        print("💡 Revisa la estructura de archivos")

if __name__ == "__main__":
    main()