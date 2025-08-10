#!/usr/bin/env python3
"""
Diagnóstico Rápido - Prototipo_chabot
TFM Vicente Caruncho
"""

import os
import sys
from pathlib import Path

def main():
    print("🔍 DIAGNÓSTICO RÁPIDO - Prototipo_chabot")
    print("=" * 50)
    
    # Directorio actual
    current = Path.cwd()
    print(f"📁 Directorio actual: {current}")
    print(f"📁 Nombre del directorio: {current.name}")
    
    # Buscar estructura
    print(f"\n📂 Contenido del directorio actual:")
    for item in sorted(current.iterdir()):
        if item.is_dir():
            print(f"   📁 {item.name}/")
        else:
            print(f"   📄 {item.name}")
    
    # Verificar si existe app/
    app_dir = current / "app"
    print(f"\n🔍 Verificando app/...")
    if app_dir.exists():
        print(f"   ✅ app/ encontrado")
        print(f"   📂 Contenido de app/:")
        for item in sorted(app_dir.iterdir()):
            if item.is_dir():
                print(f"      📁 {item.name}/")
            else:
                print(f"      📄 {item.name}")
    else:
        print(f"   ❌ app/ NO encontrado")
    
    # Verificar archivos clave
    key_files = ["run.py", "requirements.txt", ".env", ".env.example"]
    print(f"\n📋 Archivos clave:")
    for file in key_files:
        file_path = current / file
        if file_path.exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}")
    
    # Test de importación Python
    print(f"\n🐍 Test de importación:")
    sys.path.insert(0, str(current))
    
    try:
        import app
        print(f"   ✅ import app - OK")
        
        try:
            from app.core import config
            print(f"   ✅ from app.core import config - OK")
        except:
            print(f"   ❌ from app.core import config - FALLO")
            
        try:
            from app.core import logger
            print(f"   ✅ from app.core import logger - OK")  
        except:
            print(f"   ❌ from app.core import logger - FALLO")
            
    except Exception as e:
        print(f"   ❌ import app - FALLO: {e}")
    
    # Test Ollama
    print(f"\n🤖 Test Ollama:")
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=3)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            print(f"   ✅ Servidor Ollama funcionando")
            print(f"   📊 Modelos instalados: {len(models)}")
            for model in models:
                print(f"      - {model}")
            
            if not models:
                print(f"   ⚠️  No hay modelos instalados")
                print(f"   💡 Ejecuta: ollama pull llama3.2:3b")
        else:
            print(f"   ❌ Servidor responde pero con error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Ollama no disponible: {e}")
        print(f"   💡 Asegúrate de que Ollama esté ejecutándose")
    
    # Recomendaciones
    print(f"\n💡 PRÓXIMOS PASOS:")
    
    if not app_dir.exists():
        print(f"   1. ❌ Crear estructura app/")
        print(f"      mkdir app app\\core app\\models app\\services app\\routes")
        print(f"      echo. > app\\__init__.py")
        print(f"      echo. > app\\core\\__init__.py")
    else:
        print(f"   1. ✅ Estructura app/ existe")
    
    print(f"   2. 🤖 Instalar modelo Ollama:")
    print(f"      ollama pull llama3.2:3b")
    
    print(f"   3. 🧪 Ejecutar test real del sistema")
    
    print(f"\n🎯 ESTADO GENERAL:")
    if app_dir.exists():
        print(f"   🟢 BIEN - Estructura básica encontrada")
        print(f"   ➡️  Proceder con implementación del LLM Service")
    else:
        print(f"   🔴 PROBLEMA - Estructura app/ falta")
        print(f"   ➡️  Crear estructura primero")

if __name__ == "__main__":
    main()