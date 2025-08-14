#!/usr/bin/env python3
"""
Script de diagnóstico y resolución para configuración de modelos
TFM Vicente Caruncho - Prototipo Chatbot RAG
"""

import os
import sys
import requests
import subprocess
from pathlib import Path

def check_env_file():
    """Verificar archivo .env"""
    print("🔧 VERIFICANDO ARCHIVO .env...")
    
    env_path = Path(".env")
    env_example = Path(".env.example")
    
    if not env_path.exists():
        print("❌ Archivo .env no existe")
        if env_example.exists():
            print("💡 Creando .env desde .env.example...")
            import shutil
            shutil.copy2(env_example, env_path)
            print("✅ Archivo .env creado")
        else:
            print("❌ Tampoco existe .env.example")
            return False
    else:
        print("✅ Archivo .env existe")
    
    # Verificar contenido
    with open(env_path, 'r') as f:
        content = f.read()
    
    print("\n📋 Variables importantes en .env:")
    
    # OpenAI API Key
    if "OPENAI_API_KEY=" in content:
        lines = content.split('\n')
        for line in lines:
            if line.startswith('OPENAI_API_KEY='):
                key = line.split('=', 1)[1].strip()
                if key and key != 'sk-tu-api-key-aqui' and key.startswith('sk-'):
                    print("✅ OPENAI_API_KEY configurada correctamente")
                else:
                    print("❌ OPENAI_API_KEY no configurada o incorrecta")
                    print("💡 Edita .env y añade: OPENAI_API_KEY=sk-tu-api-key-real")
                break
    else:
        print("❌ OPENAI_API_KEY no encontrada en .env")
        print("💡 Añadir a .env: OPENAI_API_KEY=sk-tu-api-key-aqui")
    
    # Ollama URL
    if "OLLAMA_BASE_URL=" in content:
        print("✅ OLLAMA_BASE_URL configurada")
    else:
        print("⚠️ OLLAMA_BASE_URL no encontrada, usando default localhost:11434")
    
    return True

def check_ollama_status():
    """Verificar estado de Ollama"""
    print("\n🦙 VERIFICANDO OLLAMA...")
    
    # 1. Verificar si Ollama está instalado
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Ollama instalado: {version}")
        else:
            print("❌ Ollama no está instalado")
            print("💡 Instalar desde: https://ollama.ai/download")
            return False
    except FileNotFoundError:
        print("❌ Comando 'ollama' no encontrado")
        print("💡 Instalar Ollama desde: https://ollama.ai/download")
        return False
    
    # 2. Verificar si el servidor está ejecutándose
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Servidor Ollama ejecutándose")
            
            # 3. Verificar modelos instalados
            models_data = response.json()
            models = [model['name'] for model in models_data.get('models', [])]
            
            if models:
                print(f"✅ Modelos disponibles: {len(models)}")
                for model in models:
                    print(f"   - {model}")
            else:
                print("❌ No hay modelos instalados")
                print("💡 Instalar modelo: ollama pull llama3.2:3b")
                return False
            
            return True
        else:
            print(f"❌ Servidor Ollama no responde: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ No se puede conectar a Ollama")
        print("💡 Iniciar Ollama: ollama serve")
        return False

def check_openai_api():
    """Verificar API de OpenAI"""
    print("\n🤖 VERIFICANDO OPENAI API...")
    
    # Cargar variables de entorno
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY no configurada")
        print("💡 Añadir a .env: OPENAI_API_KEY=sk-tu-api-key-real")
        return False
    
    if not api_key.startswith("sk-"):
        print("❌ OPENAI_API_KEY tiene formato incorrecto")
        print("💡 Debe empezar con 'sk-'")
        return False
    
    print(f"✅ API Key configurada: {api_key[:10]}...{api_key[-4:]}")
    
    # Probar conexión
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        # Test simple
        models = client.models.list()
        print("✅ Conexión con OpenAI exitosa")
        print(f"✅ Modelos disponibles: {len(models.data)}")
        return True
        
    except Exception as e:
        print(f"❌ Error conectando con OpenAI: {e}")
        return False

def fix_configuration():
    """Intentar arreglar configuración automáticamente"""
    print("\n🔧 INTENTANDO ARREGLAR CONFIGURACIÓN...")
    
    # 1. Crear .env si no existe
    check_env_file()
    
    # 2. Verificar y arreglar Ollama
    print("\n2. Ollama:")
    if not check_ollama_status():
        print("⚠️ Ollama requiere configuración manual:")
        print("   1. Instalar: https://ollama.ai/download")
        print("   2. Ejecutar: ollama serve")
        print("   3. Instalar modelo: ollama pull llama3.2:3b")
    
    # 3. Verificar OpenAI
    print("\n3. OpenAI:")
    if not check_openai_api():
        print("⚠️ OpenAI requiere configuración manual:")
        print("   1. Obtener API key: https://platform.openai.com/api-keys")
        print("   2. Editar .env: OPENAI_API_KEY=sk-tu-api-key-real")

def restart_application():
    """Reiniciar aplicación después de cambios"""
    print("\n🔄 Para aplicar cambios:")
    print("1. Presiona Ctrl+C para detener el servidor")
    print("2. Ejecuta: python run.py")
    print("3. Abre: http://localhost:5000")

def main():
    """Función principal"""
    print("🎓 TFM VICENTE CARUNCHO - DIAGNÓSTICO DE CONFIGURACIÓN")
    print("🏛️ Prototipo Chatbot RAG para Administraciones Locales")
    print("🔍 Resolviendo problemas de configuración...")
    print("=" * 60)
    
    # Cambiar al directorio del proyecto
    project_root = Path(__file__).parent.parent if Path(__file__).parent.name == 'scripts' else Path(__file__).parent
    os.chdir(project_root)
    print(f"📁 Directorio de trabajo: {project_root}")
    
    # Ejecutar diagnóstico completo
    fix_configuration()
    
    print("\n" + "=" * 60)
    print("✅ DIAGNÓSTICO COMPLETADO")
    print("💡 Revisa los mensajes anteriores y aplica las correcciones")
    restart_application()

if __name__ == "__main__":
    main()