#!/usr/bin/env python3
"""
Script de Setup Automático para Ollama - CORREGIDO
TFM Vicente Caruncho - Sistemas Inteligentes UJI
"""

import os
import sys
import subprocess
import requests
import time
from pathlib import Path

def check_ollama_installed():
    """Verificar si Ollama está instalado"""
    try:
        result = subprocess.run(['ollama', '--version'], 
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Ollama encontrado: {version}")
            return True
        else:
            print("❌ Ollama no está instalado o no está en PATH")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Ollama no está instalado")
        return False

def check_ollama_running():
    """Verificar si el servidor Ollama está ejecutándose"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            print("✅ Servidor Ollama ejecutándose en localhost:11434")
            return True
        else:
            print(f"❌ Servidor Ollama no responde (status: {response.status_code})")
            return False
    except requests.exceptions.RequestException:
        print("❌ Servidor Ollama no está ejecutándose")
        return False

def get_installed_models():
    """Obtener lista de modelos instalados"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return models
        return []
    except Exception as e:
        print(f"❌ Error obteniendo modelos: {e}")
        return []

def quick_test():
    """Prueba rápida para verificar que todo funciona"""
    print("\n⚡ PRUEBA RÁPIDA DEL SISTEMA")
    print("=" * 30)
    
    try:
        # Test 1: Servidor Ollama
        if not check_ollama_running():
            print("❌ Servidor Ollama no está ejecutándose")
            return False
        
        # Test 2: Modelos disponibles
        models = get_installed_models()
        if not models:
            print("❌ No hay modelos instalados")
            return False
        
        print(f"✅ {len(models)} modelos disponibles")
        for model in models[:3]:  # Mostrar primeros 3
            print(f"   - {model}")
        
        # Test 3: Generación rápida
        test_model = models[0]
        print(f"🔄 Probando generación con {test_model}...")
        
        payload = {
            "model": test_model,
            "prompt": "Hola",
            "stream": False,
            "options": {"max_tokens": 10}
        }
        
        response = requests.post(
            'http://localhost:11434/api/generate',
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            generated_text = data.get('response', '').strip()
            print(f"✅ Generación exitosa: {generated_text[:50]}...")
            return True
        else:
            print(f"❌ Error en generación: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error en prueba rápida: {e}")
        return False

def main():
    """Función principal del setup"""
    print("🎓 TFM VICENTE CARUNCHO - SETUP OLLAMA")
    print("🏛️ Prototipo Chatbot RAG para Administraciones Locales")
    print("🤖 Configuración de Modelos de Lenguaje Locales")
    print("=" * 60)
    
    # Solo verificación básica para empezar
    print("\n🔍 Verificando instalación de Ollama...")
    if not check_ollama_installed():
        print("❌ Ollama no está instalado")
        print("💡 Descarga Ollama desde: https://ollama.ai/download")
        print("💡 Instala y vuelve a ejecutar este script")
        return 1
    
    print("\n🖥️ Verificando servidor Ollama...")
    if not check_ollama_running():
        print("❌ Servidor Ollama no está ejecutándose")
        print("💡 Intenta ejecutar: ollama serve")
        print("💡 O reinicia Ollama desde el menú de Windows")
        return 1
    
    print("\n📋 Verificando modelos instalados...")
    models = get_installed_models()
    if models:
        print(f"✅ {len(models)} modelos disponibles:")
        for model in models:
            print(f"   ✅ {model}")
    else:
        print("⚠️ No hay modelos instalados")
        print("💡 Para instalar el modelo principal: ollama pull llama3.2:3b")
        return 1
    
    print("\n✅ ¡Setup básico completado!")
    print("🚀 Ollama está funcionando correctamente")
    
    return 0

if __name__ == "__main__":
    try:
        # Verificar si se solicita prueba rápida
        if len(sys.argv) > 1 and sys.argv[1] == '--test':
            if quick_test():
                print("🎉 Sistema funcionando correctamente")
                sys.exit(0)
            else:
                print("❌ Sistema requiere configuración")
                sys.exit(1)
        else:
            # Setup completo
            exit_code = main()
            sys.exit(exit_code)
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Setup cancelado por el usuario")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        sys.exit(1)