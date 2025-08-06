#!/usr/bin/env python3
"""
Script de configuración de Ollama para TFM
Vicente Caruncho - Sistemas Inteligentes

Este script:
1. Verifica si Ollama está instalado y funcionando
2. Descarga los modelos recomendados para el TFM
3. Realiza pruebas básicas de funcionamiento
4. Configura el entorno para desarrollo

Ejecutar: python scripts/setup_ollama.py
"""

import os
import sys
import subprocess
import requests
import json
import time
from pathlib import Path

def print_header(text):
    """Imprimir cabecera con formato"""
    print(f"\n{'='*60}")
    print(f"🚀 {text}")
    print('='*60)

def print_step(step, description):
    """Imprimir paso con formato"""
    print(f"\n{step}. {description}")
    print("-" * 40)

def check_ollama_installation():
    """Verificar si Ollama está instalado"""
    print_step(1, "VERIFICANDO INSTALACIÓN DE OLLAMA")
    
    try:
        # Verificar comando ollama
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Ollama instalado: {version}")
            return True
        else:
            print("❌ Comando 'ollama' no encontrado")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Ollama no está instalado o no está en PATH")
        print("\n🔧 Para instalar Ollama:")
        print("   Windows: https://ollama.ai/download/windows")
        print("   macOS: https://ollama.ai/download/mac") 
        print("   Linux: curl -fsSL https://ollama.ai/install.sh | sh")
        return False

def check_ollama_service():
    """Verificar si el servicio de Ollama está funcionando"""
    print_step(2, "VERIFICANDO SERVICIO OLLAMA")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Servicio Ollama funcionando en puerto 11434")
            return True
        else:
            print(f"❌ Servicio respondió con código {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ No se puede conectar al servicio Ollama")
        print("\n🔧 Soluciones:")
        print("   1. Ejecutar: ollama serve")
        print("   2. Reiniciar el servicio de Ollama")
        print("   3. Verificar que no hay firewall bloqueando puerto 11434")
        return False
    except requests.exceptions.Timeout:
        print("❌ Timeout conectando con Ollama")
        return False

def get_installed_models():
    """Obtener modelos ya instalados"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return models
        return []
    except:
        return []

def install_model(model_name, description):
    """Instalar un modelo específico"""
    print(f"\n📥 Descargando {model_name} ({description})...")
    
    try:
        # Ejecutar ollama pull
        process = subprocess.Popen(
            ['ollama', 'pull', model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Mostrar