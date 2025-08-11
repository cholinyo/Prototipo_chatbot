# -*- coding: utf-8 -*-
"""
Test simple del LLM Service - Mejorado
"""

import sys
from pathlib import Path

# Añadir proyecto al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test de importaciones básicas"""
    print("=== TEST DE IMPORTACIONES ===")
    
    try:
        print("1. Importando app...")
        import app
        print("   OK - app importado")
        
        print("2. Importando app.services...")
        import app.services
        print("   OK - app.services importado")
        
        print("3. Importando app.services.llm...")
        from app.services import llm
        print("   OK - app.services.llm importado")
        
        print("4. Importando llm_service...")
        from app.services.llm import llm_service
        print("   OK - llm_service importado")
        
        return True
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ollama_direct():
    """Test directo de Ollama"""
    print("\n=== TEST OLLAMA DIRECTO ===")
    
    try:
        import requests
        
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            print(f"OK - Ollama funcionando con {len(models)} modelos:")
            for model in models:
                print(f"   - {model}")
            return len(models) > 0
        else:
            print(f"ERROR - Ollama responde con error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"ERROR - Ollama no disponible: {e}")
        return False

def test_llm_service():
    """Test del LLM Service"""
    print("\n=== TEST LLM SERVICE ===")
    
    try:
        from app.services.llm import llm_service
        
        print("1. Health check...")
        health = llm_service.health_check()
        print(f"   Status: {health['status']}")
        
        if health['ollama']['available']:
            models = health['ollama']['models']
            print(f"   Modelos: {len(models)}")
            return len(models) > 0
        else:
            print("   Ollama no disponible")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Función principal"""
    print("TEST COMPLETO LLM SERVICE")
    print("=" * 40)
    
    # Test 1: Importaciones
    imports_ok = test_basic_imports()
    
    # Test 2: Ollama directo
    ollama_ok = test_ollama_direct()
    
    # Test 3: LLM Service (solo si importaciones OK)
    service_ok = False
    if imports_ok:
        service_ok = test_llm_service()
    
    # Resumen
    print("\n=== RESUMEN ===")
    print(f"Importaciones: {'OK' if imports_ok else 'FALLO'}")
    print(f"Ollama:        {'OK' if ollama_ok else 'FALLO'}")
    print(f"LLM Service:   {'OK' if service_ok else 'FALLO'}")
    
    if imports_ok and ollama_ok and service_ok:
        print("\nEXITO - Sistema completamente funcional!")
        return True
    elif imports_ok and ollama_ok:
        print("\nPARCIAL - Ollama funciona, revisar LLM Service")
        return True
    elif imports_ok:
        print("\nPARCIAL - Importaciones OK, instalar modelo Ollama")
        print("Ejecuta: ollama pull llama3.2:3b")
        return False
    else:
        print("\nFALLO - Problemas de importación")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)