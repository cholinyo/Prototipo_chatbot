#!/usr/bin/env python3
"""
Test simple del LLM Service - Sin emojis problemáticos
"""

import sys
from pathlib import Path

# Añadir proyecto al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_llm_service():
    """Test básico del LLM Service"""
    print("TEST SIMPLE LLM SERVICE")
    print("=" * 30)
    
    try:
        # Test de importación
        print("Probando importación...")
        from app.services.llm import llm_service
        print("✅ Importación exitosa")
        
        # Health check
        print("Health check...")
        health = llm_service.health_check()
        print(f"✅ Health check: {health['status']}")
        
        # Verificar modelos
        if health['ollama']['available']:
            models = health['ollama']['models']
            print(f"✅ Modelos Ollama: {len(models)}")
            for model in models:
                print(f"   - {model}")
            
            if models:
                print("¡Sistema LLM listo!")
                return True
            else:
                print("⚠️ No hay modelos instalados")
                print("Ejecuta: ollama pull llama3.2:3b")
                return False
        else:
            print("❌ Ollama no disponible")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_llm_service()
    print(f"\n{'ÉXITO' if success else 'FALLO'}")