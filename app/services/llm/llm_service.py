# -*- coding: utf-8 -*-
"""
Servicio LLM basico
TFM Vicente Caruncho
"""

import time
import requests
from typing import Dict, Any, List, Optional

class SimpleLLMService:
    """Servicio LLM simplificado"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
    
    def is_available(self) -> bool:
        """Verificar si hay algun servicio disponible"""
        return self.check_ollama()
    
    def check_ollama(self) -> bool:
        """Verificar Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            return response.status_code == 200
        except:
            return False
    
    def get_ollama_models(self) -> List[str]:
        """Obtener modelos Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except:
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """Health check del servicio"""
        ollama_ok = self.check_ollama()
        models = self.get_ollama_models() if ollama_ok else []
        
        return {
            'status': 'healthy' if ollama_ok else 'unhealthy',
            'ollama': {
                'available': ollama_ok,
                'models': models,
                'model_count': len(models)
            }
        }

# Instancia global
llm_service = SimpleLLMService()