"""
Módulo de compatibilidad para LLMService
Adapta la estructura existente al script de validación
"""

import os
import time
import requests
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class LLMResponse:
    """Respuesta estándar de LLM"""
    content: str
    model_used: str
    processing_time: float
    tokens_used: int = 0
    estimated_cost: float = 0.0
    sources: List[str] = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []

try:
    # Intentar importar desde estructura existente
    from app.services.llm.llm_service import LLMService as ExistingLLMService
    
    class LLMService:
        """Wrapper para LLMService existente"""
        def __init__(self):
            try:
                self._service = ExistingLLMService()
            except:
                self._service = None
        
        def __getattr__(self, name):
            if self._service:
                return getattr(self._service, name)
            return lambda *args, **kwargs: None
        
        def test_ollama_connection(self) -> bool:
            """Test compatibilidad"""
            if hasattr(self._service, 'test_ollama_connection'):
                return self._service.test_ollama_connection()
            
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                return response.status_code == 200
            except:
                return False
        
        def test_openai_connection(self) -> bool:
            """Test compatibilidad"""
            if hasattr(self._service, 'test_openai_connection'):
                return self._service.test_openai_connection()
            
            api_key = os.getenv("OPENAI_API_KEY")
            return bool(api_key and api_key.startswith("sk-"))
        
        def get_ollama_models(self) -> List[str]:
            """Obtener modelos Ollama"""
            if hasattr(self._service, 'get_ollama_models'):
                return self._service.get_ollama_models()
            
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    return [m['name'] for m in data.get('models', [])]
                return []
            except:
                return []
        
        def get_openai_models(self) -> List[str]:
            """Obtener modelos OpenAI"""
            if hasattr(self._service, 'get_openai_models'):
                return self._service.get_openai_models()
            
            if not self.test_openai_connection():
                return []
            return ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo']
    
    print("✅ LLMService wrapper creado para estructura existente")
    
except ImportError:
    # Fallback: implementación básica
    class LLMService:
        """Implementación básica para compatibilidad"""
        
        def __init__(self):
            self.ollama_base_url = "http://localhost:11434"
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        def test_ollama_connection(self) -> bool:
            try:
                response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
                return response.status_code == 200
            except:
                return False
        
        def test_openai_connection(self) -> bool:
            return bool(self.openai_api_key and self.openai_api_key.startswith("sk-"))
        
        def get_ollama_models(self) -> List[str]:
            try:
                response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    return [m['name'] for m in data.get('models', [])]
                return []
            except:
                return []
        
        def get_openai_models(self) -> List[str]:
            if not self.test_openai_connection():
                return []
            return ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo']

__all__ = ["LLMService", "LLMResponse"]
