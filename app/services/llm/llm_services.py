"""
Servicio LLM - Compatible con configuración actual
"""

import os
import time
import requests
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass

class LLMRequest:
    """Request para servicios LLM"""
    query: str
    context: List[Any] = None
    temperature: float = 0.7
    max_tokens: int = 1000 
class LLMResponse:
    """Respuesta de modelo LLM"""
    content: str
    model_used: str
    processing_time: float
    tokens_used: int = 0
    estimated_cost: float = 0.0
    sources: List[str] = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []

class LLMService:
    """Servicio de gestión LLM"""
    
    def __init__(self):
        self.ollama_base_url = "http://localhost:11434"
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
    
    def test_ollama_connection(self) -> bool:
        """Test conexión Ollama"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def test_openai_connection(self) -> bool:
        """Test conexión OpenAI"""
        return bool(self.openai_api_key and self.openai_api_key.startswith("sk-"))
    
    def get_ollama_models(self) -> List[str]:
        """Obtener modelos Ollama"""
        try:
            if not self.test_ollama_connection():
                return []
            
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                return [model['name'] for model in models_data.get('models', [])]
            return []
        except:
            return []
    
    def get_openai_models(self) -> List[str]:
        """Modelos OpenAI disponibles"""
        if not self.test_openai_connection():
            return []
        # Lista estática básica; puedes extenderla si necesitas
        return ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo']
    
    def get_available_providers(self) -> Dict[str, bool]:
        """Obtener proveedores disponibles"""
        return {
            'ollama': self.test_ollama_connection(),
            'openai': self.test_openai_connection()
        }
    
    def is_available(self) -> bool:
        """Verificar si hay al menos un proveedor disponible"""
        providers = self.get_available_providers()
        return any(providers.values())
    
    def health_check(self) -> Dict[str, Any]:
        """Health check completo del servicio LLM"""
        health = {
            'status': 'healthy',
            'timestamp': time.time(),
            'services': {},
            'models': {}
        }
        
        # Test Ollama
        ollama_available = self.test_ollama_connection()
        if ollama_available:
            ollama_models = self.get_ollama_models()
            health['services']['ollama'] = {
                'status': 'available',
                'url': self.ollama_base_url,
                'models_count': len(ollama_models)
            }
            health['models']['ollama'] = ollama_models
        else:
            health['services']['ollama'] = {
                'status': 'unavailable',
                'url': self.ollama_base_url,
                'models_count': 0
            }
            health['status'] = 'degraded'
        
        # Test OpenAI
        openai_available = self.test_openai_connection()
        if openai_available:
            openai_models = self.get_openai_models()
            health['services']['openai'] = {
                'status': 'configured',
                'models_count': len(openai_models)
            }
            health['models']['openai'] = openai_models
        else:
            health['services']['openai'] = {
                'status': 'not_configured',
                'models_count': 0
            }
            if not ollama_available:
                health['status'] = 'error'
        
        return health

    # -------------------------
    # AÑADIDO: compatibilidad main.py / dashboard
    # -------------------------
    def get_available_models(self) -> Dict[str, List[str]]:
        """Modelos disponibles por proveedor, en formato esperado por main.py"""
        return {
            'ollama': self.get_ollama_models(),
            'openai': self.get_openai_models()
        }

    def get_service_stats(self) -> Dict[str, Any]:
        """Resumen de estado en el shape que usa main.py/dashboard"""
        health = self.health_check()
        providers_available = {
            'ollama': health['services'].get('ollama', {}).get('status') == 'available',
            'openai': health['services'].get('openai', {}).get('status') in ('configured', 'available')
        }
        return {
            'status': health.get('status', 'error'),
            'timestamp': health.get('timestamp'),
            'providers_available': providers_available,
            'models': health.get('models', {}),
            'services': health.get('services', {})
        }
    
    def generate_local(self, query: str, context: List[Any], model: str = "llama3.2:3b") -> LLMResponse:
        """Generar con modelo local"""
        start_time = time.time()
        try:
            # Construir contexto
            if context and hasattr(context[0], 'content'):
                context_text = "\n".join([chunk.content for chunk in context])
            else:
                context_text = "\n".join([str(chunk) for chunk in (context or [])])
            
            prompt = f"Contexto: {context_text}\n\nPregunta: {query}\n\nResponde basándote en el contexto."
            
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 200
                    }
                },
                timeout=30
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return LLMResponse(
                    content=result.get('response', '').strip(),
                    model_used=model,
                    processing_time=processing_time,
                    estimated_cost=0.0
                )
            else:
                return LLMResponse(
                    content=f"Error HTTP {response.status_code}",
                    model_used=model,
                    processing_time=processing_time
                )
                
        except Exception as e:
            return LLMResponse(
                content=f"Error: {e}",
                model_used=model,
                processing_time=time.time() - start_time
            )
    
    def generate_openai(self, query: str, context: List[Any], model: str = "gpt-3.5-turbo") -> LLMResponse:
        """Generar con OpenAI"""
        if not self.test_openai_connection():
            return LLMResponse(
                content="Error: OpenAI API key no configurada",
                model_used=model,
                processing_time=0.0
            )
        
        start_time = time.time()
        
        try:
            import openai
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            # Construir contexto
            if context and hasattr(context[0], 'content'):
                context_text = "\n".join([chunk.content for chunk in context])
            else:
                context_text = "\n".join([str(chunk) for chunk in (context or [])])
            
            messages = [
                {"role": "system", "content": "Eres un asistente para administraciones locales."},
                {"role": "user", "content": f"Contexto: {context_text}\n\nPregunta: {query}"}
            ]
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=200,
                temperature=0.3
            )
            
            processing_time = time.time() - start_time
            content = response.choices[0].message.content.strip()
            
            # Calcular costo (aproximación simple)
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            cost = (prompt_tokens * 0.000001 + completion_tokens * 0.000002) if "gpt-3.5" in model else (prompt_tokens * 0.00003 + completion_tokens * 0.00006)
            
            return LLMResponse(
                content=content,
                model_used=model,
                processing_time=processing_time,
                tokens_used=prompt_tokens + completion_tokens,
                estimated_cost=cost
            )
            
        except Exception as e:
            return LLMResponse(
                content=f"Error OpenAI: {e}",
                model_used=model,
                processing_time=time.time() - start_time
            )
# Función de conveniencia que falta
def get_llm_service() -> LLMService:
    """Obtener instancia del servicio LLM (función de conveniencia)"""
    return llm_service

# Singleton para uso desde main.py y otros módulos
llm_service = LLMService()
