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
        return ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo']
    
    def generate_local(self, query: str, context: List[Any], model: str = "llama3.2:3b") -> LLMResponse:
        """Generar con modelo local"""
        start_time = time.time()
        
        try:
            # Construir contexto
            if hasattr(context[0], 'content'):
                context_text = "\n".join([chunk.content for chunk in context])
            else:
                context_text = "\n".join([str(chunk) for chunk in context])
            
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
            if hasattr(context[0], 'content'):
                context_text = "\n".join([chunk.content for chunk in context])
            else:
                context_text = "\n".join([str(chunk) for chunk in context])
            
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
            
            # Calcular costo
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
