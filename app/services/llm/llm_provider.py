"""
Proveedor Ollama para LLM Service
Prototipo_chatbot - TFM Vicente Caruncho
"""

import time
import json
import requests
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from app.core.config import get_model_config
from app.core.logger import get_logger
from app.models import ModelResponse, DocumentChunk

@dataclass
class OllamaConfig:
    """Configuración para Ollama"""
    base_url: str = "http://localhost:11434"
    default_model: str = "llama3.1:8b"
    timeout: int = 120
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: int = 2000

class OllamaProvider:
    """Proveedor de modelos Ollama locales"""
    
    def __init__(self):
        self.config = get_model_config()
        self.logger = get_logger("prototipo_chatbot.ollama_provider")
        
        # Configuración específica de Ollama
        self.base_url = getattr(self.config, 'ollama_base_url', 'http://localhost:11434')
        self.default_model = getattr(self.config, 'ollama_default_model', 'llama3.1:8b')
        self.timeout = getattr(self.config, 'ollama_timeout', 120)
        
        self.session = requests.Session()
        self.available = False
        self._check_availability()
    
    def _check_availability(self):
        """Verificar disponibilidad del servicio Ollama"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            
            if response.status_code == 200:
                self.available = True
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                
                self.logger.info(
                    "Modelos Ollama obtenidos",
                    count=len(models),
                    models=model_names
                )
            else:
                self.available = False
                self.logger.warning(
                    "Ollama no disponible",
                    status_code=response.status_code
                )
                
        except requests.exceptions.RequestException as e:
            self.available = False
            self.logger.warning(
                "No se puede conectar a Ollama",
                error=str(e),
                url=self.base_url
            )
    
    def is_available(self) -> bool:
        """Verificar si el proveedor está disponible"""
        return self.available
    
    def list_models(self) -> List[str]:
        """Listar modelos disponibles"""
        if not self.available:
            return []
        
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m.get('name', '') for m in models if m.get('name')]
            
        except Exception as e:
            self.logger.error(f"Error listando modelos: {e}")
        
        return []
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        context: Optional[List[DocumentChunk]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> ModelResponse:
        """Generar respuesta usando Ollama"""
        
        # Limpiar el nombre del modelo si viene con prefijo
        if model and ':' in model:
            if model.startswith('ollama:'):
                model = model.replace('ollama:', '')
        
        model = model or self.default_model
        
        # Construir prompt con contexto si existe
        full_prompt = self._build_prompt(prompt, context)
        
        # Log de request
        self.logger.info(
            "Enviando request a Ollama",
            model=model,
            prompt_length=len(full_prompt)
        )
        
        start_time = time.time()
        
        try:
            # Preparar payload
            payload = {
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            # Hacer request
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Extraer respuesta
                content = data.get('response', '')
                
                # Obtener información de tokens si está disponible
                eval_count = data.get('eval_count', 0)
                
                self.logger.info(
                    "Respuesta Ollama recibida",
                    model=model,
                    response_time=response_time,
                    tokens=eval_count
                )
                
                return ModelResponse(
                    content=content,
                    model=model,
                    provider='ollama',
                    tokens_used=eval_count,
                    response_time=response_time,
                    context_used=context,
                    metadata={
                        'temperature': temperature,
                        'max_tokens': max_tokens,
                        'prompt_tokens': data.get('prompt_eval_count', 0),
                        'total_duration': data.get('total_duration', 0),
                        'load_duration': data.get('load_duration', 0)
                    }
                )
            
            else:
                error_msg = f"Error HTTP {response.status_code}: {response.text}"
                self.logger.error(
                    "Error en Ollama generation",
                    model=model,
                    error=error_msg,
                    response_time=response_time
                )
                
                return ModelResponse(
                    content="",
                    model=model,
                    provider='ollama',
                    response_time=response_time,
                    error=error_msg
                )
                
        except requests.exceptions.Timeout:
            error_msg = f"Timeout después de {self.timeout}s"
            self.logger.error(
                "Timeout en Ollama",
                model=model,
                timeout=self.timeout
            )
            
            return ModelResponse(
                content="",
                model=model,
                provider='ollama',
                response_time=time.time() - start_time,
                error=error_msg
            )
            
        except Exception as e:
            error_msg = f"Error inesperado: {str(e)}"
            self.logger.error(
                "Error en Ollama generation",
                model=model,
                error=str(e)
            )
            
            return ModelResponse(
                content="",
                model=model,
                provider='ollama',
                response_time=time.time() - start_time,
                error=error_msg
            )
    
    def _build_prompt(
        self,
        query: str,
        context: Optional[List[DocumentChunk]] = None
    ) -> str:
        """Construir prompt completo con contexto"""
        
        if not context:
            return query
        
        # Construir contexto desde chunks
        context_parts = []
        for i, chunk in enumerate(context, 1):
            chunk_text = chunk.get_text_for_llm()
            context_parts.append(f"[Documento {i}]\n{chunk_text}")
        
        context_text = "\n\n".join(context_parts)
        
        # Formato del prompt con contexto
        prompt = f"""Contexto relevante:
{context_text}

Pregunta: {query}

Por favor, responde basándote en el contexto proporcionado. Si la información no está en el contexto, indícalo claramente."""
        
        return prompt
    
    def pull_model(self, model_name: str) -> bool:
        """Descargar un modelo si no está disponible"""
        try:
            self.logger.info(f"Descargando modelo {model_name}...")
            
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=None  # Sin timeout para descargas largas
            )
            
            if response.status_code == 200:
                self.logger.info(f"Modelo {model_name} descargado exitosamente")
                return True
            else:
                self.logger.error(
                    f"Error descargando modelo: {response.status_code}",
                    response=response.text
                )
                return False
                
        except Exception as e:
            self.logger.error(f"Error descargando modelo: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Obtener información sobre un modelo"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/show",
                json={"name": model_name},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            
        except Exception as e:
            self.logger.error(f"Error obteniendo info del modelo: {e}")
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del proveedor"""
        return {
            'available': self.available,
            'base_url': self.base_url,
            'default_model': self.default_model,
            'models': self.list_models() if self.available else []
        }

# Instancia global del proveedor
ollama_provider = OllamaProvider()

# Exportar
__all__ = ['OllamaProvider', 'ollama_provider', 'OllamaConfig']