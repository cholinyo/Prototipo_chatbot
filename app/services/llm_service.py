"""
Servicio principal de modelos de lenguaje (LLM) para Prototipo_chatbot
Maneja tanto modelos locales (Ollama) como remotos (OpenAI)
TFM Vicente Caruncho - Sistemas Inteligentes
"""

import os
import time
import requests
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import json
import concurrent.futures
from abc import ABC, abstractmethod

# Imports locales
from app.core.config import get_model_config, get_openai_api_key
from app.core.logger import get_logger
from app.models import ModelResponse, DocumentChunk

@dataclass
class LLMRequest:
    """Estructura para requests a LLM"""
    query: str
    context: List[DocumentChunk] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stream: bool = False

class LLMProvider(ABC):
    """Interfaz abstracta para proveedores de LLM"""
    
    @abstractmethod
    def is_available(self) -> bool:
        """Verificar si el proveedor está disponible"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Obtener modelos disponibles"""
        pass
    
    @abstractmethod
    def generate_response(self, request: LLMRequest, model_name: str = None) -> ModelResponse:
        """Generar respuesta usando el modelo"""
        pass

class OllamaProvider(LLMProvider):
    """Proveedor para modelos locales via Ollama"""
    
    def __init__(self):
        self.config = get_model_config()
        self.logger = get_logger("ollama_provider")
        self.endpoint = self.config.local_endpoint
        self.timeout = self.config.local_timeout
        
    def is_available(self) -> bool:
        """Verificar si Ollama está disponible"""
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_available_models(self) -> List[str]:
        """Obtener modelos disponibles en Ollama"""
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except Exception as e:
            self.logger.warning("Error obteniendo modelos Ollama", error=str(e))
            return []
    
    def generate_response(self, request: LLMRequest, model_name: str = None) -> ModelResponse:
        """Generar respuesta usando Ollama"""
        start_time = time.time()
        model = model_name or self.config.local_default
        
        try:
            # Construir prompt con contexto RAG si está disponible
            prompt = self._build_prompt_with_context(request.query, request.context)
            
            # Preparar payload para Ollama
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                }
            }
            
            # Añadir parámetros opcionales
            if request.top_p is not None:
                payload["options"]["top_p"] = request.top_p
            if request.top_k is not None:
                payload["options"]["top_k"] = request.top_k
            
            # Realizar request a Ollama
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                content = data.get('response', '')
                
                # Log éxito
                self.logger.info("Respuesta Ollama generada exitosamente",
                               model=model,
                               prompt_length=len(prompt),
                               response_length=len(content),
                               response_time=response_time)
                
                return ModelResponse(
                    model_name=model,
                    model_type='local',
                    content=content,
                    response_time=response_time,
                    rag_sources_used=request.context or [],
                    rag_query=request.query if request.context else None,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    success=True
                )
            else:
                error_msg = f"Ollama error: {response.status_code} - {response.text}"
                self.logger.error("Error en respuesta Ollama",
                                model=model,
                                status_code=response.status_code,
                                error=response.text)
                
                return ModelResponse(
                    model_name=model,
                    model_type='local',
                    content="",
                    response_time=response_time,
                    error=error_msg,
                    success=False
                )
                
        except requests.exceptions.Timeout:
            error_msg = f"Timeout conectando con Ollama (>{self.timeout}s)"
            self.logger.error("Timeout Ollama", model=model, timeout=self.timeout)
            
            return ModelResponse(
                model_name=model,
                model_type='local',
                content="",
                response_time=time.time() - start_time,
                error=error_msg,
                success=False
            )
            
        except Exception as e:
            error_msg = f"Error inesperado con Ollama: {str(e)}"
            self.logger.error("Error inesperado Ollama", model=model, error=str(e))
            
            return ModelResponse(
                model_name=model,
                model_type='local',
                content="",
                response_time=time.time() - start_time,
                error=error_msg,
                success=False
            )
    
    def _build_prompt_with_context(self, query: str, context: List[DocumentChunk] = None) -> str:
        """Construir prompt incluyendo contexto RAG"""
        if not context:
            return query
        
        # Crear prompt con contexto
        context_text = "\n\n".join([
            f"[Fuente: {chunk.metadata.source_path}]\n{chunk.content}"
            for chunk in context
        ])
        
        prompt = f"""Contexto de información relevante:
{context_text}

Pregunta: {query}

Por favor responde basándote en el contexto proporcionado. Si la información del contexto no es suficiente, indícalo claramente."""
        
        return prompt

class OpenAIProvider(LLMProvider):
    """Proveedor para modelos OpenAI"""
    
    def __init__(self):
        self.config = get_model_config()
        self.logger = get_logger("openai_provider")
        self.api_key = get_openai_api_key()
        self.timeout = self.config.openai_timeout
        
    def is_available(self) -> bool:
        """Verificar si OpenAI está disponible"""
        if not self.api_key:
            return False
        
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            # Verificar con un request simple
            models = client.models.list()
            return True
        except Exception:
            return False
    
    def get_available_models(self) -> List[str]:
        """Obtener modelos disponibles de OpenAI"""
        if not self.api_key:
            return []
        
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            models = client.models.list()
            
            # Filtrar solo modelos de chat relevantes
            chat_models = [
                model.id for model in models.data 
                if any(prefix in model.id for prefix in ['gpt-3.5', 'gpt-4'])
            ]
            return sorted(chat_models)
            
        except Exception as e:
            self.logger.warning("Error obteniendo modelos OpenAI", error=str(e))
            return self.config.openai_available  # Fallback a configuración
    
    def generate_response(self, request: LLMRequest, model_name: str = None) -> ModelResponse:
        """Generar respuesta usando OpenAI"""
        if not self.api_key:
            return ModelResponse(
                model_name=model_name or "unknown",
                model_type='openai',
                content="",
                response_time=0,
                error="API key de OpenAI no configurada",
                success=False
            )
        
        start_time = time.time()
        model = model_name or self.config.openai_default
        
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
            # Construir mensajes con contexto RAG
            messages = self._build_messages_with_context(request.query, request.context)
            
            # Preparar parámetros para OpenAI
            params = {
                "model": model,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }
            
            # Añadir parámetros opcionales
            if request.top_p is not None:
                params["top_p"] = request.top_p
            
            # Realizar request a OpenAI
            response = client.chat.completions.create(**params)
            
            response_time = time.time() - start_time
            
            # Extraer información de la respuesta
            content = response.choices[0].message.content
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            # Log éxito
            self.logger.info("Respuesta OpenAI generada exitosamente",
                           model=model,
                           prompt_tokens=prompt_tokens,
                           completion_tokens=completion_tokens,
                           total_tokens=total_tokens,
                           response_time=response_time)
            
            return ModelResponse(
                model_name=model,
                model_type='openai',
                content=content,
                response_time=response_time,
                tokens_used=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                rag_sources_used=request.context or [],
                rag_query=request.query if request.context else None,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                success=True
            )
            
        except ImportError:
            error_msg = "Librería openai no instalada. Instala con: pip install openai"
            self.logger.error("Librería OpenAI faltante")
            
            return ModelResponse(
                model_name=model,
                model_type='openai',
                content="",
                response_time=time.time() - start_time,
                error=error_msg,
                success=False
            )
            
        except Exception as e:
            error_msg = f"Error con OpenAI: {str(e)}"
            self.logger.error("Error OpenAI", model=model, error=str(e))
            
            return ModelResponse(
                model_name=model,
                model_type='openai',
                content="",
                response_time=time.time() - start_time,
                error=error_msg,
                success=False
            )
    
    def _build_messages_with_context(self, query: str, context: List[DocumentChunk] = None) -> List[Dict[str, str]]:
        """Construir mensajes incluyendo contexto RAG"""
        messages = [
            {
                "role": "system",
                "content": "Eres un asistente especializado en administración pública local. Responde de forma precisa y útil basándote en el contexto proporcionado."
            }
        ]
        
        if context:
            # Añadir contexto como mensaje del sistema
            context_text = "\n\n".join([
                f"[Fuente: {chunk.metadata.source_path}]\n{chunk.content}"
                for chunk in context
            ])
            
            messages.append({
                "role": "system",
                "content": f"Contexto relevante:\n{context_text}"
            })
        
        # Añadir pregunta del usuario
        messages.append({
            "role": "user",
            "content": query
        })
        
        return messages

class LLMService:
    """Servicio principal que maneja múltiples proveedores de LLM"""
    
    def __init__(self):
        self.logger = get_logger("llm_service")
        
        # Inicializar proveedores
        self.providers = {
            'ollama': OllamaProvider(),
            'openai': OpenAIProvider()
        }
        
        self.logger.info("LLM Service inicializado",
                        providers=list(self.providers.keys()))
    
    def get_available_providers(self) -> Dict[str, bool]:
        """Verificar qué proveedores están disponibles"""
        availability = {}
        
        for name, provider in self.providers.items():
            try:
                availability[name] = provider.is_available()
                self.logger.debug("Provider verificado",
                                provider=name,
                                available=availability[name])
            except Exception as e:
                availability[name] = False
                self.logger.warning("Error verificando provider",
                                  provider=name,
                                  error=str(e))
        
        return availability
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Obtener modelos disponibles por proveedor"""
        models = {}
        
        for name, provider in self.providers.items():
            try:
                if provider.is_available():
                    models[name] = provider.get_available_models()
                else:
                    models[name] = []
            except Exception as e:
                self.logger.warning("Error obteniendo modelos",
                                  provider=name,
                                  error=str(e))
                models[name] = []
        
        return models
    
    def generate_response(self, query: str, provider: str = 'ollama',
                         model_name: str = None, context: List[DocumentChunk] = None,
                         **generation_params) -> ModelResponse:
        """Generar respuesta usando un proveedor específico"""
        
        if provider not in self.providers:
            return ModelResponse(
                model_name=model_name or 'unknown',
                model_type=provider,
                content="",
                response_time=0,
                error=f"Proveedor desconocido: {provider}",
                success=False
            )
        
        provider_instance = self.providers[provider]
        
        # Verificar disponibilidad
        if not provider_instance.is_available():
            return ModelResponse(
                model_name=model_name or 'unknown',
                model_type=provider,
                content="",
                response_time=0,
                error=f"Proveedor {provider} no disponible",
                success=False
            )
        
        # Crear request
        request = LLMRequest(
            query=query,
            context=context,
            **generation_params
        )
        
        # Generar respuesta
        self.logger.info("Generando respuesta LLM",
                        provider=provider,
                        model=model_name,
                        query_length=len(query),
                        context_chunks=len(context) if context else 0)
        
        return provider_instance.generate_response(request, model_name)
    
    def compare_responses(self, query: str, context: List[DocumentChunk] = None,
                         **generation_params) -> Dict[str, ModelResponse]:
        """Comparar respuestas de múltiples proveedores"""
        
        available_providers = self.get_available_providers()
        active_providers = [name for name, available in available_providers.items() if available]
        
        if not active_providers:
            self.logger.warning("No hay proveedores disponibles para comparación")
            return {}
        
        responses = {}
        
        # Generar respuestas en paralelo
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_providers)) as executor:
            future_to_provider = {
                executor.submit(
                    self.generate_response,
                    query=query,
                    provider=provider,
                    context=context,
                    **generation_params
                ): provider
                for provider in active_providers
            }
            
            for future in concurrent.futures.as_completed(future_to_provider):
                provider = future_to_provider[future]
                try:
                    response = future.result()
                    responses[provider] = response
                except Exception as e:
                    self.logger.error("Error en comparación",
                                    provider=provider,
                                    error=str(e))
                    responses[provider] = ModelResponse(
                        model_name='unknown',
                        model_type=provider,
                        content="",
                        response_time=0,
                        error=f"Error ejecutando {provider}: {str(e)}",
                        success=False
                    )
        
        self.logger.info("Comparación completada",
                        providers=list(responses.keys()),
                        successful=len([r for r in responses.values() if r.success]))
        
        return responses
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del servicio"""
        availability = self.get_available_providers()
        models = self.get_available_models()
        
        return {
            'providers_available': availability,
            'total_providers': len(self.providers),
            'available_providers': len([p for p in availability.values() if p]),
            'models_by_provider': models,
            'total_models': sum(len(model_list) for model_list in models.values())
        }

# Instancia global del servicio LLM
llm_service = LLMService()

# Funciones de conveniencia
def generate_llm_response(query: str, provider: str = 'ollama',
                         model_name: str = None, context: List[DocumentChunk] = None,
                         **kwargs) -> ModelResponse:
    """Función de conveniencia para generar respuesta"""
    return llm_service.generate_response(query, provider, model_name, context, **kwargs)

def compare_llm_responses(query: str, context: List[DocumentChunk] = None,
                         **kwargs) -> Dict[str, ModelResponse]:
    """Función de conveniencia para comparar respuestas"""
    return llm_service.compare_responses(query, context, **kwargs)

def get_available_models() -> Dict[str, List[str]]:
    """Función de conveniencia para obtener modelos"""
    return llm_service.get_available_models()

def get_available_providers() -> Dict[str, bool]:
    """Función de conveniencia para obtener proveedores"""
    return llm_service.get_available_providers()