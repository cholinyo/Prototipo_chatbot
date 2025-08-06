"""
Servicio principal de modelos de lenguaje (LLM) para Prototipo_chatbot
Maneja tanto modelos locales (Ollama) como remotos (OpenAI)
TFM Vicente Caruncho - Sistemas Inteligentes
"""

import os
import time
import requests
import openai
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
        except Exception as e:
            self.logger.warning("Ollama no disponible", error=str(e))
            return False
    
    def get_available_models(self) -> List[str]:
        """Obtener modelos disponibles en Ollama"""
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                self.logger.info("Modelos Ollama obtenidos", count=len(models), models=models)
                return models
            return []
        except Exception as e:
            self.logger.error("Error obteniendo modelos Ollama", error=str(e))
            return []
    
    def generate_response(self, request: LLMRequest, model_name: str = None) -> ModelResponse:
        """Generar respuesta usando Ollama"""
        start_time = time.time()
        
        # Usar modelo por defecto si no se especifica
        if not model_name:
            model_name = self.config.default_local_model
        
        try:
            # Construir prompt con contexto RAG
            prompt = self._build_rag_prompt(request.query, request.context)
            
            # Configurar parámetros del modelo
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": request.stream,
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
            
            self.logger.info("Enviando request a Ollama", 
                           model=model_name, 
                           prompt_length=len(prompt))
            
            # Realizar request
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"Error HTTP {response.status_code}: {response.text}")
            
            # Procesar respuesta
            result = response.json()
            response_time = time.time() - start_time
            
            # Extraer métricas
            total_tokens = result.get('total_duration', 0)  # En nanosegundos
            prompt_tokens = result.get('prompt_eval_count', 0)
            completion_tokens = result.get('eval_count', 0)
            
            self.logger.info("Respuesta Ollama recibida",
                           model=model_name,
                           response_time=response_time,
                           prompt_tokens=prompt_tokens,
                           completion_tokens=completion_tokens)
            
            return ModelResponse(
                response=result.get('response', ''),
                model_name=f"ollama/{model_name}",
                provider="ollama",
                response_time=response_time,
                total_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                sources=self._extract_sources(request.context) if request.context else []
            )
            
        except Exception as e:
            error_time = time.time() - start_time
            self.logger.error("Error en Ollama generation",
                            model=model_name,
                            error=str(e),
                            response_time=error_time)
            
            return ModelResponse(
                response=f"Error generando respuesta: {str(e)}",
                model_name=f"ollama/{model_name}",
                provider="ollama",
                response_time=error_time,
                error=str(e),
                sources=[]
            )
    
    def _build_rag_prompt(self, query: str, context: List[DocumentChunk] = None) -> str:
        """Construir prompt enriquecido con contexto RAG"""
        base_prompt = """Eres un asistente especializado en administración local española. 
Tu tarea es ayudar a técnicos municipales respondiendo preguntas basándote en la información oficial proporcionada.

INSTRUCCIONES:
- Responde únicamente basándote en el contexto proporcionado
- Si no tienes información suficiente, indica claramente esta limitación
- Mantén un tono profesional y técnico
- Incluye referencias a las fuentes cuando sea relevante
- Si la pregunta no está relacionada con administración local, indícalo educadamente

"""
        
        if context and len(context) > 0:
            context_text = "\n\nCONTEXTO OFICIAL:\n"
            for i, chunk in enumerate(context[:5], 1):  # Limitar a 5 chunks
                source = chunk.metadata.source_path if chunk.metadata else "Fuente desconocida"
                context_text += f"\n[Documento {i}: {source}]\n{chunk.content}\n"
            
            base_prompt += context_text
        
        base_prompt += f"\n\nPREGUNTA: {query}\n\nRESPUESTA:"
        
        return base_prompt
    
    def _extract_sources(self, context: List[DocumentChunk]) -> List[str]:
        """Extraer fuentes del contexto"""
        if not context:
            return []
        
        sources = set()
        for chunk in context:
            if chunk.metadata and chunk.metadata.source_path:
                sources.add(chunk.metadata.source_path)
        
        return list(sources)

class OpenAIProvider(LLMProvider):
    """Proveedor para modelos OpenAI"""
    
    def __init__(self):
        self.config = get_model_config()
        self.logger = get_logger("openai_provider")
        self.api_key = get_openai_api_key()
        
        # Configurar cliente OpenAI
        if self.api_key:
            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            self.client = None
            self.logger.warning("OpenAI API key no configurada")
    
    def is_available(self) -> bool:
        """Verificar si OpenAI está disponible"""
        if not self.client or not self.api_key:
            return False
        
        try:
            # Test simple de conectividad
            response = self.client.models.list()
            return True
        except Exception as e:
            self.logger.warning("OpenAI no disponible", error=str(e))
            return False
    
    def get_available_models(self) -> List[str]:
        """Obtener modelos disponibles en OpenAI"""
        if not self.client:
            return []
        
        try:
            models = self.client.models.list()
            # Filtrar solo modelos de chat relevantes
            chat_models = [
                model.id for model in models.data 
                if any(prefix in model.id for prefix in ['gpt-4', 'gpt-3.5'])
            ]
            
            self.logger.info("Modelos OpenAI obtenidos", count=len(chat_models))
            return sorted(chat_models)
            
        except Exception as e:
            self.logger.error("Error obteniendo modelos OpenAI", error=str(e))
            return []
    
    def generate_response(self, request: LLMRequest, model_name: str = None) -> ModelResponse:
        """Generar respuesta usando OpenAI"""
        start_time = time.time()
        
        if not self.client:
            return ModelResponse(
                response="Error: OpenAI no configurado correctamente",
                model_name="openai/error",
                provider="openai",
                response_time=0,
                error="API key no configurada",
                sources=[]
            )
        
        # Usar modelo por defecto si no se especifica
        if not model_name:
            model_name = self.config.default_openai_model
        
        try:
            # Construir mensajes con contexto RAG
            messages = self._build_chat_messages(request.query, request.context)
            
            self.logger.info("Enviando request a OpenAI",
                           model=model_name,
                           messages_count=len(messages))
            
            # Realizar request
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                stream=request.stream
            )
            
            response_time = time.time() - start_time
            
            # Extraer datos de la respuesta
            response_text = completion.choices[0].message.content
            total_tokens = completion.usage.total_tokens
            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens
            
            # Calcular coste estimado (precios aproximados)
            estimated_cost = self._calculate_cost(model_name, prompt_tokens, completion_tokens)
            
            self.logger.info("Respuesta OpenAI recibida",
                           model=model_name,
                           response_time=response_time,
                           total_tokens=total_tokens,
                           estimated_cost=estimated_cost)
            
            return ModelResponse(
                response=response_text,
                model_name=f"openai/{model_name}",
                provider="openai",
                response_time=response_time,
                total_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                estimated_cost=estimated_cost,
                sources=self._extract_sources(request.context) if request.context else []
            )
            
        except Exception as e:
            error_time = time.time() - start_time
            self.logger.error("Error en OpenAI generation",
                            model=model_name,
                            error=str(e),
                            response_time=error_time)
            
            return ModelResponse(
                response=f"Error generando respuesta: {str(e)}",
                model_name=f"openai/{model_name}",
                provider="openai",
                response_time=error_time,
                error=str(e),
                sources=[]
            )
    
    def _build_chat_messages(self, query: str, context: List[DocumentChunk] = None) -> List[Dict[str, str]]:
        """Construir mensajes de chat con contexto RAG"""
        messages = [
            {
                "role": "system",
                "content": """Eres un asistente especializado en administración local española. 
Tu tarea es ayudar a técnicos municipales respondiendo preguntas basándote en la información oficial proporcionada.

INSTRUCCIONES:
- Responde únicamente basándote en el contexto proporcionado
- Si no tienes información suficiente, indica claramente esta limitación
- Mantén un tono profesional y técnico
- Incluye referencias a las fuentes cuando sea relevante
- Si la pregunta no está relacionada con administración local, indícalo educadamente"""
            }
        ]
        
        if context and len(context) > 0:
            # Añadir contexto como mensaje del sistema
            context_text = "CONTEXTO OFICIAL:\n\n"
            for i, chunk in enumerate(context[:5], 1):  # Limitar a 5 chunks
                source = chunk.metadata.source_path if chunk.metadata else "Fuente desconocida"
                context_text += f"[Documento {i}: {source}]\n{chunk.content}\n\n"
            
            messages.append({
                "role": "system",
                "content": context_text
            })
        
        # Añadir pregunta del usuario
        messages.append({
            "role": "user",
            "content": query
        })
        
        return messages
    
    def _calculate_cost(self, model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calcular coste estimado basado en precios OpenAI"""
        # Precios aproximados por 1K tokens (actualizar según precios actuales)
        pricing = {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        }
        
        # Buscar precio para el modelo
        model_pricing = None
        for key, price in pricing.items():
            if key in model_name:
                model_pricing = price
                break
        
        if not model_pricing:
            return 0.0
        
        input_cost = (prompt_tokens / 1000) * model_pricing["input"]
        output_cost = (completion_tokens / 1000) * model_pricing["output"]
        
        return input_cost + output_cost
    
    def _extract_sources(self, context: List[DocumentChunk]) -> List[str]:
        """Extraer fuentes del contexto"""
        if not context:
            return []
        
        sources = set()
        for chunk in context:
            if chunk.metadata and chunk.metadata.source_path:
                sources.add(chunk.metadata.source_path)
        
        return list(sources)

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
    
    def generate_response(self, 
                         request: LLMRequest, 
                         provider_name: str = "ollama", 
                         model_name: str = None) -> ModelResponse:
        """Generar respuesta usando un proveedor específico"""
        
        if provider_name not in self.providers:
            return ModelResponse(
                response=f"Error: Proveedor '{provider_name}' no disponible",
                model_name=f"{provider_name}/error",
                provider=provider_name,
                response_time=0,
                error=f"Proveedor no encontrado: {provider_name}",
                sources=[]
            )
        
        provider = self.providers[provider_name]
        
        if not provider.is_available():
            return ModelResponse(
                response=f"Error: Proveedor '{provider_name}' no está disponible",
                model_name=f"{provider_name}/unavailable",
                provider=provider_name,
                response_time=0,
                error=f"Proveedor no disponible: {provider_name}",
                sources=[]
            )
        
        return provider.generate_response(request, model_name)
    
    def compare_models(self, 
                      request: LLMRequest,
                      local_model: str = None,
                      openai_model: str = None) -> Dict[str, ModelResponse]:
        """Comparar respuestas entre modelo local y OpenAI"""
        self.logger.info("Iniciando comparación de modelos",
                        local_model=local_model,
                        openai_model=openai_model)
        
        results = {}
        
        # Ejecutar en paralelo para mejor rendimiento
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            
            # Ollama (local)
            if self.providers['ollama'].is_available():
                futures['ollama'] = executor.submit(
                    self.generate_response, request, 'ollama', local_model
                )
            
            # OpenAI
            if self.providers['openai'].is_available():
                futures['openai'] = executor.submit(
                    self.generate_response, request, 'openai', openai_model
                )
            
            # Recoger resultados
            for provider_name, future in futures.items():
                try:
                    results[provider_name] = future.result(timeout=120)
                except Exception as e:
                    self.logger.error("Error en comparación",
                                    provider=provider_name,
                                    error=str(e))
                    results[provider_name] = ModelResponse(
                        response=f"Error en {provider_name}: {str(e)}",
                        model_name=f"{provider_name}/error",
                        provider=provider_name,
                        response_time=0,
                        error=str(e),
                        sources=[]
                    )
        
        self.logger.info("Comparación completada",
                        providers_executed=list(results.keys()))
        
        return results
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del servicio"""
        availability = self.get_available_providers()
        models = self.get_available_models()
        
        return {
            "providers": {
                "available": [name for name, avail in availability.items() if avail],
                "unavailable": [name for name, avail in availability.items() if not avail],
                "total": len(self.providers)
            },
            "models": {
                "total_count": sum(len(model_list) for model_list in models.values()),
                "by_provider": models
            },
            "service_status": "healthy" if any(availability.values()) else "degraded"
        }

# Instancia global del servicio
llm_service = None

def get_llm_service() -> LLMService:
    """Obtener instancia singleton del LLM Service"""
    global llm_service
    if llm_service is None:
        llm_service = LLMService()
    return llm_service

# Testing y debugging
if __name__ == "__main__":
    import asyncio
    
    def test_llm_service():
        """Función de prueba del LLM Service"""
        service = get_llm_service()
        
        print("🧪 Testing LLM Service...")
        
        # Verificar disponibilidad
        availability = service.get_available_providers()
        print(f"📊 Providers disponibles: {availability}")
        
        # Obtener modelos
        models = service.get_available_models()
        print(f"🤖 Modelos disponibles: {models}")
        
        # Estadísticas
        stats = service.get_service_stats()
        print(f"📈 Estadísticas: {json.dumps(stats, indent=2)}")
        
        # Test de generación simple
        request = LLMRequest(
            query="¿Qué es una licencia de obras?",
            temperature=0.7,
            max_tokens=200
        )
        
        if availability.get('ollama', False):
            print("\n🦙 Testing Ollama...")
            result = service.generate_response(request, 'ollama')
            print(f"Response: {result.response[:100]}...")
            print(f"Time: {result.response_time:.2f}s")
        
        if availability.get('openai', False):
            print("\n🤖 Testing OpenAI...")
            result = service.generate_response(request, 'openai')
            print(f"Response: {result.response[:100]}...")
            print(f"Time: {result.response_time:.2f}s")
            print(f"Cost: ${result.estimated_cost:.4f}")
    
    test_llm_service()