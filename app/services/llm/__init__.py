# app/services/llm/__init__.py
"""
LLM Service Package - Modelos de Lenguaje Local y Cloud
TFM Vicente Caruncho - Sistemas Inteligentes UJI
"""

from app.services.llm.llm_service import LLMService, llm_service
from app.models.llm_models import LLMRequest, LLMResponse, ComparisonResult

__all__ = [
    'LLMService',
    'llm_service', 
    'LLMRequest',
    'LLMResponse', 
    'ComparisonResult'
]

# app/services/llm/llm_service.py
"""
Servicio Principal de LLM - Coordinador entre Ollama y OpenAI
"""

import time
import asyncio
from typing import Dict, Any, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

from app.core.config import get_model_config
from app.core.logger import get_logger
from app.models.llm_models import LLMRequest, LLMResponse, ComparisonResult
from app.models import DocumentChunk

class LLMService:
    """Servicio principal para gestión de LLMs locales y cloud"""
    
    def __init__(self):
        self.logger = get_logger("prototipo_chatbot.llm_service")
        self.config = get_model_config()
        self.providers = {}
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'avg_response_time': 0.0
        }
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Inicializar proveedores disponibles"""
        
        # Cargar Ollama Provider
        try:
            from app.services.llm.ollama_provider import OllamaProvider
            ollama = OllamaProvider()
            if ollama.is_available():
                self.providers['ollama'] = ollama
                self.logger.info("✅ Ollama provider disponible")
            else:
                self.logger.warning("⚠️ Ollama no disponible - verificar instalación")
        except Exception as e:
            self.logger.error(f"❌ Error cargando Ollama: {e}")
        
        # Cargar OpenAI Provider
        try:
            from app.services.llm.openai_provider import OpenAIProvider
            openai = OpenAIProvider()
            if openai.is_available():
                self.providers['openai'] = openai
                self.logger.info("✅ OpenAI provider disponible")
            else:
                self.logger.warning("⚠️ OpenAI no disponible - verificar API key")
        except Exception as e:
            self.logger.error(f"❌ Error cargando OpenAI: {e}")
        
        if not self.providers:
            self.logger.error("❌ No hay proveedores LLM disponibles")
        else:
            self.logger.info(
                f"🚀 LLM Service inicializado con {len(self.providers)} proveedores: {list(self.providers.keys())}"
            )
    
    def is_available(self) -> bool:
        """Verificar si hay al menos un proveedor disponible"""
        return len(self.providers) > 0
    
    def get_available_providers(self) -> List[str]:
        """Obtener lista de proveedores disponibles"""
        return list(self.providers.keys())
    
    def get_available_models(self, provider: str = None) -> Dict[str, List[str]]:
        """Obtener modelos disponibles por proveedor"""
        if provider and provider in self.providers:
            return {provider: self.providers[provider].get_available_models()}
        
        models = {}
        for name, provider_instance in self.providers.items():
            models[name] = provider_instance.get_available_models()
        return models
    
    def generate_response(
        self,
        request: LLMRequest,
        provider: str = 'ollama',
        model: str = None
    ) -> LLMResponse:
        """
        Generar respuesta con un proveedor específico
        
        Args:
            request: Solicitud LLM con query, contexto, parámetros
            provider: 'ollama' o 'openai'
            model: Modelo específico a usar (opcional)
        
        Returns:
            LLMResponse con respuesta generada y métricas
        """
        if not self.is_available():
            raise RuntimeError("No hay proveedores LLM disponibles")
        
        if provider not in self.providers:
            available = list(self.providers.keys())
            raise ValueError(f"Proveedor '{provider}' no disponible. Disponibles: {available}")
        
        start_time = time.time()
        
        try:
            # Generar respuesta
            provider_instance = self.providers[provider]
            response = provider_instance.generate(request, model)
            
            # Calcular métricas
            response_time = time.time() - start_time
            response.response_time = response_time
            response.provider = provider
            response.model = model or provider_instance.get_default_model()
            
            # Actualizar estadísticas globales
            self._update_metrics(response, success=True)
            
            self.logger.info(
                f"✅ Respuesta generada exitosamente",
                provider=provider,
                model=response.model,
                response_time=f"{response_time:.2f}s",
                tokens=response.total_tokens
            )
            
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_metrics(None, success=False)
            
            self.logger.error(
                f"❌ Error generando respuesta",
                provider=provider,
                error=str(e),
                response_time=f"{response_time:.2f}s"
            )
            
            # Retornar respuesta de error
            return LLMResponse(
                response=f"Error: {str(e)}",
                provider=provider,
                model=model or "unknown",
                success=False,
                error=str(e),
                response_time=response_time
            )
    
    def compare_models(
        self,
        request: LLMRequest,
        ollama_model: str = "llama3.2:3b",
        openai_model: str = "gpt-4o-mini"
    ) -> ComparisonResult:
        """
        Comparar respuestas entre modelo local (Ollama) y cloud (OpenAI)
        
        Args:
            request: Solicitud base para ambos modelos
            ollama_model: Modelo Ollama a usar
            openai_model: Modelo OpenAI a usar
        
        Returns:
            ComparisonResult con ambas respuestas y análisis
        """
        if 'ollama' not in self.providers or 'openai' not in self.providers:
            missing = []
            if 'ollama' not in self.providers:
                missing.append('ollama')
            if 'openai' not in self.providers:
                missing.append('openai')
            raise RuntimeError(f"Comparación requiere ambos proveedores. Faltantes: {missing}")
        
        start_time = time.time()
        
        # Ejecutar ambos modelos en paralelo
        def generate_ollama():
            return self.generate_response(request, 'ollama', ollama_model)
        
        def generate_openai():
            return self.generate_response(request, 'openai', openai_model)
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_ollama = executor.submit(generate_ollama)
            future_openai = executor.submit(generate_openai)
            
            # Obtener resultados
            ollama_response = future_ollama.result()
            openai_response = future_openai.result()
        
        total_time = time.time() - start_time
        
        # Crear comparación
        comparison = ComparisonResult(
            query=request.query,
            ollama_response=ollama_response,
            openai_response=openai_response,
            total_comparison_time=total_time
        )
        
        # Calcular análisis comparativo
        comparison.analysis = self._analyze_comparison(ollama_response, openai_response)
        
        self.logger.info(
            f"🔍 Comparación completada",
            ollama_model=ollama_model,
            openai_model=openai_model,
            total_time=f"{total_time:.2f}s",
            ollama_success=ollama_response.success,
            openai_success=openai_response.success
        )
        
        return comparison
    
    def _analyze_comparison(self, ollama_resp: LLMResponse, openai_resp: LLMResponse) -> Dict[str, Any]:
        """Analizar diferencias entre respuestas"""
        analysis = {
            'speed_winner': None,
            'token_efficiency': None,
            'cost_analysis': {},
            'success_rates': {
                'ollama': ollama_resp.success,
                'openai': openai_resp.success
            }
        }
        
        # Análisis de velocidad
        if ollama_resp.success and openai_resp.success:
            if ollama_resp.response_time < openai_resp.response_time:
                analysis['speed_winner'] = 'ollama'
                analysis['speed_difference'] = openai_resp.response_time - ollama_resp.response_time
            else:
                analysis['speed_winner'] = 'openai'
                analysis['speed_difference'] = ollama_resp.response_time - openai_resp.response_time
        
        # Análisis de tokens
        if ollama_resp.total_tokens and openai_resp.total_tokens:
            analysis['token_efficiency'] = {
                'ollama_tokens': ollama_resp.total_tokens,
                'openai_tokens': openai_resp.total_tokens,
                'difference': abs(ollama_resp.total_tokens - openai_resp.total_tokens)
            }
        
        # Análisis de costos
        analysis['cost_analysis'] = {
            'ollama_cost': 0.0,  # Siempre gratis
            'openai_cost': openai_resp.estimated_cost or 0.0,
            'cost_difference': openai_resp.estimated_cost or 0.0
        }
        
        return analysis
    
    def _update_metrics(self, response: Optional[LLMResponse], success: bool):
        """Actualizar métricas del servicio"""
        self.metrics['total_requests'] += 1
        
        if success and response:
            self.metrics['successful_requests'] += 1
            if response.total_tokens:
                self.metrics['total_tokens'] += response.total_tokens
            if response.estimated_cost:
                self.metrics['total_cost'] += response.estimated_cost
            
            # Actualizar tiempo promedio de respuesta
            current_avg = self.metrics['avg_response_time']
            new_time = response.response_time
            total_successful = self.metrics['successful_requests']
            
            if total_successful == 1:
                self.metrics['avg_response_time'] = new_time
            else:
                self.metrics['avg_response_time'] = (
                    (current_avg * (total_successful - 1) + new_time) / total_successful
                )
        else:
            self.metrics['failed_requests'] += 1
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas completas del servicio"""
        provider_stats = {}
        for name, provider in self.providers.items():
            provider_stats[name] = {
                'available': provider.is_available(),
                'models': provider.get_available_models(),
                'default_model': provider.get_default_model()
            }
        
        return {
            'service_available': self.is_available(),
            'providers': provider_stats,
            'metrics': self.metrics.copy(),
            'health': {
                'success_rate': (
                    self.metrics['successful_requests'] / max(self.metrics['total_requests'], 1)
                ) * 100,
                'total_requests': self.metrics['total_requests'],
                'avg_response_time': self.metrics['avg_response_time']
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Health check del servicio"""
        health = {
            'status': 'healthy' if self.is_available() else 'unhealthy',
            'providers': {},
            'timestamp': time.time()
        }
        
        for name, provider in self.providers.items():
            try:
                is_available = provider.is_available()
                health['providers'][name] = {
                    'status': 'up' if is_available else 'down',
                    'models_count': len(provider.get_available_models())
                }
            except Exception as e:
                health['providers'][name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return health

# Instancia global del servicio
llm_service = LLMService()

# app/services/llm/ollama_provider.py
"""
Proveedor Ollama - Modelos Locales
"""

import requests
import json
from typing import Dict, Any, List, Optional

from app.core.config import get_model_config
from app.core.logger import get_logger
from app.models.llm_models import LLMRequest, LLMResponse

class OllamaProvider:
    """Proveedor para modelos Ollama locales"""
    
    def __init__(self):
        self.logger = get_logger("prototipo_chatbot.ollama_provider")
        self.config = get_model_config()
        self.base_url = self.config.get('ollama_base_url', 'http://localhost:11434')
        self.timeout = self.config.get('ollama_timeout', 30)
        
        # Modelos disponibles para el TFM
        self.tfm_models = [
            "llama3.2:3b",
            "mistral:7b", 
            "gemma2:2b",
            "phi3:mini"
        ]
    
    def is_available(self) -> bool:
        """Verificar si Ollama está disponible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.debug(f"Ollama no disponible: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Obtener modelos disponibles en Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                self.logger.debug(f"Modelos Ollama disponibles: {models}")
                return models
            return []
        except Exception as e:
            self.logger.error(f"Error obteniendo modelos Ollama: {e}")
            return []
    
    def get_default_model(self) -> str:
        """Obtener modelo por defecto"""
        available = self.get_available_models()
        for model in self.tfm_models:
            if model in available:
                return model
        return available[0] if available else "llama3.2:3b"
    
    def generate(self, request: LLMRequest, model: str = None) -> LLMResponse:
        """
        Generar respuesta usando Ollama
        
        Args:
            request: Solicitud con query, contexto y parámetros
            model: Modelo específico a usar
        
        Returns:
            LLMResponse con la respuesta generada
        """
        if not self.is_available():
            raise RuntimeError("Ollama no está disponible")
        
        model = model or self.get_default_model()
        
        # Construir prompt con contexto RAG si está disponible
        prompt = self._build_rag_prompt(request.query, request.context)
        
        # Parámetros para Ollama
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens
            }
        }
        
        try:
            self.logger.debug(f"Generando con Ollama modelo {model}")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            return LLMResponse(
                response=data.get('response', ''),
                model=model,
                provider='ollama',
                success=True,
                prompt_tokens=len(prompt.split()),  # Estimación simple
                completion_tokens=len(data.get('response', '').split()),
                total_tokens=len(prompt.split()) + len(data.get('response', '').split()),
                estimated_cost=0.0,  # Ollama es gratuito
                metadata={
                    'eval_count': data.get('eval_count'),
                    'eval_duration': data.get('eval_duration'),
                    'total_duration': data.get('total_duration')
                }
            )
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error de conexión con Ollama: {e}")
            raise RuntimeError(f"Error conectando con Ollama: {e}")
        except Exception as e:
            self.logger.error(f"Error generando respuesta Ollama: {e}")
            raise RuntimeError(f"Error en generación Ollama: {e}")
    
    def _build_rag_prompt(self, query: str, context: Optional[List] = None) -> str:
        """Construir prompt optimizado para RAG"""
        if not context:
            return f"""Responde a la siguiente consulta de manera precisa y útil:

Consulta: {query}

Respuesta:"""
        
        # Formatear contexto de documentos
        context_text = ""
        for i, doc in enumerate(context[:3], 1):  # Máximo 3 documentos
            if hasattr(doc, 'content'):
                content = doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
                source = getattr(doc, 'source', f'Documento {i}')
                context_text += f"\n--- Fuente {i}: {source} ---\n{content}\n"
        
        return f"""Eres un asistente especializado en administración local española. Responde a las consultas basándote únicamente en la información proporcionada.

CONTEXTO DOCUMENTAL:
{context_text}

CONSULTA: {query}

INSTRUCCIONES:
- Responde únicamente basándote en la información del contexto
- Si la información no está en el contexto, indica claramente que no puedes responder
- Cita las fuentes específicas cuando sea apropiado
- Mantén un tono profesional y claro
- Estructura la respuesta de manera organizada

RESPUESTA:"""

# app/services/llm/openai_provider.py
"""
Proveedor OpenAI - Modelos Cloud
"""

import os
import time
from typing import Dict, Any, List, Optional

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

from app.core.config import get_model_config
from app.core.logger import get_logger
from app.models.llm_models import LLMRequest, LLMResponse

class OpenAIProvider:
    """Proveedor para modelos OpenAI"""
    
    def __init__(self):
        self.logger = get_logger("prototipo_chatbot.openai_provider")
        self.config = get_model_config()
        self.client = None
        self.api_key = os.getenv('OPENAI_API_KEY')
        
        # Modelos disponibles para el TFM
        self.tfm_models = [
            "gpt-4o",
            "gpt-4o-mini", 
            "gpt-3.5-turbo"
        ]
        
        # Precios por 1K tokens (actualizar según OpenAI)
        self.pricing = {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002}
        }
        
        if self.api_key and OPENAI_AVAILABLE:
            try:
                self.client = OpenAI(api_key=self.api_key)
                self.logger.info("✅ Cliente OpenAI inicializado")
            except Exception as e:
                self.logger.error(f"❌ Error inicializando OpenAI: {e}")
        else:
            if not OPENAI_AVAILABLE:
                self.logger.warning("⚠️ Librería openai no instalada")
            else:
                self.logger.warning("⚠️ OPENAI_API_KEY no configurada")
    
    def is_available(self) -> bool:
        """Verificar si OpenAI está disponible"""
        if not OPENAI_AVAILABLE or not self.client or not self.api_key:
            return False
        
        try:
            # Test simple de conexión
            self.client.models.list()
            return True
        except Exception as e:
            self.logger.debug(f"OpenAI no disponible: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Obtener modelos disponibles (filtrados para TFM)"""
        if not self.is_available():
            return []
        
        try:
            models_response = self.client.models.list()
            available_models = [model.id for model in models_response.data]
            
            # Filtrar solo modelos TFM disponibles
            tfm_available = [model for model in self.tfm_models if model in available_models]
            self.logger.debug(f"Modelos OpenAI TFM disponibles: {tfm_available}")
            return tfm_available
            
        except Exception as e:
            self.logger.error(f"Error obteniendo modelos OpenAI: {e}")
            return self.tfm_models  # Fallback a lista estática
    
    def get_default_model(self) -> str:
        """Obtener modelo por defecto"""
        available = self.get_available_models()
        return available[0] if available else "gpt-4o-mini"
    
    def generate(self, request: LLMRequest, model: str = None) -> LLLResponse:
        """
        Generar respuesta usando OpenAI
        
        Args:
            request: Solicitud con query, contexto y parámetros
            model: Modelo específico a usar
        
        Returns:
            LLMResponse con la respuesta generada
        """
        if not self.is_available():
            raise RuntimeError("OpenAI no está disponible")
        
        model = model or self.get_default_model()
        
        # Construir mensajes con contexto RAG
        messages = self._build_rag_messages(request.query, request.context)
        
        try:
            self.logger.debug(f"Generando con OpenAI modelo {model}")
            
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens
            )
            generation_time = time.time() - start_time
            
            # Extraer información de la respuesta
            choice = response.choices[0]
            usage = response.usage
            
            # Calcular costo estimado
            estimated_cost = self._calculate_cost(model, usage.prompt_tokens, usage.completion_tokens)
            
            return LLMResponse(
                response=choice.message.content,
                model=model,
                provider='openai',
                success=True,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                estimated_cost=estimated_cost,
                response_time=generation_time,
                metadata={
                    'finish_reason': choice.finish_reason,
                    'model_version': response.model,
                    'usage': usage.model_dump() if hasattr(usage, 'model_dump') else dict(usage)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generando respuesta OpenAI: {e}")
            raise RuntimeError(f"Error en generación OpenAI: {e}")
    
    def _build_rag_messages(self, query: str, context: Optional[List] = None) -> List[Dict[str, str]]:
        """Construir mensajes para Chat Completion con contexto RAG"""
        
        system_message = """Eres un asistente especializado en administración local española. Tu función es ayudar a técnicos municipales y ciudadanos con consultas sobre trámites, procedimientos y normativas.

INSTRUCCIONES:
- Responde únicamente basándote en la información proporcionada en el contexto
- Si la información no está disponible en el contexto, indícalo claramente
- Mantén un tono profesional, claro y útil
- Estructura la respuesta de manera organizada
- Cita las fuentes cuando sea apropiado"""
        
        messages = [{"role": "system", "content": system_message}]
        
        if context:
            # Formatear contexto
            context_text = "CONTEXTO DOCUMENTAL:\n"
            for i, doc in enumerate(context[:3], 1):  # Máximo 3 documentos
                if hasattr(doc, 'content'):
                    content = doc.content[:800] + "..." if len(doc.content) > 800 else doc.content
                    source = getattr(doc, 'source', f'Documento {i}')
                    context_text += f"\n--- Fuente {i}: {source} ---\n{content}\n"
            
            messages.append({"role": "user", "content": f"{context_text}\n\nCONSULTA: {query}"})
        else:
            messages.append({"role": "user", "content": query})
        
        return messages
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calcular costo estimado de la consulta"""
        if model not in self.pricing:
            return 0.0
        
        model_pricing = self.pricing[model]
        
        prompt_cost = (prompt_tokens / 1000) * model_pricing["input"]
        completion_cost = (completion_tokens / 1000) * model_pricing["output"]
        
        return prompt_cost + completion_cost