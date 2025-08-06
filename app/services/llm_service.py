"""
Servicio de Modelos de Lenguaje (LLM)
TFM Vicente Caruncho
"""

import time
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.core.config import get_model_config
from app.core.logger import get_logger
from app.models import ModelResponse, ComparisonResult, DocumentChunk

class LLMService:
    """Servicio principal para gestión de LLMs"""
    
    def __init__(self):
        self.logger = get_logger("prototipo_chatbot.llm_service")
        self.config = get_model_config()
        self.providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Inicializar proveedores disponibles"""
        
        # Intentar cargar Ollama
        try:
            from app.services.llm.ollama_provider import ollama_provider
            if ollama_provider.is_available():
                self.providers['ollama'] = ollama_provider
                self.logger.info("Proveedor Ollama disponible")
        except ImportError as e:
            self.logger.warning(f"Ollama no disponible: {e}")
        
        # Intentar cargar OpenAI
        try:
            from app.services.llm.openai_provider import openai_provider
            if openai_provider.is_available():
                self.providers['openai'] = openai_provider
                self.logger.info("Proveedor OpenAI disponible")
        except ImportError as e:
            self.logger.warning(f"OpenAI no disponible: {e}")
        
        self.logger.info(
            "LLM Service inicializado",
            providers=list(self.providers.keys())
        )
    
    def is_available(self) -> bool:
        """Verificar si hay algún proveedor disponible"""
        return len(self.providers) > 0
    
    def generate_response(
        self,
        query: str,
        context: Optional[List[DocumentChunk]] = None,
        provider: str = 'ollama',
        model: Optional[str] = None,
        **kwargs
    ) -> ModelResponse:
        """Generar respuesta usando un proveedor específico"""
        
        if provider not in self.providers:
            available = list(self.providers.keys())
            if available:
                provider = available[0]
                self.logger.warning(
                    f"Proveedor {provider} no disponible, usando {provider}"
                )
            else:
                return ModelResponse(
                    content="No hay proveedores de LLM disponibles",
                    model="none",
                    provider="none",
                    error="No providers available"
                )
        
        try:
            return self.providers[provider].generate(
                prompt=query,
                model=model,
                context=context,
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"Error generando respuesta: {e}")
            return ModelResponse(
                content="Error generando respuesta",
                model=model or "unknown",
                provider=provider,
                error=str(e)
            )
    
    def compare_models(
        self,
        query: str,
        context: Optional[List[DocumentChunk]] = None,
        local_model: Optional[str] = None,
        openai_model: Optional[str] = None,
        **kwargs
    ) -> ComparisonResult:
        """Comparar respuestas de diferentes modelos"""
        
        self.logger.info(
            "Iniciando comparación de modelos",
            local_model=local_model,
            openai_model=openai_model
        )
        
        comparison = ComparisonResult(query=query)
        
        # Generar respuesta local
        if 'ollama' in self.providers:
            try:
                comparison.local_response = self.generate_response(
                    query=query,
                    context=context,
                    provider='ollama',
                    model=local_model,
                    **kwargs
                )
            except Exception as e:
                self.logger.error(f"Error en modelo local: {e}")
                comparison.local_response = ModelResponse(
                    content="",
                    model=local_model or "unknown",
                    provider="ollama",
                    error=str(e)
                )
        
        # Generar respuesta OpenAI
        if 'openai' in self.providers:
            try:
                comparison.openai_response = self.generate_response(
                    query=query,
                    context=context,
                    provider='openai',
                    model=openai_model,
                    **kwargs
                )
            except Exception as e:
                self.logger.error(f"Error en OpenAI: {e}")
                comparison.openai_response = ModelResponse(
                    content="",
                    model=openai_model or "unknown",
                    provider="openai",
                    error=str(e)
                )
        
        return comparison
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del servicio"""
        stats = {
            'service_available': self.is_available(),
            'providers_available': {},
            'providers_total': len(self.providers)
        }
        
        for name, provider in self.providers.items():
            stats['providers_available'][name] = provider.is_available()
        
        return stats

# Instancia global
llm_service = LLMService()

__all__ = ['LLMService', 'llm_service']
