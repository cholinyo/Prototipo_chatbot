# -*- coding: utf-8 -*-
"""
Modelos de datos para LLM Service
TFM Vicente Caruncho
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

@dataclass
class LLMRequest:
    """Solicitud a un modelo de lenguaje"""
    query: str
    context: Optional[List] = None
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 1000
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validacion"""
        if not self.query.strip():
            raise ValueError("Query no puede estar vacio")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("Temperature debe estar entre 0.0 y 2.0")

@dataclass
class LLMResponse:
    """Respuesta de un modelo de lenguaje"""
    response: str
    model: str
    provider: str
    success: bool = True
    error: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    response_time: Optional[float] = None
    estimated_cost: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    sources: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            'response': self.response,
            'model': self.model,
            'provider': self.provider,
            'success': self.success,
            'error': self.error,
            'total_tokens': self.total_tokens,
            'response_time': self.response_time,
            'estimated_cost': self.estimated_cost,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

@dataclass
class ComparisonResult:
    """Resultado de comparacion entre modelos"""
    query: str
    ollama_response: LLMResponse
    openai_response: LLMResponse
    total_comparison_time: float
    analysis: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            'query': self.query,
            'total_comparison_time': self.total_comparison_time,
            'ollama_response': self.ollama_response.to_dict(),
            'openai_response': self.openai_response.to_dict(),
            'analysis': self.analysis
        }

def create_test_request(query: str) -> LLMRequest:
    """Crear solicitud de prueba"""
    return LLMRequest(
        query=query,
        temperature=0.3,
        max_tokens=500
    )