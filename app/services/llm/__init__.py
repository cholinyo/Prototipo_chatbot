"""
LLM Services Package
TFM Vicente Caruncho
"""

from .llm_services import LLMService

# Crear instancia global
llm_service = LLMService()

def get_llm_service():
    """Obtener instancia del servicio LLM"""
    return llm_service

# Exportar para compatibilidad
__all__ = ['LLMService', 'llm_service', 'get_llm_service']
