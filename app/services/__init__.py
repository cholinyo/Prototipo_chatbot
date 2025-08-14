"""
Services Package
TFM Vicente Caruncho
"""

# Importar servicios principales
try:
    from .llm import llm_service, get_llm_service
except ImportError:
    llm_service = None
    get_llm_service = lambda: None

__all__ = ['llm_service', 'get_llm_service']
