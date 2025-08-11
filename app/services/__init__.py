# -*- coding: utf-8 -*-
"""
Services Package
TFM Vicente Caruncho
"""

# Solo importar lo que realmente existe
try:
    from .llm import llm_service
    __all__ = ['llm_service']
except ImportError:
    # Si no existe, no importar nada por ahora
    __all__ = []