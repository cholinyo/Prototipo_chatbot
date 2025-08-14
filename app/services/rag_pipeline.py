"""
RAG Pipeline - Wrapper para LLM Service existente
TFM Vicente Caruncho
"""

import time
from typing import Dict, Any

class RAGPipeline:
    """Pipeline RAG que utiliza el LLM Service existente"""
    
    def __init__(self):
        try:
            from .llm import get_llm_service
            self.llm_service = get_llm_service()
        except ImportError:
            self.llm_service = None
    
    def health_check(self) -> Dict[str, Any]:
        """Health check del pipeline completo"""
        if not self.llm_service:
            return {
                'status': 'error',
                'timestamp': time.time(),
                'error': 'LLM Service no disponible',
                'components': {
                    'llm': 'not_available'
                }
            }
        
        # Usar health check del LLM service
        try:
            return self.llm_service.health_check()
        except Exception as e:
            return {
                'status': 'error',
                'timestamp': time.time(),
                'error': str(e),
                'components': {
                    'llm': f'error: {e}'
                }
            }
    
    def is_available(self) -> bool:
        """Verificar disponibilidad"""
        if not self.llm_service:
            return False
        
        try:
            return self.llm_service.is_available()
        except:
            return False

# Instancia global
rag_pipeline = RAGPipeline()

def get_rag_pipeline() -> RAGPipeline:
    """Obtener instancia del pipeline"""
    return rag_pipeline
