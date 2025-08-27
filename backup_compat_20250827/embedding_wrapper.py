"""
Wrapper para EmbeddingService - Añade model_name faltante
"""

try:
    from app.services.rag.embeddings import embedding_service as _original_service
    
    class EmbeddingServiceWrapper:
        """Wrapper que añade compatibilidad model_name"""
        
        def __init__(self, original_service):
            self._service = original_service
            # Añadir atributo faltante
            self.model_name = getattr(original_service, 'model_name', 'all-MiniLM-L6-v2')
            self.model = getattr(original_service, 'model', None)
        
        def __getattr__(self, name):
            """Delegar al servicio original"""
            return getattr(self._service, name)
        
        def is_available(self):
            """Verificar disponibilidad"""
            return hasattr(self._service, 'model') and self._service.model is not None
        
        def encode(self, text: str):
            """Encoding compatible"""
            return self._service.encode(text)
        
        def encode_batch(self, texts: list, batch_size: int = 32):
            """Batch encoding compatible"""
            if hasattr(self._service, 'encode_batch'):
                return self._service.encode_batch(texts, batch_size)
            else:
                return [self.encode(text) for text in texts]
    
    # Crear instancia wrapped
    embedding_service_wrapped = EmbeddingServiceWrapper(_original_service)
    
except ImportError:
    embedding_service_wrapped = None

__all__ = ["embedding_service_wrapped"]
