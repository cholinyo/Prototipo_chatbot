"""
Prototipo_chatbot - TFM Vicente Caruncho Ramos
Universitat Jaume I - Sistemas Inteligentes

Sistema RAG para Administraciones Locales
"""

__version__ = "1.0.0"
__author__ = "Vicente Caruncho Ramos"
__university__ = "Universitat Jaume I"
__project__ = "Prototipo de Chatbot RAG para Administraciones Locales"

# Importaciones principales (con manejo de errores)
try:
    from .models.document import DocumentChunk, DocumentMetadata
except ImportError:
    DocumentChunk = None
    DocumentMetadata = None

try:
    from .services.rag.embeddings import embedding_service
except ImportError:
    embedding_service = None

def get_project_info():
    """Informacion del proyecto"""
    return {
        "name": __project__,
        "version": __version__,
        "author": __author__,
        "university": __university__,
        "status": "TFM Development"
    }
