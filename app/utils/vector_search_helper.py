"""
Helper para búsquedas en vector stores
Soluciona problemas de embeddings y conversión de strings
"""

import numpy as np

def safe_vector_search(vector_store, query, k=5):
    """Búsqueda segura que maneja conversión de strings"""
    try:
        # Importar embedding service
        from app.services.rag.embeddings import embedding_service
        
        # Si query es string, convertir a embedding
        if isinstance(query, str):
            query_embedding = embedding_service.encode(query)
        elif isinstance(query, (list, np.ndarray)):
            query_embedding = query
        else:
            raise ValueError(f"Query type no soportado: {type(query)}")
        
        # Asegurar que es numpy array
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        
        # Búsqueda con embedding
        return vector_store.search(query_embedding, k=k)
        
    except Exception as e:
        print(f"Error en búsqueda: {e}")
        return []

def create_documents_with_embeddings(chunks):
    """Crear documentos con embeddings incluidos"""
    try:
        from app.services.rag.embeddings import embedding_service
        
        embedded_chunks = []
        
        for chunk in chunks:
            # Generar embedding si no existe
            if not hasattr(chunk, 'embedding') or chunk.embedding is None:
                chunk.embedding = embedding_service.encode(chunk.content)
            
            embedded_chunks.append(chunk)
        
        return embedded_chunks
        
    except Exception as e:
        print(f"Error generando embeddings: {e}")
        return chunks
