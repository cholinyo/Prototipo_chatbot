#!/usr/bin/env python3
"""
Script para corregir el Embedding Service
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def fix_embedding_service():
    """Corregir el servicio de embeddings"""
    
    # Leer el archivo actual
    embeddings_file = project_root / "app" / "services" / "rag" / "embeddings.py"
    
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Código corregido para el servicio de embeddings
    fixed_code = '''"""
Servicio de Embeddings para Prototipo_chatbot
TFM Vicente Caruncho - Sistemas Inteligentes UJI
"""

import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger("prototipo_chatbot.embeddings")

class EmbeddingService:
    """Servicio de embeddings usando sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dimension = 384  # Dimensión por defecto del modelo
        self.cache_size = 10000  # ← AÑADIDO: cache_size
        self._load_model()
    
    def _load_model(self):
        """Cargar modelo de embeddings"""
        try:
            logger.info(f"Cargando modelo de embeddings: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Verificar dimensión real
            test_embedding = self.model.encode("test", show_progress_bar=False)
            self.dimension = len(test_embedding)
            
            logger.info(f"Modelo cargado correctamente - Dimensión: {self.dimension}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo de embeddings: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Verificar si el servicio está disponible"""
        return self.model is not None
    
    def encode_single_text(self, text: str) -> Optional[np.ndarray]:
        """Generar embedding para un texto"""
        if not self.is_available():
            return None
        
        try:
            embedding = self.model.encode(text, show_progress_bar=False)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error generando embedding: {e}")
            return None
    
    def encode_texts(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generar embeddings para múltiples textos"""
        if not self.is_available():
            return None
        
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Error generando embeddings: {e}")
            return None
    
    def warm_up(self):
        """Precalentar el modelo"""
        if self.is_available():
            self.encode_single_text("Warming up the embedding model.")
            logger.info("Modelo de embeddings precalentado")
    
    def get_stats(self) -> dict:
        """Obtener estadísticas del servicio"""
        return {
            'available': self.is_available(),
            'model_name': self.model_name,
            'dimension': self.dimension,
            'cache_size': self.cache_size
        }

# Instancia global del servicio
embedding_service = EmbeddingService()

def get_embedding_service() -> EmbeddingService:
    """Obtener instancia del servicio de embeddings"""
    return embedding_service
'''
    
    # Escribir el código corregido
    with open(embeddings_file, 'w', encoding='utf-8') as f:
        f.write(fixed_code)
    
    print("✅ Embedding Service corregido")

def test_fix():
    """Probar que la corrección funciona"""
    try:
        from app.services.rag.embeddings import get_embedding_service
        embedding_service = get_embedding_service()
        
        if embedding_service.is_available():
            print("✅ Embedding Service funciona correctamente")
            
            # Test de embedding
            test_text = "Esta es una prueba"
            embedding = embedding_service.encode_single_text(test_text)
            
            if embedding is not None:
                print(f"✅ Test de embedding exitoso - Dimensión: {len(embedding)}")
                return True
            else:
                print("❌ Error generando embedding de prueba")
                return False
        else:
            print("❌ Embedding Service no disponible")
            return False
            
    except Exception as e:
        print(f"❌ Error probando Embedding Service: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Corrigiendo Embedding Service...")
    fix_embedding_service()
    
    print("\n🧪 Probando corrección...")
    if test_fix():
        print("\n🎉 ¡Embedding Service completamente funcional!")
    else:
        print("\n⚠️ Aún hay problemas con el Embedding Service")