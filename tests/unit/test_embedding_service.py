"""
Tests unitarios para EmbeddingService
TFM Vicente Caruncho - Sistemas Inteligentes
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Añadir directorio raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.rag.embeddings import (
    EmbeddingService,
    EmbeddingCache,
    EmbeddingMetrics,
    embedding_service,
    encode_text,
    encode_texts,
    is_embedding_service_available
)

class TestEmbeddingService:
    """Tests para el servicio de embeddings"""
    
    def test_service_initialization(self):
        """Test que el servicio se inicializa correctamente"""
        assert embedding_service is not None
        assert isinstance(embedding_service, EmbeddingService)
        
    def test_service_availability(self):
        """Test disponibilidad del servicio"""
        is_available = is_embedding_service_available()
        assert isinstance(is_available, bool)
        
        if is_available:
            # Si está disponible, verificar que tiene modelo
            assert embedding_service.model is not None
            assert embedding_service.get_dimension() > 0
    
    def test_model_info(self):
        """Test información del modelo"""
        info = embedding_service.get_model_info()
        
        assert isinstance(info, dict)
        assert 'available' in info
        assert 'model_name' in info
        assert 'dimension' in info
        assert 'device' in info
        
        if info['available']:
            assert info['dimension'] > 0
            assert info['model_name'] != 'not_loaded'
    
    @pytest.mark.skipif(
        not is_embedding_service_available(),
        reason="Embedding service no disponible"
    )
    def test_single_text_encoding(self):
        """Test codificación de texto individual"""
        test_text = "Este es un texto de prueba para embeddings"
        
        embedding = encode_text(test_text)
        
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1
        assert embedding.shape[0] == embedding_service.get_dimension()
        
        # Verificar que el embedding está normalizado
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01  # Tolerancia para normalización
    
    @pytest.mark.skipif(
        not is_embedding_service_available(),
        reason="Embedding service no disponible"
    )
    def test_batch_encoding(self):
        """Test codificación en batch"""
        test_texts = [
            "Primera oración de prueba",
            "Segunda oración diferente",
            "Tercera oración para el batch"
        ]
        
        embeddings = encode_texts(test_texts)
        
        assert embeddings is not None
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(test_texts)
        
        for embedding in embeddings:
            assert isinstance(embedding, (list, np.ndarray))
            if isinstance(embedding, np.ndarray):
                assert embedding.shape[0] == embedding_service.get_dimension()
    
    @pytest.mark.skipif(
        not is_embedding_service_available(),
        reason="Embedding service no disponible"
    )
    def test_empty_text_handling(self):
        """Test manejo de texto vacío"""
        empty_text = ""
        
        embedding = encode_text(empty_text)
        
        # Debe devolver un embedding válido incluso para texto vacío
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
    
    @pytest.mark.skipif(
        not is_embedding_service_available(),
        reason="Embedding service no disponible"
    )
    def test_long_text_handling(self):
        """Test manejo de texto largo"""
        long_text = "Este es un texto muy largo. " * 100
        
        embedding = encode_text(long_text)
        
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == embedding_service.get_dimension()
    
    def test_cache_functionality(self):
        """Test funcionalidad del cache"""
        cache = EmbeddingCache(max_size=10)
        
        # Test cache vacío
        result = cache.get("texto inexistente")
        assert result is None
        assert cache.misses == 1
        assert cache.hits == 0
        
        # Test añadir al cache
        test_embedding = np.random.rand(384)
        cache.put("texto de prueba", test_embedding)
        
        # Test recuperar del cache
        cached = cache.get("texto de prueba")
        assert cached is not None
        assert np.array_equal(cached, test_embedding)
        assert cache.hits == 1
        
        # Test estadísticas del cache
        stats = cache.get_stats()
        assert stats['cache_size'] == 1
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
    
    def test_metrics_recording(self):
        """Test registro de métricas"""
        metrics = EmbeddingMetrics()
        
        # Test métricas iniciales
        assert metrics.total_texts_processed == 0
        assert metrics.total_batches_processed == 0
        
        # Test registro de batch
        metrics.record_batch(batch_size=5, processing_time=1.5)
        
        assert metrics.total_texts_processed == 5
        assert metrics.total_batches_processed == 1
        assert metrics.total_processing_time == 1.5
        assert metrics.last_batch_time == 1.5
        
        # Test estadísticas
        stats = metrics.get_stats()
        assert stats['total_texts_processed'] == 5
        assert stats['total_batches_processed'] == 1
        assert stats['avg_time_per_text'] == 0.3
    
    @pytest.mark.skipif(
        not is_embedding_service_available(),
        reason="Embedding service no disponible"
    )
    def test_special_characters_handling(self):
        """Test manejo de caracteres especiales"""
        special_texts = [
            "Texto con ñ y acentos: áéíóú",
            "Text with emojis 😀 🎉",
            "Símbolos especiales: @#$%^&*()",
            "中文字符 and mixed languages"
        ]
        
        for text in special_texts:
            embedding = encode_text(text)
            assert embedding is not None
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape[0] == embedding_service.get_dimension()
    
    def test_service_stats(self):
        """Test estadísticas completas del servicio"""
        stats = embedding_service.get_stats()
        
        assert isinstance(stats, dict)
        assert 'model_info' in stats
        assert 'cache_stats' in stats
        assert 'metrics' in stats
        
        # Verificar estructura de model_info
        assert 'available' in stats['model_info']
        assert 'model_name' in stats['model_info']
        
        # Verificar estructura de cache_stats
        assert 'cache_size' in stats['cache_stats']
        assert 'hit_rate' in stats['cache_stats']
        
        # Verificar estructura de metrics
        assert 'total_texts_processed' in stats['metrics']
        assert 'avg_time_per_text' in stats['metrics']

@pytest.fixture
def sample_texts():
    """Fixture con textos de ejemplo"""
    return [
        "Licencia de obras municipal",
        "Documentación necesaria para permisos",
        "Requisitos administrativos locales"
    ]

@pytest.fixture
def embedding_dimension():
    """Fixture con dimensión esperada de embeddings"""
    return 384  # all-MiniLM-L6-v2 dimension

def test_embedding_consistency(sample_texts):
    """Test que los embeddings son consistentes"""
    if not is_embedding_service_available():
        pytest.skip("Embedding service no disponible")
    
    # Codificar el mismo texto dos veces
    text = sample_texts[0]
    embedding1 = encode_text(text)
    embedding2 = encode_text(text)
    
    # Deben ser idénticos
    assert np.allclose(embedding1, embedding2)

def test_embedding_similarity(sample_texts):
    """Test que textos similares tienen embeddings similares"""
    if not is_embedding_service_available():
        pytest.skip("Embedding service no disponible")
    
    # Textos similares
    text1 = "permiso de construcción"
    text2 = "licencia de obras"
    text3 = "receta de cocina"  # Texto no relacionado
    
    emb1 = encode_text(text1)
    emb2 = encode_text(text2)
    emb3 = encode_text(text3)
    
    # Calcular similitudes coseno
    sim_12 = np.dot(emb1, emb2)  # Entre textos similares
    sim_13 = np.dot(emb1, emb3)  # Entre textos diferentes
    
    # Los textos similares deben tener mayor similitud
    assert sim_12 > sim_13