"""
Configuración centralizada y validada del sistema Prototipo_chatbot
TFM Vicente Caruncho - Sistemas Inteligentes
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv
import logging

# Cargar variables de entorno
load_dotenv()

@dataclass
class AppConfig:
    """Configuración general de la aplicación"""
    name: str = "Prototipo_chatbot"
    version: str = "1.0.0"
    description: str = "Chatbot RAG para Administraciones Locales"
    debug: bool = True
    host: str = "127.0.0.1"
    port: int = 5000
    secret_key: str = ""
    
    def __post_init__(self):
        """Validación post-inicialización"""
        if not self.secret_key:
            self.secret_key = os.getenv('SECRET_KEY', f"{self.name}_dev_secret")
        
        # Validar puerto
        if not (1024 <= self.port <= 65535):
            raise ValueError(f"Puerto inválido: {self.port}")

@dataclass
class ModelConfig:
    """Configuración de modelos de IA"""
    embedding_name: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_device: str = "cpu"
    embedding_cache_dir: str = ""
    
    # Modelos locales (Ollama)
    local_default: str = "llama3.2:3b"
    local_available: List[str] = field(default_factory=lambda: [
        "llama3.2:3b", "mistral:7b", "gemma2:2b"
    ])
    local_endpoint: str = "http://localhost:11434"
    local_timeout: int = 60
    
    # Modelos OpenAI
    openai_default: str = "gpt-4o-mini"
    openai_available: List[str] = field(default_factory=lambda: [
        "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"
    ])
    openai_timeout: int = 30
    openai_max_retries: int = 3
    
    def __post_init__(self):
        """Validación post-inicialización"""
        if not self.embedding_cache_dir:
            self.embedding_cache_dir = "data/cache/embeddings"
        
        # Crear directorio de cache si no existe
        Path(self.embedding_cache_dir).mkdir(parents=True, exist_ok=True)

@dataclass
class VectorStoreConfig:
    """Configuración de almacenes vectoriales"""
    default: str = "faiss"
    
    # FAISS
    faiss_enabled: bool = True
    faiss_path: str = "data/vectorstore/faiss"
    faiss_index_type: str = "IndexFlatL2"
    faiss_normalize_vectors: bool = True
    
    # ChromaDB  
    chromadb_enabled: bool = True
    chromadb_path: str = "data/vectorstore/chromadb"
    chromadb_collection: str = "prototipo_documents"
    chromadb_distance_function: str = "cosine"
    
    def __post_init__(self):
        """Validación y creación de directorios"""
        if self.default not in ["faiss", "chromadb"]:
            raise ValueError(f"Vector store por defecto inválido: {self.default}")
        
        # Crear directorios si no existen
        if self.faiss_enabled:
            Path(self.faiss_path).mkdir(parents=True, exist_ok=True)
        if self.chromadb_enabled:
            Path(self.chromadb_path).mkdir(parents=True, exist_ok=True)

@dataclass
class RAGConfig:
    """Configuración del sistema RAG"""
    enabled: bool = True
    k_default: int = 5
    k_max: int = 20
    chunk_size: int = 500
    chunk_overlap: int = 50
    similarity_threshold: float = 0.7
    max_query_length: int = 1000
    
    # Estrategias de recuperación
    retrieval_strategy: str = "similarity"  # similarity, mmr, hybrid
    mmr_diversity_lambda: float = 0.5
    
    # Reranking
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 10
    
    def __post_init__(self):
        """Validación post-inicialización"""
        if not (1 <= self.k_default <= self.k_max):
            raise ValueError(f"k_default debe estar entre 1 y {self.k_max}")
        
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold debe estar entre 0.0 y 1.0")
        
        if self.retrieval_strategy not in ["similarity", "mmr", "hybrid"]:
            raise ValueError(f"Estrategia de recuperación inválida: {self.retrieval_strategy}")

@dataclass  
class IngestionConfig:
    """Configuración de ingesta de datos"""
    batch_size: int = 100
    max_file_size_mb: int = 50
    parallel_processing: bool = True
    max_workers: int = 4
    
    # Formatos soportados
    document_formats: List[str] = field(default_factory=lambda: [
        ".pdf", ".docx", ".txt", ".rtf", ".md"
    ])
    spreadsheet_formats: List[str] = field(default_factory=lambda: [
        ".xlsx", ".xls", ".csv"
    ])
    web_formats: List[str] = field(default_factory=lambda: [
        ".html", ".htm"
    ])
    
    # Configuración de OCR
    ocr_enabled: bool = False
    ocr_language: str = "spa"
    
    def __post_init__(self):
        """Validación post-inicialización"""
        if self.max_workers < 1:
            self.max_workers = 1
        
        if self.max_file_size_mb < 1:
            raise ValueError("Tamaño máximo de archivo debe ser al menos 1MB")

@dataclass
class SecurityConfig:
    """Configuración de seguridad"""
    rate_limit_per_minute: int = 60
    max_session_duration_hours: int = 24
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Logs de seguridad
    log_failed_requests: bool = True
    log_successful_requests: bool = True
    
    # Validación de entrada
    sanitize_inputs: bool = True
    max_query_length: int = 1000
    
    def __post_init__(self):
        """Validación post-inicialización"""
        if self.rate_limit_per_minute < 1:
            raise ValueError("Rate limit debe ser al menos 1 por minuto")

class ConfigManager:
    """Gestor centralizado de configuración con validación"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = Path(config_path)
        self._config_data = {}
        self._configs = {}
        self._load_config()
        self._initialize_configs()
        
    def _load_config(self) -> None:
        """Cargar configuración desde archivo YAML"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config_data = yaml.safe_load(f) or {}
            else:
                # Crear configuración por defecto
                self._config_data = self._get_default_config()
                self._save_default_config()
                
        except yaml.YAMLError as e:
            raise ValueError(f"Error parseando configuración YAML: {e}")
        except Exception as e:
            raise RuntimeError(f"Error cargando configuración: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Obtener configuración por defecto"""
        return {
            "app": {
                "name": "Prototipo_chatbot",
                "version": "1.0.0",
                "description": "Chatbot RAG para Administraciones Locales - TFM",
                "debug": os.getenv('FLASK_DEBUG', 'true').lower() == 'true',
                "host": os.getenv('FLASK_HOST', '127.0.0.1'),
                "port": int(os.getenv('FLASK_PORT', '5000'))
            },
            "models": {
                "embedding": {
                    "name": os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
                    "dimension": 384,
                    "device": "cpu",
                    "cache_dir": "data/cache/embeddings"
                },
                "local": {
                    "default": "llama3.2:3b",
                    "available": ["llama3.2:3b", "mistral:7b", "gemma2:2b"],
                    "endpoint": "http://localhost:11434",
                    "timeout": 60
                },
                "openai": {
                    "default": "gpt-4o-mini",
                    "available": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
                    "timeout": 30,
                    "max_retries": 3
                }
            },
            "vector_stores": {
                "default": "faiss",
                "faiss": {
                    "enabled": True,
                    "path": "data/vectorstore/faiss",
                    "index_type": "IndexFlatL2",
                    "normalize_vectors": True
                },
                "chromadb": {
                    "enabled": True,
                    "path": "data/vectorstore/chromadb",
                    "collection": "prototipo_documents",
                    "distance_function": "cosine"
                }
            },
            "rag": {
                "enabled": True,
                "k_default": 5,
                "k_max": 20,
                "chunk_size": 500,
                "chunk_overlap": 50,
                "similarity_threshold": 0.7,
                "max_query_length": 1000,
                "retrieval_strategy": "similarity",
                "mmr_diversity_lambda": 0.5,
                "rerank_enabled": False,
                "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "rerank_top_k": 10
            },
            "ingestion": {
                "batch_size": 100,
                "max_file_size_mb": 50,
                "parallel_processing": True,
                "max_workers": 4,
                "document_formats": [".pdf", ".docx", ".txt", ".rtf", ".md"],
                "spreadsheet_formats": [".xlsx", ".xls", ".csv"],
                "web_formats": [".html", ".htm"],
                "ocr_enabled": False,
                "ocr_language": "spa"
            },
            "security": {
                "rate_limit_per_minute": 60,
                "max_session_duration_hours": 24,
                "allowed_origins": ["*"],
                "log_failed_requests": True,
                "log_successful_requests": True,
                "sanitize_inputs": True,
                "max_query_length": 1000
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/prototipo_chatbot.log",
                "max_bytes": 10485760,
                "backup_count": 5
            }
        }
    
    def _save_default_config(self) -> None:
        """Guardar configuración por defecto"""
        # Crear directorio config si no existe
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config_data, f, default_flow_style=False, allow_unicode=True)
    
    def _initialize_configs(self) -> None:
        """Inicializar objetos de configuración"""
        try:
            # App config
            app_data = self._config_data.get('app', {})
            self._configs['app'] = AppConfig(
                name=app_data.get('name', 'Prototipo_chatbot'),
                version=app_data.get('version', '1.0.0'),
                description=app_data.get('description', 'Chatbot RAG para Administraciones Locales'),
                debug=app_data.get('debug', True),
                host=app_data.get('host', '127.0.0.1'),
                port=app_data.get('port', 5000)
            )
            
            # Model config
            models_data = self._config_data.get('models', {})
            embedding_data = models_data.get('embedding', {})
            local_data = models_data.get('local', {})
            openai_data = models_data.get('openai', {})
            
            self._configs['model'] = ModelConfig(
                embedding_name=embedding_data.get('name', 'all-MiniLM-L6-v2'),
                embedding_dimension=embedding_data.get('dimension', 384),
                embedding_device=embedding_data.get('device', 'cpu'),
                embedding_cache_dir=embedding_data.get('cache_dir', 'data/cache/embeddings'),
                local_default=local_data.get('default', 'llama3.2:3b'),
                local_available=local_data.get('available', ['llama3.2:3b', 'mistral:7b', 'gemma2:2b']),
                local_endpoint=local_data.get('endpoint', 'http://localhost:11434'),
                local_timeout=local_data.get('timeout', 60),
                openai_default=openai_data.get('default', 'gpt-4o-mini'),
                openai_available=openai_data.get('available', ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo']),
                openai_timeout=openai_data.get('timeout', 30),
                openai_max_retries=openai_data.get('max_retries', 3)
            )
            
            # Vector store config
            vs_data = self._config_data.get('vector_stores', {})
            faiss_data = vs_data.get('faiss', {})
            chromadb_data = vs_data.get('chromadb', {})
            
            self._configs['vector_store'] = VectorStoreConfig(
                default=vs_data.get('default', 'faiss'),
                faiss_enabled=faiss_data.get('enabled', True),
                faiss_path=faiss_data.get('path', 'data/vectorstore/faiss'),
                faiss_index_type=faiss_data.get('index_type', 'IndexFlatL2'),
                faiss_normalize_vectors=faiss_data.get('normalize_vectors', True),
                chromadb_enabled=chromadb_data.get('enabled', True),
                chromadb_path=chromadb_data.get('path', 'data/vectorstore/chromadb'),
                chromadb_collection=chromadb_data.get('collection', 'prototipo_documents'),
                chromadb_distance_function=chromadb_data.get('distance_function', 'cosine')
            )
            
            # RAG config
            rag_data = self._config_data.get('rag', {})
            self._configs['rag'] = RAGConfig(
                enabled=rag_data.get('enabled', True),
                k_default=rag_data.get('k_default', 5),
                k_max=rag_data.get('k_max', 20),
                chunk_size=rag_data.get('chunk_size', 500),
                chunk_overlap=rag_data.get('chunk_overlap', 50),
                similarity_threshold=rag_data.get('similarity_threshold', 0.7),
                max_query_length=rag_data.get('max_query_length', 1000),
                retrieval_strategy=rag_data.get('retrieval_strategy', 'similarity'),
                mmr_diversity_lambda=rag_data.get('mmr_diversity_lambda', 0.5),
                rerank_enabled=rag_data.get('rerank_enabled', False),
                rerank_model=rag_data.get('rerank_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
                rerank_top_k=rag_data.get('rerank_top_k', 10)
            )
            
            # Ingestion config
            ing_data = self._config_data.get('ingestion', {})
            self._configs['ingestion'] = IngestionConfig(
                batch_size=ing_data.get('batch_size', 100),
                max_file_size_mb=ing_data.get('max_file_size_mb', 50),
                parallel_processing=ing_data.get('parallel_processing', True),
                max_workers=ing_data.get('max_workers', 4),
                document_formats=ing_data.get('document_formats', ['.pdf', '.docx', '.txt', '.rtf', '.md']),
                spreadsheet_formats=ing_data.get('spreadsheet_formats', ['.xlsx', '.xls', '.csv']),
                web_formats=ing_data.get('web_formats', ['.html', '.htm']),
                ocr_enabled=ing_data.get('ocr_enabled', False),
                ocr_language=ing_data.get('ocr_language', 'spa')
            )
            
            # Security config
            sec_data = self._config_data.get('security', {})
            self._configs['security'] = SecurityConfig(
                rate_limit_per_minute=sec_data.get('rate_limit_per_minute', 60),
                max_session_duration_hours=sec_data.get('max_session_duration_hours', 24),
                allowed_origins=sec_data.get('allowed_origins', ['*']),
                log_failed_requests=sec_data.get('log_failed_requests', True),
                log_successful_requests=sec_data.get('log_successful_requests', True),
                sanitize_inputs=sec_data.get('sanitize_inputs', True),
                max_query_length=sec_data.get('max_query_length', 1000)
            )
            
        except Exception as e:
            raise RuntimeError(f"Error inicializando configuraciones: {e}")
    
    def get_app_config(self) -> AppConfig:
        """Obtener configuración de aplicación"""
        return self._configs['app']
    
    def get_model_config(self) -> ModelConfig:
        """Obtener configuración de modelos"""
        return self._configs['model']
    
    def get_vector_store_config(self) -> VectorStoreConfig:
        """Obtener configuración de vector stores"""
        return self._configs['vector_store']
    
    def get_rag_config(self) -> RAGConfig:
        """Obtener configuración RAG"""
        return self._configs['rag']
    
    def get_ingestion_config(self) -> IngestionConfig:
        """Obtener configuración de ingesta"""
        return self._configs['ingestion']
    
    def get_security_config(self) -> SecurityConfig:
        """Obtener configuración de seguridad"""
        return self._configs['security']
    
    def get_log_config(self) -> Dict[str, Any]:
        """Obtener configuración de logging"""
        return self._config_data.get('logging', {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'logs/prototipo_chatbot.log',
            'max_bytes': 10485760,
            'backup_count': 5
        })
    
    def is_development(self) -> bool:
        """Verificar si estamos en modo desarrollo"""
        return self.get_app_config().debug
    
    def reload_config(self) -> None:
        """Recargar configuración desde archivo"""
        self._load_config()
        self._initialize_configs()
    
    def validate_config(self) -> Dict[str, Any]:
        """Validar toda la configuración"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Validar cada configuración
            for config_name, config_obj in self._configs.items():
                if hasattr(config_obj, '__post_init__'):
                    try:
                        config_obj.__post_init__()
                    except Exception as e:
                        validation_result['valid'] = False
                        validation_result['errors'].append(f"Error en {config_name}: {str(e)}")
            
            # Validaciones adicionales
            model_config = self.get_model_config()
            if not os.getenv('OPENAI_API_KEY'):
                validation_result['warnings'].append("OPENAI_API_KEY no configurada - modelos OpenAI no disponibles")
            
            # Verificar directorios
            for path in [model_config.embedding_cache_dir]:
                if not Path(path).exists():
                    validation_result['warnings'].append(f"Directorio no existe: {path}")
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Error general de validación: {str(e)}")
        
        return validation_result

# Instancia global del gestor de configuración
config_manager = ConfigManager()

# Funciones de conveniencia
def get_app_config() -> AppConfig:
    """Obtener configuración de aplicación"""
    return config_manager.get_app_config()

def get_model_config() -> ModelConfig:
    """Obtener configuración de modelos"""
    return config_manager.get_model_config()

def get_vector_store_config() -> VectorStoreConfig:
    """Obtener configuración de vector stores"""
    return config_manager.get_vector_store_config()

def get_rag_config() -> RAGConfig:
    """Obtener configuración RAG"""
    return config_manager.get_rag_config()

def get_ingestion_config() -> IngestionConfig:
    """Obtener configuración de ingesta"""
    return config_manager.get_ingestion_config()

def get_security_config() -> SecurityConfig:
    """Obtener configuración de seguridad"""
    return config_manager.get_security_config()

def get_log_config() -> Dict[str, Any]:
    """Obtener configuración de logging"""
    return config_manager.get_log_config()

def get_openai_api_key() -> Optional[str]:
    """Obtener API key de OpenAI"""
    return os.getenv('OPENAI_API_KEY')

def is_development() -> bool:
    """Verificar si estamos en modo desarrollo"""
    return config_manager.is_development()

def validate_configuration() -> Dict[str, Any]:
    """Validar toda la configuración"""
    return config_manager.validate_config()

def reload_configuration() -> None:
    """Recargar configuración"""
    config_manager.reload_config()  