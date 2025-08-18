#!/usr/bin/env python3
"""
Script de Diagnóstico y Reparación Automática
TFM Vicente Caruncho - Prototipo_chatbot
Detecta y corrige problemas de configuración automáticamente
"""

import sys
import os
import json
import shutil
import subprocess
import time  # ← AÑADIDO
import traceback  # ← AÑADIDO
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Añadir el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def print_header(title: str, char: str = "="):
    """Imprimir cabecera de sección"""
    print("\n" + char * 70)
    print(f"🔧 {title}")
    print(char * 70)

def print_step(step: str, description: str):
    """Imprimir paso del diagnóstico"""
    print(f"\n📋 PASO {step}: {description}")
    print("-" * 50)

def print_result(success: bool, message: str, details: Any = None):
    """Imprimir resultado"""
    icon = "✅" if success else "❌"
    print(f"   {icon} {message}")
    if details:
        if isinstance(details, dict):
            for key, value in details.items():
                print(f"      {key}: {value}")
        else:
            print(f"      {details}")

def check_python_environment():
    """Verificar entorno Python"""
    print_step("1", "VERIFICANDO ENTORNO PYTHON")
    
    # Verificar versión de Python
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print_result(True, f"Python {python_version}")
    
    # Verificar paquetes críticos
    critical_packages = {
        'torch': 'PyTorch para modelos',
        'sentence_transformers': 'Modelos de embeddings',
        'faiss': 'Vector store FAISS',
        'chromadb': 'Vector store ChromaDB',
        'flask': 'Framework web',
        'numpy': 'Computación numérica',
        'pandas': 'Manipulación de datos'
    }
    
    missing_packages = []
    installed_packages = {}
    
    for package, description in critical_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'Unknown')
            installed_packages[package] = version
            print_result(True, f"{package} {version} - {description}")
        except ImportError:
            missing_packages.append(package)
            print_result(False, f"{package} NO INSTALADO - {description}")
    
    return len(missing_packages) == 0, missing_packages, installed_packages

def install_missing_packages(missing_packages: List[str]):
    """Instalar paquetes faltantes"""
    if not missing_packages:
        return True
    
    print_step("2", "INSTALANDO PAQUETES FALTANTES")
    
    # Mapa de paquetes especiales
    package_map = {
        'torch': 'torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu',
        'faiss': 'faiss-cpu',
        'sentence_transformers': 'sentence-transformers',
        'chromadb': 'chromadb'
    }
    
    success = True
    
    for package in missing_packages:
        install_cmd = package_map.get(package, package)
        print(f"   📦 Instalando {package}...")
        
        try:
            result = subprocess.run(
                f"pip install {install_cmd}".split(),
                capture_output=True,
                text=True,
                check=True
            )
            print_result(True, f"{package} instalado correctamente")
        except subprocess.CalledProcessError as e:
            print_result(False, f"Error instalando {package}: {e}")
            success = False
    
    return success

def create_missing_directories():
    """Crear directorios faltantes"""
    print_step("3", "CREANDO ESTRUCTURA DE DIRECTORIOS")
    
    required_dirs = [
        'app/services/rag',
        'app/services/llm',
        'app/services/ingestion',
        'data/vectorstore/faiss',
        'data/vectorstore/chromadb',
        'data/documents',
        'data/cache/embeddings',
        'data/reports',
        'logs',
        'config',
        'scripts'
    ]
    
    created_dirs = []
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(dir_name)
    
    if created_dirs:
        print_result(True, f"Creados {len(created_dirs)} directorios")
        for dir_name in created_dirs:
            print(f"      📁 {dir_name}")
    else:
        print_result(True, "Todos los directorios ya existen")
    
    return True

def create_embedding_service():
    """Crear servicio de embeddings básico"""
    print_step("4", "CREANDO SERVICIO DE EMBEDDINGS")
    
    embeddings_file = project_root / "app" / "services" / "rag" / "embeddings.py"
    
    if embeddings_file.exists():
        print_result(True, "Servicio de embeddings ya existe")
        return True
    
    embeddings_code = '''"""
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

# Instancia global del servicio
embedding_service = EmbeddingService()

def get_embedding_service() -> EmbeddingService:
    """Obtener instancia del servicio de embeddings"""
    return embedding_service
'''
    
    try:
        # Crear archivo __init__.py en el directorio si no existe
        init_file = embeddings_file.parent / "__init__.py"
        if not init_file.exists():
            init_file.write_text("")
        
        embeddings_file.write_text(embeddings_code, encoding='utf-8')
        print_result(True, f"Servicio de embeddings creado: {embeddings_file}")
        return True
    except Exception as e:
        print_result(False, f"Error creando servicio de embeddings: {e}")
        return False

def create_faiss_store():
    """Crear FAISS vector store básico"""
    print_step("5", "CREANDO FAISS VECTOR STORE")
    
    faiss_file = project_root / "app" / "services" / "rag" / "faiss_store.py"
    
    if faiss_file.exists():
        print_result(True, "FAISS store ya existe")
        return True
    
    faiss_code = '''"""
FAISS Vector Store para Prototipo_chatbot
TFM Vicente Caruncho - Sistemas Inteligentes UJI
"""

import faiss
import numpy as np
import pickle
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger("prototipo_chatbot.faiss_store")

class FaissVectorStore:
    """Vector store usando FAISS para búsqueda eficiente"""
    
    def __init__(self, dimension: int = 384, index_type: str = "flat"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.metadata = []  # Lista de metadatos por vector
        self.id_mapping = {}  # Mapeo ID -> índice en FAISS
        self.counter = 0
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Inicializar índice FAISS"""
        try:
            if self.index_type == "flat":
                self.index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
            
            logger.info(f"Índice FAISS inicializado: {self.index_type}, dimensión: {self.dimension}")
            
        except Exception as e:
            logger.error(f"Error inicializando FAISS: {e}")
            self.index = None
    
    def is_available(self) -> bool:
        """Verificar si el store está disponible"""
        return self.index is not None
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> List[str]:
        """Añadir vectores al índice"""
        if not self.is_available():
            return []
        
        try:
            # Normalizar vectores
            vectors = vectors.astype(np.float32)
            faiss.normalize_L2(vectors)
            
            # Añadir al índice
            start_id = self.counter
            self.index.add(vectors)
            
            # Guardar metadatos y mapeos
            ids = []
            for i, meta in enumerate(metadata):
                doc_id = f"doc_{start_id + i}"
                self.id_mapping[doc_id] = start_id + i
                self.metadata.append(meta)
                ids.append(doc_id)
                self.counter += 1
            
            logger.info(f"Añadidos {len(vectors)} vectores al índice FAISS")
            return ids
            
        except Exception as e:
            logger.error(f"Error añadiendo vectores: {e}")
            return []
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Buscar vectores similares"""
        if not self.is_available():
            return []
        
        try:
            # Normalizar query
            query_vector = query_vector.astype(np.float32).reshape(1, -1)
            faiss.normalize_L2(query_vector)
            
            # Búsqueda
            distances, indices = self.index.search(query_vector, k)
            
            # Construir resultados
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= 0 and idx < len(self.metadata):
                    result = {
                        'id': f"doc_{idx}",
                        'score': 1.0 - distance,  # Convertir distancia a similitud
                        'metadata': self.metadata[idx]
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda FAISS: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del store"""
        if not self.is_available():
            return {'available': False}
        
        return {
            'available': True,
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'memory_usage': self.index.ntotal * self.dimension * 4  # bytes
        }
    
    def save(self, filepath: str):
        """Guardar índice a disco"""
        if not self.is_available():
            return False
        
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Guardar índice FAISS
            faiss.write_index(self.index, str(path) + ".index")
            
            # Guardar metadatos
            with open(str(path) + ".metadata", 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'id_mapping': self.id_mapping,
                    'counter': self.counter,
                    'dimension': self.dimension,
                    'index_type': self.index_type
                }, f)
            
            logger.info(f"Índice FAISS guardado en {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando índice: {e}")
            return False
    
    def load(self, filepath: str):
        """Cargar índice desde disco"""
        try:
            path = Path(filepath)
            
            if not (path.with_suffix('.index').exists() and 
                   path.with_suffix('.metadata').exists()):
                logger.info("Índice FAISS no encontrado, usando nuevo")
                return False
            
            # Cargar índice FAISS
            self.index = faiss.read_index(str(path) + ".index")
            
            # Cargar metadatos
            with open(str(path) + ".metadata", 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.id_mapping = data['id_mapping']
                self.counter = data['counter']
                self.dimension = data['dimension']
                self.index_type = data['index_type']
            
            logger.info(f"Índice FAISS cargado desde {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando índice: {e}")
            return False

# Instancia global
faiss_store = FaissVectorStore()

def get_faiss_store() -> FaissVectorStore:
    """Obtener instancia del FAISS store"""
    return faiss_store
'''
    
    try:
        faiss_file.write_text(faiss_code, encoding='utf-8')
        print_result(True, f"FAISS store creado: {faiss_file}")
        return True
    except Exception as e:
        print_result(False, f"Error creando FAISS store: {e}")
        return False

def create_chromadb_store():
    """Crear ChromaDB vector store básico"""
    print_step("6", "CREANDO CHROMADB VECTOR STORE")
    
    chromadb_file = project_root / "app" / "services" / "rag" / "chromadb_store.py"
    
    if chromadb_file.exists():
        print_result(True, "ChromaDB store ya existe")
        return True
    
    chromadb_code = '''"""
ChromaDB Vector Store para Prototipo_chatbot
TFM Vicente Caruncho - Sistemas Inteligentes UJI
"""

import chromadb
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger("prototipo_chatbot.chromadb_store")

class ChromaDBVectorStore:
    """Vector store usando ChromaDB"""
    
    def __init__(self, collection_name: str = "prototipo_chatbot", persist_directory: str = None):
        self.collection_name = collection_name
        self.persist_directory = persist_directory or str(Path("data/vectorstore/chromadb"))
        self.client = None
        self.collection = None
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Inicializar cliente ChromaDB"""
        try:
            # Crear directorio de persistencia
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            # Inicializar cliente
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Obtener o crear colección
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Documentos del sistema RAG"}
            )
            
            logger.info(f"ChromaDB inicializado: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error inicializando ChromaDB: {e}")
            self.client = None
            self.collection = None
    
    def is_available(self) -> bool:
        """Verificar si el store está disponible"""
        return self.collection is not None
    
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]], 
                     embeddings: Optional[List[List[float]]] = None) -> List[str]:
        """Añadir documentos a ChromaDB"""
        if not self.is_available():
            return []
        
        try:
            # Generar IDs únicos
            ids = [f"doc_{i}_{hash(doc)}" for i, doc in enumerate(documents)]
            
            # Preparar datos
            add_kwargs = {
                'documents': documents,
                'metadatas': metadata,
                'ids': ids
            }
            
            if embeddings:
                add_kwargs['embeddings'] = embeddings
            
            # Añadir a la colección
            self.collection.add(**add_kwargs)
            
            logger.info(f"Añadidos {len(documents)} documentos a ChromaDB")
            return ids
            
        except Exception as e:
            logger.error(f"Error añadiendo documentos a ChromaDB: {e}")
            return []
    
    def search(self, query_text: str = None, query_embedding: List[float] = None, 
              n_results: int = 5, where: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Buscar documentos similares"""
        if not self.is_available():
            return []
        
        try:
            # Preparar query
            query_kwargs = {'n_results': n_results}
            
            if query_text:
                query_kwargs['query_texts'] = [query_text]
            elif query_embedding:
                query_kwargs['query_embeddings'] = [query_embedding]
            else:
                return []
            
            if where:
                query_kwargs['where'] = where
            
            # Realizar búsqueda
            results = self.collection.query(**query_kwargs)
            
            # Formatear resultados
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i] if 'documents' in results else None,
                    'metadata': results['metadatas'][0][i] if 'metadatas' in results else {},
                    'score': 1.0 - results['distances'][0][i] if 'distances' in results else 1.0
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error en búsqueda ChromaDB: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del store"""
        if not self.is_available():
            return {'available': False}
        
        try:
            count = self.collection.count()
            return {
                'available': True,
                'total_documents': count,
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {'available': False, 'error': str(e)}
    
    def clear(self):
        """Limpiar toda la colección"""
        if not self.is_available():
            return False
        
        try:
            # ChromaDB no tiene clear directo, recrear colección
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Documentos del sistema RAG"}
            )
            logger.info("Colección ChromaDB limpiada")
            return True
        except Exception as e:
            logger.error(f"Error limpiando ChromaDB: {e}")
            return False

# Instancia global
chromadb_store = ChromaDBVectorStore()

def get_chromadb_store() -> ChromaDBVectorStore:
    """Obtener instancia del ChromaDB store"""
    return chromadb_store
'''
    
    try:
        chromadb_file.write_text(chromadb_code, encoding='utf-8')
        print_result(True, f"ChromaDB store creado: {chromadb_file}")
        return True
    except Exception as e:
        print_result(False, f"Error creando ChromaDB store: {e}")
        return False

def create_llm_service():
    """Crear servicio LLM básico"""
    print_step("7", "CREANDO SERVICIO LLM")
    
    llm_dir = project_root / "app" / "services" / "llm"
    llm_file = llm_dir / "llm_services.py"
    
    if llm_file.exists():
        print_result(True, "Servicio LLM ya existe")
        return True
    
    # Crear directorio si no existe
    llm_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear __init__.py
    (llm_dir / "__init__.py").write_text("")
    
    llm_code = '''"""
Servicio LLM para Prototipo_chatbot
TFM Vicente Caruncho - Sistemas Inteligentes UJI
Integración con Ollama y OpenAI
"""

import os
import time
import json
import requests
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger("prototipo_chatbot.llm_service")

class LLMService:
    """Servicio para gestión de modelos de lenguaje"""
    
    def __init__(self):
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # Estado de servicios
        self.ollama_available = False
        self.openai_available = False
        
        self._check_services()
    
    def _check_services(self):
        """Verificar disponibilidad de servicios"""
        # Verificar Ollama
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            self.ollama_available = response.status_code == 200
            if self.ollama_available:
                logger.info("Ollama disponible")
            else:
                logger.warning("Ollama no responde correctamente")
        except Exception as e:
            logger.warning(f"Ollama no disponible: {e}")
            self.ollama_available = False
        
        # Verificar OpenAI
        self.openai_available = bool(self.openai_api_key)
        if self.openai_available:
            logger.info("OpenAI configurado")
        else:
            logger.warning("OpenAI API key no configurada")
    
    def health_check(self) -> Dict[str, Any]:
        """Health check completo del servicio"""
        self._check_services()
        
        # Obtener modelos disponibles
        ollama_models = self._get_ollama_models() if self.ollama_available else []
        openai_models = self._get_openai_models() if self.openai_available else []
        
        status = 'healthy' if (self.ollama_available or self.openai_available) else 'error'
        
        return {
            'status': status,
            'timestamp': time.time(),
            'services': {
                'ollama': {
                    'status': 'available' if self.ollama_available else 'unavailable',
                    'url': self.ollama_url,
                    'models_count': len(ollama_models)
                },
                'openai': {
                    'status': 'configured' if self.openai_available else 'not_configured',
                    'models_count': len(openai_models)
                }
            },
            'models': {
                'ollama': ollama_models,
                'openai': openai_models
            }
        }
    
    def _get_ollama_models(self) -> List[str]:
        """Obtener modelos disponibles en Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
        except Exception as e:
            logger.debug(f"Error obteniendo modelos Ollama: {e}")
        return []
    
    def _get_openai_models(self) -> List[str]:
        """Obtener modelos disponibles en OpenAI"""
        if not self.openai_available:
            return []
        
        # Lista de modelos OpenAI comunes
        return [
            'gpt-4o',
            'gpt-4o-mini', 
            'gpt-3.5-turbo'
        ]
    
    def generate_response(self, prompt: str, provider: str = 'ollama', 
                         model: str = None, temperature: float = 0.3) -> Dict[str, Any]:
        """Generar respuesta usando el proveedor especificado"""
        start_time = time.time()
        
        try:
            if provider == 'ollama':
                return self._generate_ollama_response(prompt, model, temperature, start_time)
            elif provider == 'openai':
                return self._generate_openai_response(prompt, model, temperature, start_time)
            else:
                return {
                    'success': False,
                    'error': f"Proveedor no soportado: {provider}",
                    'response_time': time.time() - start_time
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    def _generate_ollama_response(self, prompt: str, model: str = None, 
                                 temperature: float = 0.3, start_time: float = None) -> Dict[str, Any]:
        """Generar respuesta con Ollama"""
        if not self.ollama_available:
            return {
                'success': False,
                'error': 'Ollama no disponible',
                'response_time': time.time() - (start_time or time.time())
            }
        
        # Modelo por defecto
        if not model:
            models = self._get_ollama_models()
            if not models:
                return {
                    'success': False,
                    'error': 'No hay modelos Ollama disponibles',
                    'response_time': time.time() - (start_time or time.time())
                }
            model = models[0]  # Usar el primer modelo disponible
        
        try:
            payload = {
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': temperature
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            response_time = time.time() - (start_time or time.time())
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'response': data.get('response', ''),
                    'model': model,
                    'provider': 'ollama',
                    'response_time': response_time,
                    'tokens_used': data.get('eval_count', 0)
                }
            else:
                return {
                    'success': False,
                    'error': f"Error Ollama: {response.status_code}",
                    'response_time': response_time
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': time.time() - (start_time or time.time())
            }
    
    def _generate_openai_response(self, prompt: str, model: str = None, 
                                 temperature: float = 0.3, start_time: float = None) -> Dict[str, Any]:
        """Generar respuesta con OpenAI"""
        if not self.openai_available:
            return {
                'success': False,
                'error': 'OpenAI API key no configurada',
                'response_time': time.time() - (start_time or time.time())
            }
        
        # Modelo por defecto
        if not model:
            model = 'gpt-4o-mini'
        
        try:
            import openai
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=500
            )
            
            response_time = time.time() - (start_time or time.time())
            
            return {
                'success': True,
                'response': response.choices[0].message.content,
                'model': model,
                'provider': 'openai',
                'response_time': response_time,
                'tokens_used': response.usage.total_tokens,
                'estimated_cost': response.usage.total_tokens * 0.00001  # Estimación
            }
            
        except ImportError:
            return {
                'success': False,
                'error': 'Librería openai no instalada',
                'response_time': time.time() - (start_time or time.time())
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': time.time() - (start_time or time.time())
            }

# Instancia global
llm_service = LLMService()

def get_llm_service() -> LLMService:
    """Obtener instancia del servicio LLM"""
    return llm_service
'''
    
    try:
        llm_file.write_text(llm_code, encoding='utf-8')
        print_result(True, f"Servicio LLM creado: {llm_file}")
        return True
    except Exception as e:
        print_result(False, f"Error creando servicio LLM: {e}")
        return False

def create_config_files():
    """Crear archivos de configuración"""
    print_step("8", "CREANDO ARCHIVOS DE CONFIGURACIÓN")
    
    config_dir = project_root / "config"
    config_dir.mkdir(exist_ok=True)
    
    # Crear settings.yaml
    settings_file = config_dir / "settings.yaml"
    if not settings_file.exists():
        settings_content = '''# Configuración del Sistema RAG
# TFM Vicente Caruncho - Prototipo_chatbot

app:
  name: "Prototipo_chatbot"
  version: "1.2.0"
  description: "Sistema RAG para Administraciones Locales"
  host: "localhost"
  port: 5000
  debug: true

rag:
  embedding_model: "all-MiniLM-L6-v2"
  chunk_size: 500
  chunk_overlap: 50
  similarity_threshold: 0.5
  default_k: 5

vector_stores:
  faiss:
    enabled: true
    index_type: "flat"
    persist_path: "data/vectorstore/faiss/index"
  
  chromadb:
    enabled: true
    persist_directory: "data/vectorstore/chromadb"
    collection_name: "prototipo_chatbot"

llm:
  ollama:
    url: "http://localhost:11434"
    default_model: "llama3.2:3b"
    timeout: 30
  
  openai:
    default_model: "gpt-4o-mini"
    max_tokens: 500
    temperature: 0.3

logging:
  level: "INFO"
  file: "logs/prototipo_chatbot.log"
  max_size: "10MB"
  backup_count: 5

security:
  max_query_length: 1000
  rate_limit_per_hour: 100
  allowed_file_types: [".pdf", ".docx", ".txt", ".md"]
'''
        
        settings_file.write_text(settings_content, encoding='utf-8')
        print_result(True, "Archivo settings.yaml creado")
    else:
        print_result(True, "Archivo settings.yaml ya existe")
    
    # Crear .env.example
    env_example = project_root / ".env.example"
    if not env_example.exists():
        env_content = '''# Configuración de Variables de Entorno
# TFM Vicente Caruncho - Prototipo_chatbot

# OpenAI (opcional)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Ollama (local)
OLLAMA_URL=http://localhost:11434

# Flask
SECRET_KEY=your-secret-key-change-in-production
FLASK_ENV=development

# Logging
LOG_LEVEL=INFO

# Vector Stores
FAISS_INDEX_PATH=data/vectorstore/faiss/index
CHROMADB_PATH=data/vectorstore/chromadb

# Embedding Cache
EMBEDDING_CACHE_PATH=data/cache/embeddings
'''
        env_example.write_text(env_content, encoding='utf-8')
        print_result(True, "Archivo .env.example creado")
    else:
        print_result(True, "Archivo .env.example ya existe")
    
    return True

def update_pipeline_imports():
    """Actualizar imports en el pipeline principal"""
    print_step("9", "ACTUALIZANDO PIPELINE PRINCIPAL")
    
    pipeline_file = project_root / "app" / "services" / "rag_pipeline.py"
    
    if not pipeline_file.exists():
        print_result(False, "Pipeline principal no encontrado")
        return False
    
    try:
        # Leer contenido actual
        content = pipeline_file.read_text(encoding='utf-8')
        
        # Verificar si ya tiene los imports correctos
        if "from app.services.rag.embeddings import get_embedding_service" in content:
            print_result(True, "Pipeline ya tiene imports correctos")
            return True
        
        # Añadir imports necesarios al principio
        import_block = '''# Imports corregidos automáticamente
try:
    from app.services.rag.embeddings import get_embedding_service
    from app.services.rag.faiss_store import get_faiss_store
    from app.services.rag.chromadb_store import get_chromadb_store
    from app.services.llm.llm_services import get_llm_service
    SERVICES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Algunos servicios no disponibles: {e}")
    SERVICES_AVAILABLE = False

'''
        
        # Insertar al principio del archivo después de los docstrings
        lines = content.split('\n')
        insert_position = 0
        
        # Buscar el final de los docstrings/comments iniciales
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('"""') and not line.strip().startswith("'''"):
                insert_position = i
                break
        
        # Insertar el bloque de imports
        lines.insert(insert_position, import_block)
        updated_content = '\n'.join(lines)
        
        # Guardar archivo actualizado
        backup_file = pipeline_file.with_suffix('.py.backup')
        pipeline_file.rename(backup_file)
        
        pipeline_file.write_text(updated_content, encoding='utf-8')
        
        print_result(True, f"Pipeline actualizado (backup en {backup_file.name})")
        return True
        
    except Exception as e:
        print_result(False, f"Error actualizando pipeline: {e}")
        return False

def verify_repairs():
    """Verificar que las reparaciones funcionan"""
    print_step("10", "VERIFICANDO REPARACIONES")
    
    success_count = 0
    total_tests = 0
    
    # Test 1: Embedding Service
    total_tests += 1
    try:
        from app.services.rag.embeddings import get_embedding_service
        embedding_service = get_embedding_service()
        if embedding_service.is_available():
            print_result(True, "Embedding Service disponible")
            success_count += 1
        else:
            print_result(False, "Embedding Service no disponible")
    except Exception as e:
        print_result(False, f"Error Embedding Service: {e}")
    
    # Test 2: FAISS Store
    total_tests += 1
    try:
        from app.services.rag.faiss_store import get_faiss_store
        faiss_store = get_faiss_store()
        if faiss_store.is_available():
            print_result(True, "FAISS Store disponible")
            success_count += 1
        else:
            print_result(False, "FAISS Store no disponible")
    except Exception as e:
        print_result(False, f"Error FAISS Store: {e}")
    
    # Test 3: ChromaDB Store
    total_tests += 1
    try:
        from app.services.rag.chromadb_store import get_chromadb_store
        chromadb_store = get_chromadb_store()
        if chromadb_store.is_available():
            print_result(True, "ChromaDB Store disponible")
            success_count += 1
        else:
            print_result(False, "ChromaDB Store no disponible")
    except Exception as e:
        print_result(False, f"Error ChromaDB Store: {e}")
    
    # Test 4: LLM Service
    total_tests += 1
    try:
        from app.services.llm.llm_services import get_llm_service
        llm_service = get_llm_service()
        health = llm_service.health_check()
        if health['status'] in ['healthy', 'partial']:
            print_result(True, f"LLM Service disponible - {health['status']}")
            success_count += 1
        else:
            print_result(False, f"LLM Service limitado - {health['status']}")
    except Exception as e:
        print_result(False, f"Error LLM Service: {e}")
    
    # Resumen
    success_rate = (success_count / total_tests) * 100
    print(f"\n   📊 Resultado: {success_count}/{total_tests} servicios funcionando ({success_rate:.1f}%)")
    
    if success_rate >= 75:
        print_result(True, "Sistema reparado exitosamente")
        return True
    else:
        print_result(False, "Sistema necesita más reparaciones")
        return False

def generate_repair_report():
    """Generar reporte de reparación"""
    print_step("11", "GENERANDO REPORTE DE REPARACIÓN")
    
    reports_dir = project_root / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = reports_dir / f"system_repair_report_{timestamp}.json"
    
    # Verificar estado final
    final_status = {}
    
    try:
        # Test final del pipeline
        sys.path.insert(0, str(project_root))
        from app.services.rag_pipeline import get_rag_pipeline
        
        pipeline = get_rag_pipeline()
        if pipeline:
            health = pipeline.health_check()
            stats = pipeline.get_stats()
            
            final_status = {
                'pipeline_available': True,
                'health': health,
                'stats': stats,
                'repair_successful': health['status'] != 'error'
            }
        else:
            final_status = {
                'pipeline_available': False,
                'repair_successful': False
            }
            
    except Exception as e:
        final_status = {
            'pipeline_available': False,
            'error': str(e),
            'repair_successful': False
        }
    
    # Crear reporte
    report = {
        'timestamp': time.time(),
        'repair_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'actions_performed': [
            'Verificación entorno Python',
            'Instalación paquetes faltantes',
            'Creación estructura directorios',
            'Creación Embedding Service',
            'Creación FAISS Store',
            'Creación ChromaDB Store', 
            'Creación LLM Service',
            'Creación archivos configuración',
            'Actualización pipeline principal',
            'Verificación reparaciones'
        ],
        'final_status': final_status,
        'recommendations': [
            'Instalar Ollama y descargar modelos: ollama pull llama3.2:3b',
            'Configurar OpenAI API key en .env si quieres usar modelos cloud',
            'Ejecutar scripts/test_rag_pipeline.py para testing completo',
            'Revisar logs en logs/ para más detalles'
        ]
    }
    
    # Guardar reporte
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print_result(True, f"Reporte guardado: {report_file}")
    
    # Mostrar resumen
    if final_status.get('repair_successful'):
        print_result(True, "🎉 REPARACIÓN EXITOSA - Sistema funcional")
    else:
        print_result(False, "⚠️ REPARACIÓN PARCIAL - Revisar logs")
    
    return report_file

def main():
    """Función principal del script de diagnóstico y reparación"""
    print_header("DIAGNÓSTICO Y REPARACIÓN AUTOMÁTICA DEL SISTEMA")
    print("👨‍🎓 Vicente Caruncho Ramos - Sistemas Inteligentes UJI")
    print("🔗 https://github.com/cholinyo/Prototipo_chatbot")
    
    print("\n🎯 Este script detectará y corregirá problemas automáticamente")
    
    try:
        # Ejecutar reparaciones paso a paso
        success_steps = 0
        total_steps = 11
        
        # Paso 1: Verificar entorno Python
        python_ok, missing_packages, installed_packages = check_python_environment()
        if python_ok:
            success_steps += 1
        
        # Paso 2: Instalar paquetes faltantes (si los hay)
        if missing_packages:
            if install_missing_packages(missing_packages):
                success_steps += 1
        else:
            print_step("2", "TODOS LOS PAQUETES YA ESTÁN INSTALADOS")
            print_result(True, "No se requieren instalaciones")
            success_steps += 1
        
        # Paso 3: Crear directorios
        if create_missing_directories():
            success_steps += 1
        
        # Paso 4: Crear Embedding Service
        if create_embedding_service():
            success_steps += 1
        
        # Paso 5: Crear FAISS Store
        if create_faiss_store():
            success_steps += 1
        
        # Paso 6: Crear ChromaDB Store
        if create_chromadb_store():
            success_steps += 1
        
        # Paso 7: Crear LLM Service
        if create_llm_service():
            success_steps += 1
        
        # Paso 8: Crear archivos configuración
        if create_config_files():
            success_steps += 1
        
        # Paso 9: Actualizar pipeline
        if update_pipeline_imports():
            success_steps += 1
        
        # Paso 10: Verificar reparaciones
        if verify_repairs():
            success_steps += 1
        
        # Paso 11: Generar reporte
        report_file = generate_repair_report()
        if report_file:
            success_steps += 1
        
        # Resumen final
        print_header("RESUMEN DE REPARACIÓN")
        success_rate = (success_steps / total_steps) * 100
        
        print(f"   📊 Pasos completados: {success_steps}/{total_steps} ({success_rate:.1f}%)")
        print(f"   📄 Reporte detallado: {report_file}")
        
        if success_rate >= 90:
            print_result(True, "🎉 SISTEMA COMPLETAMENTE REPARADO")
            print("\n🚀 Siguiente paso: ejecutar 'python run.py' para iniciar la aplicación")
        elif success_rate >= 70:
            print_result(True, "⚠️ SISTEMA MAYORMENTE REPARADO")
            print("\n🔧 Algunos componentes pueden necesitar configuración manual")
        else:
            print_result(False, "❌ REPARACIÓN INCOMPLETA")
            print("\n💡 Revisa el reporte y logs para más detalles")
        
        return 0 if success_rate >= 70 else 1
        
    except KeyboardInterrupt:
        print("\n⏹️ Reparación cancelada por el usuario")
        return 1
    except Exception as e:
        print(f"\n❌ Error crítico durante la reparación: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        return 1

def print_header(title: str, char: str = "="):
    """Imprimir cabecera de sección"""
    print("\n" + char * 70)
    print(f"🔧 {title}")
    print(char * 70)

def print_step(step: str, description: str):
    """Imprimir paso del diagnóstico"""
    print(f"\n📋 PASO {step}: {description}")
    print("-" * 50)

def print_result(success: bool, message: str, details: Any = None):
    """Imprimir resultado"""
    icon = "✅" if success else "❌"
    print(f"   {icon} {message}")
    if details:
        if isinstance(details, dict):
            for key, value in details.items():
                print(f"      {key}: {value}")
        else:
            print(f"      {details}")

# ... (resto de funciones permanecen iguales)

def fix_imports_issue():
    """Arreglar problemas de imports específicos"""
    print_step("12", "CORRIGIENDO PROBLEMAS DE IMPORTS")
    
    fixes_applied = []
    
    # 1. Verificar config.py tiene las funciones necesarias
    config_file = project_root / "app" / "core" / "config.py"
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'def get_embedding_config(' not in content:
            # Añadir función faltante
            embedding_config_func = '''

def get_embedding_config() -> Dict[str, Any]:
    """Obtener configuración específica de embeddings"""
    model_config = get_model_config()
    return {
        'model_name': model_config.embedding_name,
        'dimension': model_config.embedding_dimension,
        'device': model_config.embedding_device,
        'cache_dir': model_config.embedding_cache_dir,
        'normalize_vectors': True,
        'batch_size': 32
    }
'''
            content += embedding_config_func
            
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            fixes_applied.append("get_embedding_config añadida a config.py")
            
    except Exception as e:
        print_result(False, f"Error corrigiendo config.py: {e}")
    
    # 2. Verificar LLM service exports
    llm_service_file = project_root / "app" / "services" / "llm" / "llm_services.py"
    try:
        with open(llm_service_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'def get_llm_service(' not in content:
            # Añadir función faltante
            get_llm_func = '''

def get_llm_service() -> LLMService:
    """Obtener instancia del servicio LLM"""
    return llm_service
'''
            content += get_llm_func
            
            with open(llm_service_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            fixes_applied.append("get_llm_service añadida a llm_services.py")
            
    except Exception as e:
        print_result(False, f"Error corrigiendo llm_services.py: {e}")
    
    if fixes_applied:
        for fix in fixes_applied:
            print_result(True, fix)
        return True
    else:
        print_result(True, "No se requieren correcciones adicionales")
        return True

def main():
    """Función principal del script de diagnóstico y reparación"""
    print_header("DIAGNÓSTICO Y REPARACIÓN AUTOMÁTICA DEL SISTEMA")
    print("👨‍🎓 Vicente Caruncho Ramos - Sistemas Inteligentes UJI")
    print("🔗 https://github.com/cholinyo/Prototipo_chatbot")
    
    print("\n🎯 Este script detectará y corregirá problemas automáticamente")
    
    try:
        # Ejecutar reparaciones paso a paso
        success_steps = 0
        total_steps = 12  # ← AUMENTADO a 12 pasos
        
        # Pasos 1-10 como antes...
        # (código anterior permanece igual)
        
        # NUEVO PASO 11: Corregir imports específicos
        if fix_imports_issue():
            success_steps += 1
        
        # Paso 12: Generar reporte
        report_file = generate_repair_report()
        if report_file:
            success_steps += 1
        
        # Resumen final
        print_header("RESUMEN DE REPARACIÓN")
        success_rate = (success_steps / total_steps) * 100
        
        print(f"   📊 Pasos completados: {success_steps}/{total_steps} ({success_rate:.1f}%)")
        print(f"   📄 Reporte detallado: {report_file}")
        
        if success_rate >= 90:
            print_result(True, "🎉 SISTEMA COMPLETAMENTE REPARADO")
            print("\n🚀 Siguiente paso: ejecutar 'python run.py' para iniciar la aplicación")
        elif success_rate >= 70:
            print_result(True, "⚠️ SISTEMA MAYORMENTE REPARADO")
            print("\n🔧 Algunos componentes pueden necesitar configuración manual")
        else:
            print_result(False, "❌ REPARACIÓN INCOMPLETA")
            print("\n💡 Revisa el reporte y logs para más detalles")
        
        return 0 if success_rate >= 70 else 1
        
    except KeyboardInterrupt:
        print("\n⏹️ Reparación cancelada por el usuario")
        return 1
    except Exception as e:
        print(f"\n❌ Error crítico durante la reparación: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")  # ← Ahora funciona
        return 1

if __name__ == "__main__":
    sys.exit(main())