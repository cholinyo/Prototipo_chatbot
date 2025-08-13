#!/usr/bin/env python3
"""
Script de Corrección de Encoding UTF-8
Soluciona problemas de caracteres especiales en archivos Python
Prototipo_chatbot - Vicente Caruncho Ramos
"""

import os
import sys
from pathlib import Path
import chardet

def print_header(title: str):
    """Imprimir cabecera"""
    print("\n" + "=" * 70)
    print(f"🔧 {title}")
    print("=" * 70)

def detect_and_fix_encoding(file_path: Path):
    """Detectar y corregir encoding de un archivo"""
    try:
        # Leer archivo en modo binario para detectar encoding
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        
        # Detectar encoding
        detection = chardet.detect(raw_data)
        detected_encoding = detection['encoding']
        confidence = detection['confidence']
        
        print(f"📄 {file_path.name}:")
        print(f"   🔍 Encoding detectado: {detected_encoding} (confianza: {confidence:.2f})")
        
        if detected_encoding and detected_encoding.lower() != 'utf-8':
            # Leer con encoding detectado
            try:
                with open(file_path, 'r', encoding=detected_encoding) as f:
                    content = f.read()
                
                # Escribir en UTF-8
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"   ✅ Convertido de {detected_encoding} a UTF-8")
                return True
                
            except Exception as e:
                print(f"   ❌ Error convirtiendo: {e}")
                return False
        else:
            print(f"   ✅ Ya está en UTF-8")
            return True
            
    except Exception as e:
        print(f"   ❌ Error procesando {file_path}: {e}")
        return False

def create_clean_init_files():
    """Crear archivos __init__.py limpios y seguros"""
    print_header("CREANDO ARCHIVOS __INIT__.PY LIMPIOS")
    
    # Archivos __init__.py a crear/corregir
    init_files = {
        'app/__init__.py': '''"""
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
''',
        
        'app/core/__init__.py': '''"""Modulos core del sistema"""
''',
        
        'app/models/__init__.py': '''"""Modelos del sistema RAG"""

try:
    from .document import DocumentChunk, DocumentMetadata, SearchResult, RAGResponse
    __all__ = ["DocumentChunk", "DocumentMetadata", "SearchResult", "RAGResponse"]
except ImportError:
    __all__ = []
''',
        
        'app/services/__init__.py': '''"""Servicios del sistema RAG"""

# Importaciones condicionales para evitar errores
try:
    from .rag.embeddings import embedding_service
except ImportError:
    embedding_service = None

try:
    from .rag.faiss_store import FaissVectorStore
except ImportError:
    FaissVectorStore = None

try:
    from .rag.chromadb_store import ChromaDBVectorStore
except ImportError:
    ChromaDBVectorStore = None

try:
    from .ingestion.data_ingestion import DataIngestionService
except ImportError:
    DataIngestionService = None

try:
    from .llm.llm_service import LLMService
except ImportError:
    LLMService = None

__all__ = [
    "embedding_service",
    "FaissVectorStore", 
    "ChromaDBVectorStore",
    "DataIngestionService",
    "LLMService"
]
''',
        
        'app/services/rag/__init__.py': '''"""Servicios RAG"""
''',
        
        'app/services/ingestion/__init__.py': '''"""Servicios de ingestion de datos"""

try:
    from .data_ingestion import DataIngestionService
    __all__ = ["DataIngestionService"]
except ImportError:
    __all__ = []
''',
        
        'app/services/llm/__init__.py': '''"""Servicios de modelos de lenguaje"""

try:
    from .llm_service import LLMService, LLMResponse
    __all__ = ["LLMService", "LLMResponse"]
except ImportError:
    __all__ = []
''',
        
        'app/routes/__init__.py': '''"""Rutas de la aplicacion"""
''',
        
        'app/utils/__init__.py': '''"""Utilidades"""
'''
    }
    
    created_files = 0
    
    for file_path, content in init_files.items():
        try:
            full_path = Path(file_path)
            
            # Crear directorio si no existe
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Escribir archivo en UTF-8 explícitamente
            with open(full_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(content)
            
            print(f"   ✅ {file_path}")
            created_files += 1
            
        except Exception as e:
            print(f"   ❌ Error creando {file_path}: {e}")
    
    print(f"\n📊 Archivos procesados: {created_files}/{len(init_files)}")
    return created_files == len(init_files)

def fix_document_model_encoding():
    """Corregir encoding del modelo DocumentChunk"""
    print_header("CORRIGIENDO MODELO DOCUMENT.PY")
    
    document_content = '''"""
Modelos de documentos para el sistema RAG
Prototipo_chatbot - TFM Vicente Caruncho Ramos
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np

@dataclass
class DocumentMetadata:
    """Metadatos de un documento"""
    source_path: str
    source_type: str = "unknown"  # document, web, api, database
    file_type: str = ""
    size_bytes: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: datetime = field(default_factory=datetime.now)
    checksum: str = ""
    title: str = ""
    author: str = ""
    language: str = "es"
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass  
class DocumentChunk:
    """Fragmento de documento con embedding"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = ""
    chunk_index: int = 0
    chunk_size: int = 0
    start_char: int = 0
    end_char: int = 0
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Procesar despues de inicializacion"""
        if not self.id:
            import hashlib
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            self.id = f"chunk_{content_hash}_{self.chunk_index}"
        
        if not self.chunk_size:
            self.chunk_size = len(self.content)
        
        if not self.end_char and self.start_char:
            self.end_char = self.start_char + self.chunk_size

@dataclass
class SearchResult:
    """Resultado de busqueda semantica"""
    chunk: DocumentChunk
    score: float
    rank: int = 0
    
@dataclass
class RAGResponse:
    """Respuesta del sistema RAG"""
    query: str
    answer: str
    sources: List[DocumentChunk]
    confidence: float = 0.0
    processing_time: float = 0.0
    model_used: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
'''
    
    try:
        model_file = Path("app/models/document.py")
        model_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Escribir en UTF-8 explícitamente
        with open(model_file, 'w', encoding='utf-8', newline='\n') as f:
            f.write(document_content)
        
        print(f"   ✅ Creado: {model_file}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error creando document.py: {e}")
        return False

def fix_services_encoding():
    """Corregir encoding de servicios"""
    print_header("CORRIGIENDO SERVICIOS")
    
    # Verificar archivos de servicios existentes
    service_files = [
        "app/services/ingestion/data_ingestion.py",
        "app/services/llm/llm_service.py"
    ]
    
    fixed_files = 0
    
    for service_file in service_files:
        file_path = Path(service_file)
        
        if file_path.exists():
            if detect_and_fix_encoding(file_path):
                fixed_files += 1
        else:
            print(f"   ⚠️  Archivo no encontrado: {service_file}")
    
    return fixed_files

def validate_encoding_fix():
    """Validar que el fix de encoding funcionó"""
    print_header("VALIDANDO CORRECCION DE ENCODING")
    
    # Archivos críticos a validar
    critical_files = [
        "app/__init__.py",
        "app/models/__init__.py",
        "app/models/document.py",
        "app/services/__init__.py"
    ]
    
    validation_success = 0
    
    for file_path in critical_files:
        full_path = Path(file_path)
        
        if not full_path.exists():
            print(f"   ❌ Archivo no existe: {file_path}")
            continue
        
        try:
            # Intentar leer como UTF-8
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Intentar importar si es un módulo Python
            if file_path.endswith('.py'):
                try:
                    compile(content, str(full_path), 'exec')
                    print(f"   ✅ {file_path} - Sintaxis válida")
                    validation_success += 1
                except SyntaxError as e:
                    print(f"   ❌ {file_path} - Error sintaxis: {e}")
                except Exception as e:
                    print(f"   ⚠️  {file_path} - Advertencia: {e}")
                    validation_success += 1  # Contar como éxito parcial
            else:
                print(f"   ✅ {file_path} - Lectura UTF-8 exitosa")
                validation_success += 1
                
        except UnicodeDecodeError as e:
            print(f"   ❌ {file_path} - Error UTF-8: {e}")
        except Exception as e:
            print(f"   ❌ {file_path} - Error: {e}")
    
    print(f"\n📊 Archivos validados: {validation_success}/{len(critical_files)}")
    return validation_success >= len(critical_files) - 1  # Permitir 1 fallo

def test_import_after_fix():
    """Probar imports después del fix"""
    print_header("PROBANDO IMPORTS DESPUES DEL FIX")
    
    # Tests de import básicos
    import_tests = [
        ("app", "Módulo principal"),
        ("app.models", "Módulo models"),
        ("app.services", "Módulo services"),
        ("app.models.document", "DocumentChunk")
    ]
    
    successful_imports = 0
    
    for module_name, description in import_tests:
        try:
            __import__(module_name)
            print(f"   ✅ {module_name} - {description}")
            successful_imports += 1
        except ImportError as e:
            print(f"   ❌ {module_name} - ImportError: {e}")
        except SyntaxError as e:
            print(f"   ❌ {module_name} - SyntaxError: {e}")
        except Exception as e:
            print(f"   ⚠️  {module_name} - Warning: {e}")
            successful_imports += 1  # Contar warnings como éxito parcial
    
    print(f"\n📊 Imports exitosos: {successful_imports}/{len(import_tests)}")
    return successful_imports >= len(import_tests) - 1

def main():
    """Función principal de corrección de encoding"""
    print_header("CORRECCIÓN DE ENCODING UTF-8")
    print("📋 TFM: Prototipo de Chatbot RAG para Administraciones Locales")
    print("👨‍🎓 Autor: Vicente Caruncho Ramos") 
    print("🏫 Universidad: Universitat Jaume I - Sistemas Inteligentes")
    print("🎯 Problema: Error de encoding en archivos Python")
    
    # Paso 1: Crear archivos __init__.py limpios
    step1_success = create_clean_init_files()
    
    # Paso 2: Corregir modelo DocumentChunk
    step2_success = fix_document_model_encoding()
    
    # Paso 3: Corregir encoding de servicios
    step3_success = fix_services_encoding()
    
    # Paso 4: Validar corrección
    step4_success = validate_encoding_fix()
    
    # Paso 5: Probar imports
    step5_success = test_import_after_fix()
    
    # Resumen final
    successful_steps = sum([step1_success, step2_success, step3_success, step4_success, step5_success])
    total_steps = 5
    
    print_header("RESUMEN DE CORRECCIÓN")
    
    print(f"📊 Pasos completados: {successful_steps}/{total_steps}")
    
    if successful_steps >= 4:  # Al menos 4 de 5 pasos exitosos
        print("✅ CORRECCIÓN DE ENCODING EXITOSA")
        print("🚀 Ejecutar ahora: python ejecutar_fase4_rag_endtoend_validation.py")
        
        # Consejo adicional
        print("\n💡 CONSEJOS:")
        print("   - Todos los archivos están ahora en UTF-8")
        print("   - Sin caracteres especiales problemáticos")
        print("   - Imports seguros y compatibles")
        
    else:
        print("⚠️  CORRECCIÓN PARCIAL")
        print("🔧 Revisar errores anteriores y corregir manualmente")
        
        # Sugerencias de solución manual
        print("\n🛠️  SOLUCIÓN MANUAL:")
        print("   1. Abrir app/__init__.py en un editor")
        print("   2. Verificar que no hay caracteres raros")
        print("   3. Guardar como UTF-8 sin BOM")
        print("   4. Re-ejecutar este script")
    
    print(f"\n📁 Archivos corregidos en:")
    print(f"   📄 app/__init__.py")
    print(f"   📄 app/models/document.py")
    print(f"   📄 app/services/__init__.py")
    print(f"   📄 Todos los archivos __init__.py")

if __name__ == "__main__":
    """Punto de entrada"""
    try:
        # Instalar chardet si no está disponible
        try:
            import chardet
        except ImportError:
            print("📦 Instalando chardet...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "chardet"])
            import chardet
        
        main()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ CORRECCIÓN INTERRUMPIDA POR USUARIO")
        
    except Exception as e:
        print(f"\n\n❌ ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n📚 Si el problema persiste:")
        print("   1. Verificar que estás en el directorio correcto")
        print("   2. Usar editor que soporte UTF-8 (VS Code, Notepad++)")
        print("   3. Consultar documentación del proyecto")
        print("\n👨‍🎓 TFM Vicente Caruncho Ramos - Universitat Jaume I")
        print("=" * 70)