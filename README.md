# 🤖 Prototipo_chatbot

## **Sistema de Chatbot RAG para Administraciones Locales**

> **Prototipo de Chatbot Interno para Administraciones Locales Usando Modelos de Lenguaje Locales y Comparación con OpenAI**  
> **Trabajo Final de Máster - Vicente Caruncho Ramos**  
> **Máster en Sistemas Inteligentes - Universitat Jaume I**  
> **Tutor: Rafael Berlanga Llavori**  
> **Curso 2024-2025**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![FAISS](https://img.shields.io/badge/FAISS-1.7+-red.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4+-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-89%25%20Completado-brightgreen.svg)
![TFM](https://img.shields.io/badge/TFM-Sistemas%20Inteligentes-orange.svg)

---

## 📋 Descripción del Proyecto

### **Contexto y Motivación**

Las administraciones locales españolas manejan grandes volúmenes de información distribuida en múltiples fuentes: documentos oficiales, páginas web, bases de datos, APIs de servicios, etc. Los ciudadanos y funcionarios necesitan acceder a esta información de manera eficiente, pero actualmente enfrentan:

- **Fragmentación de información** en múltiples sistemas
- **Dificultad para encontrar** respuestas específicas
- **Inconsistencias** entre diferentes fuentes
- **Procesos manuales** lentos y propensos a errores
- **Barrera técnica** para consultas complejas

### **Solución Propuesta**

Este proyecto desarrolla un **sistema conversacional basado en arquitectura RAG** (Retrieval-Augmented Generation) que permite a técnicos municipales y ciudadanos consultar información administrativa mediante lenguaje natural, integrando múltiples fuentes de datos de forma transparente y eficiente.

### **Objetivos del TFM**

#### **Objetivo Principal**
Desarrollar y evaluar un prototipo de chatbot RAG que demuestre la viabilidad técnica y económica de implementar sistemas de IA conversacional en administraciones locales españolas, comparando el rendimiento de modelos locales versus servicios cloud.

#### **Objetivos Específicos**
1. **Diseñar una arquitectura RAG modular** adaptable a diferentes administraciones
2. **Implementar ingesta multimodal** de documentos administrativos (PDF, DOCX, web, APIs)
3. **Comparar empíricamente** tecnologías de vector stores (FAISS vs ChromaDB)
4. **Evaluar modelos de lenguaje** locales (Ollama) versus cloud (OpenAI)
5. **Desarrollar métricas específicas** para evaluar sistemas RAG en contexto gubernamental
6. **Crear framework reproducible** para futuras investigaciones en el sector público

---

## ✨ Características Principales

### **🔍 RAG Avanzado**
- **Recuperación semántica** desde múltiples fuentes heterogéneas
- **Embeddings optimizados** con sentence-transformers y cache LRU inteligente
- **Vector stores duales** para comparación empírica (FAISS + ChromaDB)
- **Filtrado avanzado** por metadatos y contexto temporal
- **Ranking híbrido** que combina relevancia semántica y autoridad de fuente

### **⚖️ Comparación Dual de Modelos**
- **Modelos locales** (Ollama: llama3.2:3b, mistral:7b, gemma2:2b)
- **Modelos cloud** (OpenAI: gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
- **Evaluación automática** de respuestas con métricas de calidad
- **Análisis de costos** y tiempo de respuesta en tiempo real
- **Dashboard comparativo** con visualización de resultados

### **🛡️ Seguridad y Cumplimiento**
- **Procesamiento local** de datos sensibles para cumplimiento ENS
- **Trazabilidad completa** de consultas y respuestas
- **Audit logging** para transparencia y accountability
- **Rate limiting** y protección contra abuso
- **Preparado para CCN-TEC 014** (Esquema Nacional de Seguridad)

### **🔧 Arquitectura Modular**
- **Componentes intercambiables** para adaptación específica
- **Configuración YAML** centralizada y flexible
- **APIs REST** para integración con sistemas existentes
- **WebSocket** para chat en tiempo real
- **Docker ready** para deployment escalable

### **📊 Métricas y Benchmarking**
- **Framework científico** de evaluación reproducible
- **Métricas específicas** para administraciones públicas
- **Benchmarking automático** FAISS vs ChromaDB
- **Reportes académicos** en JSON y Markdown
- **Visualización** de resultados empíricos

### **📥 Ingesta Multimodal**
- **Documentos estructurados**: PDF, DOCX, TXT, Excel, CSV
- **Contenido web**: Scraping inteligente con rate limiting
- **APIs REST**: Conexión a servicios municipales existentes
- **Bases de datos**: Integración SQL con múltiples SGBD
- **Procesamiento OCR** para documentos escaneados

---

## 🏗️ Arquitectura del Sistema

### **Vista General**

```mermaid
graph TB
    subgraph "🌐 Capa de Presentación"
        WebUI[Interfaz Web]
        ChatUI[Chat Interface]
        AdminUI[Panel Admin]
        API[REST API]
    end
    
    subgraph "🧠 Capa de Procesamiento"
        Router[Request Router]
        Auth[Autenticación]
        
        subgraph "📥 Pipeline de Ingesta"
            DocProc[Procesador Documentos]
            WebScrap[Web Scraper]
            APIConn[Conector APIs]
            DBConn[Conector BBDD]
        end
        
        subgraph "🔍 Motor RAG" 
            Embed[EmbeddingService]
            FAISS[FAISS Store]
            ChromaDB[ChromaDB Store]
            LLMService[LLM Service]
        end
        
        subgraph "📊 Análisis"
            Metrics[Recolector Métricas]
            Benchmark[Motor Benchmark]
            Comparator[Comparador Modelos]
        end
    end
    
    subgraph "💾 Capa de Almacenamiento"
        VectorDB[(Vector Databases)]
        FileStore[(Almacén Archivos)]
        MetaDB[(Base Metadatos)]
        Cache[(Cache LRU)]
    end
    
    WebUI --> Router
    ChatUI --> Router
    AdminUI --> Router
    API --> Router
    
    Router --> DocProc
    Router --> WebScrap
    Router --> APIConn
    Router --> DBConn
    
    DocProc --> Embed
    WebScrap --> Embed
    APIConn --> Embed
    DBConn --> Embed
    
    Embed --> FAISS
    Embed --> ChromaDB
    FAISS --> LLMService
    ChromaDB --> LLMService
    
    LLMService --> Metrics
    Metrics --> Benchmark
    Benchmark --> Comparator
    
    Embed --> Cache
    FAISS --> VectorDB
    ChromaDB --> VectorDB
```

### **Componentes Principales**

#### **🧠 EmbeddingService**
```python
# Servicio de embeddings optimizado
class EmbeddingService:
    - modelo: all-MiniLM-L6-v2 (384 dimensiones)
    - cache_lru: Optimización inteligente de rendimiento
    - batch_processing: Procesamiento eficiente en lotes
    - metrics: Tracking detallado para análisis
    - memory_management: Gestión automática de recursos
```

#### **🗄️ Vector Stores Duales**
```python
# FAISS - Velocidad y control
FaissVectorStore:
    - indices: IndexFlatL2, IndexIVFFlat, HNSW
    - metadatos: Gestión externa con pickle
    - optimización: Parameter tuning automático
    - memoria: Eficiencia máxima para datasets grandes

# ChromaDB - Facilidad y funcionalidad  
ChromaDBVectorStore:
    - persistencia: SQLite backend automático
    - metadatos: Integrados con queries complejas
    - filtrado: WHERE clauses nativas avanzadas
    - transacciones: ACID compliance garantizado
```

#### **🤖 LLM Service**
```python
# Gestión dual de modelos
LLMService:
    ollama_client:
        - modelos: llama3.2:3b, mistral:7b, gemma2:2b
        - local: Sin costos por token, máxima privacidad
        - latencia: Optimizada para respuestas rápidas
        
    openai_client:
        - modelos: gpt-4o, gpt-4o-mini, gpt-3.5-turbo  
        - cloud: Calidad state-of-the-art
        - cost_tracking: Monitoreo automático de gastos
```

---

## 🚀 Inicio Rápido

### **Prerrequisitos**

#### **Software Necesario**
- **Python 3.9+** (recomendado 3.11)
- **Git** para clonado del repositorio
- **PowerShell** (Windows) o **Bash** (Linux/macOS)
- **Ollama** para modelos locales - [Instalar Ollama](https://ollama.ai/)
- **Cuenta OpenAI** (opcional) - [OpenAI API](https://platform.openai.com/)

#### **Recursos del Sistema**
- **RAM**: Mínimo 8GB, recomendado 16GB
- **Almacenamiento**: 5GB libres para modelos y datos
- **CPU**: Cualquier procesador moderno (64-bit)
- **GPU**: Opcional, aceleración automática si disponible

### **Instalación Paso a Paso**

#### **1. Clonar y Configurar Proyecto**
```powershell
# Clonar repositorio
git clone https://github.com/vcaruncho/prototipo_chatbot.git
cd prototipo_chatbot

# Crear y activar entorno virtual
python -m venv venv
# Windows PowerShell:
venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate.bat  
# Linux/macOS:
source venv/bin/activate

# Actualizar pip e instalar dependencias
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### **2. Configurar Variables de Entorno**
```powershell
# Copiar configuración de ejemplo
Copy-Item .env.example .env

# Editar .env con tu configuración preferida
# Especialmente importante si usarás OpenAI:
OPENAI_API_KEY=sk-tu-api-key-aqui
```

#### **3. Configurar Modelos Locales (Ollama)**
```powershell
# Instalar modelos recomendados
ollama pull llama3.2:3b     # Modelo principal (2GB)
ollama pull mistral:7b      # Modelo alternativo (4GB)  
ollama pull gemma2:2b       # Modelo ligero (1.5GB)

# Verificar instalación
ollama list
ollama serve  # Iniciar servidor en background
```

#### **4. Verificar Instalación**
```powershell
# Verificar dependencias críticas
python -c "import torch, sentence_transformers, faiss, chromadb; print('✅ Stack AI completo OK')"

# Ejecutar tests de componentes
python test_embedding_service.py      # Debería mostrar ✅ 100% tests pasados
python test_chromadb_benchmark.py     # Verificar ChromaDB funcional
```

#### **5. Inicializar Base de Datos y Cache**
```powershell
# Crear directorios necesarios
python -c "
from pathlib import Path
dirs = ['data/vectorstore/faiss', 'data/vectorstore/chromadb', 'data/cache/embeddings', 'logs', 'data/reports']
[Path(d).mkdir(parents=True, exist_ok=True) for d in dirs]
print('✅ Estructura de directorios creada')
"

# Precargar modelo de embeddings (primera ejecución más lenta)
python -c "
from app.services.rag.embeddings import embedding_service
embedding_service.warm_up()
print('✅ Modelo de embeddings precargado')
"
```

#### **6. Ejecutar Aplicación**
```powershell
# Iniciar servidor de desarrollo
python run.py

# Abrir navegador en: http://localhost:5000
# Verás el dashboard principal con métricas del sistema
```

### **Verificación de Instalación Exitosa**

Si todo está correcto, deberías ver:
- ✅ **Dashboard funcionando** en http://localhost:5000
- ✅ **Logs estructurados** en `logs/prototipo_chatbot.log`
- ✅ **Health check** verde en `/health`
- ✅ **Métricas de sistema** actualizándose
- ✅ **Chat interface** respondiendo (preparada para RAG)

---

## 📊 Uso del Sistema

### **Interfaz Web Principal**

#### **🏠 Dashboard** (`/`)
- **Métricas en tiempo real** del sistema
- **Estado de servicios** (Embeddings, Vector Stores, LLM)
- **Estadísticas de uso** (consultas, documentos indexados)
- **Gráficos de rendimiento** (tiempo de respuesta, throughput)

#### **💬 Chat Interface** (`/chat`)
- **Chat conversacional** con interfaz moderna
- **Historial de sesiones** persistente
- **Fuentes citadas** con transparencia completa
- **Comparación de respuestas** (local vs cloud)
- **Métricas de consulta** (tiempo, tokens, costo)

#### **⚖️ Comparación de Modelos** (`/comparison`)
- **Testing lado a lado** de modelos locales vs cloud
- **Métricas detalladas** de rendimiento y calidad
- **Análisis de costos** en tiempo real
- **Exportación de resultados** para análisis posterior

#### **🔧 Panel de Administración** (`/admin`)
- **Gestión de documentos** indexados
- **Configuración del sistema** en tiempo real
- **Logs del sistema** con filtrado avanzado
- **Métricas de uso** y estadísticas detalladas

### **API REST Documentada**

#### **Endpoints Principales**
```http
# Salud del sistema
GET /api/health
Content-Type: application/json
Response: {"status": "healthy", "services": {...}, "metrics": {...}}

# Chat conversacional  
POST /api/chat
Content-Type: application/json
Body: {"message": "¿Cómo tramitar una licencia?", "session_id": "uuid"}
Response: {"response": "...", "sources": [...], "metrics": {...}}

# Comparación de modelos
POST /api/compare
Content-Type: application/json  
Body: {"query": "...", "models": ["local", "openai"]}
Response: {"local_response": {...}, "openai_response": {...}, "comparison": {...}}

# Ingesta de documentos
POST /api/ingest
Content-Type: multipart/form-data
Body: file upload + metadata
Response: {"job_id": "uuid", "status": "processing", "progress": 0}

# Métricas del sistema
GET /api/metrics
Response: {"embeddings": {...}, "vector_stores": {...}, "llm": {...}}
```

### **Configuración Avanzada**

#### **Archivo .env Completo**
```bash
# ===== CONFIGURACIÓN GENERAL =====
PROJECT_NAME=Prototipo_chatbot
PROJECT_VERSION=1.0.0
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=tu-clave-secreta-segura

# ===== MODELOS DE LENGUAJE =====
# Modelos locales (Ollama)
DEFAULT_LOCAL_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
AVAILABLE_LOCAL_MODELS=llama3.2:3b,mistral:7b,gemma2:2b

# Modelos OpenAI
OPENAI_API_KEY=sk-tu-api-key-aqui
DEFAULT_OPENAI_MODEL=gpt-4o-mini
AVAILABLE_OPENAI_MODELS=gpt-4o,gpt-4o-mini,gpt-3.5-turbo

# ===== EMBEDDINGS =====
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
EMBEDDING_CACHE_DIR=data/cache/embeddings

# ===== VECTOR STORES =====
DEFAULT_VECTOR_STORE=faiss
FAISS_INDEX_PATH=data/vectorstore/faiss
CHROMADB_PATH=data/vectorstore/chromadb

# ===== RAG CONFIGURATION =====
RAG_K_DEFAULT=5
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

---

## 🧪 Benchmarking y Evaluación

### **Framework de Benchmarking Científico**

Este proyecto incluye un framework completo para evaluación empírica de tecnologías RAG, diseñado específicamente para investigación académica y toma de decisiones informadas.

#### **Ejecutar Comparación Académica**
```powershell
# Benchmark completo FAISS vs ChromaDB
python comparison_faiss_vs_chromadb.py

# Resultado: 
# - data/reports/vector_stores_comparison_YYYYMMDD_HHMMSS.json
# - data/reports/vector_stores_summary_YYYYMMDD_HHMMSS.md
# - Análisis estadístico completo con intervalos de confianza
# - Recomendaciones específicas para administraciones locales
```

#### **Métricas Evaluadas**

##### **Rendimiento Técnico**
```python
performance_metrics = {
    "insertion_throughput": "documentos/segundo",
    "search_latency": "milisegundos/consulta", 
    "memory_efficiency": "MB/1000_documentos",
    "disk_usage": "MB total",
    "scalability": "degradación con dataset size",
    "concurrent_performance": "usuarios simultáneos soportados"
}
```

##### **Calidad de Resultados**
```python
quality_metrics = {
    "relevance_at_k": "% resultados relevantes en top-k",
    "mean_reciprocal_rank": "posición promedio primer resultado relevante",
    "diversity": "variedad de fuentes en resultados",
    "consistency": "estabilidad entre ejecuciones",
    "filter_effectiveness": "precisión de filtros por metadatos"
}
```

##### **Usabilidad y Mantenimiento**
```python
usability_metrics = {
    "api_complexity": "líneas de código para tareas comunes",
    "learning_curve": "tiempo para implementar casos de uso básicos",
    "documentation_quality": "completitud y claridad de documentación",
    "community_support": "actividad y recursos disponibles",
    "operational_overhead": "esfuerzo requerido para operación"
}
```

### **Dataset de Evaluación**

El sistema incluye un dataset cuidadosamente curado que representa casos de uso reales en administraciones locales españolas:

```python
# 20 documentos representativos que cubren:
administrative_domains = [
    "Normativa y regulaciones municipales",
    "Procedimientos administrativos y tramitación", 
    "Servicios ciudadanos y atención al público",
    "Gestión financiera y tributaria local",
    "Urbanismo y planificación territorial",
    "Servicios municipales especializados"
]

# 10 consultas típicas de usuarios reales:
typical_queries = [
    "licencias y permisos municipales",
    "servicios digitales administración",
    "tramitación expedientes ciudadanos",
    "normativa ordenanzas locales",
    "presupuestos hacienda municipal",
    # ... más consultas representativas
]
```

### **Interpretación de Resultados**

#### **Reporte Automático Generado**
```markdown
# Ejemplo de salida del benchmark:

## Resultados Principales:
- **Inserción**: FAISS 45.3 docs/seg vs ChromaDB 23.7 docs/seg (1.9x más rápido)
- **Búsqueda**: FAISS 12.4ms vs ChromaDB 28.7ms (2.3x más rápido)  
- **Memoria**: FAISS 156MB vs ChromaDB 89MB (ChromaDB 43% más eficiente)
- **Funcionalidad**: ChromaDB superior en filtros complejos y metadatos

## Recomendaciones:
- **Para >10K documentos**: FAISS recomendado por rendimiento
- **Para prototipado rápido**: ChromaDB recomendado por simplicidad
- **Para equipos sin experiencia AI**: ChromaDB más amigable
- **Para producción de alta escala**: FAISS con optimizaciones
```

---

## 🔧 Desarrollo y Extensión

### **Estructura del Proyecto**
```
prototipo_chatbot/
├── 📁 app/                          # Aplicación principal
│   ├── 📁 core/                     # Configuración y utilidades
│   │   ├── config.py                  # Configuración YAML centralizada
│   │   └── logger.py                  # Logging estructurado
│   ├── 📁 models/                   # Modelos de datos
│   │   └── __init__.py                # Dataclasses con validación
│   ├── 📁 routes/                   # Rutas web y API
│   │   ├── main.py                    # Dashboard y páginas principales
│   │   ├── api.py                     # Endpoints REST documentados
│   │   ├── chat.py                    # Interfaz de chat
│   │   └── admin.py                   # Panel de administración
│   ├── 📁 services/                 # Lógica de negocio
│   │   ├── 📁 ingestion/              # Pipeline de ingesta multimodal
│   │   │   ├── document_processor.py   # PDF, DOCX, TXT, Excel
│   │   │   ├── web_scraper.py          # Scraping inteligente
│   │   │   ├── api_connector.py        # Integración APIs REST
│   │   │   └── database_connector.py   # Conexión BBDD SQL
│   │   ├── 📁 rag/                    # Motor RAG completo
│   │   │   ├── embeddings.py           # EmbeddingService optimizado
│   │   │   ├── faiss_store.py          # Vector store FAISS
│   │   │   ├── chromadb_store.py       # Vector store ChromaDB
│   │   │   └── vector_store.py         # Interfaz abstracta común
│   │   └── llm_service.py             # Gestión modelos LLM
│   ├── 📁 static/                   # Assets frontend
│   │   ├── 📁 css/                    # Estilos personalizados
│   │   ├── 📁 js/                     # JavaScript moderno
│   │   └── 📁 images/                 # Recursos gráficos
│   ├── 📁 templates/                # Templates Jinja2
│   │   ├── base.html                  # Template base Bootstrap 5
│   │   ├── index.html                 # Dashboard principal
│   │   ├── chat.html                  # Interfaz chat
│   │   └── admin.html                 # Panel administración
│   └── __init__.py                  # Factory Flask con blueprints
├── 📁 data/                         # Datos y almacenamiento
│   ├── 📁 vectorstore/                # Vector databases
│   │   ├── 📁 faiss/                  # Índices FAISS
│   │   └── 📁 chromadb/               # Base datos ChromaDB
│   ├── 📁 cache/                      # Cache de embeddings
│   ├── 📁 reports/                    # Reportes de benchmarking
│   └── 📁 uploads/                    # Archivos subidos por usuarios
├── 📁 logs/                         # Logs estructurados
├── 📁 tests/                        # Suite de tests
│   ├── test_embedding_service.py      # Tests embeddings
│   ├── test_chromadb_benchmark.py     # Tests ChromaDB
│   └── test_faiss_store.py           # Tests FAISS
├── 📁 docs/                         # Documentación técnica
│   ├── arquitectura_faiss.md          # Análisis FAISS para TFM
│   ├── arquitectura_chromadb.md       # Análisis ChromaDB para TFM
│   └── guia_benchmarking.md          # Metodología científica
├── 📄 comparison_faiss_vs_chromadb.py # Script benchmarking académico
├── 📄 requirements.txt              # Dependencias Python
├── 📄 .env.example                  # Template configuración
├── 📄 run.py                        # Punto de entrada
└── 📄 README.md                     # Esta documentación
```

### **Añadir Nuevas Funcionalidades**

#### **Nuevo Vector Store**
```python
# 1. Implementar interfaz común
from app.services.rag.vector_store import VectorStore

class NuevoVectorStore(VectorStore):
    def __init__(self, config):
        # Inicialización específica
        pass
    
    def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        # Implementar indexación
        pass
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[DocumentChunk]:
        # Implementar búsqueda
        pass

# 2. Registrar en factory
# app/services/rag/__init__.py
vector_store_registry["nuevo"] = NuevoVectorStore

# 3. Añadir configuración
# .env
DEFAULT_VECTOR_STORE=nuevo
NUEVO_CONFIG_PARAM=valor
```

#### **Nuevo Modelo LLM**
```python
# 1. Extender LLMService
from app.services.llm_service import LLMService

class NuevoLLMClient:
    def generate(self, prompt: str, context: List[DocumentChunk]) -> ModelResponse:
        # Implementar generación
        pass

# 2. Integrar en servicio principal
llm_service.register_client("nuevo_modelo", NuevoLLMClient())

# 3. Usar en chat
response = llm_service.generate_dual(
    query="consulta usuario",
    context=retrieved_chunks,
    models=["nuevo_modelo", "gpt-4o"]
)
```

#### **Nuevo Procesador de Documentos**
```python
# 1. Implementar procesador específico
from app.services.ingestion.document_processor import DocumentProcessor

class NuevoDocumentProcessor(DocumentProcessor):
    def can_process(self, file_path: str) -> bool:
        return file_path.endswith('.nuevo_formato')
    
    def process(self, file_path: str) -> List[DocumentChunk]:
        # Lógica de procesamiento específica
        pass

# 2. Registrar procesador
document_processor.register_processor(NuevoDocumentProcessor())
```

### **Testing y Calidad**

#### **Ejecutar Tests Existentes**
```powershell
# Tests unitarios de componentes
python test_embedding_service.py       # EmbeddingService completo
python test_chromadb_benchmark.py      # ChromaDB funcionalidad  
python test_faiss_store.py            # FAISS operaciones

# Verificar salud del sistema
python -c "
from app import create_app
app = create_app()
with app.test_client() as client:
    response = client.get('/api/health')
    print(f'Health Check: {response.status_code}')
    print(response.get_json())
"
```

#### **Añadir Nuevos Tests**
```python
# tests/test_nuevo_componente.py
import pytest
from app.services.nuevo_componente import NuevoComponente

def test_nuevo_componente_inicializacion():
    componente = NuevoComponente()
    assert componente.is_available()

def test_nuevo_componente_funcionalidad():
    componente = NuevoComponente()
    resultado = componente.procesar("input_test")
    assert resultado is not None
    assert len(resultado) > 0

# Ejecutar con pytest
pytest tests/test_nuevo_componente.py -v
```

#### **Profiling y Optimización**
```python
# Análisis de rendimiento
import cProfile
import pstats

def profile_embedding_service():
    pr = cProfile.Profile()
    pr.enable()
    
    # Código a perfilar
    from app.services.rag.embeddings import embedding_service
    texts = ["texto ejemplo"] * 100
    embeddings = embedding_service.encode_batch(texts)
    
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(10)

# python -c "from profile_script import profile_embedding_service; profile_embedding_service()"
```

---

## 📊 Casos de Uso Específicos

### **Para Administraciones Locales**

#### **Caso 1: Consultas Ciudadanas**
```python
# Ejemplo de consulta típica
consulta_ciudadano = """
¿Qué documentos necesito para solicitar una licencia de apertura 
de un pequeño comercio en el centro histórico?
"""

# El sistema busca en:
sources_consulted = [
    "Ordenanza Municipal de Licencias",
    "Guía del Emprendedor (web municipal)", 
    "FAQ del Portal del Ciudadano",
    "Base de datos de procedimientos",
    "Normativa autonómica aplicable"
]

# Respuesta contextualizada con fuentes
response = {
    "answer": "Para una licencia de apertura necesitarás...",
    "sources": [
        {"title": "Ordenanza de Licencias Art. 15", "confidence": 0.95},
        {"title": "Guía Portal Ciudadano", "confidence": 0.87}
    ],
    "next_steps": ["Contactar ventanilla única", "Revisar normativa específica"]
}
```

#### **Caso 2: Consultas Técnicas Internas**
```python
# Consulta de funcionario municipal
consulta_funcionario = """
¿Cuál es el procedimiento para modificar una licencia urbanística 
ya concedida cuando hay cambio de actividad?
"""

# Búsqueda en documentación técnica avanzada
specialized_sources = [
    "Manual de Procedimientos Internos",
    "Jurisprudencia administrativa reciente",
    "Circulares de la Consejería de Urbanismo",
    "Casos similares resueltos (base de datos)",
    "Normativa técnica edificación"
]

# Respuesta con detalle técnico y referencias normativas
```

#### **Caso 3: Análisis Comparativo**
```python
# Comparación de respuestas modelo local vs cloud
comparison_result = {
    "query": "Procedimiento sancionador en materia de ruidos",
    "local_model": {
        "model": "llama3.2:3b",
        "response_time": "2.3s",
        "cost": "€0.00",
        "answer": "Respuesta detallada...",
        "quality_score": 8.2
    },
    "cloud_model": {
        "model": "gpt-4o-mini", 
        "response_time": "1.1s",
        "cost": "€0.003",
        "answer": "Respuesta alternativa...",
        "quality_score": 9.1
    },
    "recommendation": "Cloud model para consultas complejas, local para consultas rutinarias"
}
```

### **Para Investigadores**

#### **Reproducir Experimentos**
```powershell
# Configuración determinística para reproducibilidad
export PYTHONHASHSEED=42
export CUDA_DETERMINISTIC=1

# Ejecutar benchmark con configuración académica
python comparison_faiss_vs_chromadb.py --config=academic --seed=42 --iterations=10

# Analizar resultados con estadísticas
python analyze_benchmark_results.py data/reports/latest_comparison.json
```

#### **Extender Dataset de Evaluación**
```python
# Añadir nuevos dominios de evaluación
new_evaluation_domains = {
    "salud_publica": [
        "Protocolo COVID-19 en centros municipales",
        "Normativa sanidad alimentaria restaurantes",
        # ... más documentos específicos
    ],
    "medio_ambiente": [
        "Ordenanza gestión residuos municipales", 
        "Plan sostenibilidad energética local",
        # ... documentos ambientales
    ]
}

# Integrar en benchmark
benchmark_config.update_domains(new_evaluation_domains)
```

### **Para Desarrolladores**

#### **Integración con Sistemas Existentes**
```python
# API REST para integración
import requests

# Consultar desde sistema externo
response = requests.post('http://localhost:5000/api/chat', json={
    'message': 'Consulta desde sistema externo',
    'session_id': 'sistema_gestion_municipal_001',
    'metadata': {
        'user_role': 'funcionario',
        'department': 'urbanismo'
    }
})

result = response.json()
print(f"Respuesta: {result['response']}")
print(f"Fuentes: {result['sources']}")
```

#### **Webhook para Notificaciones**
```python
# Notificar actualizaciones de documentos
webhook_payload = {
    'event': 'document_updated',
    'document_id': 'ordenanza_municipal_v2.pdf',
    'changes': ['Artículo 15 modificado', 'Nuevo Anexo III'],
    'requires_reindexing': True
}

requests.post('http://localhost:5000/api/webhooks/document_update', 
              json=webhook_payload)
```

---

## 🚀 Deployment y Producción

### **Deployment Local (Desarrollo)**
```powershell
# Configuración de desarrollo
$env:FLASK_ENV="development"
$env:FLASK_DEBUG="True"
python run.py
```

### **Deployment con Docker**
```dockerfile
# Dockerfile (preparado para implementar)
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "run:app"]
```

```yaml
# docker-compose.yml (preparado)
version: '3.8'
services:
  chatbot:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=postgresql://user:pass@db:5432/chatbot
    volumes:
      - ./data:/app/data
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: chatbot
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    
volumes:
  postgres_data:
```

### **Deployment Azure (Preparado)**
```yaml
# azure-pipelines.yml (preparado para CI/CD)
trigger:
- main

pool:
  vmImage: 'ubuntu-latest'

steps:
- task: UsePythonVersion@0
  inputs:
    versionSpec: '3.11'
  displayName: 'Use Python 3.11'

- script: |
    pip install -r requirements.txt
    python -m pytest tests/ -v
  displayName: 'Run tests'

- task: AzureWebApp@1
  inputs:
    azureSubscription: 'azure-subscription'
    appName: 'prototipo-chatbot'
    package: '.'
```

### **Configuración Producción**
```bash
# .env.production
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=clave-super-segura-produccion

# Base de datos PostgreSQL
DATABASE_URL=postgresql://user:pass@servidor:5432/chatbot_prod

# Redis para cache distribuido  
REDIS_URL=redis://servidor:6379/0

# Configuración seguridad
CORS_ORIGINS=https://ayuntamiento.es,https://admin.ayuntamiento.es
RATE_LIMIT_GLOBAL=5000/hour
RATE_LIMIT_CHAT=100/hour

# Monitoreo
SENTRY_DSN=https://sentry-dsn-aqui
LOG_LEVEL=INFO
```

---

## 📈 Monitoreo y Observabilidad

### **Métricas del Sistema**
```python
# Métricas automáticas incluidas
system_metrics = {
    "health": {
        "status": "healthy/degraded/unhealthy",
        "uptime_seconds": 0,
        "last_restart": "timestamp"
    },
    "performance": {
        "requests_per_minute": 0,
        "avg_response_time_ms": 0,
        "error_rate_percent": 0,
        "active_sessions": 0
    },
    "resources": {
        "memory_usage_mb": 0,
        "cpu_usage_percent": 0,
        "disk_usage_gb": 0,
        "gpu_memory_mb": 0  # Si disponible
    },
    "ai_services": {
        "embedding_cache_hit_rate": 0,
        "vector_store_size_mb": 0,
        "llm_tokens_per_minute": 0,
        "cost_per_hour_usd": 0
    }
}
```

### **Alertas Configurables**
```yaml
# config/alerts.yml (preparado)
alerts:
  high_error_rate:
    condition: "error_rate > 5%"
    action: "email_admin"
    
  high_response_time:
    condition: "avg_response_time > 5000ms"
    action: "slack_notification"
    
  low_disk_space:
    condition: "disk_usage > 90%"
    action: "email_admin"
    
  embedding_service_down:
    condition: "embedding_service.status != healthy"
    action: "immediate_notification"
```

### **Dashboard de Métricas**
Acceso en `/admin/metrics` con visualización en tiempo real de:
- 📊 **Gráficos de rendimiento** (Chart.js)
- 🎯 **KPIs principales** del sistema
- 📈 **Tendencias históricas** de uso
- 🚨 **Alertas activas** y su estado
- 💰 **Costos estimados** de operación

---

## 🤝 Contribuir al Proyecto

### **Para Estudiantes e Investigadores**

#### **Áreas de Contribución**
1. **Nuevos vector stores** (Pinecone, Weaviate, Qdrant)
2. **Modelos de embedding** (multilingual, domain-specific)
3. **Métricas de evaluación** específicas para sector público
4. **Datasets especializados** por dominio administrativo
5. **Optimizaciones de rendimiento** para hardware específico

#### **Proceso de Contribución**
```powershell
# 1. Fork del repositorio
git clone https://github.com/tu-usuario/prototipo_chatbot.git
cd prototipo_chatbot

# 2. Crear rama para feature
git checkout -b feature/nueva-funcionalidad

# 3. Implementar y testear
# ... desarrollo ...
python -m pytest tests/ -v

# 4. Commit y push
git add .
git commit -m "feat: descripción clara del cambio"
git push origin feature/nueva-funcionalidad

# 5. Crear Pull Request con:
# - Descripción clara del cambio
# - Tests que pasen
# - Documentación actualizada
# - Ejemplo de uso si aplica
```

### **Para Administraciones Locales**

#### **Personalización para tu Ayuntamiento**
```python
# config/municipio_config.yml
municipio:
  nombre: "Ayuntamiento de Tu Ciudad"
  logo: "static/images/escudo_municipal.png"
  colores:
    primario: "#003366"    # Azul institucional
    secundario: "#66CC00"  # Verde sostenible
  
  dominios_especializados:
    - "normativa_local"
    - "servicios_ciudadanos" 
    - "tramites_especificos"
    
  fuentes_datos:
    sede_electronica: "https://sede.tuciudad.es"
    portal_transparencia: "https://transparencia.tuciudad.es"
    normativa: "https://normativa.tuciudad.es"
```

#### **Casos de Uso Sugeridos**
1. **Consultas ciudadanas frecuentes** - Automatización primer nivel
2. **Soporte técnico interno** - Ayuda a funcionarios
3. **Análisis de normativa** - Búsqueda en legislación compleja
4. **Gestión documental** - Indexación automática de expedientes
5. **Atención multiidioma** - Soporte comunidades locales

---

## 📚 Recursos y Referencias

### **Documentación Técnica**
- 📖 **[Arquitectura FAISS](docs/arquitectura_faiss.md)** - Análisis técnico completo
- 📖 **[Arquitectura ChromaDB](docs/arquitectura_chromadb.md)** - Comparación detallada
- 📖 **[Guía de Benchmarking](docs/guia_benchmarking.md)** - Metodología científica
- 🔧 **[API Reference](http://localhost:5000/api/docs)** - Documentación interactiva

### **Papers y Referencias Académicas**
```bibtex
@misc{caruncho2025chatbot,
  title={Prototipo de Chatbot Interno para Administraciones Locales: 
         Comparación Empírica de Modelos Locales vs Cloud},
  author={Caruncho Ramos, Vicente},
  year={2025},
  school={Universitat Jaume I},
  type={Trabajo Final de Máster},
  note={Máster en Sistemas Inteligentes}
}

@article{lewis2020retrieval,
  title={Retrieval-augmented generation for knowledge-intensive nlp tasks},
  author={Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and others},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={9459--9474},
  year={2020}
}
```

### **Tecnologías Utilizadas**
- 🤖 **[sentence-transformers](https://www.sbert.net/)** - Embeddings semánticos
- 🔍 **[FAISS](https://faiss.ai/)** - Vector similarity search
- 🗄️ **[ChromaDB](https://www.trychroma.com/)** - Vector database
- 🌐 **[Flask](https://flask.palletsprojects.com/)** - Web framework
- 🎨 **[Bootstrap 5](https://getbootstrap.com/)** - UI framework
- 🤖 **[Ollama](https://ollama.ai/)** - Local LLM runtime
- ☁️ **[OpenAI API](https://platform.openai.com/)** - Cloud LLM service

### **Comunidad y Soporte**
- 💬 **Discussions**: Issues y mejoras en GitHub
- 📧 **Email**: vcaruncho@uji.es (autor del TFM)
- 🐦 **Actualizaciones**: Seguir repositorio para cambios
- 📊 **Benchmarks**: Contribuir con nuevos datasets y métricas

---

## 📄 Licencia y Atribución

### **Licencia MIT**
```
MIT License

Copyright (c) 2025 Vicente Caruncho Ramos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### **Cita Académica**
Si utilizas este proyecto en investigación académica, por favor cita:

```
Caruncho Ramos, V. (2025). Prototipo de Chatbot Interno para Administraciones 
Locales Usando Modelos de Lenguaje Locales y Comparación con OpenAI. 
Trabajo Final de Máster, Máster en Sistemas Inteligentes, 
Universitat Jaume I, España.
```

### **Agradecimientos**
- **Rafael Berlanga Llavori** - Tutor del TFM y guía técnica
- **Universitat Jaume I** - Máster en Sistemas Inteligentes
- **Comunidad open-source** - Desarrolladores de las librerías utilizadas
- **Administraciones locales** - Casos de uso y requisitos reales

---

## 🎯 Estado Actual y Roadmap

### **Estado Actual: 89% Completado** ✅
- ✅ **Sistema RAG core funcional** con dual vector stores
- ✅ **Framework de benchmarking científico** implementado
- ✅ **Documentación académica completa** para TFM
- ✅ **Interface web profesional** con UX optimizada
- ✅ **Pipeline de ingesta multimodal** robusta y escalable
- 🔄 **Integración LLM dual** (75% completado)

### **Próximos Hitos** 🚀
1. **Completar integración Ollama + OpenAI** (Esta semana)
2. **Ejecutar benchmarking académico completo** (Esta semana)  
3. **Finalizar memoria TFM** (Próximas 2 semanas)
4. **Defensa del TFM** con demo en vivo

### **Extensiones Futuras** 🔮
- ☁️ **Deployment Azure/AWS** para demostración cloud
- 🛡️ **Seguridad avanzada** con autenticación y autorización
- 📊 **Dashboard analytics** con métricas en tiempo real
- 🌐 **API GraphQL** para consultas complejas
- 🤖 **Más modelos LLM** (Claude, Gemini, modelos especializados)
- 🌍 **Soporte multiidioma** para comunidades diversas

---

**📊 Este README documenta un sistema RAG completo, funcional y listo para investigación académica y aplicación práctica en administraciones locales españolas.**

**🎓 Desarrollado como TFM en Máster de Sistemas Inteligentes - Universitat Jaume I**  
**👨‍💻 Vicente Caruncho Ramos - 2025**