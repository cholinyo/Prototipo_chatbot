# Arquitectura ChromaDB para Prototipo_chatbot TFM

## 🎯 **Análisis Estratégico de ChromaDB**

### **¿Qué es ChromaDB?**
ChromaDB es una base de datos vectorial open-source diseñada específicamente para aplicaciones de AI y sistemas RAG (Retrieval-Augmented Generation). Se caracteriza por su simplicidad de uso, capacidad de ejecutarse tanto en memoria como en modo persistente, y su integración nativa con modelos de embeddings.

### **Características Principales**
```
🔹 Base de datos vectorial especializada
🔹 Open-source con licencia Apache 2.0
🔹 API simple e intuitiva
🔹 Soporte nativo para metadatos complejos
🔹 Filtrado avanzado por atributos
🔹 Modo in-memory y persistente
🔹 Integración con sentence-transformers
🔹 Arquitectura cliente/servidor opcional
```

## 📊 **Comparativa: ChromaDB vs FAISS**

### **Ventajas de ChromaDB**
```
✅ FACILIDAD DE USO:
   • API más intuitiva y menos código
   • Gestión automática de metadatos
   • No requiere mapeos manuales ID->vector

✅ FUNCIONALIDADES AVANZADAS:
   • Filtrado nativo por metadatos
   • Queries complejas con WHERE clauses
   • Soporte para múltiples colecciones
   • Versionado automático de datos

✅ PERSISTENCIA:
   • Base de datos real con SQLite backend
   • Transacciones ACID
   • Recuperación automática ante fallos
   • Backup y restore integrados

✅ FLEXIBILIDAD:
   • Múltiples funciones de distancia
   • Embeddings functions personalizables
   • Metadatos estructurados y no estructurados
   • APIs REST y Python nativas
```

### **Desventajas de ChromaDB**
```
❌ RENDIMIENTO:
   • Menor velocidad que FAISS en datasets grandes
   • Overhead de base de datos relacional
   • Mayor uso de memoria para metadatos

❌ ESCALABILIDAD:
   • Limitaciones con millones de vectores
   • SQLite como bottleneck potencial
   • Modo distribuido aún en desarrollo

❌ CONTROL:
   • Menos control sobre algoritmos de indexación
   • Optimizaciones limitadas vs FAISS
   • Dependencia de decisiones del proyecto upstream
```

## 🏗️ **Arquitectura Técnica ChromaDB**

### **Estructura Interna**
```
🗄️ ChromaDB Database Structure:
├── Collections (equivalente a "tablas")
│   ├── documents/ (contenido textual)
│   ├── embeddings/ (vectores 384D)
│   ├── metadatas/ (información estructurada)
│   └── ids/ (identificadores únicos)
│
├── SQLite Backend
│   ├── collection_metadata
│   ├── embedding_metadata  
│   ├── segment_metadata
│   └── index_metadata
│
└── Vector Index
    ├── HNSW (por defecto)
    ├── Brute Force (pequeños datasets)
    └── Custom indexes (futuro)
```

### **Modelo de Datos Propuesto**
```python
# Colección Principal: "prototipo_documents"
{
    "ids": ["doc_123_chunk_5", "web_456_chunk_2", ...],
    
    "documents": [
        "Texto del fragmento de licencias municipales...",
        "Información web sobre tramitación...",
        ...
    ],
    
    "embeddings": [
        [0.1, -0.3, 0.7, ...],  # 384 dimensiones
        [0.2, 0.1, -0.4, ...],  # Vector 2
        ...
    ],
    
    "metadatas": [
        {
            # Metadatos de DocumentMetadata
            "source_path": "/docs/ordenanza_municipal.pdf",
            "source_type": "document",
            "file_type": ".pdf", 
            "size_bytes": 156847,
            "created_at": "2025-08-04T18:30:00Z",
            "processed_at": "2025-08-04T18:35:00Z",
            "checksum": "a1b2c3d4e5f6...",
            "title": "Ordenanza Municipal de Licencias",
            
            # Metadatos de DocumentChunk
            "chunk_id": "doc_123_chunk_5",
            "chunk_index": 5,
            "chunk_size": 856,
            "start_char": 2340,
            "end_char": 3196,
            "page_number": 12,
            "section_title": "Licencias de Actividad",
            
            # Metadatos específicos para filtrado
            "department": "urbanismo",
            "document_category": "normativa",
            "language": "es",
            "publication_year": 2024,
            "last_updated": "2025-01-15T10:00:00Z",
            "relevance_score": 0.95,
            "content_type": "regulation"
        },
        {
            # Chunk de contenido web
            "source_path": "https://ayuntamiento.es/licencias",
            "source_type": "web",
            "url": "https://ayuntamiento.es/licencias",
            "title": "Guía Online de Licencias",
            "scraped_at": "2025-08-04T15:45:00Z",
            "chunk_id": "web_456_chunk_2",
            "content_type": "guide",
            "department": "urbanismo",
            "last_verified": "2025-07-30T12:00:00Z"
        }
    ]
}
```

## 🎯 **Estrategias de Organización**

### **Opción A: Colección Única (Recomendado para TFM)**
```python
# Una sola colección: "prototipo_documents"
collection = client.create_collection(
    name="prototipo_documents",
    metadata={
        "description": "Todos los documentos del prototipo",
        "embedding_model": "all-MiniLM-L6-v2",
        "created_for": "TFM_Vicente_Caruncho"
    }
)

# Ventajas:
✅ Búsqueda cross-domain (encuentra relaciones entre tipos)
✅ Comparación directa con FAISS (mismo dataset)
✅ Filtrado flexible por metadatos
✅ Gestión simplificada
✅ Análisis académico más limpio
```

### **Opción B: Múltiples Colecciones (Casos Específicos)**
```python
# Colecciones separadas por tipo de fuente
collections = {
    "official_documents": client.create_collection("official_documents"),
    "web_content": client.create_collection("web_content"), 
    "api_data": client.create_collection("api_data"),
    "database_content": client.create_collection("database_content")
}

# Casos de uso:
- Diferentes modelos de embedding por tipo
- Requisitos de seguridad específicos
- Optimizaciones por dominio
- Separación por nivel de confianza
```

## 🔧 **Implementación Técnica**

### **Configuración Optimizada**
```python
import chromadb
from chromadb.config import Settings

# Configuración para desarrollo (persistente)
client = chromadb.PersistentClient(
    path="data/vectorstore/chromadb",
    settings=Settings(
        # Optimizaciones de rendimiento
        anonymized_telemetry=False,
        allow_reset=True,
        
        # Configuración de indexing
        hnsw_construction_ef=200,        # Calidad construcción
        hnsw_search_ef=100,              # Calidad búsqueda
        hnsw_M=16,                       # Conexiones por nodo
        
        # Configuración de memoria
        persist_directory="data/vectorstore/chromadb",
        
        # Configuración de logging
        chroma_server_cors_allow_origins=["*"],
        chroma_server_host="localhost",
        chroma_server_http_port=8000
    )
)

# Crear colección con configuración específica
collection = client.create_collection(
    name="prototipo_documents",
    metadata={
        "hnsw:space": "cosine",          # Función de distancia
        "hnsw:construction_ef": 200,
        "hnsw:M": 16,
        "description": "Documentos administrativos para RAG",
        "embedding_model": "all-MiniLM-L6-v2",
        "dimension": 384,
        "created_at": "2025-08-04T18:00:00Z",
        "tfm_author": "Vicente Caruncho Ramos"
    }
)
```

### **Pipeline de Indexación**
```python
def add_documents_to_chromadb(chunks: List[DocumentChunk]):
    """Pipeline optimizado para añadir documentos"""
    
    # 1. Preparar datos para ChromaDB
    ids = []
    documents = []
    embeddings = []
    metadatas = []
    
    for chunk in chunks:
        # IDs únicos
        ids.append(chunk.id or str(uuid.uuid4()))
        
        # Contenido textual
        documents.append(chunk.content)
        
        # Embeddings (generar si no existen)
        if chunk.embedding is None:
            embedding = embedding_service.encode_single_text(chunk.content)
        else:
            embedding = chunk.embedding
        embeddings.append(embedding.tolist())
        
        # Metadatos completos
        metadata = {
            # DocumentMetadata
            "source_path": chunk.metadata.source_path,
            "source_type": chunk.metadata.source_type,
            "file_type": chunk.metadata.file_type,
            "size_bytes": chunk.metadata.size_bytes,
            "created_at": chunk.metadata.created_at.isoformat(),
            "processed_at": chunk.metadata.processed_at.isoformat(),
            "checksum": chunk.metadata.checksum,
            "title": chunk.metadata.title or "",
            
            # DocumentChunk específicos
            "chunk_id": chunk.id,
            "chunk_index": chunk.chunk_index,
            "chunk_size": chunk.chunk_size,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            
            # Campos para filtrado
            "timestamp": datetime.now().isoformat(),
            "word_count": len(chunk.content.split()),
            "char_count": len(chunk.content)
        }
        
        # Añadir metadatos opcionales
        if hasattr(chunk.metadata, 'language') and chunk.metadata.language:
            metadata["language"] = chunk.metadata.language
        if hasattr(chunk.metadata, 'url') and chunk.metadata.url:
            metadata["url"] = chunk.metadata.url
            
        metadatas.append(metadata)
    
    # 2. Inserción en lotes (recomendado para rendimiento)
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        batch_end = min(i + batch_size, len(ids))
        
        collection.add(
            ids=ids[i:batch_end],
            documents=documents[i:batch_end],
            embeddings=embeddings[i:batch_end],
            metadatas=metadatas[i:batch_end]
        )
    
    return True
```

### **Pipeline de Búsqueda Avanzada**
```python
def search_chromadb(query: str, k: int = 5, filters: Dict = None):
    """Búsqueda avanzada con filtros y ranking"""
    
    # 1. Generar embedding de consulta
    query_embedding = embedding_service.encode_single_text(query)
    
    # 2. Construir filtros WHERE
    where_clause = {}
    if filters:
        for key, value in filters.items():
            if isinstance(value, list):
                where_clause[key] = {"$in": value}
            elif isinstance(value, dict) and "range" in value:
                where_clause[key] = {
                    "$gte": value["range"][0],
                    "$lte": value["range"][1]
                }
            else:
                where_clause[key] = {"$eq": value}
    
    # 3. Ejecutar búsqueda
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k * 2,  # Obtener más para post-filtrado
        where=where_clause,
        include=["documents", "metadatas", "distances"]
    )
    
    # 4. Post-procesamiento y ranking
    chunks = []
    if results['documents'] and len(results['documents'][0]) > 0:
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        for i, (doc_text, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            # Reconstruir DocumentChunk
            doc_metadata = DocumentMetadata(
                source_path=metadata["source_path"],
                source_type=metadata["source_type"],
                file_type=metadata["file_type"],
                size_bytes=metadata["size_bytes"],
                created_at=datetime.fromisoformat(metadata["created_at"]),
                processed_at=datetime.fromisoformat(metadata["processed_at"]),
                checksum=metadata["checksum"],
                title=metadata.get("title", "")
            )
            
            chunk = DocumentChunk(
                id=metadata["chunk_id"],
                content=doc_text,
                metadata=doc_metadata,
                chunk_index=metadata["chunk_index"],
                chunk_size=metadata["chunk_size"],
                start_char=metadata["start_char"],
                end_char=metadata["end_char"]
            )
            
            # Añadir información de relevancia
            chunk.similarity_score = 1 - distance  # Convertir distancia a score
            chunk.rank = i + 1
            
            chunks.append(chunk)
    
    return chunks[:k]  # Devolver solo los k mejores
```

## 📊 **Casos de Uso para Administraciones Locales**

### **1. Búsqueda Cross-Domain**
```python
# Encontrar información sobre "licencias" en todos los tipos de fuente
results = search_chromadb(
    query="tramitación licencias municipales",
    k=10,
    filters=None  # Sin filtros = busca en todo
)

# Resultado típico:
# - Chunk 1: PDF oficial "Ordenanza de Licencias" (score: 0.95)
# - Chunk 2: Página web "Guía ciudadano" (score: 0.89) 
# - Chunk 3: API response "Procedimientos" (score: 0.87)
# - Chunk 4: BBDD "Histórico tramitaciones" (score: 0.82)
```

### **2. Búsqueda Filtrada por Departamento**
```python
# Solo información de urbanismo
urbanismo_results = search_chromadb(
    query="permisos construcción",
    k=5,
    filters={"department": "urbanismo"}
)

# Solo documentos oficiales recientes
oficiales_recientes = search_chromadb(
    query="nueva normativa municipal",
    k=3,
    filters={
        "source_type": "document",
        "publication_year": {"range": [2024, 2025]}
    }
)
```

### **3. Búsqueda Temporal**
```python
# Información actualizada en los últimos 6 meses
from datetime import datetime, timedelta
recent_date = (datetime.now() - timedelta(days=180)).isoformat()

recent_info = search_chromadb(
    query="cambios procedimientos administrativos",
    k=8,
    filters={
        "last_updated": {"range": [recent_date, datetime.now().isoformat()]}
    }
)
```

### **4. Búsqueda por Nivel de Confianza**
```python
# Solo fuentes oficiales (alta confianza)
official_sources = search_chromadb(
    query="requisitos documentación ciudadanos",
    k=5,
    filters={
        "source_type": {"$in": ["document", "database"]},
        "content_type": "regulation"
    }
)

# Incluir guías ciudadanas (confianza media)
citizen_guides = search_chromadb(
    query="cómo solicitar certificados",
    k=10,
    filters={
        "content_type": {"$in": ["regulation", "guide", "faq"]}
    }
)
```

## 🎓 **Beneficios para el TFM**

### **Ventajas Académicas de ChromaDB**
```
📊 PARA INVESTIGACIÓN:
✅ Metadatos ricos para análisis estadístico
✅ Queries SQL-like para investigación
✅ Trazabilidad completa de documentos
✅ Versionado automático de cambios

📊 PARA COMPARACIÓN:
✅ API más simple = menos variables confusas
✅ Resultados más interpretables
✅ Filtrado nativo vs post-procesamiento
✅ Gestión automática de IDs

📊 PARA PRODUCCIÓN:
✅ Base de datos real vs archivos
✅ Transacciones ACID
✅ Backup/restore integrado
✅ API REST para integración web
```

### **Métricas Específicas ChromaDB**
```python
# Métricas de rendimiento
{
    "insertion_throughput": "docs/second",
    "search_latency": "ms/query", 
    "memory_efficiency": "MB/1000_docs",
    "disk_usage": "MB total",
    
    # Métricas de funcionalidad
    "metadata_query_support": "boolean",
    "complex_filter_support": "boolean", 
    "transaction_support": "boolean",
    "backup_restore_support": "boolean",
    
    # Métricas de escalabilidad
    "max_vectors_tested": "integer",
    "concurrent_users_supported": "integer",
    "collection_management_overhead": "ms"
}
```

### **Casos de Estudio Propuestos**
```
🔬 EXPERIMENTO 1: Rendimiento vs Funcionalidad
   • Comparar velocidad FAISS vs facilidad ChromaDB
   • Medir overhead de metadatos complejos
   • Evaluar trade-offs desarrollo vs performance

🔬 EXPERIMENTO 2: Calidad de Resultados
   • Comparar precisión con/sin filtros de metadatos
   • Evaluar relevancia contextual por tipo de fuente
   • Medir impacto de normalización semántica

🔬 EXPERIMENTO 3: Escalabilidad Práctica
   • Testing con datasets crecientes (1K, 10K, 100K docs)
   • Evaluación de degradación de performance
   • Análisis de límites prácticos para administraciones
```

## 🎯 **Recomendaciones para el TFM**

### **Implementación Sugerida**
1. **Usar ChromaDB para prototipo inicial** - API más simple, desarrollo más rápido
2. **Colección única con filtros** - Máxima flexibilidad para comparación
3. **Metadatos ricos** - Aprovechar capacidad diferencial vs FAISS
4. **Benchmark exhaustivo** - Medir trade-offs funcionalidad vs rendimiento

### **Análisis Académico Recomendado**
1. **Comparación cuantitativa** - Velocidad, memoria, throughput
2. **Comparación cualitativa** - Facilidad desarrollo, mantenibilidad
3. **Casos de uso específicos** - Cuándo elegir cada tecnología
4. **Proyección futura** - Escalabilidad a largo plazo

### **Contribución Científica**
- **Framework de evaluación** para vector stores en administraciones
- **Métricas específicas** para sistemas RAG gubernamentales  
- **Análisis empírico** FAISS vs ChromaDB en casos reales
- **Buenas prácticas** para implementación en sector público

---

**Esta arquitectura ChromaDB proporciona la base técnica sólida necesaria para tu análisis comparativo en el TFM, combinando funcionalidad avanzada con capacidad de evaluación académica rigurosa.**