# Arquitectura FAISS para Prototipo_chatbot TFM

## 🎯 **Diseño Estratégico**

### **Opción A: Índice Único Global (Recomendado para TFM)**
```
📊 VENTAJAS:
✅ Búsqueda cross-domain (encuentra relaciones entre tipos)
✅ Más simple de mantener y optimizar
✅ Mejor para comparativas académicas (FAISS vs ChromaDB)
✅ Menor overhead de gestión
✅ Escalabilidad más directa

📊 ESTRUCTURA:
🔍 faiss_index.index (384 dimensiones)
├── Vector 0: Chunk PDF sobre "licencias municipales"
├── Vector 1: Chunk Web sobre "tramitación licencias"  
├── Vector 2: Chunk API sobre "normativa licencias"
└── Vector N: Cualquier tipo de contenido relacionado
```

### **Opción B: Múltiples Colecciones (Para casos específicos)**
```
📊 CASOS DE USO:
- Diferentes modelos de embedding por tipo
- Requisitos de seguridad por fuente
- Optimizaciones específicas por dominio

📊 ESTRUCTURA:
🔍 documents.index (PDFs, DOCX oficiales)
🔍 web.index (Páginas web públicas)  
🔍 api.index (Datos estructurados)
🔍 database.index (Información interna)
```

## 📊 **Datos que se Almacenarán**

### **1. En FAISS Index (.index file)**
```python
# Vector embeddings (numpy arrays)
embeddings = [
    [0.1, -0.3, 0.7, ...],  # 384 dimensiones
    [0.2, 0.1, -0.4, ...],  # Vector 2
    [...],                   # Vector N
]
```

### **2. Metadata Companion Files**
```python
# metadata.pkl - Información completa de chunks
{
    0: {
        'chunk_id': 'doc_123_chunk_5',
        'content': 'Texto del fragmento...',
        'source_type': 'document',
        'file_type': '.pdf',
        'source_path': '/docs/ordenanza_municipal.pdf',
        'page_number': 12,
        'section_title': 'Licencias de Actividad',
        'created_at': '2025-08-03T16:30:00',
        'word_count': 156,
        'chunk_index': 5,
        'start_char': 2340,
        'end_char': 2896
    },
    1: {
        'chunk_id': 'web_456_chunk_2', 
        'content': 'Información sobre tramitación...',
        'source_type': 'web',
        'url': 'https://ayuntamiento.es/licencias',
        'title': 'Guía de Licencias',
        'scraped_at': '2025-08-03T15:45:00',
        # ... más metadatos
    }
}

# id_mapping.pkl - Mapeo rápido
{
    'doc_123_chunk_5': 0,      # chunk_id -> faiss_index
    'web_456_chunk_2': 1,
    'api_789_chunk_1': 2,
    # ...
}
```

### **3. Configuración del Índice**
```yaml
# index_config.yaml
index_info:
  type: "IndexFlatL2"          # Tipo de índice FAISS
  dimension: 384               # Dimensión embeddings
  metric: "L2"                 # Métrica de distancia
  normalize_vectors: true     # Normalización activa

statistics:
  total_vectors: 15847
  total_sources: 342
  by_type:
    document: 8234             # PDFs, DOCX
    web: 4521                  # Páginas web
    api: 1987                  # Datos de APIs
    database: 1105             # Consultas BBDD
  
performance:
  last_build_time: 23.4       # Segundos
  index_size_mb: 245.8        # Tamaño en disco
  avg_search_time_ms: 12.3    # Tiempo promedio búsqueda

embedding_model:
  name: "all-MiniLM-L6-v2"
  version: "2.2.2"
  cache_hit_rate: 0.73
```

## 🎯 **Recomendación para tu TFM**

### **📊 Estrategia Híbrida Óptima:**

```python
class FaissVectorStore:
    def __init__(self):
        # ÍNDICE PRINCIPAL - Todo junto
        self.main_index = faiss.IndexFlatL2(384)
        
        # FILTROS VIRTUALES - Por metadatos
        self.filters = {
            'source_type': ['document', 'web', 'api', 'database'],
            'file_type': ['.pdf', '.docx', '.html'],
            'date_range': (start_date, end_date),
            'department': ['urbanismo', 'hacienda', 'servicios']
        }
```

### **💡 Ventajas de esta Aproximación:**

1. **🔍 Búsqueda Global**: 
   - Query: "licencias de apertura" 
   - Encuentra: PDFs oficiales + páginas web + datos API

2. **🎯 Filtrado Específico**:
   - Solo documentos: `search(query, filter={'source_type': 'document'})`
   - Solo web: `search(query, filter={'source_type': 'web'})`
   - Por fecha: `search(query, filter={'date_range': ('2024-01-01', '2025-01-01')})`

3. **📊 Comparativas TFM**:
   - FAISS vs ChromaDB con **mismos datos**
   - Benchmarks **directamente comparables**
   - Métricas **consistentes**

## 🚀 **Implementación Técnica**

### **Tipos de Índices FAISS a Usar:**

```python
# Para desarrollo y TFM (exacto, simple)
IndexFlatL2(384)              # Búsqueda exacta, ideal para benchmarks

# Para producción futura (aproximado, rápido)  
IndexIVFFlat(quantizer, 100, 384)  # 100 clusters
IndexHNSW(384, 32)            # Hierarchical NSW, muy rápido
```

### **Pipeline de Indexación:**

```python
def add_documents_to_faiss(chunks: List[DocumentChunk]):
    # 1. Generar embeddings (ya tenemos EmbeddingService ✅)
    embeddings = embedding_service.encode_documents(chunks)
    
    # 2. Añadir a FAISS index
    vectors = np.array([chunk.embedding for chunk in embeddings])
    faiss_index.add(vectors)
    
    # 3. Guardar metadatos
    update_metadata_store(chunks)
    
    # 4. Actualizar mapeos
    update_id_mapping(chunks)
    
    # 5. Persistir a disco
    save_index_to_disk()
```

### **Pipeline de Búsqueda:**

```python
def search(query: str, k: int = 5, filters: Dict = None):
    # 1. Embedding de query
    query_vector = embedding_service.encode_single_text(query)
    
    # 2. Búsqueda en FAISS
    distances, indices = faiss_index.search(query_vector, k * 2)
    
    # 3. Aplicar filtros por metadatos
    filtered_results = apply_metadata_filters(indices, filters)
    
    # 4. Recuperar chunks completos
    chunks = get_chunks_by_indices(filtered_results[:k])
    
    return chunks
```

## 🎓 **Beneficios para el TFM**

### **📊 Métricas Comparativas:**
- **Velocidad indexación**: FAISS vs ChromaDB
- **Velocidad búsqueda**: Queries/segundo
- **Uso memoria**: RAM durante operaciones
- **Precisión resultados**: Relevancia @k
- **Escalabilidad**: Rendimiento vs tamaño dataset

### **📈 Casos de Uso Académicos:**
- **Cross-domain search**: Buscar "licencias" encuentra PDF + Web + API
- **Filtered search**: Solo documentos oficiales
- **Temporal search**: Solo información reciente
- **Departmental search**: Solo contenido de urbanismo

### **🔬 Experimentos Posibles:**
- Comparar diferentes tipos de índices FAISS
- Medir impacto de normalización de vectores
- Evaluar calidad vs velocidad (exacto vs aproximado)
- Analizar distribución semántica por tipo de fuente

**¿Esta arquitectura te parece adecuada para tu TFM?** 

La propuesta combina **simplicidad para desarrollo** con **flexibilidad para análisis académico**, manteniendo la capacidad de hacer comparaciones justas con ChromaDB.