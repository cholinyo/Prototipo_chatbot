# Guía de Benchmarking y Análisis - TFM Vector Stores

## 🎯 **Objetivo del Benchmarking**

### **Propósito Académico**
Esta guía detalla la metodología para realizar una comparación empírica rigurosa entre FAISS y ChromaDB en el contexto de sistemas RAG para administraciones locales, proporcionando datos cuantitativos y cualitativos para el análisis académico del TFM.

### **Preguntas de Investigación**
```
🔬 RQ1: ¿Cuál es la diferencia de rendimiento entre FAISS y ChromaDB 
        en términos de velocidad de indexación y búsqueda?

🔬 RQ2: ¿Cómo afecta el tamaño del dataset al rendimiento relativo 
        de ambas tecnologías?

🔬 RQ3: ¿Qué trade-offs existen entre funcionalidad avanzada 
        (metadatos, filtros) y rendimiento puro?

🔬 RQ4: ¿Cuál es más adecuado para diferentes casos de uso 
        en administraciones locales?

🔬 RQ5: ¿Cómo escalan ambas tecnologías con el crecimiento 
        del volumen de documentos?
```

## 📊 **Metodología de Benchmarking**

### **Configuración del Entorno de Pruebas**
```yaml
# Especificaciones del entorno
hardware:
  cpu: "Procesador del sistema de pruebas"
  ram: "Cantidad de RAM disponible"
  storage: "Tipo de almacenamiento (SSD/HDD)"
  os: "Windows 11 / macOS / Linux"

software:
  python: "3.9+"
  faiss: "cpu version"
  chromadb: "latest version"
  sentence_transformers: "2.2+"
  numpy: "latest"

dataset:
  documents: 20  # Inicial para testing
  documents_large: [50, 100, 500, 1000]  # Para escalabilidad
  embedding_model: "all-MiniLM-L6-v2"
  embedding_dimension: 384
  avg_document_length: "~150 palabras"
  domain: "Administración local española"
```

### **Dataset de Pruebas**
```python
# Documentos representativos de administraciones locales
benchmark_documents = [
    # Normativa y regulaciones
    "Administración electrónica y digitalización de servicios públicos municipales",
    "Procedimientos de licencias urbanísticas y de actividad en administraciones locales",
    "Normativa municipal sobre ordenanzas, reglamentos y disposiciones locales",
    
    # Procedimientos administrativos
    "Gestión de expedientes administrativos y tramitación de documentos oficiales",
    "Contratación pública, licitaciones y procedimientos administrativos",
    "Personal funcionario, laborales y organización administrativa municipal",
    
    # Servicios ciudadanos
    "Atención ciudadana presencial y telemática en ayuntamientos españoles",
    "Servicios sociales municipales y políticas de bienestar ciudadano",
    "Transparencia, participación ciudadana y gobierno abierto local",
    
    # Gestión financiera y tributaria
    "Hacienda local, presupuestos municipales y gestión financiera pública",
    "Tributos locales, tasas municipales y gestión tributaria",
    "Patrimonio municipal, bienes públicos y gestión del dominio público",
    
    # Urbanismo y territorio
    "Planificación urbanística, PGOU y gestión del territorio municipal",
    "Las licencias de obra deben tramitarse siguiendo procedimientos establecidos",
    "Medio ambiente, sostenibilidad y gestión de residuos urbanos",
    
    # Servicios municipales
    "Cultura, deportes y actividades de ocio en el ámbito municipal",
    "Educación municipal, escuelas infantiles y servicios educativos",
    "Sanidad pública local y servicios de salud comunitaria",
    "Transporte público urbano y movilidad sostenible municipal",
    "Seguridad ciudadana, policía local y protección civil municipal"
]

# Queries de prueba representativas
benchmark_queries = [
    "licencias y permisos municipales",
    "servicios digitales administración",
    "tramitación expedientes ciudadanos", 
    "normativa ordenanzas locales",
    "presupuestos hacienda municipal",
    "urbanismo planificación territorial",
    "medio ambiente sostenibilidad",
    "atención ciudadana servicios",
    "transparencia gobierno abierto",
    "personal funcionario municipal"
]
```

## 🧪 **Ejecución del Benchmarking**

### **Paso 1: Preparación del Entorno**
```powershell
# Verificar que el entorno esté correcto
python -c "import faiss, chromadb, numpy; print('✅ Dependencias OK')"

# Limpiar datos anteriores
Remove-Item -Recurse -Force "data/benchmark_*" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "data/reports" -ErrorAction SilentlyContinue

# Crear directorios necesarios
New-Item -ItemType Directory -Force -Path "data/reports"
New-Item -ItemType Directory -Force -Path "data/benchmark_results"
```

### **Paso 2: Ejecutar Benchmark**
```powershell
# Ejecutar el script de comparación completo
python comparison_faiss_vs_chromadb.py

# El script generará automáticamente:
# - data/reports/vector_stores_comparison_YYYYMMDD_HHMMSS.json
# - data/reports/vector_stores_summary_YYYYMMDD_HHMMSS.md
```

### **Paso 3: Verificar Resultados**
```powershell
# Verificar que se generaron los archivos
Get-ChildItem "data/reports" | Format-Table Name, Length, LastWriteTime

# Visualizar resumen rápido
Get-Content "data/reports/vector_stores_summary_*.md" | Select-Object -First 50
```

## 📈 **Métricas de Evaluación**

### **Métricas de Rendimiento**
```python
performance_metrics = {
    # Indexación/Inserción
    "insertion_metrics": {
        "throughput_docs_per_second": float,
        "total_insertion_time": float,
        "memory_usage_during_insertion": float,
        "disk_usage_after_insertion": float,
        
        # Por tamaño de lote
        "batch_5_docs": {"time": float, "throughput": float},
        "batch_10_docs": {"time": float, "throughput": float}, 
        "batch_20_docs": {"time": float, "throughput": float}
    },
    
    # Búsqueda/Retrieval  
    "search_metrics": {
        "avg_search_time_ms": float,
        "queries_per_second": float,
        "search_time_by_k": {
            "k_1": float,
            "k_3": float, 
            "k_5": float,
            "k_10": float
        },
        "memory_usage_during_search": float
    },
    
    # Escalabilidad
    "scalability_metrics": {
        "performance_by_dataset_size": {
            "20_docs": {"insertion": float, "search": float},
            "50_docs": {"insertion": float, "search": float},
            "100_docs": {"insertion": float, "search": float},
            "500_docs": {"insertion": float, "search": float}
        }
    }
}
```

### **Métricas de Funcionalidad**
```python
functionality_metrics = {
    # Capacidades de filtrado
    "filtering_support": {
        "basic_equality_filters": bool,
        "range_filters": bool, 
        "multiple_condition_filters": bool,
        "complex_nested_filters": bool,
        "filter_performance_impact": float  # % slowdown
    },
    
    # Gestión de metadatos
    "metadata_support": {
        "structured_metadata": bool,
        "nested_metadata": bool,
        "metadata_size_limit": int,
        "metadata_query_performance": float
    },
    
    # Persistencia y gestión
    "persistence_features": {
        "automatic_persistence": bool,
        "manual_save_load": bool,
        "transaction_support": bool,
        "concurrent_access": bool,
        "backup_restore": bool
    }
}
```

### **Métricas de Calidad**
```python
quality_metrics = {
    # Precisión de resultados
    "search_quality": {
        "relevance_at_k1": float,  # % resultados relevantes en posición 1
        "relevance_at_k3": float,  # % resultados relevantes en top 3
        "relevance_at_k5": float,  # % resultados relevantes en top 5
        "mean_reciprocal_rank": float,
        "consistency_across_runs": float  # Variación entre ejecuciones
    },
    
    # Efectividad de filtros
    "filter_effectiveness": {
        "filter_precision": float,  # % resultados que cumplen filtros
        "filter_recall": float,     # % resultados válidos encontrados
        "filter_impact_on_relevance": float
    }
}
```

## 📊 **Análisis de Resultados**

### **Análisis Cuantitativo**
```python
# Ejemplo de análisis automático generado
quantitative_analysis = {
    "performance_winner": {
        "insertion": "FAISS/ChromaDB",
        "search": "FAISS