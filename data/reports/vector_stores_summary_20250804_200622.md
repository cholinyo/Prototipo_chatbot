# Comparación Empírica: FAISS vs ChromaDB

**Autor:** Vicente Caruncho Ramos  
**TFM:** Prototipo de Chatbot Interno para Administraciones Locales  
**Fecha:** 04/08/2025

## Metodología

- **Dataset:** 20 documentos de administración local
- **Queries:** 10 consultas representativas
- **Modelo embeddings:** all-MiniLM-L6-v2 (384 dimensiones)
- **Métricas:** Latencia, throughput, uso de recursos

## Resultados Principales

### Rendimiento de Inserción
- **FAISS:** 2861.7 docs/segundo
- **ChromaDB:** 128.3 docs/segundo
- **Ganador:** FAISS

### Rendimiento de Búsqueda
- **FAISS:** 0.000s promedio
- **ChromaDB:** 0.003s promedio  
- **Ganador:** FAISS

### Almacenamiento
- **FAISS:** 0.03 MB (Memory + Files)
- **ChromaDB:** 0.00 MB (Persistent DB)

## Conclusiones para Administraciones Locales

### Recomendaciones de Uso
1. FAISS superior para inserción masiva de datos
2. FAISS superior para búsquedas de alta velocidad
3. FAISS ideal para aplicaciones que priorizan velocidad
4. ChromaDB ideal para aplicaciones que requieren persistencia

### Implicaciones para el TFM

- **FAISS** se muestra superior para casos de uso que priorizan **velocidad de respuesta**
- **ChromaDB** ofrece ventajas en **persistencia** y **gestión de metadatos**
- Ambas tecnologías son **viables** para sistemas RAG en administraciones locales
- La elección depende de los **requisitos específicos** del caso de uso

### Trabajo Futuro

- Evaluación con datasets más grandes (>10,000 documentos)
- Análisis de calidad de resultados (relevancia, precisión)
- Pruebas con diferentes modelos de embeddings
- Evaluación de escalabilidad en producción
