Variables del Dashboard por Implementar:

rag_stats - Estadísticas del sistema RAG

vector_store_type: Tipo de vector store actual
total_chunks: Total de fragmentos indexados
last_update: Última actualización del índice


ingestion_stats - Estadísticas de ingesta

jobs_completed: Trabajos completados
jobs_active: Trabajos en progreso
jobs_failed: Trabajos fallidos
total_documents_processed: Total documentos procesados


charts_data - Datos reales para gráficos

usage_over_time: Datos históricos de uso
model_usage: Distribución uso modelos
response_times: Tiempos de respuesta reales



Implementación requerida en run.py:

Añadir estas variables al contexto del dashboard
Conectar con métricas reales del sistema
Implementar tracking de estadísticas de uso

Prioridad: Media - Funciona con valores por defecto pero mejorará la demo final del TFM.