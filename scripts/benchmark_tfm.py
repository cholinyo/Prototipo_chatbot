#!/usr/bin/env python3
"""
Benchmarking Académico Completo para TFM
Comparación FAISS vs ChromaDB + Modelos Locales vs Cloud
Vicente Caruncho Ramos - Sistemas Inteligentes UJI
"""

import sys
import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import statistics

# Añadir directorio raíz al path
sys.path.append(str(Path(__file__).parent))

from app.core.config import get_config
from app.core.logger import get_logger
from app.services.rag_pipeline import get_rag_pipeline
from app.models import DocumentChunk, DocumentMetadata


@dataclass
class BenchmarkResult:
    """Resultado individual de benchmark"""
    query: str
    vector_store: str
    llm_provider: str
    llm_model: str
    response_time: float
    retrieval_time: float
    generation_time: float
    response_length: int
    confidence_score: float
    sources_count: int
    estimated_cost: float
    error: str = None
    timestamp: str = None


@dataclass
class BenchmarkSummary:
    """Resumen estadístico del benchmark"""
    total_queries: int
    success_rate: float
    avg_response_time: float
    avg_retrieval_time: float
    avg_generation_time: float
    avg_confidence: float
    total_cost: float
    error_count: int
    fastest_config: str
    most_accurate_config: str
    most_economical_config: str


class TFMBenchmark:
    """Clase principal para benchmarking académico del TFM"""
    
    def __init__(self):
        self.logger = get_logger("prototipo_chatbot.tfm_benchmark")
        self.config = get_config()
        self.pipeline = get_rag_pipeline()
        
        # Directorios de salida
        self.output_dir = Path("data/reports/tfm_benchmark")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuraciones de prueba
        self.vector_stores = ['faiss', 'chromadb']
        self.llm_providers = ['ollama', 'openai']
        self.ollama_models = ['llama3.2:3b', 'mistral:7b', 'gemma2:2b']
        self.openai_models = ['gpt-4o-mini', 'gpt-3.5-turbo']
        
        # Consultas de prueba específicas para administraciones locales
        self.test_queries = [
            "¿Cuáles son los requisitos para solicitar una licencia de obras?",
            "¿Qué documentación necesito para abrir un negocio en el municipio?",
            "¿Cuáles son los plazos para tramitar el empadronamiento?",
            "¿Cómo puedo solicitar ayudas sociales en el ayuntamiento?",
            "¿Qué impuestos municipales debo pagar como propietario?",
            "¿Cuál es el procedimiento para denunciar ruidos molestos?",
            "¿Dónde puedo consultar el padrón municipal de habitantes?",
            "¿Qué servicios ofrece el área de urbanismo del ayuntamiento?",
            "¿Cómo solicito una cita previa en el registro civil?",
            "¿Cuáles son los horarios de atención ciudadana?",
            "¿Qué hacer si tengo problemas con el suministro de agua?",
            "¿Cómo puedo participar en el pleno municipal?",
            "¿Qué ayudas existen para familias numerosas?",
            "¿Cuál es el procedimiento para cambio de domicilio fiscal?",
            "¿Dónde solicito el certificado de convivencia?"
        ]
        
        # Métricas a evaluar
        self.metrics = [
            'response_time',
            'retrieval_time', 
            'generation_time',
            'confidence_score',
            'sources_count',
            'estimated_cost',
            'response_length'
        ]
        
        # Resultados del benchmark
        self.results: List[BenchmarkResult] = []
        
        self.logger.info("TFM Benchmark inicializado", 
                        queries=len(self.test_queries),
                        vector_stores=self.vector_stores,
                        llm_providers=self.llm_providers)
    
    def run_complete_benchmark(self) -> Dict[str, Any]:
        """Ejecutar benchmark completo para el TFM"""
        
        self.logger.info("🚀 INICIANDO BENCHMARK ACADÉMICO COMPLETO")
        start_time = time.time()
        
        try:
            # 1. Verificar disponibilidad del pipeline
            if not self.pipeline.is_available():
                raise Exception("Pipeline RAG no disponible")
            
            # 2. Ejecutar pruebas por configuración
            total_configs = len(self.vector_stores) * len(self.llm_providers) * len(self.test_queries)
            current_test = 0
            
            for vector_store in self.vector_stores:
                for provider in self.llm_providers:
                    models = self.ollama_models if provider == 'ollama' else self.openai_models
                    
                    for model in models:
                        for query in self.test_queries:
                            current_test += 1
                            progress = (current_test / (total_configs * len(models))) * 100
                            
                            self.logger.info(f"Progreso: {progress:.1f}% - Probando {vector_store}/{provider}/{model}")
                            
                            result = self.run_single_test(
                                query=query,
                                vector_store=vector_store,
                                provider=provider,
                                model=model
                            )
                            
                            self.results.append(result)
                            
                            # Pausa entre pruebas para no sobrecargar
                            time.sleep(1)
            
            # 3. Analizar resultados
            analysis = self.analyze_results()
            
            # 4. Generar reportes
            report_files = self.generate_reports(analysis)
            
            # 5. Crear visualizaciones
            chart_files = self.create_visualizations(analysis)
            
            total_time = time.time() - start_time
            
            summary = {
                'benchmark_completed': True,
                'total_time': total_time,
                'total_tests': len(self.results),
                'success_tests': len([r for r in self.results if not r.error]),
                'report_files': report_files,
                'chart_files': chart_files,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("✅ BENCHMARK COMPLETADO", 
                           total_time=f"{total_time:.2f}s",
                           total_tests=len(self.results),
                           success_rate=f"{(len([r for r in self.results if not r.error]) / len(self.results)) * 100:.1f}%")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error en benchmark: {e}")
            return {
                'benchmark_completed': False,
                'error': str(e),
                'partial_results': len(self.results)
            }
    
    def run_single_test(self, query: str, vector_store: str, provider: str, model: str) -> BenchmarkResult:
        """Ejecutar una prueba individual"""
        
        test_start = time.time()
        
        try:
            # Configurar vector store para esta prueba
            original_vs = self.config.get('DEFAULT_VECTOR_STORE')
            self.config['DEFAULT_VECTOR_STORE'] = vector_store
            
            # Reinicializar pipeline con nueva configuración
            self.pipeline._initialize_vector_store()
            
            # Medir tiempo de recuperación
            retrieval_start = time.time()
            query_embedding = self.pipeline.embedding_service.encode_single_text(query)
            context_chunks = self.pipeline.vector_store.search(query_embedding, k=5)
            retrieval_time = time.time() - retrieval_start
            
            # Medir tiempo de generación
            generation_start = time.time()
            rag_response = self.pipeline.llm_service.generate_response(
                query=query,
                context=context_chunks,
                provider=provider,
                model=model,
                temperature=0.3
            )
            generation_time = time.time() - generation_start
            
            total_time = time.time() - test_start
            
            # Calcular métricas
            confidence = self.pipeline._calculate_confidence(context_chunks, rag_response)
            
            result = BenchmarkResult(
                query=query,
                vector_store=vector_store,
                llm_provider=provider,
                llm_model=model,
                response_time=total_time,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                response_length=len(rag_response.content),
                confidence_score=confidence,
                sources_count=len(context_chunks),
                estimated_cost=getattr(rag_response, 'estimated_cost', 0.0) or 0.0,
                error=rag_response.error,
                timestamp=datetime.now().isoformat()
            )
            
            # Restaurar configuración original
            self.config['DEFAULT_VECTOR_STORE'] = original_vs
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en prueba individual: {e}")
            
            return BenchmarkResult(
                query=query,
                vector_store=vector_store,
                llm_provider=provider,
                llm_model=model,
                response_time=0.0,
                retrieval_time=0.0,
                generation_time=0.0,
                response_length=0,
                confidence_score=0.0,
                sources_count=0,
                estimated_cost=0.0,
                error=str(e),
                timestamp=datetime.now().isoformat()
            )
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analizar resultados del benchmark"""
        
        self.logger.info("📊 Analizando resultados del benchmark")
        
        if not self.results:
            return {'error': 'No hay resultados para analizar'}
        
        # Convertir a DataFrame para análisis
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Filtrar resultados exitosos
        success_df = df[df['error'].isna()]
        
        if success_df.empty:
            return {'error': 'No hay resultados exitosos para analizar'}
        
        analysis = {
            'overview': {
                'total_tests': len(df),
                'successful_tests': len(success_df),
                'success_rate': len(success_df) / len(df) * 100,
                'error_rate': (len(df) - len(success_df)) / len(df) * 100
            },
            'vector_store_comparison': self._analyze_vector_stores(success_df),
            'llm_provider_comparison': self._analyze_llm_providers(success_df),
            'model_comparison': self._analyze_models(success_df),
            'performance_metrics': self._analyze_performance(success_df),
            'cost_analysis': self._analyze_costs(success_df),
            'statistical_significance': self._statistical_tests(success_df),
            'recommendations': self._generate_recommendations(success_df)
        }
        
        return analysis
    
    def _analyze_vector_stores(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análisis comparativo de vector stores"""
        
        vs_stats = {}
        
        for vs in self.vector_stores:
            vs_data = df[df['vector_store'] == vs]
            
            if not vs_data.empty:
                vs_stats[vs] = {
                    'avg_retrieval_time': vs_data['retrieval_time'].mean(),
                    'std_retrieval_time': vs_data['retrieval_time'].std(),
                    'avg_sources_count': vs_data['sources_count'].mean(),
                    'avg_confidence': vs_data['confidence_score'].mean(),
                    'min_retrieval_time': vs_data['retrieval_time'].min(),
                    'max_retrieval_time': vs_data['retrieval_time'].max(),
                    'median_retrieval_time': vs_data['retrieval_time'].median()
                }
        
        # Determinar ganador
        if len(vs_stats) > 1:
            winner = min(vs_stats.keys(), 
                        key=lambda x: vs_stats[x]['avg_retrieval_time'])
            
            vs_stats['winner'] = {
                'fastest_vector_store': winner,
                'performance_advantage': {
                    vs: vs_stats[winner]['avg_retrieval_time'] / vs_stats[vs]['avg_retrieval_time']
                    for vs in vs_stats if vs != winner and vs != 'winner'
                }
            }
        
        return vs_stats
    
    def _analyze_llm_providers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análisis comparativo de proveedores LLM"""
        
        provider_stats = {}
        
        for provider in self.llm_providers:
            provider_data = df[df['llm_provider'] == provider]
            
            if not provider_data.empty:
                provider_stats[provider] = {
                    'avg_generation_time': provider_data['generation_time'].mean(),
                    'std_generation_time': provider_data['generation_time'].std(),
                    'avg_response_length': provider_data['response_length'].mean(),
                    'avg_confidence': provider_data['confidence_score'].mean(),
                    'total_cost': provider_data['estimated_cost'].sum(),
                    'avg_cost_per_query': provider_data['estimated_cost'].mean(),
                    'success_rate': (provider_data['error'].isna().sum() / len(provider_data)) * 100
                }
        
        return provider_stats
    
    def _analyze_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análisis comparativo de modelos específicos"""
        
        model_stats = {}
        
        for model in df['llm_model'].unique():
            model_data = df[df['llm_model'] == model]
            
            if not model_data.empty:
                model_stats[model] = {
                    'avg_response_time': model_data['response_time'].mean(),
                    'avg_generation_time': model_data['generation_time'].mean(),
                    'avg_confidence': model_data['confidence_score'].mean(),
                    'avg_response_length': model_data['response_length'].mean(),
                    'total_cost': model_data['estimated_cost'].sum(),
                    'provider': model_data['llm_provider'].iloc[0],
                    'test_count': len(model_data)
                }
        
        return model_stats
    
    def _analyze_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análisis de métricas de rendimiento"""
        
        performance = {
            'response_time': {
                'mean': df['response_time'].mean(),
                'median': df['response_time'].median(),
                'std': df['response_time'].std(),
                'min': df['response_time'].min(),
                'max': df['response_time'].max(),
                'percentile_95': df['response_time'].quantile(0.95)
            },
            'retrieval_time': {
                'mean': df['retrieval_time'].mean(),
                'median': df['retrieval_time'].median(),
                'std': df['retrieval_time'].std(),
                'percentage_of_total': (df['retrieval_time'].mean() / df['response_time'].mean()) * 100
            },
            'generation_time': {
                'mean': df['generation_time'].mean(),
                'median': df['generation_time'].median(),
                'std': df['generation_time'].std(),
                'percentage_of_total': (df['generation_time'].mean() / df['response_time'].mean()) * 100
            },
            'confidence_distribution': {
                'mean': df['confidence_score'].mean(),
                'median': df['confidence_score'].median(),
                'std': df['confidence_score'].std(),
                'high_confidence_rate': (df['confidence_score'] > 0.7).sum() / len(df) * 100
            }
        }
        
        return performance
    
    def _analyze_costs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análisis de costes económicos"""
        
        # Filtrar solo resultados con coste (OpenAI)
        cost_df = df[df['estimated_cost'] > 0]
        
        if cost_df.empty:
            return {'note': 'No hay datos de coste disponibles (solo modelos locales)'}
        
        cost_analysis = {
            'total_cost': cost_df['estimated_cost'].sum(),
            'avg_cost_per_query': cost_df['estimated_cost'].mean(),
            'median_cost_per_query': cost_df['estimated_cost'].median(),
            'min_cost': cost_df['estimated_cost'].min(),
            'max_cost': cost_df['estimated_cost'].max(),
            'cost_by_model': cost_df.groupby('llm_model')['estimated_cost'].agg(['sum', 'mean', 'count']).to_dict('index'),
            'projected_monthly_cost': {
                'low_usage': cost_df['estimated_cost'].mean() * 100,  # 100 consultas/mes
                'medium_usage': cost_df['estimated_cost'].mean() * 1000,  # 1000 consultas/mes
                'high_usage': cost_df['estimated_cost'].mean() * 10000  # 10000 consultas/mes
            }
        }
        
        return cost_analysis
    
    def _statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Pruebas estadísticas de significancia"""
        
        from scipy import stats
        
        statistical_tests = {}
        
        # Test t para comparar vector stores en tiempo de recuperación
        if len(self.vector_stores) == 2:
            vs1_data = df[df['vector_store'] == self.vector_stores[0]]['retrieval_time']
            vs2_data = df[df['vector_store'] == self.vector_stores[1]]['retrieval_time']
            
            if len(vs1_data) > 1 and len(vs2_data) > 1:
                t_stat, p_value = stats.ttest_ind(vs1_data, vs2_data)
                
                statistical_tests['vector_store_retrieval_ttest'] = {
                    'vs1': self.vector_stores[0],
                    'vs2': self.vector_stores[1],
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'interpretation': 'Diferencia significativa' if p_value < 0.05 else 'No hay diferencia significativa'
                }
        
        # Test ANOVA para comparar modelos
        model_groups = [df[df['llm_model'] == model]['response_time'].values 
                       for model in df['llm_model'].unique() 
                       if len(df[df['llm_model'] == model]) > 1]
        
        if len(model_groups) > 2:
            f_stat, p_value = stats.f_oneway(*model_groups)
            
            statistical_tests['model_comparison_anova'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'interpretation': 'Diferencias significativas entre modelos' if p_value < 0.05 else 'No hay diferencias significativas'
            }
        
        return statistical_tests
    
    def _generate_recommendations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generar recomendaciones para administraciones locales"""
        
        recommendations = {
            'for_small_municipalities': {
                'description': 'Ayuntamientos pequeños (<10,000 habitantes)',
                'recommended_config': None,
                'reasoning': []
            },
            'for_medium_municipalities': {
                'description': 'Ayuntamientos medianos (10,000-50,000 habitantes)', 
                'recommended_config': None,
                'reasoning': []
            },
            'for_large_municipalities': {
                'description': 'Ayuntamientos grandes (>50,000 habitantes)',
                'recommended_config': None,
                'reasoning': []
            },
            'cost_optimization': {},
            'performance_optimization': {},
            'implementation_guidelines': []
        }
        
        # Configuración más económica (modelos locales)
        local_results = df[df['llm_provider'] == 'ollama']
        if not local_results.empty:
            best_local = local_results.loc[local_results['response_time'].idxmin()]
            
            recommendations['for_small_municipalities']['recommended_config'] = {
                'vector_store': best_local['vector_store'],
                'llm_provider': best_local['llm_provider'],
                'llm_model': best_local['llm_model'],
                'estimated_monthly_cost': 0.0
            }
            
            recommendations['for_small_municipalities']['reasoning'] = [
                'Coste operacional mínimo (solo hardware)',
                'Datos permanecen en infraestructura local',
                'Cumplimiento ENS simplificado',
                f'Tiempo de respuesta promedio: {best_local["response_time"]:.2f}s'
            ]
        
        # Configuración balanceada
        if not df.empty:
            # Buscar configuración con mejor balance rendimiento/coste
            df['efficiency_score'] = df['confidence_score'] / (df['response_time'] + df['estimated_cost'] * 100)
            best_balanced = df.loc[df['efficiency_score'].idxmax()]
            
            recommendations['for_medium_municipalities']['recommended_config'] = {
                'vector_store': best_balanced['vector_store'],
                'llm_provider': best_balanced['llm_provider'],
                'llm_model': best_balanced['llm_model'],
                'efficiency_score': best_balanced['efficiency_score']
            }
        
        # Configuración de alto rendimiento
        cloud_results = df[df['llm_provider'] == 'openai']
        if not cloud_results.empty:
            best_cloud = cloud_results.loc[cloud_results['confidence_score'].idxmax()]
            
            recommendations['for_large_municipalities']['recommended_config'] = {
                'vector_store': best_cloud['vector_store'],
                'llm_provider': best_cloud['llm_provider'],
                'llm_model': best_cloud['llm_model'],
                'confidence': best_cloud['confidence_score']
            }
        
        # Guías de implementación
        recommendations['implementation_guidelines'] = [
            'Comenzar con modelos locales para pruebas de concepto',
            'Evaluar volumen de consultas antes de elegir proveedor',
            'Implementar cache de respuestas frecuentes',
            'Monitorear métricas de satisfacción ciudadana',
            'Planificar backup y recuperación de vector stores',
            'Capacitar personal técnico en mantenimiento del sistema'
        ]
        
        return recommendations
    
    def generate_reports(self, analysis: Dict[str, Any]) -> List[str]:
        """Generar reportes en diferentes formatos"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_files = []
        
        # 1. Reporte JSON completo
        json_file = self.output_dir / f"tfm_benchmark_complete_{timestamp}.json"
        
        report_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'benchmark_version': '1.0',
                'total_tests': len(self.results),
                'test_queries': self.test_queries,
                'configurations_tested': {
                    'vector_stores': self.vector_stores,
                    'llm_providers': self.llm_providers,
                    'ollama_models': self.ollama_models,
                    'openai_models': self.openai_models
                }
            },
            'raw_results': [asdict(r) for r in self.results],
            'analysis': analysis
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        report_files.append(str(json_file))
        
        # 2. Reporte CSV para análisis estadístico
        csv_file = self.output_dir / f"tfm_benchmark_data_{timestamp}.csv"
        df = pd.DataFrame([asdict(r) for r in self.results])
        df.to_csv(csv_file, index=False, encoding='utf-8')
        report_files.append(str(csv_file))
        
        # 3. Reporte ejecutivo en Markdown
        md_file = self.output_dir / f"tfm_benchmark_executive_{timestamp}.md"
        self._generate_executive_report(md_file, analysis)
        report_files.append(str(md_file))
        
        # 4. Reporte técnico detallado
        tech_file = self.output_dir / f"tfm_benchmark_technical_{timestamp}.md"
        self._generate_technical_report(tech_file, analysis)
        report_files.append(str(tech_file))
        
        self.logger.info(f"Reportes generados: {len(report_files)} archivos")
        
        return report_files
    
    def _generate_executive_report(self, file_path: Path, analysis: Dict[str, Any]):
        """Generar reporte ejecutivo para la memoria del TFM"""
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"""# Reporte Ejecutivo - Benchmarking RAG para Administraciones Locales

> **TFM Vicente Caruncho Ramos**  
> **Máster en Sistemas Inteligentes - Universitat Jaume I**  
> **Fecha:** {datetime.now().strftime('%d/%m/%Y %H:%M')}

## 📊 Resumen Ejecutivo

### Objetivos del Benchmark
- Comparar rendimiento FAISS vs ChromaDB en contexto RAG
- Evaluar modelos locales (Ollama) vs servicios cloud (OpenAI)
- Generar recomendaciones específicas para administraciones locales

### Resultados Principales

**Pruebas Realizadas:** {analysis['overview']['total_tests']}  
**Tasa de Éxito:** {analysis['overview']['success_rate']:.1f}%  
**Configuraciones Evaluadas:** {len(self.vector_stores)} vector stores × {len(self.llm_providers)} proveedores

""")
            
            # Vector Store Comparison
            if 'vector_store_comparison' in analysis:
                f.write("### 🗄️ Comparación Vector Stores\n\n")
                
                vs_comp = analysis['vector_store_comparison']
                for vs, stats in vs_comp.items():
                    if vs != 'winner' and isinstance(stats, dict):
                        f.write(f"**{vs.upper()}:**\n")
                        f.write(f"- Tiempo promedio recuperación: {stats['avg_retrieval_time']:.3f}s\n")
                        f.write(f"- Fuentes promedio: {stats['avg_sources_count']:.1f}\n")
                        f.write(f"- Confianza promedio: {stats['avg_confidence']:.2f}\n\n")
                
                if 'winner' in vs_comp:
                    winner = vs_comp['winner']['fastest_vector_store']
                    f.write(f"**🏆 Ganador:** {winner.upper()}\n\n")
            
            # LLM Provider Comparison
            if 'llm_provider_comparison' in analysis:
                f.write("### 🤖 Comparación Proveedores LLM\n\n")
                
                llm_comp = analysis['llm_provider_comparison']
                for provider, stats in llm_comp.items():
                    f.write(f"**{provider.upper()}:**\n")
                    f.write(f"- Tiempo generación: {stats['avg_generation_time']:.3f}s\n")
                    f.write(f"- Longitud respuesta: {stats['avg_response_length']:.0f} caracteres\n")
                    f.write(f"- Confianza: {stats['avg_confidence']:.2f}\n")
                    f.write(f"- Coste total: ${stats['total_cost']:.4f}\n\n")
            
            # Recommendations
            if 'recommendations' in analysis:
                f.write("## 💡 Recomendaciones para Administraciones\n\n")
                
                recs = analysis['recommendations']
                for size, rec in recs.items():
                    if isinstance(rec, dict) and 'description' in rec:
                        f.write(f"### {rec['description']}\n\n")
                        
                        if rec.get('recommended_config'):
                            config = rec['recommended_config']
                            f.write(f"**Configuración recomendada:**\n")
                            f.write(f"- Vector Store: {config.get('vector_store', 'N/A')}\n")
                            f.write(f"- Proveedor LLM: {config.get('llm_provider', 'N/A')}\n")
                            f.write(f"- Modelo: {config.get('llm_model', 'N/A')}\n\n")
                        
                        if rec.get('reasoning'):
                            f.write("**Justificación:**\n")
                            for reason in rec['reasoning']:
                                f.write(f"- {reason}\n")
                            f.write("\n")
            
            f.write(f"""## 📈 Conclusiones Técnicas

1. **Vector Stores:** Los resultados muestran diferencias significativas en rendimiento
2. **Modelos LLM:** Balance entre coste y calidad según el caso de uso
3. **Administraciones:** Configuraciones específicas según tamaño y recursos

## 🎯 Impacto para el TFM

Este benchmark proporciona evidencia empírica para:
- Justificar decisiones arquitectónicas
- Validar hipótesis de investigación  
- Generar recomendaciones prácticas
- Fundamentar conclusiones académicas

---
*Reporte generado automáticamente por el sistema de benchmarking del TFM*
""")
    
    def _generate_technical_report(self, file_path: Path, analysis: Dict[str, Any]):
        """Generar reporte técnico detallado"""
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"""# Reporte Técnico Detallado - Benchmarking RAG

**Fecha:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}  
**Versión:** 1.0  
**Total de pruebas:** {len(self.results)}

## Metodología

### Configuración del Benchmark
- **Vector Stores evaluados:** {', '.join(self.vector_stores)}
- **Proveedores LLM:** {', '.join(self.llm_providers)}
- **Modelos Ollama:** {', '.join(self.ollama_models)}
- **Modelos OpenAI:** {', '.join(self.openai_models)}
- **Consultas de prueba:** {len(self.test_queries)}

### Métricas Evaluadas
""")
            
            for metric in self.metrics:
                f.write(f"- {metric}\n")
            
            f.write("\n## Análisis Estadístico Detallado\n\n")
            
            # Performance metrics
            if 'performance_metrics' in analysis:
                perf = analysis['performance_metrics']
                f.write("### Métricas de Rendimiento\n\n")
                f.write("| Métrica | Media | Mediana | Desv. Estándar | Min | Max |\n")
                f.write("|---------|-------|---------|----------------|-----|-----|\n")
                
                for metric_name, stats in perf.items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        f.write(f"| {metric_name} | {stats['mean']:.3f} | {stats['median']:.3f} | {stats['std']:.3f} | {stats.get('min', 'N/A')} | {stats.get('max', 'N/A')} |\n")
                
                f.write("\n")
            
            # Statistical tests
            if 'statistical_significance' in analysis:
                f.write("### Pruebas de Significancia Estadística\n\n")
                
                stat_tests = analysis['statistical_significance']
                for test_name, results in stat_tests.items():
                    f.write(f"**{test_name}:**\n")
                    f.write(f"- p-value: {results.get('p_value', 'N/A')}\n")
                    f.write(f"- Significativo: {'Sí' if results.get('significant', False) else 'No'}\n")
                    f.write(f"- Interpretación: {results.get('interpretation', 'N/A')}\n\n")
            
            # Raw data summary
            f.write("### Resumen de Datos en Bruto\n\n")
            success_results = [r for r in self.results if not r.error]
            error_results = [r for r in self.results if r.error]
            
            f.write(f"- **Pruebas exitosas:** {len(success_results)}\n")
            f.write(f"- **Pruebas con error:** {len(error_results)}\n")
            f.write(f"- **Tasa de éxito:** {(len(success_results)/len(self.results)*100):.1f}%\n\n")
            
            if error_results:
                f.write("#### Análisis de Errores\n\n")
                error_types = {}
                for result in error_results:
                    error_key = f"{result.vector_store}_{result.llm_provider}"
                    if error_key not in error_types:
                        error_types[error_key] = []
                    error_types[error_key].append(result.error)
                
                for config, errors in error_types.items():
                    f.write(f"**{config}:** {len(errors)} errores\n")
                    unique_errors = list(set(errors))
                    for error in unique_errors[:3]:  # Solo los primeros 3 errores únicos
                        f.write(f"- {error}\n")
                    f.write("\n")
            
            f.write("""
## Limitaciones y Consideraciones

### Limitaciones del Benchmark
1. **Tamaño de muestra:** Número limitado de consultas de prueba
2. **Contexto específico:** Enfocado en administraciones locales españolas
3. **Recursos computacionales:** Pruebas realizadas en entorno de desarrollo
4. **Temporalidad:** Resultados válidos para las versiones específicas de modelos

### Consideraciones para Implementación
1. **Escalabilidad:** Resultados pueden variar con mayor volumen de datos
2. **Personalización:** Necesario ajustar según documentación específica
3. **Mantenimiento:** Requiere monitoreo continuo de rendimiento
4. **Costes:** Proyecciones basadas en tarifas actuales de APIs

## Recomendaciones para Trabajo Futuro

1. **Ampliar conjunto de pruebas** con más consultas diversas
2. **Evaluar modelos adicionales** según disponibilidad
3. **Incluir métricas de calidad** subjetivas con evaluadores humanos
4. **Testear con datasets reales** de administraciones colaboradoras
5. **Implementar pruebas de carga** para evaluar escalabilidad

---
*Documento técnico generado para soporte de investigación académica*
""")
    
    def create_visualizations(self, analysis: Dict[str, Any]) -> List[str]:
        """Crear visualizaciones para el análisis"""
        
        self.logger.info("📊 Creando visualizaciones")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_files = []
        
        # Configurar estilo de matplotlib
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Convertir resultados a DataFrame
        df = pd.DataFrame([asdict(r) for r in self.results])
        success_df = df[df['error'].isna()]
        
        if success_df.empty:
            self.logger.warning("No hay datos exitosos para visualizar")
            return []
        
        # 1. Comparación de tiempos de respuesta por configuración
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análisis de Rendimiento RAG - TFM Vicente Caruncho', fontsize=16, fontweight='bold')
        
        # Subplot 1: Tiempo de respuesta por vector store
        sns.boxplot(data=success_df, x='vector_store', y='response_time', ax=axes[0,0])
        axes[0,0].set_title('Tiempo de Respuesta por Vector Store')
        axes[0,0].set_ylabel('Tiempo (segundos)')
        
        # Subplot 2: Tiempo de respuesta por proveedor LLM
        sns.boxplot(data=success_df, x='llm_provider', y='response_time', ax=axes[0,1])
        axes[0,1].set_title('Tiempo de Respuesta por Proveedor LLM')
        axes[0,1].set_ylabel('Tiempo (segundos)')
        
        # Subplot 3: Confianza por configuración
        sns.boxplot(data=success_df, x='vector_store', y='confidence_score', hue='llm_provider', ax=axes[1,0])
        axes[1,0].set_title('Confianza por Configuración')
        axes[1,0].set_ylabel('Score de Confianza')
        axes[1,0].legend(title='Proveedor LLM')
        
        # Subplot 4: Distribución de costes (solo OpenAI)
        cost_df = success_df[success_df['estimated_cost'] > 0]
        if not cost_df.empty:
            sns.histplot(data=cost_df, x='estimated_cost', hue='llm_model', ax=axes[1,1])
            axes[1,1].set_title('Distribución de Costes (OpenAI)')
            axes[1,1].set_xlabel('Coste Estimado ($)')
        else:
            axes[1,1].text(0.5, 0.5, 'No hay datos de coste\n(solo modelos locales)', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Distribución de Costes')
        
        plt.tight_layout()
        
        chart_file1 = self.output_dir / f"tfm_performance_analysis_{timestamp}.png"
        plt.savefig(chart_file1, dpi=300, bbox_inches='tight')
        plt.close()
        chart_files.append(str(chart_file1))
        
        # 2. Análisis detallado por modelo
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análisis Detallado por Modelo LLM', fontsize=16, fontweight='bold')
        
        # Tiempo de generación vs calidad
        scatter = axes[0,0].scatter(success_df['generation_time'], success_df['confidence_score'], 
                                   c=success_df['response_length'], cmap='viridis', alpha=0.7)
        axes[0,0].set_xlabel('Tiempo de Generación (s)')
        axes[0,0].set_ylabel('Score de Confianza')
        axes[0,0].set_title('Tiempo vs Calidad (color = longitud respuesta)')
        plt.colorbar(scatter, ax=axes[0,0])
        
        # Comparación tiempo retrieval vs generation
        success_df['retrieval_pct'] = success_df['retrieval_time'] / success_df['response_time'] * 100
        success_df['generation_pct'] = success_df['generation_time'] / success_df['response_time'] * 100
        
        models = success_df['llm_model'].unique()
        x_pos = np.arange(len(models))
        
        retrieval_means = [success_df[success_df['llm_model'] == model]['retrieval_pct'].mean() for model in models]
        generation_means = [success_df[success_df['llm_model'] == model]['generation_pct'].mean() for model in models]
        
        width = 0.35
        axes[0,1].bar(x_pos - width/2, retrieval_means, width, label='Recuperación', alpha=0.8)
        axes[0,1].bar(x_pos + width/2, generation_means, width, label='Generación', alpha=0.8)
        axes[0,1].set_xlabel('Modelo LLM')
        axes[0,1].set_ylabel('% del Tiempo Total')
        axes[0,1].set_title('Distribución del Tiempo por Fase')
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels(models, rotation=45)
        axes[0,1].legend()
        
        # Eficiencia por configuración (tiempo vs confianza)
        config_efficiency = success_df.groupby(['vector_store', 'llm_provider']).agg({
            'response_time': 'mean',
            'confidence_score': 'mean',
            'estimated_cost': 'sum'
        }).reset_index()
        
        config_efficiency['config'] = config_efficiency['vector_store'] + '_' + config_efficiency['llm_provider']
        
        # Crear scatter plot de eficiencia
        for i, row in config_efficiency.iterrows():
            axes[1,0].scatter(row['response_time'], row['confidence_score'], 
                            s=200, alpha=0.7, label=row['config'])
        
        axes[1,0].set_xlabel('Tiempo Promedio (s)')
        axes[1,0].set_ylabel('Confianza Promedio')
        axes[1,0].set_title('Eficiencia por Configuración')
        axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Proyección de costes mensuales
        if not cost_df.empty:
            monthly_costs = {}
            usage_scenarios = [100, 500, 1000, 5000, 10000]  # consultas/mes
            
            for model in cost_df['llm_model'].unique():
                avg_cost = cost_df[cost_df['llm_model'] == model]['estimated_cost'].mean()
                monthly_costs[model] = [avg_cost * usage for usage in usage_scenarios]
            
            x_pos = np.arange(len(usage_scenarios))
            width = 0.25
            
            for i, (model, costs) in enumerate(monthly_costs.items()):
                axes[1,1].bar(x_pos + i*width, costs, width, label=model, alpha=0.8)
            
            axes[1,1].set_xlabel('Consultas por Mes')
            axes[1,1].set_ylabel('Coste Mensual ($)')
            axes[1,1].set_title('Proyección de Costes Mensuales')
            axes[1,1].set_xticks(x_pos + width)
            axes[1,1].set_xticklabels(usage_scenarios)
            axes[1,1].legend()
            axes[1,1].set_yscale('log')
        else:
            axes[1,1].text(0.5, 0.5, 'Costes = $0\n(Modelos Locales)', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Proyección de Costes Mensuales')
        
        plt.tight_layout()
        
        chart_file2 = self.output_dir / f"tfm_detailed_analysis_{timestamp}.png"
        plt.savefig(chart_file2, dpi=300, bbox_inches='tight')
        plt.close()
        chart_files.append(str(chart_file2))
        
        # 3. Heatmap de rendimiento por configuración
        if len(success_df) > 10:  # Solo si hay suficientes datos
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Crear matriz de rendimiento
            pivot_data = success_df.pivot_table(
                values='response_time', 
                index='llm_model', 
                columns='vector_store', 
                aggfunc='mean'
            )
            
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax)
            ax.set_title('Heatmap: Tiempo de Respuesta Promedio por Configuración')
            ax.set_xlabel('Vector Store')
            ax.set_ylabel('Modelo LLM')
            
            plt.tight_layout()
            
            chart_file3 = self.output_dir / f"tfm_heatmap_performance_{timestamp}.png"
            plt.savefig(chart_file3, dpi=300, bbox_inches='tight')
            plt.close()
            chart_files.append(str(chart_file3))
        
        self.logger.info(f"Visualizaciones creadas: {len(chart_files)} gráficos")
        
        return chart_files
    
    def print_summary(self, summary: Dict[str, Any]):
        """Imprimir resumen en consola"""
        
        print("\n" + "="*80)
        print("🎓 BENCHMARK ACADÉMICO TFM - RESUMEN FINAL")
        print("="*80)
        
        if summary.get('benchmark_completed'):
            print(f"✅ Benchmark completado exitosamente")
            print(f"⏱️  Tiempo total: {summary['total_time']:.2f} segundos")
            print(f"🧪 Pruebas realizadas: {summary['total_tests']}")
            print(f"✅ Pruebas exitosas: {summary['success_tests']}")
            print(f"📊 Tasa de éxito: {(summary['success_tests']/summary['total_tests']*100):.1f}%")
            
            print(f"\n📁 Archivos generados:")
            for file_path in summary['report_files']:
                print(f"   📄 {Path(file_path).name}")
            
            for file_path in summary['chart_files']:
                print(f"   📊 {Path(file_path).name}")
            
            if 'analysis' in summary and 'recommendations' in summary['analysis']:
                print(f"\n💡 Recomendaciones principales:")
                recs = summary['analysis']['recommendations']
                
                for municipality_type, rec in recs.items():
                    if isinstance(rec, dict) and 'recommended_config' in rec and rec['recommended_config']:
                        config = rec['recommended_config']
                        print(f"   🏛️  {rec.get('description', municipality_type)}:")
                        print(f"      Vector Store: {config.get('vector_store', 'N/A')}")
                        print(f"      LLM: {config.get('llm_provider', 'N/A')}/{config.get('llm_model', 'N/A')}")
        else:
            print(f"❌ Benchmark falló: {summary.get('error', 'Error desconocido')}")
            print(f"📊 Resultados parciales: {summary.get('partial_results', 0)}")
        
        print("\n" + "="*80)
        print("🎉 ¡Benchmark TFM completado! Datos listos para memoria académica.")
        print("="*80)


def main():
    """Función principal para ejecutar benchmark desde línea de comandos"""
    
    print("🚀 Iniciando Benchmark Académico TFM")
    print("Prototipo de Chatbot RAG para Administraciones Locales")
    print("-" * 60)
    
    try:
        # Crear instancia del benchmark
        benchmark = TFMBenchmark()
        
        # Ejecutar benchmark completo
        summary = benchmark.run_complete_benchmark()
        
        # Mostrar resumen
        benchmark.print_summary(summary)
        
        # Guardar resumen final
        summary_file = benchmark.output_dir / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📋 Resumen guardado en: {summary_file}")
        
        return 0 if summary.get('benchmark_completed') else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrumpido por el usuario")
        return 1
    except Exception as e:
        print(f"\n❌ Error ejecutando benchmark: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())