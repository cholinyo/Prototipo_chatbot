#!/usr/bin/env python3
"""
Script de pruebas para LLM Service
TFM Vicente Caruncho - Sistemas Inteligentes

Ejecutar desde el directorio raíz del proyecto:
python tests/test_llm_service.py
"""

import sys
import os
import json
import time
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.llm_service import get_llm_service, LLMRequest
from app.models import DocumentChunk, DocumentMetadata

def print_section(title: str):
    """Imprimir sección con formato"""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print('='*60)

def test_service_initialization():
    """Test 1: Inicialización del servicio"""
    print_section("INICIALIZACIÓN DEL SERVICIO")
    
    try:
        service = get_llm_service()
        print("✅ Servicio LLM inicializado correctamente")
        
        # Verificar proveedores
        providers = list(service.providers.keys())
        print(f"📋 Proveedores registrados: {providers}")
        
        return service
    except Exception as e:
        print(f"❌ Error inicializando servicio: {e}")
        return None

def test_provider_availability(service):
    """Test 2: Disponibilidad de proveedores"""
    print_section("DISPONIBILIDAD DE PROVEEDORES")
    
    try:
        availability = service.get_available_providers()
        
        for provider, available in availability.items():
            status = "✅ DISPONIBLE" if available else "❌ NO DISPONIBLE"
            print(f"{provider}: {status}")
        
        return availability
    except Exception as e:
        print(f"❌ Error verificando disponibilidad: {e}")
        return {}

def test_available_models(service):
    """Test 3: Modelos disponibles"""
    print_section("MODELOS DISPONIBLES")
    
    try:
        models = service.get_available_models()
        
        for provider, model_list in models.items():
            print(f"\n{provider.upper()}:")
            if model_list:
                for model in model_list:
                    print(f"  - {model}")
            else:
                print("  No hay modelos disponibles")
        
        return models
    except Exception as e:
        print(f"❌ Error obteniendo modelos: {e}")
        return {}

def test_simple_generation(service, availability):
    """Test 4: Generación simple sin contexto"""
    print_section("GENERACIÓN SIMPLE SIN CONTEXTO")
    
    request = LLMRequest(
        query="¿Qué es una licencia de obras municipal?",
        temperature=0.7,
        max_tokens=200
    )
    
    results = {}
    
    # Test Ollama
    if availability.get('ollama', False):
        print("\n🦙 Testing Ollama...")
        try:
            start_time = time.time()
            result = service.generate_response(request, 'ollama')
            test_time = time.time() - start_time
            
            print(f"✅ Modelo: {result.model_name}")
            print(f"⏱️  Tiempo total: {test_time:.2f}s")
            print(f"⏱️  Tiempo generación: {result.response_time:.2f}s")
            print(f"🪙 Tokens: {result.total_tokens}")
            print(f"📝 Respuesta (100 chars): {result.response[:100]}...")
            
            if result.error:
                print(f"⚠️  Error: {result.error}")
            
            results['ollama'] = result
            
        except Exception as e:
            print(f"❌ Error en Ollama: {e}")
    
    # Test OpenAI
    if availability.get('openai', False):
        print("\n🤖 Testing OpenAI...")
        try:
            start_time = time.time()
            result = service.generate_response(request, 'openai')
            test_time = time.time() - start_time
            
            print(f"✅ Modelo: {result.model_name}")
            print(f"⏱️  Tiempo total: {test_time:.2f}s")
            print(f"⏱️  Tiempo generación: {result.response_time:.2f}s")
            print(f"🪙 Tokens: {result.total_tokens}")
            print(f"💰 Coste estimado: ${result.estimated_cost:.4f}")
            print(f"📝 Respuesta (100 chars): {result.response[:100]}...")
            
            if result.error:
                print(f"⚠️  Error: {result.error}")
            
            results['openai'] = result
            
        except Exception as e:
            print(f"❌ Error en OpenAI: {e}")
    
    return results

def test_rag_generation(service, availability):
    """Test 5: Generación con contexto RAG"""
    print_section("GENERACIÓN CON CONTEXTO RAG")
    
    # Crear chunks de prueba simulando documentos municipales
    chunks = [
        DocumentChunk(
            content="""La licencia de obras es un acto administrativo por el que el Ayuntamiento 
            autoriza la realización de obras de construcción, instalación o demolición, 
            así como el uso del suelo para dichos fines. Es obligatoria para obras mayores 
            que requieran proyecto técnico.""",
            metadata=DocumentMetadata(
                source_path="ordenanza_urbanistica_2024.pdf",
                source_type="pdf",
                page_number=15,
                title="Ordenanza de Urbanismo - Licencias de Obra"
            )
        ),
        DocumentChunk(
            content="""Los plazos para resolver las solicitudes de licencia de obras son:
            - Obras menores: 15 días hábiles desde presentación completa
            - Obras mayores: 30 días hábiles desde presentación completa
            - Obras en suelo rústico: 45 días hábiles
            El silencio administrativo tendrá efectos desestimatorios.""",
            metadata=DocumentMetadata(
                source_path="reglamento_tramitacion_2024.pdf",
                source_type="pdf",
                page_number=8,
                title="Reglamento de Tramitación - Plazos"
            )
        )
    ]
    
    request = LLMRequest(
        query="¿Cuáles son los plazos para obtener una licencia de obras?",
        context=chunks,
        temperature=0.3,  # Más determinista para RAG
        max_tokens=300
    )
    
    results = {}
    
    # Test Ollama con RAG
    if availability.get('ollama', False):
        print("\n🦙 Testing Ollama con RAG...")
        try:
            result = service.generate_response(request, 'ollama')
            
            print(f"✅ Modelo: {result.model_name}")
            print(f"⏱️  Tiempo: {result.response_time:.2f}s")
            print(f"📚 Fuentes: {result.sources}")
            print(f"📝 Respuesta: {result.response}")
            
            results['ollama_rag'] = result
            
        except Exception as e:
            print(f"❌ Error en Ollama RAG: {e}")
    
    # Test OpenAI con RAG
    if availability.get('openai', False):
        print("\n🤖 Testing OpenAI con RAG...")
        try:
            result = service.generate_response(request, 'openai')
            
            print(f"✅ Modelo: {result.model_name}")
            print(f"⏱️  Tiempo: {result.response_time:.2f}s")
            print(f"💰 Coste: ${result.estimated_cost:.4f}")
            print(f"📚 Fuentes: {result.sources}")
            print(f"📝 Respuesta: {result.response}")
            
            results['openai_rag'] = result
            
        except Exception as e:
            print(f"❌ Error en OpenAI RAG: {e}")
    
    return results

def test_model_comparison(service, availability):
    """Test 6: Comparación directa entre modelos"""
    print_section("COMPARACIÓN DIRECTA ENTRE MODELOS")
    
    if not (availability.get('ollama') and availability.get('openai')):
        print("⚠️  Comparación requiere ambos proveedores disponibles")
        return {}
    
    # Crear contexto de prueba
    chunks = [
        DocumentChunk(
            content="""El procedimiento de solicitud de licencia de obras requiere:
            1. Instancia de solicitud cumplimentada
            2. Proyecto técnico visado (obras mayores)
            3. Justificante de pago de tasas municipales
            4. Documentación catastral actualizada
            5. Autorización de la comunidad (si procede)""",
            metadata=DocumentMetadata(
                source_path="guia_tramites_2024.pdf",
                source_type="pdf",
                title="Guía de Trámites Municipales"
            )
        )
    ]
    
    request = LLMRequest(
        query="¿Qué documentación necesito para solicitar una licencia de obras mayor?",
        context=chunks,
        temperature=0.3,
        max_tokens=250
    )
    
    print("🚀 Ejecutando comparación en paralelo...")
    try:
        results = service.compare_models(request)
        
        print(f"\n📊 RESULTADOS DE COMPARACIÓN:")
        print("-" * 50)
        
        for provider, result in results.items():
            print(f"\n{provider.upper()}:")
            print(f"  Modelo: {result.model_name}")
            print(f"  Tiempo: {result.response_time:.2f}s")
            print(f"  Tokens: {result.total_tokens}")
            if result.estimated_cost:
                print(f"  Coste: ${result.estimated_cost:.4f}")
            print(f"  Respuesta: {result.response[:150]}...")
            if result.error:
                print(f"  Error: {result.error}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error en comparación: {e}")
        return {}

def test_service_stats(service):
    """Test 7: Estadísticas del servicio"""
    print_section("ESTADÍSTICAS DEL SERVICIO")
    
    try:
        stats = service.get_service_stats()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return stats
    except Exception as e:
        print(f"❌ Error obteniendo estadísticas: {e}")
        return {}

def test_error_handling(service):
    """Test 8: Manejo de errores"""
    print_section("MANEJO DE ERRORES")
    
    # Test proveedor inexistente
    print("🧪 Testing proveedor inexistente...")
    request = LLMRequest(query="Test", max_tokens=50)
    result = service.generate_response(request, 'proveedor_falso')
    print(f"Resultado: {result.error or 'Error no capturado'}")
    
    # Test modelo inexistente en Ollama
    print("\n🧪 Testing modelo inexistente...")
    result = service.generate_response(request, 'ollama', 'modelo_inexistente')
    print(f"Resultado: {result.response[:100]}...")

def generate_test_report(all_results):
    """Generar reporte de pruebas"""
    print_section("REPORTE FINAL DE PRUEBAS")
    
    print("📋 RESUMEN EJECUTIVO:")
    print("-" * 30)
    
    # Analizar disponibilidad
    available_providers = []
    for provider, available in all_results.get('availability', {}).items():
        if available:
            available_providers.append(provider)
    
    print(f"✅ Proveedores disponibles: {len(available_providers)}/2")
    print(f"   Activos: {', '.join(available_providers)}")
    
    # Analizar modelos
    total_models = sum(len(models) for models in all_results.get('models', {}).values())
    print(f"🤖 Total modelos disponibles: {total_models}")
    
    # Analizar pruebas de generación
    simple_tests = all_results.get('simple_generation', {})
    rag_tests = all_results.get('rag_generation', {})
    comparison_tests = all_results.get('comparison', {})
    
    print(f"🧪 Pruebas de generación simple: {len(simple_tests)} completadas")
    print(f"📚 Pruebas con RAG: {len(rag_tests)} completadas")
    print(f"⚖️  Pruebas de comparación: {'✅' if comparison_tests else '❌'}")
    
    # Status general
    if available_providers and total_models > 0:
        print(f"\n🎉 ESTADO GENERAL: ✅ SISTEMA FUNCIONAL")
        print("   El LLM Service está listo para integración")
    else:
        print(f"\n⚠️  ESTADO GENERAL: 🔧 REQUIERE CONFIGURACIÓN")
        print("   Verificar instalación de Ollama y/o API key de OpenAI")

def main():
    """Función principal de pruebas"""
    print("🚀 INICIANDO SUITE DE PRUEBAS LLM SERVICE")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Inicialización
    service = test_service_initialization()
    if not service:
        print("❌ No se puede continuar sin servicio inicializado")
        return
    
    # Test 2: Disponibilidad
    availability = test_provider_availability(service)
    results['availability'] = availability
    
    # Test 3: Modelos
    models = test_available_models(service)
    results['models'] = models
    
    # Test 4: Generación simple
    simple_results = test_simple_generation(service, availability)
    results['simple_generation'] = simple_results
    
    # Test 5: RAG
    rag_results = test_rag_generation(service, availability)
    results['rag_generation'] = rag_results
    
    # Test 6: Comparación (si ambos disponibles)
    comparison_results = test_model_comparison(service, availability)
    results['comparison'] = comparison_results
    
    # Test 7: Estadísticas
    stats = test_service_stats(service)
    results['stats'] = stats
    
    # Test 8: Errores
    test_error_handling(service)
    
    # Reporte final
    generate_test_report(results)
    
    print(f"\n🏁 PRUEBAS COMPLETADAS")
    print("=" * 60)

if __name__ == "__main__":
    main()