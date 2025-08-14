#!/usr/bin/env python3
"""
Diagnóstico Completo del Sistema
TFM Vicente Caruncho - Prototipo Chatbot RAG
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path

def test_health_endpoint_direct():
    """Test directo del endpoint /health"""
    print("\n🔍 TESTEANDO ENDPOINT /health DIRECTAMENTE...")
    
    try:
        import requests
        
        # Test del endpoint
        url = "http://localhost:5000/health"
        print(f"📡 Haciendo petición a: {url}")
        
        response = requests.get(url, timeout=10)
        
        print(f"📊 Status Code: {response.status_code}")
        print(f"📊 Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Respuesta JSON recibida:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
            # Analizar respuesta
            print("\n🔍 ANÁLISIS DE LA RESPUESTA:")
            print(f"   Status general: {data.get('status', 'NO ENCONTRADO')}")
            
            services = data.get('services', {})
            print(f"   Servicios encontrados: {list(services.keys())}")
            
            for service, status in services.items():
                print(f"   - {service}: {status}")
            
            models = data.get('models', {})
            if models:
                print(f"   Modelos disponibles:")
                for provider, model_list in models.items():
                    print(f"   - {provider}: {model_list}")
            
            return True, data
        else:
            print(f"❌ Error HTTP: {response.status_code}")
            print(f"   Contenido: {response.text}")
            return False, None
            
    except requests.exceptions.ConnectionError:
        print("❌ No se puede conectar - ¿Está ejecutándose la aplicación?")
        print("💡 Ejecuta: python run.py")
        return False, None
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        traceback.print_exc()
        return False, None

def test_llm_service_direct():
    """Test directo del LLM Service"""
    print("\n🤖 TESTEANDO LLM SERVICE DIRECTAMENTE...")
    
    try:
        # Cambiar al directorio del proyecto
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        print("📁 Directorio del proyecto:", project_root)
        print("📁 Archivos en app/services/llm:")
        
        llm_dir = project_root / "app" / "services" / "llm"
        if llm_dir.exists():
            for file in llm_dir.iterdir():
                print(f"   - {file.name}")
        else:
            print("   ❌ Directorio app/services/llm no existe")
            return False
        
        # Intentar importar el servicio
        print("\n🔄 Importando LLM Service...")
        
        try:
            from app.services.llm.llm_services import LLMService
            print("✅ LLMService importado correctamente")
            
            # Crear instancia
            llm = LLMService()
            print("✅ Instancia LLMService creada")
            
            # Test health check
            print("\n🏥 Ejecutando health_check()...")
            health = llm.health_check()
            
            print("✅ Health check ejecutado:")
            print(json.dumps(health, indent=2, ensure_ascii=False))
            
            return True, health
            
        except ImportError as e:
            print(f"❌ Error importando LLMService: {e}")
            print("💡 Verificar estructura de archivos")
            return False, None
            
    except Exception as e:
        print(f"❌ Error en test directo: {e}")
        traceback.print_exc()
        return False, None

def test_ollama_connection():
    """Test conexión directa con Ollama"""
    print("\n🦙 TESTEANDO OLLAMA DIRECTAMENTE...")
    
    try:
        import requests
        
        url = "http://localhost:11434/api/tags"
        print(f"📡 Probando conexión: {url}")
        
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            
            print("✅ Ollama conectado correctamente")
            print(f"✅ Modelos disponibles: {models}")
            
            return True, models
        else:
            print(f"❌ Ollama error HTTP: {response.status_code}")
            return False, []
            
    except requests.exceptions.ConnectionError:
        print("❌ Ollama no está ejecutándose")
        print("💡 Ejecutar: ollama serve")
        return False, []
    except Exception as e:
        print(f"❌ Error conectando Ollama: {e}")
        return False, []

def test_openai_config():
    """Test configuración OpenAI"""
    print("\n🌐 TESTEANDO CONFIGURACIÓN OPENAI...")
    
    try:
        # Cargar variables de entorno
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            print("❌ OPENAI_API_KEY no encontrada en variables de entorno")
            print("💡 Verificar archivo .env")
            return False, None
        
        if not api_key.startswith("sk-"):
            print("❌ OPENAI_API_KEY tiene formato incorrecto")
            print(f"   Valor actual: {api_key[:20]}...")
            return False, None
        
        print(f"✅ API Key encontrada: {api_key[:10]}...{api_key[-4:]}")
        
        # Test básico de conexión
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            # Intentar listar modelos
            models = client.models.list()
            print("✅ Conexión OpenAI exitosa")
            print(f"✅ Modelos accesibles: {len(models.data)}")
            
            return True, api_key
            
        except Exception as e:
            print(f"❌ Error conectando OpenAI: {e}")
            return False, api_key
            
    except Exception as e:
        print(f"❌ Error verificando OpenAI: {e}")
        return False, None

def test_frontend_elements():
    """Verificar que existen los elementos del frontend"""
    print("\n🌐 VERIFICANDO ELEMENTOS DEL FRONTEND...")
    
    try:
        # Verificar template index.html
        template_file = Path("app/templates/index.html")
        
        if not template_file.exists():
            print("❌ Template index.html no encontrado")
            return False
        
        with open(template_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Buscar elementos específicos
        elements_to_check = [
            'local-models-status',
            'openai-status',
            'status-indicator'
        ]
        
        found_elements = []
        for element_id in elements_to_check:
            if f'id="{element_id}"' in content or f"id='{element_id}'" in content:
                found_elements.append(element_id)
                print(f"   ✅ Elemento {element_id} encontrado")
            else:
                print(f"   ❌ Elemento {element_id} NO encontrado")
        
        print(f"\n📊 Elementos encontrados: {len(found_elements)}/{len(elements_to_check)}")
        
        return len(found_elements) == len(elements_to_check)
        
    except Exception as e:
        print(f"❌ Error verificando frontend: {e}")
        return False

def generate_debug_report():
    """Generar reporte completo de debug"""
    print("\n📋 GENERANDO REPORTE DE DEBUG...")
    
    report = {
        "timestamp": time.time(),
        "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {}
    }
    
    # Test 1: Health endpoint
    health_ok, health_data = test_health_endpoint_direct()
    report["tests"]["health_endpoint"] = {
        "success": health_ok,
        "data": health_data
    }
    
    # Test 2: LLM Service
    llm_ok, llm_data = test_llm_service_direct()
    report["tests"]["llm_service"] = {
        "success": llm_ok,
        "data": llm_data
    }
    
    # Test 3: Ollama
    ollama_ok, ollama_models = test_ollama_connection()
    report["tests"]["ollama"] = {
        "success": ollama_ok,
        "models": ollama_models
    }
    
    # Test 4: OpenAI
    openai_ok, openai_key = test_openai_config()
    report["tests"]["openai"] = {
        "success": openai_ok,
        "has_key": bool(openai_key)
    }
    
    # Test 5: Frontend
    frontend_ok = test_frontend_elements()
    report["tests"]["frontend"] = {
        "success": frontend_ok
    }
    
    # Guardar reporte
    report_file = Path("debug_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Reporte guardado en: {report_file}")
    
    return report

def main():
    """Función principal de diagnóstico"""
    print("🔍 DIAGNÓSTICO COMPLETO DEL SISTEMA")
    print("🎓 TFM Vicente Caruncho - Prototipo Chatbot RAG")
    print("=" * 70)
    
    # Verificar que estamos en el directorio correcto
    if not Path("run.py").exists():
        print("❌ No se encuentra run.py en el directorio actual")
        print("💡 Ejecutar desde el directorio del proyecto")
        return
    
    # Generar reporte completo
    report = generate_debug_report()
    
    # Mostrar resumen
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE DIAGNÓSTICO")
    print("=" * 70)
    
    tests = report["tests"]
    
    print(f"Health Endpoint: {'✅' if tests['health_endpoint']['success'] else '❌'}")
    print(f"LLM Service:     {'✅' if tests['llm_service']['success'] else '❌'}")
    print(f"Ollama:          {'✅' if tests['ollama']['success'] else '❌'}")
    print(f"OpenAI:          {'✅' if tests['openai']['success'] else '❌'}")
    print(f"Frontend:        {'✅' if tests['frontend']['success'] else '❌'}")
    
    # Recomendaciones
    print("\n💡 RECOMENDACIONES:")
    
    if not tests['ollama']['success']:
        print("   🦙 Ollama: Ejecutar 'ollama serve' y 'ollama pull llama3.2:3b'")
    
    if not tests['openai']['success']:
        print("   🌐 OpenAI: Verificar OPENAI_API_KEY en archivo .env")
    
    if not tests['llm_service']['success']:
        print("   🤖 LLM Service: Verificar importaciones y estructura de archivos")
    
    if not tests['health_endpoint']['success']:
        print("   🏥 Health Endpoint: Verificar que la aplicación esté ejecutándose")
    
    if not tests['frontend']['success']:
        print("   🌐 Frontend: Verificar elementos HTML en template")
    
    print(f"\n📄 Reporte completo en: debug_report.json")

if __name__ == "__main__":
    main()