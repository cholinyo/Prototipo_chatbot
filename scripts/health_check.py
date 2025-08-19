#!/usr/bin/env python3
"""
Health Check Rápido - TFM Chatbot RAG
Verificación rápida del estado del sistema

Autor: Vicente Caruncho Ramos
"""
import sys
import json
import time
from pathlib import Path

# Añadir el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def quick_health_check():
    """Health check rápido del sistema"""
    status = {
        "timestamp": time.time(),
        "status": "unknown",
        "services": {},
        "errors": []
    }
    
    try:
        # Verificar Flask
        import flask
        status["services"]["flask"] = {"status": "available", "version": flask.__version__}
    except ImportError as e:
        status["services"]["flask"] = {"status": "error", "error": str(e)}
        status["errors"].append("Flask no disponible")
    
    try:
        # Verificar Ollama
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=3)
        if response.status_code == 200:
            models = response.json().get('models', [])
            status["services"]["ollama"] = {
                "status": "healthy", 
                "models_count": len(models)
            }
        else:
            status["services"]["ollama"] = {"status": "error", "code": response.status_code}
    except Exception as e:
        status["services"]["ollama"] = {"status": "unavailable", "error": str(e)}
    
    try:
        # Verificar configuración
        from app.core.config import get_config
        config = get_config()
        status["services"]["config"] = {"status": "healthy", "environment": getattr(config, 'environment', 'unknown')}
    except Exception as e:
        status["services"]["config"] = {"status": "error", "error": str(e)}
        status["errors"].append("Configuración no disponible")
    
    # Determinar estado general
    if len(status["errors"]) == 0:
        status["status"] = "healthy"
    elif len(status["errors"]) < 2:
        status["status"] = "degraded"
    else:
        status["status"] = "unhealthy"
    
    return status


def print_health_status(status):
    """Imprimir estado de salud en formato legible"""
    print("🩺 HEALTH CHECK - TFM Chatbot RAG")
    print("=" * 40)
    
    # Estado general
    status_emoji = {
        "healthy": "🟢",
        "degraded": "🟡", 
        "unhealthy": "🔴",
        "unknown": "⚪"
    }
    
    print(f"\n{status_emoji.get(status['status'], '⚪')} Estado General: {status['status'].upper()}")
    
    # Servicios
    print("\n📋 Servicios:")
    for service_name, service_data in status["services"].items():
        service_status = service_data.get("status", "unknown")
        emoji = "✅" if service_status in ["healthy", "available"] else "❌"
        print(f"   {emoji} {service_name}: {service_status}")
        
        if "version" in service_data:
            print(f"      📦 Versión: {service_data['version']}")
        if "models_count" in service_data:
            print(f"      📦 Modelos: {service_data['models_count']}")
    
    # Errores
    if status["errors"]:
        print("\n⚠️ Errores:")
        for error in status["errors"]:
            print(f"   ❌ {error}")
    
    print(f"\n⏰ Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(status['timestamp']))}")


if __name__ == "__main__":
    try:
        if "--json" in sys.argv:
            # Salida JSON para integración con otros sistemas
            status = quick_health_check()
            print(json.dumps(status, indent=2))
        else:
            # Salida legible para humanos
            status = quick_health_check()
            print_health_status(status)
            
            # Exit code basado en el estado
            if status["status"] == "healthy":
                sys.exit(0)
            elif status["status"] == "degraded":
                sys.exit(1)
            else:
                sys.exit(2)
                
    except KeyboardInterrupt:
        print("\n⏹️ Health check cancelado")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Error en health check: {e}")
        sys.exit(3)
