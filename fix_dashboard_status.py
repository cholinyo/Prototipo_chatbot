#!/usr/bin/env python3
"""
Fix Dashboard Status - Conectar el frontend con el backend
TFM Vicente Caruncho
"""

from pathlib import Path

def create_updated_health_endpoint():
    """Crear endpoint /health que devuelva el formato correcto para el frontend"""
    
    # Verificar si ya existe un run.py
    run_file = Path("run.py")
    
    if run_file.exists():
        # Leer contenido actual
        with open(run_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Buscar y reemplazar el endpoint /health
        health_endpoint = '''
        @app.route('/health')
        def health_check():
            """Endpoint de verificación de salud actualizado"""
            try:
                from app.services.llm.llm_services import LLMService
                llm_service = LLMService()
                
                # Obtener estado real
                health = llm_service.health_check()
                
                # Formatear para el frontend
                response = {
                    'status': health['status'],
                    'timestamp': health['timestamp'],
                    'services': {
                        'llm': 'available' if health['status'] in ['healthy', 'degraded'] else 'unavailable',
                        'ollama': health['services']['ollama']['status'],
                        'openai': health['services']['openai']['status']
                    },
                    'models': health['models'],
                    'components': {
                        'embeddings': 'available',
                        'vector_store': 'available',
                        'llm': health['status']
                    }
                }
                
                return jsonify(response)
                
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'timestamp': time.time(),
                    'error': str(e),
                    'services': {
                        'llm': 'unavailable',
                        'ollama': 'unavailable', 
                        'openai': 'unavailable'
                    },
                    'components': {}
                }), 500
'''
        
        # Reemplazar la función health_check existente
        import re
        pattern = r'@app\.route\(\'/health\'\).*?return.*?(?=\n        @|\n\s*if __name__|\nif __name__|\Z)'
        
        if '@app.route(\'/health\')' in content:
            new_content = re.sub(pattern, health_endpoint.strip(), content, flags=re.DOTALL)
        else:
            # Si no existe, añadirlo antes del final
            new_content = content.replace(
                'if __name__ == "__main__":',
                health_endpoint + '\n        if __name__ == "__main__":'
            )
        
        # Asegurar importaciones necesarias
        if 'import time' not in new_content:
            new_content = new_content.replace(
                'from flask import',
                'import time\nfrom flask import'
            )
        
        with open(run_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ Actualizado: {run_file}")

def create_updated_javascript():
    """Actualizar JavaScript para mostrar estado correctamente"""
    
    js_file = Path("app/static/js/main.js")
    
    if js_file.exists():
        js_content = '''// JavaScript principal para la aplicación
document.addEventListener('DOMContentLoaded', function() {
    console.log('Prototipo_chatbot cargado correctamente');
    
    // Inicializar tooltips de Bootstrap
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Verificar estado del sistema al cargar y cada minuto
    checkSystemStatus();
    setInterval(checkSystemStatus, 60000);
});

function checkSystemStatus() {
    console.log('Verificando estado del sistema...');
    
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            console.log('Estado del sistema:', data);
            updateSystemStatus(data);
        })
        .catch(error => {
            console.error('Error verificando estado:', error);
            updateSystemStatus({
                status: 'error',
                services: {
                    ollama: 'unavailable',
                    openai: 'unavailable'
                }
            });
        });
}

function updateSystemStatus(data) {
    // Actualizar indicador principal
    const indicator = document.getElementById('status-indicator');
    if (indicator) {
        updateStatusIndicator(data.status);
    }
    
    // Actualizar modelos locales
    const localModelsStatus = document.getElementById('local-models-status');
    if (localModelsStatus) {
        if (data.services && data.services.ollama === 'available') {
            localModelsStatus.className = 'badge bg-success';
            localModelsStatus.innerHTML = '<i class="fas fa-check me-1"></i>Disponible';
        } else {
            localModelsStatus.className = 'badge bg-warning';
            localModelsStatus.innerHTML = '<i class="fas fa-exclamation-triangle me-1"></i>No configurado';
        }
    }
    
    // Actualizar OpenAI API
    const openaiStatus = document.getElementById('openai-status');
    if (openaiStatus) {
        if (data.services && data.services.openai === 'configured') {
            openaiStatus.className = 'badge bg-success';
            openaiStatus.innerHTML = '<i class="fas fa-check me-1"></i>Configurado';
        } else {
            openaiStatus.className = 'badge bg-secondary';
            openaiStatus.innerHTML = '<i class="fas fa-times me-1"></i>No configurado';
        }
    }
}

function updateStatusIndicator(status) {
    const indicator = document.getElementById('status-indicator');
    if (!indicator) return;
    
    const statusClasses = {
        'healthy': 'bg-success',
        'degraded': 'bg-warning', 
        'error': 'bg-danger'
    };
    
    const statusTexts = {
        'healthy': 'Sistema Activo',
        'degraded': 'Sistema con Alertas',
        'error': 'Sistema con Errores'
    };
    
    // Limpiar clases anteriores
    indicator.className = indicator.className.replace(/bg-\\w+/g, '');
    
    // Aplicar nueva clase
    indicator.classList.add('badge', statusClasses[status] || 'bg-secondary');
    indicator.innerHTML = `<i class="fas fa-circle me-1"></i>${statusTexts[status] || 'Estado Desconocido'}`;
}

// Funciones para navegación (placeholder)
function navigateToChat() {
    window.location.href = '/chat';
}

function navigateToComparison() {
    alert('Comparador de modelos en desarrollo');
}

function navigateToAdmin() {
    alert('Panel de administración en desarrollo');
}

function navigateToConfig() {
    alert('Configuración de fuentes de datos en desarrollo');
}

function showComingSoon(feature) {
    alert(feature + ' próximamente disponible');
}
'''
        
        with open(js_file, 'w', encoding='utf-8') as f:
            f.write(js_content)
        
        print(f"✅ Actualizado: {js_file}")

def test_health_endpoint():
    """Test del endpoint de health"""
    
    print("\n🧪 TESTEANDO ENDPOINT /health...")
    
    try:
        import requests
        
        # Test directo del endpoint
        response = requests.get("http://localhost:5000/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Endpoint responde correctamente")
            print(f"✅ Estado: {data.get('status', 'unknown')}")
            
            services = data.get('services', {})
            print(f"✅ Ollama: {services.get('ollama', 'unknown')}")
            print(f"✅ OpenAI: {services.get('openai', 'unknown')}")
            
            return True
        else:
            print(f"❌ Endpoint error: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Aplicación no está ejecutándose")
        print("💡 Ejecutar: python run.py")
        return False
    except Exception as e:
        print(f"❌ Error en test: {e}")
        return False

def main():
    """Función principal"""
    print("🔧 CONECTANDO FRONTEND CON BACKEND")
    print("🎓 TFM Vicente Caruncho - Prototipo Chatbot RAG")
    print("=" * 60)
    
    # Actualizar archivos
    create_updated_health_endpoint()
    create_updated_javascript()
    
    print("\n✅ ARCHIVOS ACTUALIZADOS")
    print("\n🚀 Próximos pasos:")
    print("1. Reiniciar la aplicación: Ctrl+C y luego python run.py")
    print("2. Actualizar navegador: F5 en http://localhost:5000")
    print("3. Los estados deberían actualizarse automáticamente")
    
    # Test si la aplicación está ejecutándose
    test_health_endpoint()

if __name__ == "__main__":
    main()