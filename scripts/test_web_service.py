# scripts/check_blueprint.py
"""
Verificar que el blueprint de web-sources esté registrado correctamente
"""

import requests
import sys
import os

# Añadir al path
sys.path.insert(0, os.path.abspath('.'))

def check_blueprint_registration():
    print("Verificando registro de blueprint web-sources")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:5000"
    
    # 1. Verificar que Flask esté corriendo
    print("\n1. Verificando que Flask esté activo...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"   Flask activo: {response.status_code == 200}")
    except:
        print("   ERROR: Flask no está corriendo")
        print("   Ejecuta primero: python run.py")
        return False
    
    # 2. Verificar endpoints específicos
    print("\n2. Verificando endpoints...")
    
    endpoints_to_test = [
        "/api/web-sources",
        "/api/web-sources/",
        "/webs",
        "/api/data-sources"
    ]
    
    for endpoint in endpoints_to_test:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"   ✓ {endpoint}: OK (200)")
                if 'sources' in response.text:
                    data = response.json()
                    print(f"     Fuentes: {data.get('total', 0)}")
            elif response.status_code == 404:
                print(f"   ✗ {endpoint}: No encontrado (404)")
            else:
                print(f"   ? {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"   ✗ {endpoint}: Error - {e}")
    
    # 3. Verificar rutas disponibles (si existe endpoint de debug)
    print("\n3. Intentando obtener lista de rutas...")
    debug_endpoints = ["/debug/routes", "/routes", "/api/routes"]
    
    for debug_endpoint in debug_endpoints:
        try:
            response = requests.get(f"{base_url}{debug_endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"   ✓ {debug_endpoint}: Encontrado")
                # Si es JSON, mostrar rutas
                if 'application/json' in response.headers.get('content-type', ''):
                    routes = response.json()
                    web_routes = [r for r in routes if 'web' in r.get('rule', '').lower()]
                    print(f"   Rutas relacionadas con 'web': {len(web_routes)}")
                    for route in web_routes[:5]:  # Mostrar máximo 5
                        print(f"     - {route.get('rule', 'N/A')}")
                break
        except:
            continue
    
    # 4. Verificar directamente desde el código
    print("\n4. Verificando desde el código...")
    try:
        from app import create_app
        app = create_app()
        
        # Obtener todas las rutas registradas
        routes = []
        for rule in app.url_map.iter_rules():
            routes.append({
                'rule': rule.rule,
                'endpoint': rule.endpoint,
                'methods': list(rule.methods)
            })
        
        # Filtrar rutas de web-sources
        web_source_routes = [r for r in routes if 'web' in r['rule'].lower() or 'web' in r['endpoint'].lower()]
        
        print(f"   Total rutas: {len(routes)}")
        print(f"   Rutas web-sources: {len(web_source_routes)}")
        
        if web_source_routes:
            print("   Rutas encontradas:")
            for route in web_source_routes:
                print(f"     {route['rule']} -> {route['endpoint']} {route['methods']}")
        else:
            print("   ⚠️ No se encontraron rutas de web-sources")
            
            # Buscar blueprints registrados
            print("\n   Blueprints registrados:")
            for blueprint_name, blueprint in app.blueprints.items():
                print(f"     - {blueprint_name}: {blueprint.url_prefix}")
        
        return len(web_source_routes) > 0
        
    except Exception as e:
        print(f"   Error verificando código: {e}")
        return False

if __name__ == "__main__":
    success = check_blueprint_registration()
    print(f"\nResultado: {'BLUEPRINT CORRECTO' if success else 'PROBLEMA CON BLUEPRINT'}")