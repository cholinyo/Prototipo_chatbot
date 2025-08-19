"""
Script para corregir rápidamente el error en run.py
Ejecutar desde el directorio raíz del proyecto
"""

import re
from pathlib import Path

def fix_run_py():
    """Corregir el error en run.py"""
    
    run_py_path = Path("run.py")
    
    if not run_py_path.exists():
        print("❌ Archivo run.py no encontrado")
        return False
    
    try:
        # Leer contenido actual
        with open(run_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Buscar y reemplazar la línea problemática
        # Patrón para encontrar la llamada a register_blueprints con 2 argumentos
        pattern = r'register_blueprints\(app,\s*logger\)'
        replacement = 'register_blueprints(app)'
        
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            print("✅ Corregida llamada a register_blueprints(app, logger)")
        else:
            print("⚠️ No se encontró la línea específica, buscando alternativas...")
            
            # Buscar patrones alternativos
            alt_patterns = [
                r'registered_blueprints\s*=\s*register_blueprints\(app,.*?\)',
                r'register_blueprints\(app,.*?\)'
            ]
            
            for pattern in alt_patterns:
                if re.search(pattern, content):
                    content = re.sub(pattern, 'register_blueprints(app)', content)
                    print("✅ Corregida llamada alternativa a register_blueprints")
                    break
        
        # También asegurar que la función register_blueprints tenga la definición correcta
        func_pattern = r'def register_blueprints\(app\):'
        if re.search(func_pattern, content):
            print("✅ Definición de función register_blueprints correcta")
        else:
            # Buscar la definición actual y corregirla si es necesaria
            current_def_pattern = r'def register_blueprints\([^)]+\):'
            if re.search(current_def_pattern, content):
                content = re.sub(current_def_pattern, 'def register_blueprints(app, logger=None):', content)
                print("✅ Actualizada definición de register_blueprints para aceptar logger opcional")
        
        # Escribir el archivo corregido
        with open(run_py_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Archivo run.py corregido exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error corrigiendo run.py: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Corrigiendo error en run.py...")
    if fix_run_py():
        print("\n🚀 Ahora puedes ejecutar: python run.py")
    else:
        print("\n❌ Corrección manual necesaria")
        print("Busca en run.py la línea que dice:")
        print("   register_blueprints(app, logger)")
        print("Y cámbiala por:")
        print("   register_blueprints(app)")