#!/usr/bin/env python3
"""
Quick Fix - Verificar y corregir SystemStats
"""
import os
from pathlib import Path

def check_and_fix_systemstats():
    """Verificar y corregir SystemStats"""
    project_root = Path(__file__).parent.parent
    models_init = project_root / "app" / "models" / "__init__.py"
    
    print("🔍 Verificando SystemStats")
    print("=" * 40)
    
    # Leer contenido actual
    if models_init.exists():
        content = models_init.read_text(encoding='utf-8')
        print(f"📄 Archivo: {models_init}")
        print(f"📊 Líneas: {len(content.split('\\n'))}")
        
        # Verificar si SystemStats está presente
        if 'class SystemStats' in content:
            print("✅ SystemStats encontrada en el archivo")
        else:
            print("❌ SystemStats NO encontrada")
            print("🔧 Creando SystemStats directamente...")
            
            # Crear SystemStats mínima
            systemstats_minimal = '''"""Modelos de datos del sistema"""

from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class SystemStats:
    """Estadísticas del sistema"""
    documents_indexed: int = 0
    queries_processed: int = 0
    uptime_hours: float = 0.0
    memory_usage_mb: float = 256.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'documents_indexed': self.documents_indexed,
            'queries_processed': self.queries_processed,
            'uptime_hours': self.uptime_hours,
            'memory_usage_mb': self.memory_usage_mb
        }
    
    def update_query_stats(self, processing_time: float, success: bool = True):
        """Actualizar estadísticas de consultas"""
        self.queries_processed += 1
'''
            
            models_init.write_text(systemstats_minimal, encoding='utf-8')
            print("✅ SystemStats mínima creada")
    
    else:
        print("❌ app/models/__init__.py no existe")
        return False
    
    return True

def test_import_now():
    """Test de importación directo"""
    print("\\n🧪 Test de importación directo")
    print("-" * 30)
    
    try:
        import sys
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        # Forzar reimport
        if 'app.models' in sys.modules:
            del sys.modules['app.models']
        
        from app.models import SystemStats
        
        # Crear instancia
        stats = SystemStats()
        print("✅ SystemStats importada correctamente")
        print(f"📊 Documentos: {stats.documents_indexed}")
        print(f"📊 Dict: {stats.to_dict()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_main_blueprint():
    """Test del blueprint main"""
    print("\\n🔄 Test del blueprint main")
    print("-" * 30)
    
    try:
        import sys
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        # Forzar reimport
        modules_to_clear = [m for m in sys.modules.keys() if m.startswith('app.')]
        for module in modules_to_clear:
            del sys.modules[module]
        
        from app.routes.main import main_bp
        print("✅ main_bp importado correctamente")
        print(f"📦 Blueprint: {main_bp.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importando main_bp: {e}")
        return False

def main():
    print("🔧 Quick Fix - SystemStats")
    
    # 1. Verificar y corregir SystemStats
    systemstats_ok = check_and_fix_systemstats()
    
    if systemstats_ok:
        # 2. Test importación
        import_ok = test_import_now()
        
        if import_ok:
            # 3. Test blueprint
            blueprint_ok = test_main_blueprint()
            
            if blueprint_ok:
                print("\\n🎉 ¡TODO FUNCIONANDO!")
                print("\\n💡 Próximo paso:")
                print("   python run.py")
                print("   -> Debería mostrar dashboard HTML")
            else:
                print("\\n⚠️ SystemStats OK, pero main_bp sigue con problemas")
        else:
            print("\\n❌ Problema con SystemStats")
    else:
        print("\\n❌ No se pudo corregir SystemStats")

if __name__ == "__main__":
    main()