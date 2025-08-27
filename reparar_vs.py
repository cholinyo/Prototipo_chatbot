#!/usr/bin/env python3
"""
Script para corregir el método __post_init__ de DocumentChunk
Preservar objetos DocumentMetadata en lugar de convertirlos a dict
"""

import os
from pathlib import Path

def fix_document_chunk():
    """Arreglar el método __post_init__ problemático de DocumentChunk"""
    
    file_path = Path("app/models/document.py")
    
    if not file_path.exists():
        print(f"❌ Archivo no encontrado: {file_path}")
        return False
    
    print(f"🔧 Arreglando DocumentChunk en: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error leyendo archivo: {e}")
        return False
    
    # Código problemático a reemplazar
    old_code = '''    def __post_init__(self):
        """Validaciones y setup automático"""
        # Asegurar que el ID no esté vacío
        if not self.id:
            import uuid
            self.id = str(uuid.uuid4())
        
        # Asegurar que metadata sea un dict
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        
        # Asegurar que chunk_index sea válido
        if self.chunk_index < 0:
            self.chunk_index = 0'''
    
    # Código corregido
    new_code = '''    def __post_init__(self):
        """Validaciones y setup automático"""
        # Asegurar que el ID no esté vacío
        if not self.id:
            import uuid
            self.id = str(uuid.uuid4())
        
        # ✅ CORRECCIÓN: Preservar DocumentMetadata, solo convertir None o tipos inválidos
        if self.metadata is None:
            self.metadata = {}
        elif hasattr(self.metadata, '__dict__') and not isinstance(self.metadata, dict):
            # Si es un objeto tipo DocumentMetadata, mantenerlo como está
            pass
        elif not isinstance(self.metadata, (dict, object)):
            # Solo convertir si no es ni dict ni objeto con atributos
            self.metadata = {}
        
        # Asegurar que chunk_index sea válido
        if self.chunk_index < 0:
            self.chunk_index = 0'''
    
    # También actualizar el type hint para ser más flexible
    old_type_hint = "    metadata: Dict[str, Any]"
    new_type_hint = "    metadata: Union[Dict[str, Any], DocumentMetadata, Any]"
    
    # Verificar si el código problemático existe
    if old_code not in content:
        print("❌ No se encontró el código problemático exacto")
        
        # Buscar patrones problemáticos alternativos
        if "if not isinstance(self.metadata, dict):" in content:
            print("🔍 Encontrado patrón problemático alternativo")
            
            # Reemplazo más específico
            problem_pattern = '''        # Asegurar que metadata sea un dict
        if not isinstance(self.metadata, dict):
            self.metadata = {}'''
            
            solution_pattern = '''        # ✅ CORRECCIÓN: Preservar DocumentMetadata
        if self.metadata is None:
            self.metadata = {}
        elif hasattr(self.metadata, '__dict__') and not isinstance(self.metadata, dict):
            # Si es un objeto tipo DocumentMetadata, mantenerlo como está
            pass'''
            
            if problem_pattern in content:
                content = content.replace(problem_pattern, solution_pattern)
                print("✅ Patrón alternativo reemplazado")
            else:
                print("❌ No se pudo encontrar el patrón problemático")
                return False
        else:
            print("✅ No se encuentra código problemático. El archivo puede estar correcto.")
            return True
    else:
        # Reemplazar código completo
        content = content.replace(old_code, new_code)
        print("✅ Método __post_init__ reemplazado")
    
    # Actualizar type hint para ser más flexible
    if old_type_hint in content:
        # Añadir import para Union si no existe
        if "from typing import" in content and "Union" not in content:
            content = content.replace(
                "from typing import Dict, Any, Optional, List, TYPE_CHECKING",
                "from typing import Dict, Any, Optional, List, TYPE_CHECKING, Union"
            )
        
        content = content.replace(old_type_hint, new_type_hint)
        print("✅ Type hint actualizado")
    
    # Crear backup
    backup_path = file_path.with_suffix('.py.backup')
    try:
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(open(file_path, 'r', encoding='utf-8').read())
        print(f"📁 Backup creado: {backup_path}")
    except Exception as e:
        print(f"⚠️ No se pudo crear backup: {e}")
    
    # Escribir archivo corregido
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ Archivo corregido exitosamente")
        return True
    except Exception as e:
        print(f"❌ Error escribiendo archivo corregido: {e}")
        return False

def verify_fix():
    """Verificar que la corrección se aplicó"""
    file_path = Path("app/models/document.py")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return False
    
    # Verificar que el código problemático no existe
    problematic_patterns = [
        "if not isinstance(self.metadata, dict):\n            self.metadata = {}"
    ]
    
    for pattern in problematic_patterns:
        if pattern in content:
            print(f"❌ Código problemático encontrado: {pattern}")
            return False
    
    # Verificar que el código corregido existe
    if "# ✅ CORRECCIÓN: Preservar DocumentMetadata" in content:
        print("✅ Código corregido encontrado")
        return True
    else:
        print("❌ Código corregido no encontrado")
        return False

if __name__ == "__main__":
    print("🔧 Script para corregir DocumentChunk.__post_init__")
    print("="*55)
    
    success = fix_document_chunk()
    
    if success:
        print("\n🔍 Verificando corrección...")
        if verify_fix():
            print("\n✅ CORRECCIÓN APLICADA EXITOSAMENTE")
            print("Ahora el DocumentChunk preservará objetos DocumentMetadata")
            print("Ejecuta el test:")
            print("python test_integracion_vector_store.py")
        else:
            print("\n❌ La corrección no se aplicó correctamente")
    else:
        print("\n❌ No se pudo aplicar la corrección")
        print("Revisa el archivo manualmente")
    
    print("="*55)