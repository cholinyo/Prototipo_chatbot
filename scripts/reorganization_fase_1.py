#!/usr/bin/env python3
"""
Script de Reorganización Fase 1 - TFM Chatbot RAG
Corrige archivos mal ubicados y nombres problemáticos

Autor: Vicente Caruncho Ramos
TFM: Sistemas Inteligentes - UJI
"""
import os
import shutil
import sys
from pathlib import Path


class ProjectReorganizer:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.moves_completed = []
        self.errors = []
        self.warnings = []
    
    def print_header(self):
        """Mostrar cabecera del script"""
        print("🔧 REORGANIZACIÓN PROYECTO TFM - FASE 1")
        print("=" * 50)
        print(f"📁 Directorio: {self.project_root}")
        print(f"🎯 Objetivo: Corregir archivos mal ubicados")
        print("-" * 50)
    
    def backup_important_files(self):
        """Crear backup de archivos importantes antes de mover"""
        print("\n💾 Creando backup de seguridad...")
        
        backup_dir = self.project_root / "backup_reorganization"
        backup_dir.mkdir(exist_ok=True)
        
        important_files = [
            "setup_completo.ps1",
            "Documentación para TFM+",
            ".env"  # Si existe (no debería estar en repo)
        ]
        
        for file_name in important_files:
            source_path = self.project_root / file_name
            if source_path.exists():
                try:
                    if source_path.is_file():
                        shutil.copy2(source_path, backup_dir)
                    else:
                        shutil.copytree(source_path, backup_dir / file_name, dirs_exist_ok=True)
                    print(f"✅ Backup creado: {file_name}")
                except Exception as e:
                    self.warnings.append(f"No se pudo hacer backup de {file_name}: {e}")
                    print(f"⚠️ Warning backup {file_name}: {e}")
        
        print(f"📦 Backup guardado en: {backup_dir}")
    
    def phase1_move_misplaced_files(self):
        """Fase 1: Mover archivos mal ubicados"""
        print("\n🔄 FASE 1: Moviendo archivos mal ubicados...")
        
        # Definir movimientos necesarios
        moves = [
            # (source, destination, description)
            ("setup_completo.ps1", "scripts/powershell/setup_completo.ps1", "Script PowerShell"),
            ("Documentación para TFM+", "docs_tfm", "Documentación académica"),
        ]
        
        for source, destination, description in moves:
            success = self._move_file_or_dir(source, destination, description)
            if success:
                self.moves_completed.append((source, destination))
    
    def phase2_create_missing_directories(self):
        """Fase 2: Crear directorios faltantes"""
        print("\n📁 FASE 2: Creando estructura de directorios...")
        
        directories_to_create = [
            # Estructura app/
            ("app/utils", "Funciones auxiliares"),
            ("app/exceptions", "Excepciones personalizadas"),
            ("app/middleware", "Middleware Flask"),
            
            # Estructura tests/
            ("tests/integration", "Tests de integración"),
            ("tests/fixtures", "Datos de prueba"),
            ("tests/data", "Datasets para testing"),
            
            # Estructura docs/
            ("docs/api", "Documentación API"),
            ("docs/deployment", "Guías de despliegue"),
            ("docs/architecture", "Diagramas de arquitectura"),
            
            # Estructura templates/
            ("app/templates/admin", "Templates administrativos"),
            ("app/templates/chat", "Templates de chat"),
            ("app/templates/errors", "Templates de error"),
            ("app/templates/layouts", "Layouts base"),
            ("app/templates/components", "Componentes reutilizables"),
            
            # Estructura config/
            ("config/environments", "Configuraciones por entorno"),
            ("config/models", "Configuraciones de modelos"),
            
            # Estructura scripts/
            ("scripts/powershell", "Scripts PowerShell"),
            ("scripts/setup", "Scripts de configuración"),
            ("scripts/maintenance", "Scripts de mantenimiento"),
        ]
        
        for dir_path, description in directories_to_create:
            success = self._create_directory(dir_path, description)
    
    def phase3_create_init_files(self):
        """Fase 3: Crear archivos __init__.py faltantes"""
        print("\n📄 FASE 3: Creando archivos __init__.py...")
        
        python_packages = [
            "app/utils",
            "app/exceptions", 
            "app/middleware",
            "tests",
            "tests/integration",
            "tests/fixtures",
        ]
        
        for package_dir in python_packages:
            init_file = self.project_root / package_dir / "__init__.py"
            if not init_file.exists() and (self.project_root / package_dir).exists():
                try:
                    init_file.write_text('"""Package initialization"""\n')
                    print(f"✅ Creado: {package_dir}/__init__.py")
                except Exception as e:
                    self.errors.append(f"Error creando {init_file}: {e}")
                    print(f"❌ Error: {package_dir}/__init__.py - {e}")
    
    def phase4_move_specific_templates(self):
        """Fase 4: Reorganizar templates específicos"""
        print("\n📄 FASE 4: Reorganizando templates...")
        
        template_moves = [
            ("app/templates/data_sources.html", "app/templates/admin/data_sources.html", "Template fuentes datos"),
            ("app/templates/admin.html", "app/templates/admin/dashboard.html", "Dashboard admin"),
        ]
        
        for source, destination, description in template_moves:
            self._move_file_or_dir(source, destination, description)
    
    def _move_file_or_dir(self, source, destination, description):
        """Mover archivo o directorio con manejo de errores"""
        source_path = self.project_root / source
        dest_path = self.project_root / destination
        
        if not source_path.exists():
            print(f"⏭️ Saltando: {source} (no existe)")
            return False
        
        try:
            # Crear directorio destino si no existe
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Mover archivo/directorio
            shutil.move(str(source_path), str(dest_path))
            print(f"✅ Movido: {source} → {destination}")
            print(f"   📝 {description}")
            return True
            
        except Exception as e:
            self.errors.append(f"Error moviendo {source}: {e}")
            print(f"❌ Error moviendo {source}: {e}")
            return False
    
    def _create_directory(self, dir_path, description):
        """Crear directorio con manejo de errores"""
        full_path = self.project_root / dir_path
        
        try:
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Creado: {dir_path}")
            print(f"   📝 {description}")
            return True
        except Exception as e:
            self.errors.append(f"Error creando directorio {dir_path}: {e}")
            print(f"❌ Error creando {dir_path}: {e}")
            return False
    
    def show_summary(self):
        """Mostrar resumen de la reorganización"""
        print("\n" + "=" * 50)
        print("📋 RESUMEN DE REORGANIZACIÓN FASE 1")
        print("=" * 50)
        
        print(f"\n✅ Movimientos completados: {len(self.moves_completed)}")
        for source, dest in self.moves_completed:
            print(f"   📁 {source} → {dest}")
        
        if self.warnings:
            print(f"\n⚠️ Advertencias: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"   ⚠️ {warning}")
        
        if self.errors:
            print(f"\n❌ Errores: {len(self.errors)}")
            for error in self.errors:
                print(f"   ❌ {error}")
        else:
            print("\n🎉 ¡Reorganización completada sin errores!")
        
        print(f"\n💡 Próximos pasos:")
        print("   1. Verificar que todo funciona: python run.py")
        print("   2. Actualizar imports si es necesario")
        print("   3. Ejecutar tests: pytest")
        print("   4. Commit cambios: git add . && git commit -m 'refactor: reorganizar estructura proyecto'")
    
    def run_reorganization(self):
        """Ejecutar reorganización completa"""
        self.print_header()
        
        try:
            # Crear backup
            self.backup_important_files()
            
            # Ejecutar fases
            self.phase1_move_misplaced_files()
            self.phase2_create_missing_directories()
            self.phase3_create_init_files()
            self.phase4_move_specific_templates()
            
            # Mostrar resumen
            self.show_summary()
            
            return len(self.errors) == 0
            
        except KeyboardInterrupt:
            print("\n⏹️ Reorganización cancelada por el usuario")
            return False
        except Exception as e:
            print(f"\n💥 Error crítico: {e}")
            return False


def main():
    """Función principal"""
    print("🎓 TFM Vicente Caruncho - Reorganización Estructura")
    print("🏛️ Prototipo Chatbot RAG para Administraciones Locales")
    
    reorganizer = ProjectReorganizer()
    
    # Preguntar confirmación
    print(f"\n📁 Directorio del proyecto: {reorganizer.project_root}")
    confirm = input("\n¿Proceder con la reorganización? (s/N): ").lower().strip()
    
    if confirm not in ['s', 'si', 'sí', 'y', 'yes']:
        print("❌ Reorganización cancelada")
        sys.exit(0)
    
    # Ejecutar reorganización
    success = reorganizer.run_reorganization()
    
    if success:
        print("\n🚀 ¡Reorganización exitosa! El proyecto está ahora mejor estructurado.")
        sys.exit(0)
    else:
        print("\n⚠️ Reorganización completada con errores. Revisar el resumen.")
        sys.exit(1)


if __name__ == "__main__":
    main()