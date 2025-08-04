#!/usr/bin/env python3
"""
Script de inicialización del proyecto Prototipo_chatbot
Configura el entorno y verifica dependencias
"""
import os
import sys
import subprocess
import platform
from pathlib import Path
import shutil

class ProjectSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.errors = []
        self.warnings = []
        
    def run_setup(self):
        """Ejecutar setup completo del proyecto"""
        print("🚀 Configurando Prototipo_chatbot...")
        print(f"📁 Directorio del proyecto: {self.project_root}")
        print("=" * 60)
        
        # Verificar Python
        self.check_python_version()
        
        # Crear estructura de directorios
        self.create_directory_structure()
        
        # Verificar dependencias del sistema
        self.check_system_dependencies()
        
        # Configurar entorno virtual
        self.setup_virtual_environment()
        
        # Instalar dependencias Python
        self.install_python_dependencies()
        
        # Crear archivos de configuración
        self.create_config_files()
        
        # Verificar modelos y servicios
        self.check_ai_services()
        
        # Mostrar resumen
        self.show_summary()
        
    def check_python_version(self):
        """Verificar versión de Python"""
        print("🐍 Verificando versión de Python...")
        
        version = sys.version_info
        if version.major != 3 or version.minor < 9:
            self.errors.append("Python 3.9+ requerido. Versión actual: {}.{}.{}".format(
                version.major, version.minor, version.micro
            ))
            print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Se requiere Python 3.9+")
        else:
            print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    
    def create_directory_structure(self):
        """Crear estructura de directorios necesaria"""
        print("\n📁 Creando estructura de directorios...")
        
        directories = [
            "data",
            "data/documents",
            "data/vectorstore",
            "data/vectorstore/faiss",
            "data/vectorstore/chromadb",
            "data/temp",
            "logs",
            "tests/fixtures",
            "app/templates/errors"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ {directory}")
    
    def check_system_dependencies(self):
        """Verificar dependencias del sistema"""
        print("\n🔧 Verificando dependencias del sistema...")
        
        # Verificar Git
        if shutil.which("git"):
            print("✅ Git - Disponible")
        else:
            self.warnings.append("Git no encontrado - Recomendado para desarrollo")
            print("⚠️ Git - No encontrado")
        
        # Verificar Ollama
        if shutil.which("ollama"):
            print("✅ Ollama - Disponible")
            try:
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    models = [line.split()[0] for line in result.stdout.strip().split('\n')[1:] if line.strip()]
                    if models:
                        print(f"   📦 Modelos disponibles: {', '.join(models)}")
                    else:
                        print("   📦 No hay modelos instalados")
                        self.warnings.append("No hay modelos Ollama instalados. Usa: ollama pull llama3.2")
            except subprocess.TimeoutExpired:
                self.warnings.append("Ollama no responde - Verifica que esté ejecutándose")
        else:
            self.warnings.append("Ollama no encontrado - Los modelos locales no funcionarán")
            print("⚠️ Ollama - No encontrado")
    
    def setup_virtual_environment(self):
        """Configurar entorno virtual si no existe"""
        print("\n🔨 Configurando entorno virtual...")
        
        venv_path = self.project_root / "venv"
        
        if venv_path.exists():
            print("✅ Entorno virtual ya existe")
            return
        
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
            print("✅ Entorno virtual creado")
            
            # Determinar comando de activación según el SO
            if platform.system() == "Windows":
                activation_cmd = f"{venv_path}\\Scripts\\activate.bat"
                pip_cmd = f"{venv_path}\\Scripts\\pip.exe"
            else:
                activation_cmd = f"source {venv_path}/bin/activate"
                pip_cmd = f"{venv_path}/bin/pip"
            
            print(f"💡 Para activar: {activation_cmd}")
            
        except subprocess.CalledProcessError:
            self.errors.append("No se pudo crear el entorno virtual")
            print("❌ Error creando entorno virtual")
    
    def install_python_dependencies(self):
        """Instalar dependencias de Python"""
        print("\n📦 Instalando dependencias de Python...")
        
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            self.errors.append("Archivo requirements.txt no encontrado")
            return
        
        venv_path = self.project_root / "venv"
        if platform.system() == "Windows":
            pip_cmd = venv_path / "Scripts" / "pip.exe"
        else:
            pip_cmd = venv_path / "bin" / "pip"
        
        if not pip_cmd.exists():
            print("⚠️ Entorno virtual no encontrado, usando pip del sistema")
            pip_cmd = "pip"
        
        try:
            print("   Actualizando pip...")
            subprocess.run([str(pip_cmd), "install", "--upgrade", "pip"], check=True, capture_output=True)
            
            print("   Instalando dependencias...")
            subprocess.run([str(pip_cmd), "install", "-r", str(requirements_file)], check=True)
            print("✅ Dependencias instaladas correctamente")
            
        except subprocess.CalledProcessError as e:
            self.errors.append(f"Error instalando dependencias: {e}")
            print("❌ Error instalando dependencias")
    
    def create_config_files(self):
        """Crear archivos de configuración si no existen"""
        print("\n⚙️ Creando archivos de configuración...")
        
        config_files = [
            (".env", ".env.example"),
            ("config/data_sources.yaml", None)
        ]
        
        for config_file, example_file in config_files:
            config_path = self.project_root / config_file
            
            if config_path.exists():
                print(f"✅ {config_file} - Ya existe")
                continue
            
            if example_file:
                example_path = self.project_root / example_file
                if example_path.exists():
                    shutil.copy2(example_path, config_path)
                    print(f"✅ {config_file} - Creado desde {example_file}")
                else:
                    print(f"⚠️ {config_file} - Archivo ejemplo no encontrado")
            else:
                # Crear archivo de configuración de fuentes de datos básico
                self.create_data_sources_config(config_path)
                print(f"✅ {config_file} - Creado")
    
    def create_data_sources_config(self, config_path):
        """Crear configuración básica de fuentes de datos"""
        config_content = """# Configuración de fuentes de datos - Prototipo_chatbot
data_sources:
  documents:
    enabled: true
    paths: []
    supported_formats: [".pdf", ".docx", ".txt", ".rtf"]
    recursive: true
    
  web:
    enabled: false
    sources: []
    
  apis:
    enabled: false
    sources: []
    
  databases:
    enabled: false
    sources: []
    
  spreadsheets:
    enabled: false
    sources: []

ingestion:
  batch_size: 100
  chunk_size: 500
  chunk_overlap: 50
  embedding_model: "all-MiniLM-L6-v2"
  vector_stores: ["faiss", "chromadb"]
"""
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
    
    def check_ai_services(self):
        """Verificar servicios de IA disponibles"""
        print("\n🧠 Verificando servicios de IA...")
        
        # Verificar OpenAI API Key
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            print("✅ OpenAI API Key - Configurada")
        else:
            self.warnings.append("OpenAI API Key no configurada - Modelos OpenAI no funcionarán")
            print("⚠️ OpenAI API Key - No configurada")
        
        # Verificar sentence-transformers
        try:
            import sentence_transformers
            print("✅ sentence-transformers - Disponible")
        except ImportError:
            self.errors.append("sentence-transformers no instalado")
            print("❌ sentence-transformers - No disponible")
        
        # Verificar FAISS
        try:
            import faiss
            print("✅ FAISS - Disponible")
        except ImportError:
            self.errors.append("FAISS no instalado")
            print("❌ FAISS - No disponible")
        
        # Verificar ChromaDB
        try:
            import chromadb
            print("✅ ChromaDB - Disponible")
        except ImportError:
            self.warnings.append("ChromaDB no instalado - Solo estará disponible FAISS")
            print("⚠️ ChromaDB - No disponible")
    
    def show_summary(self):
        """Mostrar resumen del setup"""
        print("\n" + "=" * 60)
        print("📋 RESUMEN DEL SETUP")
        print("=" * 60)
        
        if not self.errors:
            print("✅ Setup completado exitosamente!")
        else:
            print("❌ Setup completado con errores:")
            for error in self.errors:
                print(f"   • {error}")
        
        if self.warnings:
            print(f"\n⚠️ Advertencias ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        print(f"\n🚀 Para iniciar la aplicación:")
        print(f"   1. Activar entorno virtual:")
        
        if platform.system() == "Windows":
            print(f"      venv\\Scripts\\activate.bat")
        else:
            print(f"      source venv/bin/activate")
        
        print(f"   2. Ejecutar aplicación:")
        print(f"      python run.py")
        
        print(f"\n📚 Archivos importantes:")
        print(f"   • Configuración: config/settings.yaml")
        print(f"   • Variables de entorno: .env")
        print(f"   • Fuentes de datos: config/data_sources.yaml")
        print(f"   • Logs: logs/prototipo_chatbot.log")
        
        if self.warnings:
            print(f"\n💡 Recomendaciones:")
            if "Ollama" in str(self.warnings):
                print(f"   • Instalar Ollama: https://ollama.ai/")
                print(f"   • Descargar modelo: ollama pull llama3.2")
            if "OpenAI" in str(self.warnings):
                print(f"   • Configurar OpenAI: Añadir OPENAI_API_KEY al archivo .env")

def main():
    """Función principal"""
    try:
        setup = ProjectSetup()
        setup.run_setup()
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error inesperado durante el setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()