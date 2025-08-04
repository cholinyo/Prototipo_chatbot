# Script completo para crear Prototipo_chatbot
# Ejecutar desde PowerShell en la ubicación deseada

Write-Host "🚀 Creando proyecto Prototipo_chatbot completo..." -ForegroundColor Green

# Crear directorio principal
New-Item -ItemType Directory -Name "Prototipo_chatbot" -Force
Set-Location "Prototipo_chatbot"

Write-Host "📁 Creando estructura de carpetas..." -ForegroundColor Cyan

# Crear estructura de carpetas
$folders = @(
    "app",
    "app/core",
    "app/models", 
    "app/services",
    "app/services/rag",
    "app/services/ingestion",
    "app/routes",
    "app/templates",
    "app/templates/errors",
    "app/static",
    "app/static/css",
    "app/static/js",
    "app/utils",
    "config",
    "data",
    "data/documents",
    "data/vectorstore",
    "data/vectorstore/faiss",
    "data/vectorstore/chromadb", 
    "data/temp",
    "scripts",
    "tests",
    "tests/unit",
    "tests/integration",
    "docker",
    "logs"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Path $folder -Force | Out-Null
    Write-Host "✅ $folder" -ForegroundColor Yellow
}

Write-Host "📄 Creando archivos __init__.py..." -ForegroundColor Cyan

# Crear archivos __init__.py
$init_files = @(
    "app/__init__.py",
    "app/core/__init__.py",
    "app/models/__init__.py", 
    "app/services/__init__.py",
    "app/services/rag/__init__.py",
    "app/services/ingestion/__init__.py",
    "app/routes/__init__.py",
    "app/utils/__init__.py",
    "tests/__init__.py"
)

foreach ($init_file in $init_files) {
    New-Item -ItemType File -Path $init_file -Force | Out-Null
}

Write-Host "📋 Creando requirements.txt..." -ForegroundColor Cyan

# Crear requirements.txt
@"
# Prototipo_chatbot - Dependencies

# ===== CORE FRAMEWORK =====
flask>=2.3.0
pydantic>=2.0.0
pyyaml>=6.0.1
python-dotenv>=1.0.0
click>=8.1.0

# ===== AI & ML =====
sentence-transformers>=2.2.0
transformers>=4.30.0
torch>=2.0.0
numpy>=1.24.0

# ===== VECTOR DATABASES =====
faiss-cpu>=1.7.4
chromadb>=0.4.15

# ===== LOCAL MODELS =====
openai>=1.3.0

# ===== DOCUMENT PROCESSING =====
PyPDF2>=3.0.1
python-docx>=0.8.11
openpyxl>=3.1.0
pandas>=2.0.0
beautifulsoup4>=4.12.0
lxml>=4.9.0

# ===== WEB SCRAPING =====
requests>=2.31.0
selenium>=4.15.0
webdriver-manager>=4.0.0

# ===== DATABASE =====
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0

# ===== LOGGING & MONITORING =====
structlog>=23.0.0
colorlog>=6.7.0

# ===== DEVELOPMENT =====
pytest>=7.4.0
pytest-asyncio>=0.21.0
black>=23.0.0
ruff>=0.1.0

# ===== DEPLOYMENT =====
gunicorn>=21.2.0
"@ | Out-File -FilePath "requirements.txt" -Encoding UTF8

Write-Host "⚙️ Creando .env.example..." -ForegroundColor Cyan

# Crear .env.example
@"
# Prototipo_chatbot - Variables de Entorno

# ===== FLASK CONFIGURATION =====
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-change-in-production

# ===== OPENAI API =====
OPENAI_API_KEY=sk-your-openai-api-key-here

# ===== MODELS CONFIGURATION =====
DEFAULT_LOCAL_MODEL=ollama:llama3.1:8b
DEFAULT_OPENAI_MODEL=gpt-4
EMBEDDING_MODEL=all-MiniLM-L6-v2

# ===== VECTOR STORES =====
DEFAULT_VECTOR_STORE=faiss
FAISS_INDEX_PATH=data/vectorstore/faiss
CHROMADB_PATH=data/vectorstore/chromadb

# ===== RAG CONFIGURATION =====
RAG_K_DEFAULT=5
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# ===== LOGGING =====
LOG_LEVEL=INFO
LOG_FILE=logs/prototipo_chatbot.log

# ===== DEVELOPMENT =====
RELOAD_ON_CHANGE=True
DEVELOPMENT_MODE=True
"@ | Out-File -FilePath ".env.example" -Encoding UTF8

Write-Host "🔧 Creando run.py..." -ForegroundColor Cyan

# Crear run.py
@"
#!/usr/bin/env python3
"""
Prototipo_chatbot - Punto de entrada principal
Chatbot RAG para Administraciones Locales
"""
import sys
import os
from pathlib import Path

# Añadir el directorio raíz al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Función principal de arranque"""
    print("🚀 Iniciando Prototipo_chatbot...")
    print("📁 Directorio del proyecto:", project_root)
    
    try:
        # Verificar estructura básica
        required_dirs = ['app', 'config', 'data', 'logs']
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Creado directorio: {dir_name}")
        
        # Importar después de verificar estructura
        from app import create_app
        
        # Crear aplicación Flask
        app = create_app()
        
        print("✅ Aplicación Flask creada exitosamente")
        print("🌐 Accede a: http://localhost:5000")
        print("⚠️  Usa Ctrl+C para detener el servidor")
        
        # Iniciar servidor
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=True,
            use_reloader=True
        )
        
    except KeyboardInterrupt:
        print("\n⏹️  Aplicación detenida por el usuario")
    except Exception as e:
        print(f"❌ Error crítico: {e}")
        print("💡 Asegúrate de que todas las dependencias están instaladas:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()
"@ | Out-File -FilePath "run.py" -Encoding UTF8

Write-Host "🎯 Creando app/__init__.py básico..." -ForegroundColor Cyan

# Crear app/__init__.py básico
@"
"""
Prototipo_chatbot - Aplicación Flask principal
"""
from flask import Flask, render_template

def create_app():
    """Factory de aplicación Flask"""
    
    # Crear aplicación Flask
    app = Flask(__name__)
    
    # Configurar Flask
    app.config.update(
        SECRET_KEY="prototipo-chatbot-dev-key",
        DEBUG=True,
        TESTING=False
    )
    
    # Ruta de salud básica
    @app.route('/health')
    def health_check():
        return {
            'status': 'healthy',
            'version': '1.0.0',
            'name': 'Prototipo_chatbot'
        }
    
    # Ruta principal
    @app.route('/')
    def index():
        return '''
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Prototipo_chatbot</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
            <div class="container mt-5">
                <div class="text-center">
                    <h1 class="display-4">🤖 Prototipo_chatbot</h1>
                    <p class="lead">Sistema RAG para Administraciones Locales</p>
                    <div class="alert alert-success">
                        <h4>✅ ¡Aplicación funcionando correctamente!</h4>
                        <p>La estructura base del proyecto está lista.</p>
                    </div>
                    <div class="row mt-4">
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-body">
                                    <h5 class="card-title">📊 Estado</h5>
                                    <p class="card-text">Sistema inicializado</p>
                                    <span class="badge bg-success">Activo</span>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-body">
                                    <h5 class="card-title">🔧 Siguiente</h5>
                                    <p class="card-text">Implementar modelos IA</p>
                                    <span class="badge bg-warning">Pendiente</span>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card">
                                <div class="card-body">
                                    <h5 class="card-title">📚 TFM</h5>
                                    <p class="card-text">Chatbot RAG</p>
                                    <span class="badge bg-info">En desarrollo</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        '''
    
    return app
"@ | Out-File -FilePath "app/__init__.py" -Encoding UTF8

Write-Host "📖 Creando README.md..." -ForegroundColor Cyan

# Crear README.md básico
@"
# 🤖 Prototipo_chatbot

Sistema de Chatbot RAG para Administraciones Locales - TFM

## 🚀 Inicio Rápido

1. **Crear entorno virtual:**
   ```bash
   python -m venv venv
   ```

2. **Activar entorno:**
   ```bash
   # Windows:
   venv\Scripts\activate.bat
   
   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Ejecutar aplicación:**
   ```bash
   python run.py
   ```

5. **Abrir navegador:**
   http://localhost:5000

## ✅ Estado Actual

- ✅ Estructura del proyecto creada
- ✅ Aplicación Flask básica funcionando
- ⏳ Pendiente: Implementar modelos IA
- ⏳ Pendiente: Sistema RAG
- ⏳ Pendiente: Ingesta multimodal

## 🎓 TFM - Vicente Caruncho Ramos

Trabajo Final de Máster en Sistemas Inteligentes
"@ | Out-File -FilePath "README.md" -Encoding UTF8

Write-Host "🎉 ¡Proyecto creado exitosamente!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Próximos pasos:" -ForegroundColor Yellow
Write-Host "1. cd Prototipo_chatbot" -ForegroundColor White
Write-Host "2. python -m venv venv" -ForegroundColor White  
Write-Host "3. venv\Scripts\activate.bat" -ForegroundColor White
Write-Host "4. pip install --upgrade pip" -ForegroundColor White
Write-Host "5. pip install -r requirements.txt" -ForegroundColor White
Write-Host "6. python run.py" -ForegroundColor White
Write-Host ""
Write-Host "🌐 Luego abrir: http://localhost:5000" -ForegroundColor Cyan