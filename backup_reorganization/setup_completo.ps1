# =====================================================
# SETUP CON ENTORNO VIRTUAL EXISTENTE (tm_env)
# TFM Vicente Caruncho - Prototipo Chatbot RAG
# =====================================================

Write-Host "🎓 TFM VICENTE CARUNCHO - SETUP CON ENTORNO ACTUAL" -ForegroundColor Cyan
Write-Host "🏛️ Usando entorno virtual: tm_env" -ForegroundColor Green
Write-Host "📁 Directorio: C:\Users\vcaruncho\Documents\Desarrollo SW\Prototipo_chabot" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Gray

# 1. VERIFICAR QUE ESTAMOS EN EL ENTORNO CORRECTO
Write-Host "`n🐍 Verificando entorno virtual..." -ForegroundColor Yellow
$envName = $env:VIRTUAL_ENV
if ($envName) {
    Write-Host "✅ Entorno virtual activo: $envName" -ForegroundColor Green
} else {
    Write-Host "⚠️ No se detecta entorno virtual activo" -ForegroundColor Yellow
    Write-Host "💡 Asegurándonos de que tm_env esté activo..." -ForegroundColor Cyan
}

# 2. INSTALAR/ACTUALIZAR DEPENDENCIAS NECESARIAS
Write-Host "`n📦 Instalando dependencias necesarias..." -ForegroundColor Yellow

# Dependencias críticas para el diagnóstico
$criticalPackages = @(
    "requests",
    "python-dotenv", 
    "flask",
    "openai"
)

foreach ($package in $criticalPackages) {
    Write-Host "Instalando $package..." -ForegroundColor Cyan
    try {
        pip install $package --quiet
        Write-Host "✅ $package instalado" -ForegroundColor Green
    } catch {
        Write-Host "❌ Error instalando $package" -ForegroundColor Red
    }
}

# 3. INSTALAR REQUIREMENTS.TXT SI EXISTE
if (Test-Path "requirements.txt") {
    Write-Host "`n📋 Instalando desde requirements.txt..." -ForegroundColor Yellow
    try {
        pip install -r requirements.txt
        Write-Host "✅ Dependencias de requirements.txt instaladas" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Algunos paquetes pueden haber fallado, continuando..." -ForegroundColor Yellow
    }
}

# 4. CREAR ARCHIVO .env
Write-Host "`n🔧 Configurando archivo .env..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "✅ Archivo .env creado desde .env.example" -ForegroundColor Green
    } else {
        # Crear .env básico
        $envContent = @"
# ===== CONFIGURACIÓN GENERAL =====
PROJECT_NAME=Prototipo_chatbot
PROJECT_VERSION=1.0.0
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=tu-clave-secreta-segura

# ===== MODELOS DE LENGUAJE =====
# Modelos locales (Ollama)
DEFAULT_LOCAL_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
AVAILABLE_LOCAL_MODELS=llama3.2:3b,mistral:7b,gemma2:2b

# Modelos OpenAI (CAMBIAR POR TU API KEY REAL)
OPENAI_API_KEY=sk-tu-api-key-aqui
DEFAULT_OPENAI_MODEL=gpt-4o-mini
AVAILABLE_OPENAI_MODELS=gpt-4o,gpt-4o-mini,gpt-3.5-turbo

# ===== EMBEDDINGS =====
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
EMBEDDING_CACHE_DIR=data/cache/embeddings

# ===== VECTOR STORES =====
DEFAULT_VECTOR_STORE=faiss
FAISS_INDEX_PATH=data/vectorstore/faiss
CHROMADB_PATH=data/vectorstore/chromadb

# ===== RAG CONFIGURATION =====
RAG_K_DEFAULT=5
CHUNK_SIZE=500
CHUNK_OVERLAP=50
"@
        $envContent | Out-File -FilePath ".env" -Encoding UTF8
        Write-Host "✅ Archivo .env básico creado" -ForegroundColor Green
    }
} else {
    Write-Host "✅ Archivo .env ya existe" -ForegroundColor Green
}

# 5. CREAR DIRECTORIOS NECESARIOS
Write-Host "`n📁 Creando estructura de directorios..." -ForegroundColor Yellow
$directories = @(
    "data",
    "data\vectorstore", 
    "data\vectorstore\faiss",
    "data\vectorstore\chromadb",
    "data\cache",
    "data\cache\embeddings",
    "data\reports",
    "data\temp",
    "logs"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✅ Creado: $dir" -ForegroundColor Green
    }
}

# 6. VERIFICAR DEPENDENCIAS PYTHON
Write-Host "`n🔍 Verificando dependencias Python..." -ForegroundColor Yellow
$testPackages = @{
    "requests" = "import requests; print('✅ requests OK')"
    "flask" = "import flask; print('✅ flask OK')" 
    "dotenv" = "from dotenv import load_dotenv; print('✅ python-dotenv OK')"
    "openai" = "import openai; print('✅ openai OK')"
}

foreach ($pkg in $testPackages.Keys) {
    try {
        $result = python -c $testPackages[$pkg] 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host $result -ForegroundColor Green
        } else {
            Write-Host "❌ $pkg no funciona correctamente" -ForegroundColor Red
        }
    } catch {
        Write-Host "❌ Error verificando $pkg" -ForegroundColor Red
    }
}

# 7. CREAR SCRIPT DE DIAGNÓSTICO SIMPLIFICADO
Write-Host "`n🔧 Creando script de diagnóstico simplificado..." -ForegroundColor Yellow
$diagnosticoSimple = @"
#!/usr/bin/env python3
"""
Diagnóstico simplificado - Solo verifica configuración básica
"""
import os
from pathlib import Path

def check_env_file():
    print("🔧 VERIFICANDO ARCHIVO .env...")
    env_path = Path(".env")
    if env_path.exists():
        print("✅ Archivo .env existe")
        
        # Leer contenido
        with open(env_path, 'r') as f:
            content = f.read()
        
        # Verificar OpenAI
        if "OPENAI_API_KEY=" in content:
            for line in content.split('\n'):
                if line.startswith('OPENAI_API_KEY='):
                    key = line.split('=', 1)[1].strip()
                    if key and key != 'sk-tu-api-key-aqui' and key.startswith('sk-'):
                        print("✅ OPENAI_API_KEY configurada correctamente")
                    else:
                        print("❌ OPENAI_API_KEY necesita configuración")
                        print("💡 Edita .env: OPENAI_API_KEY=sk-tu-api-key-real")
                    break
        else:
            print("❌ OPENAI_API_KEY no encontrada")
    else:
        print("❌ Archivo .env no existe")

def check_ollama():
    print("\n🦙 VERIFICANDO OLLAMA...")
    try:
        import subprocess
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ Ollama instalado: {result.stdout.strip()}")
            
            # Verificar servidor
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    print("✅ Servidor Ollama ejecutándose")
                    models_data = response.json()
                    models = [model['name'] for model in models_data.get('models', [])]
                    if models:
                        print(f"✅ Modelos disponibles: {', '.join(models)}")
                    else:
                        print("❌ No hay modelos instalados")
                        print("💡 Ejecutar: ollama pull llama3.2:3b")
                else:
                    print("❌ Servidor Ollama no responde")
            except:
                print("❌ No se puede conectar a Ollama")
                print("💡 Ejecutar: ollama serve")
        else:
            print("❌ Ollama no instalado")
            print("💡 Descargar de: https://ollama.ai/download")
    except:
        print("❌ Error verificando Ollama")

def main():
    print("🎓 DIAGNÓSTICO SIMPLIFICADO - TFM Vicente Caruncho")
    print("=" * 50)
    check_env_file()
    check_ollama()
    print("\n✅ Diagnóstico completado")

if __name__ == "__main__":
    main()
"@

$diagnosticoSimple | Out-File -FilePath "diagnostico_simple.py" -Encoding UTF8
Write-Host "✅ Script diagnostico_simple.py creado" -ForegroundColor Green

# 8. INSTRUCCIONES FINALES
Write-Host "`n" + "=" * 70 -ForegroundColor Gray
Write-Host "✅ SETUP COMPLETADO CON ENTORNO tm_env" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Gray

Write-Host "`n🔍 Ahora ejecuta el diagnóstico:" -ForegroundColor Cyan
Write-Host "python diagnostico_simple.py" -ForegroundColor Yellow

Write-Host "`n⚠️ Configuraciones que pueden estar pendientes:" -ForegroundColor Yellow
Write-Host "• OpenAI API Key: Editar .env con tu clave real" -ForegroundColor White
Write-Host "• Ollama: Instalar y ejecutar 'ollama serve'" -ForegroundColor White
Write-Host "• Modelos: 'ollama pull llama3.2:3b'" -ForegroundColor White

Write-Host "`n🚀 Una vez configurado, iniciar con:" -ForegroundColor Cyan
Write-Host "python run.py" -ForegroundColor Yellow