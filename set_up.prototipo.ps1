# Script de Setup para Prototipo_chatbot
# Ejecutar desde PowerShell en la ubicación deseada

Write-Host "🚀 Creando estructura del proyecto Prototipo_chatbot..." -ForegroundColor Green

# Crear directorio principal
New-Item -ItemType Directory -Name "Prototipo_chatbot" -Force
Set-Location "Prototipo_chatbot"

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
    New-Item -ItemType Directory -Path $folder -Force
    Write-Host "Creada carpeta: $folder" -ForegroundColor Cyan
}

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
    New-Item -ItemType File -Path $init_file -Force
    Write-Host " Creado: $init_file" -ForegroundColor Yellow
}

Write-Host " Estructura base creada exitosamente!" -ForegroundColor Green
Write-Host " Ubicación: $PWD" -ForegroundColor White