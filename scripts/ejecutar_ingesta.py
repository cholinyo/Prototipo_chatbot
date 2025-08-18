#!/usr/bin/env python3
"""
Script de Ejecución de Ingesta de Documentos
TFM Vicente Caruncho - Sistemas Inteligentes

Usa tu IngestionService existente + el nuevo procesador real

Uso:
    python scripts/ejecutar_ingesta.py --help
    python scripts/ejecutar_ingesta.py --setup
    python scripts/ejecutar_ingesta.py --dir data/documentos
    python scripts/ejecutar_ingesta.py --file documento.pdf
"""

import argparse
import sys
import os
from pathlib import Path
import time

# Agregar el directorio del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_directories():
    """Crear estructura de directorios básica"""
    dirs_to_create = [
        "data/documentos",
        "data/normativas", 
        "data/ordenanzas",
        "data/vectorstore/faiss",
        "data/vectorstore/chromadb",
        "logs"
    ]
    
    print("📁 Configurando estructura de directorios...")
    for dir_path in dirs_to_create:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {dir_path}")
    
    print("✅ Estructura creada correctamente")

def create_test_documents():
    """Crear documentos de prueba"""
    docs_dir = project_root / "data" / "documentos"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    test_docs = {
        "ordenanza_ruidos.txt": """
ORDENANZA MUNICIPAL REGULADORA DE RUIDOS Y VIBRACIONES

Artículo 1. Objeto y ámbito de aplicación
La presente ordenanza tiene por objeto regular las actividades generadoras de ruidos y vibraciones en el término municipal.

Artículo 2. Niveles sonoros máximos
- Zona residencial: 55 dB(A) período diurno, 45 dB(A) período nocturno
- Zona comercial: 65 dB(A) período diurno, 55 dB(A) período nocturno  
- Zona industrial: 70 dB(A) período diurno, 60 dB(A) período nocturno

Artículo 3. Procedimiento sancionador
Las infracciones se clasifican en leves, graves y muy graves, con sanciones de 300€ a 30.000€.
        """,
        
        "licencias_actividad.txt": """
PROCEDIMIENTO DE TRAMITACIÓN DE LICENCIAS DE ACTIVIDAD

1. DOCUMENTACIÓN REQUERIDA
- Solicitud normalizada firmada
- Proyecto técnico visado por colegio profesional
- Justificante pago tasa municipal

2. PLAZOS DE TRAMITACIÓN
- Actividades inocuas: 30 días hábiles
- Actividades clasificadas: 6 meses

3. ÓRGANOS COMPETENTES
- Concejal de Urbanismo: actividades menores
- Junta de Gobierno Local: actividades mayores

4. SILENCIO ADMINISTRATIVO
Transcurridos los plazos sin resolución expresa se entiende estimada la solicitud.
        """,
        
        "servicios_sociales.txt": """
CARTA DE SERVICIOS SOCIALES MUNICIPALES

SERVICIOS DE ATENCIÓN PRIMARIA:
- Información, valoración y orientación social
- Servicio de ayuda a domicilio  
- Teleasistencia domiciliaria

PRESTACIONES ECONÓMICAS:
- Ayudas de emergencia social (máximo 600€)
- Becas comedor escolar
- Ayudas transporte personas discapacitadas

REQUISITOS GENERALES:
- Empadronamiento mínimo 12 meses
- Carecer de recursos económicos suficientes

HORARIO ATENCIÓN:
Lunes a viernes: 9:00 a 14:00 horas
        """
    }
    
    print("📝 Creando documentos de prueba...")
    for filename, content in test_docs.items():
        file_path = docs_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        print(f"   ✅ {filename}")
    
    return len(test_docs)

def process_single_file(file_path: str):
    """Procesar un archivo individual"""
    try:
        from app.services.ingestion import ingestion_service
        
        print(f"📄 Procesando archivo: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"❌ Archivo no encontrado: {file_path}")
            return False
        
        # Verificar que el servicio está disponible
        if not ingestion_service.is_available():
            print("❌ Servicio de ingesta no disponible")
            return False
        
        # Procesar archivo
        start_time = time.time()
        chunks = ingestion_service.process_file(file_path)
        processing_time = time.time() - start_time
        
        if chunks:
            print(f"✅ Procesado exitosamente:")
            print(f"   📊 Chunks generados: {len(chunks)}")
            print(f"   ⏱️ Tiempo: {processing_time:.2f}s")
            
            # Mostrar primer chunk como ejemplo
            if chunks:
                first_chunk = chunks[0]
                print(f"   📝 Ejemplo chunk:")
                print(f"      ID: {first_chunk.id}")
                print(f"      Contenido: {first_chunk.content[:100]}...")
                print(f"      Metadatos: {list(first_chunk.metadata.keys())}")
            
            return True
        else:
            print("❌ No se pudieron generar chunks")
            return False
            
    except ImportError as e:
        print(f"❌ Error importando servicio: {e}")
        print("💡 Asegúrate de estar en el directorio raíz del proyecto")
        return False
    except Exception as e:
        print(f"❌ Error procesando archivo: {e}")
        return False

def process_directory(dir_path: str):
    """Procesar todos los archivos de un directorio"""
    try:
        from app.services.ingestion import ingestion_service
        
        print(f"📂 Procesando directorio: {dir_path}")
        
        if not os.path.exists(dir_path):
            print(f"❌ Directorio no encontrado: {dir_path}")
            return False
        
        # Buscar archivos soportados
        supported_extensions = ingestion_service.processor.get_supported_extensions()
        files_found = []
        
        for ext in supported_extensions:
            pattern = f"*{ext}"
            files_found.extend(Path(dir_path).glob(pattern))
        
        if not files_found:
            print(f"❌ No se encontraron archivos soportados en {dir_path}")
            print(f"💡 Extensiones soportadas: {supported_extensions}")
            return False
        
        print(f"📋 Encontrados {len(files_found)} archivos")
        
        # Procesar cada archivo
        successful = 0
        failed = 0
        total_chunks = 0
        start_time = time.time()
        
        for file_path in files_found:
            print(f"\n🔄 Procesando: {file_path.name}")
            
            chunks = ingestion_service.process_file(str(file_path))
            
            if chunks:
                successful += 1
                total_chunks += len(chunks)
                print(f"   ✅ {len(chunks)} chunks")
            else:
                failed += 1
                print(f"   ❌ Error")
        
        # Resumen
        processing_time = time.time() - start_time
        print(f"\n📊 RESUMEN:")
        print(f"   📁 Directorio: {dir_path}")
        print(f"   📄 Archivos procesados: {successful}/{len(files_found)}")
        print(f"   ❌ Archivos fallidos: {failed}")
        print(f"   📦 Total chunks: {total_chunks}")
        print(f"   ⏱️ Tiempo total: {processing_time:.2f}s")
        print(f"   ⚡ Throughput: {successful/processing_time:.1f} archivos/s")
        
        return successful > 0
        
    except ImportError as e:
        print(f"❌ Error importando servicio: {e}")
        return False
    except Exception as e:
        print(f"❌ Error procesando directorio: {e}")
        return False

def show_service_status():
    """Mostrar estado del servicio"""
    try:
        from app.services.ingestion import ingestion_service
        
        stats = ingestion_service.get_service_stats()
        
        print("🔍 ESTADO DEL SERVICIO:")
        print(f"   🟢 Disponible: {'Sí' if stats['service_available'] else 'No'}")
        print(f"   🔧 Procesadores: {stats['processors_available']}")
        print(f"   📄 Formatos soportados: {stats['supported_extensions']}")
        print(f"   💼 Trabajos activos: {stats['active_jobs']}")
        print(f"   📈 Total trabajos: {stats['total_jobs']}")
        
        return stats['service_available']
        
    except ImportError as e:
        print(f"❌ Error importando servicio: {e}")
        return False
    except Exception as e:
        print(f"❌ Error obteniendo estado: {e}")
        return False

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Sistema de Ingesta de Documentos - TFM Vicente Caruncho",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/ejecutar_ingesta.py --setup
  python scripts/ejecutar_ingesta.py --create-test-docs
  python scripts/ejecutar_ingesta.py --file data/documentos/ordenanza.txt
  python scripts/ejecutar_ingesta.py --dir data/documentos
  python scripts/ejecutar_ingesta.py --status
        """
    )
    
    parser.add_argument('--setup', action='store_true',
                       help='Configurar estructura de directorios')
    parser.add_argument('--create-test-docs', action='store_true',
                       help='Crear documentos de prueba')
    parser.add_argument('--file', type=str,
                       help='Procesar archivo específico')
    parser.add_argument('--dir', type=str,
                       help='Procesar directorio completo')
    parser.add_argument('--status', action='store_true',
                       help='Mostrar estado del servicio')
    
    args = parser.parse_args()
    
    print("🤖 SISTEMA DE INGESTA DE DOCUMENTOS")
    print("🎓 TFM Vicente Caruncho - Sistemas Inteligentes")
    print("=" * 50)
    
    # Setup inicial
    if args.setup:
        setup_directories()
        print("\n✅ Setup completado")
        return 0
    
    # Crear documentos de prueba
    if args.create_test_docs:
        count = create_test_documents()
        print(f"\n✅ Creados {count} documentos de prueba")
        return 0
    
    # Mostrar estado
    if args.status:
        available = show_service_status()
        return 0 if available else 1
    
    # Procesar archivo específico
    if args.file:
        success = process_single_file(args.file)
        return 0 if success else 1
    
    # Procesar directorio
    if args.dir:
        success = process_directory(args.dir)
        return 0 if success else 1
    
    # Si no se especifica acción, mostrar ayuda
    parser.print_help()
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️ Proceso cancelado")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        sys.exit(1)