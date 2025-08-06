#!/usr/bin/env python3
import sys
import os
import time
import shutil
from pathlib import Path
from datetime import datetime

# Añadir directorio raíz al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_chromadb_simple():
    """Test básico de ChromaDB"""
    print("🧪 Test ChromaDB Simple")
    
    try:
        import chromadb
        from app.models import DocumentChunk, DocumentMetadata
        
        # Crear cliente ChromaDB [[1](https://docs.trychroma.com/getting-started)]
        client = chromadb.Client()
        collection = client.create_collection("test_collection")
        
        # Crear documento simple
        metadata = DocumentMetadata(
            source_path="test.pdf",
            source_type="document", 
            file_type=".pdf",
            size_bytes=100,
            created_at=datetime.now(),
            processed_at=datetime.now(),
            checksum="test123"
        )
        
        chunk = DocumentChunk(
            id="test1",
            content="Documento de prueba",
            metadata=metadata,
            chunk_index=0,
            chunk_size=18,
            start_char=0,
            end_char=18
        )
        
        print("✅ Test completado exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_chromadb_simple()