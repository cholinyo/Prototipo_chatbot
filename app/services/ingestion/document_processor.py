"""
Procesador de Documentos Básico
TFM Vicente Caruncho - Sistemas Inteligentes
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any

from app.core.logger import get_logger
from app.models import DocumentChunk

class DocumentProcessor:
    """Procesador básico de documentos"""
    
    def __init__(self):
        self.logger = get_logger("prototipo_chatbot.document_processor")
        self.processors = {
            '.txt': self._process_text,
            '.md': self._process_text,
            '.pdf': self._process_pdf_mock,
            '.docx': self._process_docx_mock,
            '.html': self._process_html_mock
        }
    
    def process(self, file_path: str, source_type: str = 'document') -> List[DocumentChunk]:
        """Procesar archivo según su extensión"""
        if not os.path.exists(file_path):
            self.logger.warning(f"Archivo no encontrado: {file_path}")
            return []
        
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in self.processors:
            self.logger.warning(f"Extensión no soportada: {file_ext}")
            return []
        
        try:
            return self.processors[file_ext](file_path, source_type)
        except Exception as e:
            self.logger.error(f"Error procesando {file_path}: {e}")
            return []
    
    def _process_text(self, file_path: str, source_type: str) -> List[DocumentChunk]:
        """Procesar archivos de texto"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Dividir en chunks básicos
            chunks = []
            chunk_size = 500  # caracteres por chunk
            
            for i in range(0, len(content), chunk_size):
                chunk_content = content[i:i + chunk_size]
                
                chunk = DocumentChunk(
                    id=f"chunk_{Path(file_path).stem}_{i}",
                    content=chunk_content,
                    metadata={
                        'source_path': file_path,
                        'source_type': source_type,
                        'title': Path(file_path).stem,
                        'chunk_index': i // chunk_size
                    },
                    embedding=None
                )
                chunks.append(chunk)
            
            self.logger.info(f"Procesado archivo texto: {file_path} -> {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error procesando texto {file_path}: {e}")
            return []
    
    def _process_pdf_mock(self, file_path: str, source_type: str) -> List[DocumentChunk]:
        """Mock para procesamiento PDF"""
        chunk = DocumentChunk(
            id=f"pdf_chunk_{int(time.time())}",
            content=f"[CONTENIDO PDF MOCK] Archivo: {Path(file_path).name}",
            metadata={
                'source_path': file_path,
                'source_type': source_type,
                'title': Path(file_path).stem,
                'note': 'Procesamiento PDF requiere PyPDF2 o pymupdf'
            },
            embedding=None
        )
        return [chunk]
    
    def _process_docx_mock(self, file_path: str, source_type: str) -> List[DocumentChunk]:
        """Mock para procesamiento DOCX"""
        chunk = DocumentChunk(
            id=f"docx_chunk_{int(time.time())}",
            content=f"[CONTENIDO DOCX MOCK] Archivo: {Path(file_path).name}",
            metadata={
                'source_path': file_path,
                'source_type': source_type,
                'title': Path(file_path).stem,
                'note': 'Procesamiento DOCX requiere python-docx'
            },
            embedding=None
        )
        return [chunk]
    
    def _process_html_mock(self, file_path: str, source_type: str) -> List[DocumentChunk]:
        """Mock para procesamiento HTML"""
        chunk = DocumentChunk(
            id=f"html_chunk_{int(time.time())}",
            content=f"[CONTENIDO HTML MOCK] Archivo: {Path(file_path).name}",
            metadata={
                'source_path': file_path,
                'source_type': source_type,
                'title': Path(file_path).stem,
                'note': 'Procesamiento HTML requiere BeautifulSoup4'
            },
            embedding=None
        )
        return [chunk]
    
    def get_supported_extensions(self) -> List[str]:
        """Obtener extensiones soportadas"""
        return list(self.processors.keys())

# Instancia global
document_processor = DocumentProcessor()

__all__ = ['DocumentProcessor', 'document_processor']
