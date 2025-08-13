"""
Servicio de ingesta de datos - Compatible con sistema existente
"""

from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
import tempfile

# Usar DocumentChunk del sistema existente
try:
    from app.models.document import DocumentChunk
except ImportError:
    # Definición mínima si no existe
    class DocumentChunk:
        def __init__(self, content: str, metadata: Dict[str, Any] = None, **kwargs):
            self.content = content
            self.metadata = metadata or {}
            for key, value in kwargs.items():
                setattr(self, key, value)

class DataIngestionService:
    """Servicio de ingesta compatible con el sistema actual"""
    
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200
    
    def ingest_text_file(self, file_path: str) -> List[DocumentChunk]:
        """Ingerir archivo de texto"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = self._chunk_text(content, {"source": file_path, "type": "text"})
            return chunks
            
        except Exception as e:
            print(f"Error procesando {file_path}: {e}")
            return []
    
    def ingest_pdf(self, file_path: str) -> List[DocumentChunk]:
        """Simular ingesta PDF"""
        mock_content = f"Contenido simulado de PDF: {Path(file_path).name}"
        return [DocumentChunk(
            content=mock_content,
            metadata={"source": file_path, "type": "pdf_simulation"}
        )]
    
    def ingest_docx(self, file_path: str) -> List[DocumentChunk]:
        """Simular ingesta DOCX"""
        mock_content = f"Contenido simulado de DOCX: {Path(file_path).name}"
        return [DocumentChunk(
            content=mock_content,
            metadata={"source": file_path, "type": "docx_simulation"}
        )]
    
    def ingest_url(self, url: str) -> List[DocumentChunk]:
        """Simular ingesta web"""
        mock_content = f"Contenido web simulado de: {url}"
        return [DocumentChunk(
            content=mock_content,
            metadata={"source": url, "type": "web_simulation"}
        )]
    
    def ingest_api(self, endpoint: str, auth_token: str = None) -> List[DocumentChunk]:
        """Simular ingesta API"""
        mock_content = f"Datos API simulados de: {endpoint}"
        return [DocumentChunk(
            content=mock_content,
            metadata={"source": endpoint, "type": "api_simulation"}
        )]
    
    def _chunk_text(self, text: str, base_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Fragmentar texto en chunks"""
        chunks = []
        chunk_index = 0
        start_pos = 0
        
        while start_pos < len(text):
            end_pos = min(start_pos + self.chunk_size, len(text))
            
            # Buscar final natural
            if end_pos < len(text):
                for i in range(end_pos, max(start_pos, end_pos - 100), -1):
                    if text[i] in '.!?\n':
                        end_pos = i + 1
                        break
            
            chunk_content = text[start_pos:end_pos].strip()
            
            if chunk_content:
                metadata = base_metadata.copy()
                metadata.update({
                    "chunk_index": chunk_index,
                    "start_char": start_pos,
                    "end_char": end_pos
                })
                
                chunk = DocumentChunk(
                    content=chunk_content,
                    metadata=metadata,
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
            
            start_pos = max(start_pos + 1, end_pos - self.chunk_overlap)
        
        return chunks
