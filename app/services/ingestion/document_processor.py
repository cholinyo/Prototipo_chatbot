"""
Procesador Real de Documentos
Se integra automáticamente con tu IngestionService existente
"""

import os
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from app.core.logger import get_logger
from app.models import DocumentChunk

class RealDocumentProcessor:
    """
    Procesador real que reemplaza automáticamente al MockDocumentProcessor
    Tu código existente lo cargará automáticamente sin cambios
    """
    
    def __init__(self):
        self.supported_extensions = ['.pdf', '.docx', '.txt', '.md', '.html', '.rtf', '.odt']
        self.logger = get_logger("prototipo_chatbot.document_processor")
        self.logger.info("Procesador REAL de documentos inicializado")
    
    def process(self, file_path: str, source_type: str = 'document') -> List[DocumentChunk]:
        """
        Método principal que tu IngestionService ya llama
        Mantiene la misma interfaz, pero hace procesamiento real
        """
        if not os.path.exists(file_path):
            self.logger.error(f"Archivo no encontrado: {file_path}")
            return []
        
        file_ext = Path(file_path).suffix.lower()
        
        try:
            # Decidir procesador según extensión
            if file_ext == '.pdf':
                chunks = self._process_pdf(file_path, source_type)
            elif file_ext == '.docx':
                chunks = self._process_docx(file_path, source_type)
            elif file_ext in ['.txt', '.md', '.rtf', '.odt']:
                chunks = self._process_text(file_path, source_type)
            elif file_ext in ['.html', '.htm']:
                chunks = self._process_html(file_path, source_type)
            else:
                self.logger.warning(f"Extensión no soportada: {file_ext}")
                return []
            
            self.logger.info(f"Procesado {file_path}: {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error procesando {file_path}: {e}")
            return []
    
    def _process_pdf(self, file_path: str, source_type: str) -> List[DocumentChunk]:
        """Procesar archivo PDF"""
        try:
            import PyPDF2
            chunks = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        chunk = self._create_chunk(
                            content=text,
                            file_path=file_path,
                            source_type=source_type,
                            page_number=page_num + 1,
                            file_type='pdf'
                        )
                        chunks.append(chunk)
            
            return chunks
            
        except ImportError:
            self.logger.warning("PyPDF2 no instalado, usando procesamiento básico")
            return self._process_as_text_fallback(file_path, source_type, 'pdf')
        except Exception as e:
            self.logger.error(f"Error procesando PDF: {e}")
            return []
    
    def _process_docx(self, file_path: str, source_type: str) -> List[DocumentChunk]:
        """Procesar archivo DOCX"""
        try:
            from docx import Document
            chunks = []
            
            doc = Document(file_path)
            current_text = ""
            paragraph_count = 0
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    current_text += paragraph.text + "\n\n"
                    paragraph_count += 1
                    
                    # Crear chunk cada 3-5 párrafos o si es muy largo
                    if paragraph_count >= 3 or len(current_text) > 1000:
                        chunk = self._create_chunk(
                            content=current_text.strip(),
                            file_path=file_path,
                            source_type=source_type,
                            paragraph_count=paragraph_count,
                            file_type='docx'
                        )
                        chunks.append(chunk)
                        current_text = ""
                        paragraph_count = 0
            
            # Último chunk si queda contenido
            if current_text.strip():
                chunk = self._create_chunk(
                    content=current_text.strip(),
                    file_path=file_path,
                    source_type=source_type,
                    paragraph_count=paragraph_count,
                    file_type='docx'
                )
                chunks.append(chunk)
            
            return chunks
            
        except ImportError:
            self.logger.warning("python-docx no instalado, usando procesamiento básico")
            return self._process_as_text_fallback(file_path, source_type, 'docx')
        except Exception as e:
            self.logger.error(f"Error procesando DOCX: {e}")
            return []
    
    def _process_text(self, file_path: str, source_type: str) -> List[DocumentChunk]:
        """Procesar archivo de texto plano"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Dividir en chunks por párrafos
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            chunks = []
            
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph) > 50:  # Solo párrafos significativos
                    chunk = self._create_chunk(
                        content=paragraph,
                        file_path=file_path,
                        source_type=source_type,
                        paragraph_number=i + 1,
                        file_type=Path(file_path).suffix
                    )
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error procesando texto: {e}")
            return []
    
    def _process_html(self, file_path: str, source_type: str) -> List[DocumentChunk]:
        """Procesar archivo HTML"""
        try:
            from bs4 import BeautifulSoup
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extraer texto limpio
            text = soup.get_text()
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
            
            chunks = []
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph) > 50:
                    chunk = self._create_chunk(
                        content=paragraph,
                        file_path=file_path,
                        source_type=source_type,
                        paragraph_number=i + 1,
                        file_type='html'
                    )
                    chunks.append(chunk)
            
            return chunks
            
        except ImportError:
            self.logger.warning("BeautifulSoup no instalado, usando procesamiento básico")
            return self._process_as_text_fallback(file_path, source_type, 'html')
        except Exception as e:
            self.logger.error(f"Error procesando HTML: {e}")
            return []
    
    def _process_as_text_fallback(self, file_path: str, source_type: str, file_type: str) -> List[DocumentChunk]:
        """Fallback: procesar como texto plano"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            
            # Limitar contenido para fallback
            if len(content) > 2000:
                content = content[:2000] + "..."
            
            chunk = self._create_chunk(
                content=content,
                file_path=file_path,
                source_type=source_type,
                file_type=file_type,
                fallback=True
            )
            
            return [chunk]
            
        except Exception as e:
            self.logger.error(f"Error en fallback: {e}")
            return []
    
    def _create_chunk(self, content: str, file_path: str, source_type: str, **kwargs) -> DocumentChunk:
        """
        Crear DocumentChunk compatible con tu modelo existente
        Mantiene el formato que tu código ya espera
        """
        # Generar ID único basado en contenido y archivo
        chunk_id = f"chunk_{int(time.time())}_{hash(content) % 10000}"
        
        # Metadatos base + específicos
        metadata = {
            'source_path': file_path,
            'source_type': source_type,
            'title': Path(file_path).stem,
            'file_size': os.path.getsize(file_path),
            'processed_at': datetime.now().isoformat(),
            'checksum': self._calculate_checksum(file_path),
            **kwargs  # Metadatos adicionales específicos del tipo
        }
        
        return DocumentChunk(
            id=chunk_id,
            content=content,
            metadata=metadata,
            embedding=None  # Se generará después en el pipeline
        )
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calcular checksum MD5 del archivo"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def get_supported_extensions(self) -> List[str]:
        """Método que tu código existente ya llama"""
        return self.supported_extensions

# Instancia global que tu código existente cargará automáticamente
document_processor = RealDocumentProcessor()