"""
Procesador de Documentos para Pipeline de Ingesta
Prototipo_chatbot - TFM Vicente Caruncho
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import hashlib
import mimetypes

from app.core.logger import get_logger
from app.models import DocumentChunk, DocumentMetadata, create_document_chunk

class BaseDocumentProcessor(ABC):
    """Clase base abstracta para procesadores de documentos"""
    
    @abstractmethod
    def can_process(self, file_path: str) -> bool:
        """Verificar si puede procesar el archivo"""
        pass
    
    @abstractmethod
    def process(self, file_path: str, **kwargs) -> List[DocumentChunk]:
        """Procesar archivo y retornar chunks"""
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Obtener extensiones soportadas"""
        pass

class TextDocumentProcessor(BaseDocumentProcessor):
    """Procesador para documentos de texto plano"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = get_logger("text_processor")
        self.supported_extensions = ['.txt', '.md', '.rst', '.log']
    
    def can_process(self, file_path: str) -> bool:
        """Verificar si puede procesar el archivo"""
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_extensions
    
    def get_supported_extensions(self) -> List[str]:
        """Obtener extensiones soportadas"""
        return self.supported_extensions
    
    def process(self, file_path: str, **kwargs) -> List[DocumentChunk]:
        """Procesar archivo de texto"""
        chunks = []
        
        try:
            # Leer archivo
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Crear metadata
            file_stats = os.stat(file_path)
            metadata = DocumentMetadata(
                source_path=str(file_path),
                source_type='text_file',
                file_type=Path(file_path).suffix[1:],
                file_size=file_stats.st_size,
                created_at=time.time(),
                processing_metadata={
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap,
                    'total_characters': len(content)
                }
            )
            
            # Dividir en chunks
            chunks = self._split_text_into_chunks(content, metadata)
            
            self.logger.info(
                f"Procesado archivo de texto: {file_path}",
                chunks_created=len(chunks)
            )
            
        except Exception as e:
            self.logger.error(f"Error procesando archivo {file_path}: {e}")
            raise
        
        return chunks
    
    def _split_text_into_chunks(
        self, 
        text: str, 
        metadata: DocumentMetadata
    ) -> List[DocumentChunk]:
        """Dividir texto en chunks con overlap"""
        chunks = []
        
        # Dividir por párrafos primero
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # Si el párrafo es muy largo, dividirlo
            if len(paragraph) > self.chunk_size:
                # Dividir párrafo largo
                sentences = self._split_into_sentences(paragraph)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= self.chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk.strip():
                            chunks.append(create_document_chunk(
                                content=current_chunk.strip(),
                                metadata=metadata,
                                chunk_index=chunk_index
                            ))
                            chunk_index += 1
                        current_chunk = sentence + " "
            else:
                # Agregar párrafo al chunk actual
                if len(current_chunk) + len(paragraph) <= self.chunk_size:
                    current_chunk += paragraph + "\n\n"
                else:
                    if current_chunk.strip():
                        chunks.append(create_document_chunk(
                            content=current_chunk.strip(),
                            metadata=metadata,
                            chunk_index=chunk_index
                        ))
                        chunk_index += 1
                    current_chunk = paragraph + "\n\n"
        
        # Agregar último chunk si existe
        if current_chunk.strip():
            chunks.append(create_document_chunk(
                content=current_chunk.strip(),
                metadata=metadata,
                chunk_index=chunk_index
            ))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Dividir texto en oraciones (simplificado)"""
        # Implementación simple, mejorar con spaCy o NLTK si es necesario
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in '.!?' and len(current) > 20:
                sentences.append(current.strip())
                current = ""
        
        if current.strip():
            sentences.append(current.strip())
        
        return sentences

class DocumentProcessor:
    """Procesador principal que coordina los procesadores específicos"""
    
    def __init__(self):
        self.logger = get_logger("prototipo_chatbot.document_processor")
        self.processors: List[BaseDocumentProcessor] = []
        self._register_default_processors()
    
    def _register_default_processors(self):
        """Registrar procesadores por defecto"""
        self.register_processor(TextDocumentProcessor())
        self.logger.info("Procesadores por defecto registrados")
    
    def register_processor(self, processor: BaseDocumentProcessor):
        """Registrar un nuevo procesador"""
        self.processors.append(processor)
        extensions = processor.get_supported_extensions()
        self.logger.info(
            f"Procesador registrado: {processor.__class__.__name__}",
            extensions=extensions
        )
    
    def can_process(self, file_path: str) -> bool:
        """Verificar si algún procesador puede manejar el archivo"""
        for processor in self.processors:
            if processor.can_process(file_path):
                return True
        return False
    
    def process(
        self, 
        file_path: str,
        source_type: str = 'document',
        **kwargs
    ) -> List[DocumentChunk]:
        """Procesar documento con el procesador apropiado"""
        
        # Verificar que el archivo existe
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        # Encontrar procesador apropiado
        for processor in self.processors:
            if processor.can_process(file_path):
                self.logger.info(
                    f"Procesando con {processor.__class__.__name__}",
                    file_path=file_path
                )
                
                try:
                    chunks = processor.process(file_path, **kwargs)
                    
                    # Actualizar source_type en metadata si es necesario
                    for chunk in chunks:
                        if chunk.metadata and not chunk.metadata.source_type:
                            chunk.metadata.source_type = source_type
                    
                    return chunks
                    
                except Exception as e:
                    self.logger.error(
                        f"Error procesando archivo: {e}",
                        file_path=file_path,
                        processor=processor.__class__.__name__
                    )
                    raise
        
        # No hay procesador disponible
        ext = Path(file_path).suffix
        self.logger.warning(
            f"No hay procesador disponible para extensión {ext}",
            file_path=file_path
        )
        
        # Intentar procesamiento genérico de texto
        return self._generic_text_processing(file_path, source_type)
    
    def _generic_text_processing(
        self, 
        file_path: str,
        source_type: str
    ) -> List[DocumentChunk]:
        """Procesamiento genérico para archivos de texto"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            metadata = DocumentMetadata(
                source_path=str(file_path),
                source_type=source_type,
                file_type='generic_text',
                file_size=os.path.getsize(file_path),
                created_at=time.time()
            )
            
            # Crear un solo chunk con todo el contenido
            chunk = create_document_chunk(
                content=content[:5000],  # Limitar tamaño
                metadata=metadata,
                chunk_index=0
            )
            
            return [chunk]
            
        except Exception as e:
            self.logger.error(f"Error en procesamiento genérico: {e}")
            return []
    
    def get_supported_extensions(self) -> List[str]:
        """Obtener todas las extensiones soportadas"""
        extensions = set()
        for processor in self.processors:
            extensions.update(processor.get_supported_extensions())
        return list(extensions)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del procesador"""
        return {
            'processors_count': len(self.processors),
            'supported_extensions': self.get_supported_extensions(),
            'processors': [p.__class__.__name__ for p in self.processors]
        }

# Instancia global
document_processor = DocumentProcessor()

# Exportar lo necesario
__all__ = [
    'DocumentProcessor',
    'BaseDocumentProcessor',
    'TextDocumentProcessor',
    'document_processor'
]