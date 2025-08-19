"""
Modelos de documentos y chunks para el sistema RAG
TFM Vicente Caruncho - Sistemas Inteligentes
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime


@dataclass
class DocumentMetadata:
    """Metadatos de un documento"""
    source_path: str
    source_type: str = "document"
    file_type: str = ""
    size_bytes: int = 0
    created_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    checksum: str = ""
    
    def __post_init__(self):
        """Inicialización automática de campos"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.processed_at is None:
            self.processed_at = datetime.now()


@dataclass
class DocumentChunk:
    """Chunk de documento procesado - Compatible con sistema existente"""
    id: str
    content: str
    metadata: Dict[str, Any]
    source_file: str
    chunk_index: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    
    # Campos adicionales para compatibilidad con vector stores
    embedding: Optional[List[float]] = None
    relevance_score: Optional[float] = None
    
    def __post_init__(self):
        """Validaciones y setup automático"""
        # Asegurar que el ID no esté vacío
        if not self.id:
            import uuid
            self.id = str(uuid.uuid4())
        
        # Asegurar que metadata sea un dict
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        
        # Asegurar que chunk_index sea válido
        if self.chunk_index < 0:
            self.chunk_index = 0
    
    @property
    def chunk_size(self) -> int:
        """Tamaño del chunk en caracteres"""
        return len(self.content) if self.content else 0
    
    @property
    def word_count(self) -> int:
        """Número de palabras en el chunk"""
        return len(self.content.split()) if self.content else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización"""
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata,
            'source_file': self.source_file,
            'chunk_index': self.chunk_index,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'embedding': self.embedding,
            'relevance_score': self.relevance_score,
            'chunk_size': self.chunk_size,
            'word_count': self.word_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """Crear desde diccionario"""
        return cls(
            id=data.get('id', ''),
            content=data.get('content', ''),
            metadata=data.get('metadata', {}),
            source_file=data.get('source_file', ''),
            chunk_index=data.get('chunk_index', 0),
            start_char=data.get('start_char'),
            end_char=data.get('end_char'),
            embedding=data.get('embedding'),
            relevance_score=data.get('relevance_score')
        )
    
    def copy(self) -> 'DocumentChunk':
        """Crear una copia del chunk"""
        return DocumentChunk(
            id=self.id,
            content=self.content,
            metadata=self.metadata.copy(),
            source_file=self.source_file,
            chunk_index=self.chunk_index,
            start_char=self.start_char,
            end_char=self.end_char,
            embedding=self.embedding.copy() if self.embedding else None,
            relevance_score=self.relevance_score
        )


# Funciones de utilidad para crear chunks
def create_chunk(
    content: str,
    source_file: str,
    chunk_index: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_id: Optional[str] = None
) -> DocumentChunk:
    """Función de conveniencia para crear chunks"""
    import uuid
    
    if chunk_id is None:
        chunk_id = str(uuid.uuid4())
    
    if metadata is None:
        metadata = {}
    
    return DocumentChunk(
        id=chunk_id,
        content=content,
        metadata=metadata,
        source_file=source_file,
        chunk_index=chunk_index
    )


def create_chunks_from_text(
    text: str,
    source_file: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    base_metadata: Optional[Dict[str, Any]] = None
) -> List[DocumentChunk]:
    """Crear múltiples chunks desde un texto"""
    if base_metadata is None:
        base_metadata = {}
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        # Calcular fin del chunk
        end = min(start + chunk_size, len(text))
        
        # Buscar un punto de corte natural (espacio, punto, etc.)
        if end < len(text):
            # Buscar hacia atrás desde el final para encontrar un buen punto de corte
            for i in range(min(100, chunk_size // 4)):
                if text[end - i] in '.!?\n ':
                    end = end - i + 1
                    break
        
        # Extraer contenido del chunk
        chunk_content = text[start:end].strip()
        
        if chunk_content:  # Solo crear chunk si tiene contenido
            # Crear metadatos específicos del chunk
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'start_char': start,
                'end_char': end,
                'chunk_size': len(chunk_content),
                'word_count': len(chunk_content.split())
            })
            
            chunk = DocumentChunk(
                id=str(__import__('uuid').uuid4()),
                content=chunk_content,
                metadata=chunk_metadata,
                source_file=source_file,
                chunk_index=chunk_index,
                start_char=start,
                end_char=end
            )
            
            chunks.append(chunk)
            chunk_index += 1
        
        # Mover al siguiente chunk con overlap
        start = max(start + 1, end - chunk_overlap)
        
        # Prevenir loops infinitos
        if start >= len(text):
            break
    
    return chunks