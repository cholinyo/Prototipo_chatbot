"""
Modelos de documentos y chunks para el sistema RAG
TFM Vicente Caruncho - Sistemas Inteligentes

PROPÓSITO: Estructura interna de contenido para RAG
- DocumentChunk: Pedazos de texto con embeddings para búsqueda
- Funciones para dividir documentos en chunks (universales para archivos Y webs)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, TYPE_CHECKING, Union
from datetime import datetime

# Import condicional para evitar circular dependency
if TYPE_CHECKING:
    from app.models.data_sources import ScrapedPage


@dataclass
class DocumentMetadata:
    """
    Metadatos de un documento procesado
    
    Información adicional sobre el documento original antes de ser
    dividido en chunks para el sistema RAG.
    """
    source_path: str
    source_type: str = "document"
    file_type: str = ""
    size_bytes: int = 0
    created_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    checksum: str = ""
    
    def __post_init__(self):
        """Inicialización automática de campos de fecha"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.processed_at is None:
            self.processed_at = datetime.now()


@dataclass
class DocumentChunk:
    """
    Chunk de documento procesado - Compatible con sistema existente
    
    Representa una porción de texto de cualquier fuente (archivo, web, API)
    junto con metadatos y embeddings para búsqueda semántica en el vector store.
    """
    id: str
    content: str
    metadata: Union[Dict[str, Any], DocumentMetadata, Any]
    source_file: str  # Para webs: será la URL
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
        
        # ✅ CORRECCIÓN: Preservar DocumentMetadata, solo convertir None o tipos inválidos
        if self.metadata is None:
            self.metadata = {}
        elif hasattr(self.metadata, '__dict__') and not isinstance(self.metadata, dict):
            # Si es un objeto tipo DocumentMetadata, mantenerlo como está
            pass
        elif not isinstance(self.metadata, (dict, object)):
            # Solo convertir si no es ni dict ni objeto con atributos
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
    
    @property
    def is_web_content(self) -> bool:
        """Verificar si el chunk proviene de contenido web"""
        return self.metadata.get('source_type') == 'web'
    
    @property
    def is_document_content(self) -> bool:
        """Verificar si el chunk proviene de documento local"""
        return self.metadata.get('source_type') == 'document'
    
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
            'word_count': self.word_count,
            'is_web_content': self.is_web_content,
            'is_document_content': self.is_document_content
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """Crear DocumentChunk desde diccionario"""
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
        """Crear una copia independiente del chunk"""
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


# =============================================================================
# FUNCIONES UNIVERSALES - Para cualquier tipo de contenido
# =============================================================================

def create_chunk(
    content: str,
    source_file: str,
    chunk_index: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_id: Optional[str] = None
) -> DocumentChunk:
    """
    Función de conveniencia para crear chunks individuales
    
    Args:
        content: Contenido de texto del chunk
        source_file: Ruta del archivo o URL de origen
        chunk_index: Índice del chunk en la secuencia
        metadata: Metadatos adicionales
        chunk_id: ID específico (se genera automáticamente si es None)
        
    Returns:
        Nueva instancia de DocumentChunk
    """
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
    """
    Crear múltiples chunks desde un texto (UNIVERSAL para archivos y webs)
    
    Args:
        text: Texto completo a dividir en chunks
        source_file: Ruta del archivo o URL de origen
        chunk_size: Tamaño máximo de cada chunk en caracteres
        chunk_overlap: Solapamiento entre chunks consecutivos
        base_metadata: Metadatos base para todos los chunks
        
    Returns:
        Lista de DocumentChunk generados desde el texto
    """
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
                'word_count': len(chunk_content.split()),
                'total_source_length': len(text)
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


# =============================================================================
# FUNCIONES ESPECÍFICAS PARA WEB - Crear chunks desde páginas scrapeadas
# =============================================================================

def create_web_chunks(
    scraped_page: 'ScrapedPage',
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[DocumentChunk]:
    """
    Crear chunks desde página web scrapeada con metadatos específicos de web
    
    Args:
        scraped_page: Página web scrapeada con contenido
        chunk_size: Tamaño máximo de cada chunk en caracteres
        chunk_overlap: Solapamiento entre chunks consecutivos
        
    Returns:
        Lista de DocumentChunk con metadatos específicos de web
    """
    from urllib.parse import urlparse
    
    # Verificar que hay contenido para procesar
    if not scraped_page.content or not scraped_page.content.strip():
        return []
    
    # Metadatos base específicos para contenido web
    web_metadata = {
        'source_type': 'web',
        'url': scraped_page.url,
        'title': scraped_page.title,
        'domain': urlparse(scraped_page.url).netloc,
        'path': urlparse(scraped_page.url).path,
        'scraped_at': scraped_page.scraped_at.isoformat(),
        'content_hash': scraped_page.content_hash,
        'page_id': scraped_page.id,
        'source_id': scraped_page.source_id,
        'status_code': scraped_page.status_code,
        'original_content_length': scraped_page.content_length,
        'links_found_count': len(scraped_page.links_found),
        'extraction_method': 'web_scraping'
    }
    
    # Añadir metadatos adicionales de la página si existen
    if scraped_page.metadata:
        web_metadata.update({
            f'page_{key}': value 
            for key, value in scraped_page.metadata.items()
            if key not in web_metadata  # Evitar duplicados
        })
    
    # Usar función universal de chunking con metadatos específicos de web
    chunks = create_chunks_from_text(
        text=scraped_page.content,
        source_file=scraped_page.url,  # Para webs, "file" es la URL
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        base_metadata=web_metadata
    )
    
    # Añadir información específica de cada chunk para webs
    for i, chunk in enumerate(chunks):
        # Añadir información específica del chunk web
        chunk.metadata.update({
            'chunk_url': f"{scraped_page.url}#chunk-{i}",  # URL única para cada chunk
            'chunk_type': 'web_content',
            'page_title': scraped_page.title,
            'chunk_position': f"{i+1}/{len(chunks)}"  # Posición en la página
        })
        
        # Si es el primer chunk, incluir información del título
        if i == 0 and scraped_page.title:
            chunk.metadata['contains_title'] = True
            # Opcionalmente incluir título al inicio del contenido del primer chunk
            if not chunk.content.startswith(scraped_page.title):
                chunk.content = f"{scraped_page.title}\n\n{chunk.content}"
                chunk.metadata['title_prepended'] = True
    
    return chunks


def create_web_chunks_from_pages(
    scraped_pages: List['ScrapedPage'],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[DocumentChunk]:
    """
    Crear chunks desde múltiples páginas web scrapeadas
    
    Procesa múltiples páginas web y consolida todos los chunks en una sola lista,
    manteniendo la información de origen de cada página.
    
    Args:
        scraped_pages: Lista de páginas web scrapeadas
        chunk_size: Tamaño máximo de cada chunk
        chunk_overlap: Solapamiento entre chunks
        
    Returns:
        Lista consolidada de DocumentChunk de todas las páginas
    """
    all_chunks = []
    
    for page_idx, page in enumerate(scraped_pages):
        try:
            page_chunks = create_web_chunks(
                scraped_page=page,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Añadir información de batch a cada chunk
            for chunk in page_chunks:
                chunk.metadata.update({
                    'batch_page_index': page_idx,
                    'batch_total_pages': len(scraped_pages),
                    'batch_processing': True
                })
            
            all_chunks.extend(page_chunks)
            
        except Exception as e:
            # Log error pero continúa con otras páginas
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error creando chunks para página {page.url}: {e}")
            continue
    
    return all_chunks


def create_document_chunks(
    file_path: str,
    content: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> List[DocumentChunk]:
    """
    Crear chunks desde documento local con metadatos específicos de archivo
    
    Args:
        file_path: Ruta del archivo de origen
        content: Contenido extraído del archivo
        chunk_size: Tamaño máximo de cada chunk
        chunk_overlap: Solapamiento entre chunks
        additional_metadata: Metadatos adicionales del documento
        
    Returns:
        Lista de DocumentChunk con metadatos específicos de documento
    """
    from pathlib import Path
    
    if not content or not content.strip():
        return []
    
    path = Path(file_path)
    
    # Metadatos base específicos para documentos
    document_metadata = {
        'source_type': 'document',
        'file_path': str(path.absolute()),
        'file_name': path.name,
        'file_extension': path.suffix.lower(),
        'file_size': len(content.encode('utf-8')),
        'extraction_method': 'document_processing'
    }
    
    # Añadir metadatos adicionales si se proporcionan
    if additional_metadata:
        document_metadata.update(additional_metadata)
    
    # Usar función universal de chunking
    chunks = create_chunks_from_text(
        text=content,
        source_file=str(path),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        base_metadata=document_metadata
    )
    
    # Añadir información específica de cada chunk para documentos
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            'chunk_type': 'document_content',
            'file_name': path.name,
            'chunk_position': f"{i+1}/{len(chunks)}"
        })
    
    return chunks


# =============================================================================
# EXPORTACIONES - API pública del módulo
# =============================================================================

__all__ = [
    # Clases principales
    'DocumentMetadata',
    'DocumentChunk', 
    
    # Funciones universales
    'create_chunk',
    'create_chunks_from_text',
    
    # Funciones específicas para web
    'create_web_chunks',
    'create_web_chunks_from_pages',
    
    # Funciones específicas para documentos
    'create_document_chunks'
]