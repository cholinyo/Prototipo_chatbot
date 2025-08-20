"""
Modelos de datos para fuentes de datos y gestión de documentos
TFM Vicente Caruncho - Sistemas Inteligentes
"""

from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import os
import json


class DataSourceType(Enum):
    """Tipos de fuentes de datos soportadas"""
    DOCUMENTS = "documents"
    WEB = "web"
    API = "api" 
    DATABASE = "database"


class DataSourceStatus(Enum):
    """Estados de las fuentes de datos"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PROCESSING = "processing"
    ERROR = "error"
    PENDING = "pending"


class ProcessingStatus(Enum):
    """Estados de procesamiento de documentos"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"


class FileChangeType(Enum):
    """Tipos de cambios en archivos"""
    NEW = "new"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class FileInfo:
    """Información de un archivo del sistema"""
    path: str
    size: int
    modified_time: datetime
    hash: str
    extension: str
    
    @classmethod
    def from_path(cls, file_path: str) -> 'FileInfo':
        """Crear FileInfo desde una ruta"""
        path = Path(file_path)
        stat = path.stat()
        
        # Calcular hash del archivo
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        except (IOError, OSError):
            hash_md5.update(b"")  # Hash vacío si no se puede leer
        
        return cls(
            path=str(path.absolute()),
            size=stat.st_size,
            modified_time=datetime.fromtimestamp(stat.st_mtime),
            hash=hash_md5.hexdigest(),
            extension=path.suffix.lower()
        )


@dataclass 
class FileChange:
    """Representa un cambio detectado en un archivo"""
    type: FileChangeType
    file_info: FileInfo
    previous_info: Optional['ProcessedDocument'] = None


@dataclass
class DataSource:
    """Fuente de datos base"""
    id: str
    name: str
    type: DataSourceType
    status: DataSourceStatus = DataSourceStatus.PENDING
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_sync: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para JSON"""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.value,
            'status': self.status.value,
            'config': self.config,
            'created_at': self.created_at.isoformat(),
            'last_sync': self.last_sync.isoformat() if self.last_sync else None,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataSource':
        """Crear desde diccionario"""
        return cls(
            id=data['id'],
            name=data['name'],
            type=DataSourceType(data['type']),
            status=DataSourceStatus(data['status']),
            config=data.get('config', {}),
            created_at=datetime.fromisoformat(data['created_at']),
            last_sync=datetime.fromisoformat(data['last_sync']) if data.get('last_sync') else None,
            metadata=data.get('metadata', {})
        )


@dataclass
class DocumentSource(DataSource):
    """Fuente de datos específica para documentos"""
    directories: List[str] = field(default_factory=list)
    file_extensions: List[str] = field(default_factory=lambda: ['.pdf', '.docx', '.txt'])
    recursive: bool = True
    exclude_patterns: List[str] = field(default_factory=list)
    max_file_size: int = 100 * 1024 * 1024  # 100MB por defecto
    
    def __post_init__(self):
        """Inicialización después de crear el objeto"""
        if self.type != DataSourceType.DOCUMENTS:
            self.type = DataSourceType.DOCUMENTS
        
        # Asegurar que las configuraciones estén en config
        self.config.update({
            'directories': self.directories,
            'file_extensions': self.file_extensions,
            'recursive': self.recursive,
            'exclude_patterns': self.exclude_patterns,
            'max_file_size': self.max_file_size
        })
    
    def is_file_supported(self, file_path: str) -> bool:
        """Verificar si un archivo es soportado"""
        path = Path(file_path)
        
        # Verificar extensión
        if path.suffix.lower() not in self.file_extensions:
            return False
        
        # Verificar patrones de exclusión
        for pattern in self.exclude_patterns:
            if pattern in str(path):
                return False
        
        # Verificar tamaño
        try:
            if path.stat().st_size > self.max_file_size:
                return False
        except (OSError, IOError):
            return False
        
        return True
    
    def scan_directories(self) -> List[FileInfo]:
        """Escanear directorios configurados y retornar archivos válidos"""
        files = []
        
        for directory in self.directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                continue
            
            # Escanear según configuración recursiva
            pattern = "**/*" if self.recursive else "*"
            
            for file_path in dir_path.glob(pattern):
                if file_path.is_file() and self.is_file_supported(str(file_path)):
                    try:
                        file_info = FileInfo.from_path(str(file_path))
                        files.append(file_info)
                    except (OSError, IOError):
                        # Saltar archivos que no se pueden leer
                        continue
        
        return files


@dataclass
class ProcessedDocument:
    """Documento que ha sido procesado"""
    id: str
    source_id: str
    file_path: str
    file_hash: str
    file_size: int
    modified_time: datetime
    processed_at: Optional[datetime] = None
    chunks_count: int = 0
    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            'id': self.id,
            'source_id': self.source_id,
            'file_path': self.file_path,
            'file_hash': self.file_hash,
            'file_size': self.file_size,
            'modified_time': self.modified_time.isoformat(),
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'chunks_count': self.chunks_count,
            'status': self.status.value,
            'error_message': self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessedDocument':
        """Crear desde diccionario"""
        return cls(
            id=data['id'],
            source_id=data['source_id'],
            file_path=data['file_path'],
            file_hash=data['file_hash'],
            file_size=data['file_size'],
            modified_time=datetime.fromisoformat(data['modified_time']),
            processed_at=datetime.fromisoformat(data['processed_at']) if data.get('processed_at') else None,
            chunks_count=data.get('chunks_count', 0),
            status=ProcessingStatus(data.get('status', 'pending')),
            error_message=data.get('error_message', '')
        )


@dataclass
class IngestionStats:
    """Estadísticas de ingesta para una fuente"""
    source_id: str
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    total_size_bytes: int = 0
    last_scan: Optional[datetime] = None
    processing_time_seconds: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Tasa de éxito de procesamiento"""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100
    
    @property
    def total_size_mb(self) -> float:
        """Tamaño total en MB"""
        return self.total_size_bytes / (1024 * 1024)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            'source_id': self.source_id,
            'total_files': self.total_files,
            'processed_files': self.processed_files,
            'failed_files': self.failed_files,
            'total_chunks': self.total_chunks,
            'total_size_bytes': self.total_size_bytes,
            'total_size_mb': self.total_size_mb,
            'last_scan': self.last_scan.isoformat() if self.last_scan else None,
            'processing_time_seconds': self.processing_time_seconds,
            'success_rate': self.success_rate
        }


# Configuraciones por defecto
DEFAULT_DOCUMENT_EXTENSIONS = [
    '.pdf', '.docx', '.doc', '.txt', '.md', '.rtf', '.odt', '.epub'
]

DEFAULT_EXCLUDE_PATTERNS = [
    '__pycache__', '.git', '.svn', 'node_modules', 
    'temp', 'tmp', '.DS_Store', 'Thumbs.db'
]

DEFAULT_CHUNK_SETTINGS = {
    'chunk_size': 500,
    'chunk_overlap': 50,
    'separator': '\n\n'
}


def create_document_source(
    name: str,
    directories: List[str],
    source_id: Optional[str] = None,
    **kwargs
) -> DocumentSource:
    """Factory function para crear fuentes de documentos"""
    import uuid
    
    if source_id is None:
        source_id = str(uuid.uuid4())
    
    return DocumentSource(
        id=source_id,
        name=name,
        type=DataSourceType.DOCUMENTS,
        directories=directories,
        file_extensions=kwargs.get('file_extensions', DEFAULT_DOCUMENT_EXTENSIONS.copy()),
        recursive=kwargs.get('recursive', True),
        exclude_patterns=kwargs.get('exclude_patterns', DEFAULT_EXCLUDE_PATTERNS.copy()),
        max_file_size=kwargs.get('max_file_size', 100 * 1024 * 1024),
        metadata=kwargs.get('metadata', {})
    )
@dataclass
class WebSource(DataSource):
    """Fuente de datos específica para sitios web"""
    base_urls: List[str] = field(default_factory=list)
    allowed_domains: List[str] = field(default_factory=list)
    max_depth: int = 2
    follow_links: bool = True
    respect_robots_txt: bool = True
    delay_seconds: float = 1.0
    user_agent: str = "Mozilla/5.0 (Prototipo_chatbot TFM UJI)"
    
    # Selectores CSS para extracción
    content_selectors: List[str] = field(default_factory=lambda: ['main', 'article', '.content', '#content'])
    title_selectors: List[str] = field(default_factory=lambda: ['h1', 'title', '.page-title'])
    exclude_selectors: List[str] = field(default_factory=lambda: ['nav', 'footer', '.sidebar', '#sidebar', '.menu'])
    
    # Filtros de contenido
    min_content_length: int = 100
    exclude_file_extensions: List[str] = field(default_factory=lambda: ['.pdf', '.doc', '.jpg', '.png', '.zip'])
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=lambda: ['/admin', '/login', '/api/'])
    
    # Configuración de headers
    custom_headers: Dict[str, str] = field(default_factory=dict)
    use_javascript: bool = False  # Si requiere Selenium
    
    def __post_init__(self):
        """Validación específica para fuentes web"""
        if self.type != DataSourceType.WEB:
            self.type = DataSourceType.WEB
        
        # Validar URLs
        if not self.base_urls:
            raise ValueError("Se debe proporcionar al menos una URL base")
        
        # Configurar dominios permitidos si no están especificados
        if not self.allowed_domains:
            from urllib.parse import urlparse
            self.allowed_domains = list(set(
                urlparse(url).netloc for url in self.base_urls
            ))
    
    def is_url_allowed(self, url: str) -> bool:
        """Verificar si una URL está permitida"""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        
        # Verificar dominio permitido
        if self.allowed_domains and domain not in self.allowed_domains:
            return False
        
        # Verificar patrones de exclusión
        for pattern in self.exclude_patterns:
            if pattern in url:
                return False
        
        # Verificar extensiones de archivo
        for ext in self.exclude_file_extensions:
            if url.lower().endswith(ext):
                return False
        
        # Verificar patrones de inclusión (si están definidos)
        if self.include_patterns:
            return any(pattern in url for pattern in self.include_patterns)
        
        return True


# Añadir al enum DataSourceType
# (modificar la definición existente)
class DataSourceType(Enum):
    DOCUMENTS = "documents"
    WEB = "web"  
    API = "api"
    DATABASE = "database"


def create_web_source(
    name: str,
    base_urls: List[str],
    source_id: Optional[str] = None,
    **kwargs
) -> WebSource:
    """Factory function para crear fuentes web"""
    import uuid
    
    if source_id is None:
        source_id = str(uuid.uuid4())
    
    return WebSource(
        id=source_id,
        name=name,
        type=DataSourceType.WEB,
        base_urls=base_urls,
        allowed_domains=kwargs.get('allowed_domains', []),
        max_depth=kwargs.get('max_depth', 2),
        follow_links=kwargs.get('follow_links', True),
        respect_robots_txt=kwargs.get('respect_robots_txt', True),
        delay_seconds=kwargs.get('delay_seconds', 1.0),
        user_agent=kwargs.get('user_agent', "Mozilla/5.0 (Prototipo_chatbot TFM UJI)"),
        content_selectors=kwargs.get('content_selectors', ['main', 'article', '.content']),
        title_selectors=kwargs.get('title_selectors', ['h1', 'title']),
        exclude_selectors=kwargs.get('exclude_selectors', ['nav', 'footer', '.sidebar']),
        min_content_length=kwargs.get('min_content_length', 100),
        exclude_file_extensions=kwargs.get('exclude_file_extensions', ['.pdf', '.doc', '.jpg']),
        include_patterns=kwargs.get('include_patterns', []),
        exclude_patterns=kwargs.get('exclude_patterns', ['/admin', '/login']),
        custom_headers=kwargs.get('custom_headers', {}),
        use_javascript=kwargs.get('use_javascript', False),
        metadata=kwargs.get('metadata', {})
    )