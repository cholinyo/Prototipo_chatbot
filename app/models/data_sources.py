"""
Modelos de datos para fuentes de datos y gestión de documentos
TFM Vicente Caruncho - Sistemas Inteligentes

PROPÓSITO: Gestión de fuentes de datos y monitoreo de archivos/páginas
- DataSource: De dónde vienen los datos (base para todas las fuentes)
- ProcessedDocument: Trackeo de qué archivos se han procesado  
- ScrapedPage: Trackeo de qué páginas web se han scrapeado
- FileInfo: Información de archivos del sistema

Este módulo define las clases base y específicas para diferentes tipos de fuentes de datos:
- DocumentSource: Para archivos locales (PDF, DOCX, etc.)
- WebSource: Para sitios web (web scraping)
- APISource: Para APIs REST (futuro)
- DatabaseSource: Para bases de datos (futuro)
"""

from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import os
import json


# =============================================================================
# ENUMS - Definiciones de tipos y estados
# =============================================================================

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
    """Estados de procesamiento de documentos y páginas"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"


class FileChangeType(Enum):
    """Tipos de cambios detectados en archivos"""
    NEW = "new"
    MODIFIED = "modified"
    DELETED = "deleted"


# =============================================================================
# CLASES DE UTILIDAD - Información de archivos y cambios
# =============================================================================

@dataclass
class FileInfo:
    """Información completa de un archivo del sistema de archivos"""
    path: str
    size: int
    modified_time: datetime
    hash: str
    extension: str
    
    @classmethod
    def from_path(cls, file_path: str) -> 'FileInfo':
        """Crear FileInfo desde una ruta de archivo"""
        path = Path(file_path)
        stat = path.stat()
        
        # Calcular hash MD5 del contenido del archivo
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


# =============================================================================
# CLASES BASE - DataSource y clases relacionadas
# =============================================================================

@dataclass
class DataSource:
    """
    Fuente de datos base - Clase padre para todos los tipos de fuentes
    
    Todos los tipos específicos (DocumentSource, WebSource, etc.) heredan de esta clase.
    El campo 'config' almacena los parámetros específicos de cada tipo de fuente.
    """
    id: str
    name: str
    type: DataSourceType
    status: DataSourceStatus = DataSourceStatus.PENDING
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_sync: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para JSON - Usado por todas las subclases"""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.value,
            'status': self.status.value,
            'config': self.config,  # Aquí van los parámetros específicos de cada tipo
            'created_at': self.created_at.isoformat(),
            'last_sync': self.last_sync.isoformat() if self.last_sync else None,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataSource':
        """Crear DataSource genérico desde diccionario"""
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
class ProcessedDocument:
    """
    Documento que ha sido procesado por el sistema
    
    Trackea archivos individuales que han pasado por el pipeline de ingesta,
    manteniendo información sobre su estado y resultados del procesamiento.
    """
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
        """Convertir a diccionario para serialización"""
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
        """Crear ProcessedDocument desde diccionario"""
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
    """
    Estadísticas de ingesta para una fuente de datos
    
    Proporciona métricas sobre el rendimiento y estado de procesamiento
    de documentos o páginas web de una fuente específica.
    """
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
        """Tasa de éxito de procesamiento como porcentaje"""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100
    
    @property
    def total_size_mb(self) -> float:
        """Tamaño total en megabytes"""
        return self.total_size_bytes / (1024 * 1024)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización"""
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


# =============================================================================
# FUENTES ESPECÍFICAS - DocumentSource
# =============================================================================

@dataclass
class DocumentSource(DataSource):
    """
    Fuente de datos específica para documentos locales
    
    Maneja archivos PDF, DOCX, TXT, etc. desde directorios del sistema de archivos.
    Proporciona funcionalidades para escanear, filtrar y validar archivos.
    """
    directories: List[str] = field(default_factory=list)
    file_extensions: List[str] = field(default_factory=lambda: ['.pdf', '.docx', '.txt'])
    recursive: bool = True
    exclude_patterns: List[str] = field(default_factory=list)
    max_file_size: int = 100 * 1024 * 1024  # 100MB por defecto
    
    def __post_init__(self):
        """Inicialización y validación después de crear el objeto"""
        if self.type != DataSourceType.DOCUMENTS:
            self.type = DataSourceType.DOCUMENTS
        
        # CRÍTICO: Sincronizar configuraciones con el campo config para serialización
        self.config.update({
            'directories': self.directories,
            'file_extensions': self.file_extensions,
            'recursive': self.recursive,
            'exclude_patterns': self.exclude_patterns,
            'max_file_size': self.max_file_size
        })
    
    def is_file_supported(self, file_path: str) -> bool:
        """
        Verificar si un archivo cumple con los criterios de soporte
        
        Args:
            file_path: Ruta del archivo a verificar
            
        Returns:
            True si el archivo es soportado, False en caso contrario
        """
        path = Path(file_path)
        
        # Verificar extensión válida
        if path.suffix.lower() not in self.file_extensions:
            return False
        
        # Verificar patrones de exclusión
        for pattern in self.exclude_patterns:
            if pattern in str(path):
                return False
        
        # Verificar tamaño máximo
        try:
            if path.stat().st_size > self.max_file_size:
                return False
        except (OSError, IOError):
            return False
        
        return True
    
    def scan_directories(self) -> List[FileInfo]:
        """
        Escanear directorios configurados y retornar archivos válidos
        
        Returns:
            Lista de FileInfo para todos los archivos válidos encontrados
        """
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


# =============================================================================
# FUENTES ESPECÍFICAS - WebSource
# =============================================================================

@dataclass
class WebSource(DataSource):
    """
    Fuente de datos específica para sitios web
    
    Maneja el web scraping de sitios web con configuraciones avanzadas de extracción.
    Incluye funcionalidades para navegación controlada, extracción de contenido
    y filtrado de URLs según patrones definidos.
    """
    # URLs y dominios
    base_urls: List[str] = field(default_factory=list)
    allowed_domains: List[str] = field(default_factory=list)
    
    # Configuración de navegación
    max_depth: int = 2
    follow_links: bool = True
    respect_robots_txt: bool = True
    delay_seconds: float = 1.0
    user_agent: str = "Mozilla/5.0 (Prototipo_chatbot TFM UJI)"
    
    # Selectores CSS para extracción de contenido
    content_selectors: List[str] = field(default_factory=lambda: ['main', 'article', '.content', '#content'])
    title_selectors: List[str] = field(default_factory=lambda: ['h1', 'title', '.page-title'])
    exclude_selectors: List[str] = field(default_factory=lambda: ['nav', 'footer', '.sidebar', '#sidebar', '.menu'])
    
    # Filtros de contenido
    min_content_length: int = 100
    exclude_file_extensions: List[str] = field(default_factory=lambda: ['.pdf', '.doc', '.jpg', '.png', '.zip'])
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=lambda: ['/admin', '/login', '/api/'])
    
    # Configuración avanzada
    custom_headers: Dict[str, str] = field(default_factory=dict)
    use_javascript: bool = False  # Si requiere Selenium para JavaScript
    
    def __post_init__(self):
        """Validación y configuración específica para fuentes web"""
        if self.type != DataSourceType.WEB:
            self.type = DataSourceType.WEB
        
        # Validar que se proporcionen URLs base
        if not self.base_urls:
            raise ValueError("Se debe proporcionar al menos una URL base")
        
        # Configurar dominios permitidos automáticamente si no están especificados
        if not self.allowed_domains:
            from urllib.parse import urlparse
            self.allowed_domains = list(set(
                urlparse(url).netloc for url in self.base_urls
            ))
        
        # CRÍTICO: Sincronizar configuraciones con el campo config para serialización
        self.config.update({
            'base_urls': self.base_urls,
            'allowed_domains': self.allowed_domains,
            'max_depth': self.max_depth,
            'follow_links': self.follow_links,
            'respect_robots_txt': self.respect_robots_txt,
            'delay_seconds': self.delay_seconds,
            'user_agent': self.user_agent,
            'content_selectors': self.content_selectors,
            'title_selectors': self.title_selectors,
            'exclude_selectors': self.exclude_selectors,
            'min_content_length': self.min_content_length,
            'exclude_file_extensions': self.exclude_file_extensions,
            'include_patterns': self.include_patterns,
            'exclude_patterns': self.exclude_patterns,
            'custom_headers': self.custom_headers,
            'use_javascript': self.use_javascript
        })
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebSource':
        """Crear WebSource desde diccionario - Reconstruye desde config"""
        config = data.get('config', {})
        
        return cls(
            id=data['id'],
            name=data['name'],
            status=DataSourceStatus(data['status']),
            created_at=datetime.fromisoformat(data['created_at']),
            last_sync=datetime.fromisoformat(data['last_sync']) if data.get('last_sync') else None,
            metadata=data.get('metadata', {}),
            base_urls=config.get('base_urls', []),
            allowed_domains=config.get('allowed_domains', []),
            max_depth=config.get('max_depth', 2),
            follow_links=config.get('follow_links', True),
            respect_robots_txt=config.get('respect_robots_txt', True),
            delay_seconds=config.get('delay_seconds', 1.0),
            user_agent=config.get('user_agent', "Mozilla/5.0 (Prototipo_chatbot TFM UJI)"),
            content_selectors=config.get('content_selectors', ['main', 'article', '.content']),
            title_selectors=config.get('title_selectors', ['h1', 'title']),
            exclude_selectors=config.get('exclude_selectors', ['nav', 'footer', '.sidebar']),
            min_content_length=config.get('min_content_length', 100),
            exclude_file_extensions=config.get('exclude_file_extensions', ['.pdf', '.doc', '.jpg']),
            include_patterns=config.get('include_patterns', []),
            exclude_patterns=config.get('exclude_patterns', ['/admin', '/login']),
            custom_headers=config.get('custom_headers', {}),
            use_javascript=config.get('use_javascript', False)
        )
    
    def is_url_allowed(self, url: str) -> bool:
        """
        Verificar si una URL está permitida según las reglas configuradas
        
        Args:
            url: URL a verificar
            
        Returns:
            True si la URL está permitida, False en caso contrario
        """
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        
        # Verificar dominio permitido
        if self.allowed_domains and domain not in self.allowed_domains:
            return False
        
        # Verificar patrones de exclusión
        for pattern in self.exclude_patterns:
            if pattern in url:
                return False
        
        # Verificar extensiones de archivo excluidas
        for ext in self.exclude_file_extensions:
            if url.lower().endswith(ext):
                return False
        
        # Verificar patrones de inclusión (si están definidos)
        if self.include_patterns:
            return any(pattern in url for pattern in self.include_patterns)
        
        return True


# =============================================================================
# FUENTES ESPECÍFICAS - ScrapedPage (Trackeo de páginas web)
# =============================================================================

@dataclass
class ScrapedPage:
    """
    Página web scrapeada y procesada - versión consolidada
    
    Equivalente a ProcessedDocument para contenido web. Combina funcionalidades 
    de tracking (como ProcessedDocument) con detalles específicos de web scraping.
    Mantiene información sobre páginas individuales que han sido extraídas.
    """
    # Campos principales de identificación
    id: str
    source_id: str
    url: str
    title: str
    content: str
    
    # Información de scraping
    links_found: List[str] = field(default_factory=list)
    scraped_at: datetime = field(default_factory=datetime.now)
    content_length: int = 0
    content_hash: str = ""
    status_code: int = 200
    
    # Estado de procesamiento (siguiendo patrón de ProcessedDocument)
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    chunks_count: int = 0
    processed_at: Optional[datetime] = None
    error_message: str = ""
    
    # Metadatos adicionales
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Inicialización automática de campos calculados"""
        # Calcular content_length si no está establecido
        if not self.content_length and self.content:
            self.content_length = len(self.content)
        
        # Calcular hash MD5 del contenido si no está establecido
        if not self.content_hash and self.content:
            import hashlib
            self.content_hash = hashlib.md5(self.content.encode('utf-8')).hexdigest()
        
        # Asegurar que el ID no esté vacío
        if not self.id:
            import uuid
            self.id = str(uuid.uuid4())
        
        # Añadir metadatos automáticos de dominio
        if self.url and 'domain' not in self.metadata:
            from urllib.parse import urlparse
            self.metadata['domain'] = urlparse(self.url).netloc
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización/persistencia"""
        return {
            'id': self.id,
            'source_id': self.source_id,
            'url': self.url,
            'title': self.title,
            'content': self.content,
            'links_found': self.links_found,
            'scraped_at': self.scraped_at.isoformat(),
            'content_length': self.content_length,
            'content_hash': self.content_hash,
            'status_code': self.status_code,
            'processing_status': self.processing_status.value,
            'chunks_count': self.chunks_count,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'error_message': self.error_message,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScrapedPage':
        """Crear ScrapedPage desde diccionario"""
        return cls(
            id=data['id'],
            source_id=data['source_id'],
            url=data['url'],
            title=data['title'],
            content=data['content'],
            links_found=data.get('links_found', []),
            scraped_at=datetime.fromisoformat(data['scraped_at']),
            content_length=data.get('content_length', 0),
            content_hash=data.get('content_hash', ''),
            status_code=data.get('status_code', 200),
            processing_status=ProcessingStatus(data.get('processing_status', 'pending')),
            chunks_count=data.get('chunks_count', 0),
            processed_at=datetime.fromisoformat(data['processed_at']) if data.get('processed_at') else None,
            error_message=data.get('error_message', ''),
            metadata=data.get('metadata', {})
        )
    
    @classmethod
    def from_response(cls, url: str, title: str, content: str, 
                     links: List[str], source_id: str,
                     response: Optional[object] = None) -> 'ScrapedPage':
        """
        Crear ScrapedPage desde respuesta HTTP (compatibilidad con web_scraper_service)
        
        Args:
            url: URL de la página
            title: Título extraído de la página
            content: Contenido principal extraído
            links: Enlaces encontrados en la página
            source_id: ID de la fuente web asociada
            response: Objeto response HTTP opcional para metadatos
            
        Returns:
            Nueva instancia de ScrapedPage
        """
        import hashlib
        import uuid
        
        # Calcular hash del contenido
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        # Extraer información de la respuesta HTTP si está disponible
        status_code = 200
        metadata = {}
        
        if response and hasattr(response, 'status_code'):
            status_code = response.status_code
            if hasattr(response, 'headers'):
                metadata.update({
                    'content_type': response.headers.get('content-type', ''),
                    'last_modified': response.headers.get('last-modified', ''),
                    'server': response.headers.get('server', ''),
                })
        
        # Añadir metadatos de la página
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        metadata.update({
            'domain': parsed_url.netloc,
            'path': parsed_url.path,
            'links_count': len(links),
            'content_length': len(content)
        })
        
        return cls(
            id=str(uuid.uuid4()),
            source_id=source_id,
            url=url,
            title=title,
            content=content,
            links_found=links,
            content_hash=content_hash,
            status_code=status_code,
            metadata=metadata
        )
    
    def update_processing_status(self, status: ProcessingStatus, 
                               chunks_count: int = 0, 
                               error_message: str = ""):
        """
        Actualizar estado de procesamiento de la página
        
        Args:
            status: Nuevo estado de procesamiento
            chunks_count: Número de chunks creados
            error_message: Mensaje de error si aplica
        """
        self.processing_status = status
        self.chunks_count = chunks_count
        self.error_message = error_message
        
        if status == ProcessingStatus.COMPLETED:
            self.processed_at = datetime.now()
    
    @property
    def word_count(self) -> int:
        """Número de palabras en el contenido"""
        return len(self.content.split()) if self.content else 0
    
    @property
    def content_size_mb(self) -> float:
        """Tamaño del contenido en megabytes"""
        return self.content_length / (1024 * 1024)


# =============================================================================
# CONFIGURACIONES POR DEFECTO
# =============================================================================

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


# =============================================================================
# FACTORY FUNCTIONS - Funciones para crear fuentes de datos
# =============================================================================

def create_document_source(
    name: str,
    directories: List[str],
    source_id: Optional[str] = None,
    **kwargs
) -> DocumentSource:
    """
    Factory function para crear fuentes de documentos
    
    Args:
        name: Nombre descriptivo de la fuente
        directories: Lista de directorios a escanear
        source_id: ID específico (se genera automáticamente si es None)
        **kwargs: Parámetros adicionales (file_extensions, recursive, etc.)
        
    Returns:
        Nueva instancia de DocumentSource configurada
    """
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


def create_web_source(
    name: str,
    base_urls: List[str],
    source_id: Optional[str] = None,
    **kwargs
) -> WebSource:
    """
    Factory function para crear fuentes web
    
    Args:
        name: Nombre descriptivo de la fuente
        base_urls: Lista de URLs base para iniciar el scraping
        source_id: ID específico (se genera automáticamente si es None)
        **kwargs: Parámetros adicionales (max_depth, selectors, etc.)
        
    Returns:
        Nueva instancia de WebSource configurada
    """
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


# =============================================================================
# EXPORTACIONES - API pública del módulo
# =============================================================================

__all__ = [
    # Enums
    'DataSourceType', 
    'DataSourceStatus', 
    'ProcessingStatus', 
    'FileChangeType',
    
    # Clases de utilidad
    'FileInfo', 
    'FileChange',
    
    # Clases base
    'DataSource', 
    'ProcessedDocument', 
    'IngestionStats',
    
    # Fuentes específicas
    'DocumentSource', 
    'WebSource',
    
    # Páginas web scrapeadas
    'ScrapedPage',
    
    # Factory functions
    'create_document_source', 
    'create_web_source',
    
    # Configuraciones por defecto
    'DEFAULT_DOCUMENT_EXTENSIONS',
    'DEFAULT_EXCLUDE_PATTERNS',
    'DEFAULT_CHUNK_SETTINGS'
]