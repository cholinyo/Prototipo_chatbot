"""
Sistema de logging estructurado para Prototipo_chatbot
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import structlog
from structlog.stdlib import LoggerFactory
import colorlog
from app.core.config import config_manager

class ChatbotLogger:
    """Configurador de logging para el sistema"""
    
    def __init__(self):
        self.logger_configured = False
        self._setup_directories()
    
    def _setup_directories(self):
        """Crear directorios de logs si no existen"""
        log_config = config_manager.get_log_config()
        log_file = log_config.get('file', 'logs/prototipo_chatbot.log')
        
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self, log_level: Optional[str] = None) -> None:
        """Configurar sistema de logging"""
        if self.logger_configured:
            return
        
        log_config = config_manager.get_log_config()
        level = log_level or log_config.get('level', 'INFO')
        log_file = log_config.get('file', 'logs/prototipo_chatbot.log')
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Configurar structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer() if not config_manager.is_development() else structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Configurar logging estándar
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper()))
        
        # Remover handlers existentes
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Handler para consola con colores (desarrollo)
        if config_manager.is_development():
            console_handler = colorlog.StreamHandler(sys.stdout)
            console_handler.setFormatter(colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green', 
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            ))
            console_handler.setLevel(getattr(logging, level.upper()))
            root_logger.addHandler(console_handler)
        
        # Handler para archivo con rotación
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=log_config.get('max_bytes', 10485760),  # 10MB
            backupCount=log_config.get('backup_count', 5),
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        file_handler.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(file_handler)
        
        # Configurar loggers específicos
        self._configure_specific_loggers()
        
        self.logger_configured = True
        
        # Log inicial
        logger = self.get_logger("system")
        logger.info("Sistema de logging inicializado", 
                   level=level, 
                   file=log_file,
                   development=config_manager.is_development())
    
    def _configure_specific_loggers(self):
        """Configurar loggers específicos con niveles apropiados"""
        # Reducir verbosidad de librerías externas
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
        logging.getLogger('transformers').setLevel(logging.WARNING)
        logging.getLogger('chromadb').setLevel(logging.INFO)
        logging.getLogger('faiss').setLevel(logging.WARNING)
        
        # Loggers específicos del sistema
        loggers_config = {
            'prototipo_chatbot': logging.INFO,
            'prototipo_chatbot.models': logging.INFO,
            'prototipo_chatbot.rag': logging.INFO,
            'prototipo_chatbot.ingestion': logging.INFO,
            'prototipo_chatbot.vectorstore': logging.INFO,
            'prototipo_chatbot.chat': logging.INFO,
            'prototipo_chatbot.api': logging.INFO,
        }
        
        for logger_name, level in loggers_config.items():
            logging.getLogger(logger_name).setLevel(level)
    
    def get_logger(self, name: str) -> structlog.stdlib.BoundLogger:
        """Obtener logger estructurado para un módulo específico"""
        if not self.logger_configured:
            self.setup_logging()
        
        return structlog.get_logger(f"prototipo_chatbot.{name}")
    
    def log_model_interaction(self, logger: structlog.stdlib.BoundLogger, 
                            model_name: str, 
                            query: str, 
                            response_time: float,
                            success: bool = True,
                            error: Optional[str] = None):
        """Log específico para interacciones con modelos"""
        log_data = {
            'event': 'model_interaction',
            'model': model_name,
            'query_length': len(query),
            'response_time_seconds': response_time,
            'success': success
        }
        
        if error:
            log_data['error'] = error
            logger.error("Error en interacción con modelo", **log_data)
        else:
            logger.info("Interacción con modelo exitosa", **log_data)
    
    def log_rag_operation(self, logger: structlog.stdlib.BoundLogger,
                         query: str,
                         vector_store: str,
                         k: int,
                         results_found: int,
                         search_time: float):
        """Log específico para operaciones RAG"""
        logger.info("Operación RAG completada",
                   event='rag_search',
                   query_length=len(query),
                   vector_store=vector_store,
                   k_requested=k,
                   results_found=results_found,
                   search_time_seconds=search_time)
    
    def log_ingestion_operation(self, logger: structlog.stdlib.BoundLogger,
                              source_type: str,
                              documents_processed: int,
                              processing_time: float,
                              success: bool = True,
                              errors: int = 0):
        """Log específico para operaciones de ingesta"""
        log_data = {
            'event': 'ingestion_operation',
            'source_type': source_type,
            'documents_processed': documents_processed,
            'processing_time_seconds': processing_time,
            'success': success,
            'errors': errors
        }
        
        if success:
            logger.info("Ingesta completada exitosamente", **log_data)
        else:
            logger.error("Ingesta completada con errores", **log_data)
    
    def log_user_interaction(self, logger: structlog.stdlib.BoundLogger,
                           user_ip: str,
                           endpoint: str,
                           query: Optional[str] = None,
                           response_time: Optional[float] = None,
                           status_code: int = 200):
        """Log específico para interacciones de usuarios"""
        log_data = {
            'event': 'user_interaction',
            'user_ip': user_ip,
            'endpoint': endpoint,
            'status_code': status_code
        }
        
        if query:
            log_data['query_length'] = len(query)
        
        if response_time:
            log_data['response_time_seconds'] = response_time
        
        if status_code >= 400:
            logger.warning("Interacción de usuario con error", **log_data)
        else:
            logger.info("Interacción de usuario exitosa", **log_data)

# Instancia global del logger
chatbot_logger = ChatbotLogger()

# Funciones de conveniencia
def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Obtener logger para un módulo específico"""
    return chatbot_logger.get_logger(name)

def setup_logging(log_level: Optional[str] = None) -> None:
    """Configurar sistema de logging"""
    chatbot_logger.setup_logging(log_level)