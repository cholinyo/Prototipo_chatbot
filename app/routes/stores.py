"""
Stores Routes para Dashboard de Vector Stores
TFM Vicente Caruncho - Sistemas Inteligentes

Rutas principales para la gestión de Vector Stores
"""

from flask import Blueprint, render_template
import time
from app.core.logger import get_logger

# Crear blueprint
stores_bp = Blueprint('stores', __name__)
logger = get_logger("stores_routes")

@stores_bp.route('/admin')
def admin_stores_dashboard():
    """Dashboard administrativo de Vector Stores"""
    try:
        # Obtener datos básicos para el template
        from app.services.vector_store_service import get_vector_store_service
        
        vector_service = get_vector_store_service()
        
        context = {
            'title': 'Vector Stores Administration',
            'system_available': vector_service.is_available(),
            'active_store': vector_service.get_active_store_type(),
            'faiss_available': vector_service.faiss_available,
            'chromadb_available': vector_service.chromadb_available,
            'timestamp': time.time()
        }
        
        return render_template('admin_stores.html', **context)
        
    except Exception as e:
        logger.error(f"Error cargando dashboard: {e}")
        return render_template('admin_stores.html', 
                             title='Vector Stores Administration',
                             error=f"Error: {str(e)}")

@stores_bp.route('/admin/status')
def stores_status():
    """Página de estado simple (alternativa sin JS)"""
    try:
        from app.services.vector_store_service import get_vector_store_service
        
        vector_service = get_vector_store_service()
        stats = vector_service.get_stats()
        health = vector_service.get_health_status()
        
        return render_template('stores_status.html', 
                             stats=stats, 
                             health=health)
    except Exception as e:
        logger.error(f"Error obteniendo estado: {e}")
        return f"Error: {str(e)}", 500