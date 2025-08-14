// JavaScript principal para la aplicación
document.addEventListener('DOMContentLoaded', function() {
    console.log('Prototipo_chatbot cargado correctamente');
    
    // Inicializar tooltips de Bootstrap
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Verificar estado del sistema al cargar y cada minuto
    checkSystemStatus();
    setInterval(checkSystemStatus, 60000);
});

function checkSystemStatus() {
    console.log('Verificando estado del sistema...');
    
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            console.log('Estado del sistema:', data);
            updateSystemStatus(data);
        })
        .catch(error => {
            console.error('Error verificando estado:', error);
            updateSystemStatus({
                status: 'error',
                services: {
                    ollama: 'unavailable',
                    openai: 'unavailable'
                }
            });
        });
}

function updateSystemStatus(data) {
    // Actualizar indicador principal
    const indicator = document.getElementById('status-indicator');
    if (indicator) {
        updateStatusIndicator(data.status);
    }
    
    // Actualizar modelos locales
    const localModelsStatus = document.getElementById('local-models-status');
    if (localModelsStatus) {
        if (data.services && data.services.ollama === 'available') {
            localModelsStatus.className = 'badge bg-success';
            localModelsStatus.innerHTML = '<i class="fas fa-check me-1"></i>Disponible';
        } else {
            localModelsStatus.className = 'badge bg-warning';
            localModelsStatus.innerHTML = '<i class="fas fa-exclamation-triangle me-1"></i>No configurado';
        }
    }
    
    // Actualizar OpenAI API
    const openaiStatus = document.getElementById('openai-status');
    if (openaiStatus) {
        if (data.services && data.services.openai === 'configured') {
            openaiStatus.className = 'badge bg-success';
            openaiStatus.innerHTML = '<i class="fas fa-check me-1"></i>Configurado';
        } else {
            openaiStatus.className = 'badge bg-secondary';
            openaiStatus.innerHTML = '<i class="fas fa-times me-1"></i>No configurado';
        }
    }
}

function updateStatusIndicator(status) {
    const indicator = document.getElementById('status-indicator');
    if (!indicator) return;
    
    const statusClasses = {
        'healthy': 'bg-success',
        'degraded': 'bg-warning', 
        'error': 'bg-danger'
    };
    
    const statusTexts = {
        'healthy': 'Sistema Activo',
        'degraded': 'Sistema con Alertas',
        'error': 'Sistema con Errores'
    };
    
    // Limpiar clases anteriores
    indicator.className = indicator.className.replace(/bg-\w+/g, '');
    
    // Aplicar nueva clase
    indicator.classList.add('badge', statusClasses[status] || 'bg-secondary');
    indicator.innerHTML = `<i class="fas fa-circle me-1"></i>${statusTexts[status] || 'Estado Desconocido'}`;
}

// Funciones para navegación (placeholder)
function navigateToChat() {
    window.location.href = '/chat';
}

function navigateToComparison() {
    alert('Comparador de modelos en desarrollo');
}

function navigateToAdmin() {
    alert('Panel de administración en desarrollo');
}

function navigateToConfig() {
    alert('Configuración de fuentes de datos en desarrollo');
}

function showComingSoon(feature) {
    alert(feature + ' próximamente disponible');
}
