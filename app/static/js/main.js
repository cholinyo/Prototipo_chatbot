/**
 * Prototipo_chatbot - JavaScript Principal
 * TFM Vicente Caruncho - Sistemas Inteligentes UJI
 * 
 * Maneja la interfaz del dashboard, verificación de estado del sistema
 * y comunicación con el backend para mostrar métricas en tiempo real.
 */

/**
 * Configuración global del sistema
 */
const CONFIG = {
    HEALTH_CHECK_INTERVAL: 60000, // 1 minuto
    UPDATE_ANIMATION_DURATION: 500,
    RETRY_ATTEMPTS: 3,
    RETRY_DELAY: 2000
};

/**
 * Estado global de la aplicación
 */
const AppState = {
    lastHealthCheck: null,
    isUpdating: false,
    retryCount: 0,
    tooltips: []
};

/**
 * Inicialización principal de la aplicación
 * Se ejecuta cuando el DOM está completamente cargado
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Prototipo_chatbot cargado correctamente');
    console.log('📊 TFM Vicente Caruncho - Sistemas Inteligentes UJI');
    
    initializeApplication();
});

/**
 * Inicializa todos los componentes de la aplicación
 */
function initializeApplication() {
    try {
        initializeBootstrapComponents();
        setupEventListeners();
        startSystemMonitoring();
        
        console.log('✅ Aplicación inicializada correctamente');
    } catch (error) {
        console.error('❌ Error inicializando aplicación:', error);
        showErrorNotification('Error inicializando la aplicación');
    }
}

/**
 * Inicializa componentes de Bootstrap (tooltips, etc.)
 */
function initializeBootstrapComponents() {
    console.log('🎨 Inicializando componentes Bootstrap...');
    
    // Inicializar tooltips
    const tooltipElements = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    AppState.tooltips = Array.from(tooltipElements).map(element => {
        return new bootstrap.Tooltip(element, {
            trigger: 'hover focus',
            placement: 'top',
            animation: true
        });
    });
    
    console.log(`✅ ${AppState.tooltips.length} tooltips inicializados`);
}

/**
 * Configura todos los event listeners de la interfaz
 */
function setupEventListeners() {
    console.log('⚡ Configurando event listeners...');
    
    // Botón de actualización manual
    const updateButton = document.getElementById('update-status-btn');
    if (updateButton) {
        updateButton.addEventListener('click', handleManualUpdate);
        console.log('✅ Event listener para botón de actualización configurado');
    }
    
    // Botón "Actualizar Estado" (si existe)
    const refreshButton = document.querySelector('[onclick*="checkSystemStatus"]');
    if (refreshButton) {
        refreshButton.removeAttribute('onclick');
        refreshButton.addEventListener('click', handleManualUpdate);
        console.log('✅ Event listener para botón de refresh configurado');
    }
    
    // Listener para errores globales de JavaScript
    window.addEventListener('error', function(event) {
        console.error('❌ Error JavaScript global:', event.error);
    });
}

/**
 * Inicia el monitoreo automático del sistema
 */
function startSystemMonitoring() {
    console.log('🔍 Iniciando monitoreo del sistema...');
    
    // Primera verificación inmediata
    performSystemCheck();
    
    // Configurar verificación periódica
    setInterval(() => {
        if (!AppState.isUpdating) {
            performSystemCheck();
        }
    }, CONFIG.HEALTH_CHECK_INTERVAL);
    
    console.log(`✅ Monitoreo configurado (cada ${CONFIG.HEALTH_CHECK_INTERVAL / 1000}s)`);
}

/**
 * Maneja la actualización manual del estado
 * @param {Event} event - Evento del click
 */
async function handleManualUpdate(event) {
    event.preventDefault();
    
    const button = event.target.closest('button');
    if (!button || AppState.isUpdating) return;
    
    console.log('🔄 Actualización manual solicitada');
    
    // Actualizar UI del botón
    const originalContent = button.innerHTML;
    button.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Actualizando...';
    button.disabled = true;
    AppState.isUpdating = true;
    
    try {
        await performSystemCheck(true);
        showSuccessNotification('Estado actualizado correctamente');
    } catch (error) {
        showErrorNotification('Error actualizando el estado');
    } finally {
        // Restaurar botón después de un delay mínimo
        setTimeout(() => {
            button.innerHTML = originalContent;
            button.disabled = false;
            AppState.isUpdating = false;
        }, CONFIG.UPDATE_ANIMATION_DURATION);
    }
}

/**
 * Realiza la verificación del estado del sistema
 * @param {boolean} isManual - Indica si es una verificación manual
 * @returns {Promise<Object>} Datos del estado del sistema
 */
async function performSystemCheck(isManual = false) {
    const startTime = performance.now();
    console.log(`🔍 ${isManual ? 'Manual' : 'Automática'} verificación del sistema iniciada`);
    
    try {
        const response = await fetch('/health', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache'
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        const duration = Math.round(performance.now() - startTime);
        
        console.log(`✅ Estado del sistema obtenido (${duration}ms):`, data);
        
        // Actualizar interfaz con los datos recibidos
        updateSystemInterface(data);
        
        // Actualizar estado de la aplicación
        AppState.lastHealthCheck = new Date();
        AppState.retryCount = 0;
        
        return data;
        
    } catch (error) {
        const duration = Math.round(performance.now() - startTime);
        console.error(`❌ Error verificando estado del sistema (${duration}ms):`, error);
        
        // Manejar el error y mostrar estado de error
        handleSystemCheckError(error);
        
        throw error;
    }
}

/**
 * Actualiza toda la interfaz con los datos del sistema
 * @param {Object} data - Datos del estado del sistema
 */
function updateSystemInterface(data) {
    console.log('🔄 Actualizando interfaz del sistema...');
    
    try {
        // Actualizar indicadores principales
        updateMainStatusIndicator(data.status);
        updateLocalModelsStatus(data);
        updateOpenAIStatus(data);
        updateAdditionalIndicators(data);
        
        // Actualizar timestamp de última verificación
        updateLastCheckTimestamp();
        
        console.log('✅ Interfaz actualizada correctamente');
        
    } catch (error) {
        console.error('❌ Error actualizando interfaz:', error);
        showErrorNotification('Error actualizando la interfaz');
    }
}

/**
 * Actualiza el estado de los modelos locales (Ollama)
 * @param {Object} data - Datos del sistema
 */
function updateLocalModelsStatus(data) {
    const element = document.getElementById('local-models-status');
    if (!element) {
        console.warn('⚠️ Elemento local-models-status no encontrado');
        return;
    }
    
    const services = data.services || {};
    const ollamaStatus = services.ollama;
    
    console.log('🦙 Actualizando estado Ollama:', ollamaStatus);
    
    // Limpiar clases anteriores
    element.className = 'badge';
    
    switch (ollamaStatus) {
        case 'available':
            element.classList.add('bg-success');
            element.innerHTML = '<i class="fas fa-check me-1"></i>Disponible';
            
            // Añadir información de modelos en tooltip
            const models = data.models?.ollama || [];
            if (models.length > 0) {
                element.title = `Modelos disponibles: ${models.join(', ')}`;
                console.log(`📦 Modelos Ollama: ${models.join(', ')}`);
            }
            break;
            
        case 'unavailable':
            element.classList.add('bg-danger');
            element.innerHTML = '<i class="fas fa-times me-1"></i>No disponible';
            element.title = 'Ollama no está ejecutándose. Ejecutar: ollama serve';
            break;
            
        default:
            element.classList.add('bg-warning');
            element.innerHTML = '<i class="fas fa-exclamation-triangle me-1"></i>No configurado';
            element.title = 'Estado de Ollama desconocido';
            console.warn('⚠️ Estado Ollama desconocido:', ollamaStatus);
    }
}

/**
 * Actualiza el estado de la API de OpenAI
 * @param {Object} data - Datos del sistema
 */
function updateOpenAIStatus(data) {
    const element = document.getElementById('openai-status');
    if (!element) {
        console.warn('⚠️ Elemento openai-status no encontrado');
        return;
    }
    
    const services = data.services || {};
    const openaiStatus = services.openai;
    
    console.log('🌐 Actualizando estado OpenAI:', openaiStatus);
    
    // Limpiar clases anteriores
    element.className = 'badge';
    
    switch (openaiStatus) {
        case 'configured':
            element.classList.add('bg-success');
            element.innerHTML = '<i class="fas fa-check me-1"></i>Configurado';
            element.title = 'API Key de OpenAI configurada correctamente';
            
            // Mostrar modelos disponibles si existen
            const models = data.models?.openai || [];
            if (models.length > 0) {
                console.log(`🤖 Modelos OpenAI: ${models.join(', ')}`);
            }
            break;
            
        case 'not_configured':
            element.classList.add('bg-secondary');
            element.innerHTML = '<i class="fas fa-times me-1"></i>No configurado';
            element.title = 'API Key de OpenAI no configurada. Revisar archivo .env';
            break;
            
        default:
            element.classList.add('bg-warning');
            element.innerHTML = '<i class="fas fa-exclamation-triangle me-1"></i>Estado desconocido';
            element.title = 'No se pudo verificar el estado de OpenAI';
            console.warn('⚠️ Estado OpenAI desconocido:', openaiStatus);
    }
}

/**
 * Actualiza el indicador principal de estado del sistema
 * @param {string} status - Estado general del sistema
 */
function updateMainStatusIndicator(status) {
    const element = document.getElementById('status-indicator');
    if (!element) {
        console.warn('⚠️ Elemento status-indicator no encontrado');
        return;
    }
    
    console.log('📊 Actualizando indicador principal:', status);
    
    const statusConfig = {
        'healthy': {
            class: 'bg-success',
            text: 'Sistema Activo',
            icon: 'fas fa-circle'
        },
        'degraded': {
            class: 'bg-warning',
            text: 'Sistema con Alertas', 
            icon: 'fas fa-exclamation-circle'
        },
        'error': {
            class: 'bg-danger',
            text: 'Sistema con Errores',
            icon: 'fas fa-times-circle'
        }
    };
    
    const config = statusConfig[status] || {
        class: 'bg-secondary',
        text: 'Estado Desconocido',
        icon: 'fas fa-question-circle'
    };
    
    // Actualizar clases y contenido
    element.className = `badge ${config.class}`;
    element.innerHTML = `<i class="${config.icon} me-1"></i>${config.text}`;
}

/**
 * Actualiza indicadores adicionales del sistema
 * @param {Object} data - Datos del sistema
 */
function updateAdditionalIndicators(data) {
    // Actualizar estado del servidor Flask
    const flaskElement = document.getElementById('flask-status');
    if (flaskElement) {
        flaskElement.className = 'badge bg-success';
        flaskElement.innerHTML = '<i class="fas fa-check me-1"></i>Activo';
    }
    
    // Actualizar estado del vector store
    const vectorElement = document.getElementById('vector-store-status');
    if (vectorElement) {
        vectorElement.className = 'badge bg-success';
        vectorElement.innerHTML = '<i class="fas fa-check me-1"></i>Disponible';
    }
    
    // Actualizar componentes adicionales si existen
    const components = data.components || {};
    Object.entries(components).forEach(([component, status]) => {
        const element = document.getElementById(`${component}-status`);
        if (element) {
            const isHealthy = status === 'available' || status === 'healthy';
            element.className = `badge ${isHealthy ? 'bg-success' : 'bg-warning'}`;
            element.innerHTML = `<i class="fas ${isHealthy ? 'fa-check' : 'fa-exclamation-triangle'} me-1"></i>${status}`;
        }
    });
}

/**
 * Actualiza el timestamp de la última verificación
 */
function updateLastCheckTimestamp() {
    const element = document.getElementById('last-status-check');
    if (element) {
        element.textContent = new Date().toLocaleTimeString('es-ES');
    }
}

/**
 * Maneja errores en la verificación del sistema
 * @param {Error} error - Error ocurrido
 */
function handleSystemCheckError(error) {
    console.error('🚨 Manejando error del sistema:', error);
    
    AppState.retryCount++;
    
    // Mostrar estado de error en la interfaz
    updateSystemInterface({
        status: 'error',
        services: {
            ollama: 'unavailable',
            openai: 'unavailable'
        },
        error: error.message
    });
    
    // Intentar reconexión si no se han agotado los intentos
    if (AppState.retryCount < CONFIG.RETRY_ATTEMPTS) {
        console.log(`🔄 Reintentando en ${CONFIG.RETRY_DELAY / 1000}s (intento ${AppState.retryCount}/${CONFIG.RETRY_ATTEMPTS})`);
        
        setTimeout(() => {
            performSystemCheck();
        }, CONFIG.RETRY_DELAY);
    } else {
        console.error('❌ Se agotaron los intentos de reconexión');
        showErrorNotification('No se puede conectar con el sistema. Verificar que la aplicación esté ejecutándose.');
    }
}

/**
 * Muestra una notificación de éxito
 * @param {string} message - Mensaje a mostrar
 */
function showSuccessNotification(message) {
    console.log('✅', message);
    // Aquí se podría integrar con un sistema de notificaciones
}

/**
 * Muestra una notificación de error
 * @param {string} message - Mensaje a mostrar
 */
function showErrorNotification(message) {
    console.error('❌', message);
    // Aquí se podría integrar con un sistema de notificaciones
}

// =============================================================================
// FUNCIONES DE NAVEGACIÓN Y UTILIDADES
// =============================================================================

/**
 * Navega al chat RAG
 */
function navigateToChat() {
    console.log('🗨️ Navegando al chat RAG...');
    window.location.href = '/chat';
}

/**
 * Navega al comparador de modelos
 */
function navigateToComparison() {
    console.log('⚖️ Navegando al comparador...');
    showComingSoon('Comparador de modelos');
}

/**
 * Navega al panel de administración
 */
function navigateToAdmin() {
    console.log('👨‍💼 Navegando al panel admin...');
    showComingSoon('Panel de administración');
}

/**
 * Navega a la configuración
 */
function navigateToConfig() {
    console.log('⚙️ Navegando a configuración...');
    showComingSoon('Configuración de fuentes de datos');
}

/**
 * Muestra mensaje de función próximamente disponible
 * @param {string} feature - Nombre de la función
 */
function showComingSoon(feature) {
    alert(`${feature} próximamente disponible\n\nEsta funcionalidad está en desarrollo y estará lista pronto.`);
}

// =============================================================================
// FUNCIONES DE DEBUG Y DESARROLLO
// =============================================================================

/**
 * Función de debug para verificación manual del health check
 */
function debugHealthCheck() {
    console.log('🔍 === DEBUG: Health Check Manual ===');
    
    fetch('/health')
        .then(response => {
            console.log('📡 Response status:', response.status);
            console.log('📡 Response headers:', Object.fromEntries(response.headers));
            return response.json();
        })
        .then(data => {
            console.log('📊 Respuesta completa:', data);
            console.log('📊 Status general:', data.status);
            console.log('📊 Servicios:', data.services);
            console.log('📊 Modelos:', data.models);
            console.log('📊 Componentes:', data.components);
            
            // Test manual de actualización
            console.log('🧪 Ejecutando actualización manual...');
            updateSystemInterface(data);
        })
        .catch(error => {
            console.error('❌ Error en debug:', error);
        });
}

/**
 * Función de debug para verificar elementos del DOM
 */
function debugDOMElements() {
    console.log('🔍 === DEBUG: Elementos DOM ===');
    
    const elements = [
        'local-models-status',
        'openai-status', 
        'status-indicator',
        'flask-status',
        'vector-store-status'
    ];
    
    elements.forEach(id => {
        const element = document.getElementById(id);
        console.log(`📍 ${id}:`, element ? '✅ Encontrado' : '❌ No encontrado', element);
    });
}

/**
 * Función de debug para verificar el estado de la aplicación
 */
function debugAppState() {
    console.log('🔍 === DEBUG: Estado de la Aplicación ===');
    console.log('📊 AppState:', AppState);
    console.log('📊 CONFIG:', CONFIG);
    console.log('📊 Tooltips activos:', AppState.tooltips.length);
    console.log('📊 Última verificación:', AppState.lastHealthCheck);
    console.log('📊 ¿Actualizando?:', AppState.isUpdating);
}

// =============================================================================
// EXPOSICIÓN GLOBAL PARA DEBUG
// =============================================================================

// Hacer funciones disponibles globalmente para debugging en consola
window.debugHealthCheck = debugHealthCheck;
window.debugDOMElements = debugDOMElements;
window.debugAppState = debugAppState;
window.checkSystemStatus = performSystemCheck;
window.AppState = AppState;

// Log de inicialización completa
console.log('🎉 main.js cargado completamente');
console.log('🔧 Funciones de debug disponibles: debugHealthCheck(), debugDOMElements(), debugAppState()');