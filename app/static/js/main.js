// JavaScript principal para la aplicación
document.addEventListener('DOMContentLoaded', function() {
    console.log('Prototipo_chatbot cargado correctamente');
    
    // Inicializar tooltips de Bootstrap
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Verificar estado del sistema cada minuto
    setInterval(checkSystemStatus, 60000);
    
    // Verificar estado inicial
    checkSystemStatus();
});

function checkSystemStatus() {
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            updateStatusIndicator(data.status);
        })
        .catch(error => {
            console.error('Error verificando estado:', error);
            updateStatusIndicator('error');
        });
}

function updateStatusIndicator(status) {
    const indicator = document.getElementById('status-indicator');
    if (!indicator) return;
    
    const statusClasses = {
        'healthy': 'bg-success',
        'warning': 'bg-warning',
        'error': 'bg-danger'
    };
    
    const statusTexts = {
        'healthy': 'Sistema Activo',
        'warning': 'Sistema con Alertas',
        'error': 'Sistema con Errores'
    };
    
    // Limpiar clases anteriores
    indicator.className = indicator.className.replace(/bg-\w+/g, '');
    
    // Aplicar nueva clase
    indicator.classList.add('badge', statusClasses[status] || 'bg-secondary');
    indicator.innerHTML = `<i class="fas fa-circle me-1"></i>${statusTexts[status] || 'Estado Desconocido'}`;
}

// Funciones para navegación
function navigateToChat() {
    alert('Chat RAG en desarrollo');
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

function navigateToVectorStore() {
    alert('Gestión de Vector Store en desarrollo');
}

function showHelp() {
    alert('Sistema de ayuda en desarrollo');
}

// Event listeners para navegación
document.addEventListener('DOMContentLoaded', function() {
    const chatLink = document.getElementById('chat-link');
    const comparisonLink = document.getElementById('comparison-link');
    const adminLink = document.getElementById('admin-link');
    const configLink = document.getElementById('config-link');
    const vectorstoreLink = document.getElementById('vectorstore-link');
    const helpLink = document.getElementById('help-link');
    
    if (chatLink) chatLink.addEventListener('click', navigateToChat);
    if (comparisonLink) comparisonLink.addEventListener('click', navigateToComparison);
    if (adminLink) adminLink.addEventListener('click', navigateToAdmin);
    if (configLink) configLink.addEventListener('click', navigateToConfig);
    if (vectorstoreLink) vectorstoreLink.addEventListener('click', navigateToVectorStore);
    if (helpLink) helpLink.addEventListener('click', showHelp);
});