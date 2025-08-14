/**
 * Frontend JavaScript para Chat RAG
 * TFM Vicente Caruncho - Administraciones Locales
 */

class ChatRAGInterface {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.isProcessing = false;
        this.currentProvider = 'ollama';
        this.messageHistory = [];
        this.currentSettings = {
            temperature: 0.3,
            maxTokens: 500,
            topK: 5
        };
        
        this.initializeElements();
        this.attachEventListeners();
        this.checkSystemStatus();
        this.loadChatHistory();
    }
    
    generateSessionId() {
        return 'chat_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    initializeElements() {
        // Elementos principales
        this.chatContainer = document.getElementById('chat-container');
        this.messageInput = document.getElementById('message-input');
        this.sendButton = document.getElementById('send-button');
        this.compareButton = document.getElementById('compare-button');
        this.providerSelect = document.getElementById('provider-select');
        this.settingsModal = document.getElementById('settings-modal');
        this.statusIndicator = document.getElementById('status-indicator');
        this.typingIndicator = document.getElementById('typing-indicator');
        
        // Elementos de configuración
        this.temperatureSlider = document.getElementById('temperature-slider');
        this.temperatureValue = document.getElementById('temperature-value');
        this.maxTokensInput = document.getElementById('max-tokens-input');
        this.topKInput = document.getElementById('top-k-input');
        
        // Elementos de estadísticas
        this.statsPanel = document.getElementById('stats-panel');
        this.responseTimeChart = document.getElementById('response-time-chart');
        
        // Verificar que elementos críticos existen
        if (!this.chatContainer || !this.messageInput || !this.sendButton) {
            console.error('Elementos críticos del chat no encontrados');
            this.showError('Error de interfaz: elementos no encontrados');
            return;
        }
        
        console.log('Chat RAG Interface inicializada', {
            sessionId: this.sessionId,
            provider: this.currentProvider
        });
    }
    
    attachEventListeners() {
        // Evento enviar mensaje
        this.sendButton.addEventListener('click', () => this.sendMessage());
        
        // Enviar con Enter (Shift+Enter para nueva línea)
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Cambio de proveedor
        if (this.providerSelect) {
            this.providerSelect.addEventListener('change', (e) => {
                this.currentProvider = e.target.value;
                this.updateProviderStatus();
            });
        }
        
        // Botón comparar proveedores
        if (this.compareButton) {
            this.compareButton.addEventListener('click', () => this.compareProviders());
        }
        
        // Configuración de temperatura
        if (this.temperatureSlider) {
            this.temperatureSlider.addEventListener('input', (e) => {
                this.currentSettings.temperature = parseFloat(e.target.value);
                if (this.temperatureValue) {
                    this.temperatureValue.textContent = e.target.value;
                }
            });
        }
        
        // Auto-resize del textarea
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = this.messageInput.scrollHeight + 'px';
        });
        
        // Limpiar chat
        const clearButton = document.getElementById('clear-chat');
        if (clearButton) {
            clearButton.addEventListener('click', () => this.clearChat());
        }
        
        // Exportar conversación
        const exportButton = document.getElementById('export-chat');
        if (exportButton) {
            exportButton.addEventListener('click', () => this.exportConversation());
        }
    }
    
    async checkSystemStatus() {
        try {
            const response = await fetch('/api/chat/status');
            const data = await response.json();
            
            if (data.success) {
                this.updateStatusIndicator(data.status, data.components);
                this.updateStatsPanel(data.pipeline_stats);
            } else {
                this.updateStatusIndicator('error', {});
            }
        } catch (error) {
            console.error('Error verificando estado del sistema:', error);
            this.updateStatusIndicator('error', {});
        }
    }
    
    updateStatusIndicator(status, components) {
        if (!this.statusIndicator) return;
        
        const statusConfig = {
            'healthy': { color: 'success', text: 'Sistema Operativo', icon: '🟢' },
            'degraded': { color: 'warning', text: 'Funcionalidad Limitada', icon: '🟡' },
            'error': { color: 'danger', text: 'Sistema No Disponible', icon: '🔴' }
        };
        
        const config = statusConfig[status] || statusConfig.error;
        
        this.statusIndicator.className = `badge bg-${config.color}`;
        this.statusIndicator.innerHTML = `${config.icon} ${config.text}`;
        
        // Tooltip con detalles de componentes
        if (components) {
            const details = Object.entries(components)
                .map(([comp, status]) => `${comp}: ${status}`)
                .join('\n');
            this.statusIndicator.title = details;
        }
    }
    
    updateStatsPanel(stats) {
        if (!this.statsPanel || !stats) return;
        
        const statsHtml = `
            <div class="row">
                <div class="col-md-4">
                    <small class="text-muted">Sesión Activa</small>
                    <div class="fw-bold">${this.sessionId.split('_')[1]}</div>
                </div>
                <div class="col-md-4">
                    <small class="text-muted">Proveedor LLM</small>
                    <div class="fw-bold">${this.currentProvider.toUpperCase()}</div>
                </div>
                <div class="col-md-4">
                    <small class="text-muted">Mensajes</small>
                    <div class="fw-bold">${this.messageHistory.length}</div>
                </div>
            </div>
            <div class="row mt-2">
                <div class="col-md-6">
                    <small class="text-muted">Vector Store</small>
                    <div class="fw-bold">${stats.vector_store?.type || 'N/A'}</div>
                </div>
                <div class="col-md-6">
                    <small class="text-muted">Documentos</small>
                    <div class="fw-bold">${stats.vector_store?.documents || 0}</div>
                </div>
            </div>
        `;
        
        this.statsPanel.innerHTML = statsHtml;
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        
        if (!message) {
            this.showWarning('Por favor, escriba un mensaje');
            return;
        }
        
        if (this.isProcessing) {
            this.showWarning('Esperando respuesta anterior...');
            return;
        }
        
        try {
            // Deshabilitar entrada
            this.setProcessingState(true);
            
            // Añadir mensaje del usuario al chat
            this.addMessage('user', message);
            
            // Mostrar indicador de escritura
            this.showTypingIndicator();
            
            // Enviar solicitud al backend
            const response = await fetch('/api/chat/send', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    provider: this.currentProvider,
                    session_id: this.sessionId,
                    temperature: this.currentSettings.temperature,
                    max_tokens: this.currentSettings.maxTokens
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Añadir respuesta del asistente
                this.addMessage('assistant', data.response, data.metadata);
                this.updateResponseMetrics(data.metadata);
            } else {
                this.addMessage('error', data.error || 'Error desconocido');
                this.showError(data.error || 'Error procesando mensaje');
            }
            
        } catch (error) {
            console.error('Error enviando mensaje:', error);
            this.addMessage('error', 'Error de conexión con el servidor');
            this.showError('Error de conexión');
        } finally {
            this.hideTypingIndicator();
            this.setProcessingState(false);
            this.messageInput.value = '';
            this.messageInput.style.height = 'auto';
        }
    }
    
    async compareProviders() {
        const message = this.messageInput.value.trim();
        
        if (!message) {
            this.showWarning('Escriba un mensaje para comparar');
            return;
        }
        
        if (this.isProcessing) {
            this.showWarning('Esperando respuesta anterior...');
            return;
        }
        
        try {
            this.setProcessingState(true);
            this.addMessage('user', message);
            this.showTypingIndicator();
            
            const response = await fetch('/api/chat/compare', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    session_id: this.sessionId,
                    temperature: this.currentSettings.temperature
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.addComparisonMessage(data);
            } else {
                this.showError(data.error || 'Error en comparación');
            }
            
        } catch (error) {
            console.error('Error comparando proveedores:', error);
            this.showError('Error de conexión en comparación');
        } finally {
            this.hideTypingIndicator();
            this.setProcessingState(false);
            this.messageInput.value = '';
        }
    }
    
    addMessage(type, content, metadata = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const timestamp = new Date().toLocaleTimeString('es-ES');
        
        let messageHtml = '';
        
        switch (type) {
            case 'user':
                messageHtml = `
                    <div class="message-header">
                        <span class="message-sender">👤 Usuario</span>
                        <span class="message-time">${timestamp}</span>
                    </div>
                    <div class="message-content user-message">
                        ${this.formatMessage(content)}
                    </div>
                `;
                break;
                
            case 'assistant':
                const sourcesHtml = metadata?.sources?.length > 0 ? 
                    `<div class="message-sources">
                        <small class="text-muted">📚 Fuentes: ${metadata.sources.slice(0, 3).join(', ')}</small>
                    </div>` : '';
                
                const metricsHtml = metadata ? 
                    `<div class="message-metrics">
                        <small class="text-muted">
                            🤖 ${metadata.model} | 
                            ⏱️ ${metadata.response_time}s | 
                            📊 ${Math.round(metadata.confidence * 100)}% confianza
                            ${metadata.estimated_cost ? ` | 💰 ${metadata.estimated_cost.toFixed(4)}` : ''}
                        </small>
                    </div>` : '';
                
                messageHtml = `
                    <div class="message-header">
                        <span class="message-sender">🤖 Asistente IA</span>
                        <span class="message-time">${timestamp}</span>
                    </div>
                    <div class="message-content assistant-message">
                        ${this.formatMessage(content)}
                    </div>
                    ${sourcesHtml}
                    ${metricsHtml}
                `;
                break;
                
            case 'error':
                messageHtml = `
                    <div class="message-header">
                        <span class="message-sender">❌ Error</span>
                        <span class="message-time">${timestamp}</span>
                    </div>
                    <div class="message-content error-message">
                        ${content}
                    </div>
                `;
                break;
        }
        
        messageDiv.innerHTML = messageHtml;
        this.chatContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Guardar en historial
        this.messageHistory.push({
            type,
            content,
            metadata,
            timestamp: new Date().toISOString()
        });
    }
    
    addComparisonMessage(data) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message comparison';
        
        const timestamp = new Date().toLocaleTimeString('es-ES');
        const providers = Object.keys(data.results);
        
        let comparisonHtml = `
            <div class="message-header">
                <span class="message-sender">⚖️ Comparación de Proveedores</span>
                <span class="message-time">${timestamp}</span>
            </div>
            <div class="comparison-content">
        `;
        
        // Resumen de comparación
        if (data.summary) {
            comparisonHtml += `
                <div class="comparison-summary mb-3">
                    <h6>📊 Resumen de Comparación</h6>
                    <div class="row">
                        <div class="col-md-4">
                            <small>🏃 Más Rápido:</small><br>
                            <strong>${data.summary.fastest_provider || 'N/A'}</strong>
                            <small class="text-muted">(${data.summary.fastest_time?.toFixed(2)}s)</small>
                        </div>
                        <div class="col-md-4">
                            <small>🎯 Mayor Confianza:</small><br>
                            <strong>${data.summary.most_confident || 'N/A'}</strong>
                            <small class="text-muted">(${Math.round((data.summary.highest_confidence || 0) * 100)}%)</small>
                        </div>
                        <div class="col-md-4">
                            <small>💰 Más Económico:</small><br>
                            <strong>${data.summary.cheapest_provider || 'N/A'}</strong>
                            ${data.summary.lowest_cost ? `<small class="text-muted">(${data.summary.lowest_cost.toFixed(4)})</small>` : ''}
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Respuestas individuales
        comparisonHtml += '<div class="provider-responses">';
        
        for (const [provider, result] of Object.entries(data.results)) {
            const statusBadge = result.success ? 
                '<span class="badge bg-success">✅ Éxito</span>' : 
                '<span class="badge bg-danger">❌ Error</span>';
            
            comparisonHtml += `
                <div class="provider-result mb-3">
                    <div class="provider-header d-flex justify-content-between align-items-center">
                        <h6 class="mb-0">🤖 ${provider.toUpperCase()}</h6>
                        ${statusBadge}
                    </div>
                    <div class="provider-content">
                        ${result.success ? 
                            `<div class="response-text">${this.formatMessage(result.response)}</div>
                             <div class="response-metrics mt-2">
                                <small class="text-muted">
                                    ⏱️ ${result.response_time}s | 
                                    📊 ${Math.round(result.confidence * 100)}% | 
                                    🤖 ${result.model}
                                    ${result.estimated_cost ? ` | 💰 ${result.estimated_cost.toFixed(4)}` : ''}
                                </small>
                             </div>` :
                            `<div class="error-text text-danger">${result.error}</div>`
                        }
                    </div>
                </div>
            `;
        }
        
        comparisonHtml += '</div></div>';
        
        messageDiv.innerHTML = comparisonHtml;
        this.chatContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Guardar en historial
        this.messageHistory.push({
            type: 'comparison',
            content: data,
            timestamp: new Date().toISOString()
        });
    }
    
    formatMessage(text) {
        // Formateo básico de texto con markdown simple
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }
    
    setProcessingState(processing) {
        this.isProcessing = processing;
        this.sendButton.disabled = processing;
        this.compareButton.disabled = processing;
        this.messageInput.disabled = processing;
        
        if (processing) {
            this.sendButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Enviando...';
            this.compareButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Comparando...';
        } else {
            this.sendButton.innerHTML = '📤 Enviar';
            this.compareButton.innerHTML = '⚖️ Comparar';
        }
    }
    
    showTypingIndicator() {
        if (this.typingIndicator) {
            this.typingIndicator.style.display = 'block';
            this.scrollToBottom();
        }
    }
    
    hideTypingIndicator() {
        if (this.typingIndicator) {
            this.typingIndicator.style.display = 'none';
        }
    }
    
    scrollToBottom() {
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }
    
    showError(message) {
        this.showToast('error', '❌ Error', message);
    }
    
    showWarning(message) {
        this.showToast('warning', '⚠️ Aviso', message);
    }
    
    showSuccess(message) {
        this.showToast('success', '✅ Éxito', message);
    }
    
    showToast(type, title, message) {
        // Crear toast dinámicamente
        const toastHtml = `
            <div class="toast align-items-center text-white bg-${type}" role="alert">
                <div class="d-flex">
                    <div class="toast-body">
                        <strong>${title}</strong><br>${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;
        
        let toastContainer = document.getElementById('toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.id = 'toast-container';
            toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
            document.body.appendChild(toastContainer);
        }
        
        toastContainer.insertAdjacentHTML('beforeend', toastHtml);
        const toastElement = toastContainer.lastElementChild;
        const toast = new bootstrap.Toast(toastElement);
        toast.show();
        
        // Limpiar toast después de mostrarlo
        toastElement.addEventListener('hidden.bs.toast', () => {
            toastElement.remove();
        });
    }
    
    updateResponseMetrics(metadata) {
        // Actualizar gráfico de tiempos de respuesta si existe
        if (this.responseTimeChart && metadata.response_time) {
            // Implementar actualización de Chart.js aquí
            console.log('Actualizando métricas:', metadata);
        }
    }
    
    async loadChatHistory() {
        try {
            const response = await fetch(`/api/chat/history/${this.sessionId}`);
            if (response.ok) {
                const data = await response.json();
                if (data.success && data.messages) {
                    // Cargar mensajes previos
                    data.messages.forEach(msg => {
                        if (msg.type === 'comparison') {
                            this.addComparisonMessage(msg.content);
                        } else {
                            this.addMessage(msg.type, msg.content, msg.metadata || null);
                        }
                    });
                }
            }
        } catch (error) {
            console.log('No hay historial previo para esta sesión');
        }
    }
    
    async clearChat() {
        if (confirm('¿Está seguro de que desea limpiar toda la conversación?')) {
            try {
                await fetch(`/api/chat/clear/${this.sessionId}`, { method: 'DELETE' });
                this.chatContainer.innerHTML = '';
                this.messageHistory = [];
                this.showSuccess('Conversación limpiada');
            } catch (error) {
                console.error('Error limpiando chat:', error);
                this.showError('Error limpiando conversación');
            }
        }
    }
    
    exportConversation() {
        const conversation = this.messageHistory.map(msg => ({
            timestamp: msg.timestamp,
            type: msg.type,
            content: msg.content,
            metadata: msg.metadata
        }));
        
        const dataStr = JSON.stringify(conversation, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `chat_rag_${this.sessionId}_${new Date().toISOString().split('T')[0]}.json`;
        link.click();
        
        this.showSuccess('Conversación exportada');
    }
}

// Inicializar interfaz cuando se carga la página
document.addEventListener('DOMContentLoaded', () => {
    window.chatRAG = new ChatRAGInterface();
    
    // Verificar estado del sistema cada 30 segundos
    setInterval(() => {
        window.chatRAG.checkSystemStatus();
    }, 30000);
});

// Estilos CSS adicionales para el chat
const chatStyles = `
<style>
.message {
    margin-bottom: 1rem;
    padding: 0.75rem;
    border-radius: 0.5rem;
    border-left: 4px solid #007bff;
}

.message.user {
    background-color: #e3f2fd;
    border-left-color: #2196f3;
    margin-left: 2rem;
}

.message.assistant {
    background-color: #f8f9fa;
    border-left-color: #28a745;
    margin-right: 2rem;
}

.message.error {
    background-color: #fff5f5;
    border-left-color: #dc3545;
}

.message.comparison {
    background-color: #fff8e1;
    border-left-color: #ff9800;
}

.message-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
    font-size: 0.875rem;
}

.message-sender {
    font-weight: 600;
    color: #495057;
}

.message-time {
    color: #6c757d;
    font-size: 0.75rem;
}

.message-content {
    line-height: 1.5;
}

.message-sources, .message-metrics {
    margin-top: 0.5rem;
    padding-top: 0.5rem;
    border-top: 1px solid #e9ecef;
}

.comparison-summary {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.375rem;
    border: 1px solid #dee2e6;
}

.provider-result {
    border: 1px solid #dee2e6;
    border-radius: 0.375rem;
    padding: 0.75rem;
    background-color: white;
}

.provider-header {
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e9ecef;
    margin-bottom: 0.5rem;
}

.response-text {
    margin-bottom: 0.5rem;
}

.error-text {
    font-style: italic;
}

#typing-indicator {
    padding: 0.75rem;
    background-color: #f8f9fa;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: none;
}

.typing-dots {
    display: inline-block;
}

.typing-dots::after {
    content: '...';
    animation: typing 1.5s infinite;
}

@keyframes typing {
    0%, 60% { content: '...'; }
    30% { content: '..'; }
    90% { content: '.'; }
}

#chat-container {
    max-height: 60vh;
    overflow-y: auto;
    padding: 1rem;
    border: 1px solid #dee2e6;
    border-radius: 0.5rem;
    background-color: white;
}

.toast-container {
    z-index: 1050;
}
</style>
`;

// Inyectar estilos en el head
document.head.insertAdjacentHTML('beforeend', chatStyles);