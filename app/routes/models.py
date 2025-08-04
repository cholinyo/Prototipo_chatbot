from flask import Blueprint, render_template, jsonify, request

# Crear blueprint
models_bp = Blueprint('models', __name__)

@models_bp.route('/')
@models_bp.route('/dashboard')
def models_dashboard():
    """Dashboard principal de modelos IA"""
    return render_template('models/dashboard.html',
                         app_name='Prototipo_chatbot',
                         page_title='Dashboard de Modelos IA',
                         breadcrumb='Modelos > Dashboard')

@models_bp.route('/status')
def models_status():
    """Estado de los modelos"""
    return jsonify({
        'ollama': 'checking',
        'models_loaded': [],
        'models_available': ['llama2', 'mistral', 'codellama'],
        'status': 'development',
        'last_update': '2025-08-02'
    })

@models_bp.route('/list')
def models_list():
    """Lista de modelos disponibles"""
    return jsonify({
        'models': [
            {'name': 'llama2', 'size': '7B', 'status': 'available'},
            {'name': 'mistral', 'size': '7B', 'status': 'downloading'},
            {'name': 'codellama', 'size': '13B', 'status': 'not_downloaded'}
        ]
    })

@models_bp.route('/download', methods=['POST'])
def download_model():
    """Descargar un modelo específico"""
    model_name = request.json.get('model_name') if request.json else None
    
    if not model_name:
        return jsonify({'error': 'model_name requerido'}), 400
    
    return jsonify({
        'message': f'Descarga iniciada para {model_name}',
        'status': 'downloading',
        'model': model_name
    })