# SISTEMA COMPLETO DE GESTIÓN DE MODELOS
# Ollama + OpenAI + Fine-tuning + Estadísticas

"""
ARCHIVO 1: app/services/models/ollama_manager.py
Gestión completa de modelos Ollama locales
"""

import requests
import subprocess
import json
import time
import psutil
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from app.core.logger import get_logger

@dataclass
class ModelStats:
    """Estadísticas de rendimiento del modelo"""
    response_time: float
    cpu_usage: float
    memory_usage: float
    tokens_per_second: float
    total_requests: int
    error_rate: float

@dataclass
class OllamaModel:
    """Información del modelo Ollama"""
    name: str
    size: str
    modified: str
    digest: str
    family: str
    format: str
    parameters: Dict[str, Any]

class OllamaManager:
    """Gestor completo de modelos Ollama"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.logger = get_logger("ollama_manager")
        self.model_stats = {}
        
    # ================================
    # GESTIÓN DE MODELOS
    # ================================
    
    def get_available_models(self) -> List[OllamaModel]:
        """Obtener lista de modelos disponibles"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            models_data = response.json().get('models', [])
            models = []
            
            for model_data in models_data:
                model = OllamaModel(
                    name=model_data.get('name', ''),
                    size=model_data.get('size', ''),
                    modified=model_data.get('modified_at', ''),
                    digest=model_data.get('digest', ''),
                    family=model_data.get('details', {}).get('family', ''),
                    format=model_data.get('details', {}).get('format', ''),
                    parameters=model_data.get('details', {}).get('parameters', {})
                )
                models.append(model)
                
            self.logger.info("Modelos Ollama obtenidos", count=len(models))
            return models
            
        except Exception as e:
            self.logger.error("Error obteniendo modelos Ollama", error=str(e))
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Descargar un modelo"""
        try:
            self.logger.info("Iniciando descarga de modelo", model=model_name)
            
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if data.get('status'):
                        self.logger.info("Progreso descarga", 
                                       model=model_name,
                                       status=data['status'])
                        
            return True
            
        except Exception as e:
            self.logger.error("Error descargando modelo", 
                            model=model_name, error=str(e))
            return False
    
    def delete_model(self, model_name: str) -> bool:
        """Eliminar un modelo"""
        try:
            response = requests.delete(f"{self.base_url}/api/delete", 
                                     json={"name": model_name})
            response.raise_for_status()
            
            self.logger.info("Modelo eliminado", model=model_name)
            return True
            
        except Exception as e:
            self.logger.error("Error eliminando modelo", 
                            model=model_name, error=str(e))
            return False
    
    # ================================
    # FINE-TUNING Y CONFIGURACIÓN
    # ================================
    
    def create_modelfile(self, base_model: str, parameters: Dict[str, Any], 
                        system_prompt: str = "") -> str:
        """Crear Modelfile para fine-tuning"""
        modelfile_content = f"""FROM {base_model}

# Parámetros del modelo
PARAMETER temperature {parameters.get('temperature', 0.7)}
PARAMETER top_k {parameters.get('top_k', 40)}
PARAMETER top_p {parameters.get('top_p', 0.9)}
PARAMETER repeat_penalty {parameters.get('repeat_penalty', 1.1)}
PARAMETER num_ctx {parameters.get('num_ctx', 2048)}
PARAMETER num_predict {parameters.get('num_predict', 128)}

# System prompt personalizado
SYSTEM \"\"\"
{system_prompt or 'Eres un asistente especializado en administración pública local.'}
\"\"\"

# Template personalizado
TEMPLATE \"\"\"{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
\"\"\"
"""
        return modelfile_content
    
    def fine_tune_model(self, base_model: str, new_model_name: str,
                       parameters: Dict[str, Any], system_prompt: str = "") -> bool:
        """Realizar fine-tuning de un modelo"""
        try:
            # Crear Modelfile
            modelfile = self.create_modelfile(base_model, parameters, system_prompt)
            
            # Crear modelo personalizado
            response = requests.post(
                f"{self.base_url}/api/create",
                json={
                    "name": new_model_name,
                    "modelfile": modelfile
                },
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if data.get('status'):
                        self.logger.info("Progreso fine-tuning",
                                       model=new_model_name,
                                       status=data['status'])
            
            self.logger.info("Fine-tuning completado", 
                           base_model=base_model,
                           new_model=new_model_name)
            return True
            
        except Exception as e:
            self.logger.error("Error en fine-tuning", 
                            model=new_model_name, error=str(e))
            return False
    
    # ================================
    # PRUEBAS DE CONEXIÓN
    # ================================
    
    def test_connection(self) -> Dict[str, Any]:
        """Probar conexión con Ollama"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                models_count = len(response.json().get('models', []))
                return {
                    'status': 'connected',
                    'response_time': response_time,
                    'models_available': models_count,
                    'ollama_version': self._get_ollama_version()
                }
            else:
                return {
                    'status': 'error',
                    'error': f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                'status': 'disconnected',
                'error': str(e)
            }
    
    def test_model_inference(self, model_name: str, 
                           test_prompt: str = "Hola, ¿cómo estás?") -> Dict[str, Any]:
        """Probar inferencia de un modelo específico"""
        try:
            start_time = time.time()
            cpu_before = psutil.cpu_percent()
            memory_before = psutil.virtual_memory().percent
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": test_prompt,
                    "stream": False
                }
            )
            
            response_time = time.time() - start_time
            cpu_after = psutil.cpu_percent()
            memory_after = psutil.virtual_memory().percent
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'status': 'success',
                    'response_time': response_time,
                    'response_length': len(data.get('response', '')),
                    'cpu_usage': cpu_after - cpu_before,
                    'memory_usage': memory_after - memory_before,
                    'tokens_evaluated': data.get('eval_count', 0),
                    'tokens_per_second': data.get('eval_count', 0) / response_time if response_time > 0 else 0
                }
            else:
                return {
                    'status': 'error',
                    'error': f"HTTP {response.status_code}",
                    'response_time': response_time
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'response_time': time.time() - start_time if 'start_time' in locals() else 0
            }
    
    # ================================
    # ESTADÍSTICAS Y MONITOREO
    # ================================
    
    def get_model_stats(self, model_name: str) -> ModelStats:
        """Obtener estadísticas de un modelo"""
        return self.model_stats.get(model_name, ModelStats(
            response_time=0.0,
            cpu_usage=0.0,
            memory_usage=0.0,
            tokens_per_second=0.0,
            total_requests=0,
            error_rate=0.0
        ))
    
    def update_model_stats(self, model_name: str, response_time: float,
                          cpu_usage: float, memory_usage: float,
                          tokens_per_second: float, success: bool):
        """Actualizar estadísticas de un modelo"""
        if model_name not in self.model_stats:
            self.model_stats[model_name] = ModelStats(
                response_time=0.0, cpu_usage=0.0, memory_usage=0.0,
                tokens_per_second=0.0, total_requests=0, error_rate=0.0
            )
        
        stats = self.model_stats[model_name]
        stats.total_requests += 1
        
        # Promedio móvil para métricas
        alpha = 0.1  # Factor de suavizado
        stats.response_time = (1 - alpha) * stats.response_time + alpha * response_time
        stats.cpu_usage = (1 - alpha) * stats.cpu_usage + alpha * cpu_usage
        stats.memory_usage = (1 - alpha) * stats.memory_usage + alpha * memory_usage
        stats.tokens_per_second = (1 - alpha) * stats.tokens_per_second + alpha * tokens_per_second
        
        if not success:
            stats.error_rate = (stats.error_rate * (stats.total_requests - 1) + 1) / stats.total_requests
        else:
            stats.error_rate = (stats.error_rate * (stats.total_requests - 1)) / stats.total_requests
    
    def _get_ollama_version(self) -> str:
        """Obtener versión de Ollama"""
        try:
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        except:
            return "Unknown"


"""
ARCHIVO 2: app/services/models/openai_manager.py
Gestión de modelos OpenAI
"""

import openai
from typing import Dict, List, Any
from app.core.config import get_openai_api_key

class OpenAIManager:
    """Gestor de modelos OpenAI"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=get_openai_api_key())
        self.logger = get_logger("openai_manager")
        self.model_stats = {}
        
        # Modelos disponibles via API
        self.available_models = {
            'gpt-4': {
                'name': 'GPT-4',
                'description': 'Modelo más capaz, mejor para tareas complejas',
                'max_tokens': 8192,
                'cost_per_1k_tokens': 0.03
            },
            'gpt-4-turbo': {
                'name': 'GPT-4 Turbo',
                'description': 'Más rápido y económico que GPT-4',
                'max_tokens': 128000,
                'cost_per_1k_tokens': 0.01
            },
            'gpt-3.5-turbo': {
                'name': 'GPT-3.5 Turbo',
                'description': 'Rápido y económico para la mayoría de tareas',
                'max_tokens': 16385,
                'cost_per_1k_tokens': 0.001
            },
            'gpt-4o': {
                'name': 'GPT-4o',
                'description': 'Optimizado para velocidad y eficiencia',
                'max_tokens': 128000,
                'cost_per_1k_tokens': 0.005
            }
        }
    
    def get_available_models(self) -> Dict[str, Any]:
        """Obtener modelos OpenAI disponibles"""
        return self.available_models
    
    def test_connection(self) -> Dict[str, Any]:
        """Probar conexión con OpenAI"""
        try:
            start_time = time.time()
            
            # Hacer una llamada simple para probar la conexión
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1
            )
            
            response_time = time.time() - start_time
            
            return {
                'status': 'connected',
                'response_time': response_time,
                'api_key_valid': True,
                'models_available': len(self.available_models)
            }
            
        except openai.AuthenticationError:
            return {
                'status': 'error',
                'error': 'API key inválida o no configurada'
            }
        except openai.RateLimitError:
            return {
                'status': 'error',
                'error': 'Límite de rate excedido'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def test_model_inference(self, model_name: str, 
                           test_prompt: str = "Hola, ¿cómo estás?") -> Dict[str, Any]:
        """Probar inferencia de un modelo OpenAI"""
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": test_prompt}],
                max_tokens=50
            )
            
            response_time = time.time() - start_time
            
            return {
                'status': 'success',
                'response_time': response_time,
                'tokens_used': response.usage.total_tokens,
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'estimated_cost': self._calculate_cost(model_name, response.usage.total_tokens),
                'response_length': len(response.choices[0].message.content)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'response_time': time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def _calculate_cost(self, model_name: str, tokens: int) -> float:
        """Calcular costo estimado"""
        model_info = self.available_models.get(model_name, {})
        cost_per_1k = model_info.get('cost_per_1k_tokens', 0.001)
        return (tokens / 1000) * cost_per_1k


"""
ARCHIVO 3: app/routes/models.py
Rutas Flask para gestión de modelos
"""

from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from app.services.models.ollama_manager import OllamaManager
from app.services.models.openai_manager import OpenAIManager

models_bp = Blueprint('models', __name__, url_prefix='/models')

ollama_manager = OllamaManager()
openai_manager = OpenAIManager()

@models_bp.route('/')
def models_dashboard():
    """Dashboard principal de modelos"""
    return render_template('models/dashboard.html')

# ================================
# RUTAS OLLAMA
# ================================

@models_bp.route('/ollama')
def ollama_models():
    """Página de gestión de modelos Ollama"""
    models = ollama_manager.get_available_models()
    connection_status = ollama_manager.test_connection()
    
    return render_template('models/ollama.html', 
                         models=models,
                         connection_status=connection_status)

@models_bp.route('/api/ollama/models', methods=['GET'])
def api_get_ollama_models():
    """API: Obtener modelos Ollama"""
    models = ollama_manager.get_available_models()
    return jsonify([{
        'name': m.name,
        'size': m.size,
        'family': m.family,
        'modified': m.modified
    } for m in models])

@models_bp.route('/api/ollama/pull', methods=['POST'])
def api_pull_model():
    """API: Descargar modelo"""
    data = request.get_json()
    model_name = data.get('model_name')
    
    if not model_name:
        return jsonify({'error': 'Nombre de modelo requerido'}), 400
    
    success = ollama_manager.pull_model(model_name)
    
    if success:
        return jsonify({'message': f'Modelo {model_name} descargado correctamente'})
    else:
        return jsonify({'error': 'Error descargando modelo'}), 500

@models_bp.route('/api/ollama/delete', methods=['DELETE'])
def api_delete_model():
    """API: Eliminar modelo"""
    data = request.get_json()
    model_name = data.get('model_name')
    
    success = ollama_manager.delete_model(model_name)
    
    if success:
        return jsonify({'message': f'Modelo {model_name} eliminado correctamente'})
    else:
        return jsonify({'error': 'Error eliminando modelo'}), 500

@models_bp.route('/api/ollama/fine-tune', methods=['POST'])
def api_fine_tune_model():
    """API: Fine-tuning de modelo"""
    data = request.get_json()
    
    base_model = data.get('base_model')
    new_model_name = data.get('new_model_name')
    parameters = data.get('parameters', {})
    system_prompt = data.get('system_prompt', '')
    
    if not base_model or not new_model_name:
        return jsonify({'error': 'Modelo base y nombre nuevo requeridos'}), 400
    
    success = ollama_manager.fine_tune_model(
        base_model, new_model_name, parameters, system_prompt
    )
    
    if success:
        return jsonify({'message': f'Fine-tuning completado: {new_model_name}'})
    else:
        return jsonify({'error': 'Error en fine-tuning'}), 500

@models_bp.route('/api/ollama/test/<model_name>', methods=['POST'])
def api_test_ollama_model(model_name):
    """API: Probar modelo Ollama"""
    data = request.get_json()
    test_prompt = data.get('prompt', 'Hola, ¿cómo estás?')
    
    result = ollama_manager.test_model_inference(model_name, test_prompt)
    return jsonify(result)

@models_bp.route('/api/ollama/stats/<model_name>', methods=['GET'])
def api_get_ollama_stats(model_name):
    """API: Obtener estadísticas de modelo"""
    stats = ollama_manager.get_model_stats(model_name)
    return jsonify({
        'response_time': stats.response_time,
        'cpu_usage': stats.cpu_usage,
        'memory_usage': stats.memory_usage,
        'tokens_per_second': stats.tokens_per_second,
        'total_requests': stats.total_requests,
        'error_rate': stats.error_rate
    })

# ================================
# RUTAS OPENAI
# ================================

@models_bp.route('/openai')
def openai_models():
    """Página de gestión de modelos OpenAI"""
    models = openai_manager.get_available_models()
    connection_status = openai_manager.test_connection()
    
    return render_template('models/openai.html',
                         models=models,
                         connection_status=connection_status)

@models_bp.route('/api/openai/models', methods=['GET'])
def api_get_openai_models():
    """API: Obtener modelos OpenAI"""
    models = openai_manager.get_available_models()
    return jsonify(models)

@models_bp.route('/api/openai/test/<model_name>', methods=['POST'])
def api_test_openai_model(model_name):
    """API: Probar modelo OpenAI"""
    data = request.get_json()
    test_prompt = data.get('prompt', 'Hola, ¿cómo estás?')
    
    result = openai_manager.test_model_inference(model_name, test_prompt)
    return jsonify(result)

@models_bp.route('/api/connection/test', methods=['GET'])
def api_test_connections():
    """API: Probar todas las conexiones"""
    return jsonify({
        'ollama': ollama_manager.test_connection(),
        'openai': openai_manager.test_connection()
    })


"""
ARCHIVO 4: app/templates/models/dashboard.html
Template principal del dashboard de modelos
"""

# Este contenido se creará en el siguiente artifact debido a limitaciones de espacio