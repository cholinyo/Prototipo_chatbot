# -*- coding: utf-8 -*-
"""
API Routes para LLM Service - Basico
TFM Vicente Caruncho
"""

from flask import Blueprint, jsonify

# Crear blueprint
llm_bp = Blueprint('llm_api', __name__, url_prefix='/api/llm')

@llm_bp.route('/health', methods=['GET'])
def health_check():
    """Health check del servicio LLM"""
    try:
        from app.services.llm import llm_service
        health = llm_service.health_check()
        status_code = 200 if health['status'] == 'healthy' else 503
        return jsonify(health), status_code
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@llm_bp.route('/status', methods=['GET'])
def service_status():
    """Estado del servicio"""
    return jsonify({
        'service': 'LLM Service',
        'version': '1.0.0',
        'status': 'active'
    })