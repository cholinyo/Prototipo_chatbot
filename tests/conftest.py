"""
Configuración pytest para Prototipo_chatbot
"""
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def app():
    from app import create_app
    
    test_config = {
        'TESTING': True,
        'DEBUG': False,
        'SECRET_KEY': 'test-secret-key'
    }
    
    app = create_app(test_config)
    with app.app_context():
        yield app

@pytest.fixture
def client(app):
    return app.test_client()
