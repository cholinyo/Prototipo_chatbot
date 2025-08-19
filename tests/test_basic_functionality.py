"""
Tests básicos de funcionalidad
"""
def test_app_creation(app):
    assert app is not None

def test_index_route(client):
    response = client.get('/')
    assert response.status_code == 200

def test_health_endpoint(client):
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'

def test_api_status_endpoint(client):
    response = client.get('/api/status')
    assert response.status_code == 200
