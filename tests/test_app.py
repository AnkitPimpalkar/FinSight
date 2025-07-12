from app import app

def test_index_route():
    """Test that the index route returns 200"""
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200

def test_app_running():
    """Test that the Flask app exists"""
    assert app is not None
