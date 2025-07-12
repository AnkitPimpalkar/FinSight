from app import app

def test_app_creation():
    """Test that the Flask app exists and is configured"""
    assert app is not None
    assert app.config['TESTING'] is not True  # Ensure we're not in testing mode by default
