
import json
from server import create_app

def test_route_ok():
    app = create_app()
    client = app.test_client()
    resp = client.get("/emotionDetector?textToAnalyze=I%20am%20happy")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "dominant_emotion" in data

def test_route_400():
    app = create_app()
    client = app.test_client()
    resp = client.get("/emotionDetector?textToAnalyze=")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data.get("error") == "Text is required."
