from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_positive():
    response = client.post("/predict", json={"text": "I love MLOps!"})
    assert response.status_code == 200
    assert "label" in response.json()["prediction"]

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200