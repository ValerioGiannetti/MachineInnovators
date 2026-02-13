from fastapi.testclient import TestClient
import pytest
from app.main import app 

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

# --- TEST VALIDAZIONE  ---

def test_predict_empty_text():
    """Verifica che un testo vuoto venga bloccato (min_length=1)"""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422  

def test_predict_too_long_text():
    """Verifica che un testo > 280 caratteri venga bloccato"""
    long_text = "a" * 281
    response = client.post("/predict", json={"text": long_text})
    assert response.status_code == 422

def test_predict_special_characters():
    """Verifica che caratteri speciali ed emoji siano gestiti correttamente"""
    response = client.post("/predict", json={"text": " @test #python #MLOps 12345"})
    assert response.status_code == 200
    assert "sentiment" in response.json()["results"][0]

def test_batch_too_many_items():
    """Verifica che superare il max_items (20) sollevi errore"""
    tweets = [{"text": "test"}] * 21
    response = client.post("/predict/list", json={"tweets": tweets})
    assert response.status_code == 422

# --- TEST COERENZA SENTIMENT ---

@pytest.mark.parametrize("input_text, expected_sentiment", [
    ("I am so happy today, everything is wonderful!", "Positivo"),
    ("This is the worst day of my life, I hate everything.", "Negativo"),
    ("The sky is blue and the grass is green.", "Neutro"),
])
def test_sentiment_coherence(input_text, expected_sentiment):
    """Verifica che il sentiment restituito sia quello atteso per frasi ovvie"""
    response = client.post("/predict", json={"text": input_text})
    assert response.status_code == 200
    
    # Verifichiamo la struttura corretta
    result = response.json()["results"][0]
    assert result["sentiment"] == expected_sentiment
    assert 0 <= result["confidence"] <= 1

def test_batch_prediction_logic():
    """Verifica che l'endpoint list restituisca lo stesso numero di input"""
    payload = {
        "tweets": [
            {"text": "I love it"},
            {"text": "I hate it"}
        ]
    }
    response = client.post("/predict/list", json=payload)
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert results[0]["sentiment"] == "Positivo"
    assert results[1]["sentiment"] == "Negativo"