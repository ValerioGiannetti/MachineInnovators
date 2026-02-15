from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel, Field
from typing import List
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn
from prometheus_client import Counter, Gauge
import os

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

app = FastAPI(title="Twitter Sentiment API")

model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

LABEL_MAPPING = {
    "negative": "Negativo",
    "neutral": "Neutro",
    "positive": "Positivo"
}

# Gauge per la confidenza media (mostra l'ultimo valore ricevuto)
MODEL_CONFIDENCE = Gauge(
    'model_confidence_score', 
    'Confidenza della predizione attuale',
    ['model_version']
)

# Counter per la distribuzione delle classi  per rilevare il Data Drift
SENTIMENT_DISTRIBUTION = Counter(
    'sentiment_analysis_total', 
    'Distribuzione delle predizioni di sentiment', 
    ['label']
)

class Tweet(BaseModel):
    text: str = Field(..., min_length=1, max_length=280)

class TweetList(BaseModel):
    # classe per gestire una lista di tweet
    tweets: List[Tweet] = Field(..., min_length=1, max_length=20)

@app.get("/")
def read_root():
    return {"status": "healthy"}

@app.post("/predict")
def predict(tweet: Tweet):
    # Esecuzione dell'inferenza
    prediction = sentiment_task(tweet.text)[0] 
    
    label_raw = prediction['label']
    score = prediction['score']
    label_human = LABEL_MAPPING.get(label_raw, label_raw)
    
    # --- AGGIORNAMENTO METRICHE CUSTOM ---
    # 1. Registra la confidenza (Gauge)
    MODEL_CONFIDENCE.labels(model_version=MODEL_VERSION).set(score)
    
    # 2. Incrementa il contatore per la label specifica (Counter)
    SENTIMENT_DISTRIBUTION.labels(label=label_human).inc()
    # -------------------------------------

    return {
        "results": [{
            "sentiment": label_human,
            "confidence": round(score, 4)
        }]
    }

@app.post("/predict/list")
def predict_batch(data: TweetList):
    texts = [t.text for t in data.tweets]
    raw_results = sentiment_task(texts)
    
    formatted_results = []
    total_confidence = 0
    
    for res in raw_results:
        label_raw = res['label']
        score = res['score']
        label_human = LABEL_MAPPING.get(label_raw, label_raw)
        
        # 1. Incrementa il contatore per ogni predizione nel batch
        SENTIMENT_DISTRIBUTION.labels(label=label_human).inc()
        
        total_confidence += score
        
        formatted_results.append({
            "sentiment": label_human,
            "confidence": round(score, 4)
        })
    
    # 2. Registra la confidenza media del batch nel Gauge
    if raw_results:
        mean_confidence = total_confidence / len(raw_results)
        MODEL_CONFIDENCE.labels(model_version=MODEL_VERSION).set(mean_confidence)
    
    return {"results": formatted_results}

MODEL_PERFORMANCE = Counter(
    'model_performance_total', 
    'Conteggio per il calcolo di Accuracy, Precision e Recall', 
    ['metric_type', 'model_version']
)

@app.post("/feedback/batch")
def feedback_batch(results: List[dict]):
    """
    Consente di calcolare metriche di performance statistiche (Accuracy, Precision, Recall, F1-Score)
    confrontando le etichette predette dal modello con le etichette reali
    Riceve una lista di risultati: [{"correct": "Positivo", "predicted": "Negativo"}, ...]
    """
    for item in results:
        c = item['correct']
        p = item['predicted']
        
        if c == "Positivo":
            metric = 'tp' if p == "Positivo" else 'fn'
        else: # Casi Negativo/Neutro (semplificato per Precision/Recall su Positivo)
            metric = 'fp' if p == "Positivo" else 'tn'
            
        MODEL_PERFORMANCE.labels(metric_type=metric, model_version=MODEL_VERSION).inc()
    
    return {"status": "Metriche aggiornate"}

Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)