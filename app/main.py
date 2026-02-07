from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Twitter Sentiment API")

# Caricamento modello (Viene scaricato alla prima esecuzione)
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

class Tweet(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "healthy"}

@app.post("/predict")
def predict(tweet: Tweet):
    result = sentiment_task(tweet.text)
    return {"prediction": result[0]}

# Integrazione Prometheus
Instrumentator().instrument(app).expose(app)