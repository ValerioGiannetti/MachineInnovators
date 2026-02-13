from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel, Field
from typing import List
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn

app = FastAPI(title="Twitter Sentiment API")

model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

LABEL_MAPPING = {
    "negative": "Negativo",
    "neutral": "Neutro",
    "positive": "Positivo"
}

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
    # result è una lista, es: [{'label': 'positive', 'score': 0.9}]
    prediction = sentiment_task(tweet.text)[0] 
    
    label_human = LABEL_MAPPING.get(prediction['label'], prediction['label'])
    
    return {
        "results": [{
            "sentiment": label_human,
            "confidence": round(prediction['score'], 4)
        }]
    }

@app.post("/predict/list")
def predict_batch(data: TweetList):
    texts = [t.text for t in data.tweets]
    raw_results = sentiment_task(texts)
    
    formatted_results = []
    for res in raw_results:
        label_human = LABEL_MAPPING.get(res['label'], res['label'])
        formatted_results.append({
            "sentiment": label_human,
            "confidence": round(res['score'], 4)
        })
    
    return {"results": formatted_results}

Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)