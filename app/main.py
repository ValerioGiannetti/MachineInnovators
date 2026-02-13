from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from typing import List
app = FastAPI(title="Twitter Sentiment API")

# Caricamento modello (Viene scaricato alla prima esecuzione)
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

# Mappatura delle label per un output più leggibile
LABEL_MAPPING = {
    "negative": "Negativo",
    "neutral": "Neutro",
    "positive": "Positivo"
}

class Tweet(BaseModel):
    #Aggiunto controllo su lunghezza testo
    text: str = Field(..., min_length=1, max_length=280)

class TweetList(BaseModel):
    #Classe che può gestire una lista di tweet
    tweets: List[Tweet] = Field(..., min_items=1, max_items=20)

@app.get("/")
def read_root():
    return {"status": "healthy"}

@app.post("/predict")
def predict(tweet: Tweet):
    result = sentiment_task(tweet.text)
    # Prendo la label originale (es. 'positive') e cerco nel dizionario
    # Se non la trovo, tengo quella originale
    label_human = LABEL_MAPPING.get(result['label'], result['label'])
    formatted_results = []
    formatted_results.append({
            "sentiment": label_human,
            "confidence": round(result['score'], 4) # Arrotondo a 4 decimali
        })
    return {"results": formatted_results}



@app.post("/predict/list")
def predict_batch(data: TweetList):
    # Estrazione testi
    texts = [t.text for t in data.tweets]
    
    # Inferenza
    raw_results = sentiment_task(texts)
    
    # Formattazione e Mappatura delle Label
    formatted_results = []
    for res in raw_results:
        # Prendo la label originale (es. 'positive') e cerco nel dizionario
        # Se non la trovo, tengo quella originale
        label_human = LABEL_MAPPING.get(res['label'], res['label'])
        
        formatted_results.append({
            "sentiment": label_human,
            "confidence": round(res['score'], 4) # Arrotondo a 4 decimali
        })
    
    return {"results": formatted_results}

# Integrazione Prometheus
Instrumentator().instrument(app).expose(app)

    
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)