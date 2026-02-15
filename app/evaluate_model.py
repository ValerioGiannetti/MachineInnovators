import requests
from datasets import load_dataset
from transformers import pipeline
import evaluate
import os

'''
Cambiato logica di evalue model,
ora si aggancia a prometheus e i grafici si aggiornano automaticamente
'''

# Configurazione endpoint API (dove gira il tuo main.py)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Mapping per il dataset 

LABEL = {0: "Negativo", 1: "Neutro", 2: "Positivo"}

# Mapping per le predizioni del modello (Label -> Nome)
MODEL_LABEL = {"negative": "Negativo", "neutral": "Neutro", "positive": "Positivo"}

# Carica il dataset
dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="test")

# Inizializza la pipeline
pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=-1) # -1 per CPU

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def run_evaluation_and_send_feedback():
    print(f"Avvio valutazione su {len(dataset)} tweet...")
    
    texts = [preprocess(str(t)) for t in dataset["text"]]
    references = dataset["label"]
    
    # 1. Eseguiamo l'inferenza
    results = pipe(texts, truncation=True, max_length=512, batch_size=32)
    
    # 2. Prepariamo il payload per l'endpoint /feedback/batch
    feedback_payload = []
    predictions_for_metrics = []
    
    for i in range(len(results)):
        pred_label_raw = results[i]["label"].lower()
        pred_human = MODEL_LABEL.get(pred_label_raw, pred_label_raw)
        correct_human = LABEL[references[i]]
        
        feedback_payload.append({
            "correct": correct_human,
            "predicted": pred_human
        })
        
        # Mappa per il calcolo locale delle metriche (opzionale)
        label_mapping_id = {"negative": 0, "neutral": 1, "positive": 2}
        predictions_for_metrics.append(label_mapping_id[pred_label_raw])

    # 3. INVIO DATI A PROMETHEUS (tramite l'API)
    try:
        response = requests.post(f"{API_URL}/feedback/batch", json=feedback_payload)
        if response.status_code == 200:
            print(" Metriche inviate con successo a Prometheus via API.")
        else:
            print(f" Errore nell'invio delle metriche: {response.status_code}")
    except Exception as e:
        print(f" Impossibile connettersi all'API: {e}")

    # 4. Calcolo locale (per log)
    calcolo_metriche_locali(predictions_for_metrics, references)

def calcolo_metriche_locali(predictions, references):
    accuracy_metric = evaluate.load("accuracy")
    results = accuracy_metric.compute(predictions=predictions, references=references)
    print(f" Accuracy calcolata: {results['accuracy']:.4f}")

if __name__ == "__main__":
    run_evaluation_and_send_feedback()