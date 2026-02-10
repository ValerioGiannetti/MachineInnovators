from datasets import load_dataset
from transformers import pipeline
import numpy as np
import evaluate


# Carica il dataset (subset sentiment)
dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="test")

# Inizializza la pipeline di inferenza
# Il modello mappa internamente: 0 -> Negative, 1 -> Neutral, 2 -> Positive
pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0)

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Funzione per mappare l'output del modello all'ID numerico del dataset
label_mapping = {"negative": 0, "neutral": 1, "positive": 2}



def run_evaluation(dataset_split):
    # 1. Convertiamo esplicitamente in una lista di stringhe
    # Forza ogni elemento a essere stringa per evitare valori None o inaspettati
    texts = [preprocess(str(t)) for t in dataset_split["text"]]

    # 2. Eseguiamo l'inferenza con batch_size per velocità
    results = pipe(texts, truncation=True, max_length=512, batch_size=32)

    # 3. Mapping delle label
    # Il modello usa 'Positive', 'Negative', 'Neutral' (occhio al case-sensitive!)
    return [label_mapping[res["label"].lower()] for res in results]

# Carica le metriche di evaluate di hugginface
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")




def calcolo_metriche(predictions):
    # Calcola i risultati
    results = {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=references),
        "precision": precision_metric.compute(predictions=predictions, references=references, average="macro"),
        "recall": recall_metric.compute(predictions=predictions, references=references, average="macro"),
        "f1": f1_metric.compute(predictions=predictions, references=references, average="macro"),
    }

    print(f"Accuracy: {results['accuracy']['accuracy']:.4f}")
    print(f"Macro Recall: {results['recall']['recall']:.4f}")
    print(f"Precision: {results['precision']['precision']:.4f}")

if __name__ == "__main__":
    # Estrai testi e label reali dal dataset
    references = dataset["label"]
    predictions = run_evaluation(dataset)
    calcolo_metriche(predictions)