from datasets import load_dataset
from transformers import pipeline
import evaluate

def run_evaluation():
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="test")
    pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=-1)
    
    # Mappatura label: il modello usa LABEL_0, LABEL_1, LABEL_2
    # Tweet_eval usa 0 (neg), 1 (neu), 2 (pos)
    results = pipe(dataset["text"], truncation=True)
    
    predictions = [int(res['label'].split('_')[1]) for res in results]
    references = dataset["label"]

    acc = evaluate.load("accuracy").compute(predictions=predictions, references=references)
    prec = evaluate.load("precision").compute(predictions=predictions, references=references, average="macro")
    rec = evaluate.load("recall").compute(predictions=predictions, references=references, average="macro")

    print(f"Metrics: {acc}, {prec}, {rec}")

if __name__ == "__main__":
    run_evaluation()