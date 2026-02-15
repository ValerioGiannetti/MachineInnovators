## Monitoraggio MLOps Avanzato

Questo stack traccia la qualità del modello (ML Metrics) oltre che le performance del server (HTTP Metrics).

### 1. Metriche Custom Esposte
Il sistema espone automaticamente su Prometheus:
* **Model Confidence**: La confidenza media del modello per ogni predizione (per rilevare se il modello diventa "insicuro").
* **Sentiment Distribution**: Tracciamento delle classi predette (Positivo, Neutro, Negativo) per il rilevamento del **Data Drift**.
* **Confusion Matrix (Feedback Loop)**: Conteggio di True Positives, False Positives, etc., inviati tramite l'endpoint di feedback.

### 2. Ciclo di Valutazione (Evaluator Job)
Il progetto include un servizio **Evaluator** (container Docker dedicato) che:
* Esegue periodicamente lo script `evaluate_model.py`.
* Valuta il modello su un "Golden Dataset" (Twitter Sentiment Test Set).
* Invia i risultati all'API tramite l'endpoint `/feedback/batch`.
* Permette a Grafana di calcolare in tempo reale **Accuracy, Precision e Recall**.

### 3. Gestione Versioni (Champion-Challenger)
Tramite la variabile d'ambiente `MODEL_VERSION` nel file `docker-compose.yml`, è possibile etichettare le metriche. Questo permette di:
* Confrontare le performance tra due versioni diverse del modello.
* Visualizzare grafici comparativi su Grafana.
* Implementare strategie di deploy sicuro.

---

## Dashboard e Alerting

### Query Utili per Grafana
* **Accuracy**: `(sum(model_performance_total{metric_type="tp"}) + sum(model_performance_total{metric_type="tn"})) / sum(model_performance_total)`
* **Data Drift**: `sum by (label) (rate(sentiment_analysis_total[1h]))`

### Alerting Schedulato
Il sistema è predisposto per inviare notifiche se:
1. L'**Accuracy** scende sotto lo **0.70**.
2. La **Confidenza media** scende sotto **0.65** per più di un'ora.
3. Si rileva una distribuzione anomala delle classi (es. >90% Negativo).

---

##  Manutenzione e Retraining
Quando ricevi un alert di bassa accuracy:
1. Analizza i dati di input per identificare nuovi trend linguistici.
2. Aggiorna il dataset di training.
3. Effettua il deploy della nuova immagine Docker incrementando la variabile `MODEL_VERSION=v2`.