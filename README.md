# Twitter Sentiment API with Monitoring Stack

Questo progetto fornisce un'API per l'analisi del sentiment basata sul modello **RoBERTa**, integrando un sistema di monitoraggio completo con **Prometheus** e **Grafana**.

## Caratteristiche
* **API Sentiment**: Analisi dei testi tramite modelli di Deep Learning.
* **Monitoraggio**: Performance e metriche dell'API raccolte in tempo reale.
* **Visualizzazione**: Dashboard Grafana pre-configurata (opzionale).
* **Dockerized**: Avvio immediato con un singolo comando.

##  Stack Tecnologico
* **Backend**: Python (FastAPI) + Model (RoBERTa)
* **Metrics**: Prometheus
* **Dashboard**: Grafana
* **Infrastructure**: Docker & Docker Compose

---

##  Guida all'avvio

### 1. Prerequisiti
* Docker e Docker Compose installati.
* Un account Docker Hub (se desideri usare l'immagine dalla CD).

### 2. Configurazione Segreti
Crea un file `.env` nella cartella principale del progetto per gestire la sicurezza:

```env
GRAFANA_SECRET=la_tua_password_sicura