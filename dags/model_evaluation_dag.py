from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
# Importiamo la tua funzione di valutazione
from evaluate_model import run_evaluation_and_send_feedback

'''
Questo script si collega ad AirFlow creando un DAG
Qui viene definito il flusso di lavoro:
Scarico dataset, valuto modello e invio feedback
'''


default_args = {
    'owner': 'machine_team',
    'depends_on_past': False,
    'email_on_failure': False, #Per il momento metto False
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'sentiment_model_evaluation',
    default_args=default_args,
    description='Job periodico per calcolo Accuracy e Precision',
    schedule_interval='@hourly', # Esegue ogni ora
    start_date=datetime(2026, 2, 16),
    catchup=False,
    tags=['machine_innovators', 'sentiment'],
) as dag:

    eval_task = PythonOperator(
        task_id='evaluate_and_feedback',
        python_callable=run_evaluation_and_send_feedback,
    )

    eval_task