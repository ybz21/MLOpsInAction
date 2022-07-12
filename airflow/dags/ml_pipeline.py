from datetime import timedelta
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': days_ago(31),
    # 填入邮箱，方便失败时发送邮件
    'email': ['mlops@4paradigm.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),bu
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    'trigger_rule': 'all_success'
}

# instantiates a directed acyclic graph
dag = DAG(
    'ml_pipeline',
    default_args=default_args,
    description='A simple Machine Learning pipeline for spam message classification',
    schedule_interval=timedelta(days=1),
)

# 下载训练数据环节，可以用于
download_train_data = BashOperator(
    task_id='download_train_data',
    bash_command='python3 /opt/airflow/dags/download.py --mode train',
    dag=dag,
)
#
train = BashOperator(
    task_id='train',
    depends_on_past=False,
    bash_command='python3 /opt/airflow/dags/train.py',
    retries=3,
    dag=dag,
)

download_inference_data = BashOperator(
    task_id='download_inference_data',
    depends_on_past=False,
    bash_command='python3 /opt/airflow/dags/download.py --mode inference',
    retries=3,
    dag=dag,
)

inference = BashOperator(
    task_id='inference',
    depends_on_past=False,
    bash_command='python3 /opt/airflow/dags/inference.py',
    retries=3,
    dag=dag,
)

# download images runs first, then train, then down.
download_train_data >> train >> download_inference_data >> inference
