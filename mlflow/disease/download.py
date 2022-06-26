import mlflow
import click
import os
import requests

current_dir = os.path.dirname(__file__)


@click.command(help="Download csv data ...")
def task():
    with mlflow.start_run() as mlrun:
        url = 'https://github.com/ybz21/MLOpsInAction/raw/master/data/heart_disease_uci.csv'
        file_dir = os.path.join(current_dir, "data")
        filepath = os.path.join(file_dir, "raw.csv")
        os.system(f"mkdir -p {file_dir}")

        response = requests.get(url)
        with open(filepath, "wb") as f:
            f.write(response.content)

        mlflow.log_artifacts("data", artifact_path="data")


if __name__ == '__main__':
    task()
