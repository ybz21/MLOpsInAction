import time

import mlflow
import os

import click

current_dir = os.path.dirname(__file__)


# main.py
@click.command()
def workflow():
    with mlflow.start_run() as active_run:
        print("Launching 'download'")
        download_run = mlflow.run(".", "download", parameters={})
        download_run = mlflow.tracking.MlflowClient().get_run(download_run.run_id)

        file_path = os.path.join(download_run.info.artifact_uri, "data/raw.csv")
        # file_path_uri = download_run.info.filepath
        print(f'=====artifact_uri:{file_path}')
        print("Finish 'download'")

        print("Launching 'process'")
        process_run = mlflow.run(".", "process", parameters={"file_path": file_path})
        process_run = mlflow.tracking.MlflowClient().get_run(process_run.run_id)
        data_path_uri = os.path.join(download_run.info.artifact_uri, "data/data.csv")

        print("Launching 'train'")
        train_run = mlflow.run(".", "train", parameters={"data_path": data_path_uri})
        train_run = mlflow.tracking.MlflowClient().get_run(train_run.run_id)

        # mlflow.log_artifacts(train_run.run_id, os.path.join(current_dir, 'artifacts'))


if __name__ == '__main__':
    workflow()
