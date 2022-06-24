import mlflow
import click


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


@click.command(help="This program does ...")
@click.option("--data_path")
def task(data_path):
    with mlflow.start_run() as mlrun:
        final_pdf = pd.read_csv(data_path)

        data = final_pdf.drop('num', axis=1)
        label = final_pdf['num']
        x = data.values
        y = label.values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        lin_model = RidgeClassifier()
        lin_model.fit(x_train, y_train)

        y_preds = lin_model.predict(x_test)
        print(classification_report(y_test, y_preds))


def ml_alg(x, y):
    pass


if __name__ == '__main__':
    task()
