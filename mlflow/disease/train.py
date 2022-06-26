import mlflow
import click

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score


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

        mlflow.sklearn.log_model(lin_model, "linear-model")

        y_pred = lin_model.predict(x_test)
        report = classification_report(y_test, y_pred)
        print(report)

        clsf_report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        clsf_report.to_csv('classification_report.csv', index=True)
        mlflow.log_artifact('classification_report.csv')

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('f1', f1)


if __name__ == '__main__':
    task()
