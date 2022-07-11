import os.path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(current_dir)


def main():
    path = os.path.join(project_dir, 'data', 'sms_cls_train.csv')
    df = pd.read_csv(path, encoding='latin', sep='\t', header=None, names=['label', 'text'])
    df['type'] = df['label'].map(lambda a: 1 if a == 'ham' else 0)

    # 使用tf-idf构建词频
    tf_vect = TfidfVectorizer(binary=True)
    nb_model = MultinomialNB(alpha=1, fit_prior=True)

    pipe_model = Pipeline([("vectorizer", tf_vect), ("classifier", nb_model)])

    x_train, x_test, y_train, y_test = train_test_split(df.text, df.type, test_size=0.20, random_state=100)
    print("train count: ", x_train.shape[0], "test count: ", x_test.shape[0])

    pipe_model.fit(x_train, y_train)

    y_pred = pipe_model.predict(x_test)
    print("accuracy on test data: ", accuracy_score(y_test, y_pred))

    model_dir = os.path.join(project_dir, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'naive_bayes.pkl')

    joblib.dump(pipe_model, model_path)


if __name__ == '__main__':
    main()
