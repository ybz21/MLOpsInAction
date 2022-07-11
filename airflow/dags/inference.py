import os

import joblib

current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(current_dir)


def main():
    model_path = os.path.join(project_dir, 'model', 'naive_bayes.pkl')
    pipe_model = joblib.load(model_path)

    msg = ['aa hello']
    result = pipe_model.predict(msg)
    print(result)
    label = 'ham' if result == [1] else 'spam'
    print(label)


if __name__ == '__main__':
    main()
