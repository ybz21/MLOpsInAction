import os
import requests

current_dir = os.path.dirname(__file__)


def main():

    data_url = 'https://github.com/ybz21/MLOpsInAction/raw/master/data/sms_cls_train.csv'

    data_dir = os.path.join(current_dir, "data")
    data_path = os.path.join(data_dir, "sms_cls_train.csv")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    response = requests.get(data_url)
    with open(data_path, "wb") as f:
        f.write(response.content)


if __name__ == '__main__':
    main()
