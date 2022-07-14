# Pipeline
## 6.1 Pipeline工具概述
在3.1.5章节中，我们介绍了ML为何难以落地的痛点，其中工程层面上存在着工具，流程复杂且定制化的问题，
本章则针对这个痛点，主要介绍MLOps领域一些开源的、通用的Pipeline编排工具，帮助提升MLOps流程的串联效率。

## 6.2 AirFlow
Apache AirFlow是由Airbnb孵化的一款用于计划性的调度、监控流水线的开源工具。其主要提供以下功能：
* 用于定义各个节点执行的工作，节点间的关系，同时定义执行计划，失败策略等标准的Python SDK
* 可视化操作的Web UI
  下图展示了airflow的交互界面，在这个界面上可以创建任务，对具体的任务使用进行代码编辑等操作。
* ![images/chapter-6-airflow-screenshot.png](images/chapter-6-airflow-screenshot.png)

AirFlow的设计原则包括以下几点：
* 高伸缩性

  Airflows使用模块化的架构，利用消息队列来编排任意数量的任务，可以支持海量的任务。
* 动态调用

  Airflow流水线使用python定义，允许通过Python脚本动态生成流水线。
* 高可扩展

  Airlfow允许用户自定义算子库，以帮助用户完成自己所要适配的环境。
  下面的章节将对AirFlow核心概念和使用方法进行详细描述。
### 6.2.1 核心概念
Airflow中，流水线以DAG(有向无环图，Directed Acyclic Graph)形式展现，每个DAG由多个独立的Task组成，这些Task间的依赖关系和数据流转使用连线来表示。

下面是对Airflow的一些核心概念的介绍：
* DAG  is the core concept of Airflow, collecting Tasks together, organized with dependencies and relationships to say how they should run.
  下图是dag的一个示例：![img.png](images/img.png)
  It defines four Tasks - A, B, C, and D - and dictates the order in which they have to run, and which tasks depend on what others. It will also say how often to run the DAG - maybe “every 5 minutes starting tomorrow”, or “every day since January 1st, 2020”.
* Task
  A Task is the basic unit of execution in Airflow. Tasks are arranged into DAGs, and then have upstream and downstream dependencies set between them into order to express the order they should run in.
* Operator
  An Operator is conceptually a template for a predefined Task, that you can just define declaratively inside your DAG:


### 6.2.2 使用方法

安装 Airflow

本文在github MLOpsInAction仓库中集成好了Airflow镜像定制、服务启动的脚本。本文采用devops的方法，通过docker安装airflow，用户可以在Linux、Mac、Windows WSL Ubuntu系统上执行下面脚本，即可实现Airflow服务启动。
```shell
# git pull https://github.com/ybz21/MLOpsInAction.git
# cd MLOpsInAction/airflow
# sh start.sh
```
其中start.sh包含两部分: 一部分是对官方的apache/airflow镜像做定制化，另一部分是使用docker-compose启动Airflow服务及其支撑组件。如果您想将本文的安装方法使用到自己的业务上，可以对这两部分进行改造，下面是这两部分的说明。

镜像定制化：本文在官方镜像的基础上，增加了Python的scikit-learn库的安装，方便后面进行机器学习的训练、和推理。
用户也可以根据自己需求，修改本Repo中的MLOpsInAction/airflow/requirements.txt文件，增加自己所需的Python库，也可以修改Dockerfile，安装自己所需的软件。

服务启动：本文使用官方的docker-compose.yaml文件，仅将其中Airflow服务使用的镜像变为了上一步定制化的镜像。
如果您对Airflow服务有定制化的需求，也可以修改该文件。

当执行完该脚本后，在浏览器中输入地址 http://{脚本执行机器ip}:8080 （用户名密码均为airflow），，即可看到下图所示的Airflow的界面。

在日常生活中，我们经常会被垃圾邮件、垃圾短信以及各种垃圾消息困扰，如何通过使用机器学习的方式，将这些垃圾信息进行自动分类并过滤，减少生活工作中的不必要的烦扰成为一个较好的问题。
本节以训练一个垃圾信息分类器为例，利用Airflow构建一个定时下载训练数据、重训练、下载推理数据、推理的Pipeline。
该Pipeline具有一定的可迁移性，如果您的业务具有数据变更快且需要定期更新模型或者需要定时执行离线批量预测任务，可以基于本节中的Pipeline进行改造。

* 定义pipeline
  再dag目录下，新建ml_pipeline.py 文件，文件内容如下：
  ![img_1.png](img_1.png)
```python
from datetime import timedelta
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago

# DAG属性定义
default_args = {
  'owner': 'mlops',
  'depends_on_past': False,
  'start_date': days_ago(31),
  # 填入邮箱，方便失败、重试时发送邮件
  'email': ['mlops@4paradigm.com'],
  # 失败时发邮件告警
  'email_on_failure': True,
  'email_on_retry': False,
  # 重试次数
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

# 定义dag
dag = DAG(
  'ml_pipeline',
  default_args=default_args,
  description='A simple Machine Learning pipeline for spam message classification',
  schedule_interval=timedelta(days=1),
)

# 下载训练数据环节，用于下载标记好的垃圾短信、邮件数据，每天都有增量数据加入
download_train_data = BashOperator(
  task_id='download_train_data',
  bash_command='python3 /opt/airflow/dags/download.py --mode train',
  dag=dag,
)

# 训练环节，基于新的数据进行训练，得到更鲁棒的模型
train = BashOperator(
  task_id='train',
  depends_on_past=False,
  bash_command='python3 /opt/airflow/dags/train.py',
  retries=3,
  dag=dag,
)

# 下载推理数据环节，用于拉取线上业务需要推理的数据
download_inference_data = BashOperator(
  task_id='download_inference_data',
  depends_on_past=False,
  bash_command='python3 /opt/airflow/dags/download.py --mode inference',
  retries=3,
  dag=dag,
)

# 推理环节，用于对每天需要推理的业务数据进行推理
inference = BashOperator(
  task_id='inference',
  depends_on_past=False,
  bash_command='python3 /opt/airflow/dags/inference.py',
  retries=3,
  dag=dag,
)

# download images runs first, then train, then down.
download_train_data >> train >> download_inference_data >> inference
```

* 定义下载
```python
import os
import requests
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, required=False, default='train')


def main():
  args = parser.parse_args()
  data_dir = os.path.join(project_dir, "data")
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  if args.mode == 'train':
    data_url = 'https://github.com/ybz21/MLOpsInAction/raw/master/data/sms_cls_train.csv'
    data_path = os.path.join(data_dir, "sms_cls_train.csv")
  else:
    data_url = 'https://github.com/ybz21/MLOpsInAction/raw/master/data/sms_cls_inference.csv'
    data_path = os.path.join(data_dir, "sms_cls_inference.csv")

  response = requests.get(data_url)
  with open(data_path, "wb") as f:
    f.write(response.content)

  print('finish download')


if __name__ == '__main__':
  main()
```

* 定义训练
```python
import os.path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)


def main():
    path = os.path.join(project_dir, 'data', 'sms_cls_train.csv')
    df = pd.read_csv(path, encoding='latin', sep='\t', header=None, names=['label', 'text'])
    df['type'] = df['label'].map(lambda a: 1 if a == 'ham' else 0)

    # tf-idf用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。
    # 使用tf-idf构建词向量，可以较好的提取出文本内容强相关的词汇，把一些没有实际的词语筛除掉
    tf_vect = TfidfVectorizer(binary=True)
    nb_model = MultinomialNB(alpha=1, fit_prior=True)

    # 创建pipe的模型，如果模型不包含前面的vectorizer，会导致在推理计算的时候找不到相应的词向量
    pipe_model = Pipeline([("vectorizer", tf_vect), ("classifier", nb_model)])

    # 训练集、测试集切分
    x_train, x_test, y_train, y_test = train_test_split(df.text, df.type, test_size=0.20, random_state=100)
    print("train count: ", x_train.shape[0], "test count: ", x_test.shape[0])

  # 训练模型
    pipe_model.fit(x_train, y_train)

    # 评估模型
    y_pred = pipe_model.predict(x_test)
    print("accuracy on test data: ", accuracy_score(y_test, y_pred))

  # 保存模型
    model_dir = os.path.join(project_dir, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'naive_bayes.pkl')
    joblib.dump(pipe_model, model_path)

    print('finish train')


if __name__ == '__main__':
    main()

```

* 定义推理
```python
import os
import pandas as pd
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)


def main():
  model_path = os.path.join(project_dir, 'model', 'naive_bayes.pkl')
  data_path = os.path.join(project_dir, 'data', 'sms_cls_inference.csv')
  result_path = os.path.join(project_dir, 'data', 'sms_cls_inference_result.csv')

  pipe_model = joblib.load(model_path)

  df = pd.read_csv(data_path, encoding='latin', sep='\t', header=None, names=['text', 'type', 'label'])

  result = pipe_model.predict(df.text)
  df['type'] = result
  df['label'] = df['type'].map(lambda a: 'spam' if a == 1 else 'ham')

  # 保存推理结果，实际业务中可以上传到hive等数据仓库中
  df.to_csv(result_path, columns=['label', 'text'], index=False)

  print('finish inference')


if __name__ == '__main__':
  main()
```


## 6.3 MLflow
MLFlow是由 Apache Spark 技术团队开源的一个机器学习平台，主要提供以下功能：
* 1 跟踪、记录实验过程，交叉比较实验参数和对应的结果。
* 2 把代码打包成可复用、可复现的格式，可用于成员分享和针对线上部署。
* 3 管理、部署来自多个不同机器学习框架的模型到大部分模型部署和推理平台。
* 4 针对模型的全生命周期管理的需求，提供集中式协同管理，包括模型版本管理、模型状态转换、数据标注。

这个其中第1、3、4两个功能点解决了3.1.5章节的问题3（版本化及重现）。 MLFlow从开发者的角度出发，为开发者提供代码、模型追踪，方便开发者对不同历史版本的代码和模型进行比较。
下面的章节将对MLFlow核心概念和使用方法进行详细描述。
### 6.3.1 核心概念
* 跟踪（Tracking）

MLFlow Tracking提供了API和UI两种工具，用于在运行机器学习代码时记录参数、代码版本、结果指标，并且可以可视化。
开发者可以在任何环境（如python脚本或notebook文件）中使用 MLflow Tracking 将结果记录到本地文件或服务器，然后比较多次运行的结果，团队还可以使用它来比较不同开发着的结果。
MLflow Tracking是围绕实验代码运行概念组织的， 每次运行都会记录以下信息：
* 代码版本
  Git的commit号会记录在MLflow 项目的运行记录中。
* 运行起止时间
* 源文件（Source）
  MLflow项目中，具体运行的
* 参数（Parameters）
  运行文件的具体参数，以key-value键值对的形式出现。
* 指标（Metrics）
  指标用于记录一些关键数值型的数据，如模型的准确率、精确率、召回率等，MLflow提供了以UI的形式展现这些指标。
* 文件神器（Artifacts）
  可以使用MLflow的API，将任何格式的文件（如图片、数据、训练代码、模型）记录下来，方便跟踪对比。

* 项目（Projects）

  MLflow 项目是一种以可重用和可重现的方式打包数据科学代码的格式。 项目组件包括用于运行项目的 API 和命令行工具，从而可以将项目链接到工作流中。
  从本质上讲，MLflow 项目是对机器学习代码运行的约定，以方便开发者运行、追踪。 每个项目只是一个文件目录或一个 Git 存储库，
  这个目录中包含三大部分：用于描述项目的MLProject文件、用于描述项目依赖的conda.yaml文件、代码文件。在6.3.2中会有项目的详细介绍如何编写一个Project。
* 模型（Models）

  MLflow 模型是一种用于打包机器学习模型的标准格式，可被各种下游工具进行使用。例如，通过 REST API 进行实时服务或在 Apache Spark 上进行批量推理。
* 模型注册中心（Model Registry）

  MLflow 模型注册中心可以认为是一个存储所有模型的模型仓库，开发者可以以 API 和 UI的形式，调用模型注册中心，管理 MLflow 模型的整个生命周期。

*
### 6.3.2 使用方法
首先安装MLflow， MLflow的安装我们分为本地模式和服务器模式，本地模式是指将所有的tracking数据、模型数据放在本地；服务器模式是指将tracking数据、模型数据放在服务器端。下面是两种模式的具体介绍。
* 本地模式
```
# pip3 install mlflow
# mlflow ui
```
在命令行中执上面命令，即可完成安装，在浏览器中输入网址 http://127.0.0.1:5000 即可看到如下图所示的MLFlow页面。
![images/chapter-6-mlflowui.png](images/img.png)
基于作者的实践，此种做法缺少一些依赖，导致MLFlow在模型查看功能上受限，因此更推荐下面的服务器模式。

* 服务器模式

由于tracking服务器依赖database、对象存储等组件，且服务器依赖较多的python库，安装部署过程较为复杂，本文基于devops的概念，使用docker，在服务器或则个人电脑本地的命令行工具中运行以下命令即可将。
```
# git clone https://github.com/ybz21/mlflow-docker-compose.git
# cd mlflow-docker-compose
# docker-compose up -d --build
# cd -
```
该过程会拉取github上的成熟的MLFlow服务器构建运行仓库，然后基于仓库代码构建MLflow服务器镜像，
并拉取MLflow的依赖组件：mysql、minio等，最终将整个服务启动起来。

本节以UCI心脏病数据集为例，使用MLflow工具构建从数据下载、特征处理、模型训练、模型上线的pipeline。

UCI心脏病数据集包含76个属性，但是所有已发布的实验都引用了其中14个属性的子集。本文选取的数据集(https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)包含16个属性。比14个属性多出id和database两个属性，下面是这16个属性的名称、含义、类型及解释说明。

|  列名   | 含义  | 类型  | 说明  |
|  ----  | ----  | ----  |----  |
| id  | 唯一标识 | int |  |
| age  | 年龄 | int |  |
| sex  | 性别 | str|  取值范围：Male/Femal|
| database  | 数据来源 |str |来源数据库的名称 |
| cp  | 胸痛类型 | str| 取值范围：typical angina, atypical angina, non-anginal, asymptomatic |
| trestbps  | 血压  |int|  |
| chol  | 胆固醇 | 单元格| |
| fbs  | 血糖 |bool|  是否大于120mg/dl |
| restecg  | 心电图结果 | str|取值范围：normal, stt abnormality, lv hypertrophy |
| thalach  | 最大心率 | int| |
| exang  | 运动性心绞痛 |bool|  |
| oldpeak  | 峰值运动ST段的斜率 | float| |
| slope  | 单元格 | str|取值范围：upsloping，flat，downsloping |
| ca  | 主要血管数量 |int|取值范围：0-3|
| thal  | | str|取值范围：normal; fixed defect; reversible defect |
| num  | 类别 | int|取值范围：1、2、3、4 |

下面具体讲述pipeline具体过程， 首先创建heart_disease_cls文件夹，在文件夹下进行下面的具体操作步骤：

* 1 创建工程

在当前目录创建MLProject文件，文件内容如下
```
name: heart_disease_cls

conda_env: conda.yaml

entry_points:
  download:
    command: "python download.py"

  feature:
    parameters:
      file_path: {type: str}
    command: "python feature.py --file_path {file_path}"

  train:
    parameters:
      data_path: {type: str}
    command: "python train.py  --data_path {data_path}"

  main:
    command: 'python main.py'
```
其中name为该MLFlow工程的名字， conda_env指向改工程的依赖文件conda.yaml，entry_points中罗列了工程中所涉及的程序入口、参数及启动命令。

* 2 定义环境依赖

MLflow会为每个工程初始化一个conda的独立运行环境，因此在在conda.yaml中约定环境的依赖包，本文中的约定如下：
```
name: heart_disease_cls
channels:
  - defaults
dependencies:
  - python=3.8
  - requests
  - pip:
      - mlflow>=1.0
      - scikit-learn
      - numpy
      - pandas
      - boto3
      - fsspec
      - s3fs
      - cloudpickle

```
该conda.yaml文件主要声明了项目依赖mlflow、scikit-learn、numpy、pandas。
其中pandas、numpy用于数据处理，scikit-learn用于模型训练，mlflow用于追踪记录实验和模型。

* 3.1 pipeline-数据下载

创建download.py，文件内容如下：
```
import mlflow
import click
import os
import requests

current_dir = os.path.dirname(__file__)

@click.command(help="Run download")
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
```
该文件主要负责下载数据集，并将其记录下来，方便在溯源的时候查看原始数据。

* 3.2 pipeline-特征处理


```
import mlflow
import click
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

current_dir = os.path.dirname(__file__)

@click.command(help="Run feature")
@click.option("--file_path")
def task(file_path):
    with mlflow.start_run() as mlrun:
        heart_disease = pd.read_csv(file_path)
        # 数据清理
        heart_disease.drop(['id', 'dataset'], axis=1, inplace=True)

        # 缺失值处理
        heart_disease["trestbps"].fillna(heart_disease["trestbps"].mean(), inplace=True)
        heart_disease["chol"].fillna(heart_disease["chol"].mean(), inplace=True)
        heart_disease["fbs"].fillna(heart_disease["fbs"].mode()[0], inplace=True)
        heart_disease["restecg"].fillna(heart_disease["restecg"].mode()[0], inplace=True)
        heart_disease["thalch"].fillna(heart_disease["thalch"].mean(), inplace=True)
        heart_disease["exang"].fillna(heart_disease["exang"].mode()[0], inplace=True)
        heart_disease["oldpeak"].fillna(heart_disease["oldpeak"].mean(), inplace=True)
        heart_disease["slope"].fillna(heart_disease["slope"].mode()[0], inplace=True)
        heart_disease["ca"].fillna(heart_disease["ca"].mean(), inplace=True)
        heart_disease["thal"].fillna(heart_disease["thal"].mode()[0], inplace=True)

        # one hot 编码
        cat_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(heart_disease[cat_features].values)

        matrix = enc.transform(heart_disease[cat_features].values).toarray()
        feature_labels = np.array(enc.categories_).ravel()

        col_names = []
        for col in cat_features:
            for val in heart_disease[col].unique():
                col_names.append("{}_{}".format(col, val))

        # 拼接onehot编码后的列和剩余列
        onehot_pdf = pd.DataFrame(data=matrix, columns=col_names, dtype=int)
        remian_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'num']
        remain_pdf = heart_disease[remian_features]
        final_pdf = pd.concat([onehot_pdf, remain_pdf], axis=1)

        # 将做完特征的数据写入csv，等待训练处理
        csv_file_path = os.path.join(current_dir, 'data/data.csv')
        final_pdf.to_csv(csv_file_path)

        mlflow.log_artifacts("data", artifact_path="data")

if __name__ == '__main__':
    task()

```
该步骤包含了对原始数据进行数据预处理，数据特征工程，最终将数据进行存储，并使用追踪的手段将数据log下来。

* 3.3 pipeline-算法训练
  创建train.py，内容如下
```
import mlflow
import click

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score

@click.command(help="Run train")
@click.option("--data_path")
def task(data_path):
    with mlflow.start_run() as mlrun:
        final_pdf = pd.read_csv(data_path)

        # 数据整理
        data = final_pdf.drop('num', axis=1)
        label = final_pdf['num']
        x = data.values
        y = label.values

        # 数据切分，分为训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        # 选择分类算法，此处选择随机森林，你也可以切换算法做实验，mlflow正好可以帮你完成切换的算法及生成的模型记录
        model = RandomForestClassifier()
        model.fit(x_train, y_train)

        # 记录模型，方便在MLflow ui中查看模型
        mlflow.sklearn.log_model(model, "model")

        # 测试数据预测
        y_pred = model.predict(x_test)

        # 指标计算，并将所有指标记载下来
        report = classification_report(y_test, y_pred, output_dict=True)
        clsf_report = pd.DataFrame(report).transpose()
        clsf_report.to_csv('classification_report.csv', index=True)
        mlflow.log_artifact('classification_report.csv')
        print(report)

        # 总体的大指标，记录下来，可在mlflow ui上直观可视
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # 记录模型指标，方便在今后换模型的时候进行对比
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('f1', f1)

        # 记录训练全过程，选用的算法
        mlflow.log_artifact('train.py')


if __name__ == '__main__':
    task()
```
该步骤展现的模型的训练环节，本文中采用了随机森林算法进行分类，通过训练，最终将模型进行存储，并连带训练文件、训练指标等一并进行存储，方便日后不断迭代，切换算法时，进行不同算法的指标比较。
至此，我们完成了pipeline的定义，接下来是运行pipe和上线。

* 4 运行pipeline

在物理机上，运行下面脚本
```
# export MLFLOW_TRACKING_URI=http://127.0.0.1:5000   # 这里ip为部署的mlflow tracking server的ip
# export MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000 # 这里ip为部署的mlflow tracking server的ip
# mkdir -p ~/.aws
cat <<EOF > ~/.aws/credentials
[default]
aws_access_key_id=minio
aws_secret_access_key=minio123
EOF

# mlflow run heart_disease_cls
```
由于记录追踪模型、指标等文件，需要对S3的存储对象进行远程操作，所以需要填写S3的一些鉴权信息，MLflow默认读取 ~/.aws/credentials作为存储配置。
即在http://127.0.0.1:5000 可以看到如下界面:
![img.png](images/chapter-6-mlflow-training.png)

* 5 模型上线
  从上图界面，点击最上面model列带有sklearn标志的，即为train.py的执行步骤，点击该步骤，进入如下页面
  ![img.png](images/chapter-6-mlflow-register-model.png)
  可以点击右下角的register model将模型发布到模型仓库，并在模型仓库将该模型的stage改为Production（可产品化），然后使用线面命令对模型仓库的服务进行上线即可
```
# mlflow models serve -m "models:/heart_disease_cls/Production"
```

至此，就使用mlflow完成了利用uci heart disease 数据集完成心脏病分类这一mlops的流程。

## 6.4 小结


## 6.4 TFX
TFX是TensorFlow Extended的简称，它是由谷歌发起并开源的一个端到端平台，用于部署生产型机器学习流水线，平台提供了一个配置框架和众多共享库，用来集成定义、启动和监控机器学习系统所需的常见组件。
TFX 主要提供以下功能：

* 用于构建机器学习流水线的工具包。借助 TFX 流水线，您可以在多个平台上编排机器学习工作流，例如 Apache Airflow、Apache Beam 和 Kubeflow Pipelines 平台。
* 一组标准组件，可用作流水线的一部分，或用作机器学习训练脚本的一部分。TFX 标准组件提供久经考验的功能，可帮助您轻松开始构建机器学习流程。
* 为许多标准组件提供基本功能的库。您可以使用 TFX 库将此功能添加到自己的自定义组件中，也可以单独使用它们。

TFX的核心设计原则包括以下几点：

* 构建可服务于多个学习任务的统一平台

  &ensp; &ensp; 这要求系统具有足够的通用性和可扩展性。（TODO: 展开写）
* 支持持续训练和服务

  &ensp; &ensp;这两个事情看起来简单，但是如果考虑到其中的风险控制和自动化问题发现等细节的话，也并不简单。（TODO: 展开写）
* 人工干预

  &ensp; &ensp;如何优雅地让人参与整个流程，解决机器不好解决的问题，也是一个挑战。（TODO: 展开写）
* 可靠性和稳定性

  &ensp; &ensp;可靠性和稳定性不仅包括服务级别，还包括在数据层面发生问题的时候服务的效果依然可以保持稳定可靠。（TODO: 展开写）


下面的章节主要围绕TFX的核心概念、使用方法、使用示例进行详细描述。
## 6.4.1 TFX 核心概念
* 流水线

  TFX流水线是实现机器学习流水线的一系列组件，这些组件包括数据提取与验证、训练和分析模型、生产环境部署等，这些组件有TFX库构建而成，。
  ![avatar](6-Pipeline/imgs/tfx-arch.png)
* 组件

* 工件
## 6.4.2 TFX 使用方法
* 数据探索、可视化和清理
* 数据可视化
* 模型开发训练
* 模型效果评估
* 模型部署
## 6.4.3 TFX 经典案例
举一个tf的训练到上线的案例

## 参考
https://zhuanlan.zhihu.com/p/31041536
https://zhuanlan.zhihu.com/p/269133610
