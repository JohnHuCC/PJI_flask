import numpy as np
import pandas as pd
import plotly.express as px
import joblib
from sklearn import ensemble, preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import matplotlib

import platform
if platform.system() == 'Darwin':
    matplotlib.use("tkAgg")

import matplotlib.pyplot as plt

df_data = pd.read_csv(
    'PJI_Dataset/PJI_train.csv')

PJI_X = pd.DataFrame([df_data["Age"],
                      df_data["Segment"],
                      df_data["HGB"],
                      df_data["PLATELET"],
                      df_data["Serum "],
                      df_data["P.T"],
                      df_data["APTT"],
                      df_data["Total CCI"],
                      df_data["Total Elixhauser Groups per record"],
                      df_data["Primary,Revision,native hip"],
                      df_data["ASA_2"],
                      df_data["2X positive culture"],
                      df_data["Serum CRP"],
                      df_data["Serum ESR"],
                      df_data["Synovial WBC"],
                      df_data["Single Positive culture"],
                      df_data["Synovial_PMN"],
                      df_data["Positive Histology"],
                      df_data["Purulence"],
                      ]).T
PJI_y = df_data["Group"]

train_X, test_X, train_y, test_y = train_test_split(
    PJI_X, PJI_y, test_size=0.2)


def tran_df(arr):
    predict_data = pd.DataFrame(
        np.array([arr]), columns=['Age', 'Segment (%)', 'HGB', 'PLATELET', 'Serum WBC ', 'P.T', 'APTT', 'Total CCI', 'Total Elixhauser Groups per record', 'Primary, Revision\nnative hip', 'ASA_2',
                                  '2X positive culture', 'Serum CRP', 'Serum ESR', 'Synovial WBC', 'Single Positive culture', 'Synovial_PMN', 'Positive Histology', 'Purulence'
                                  ])
    return predict_data


def stacking_predict(df):
    loaded_model = joblib.load('Stacking_model')
    result = loaded_model.predict(df)
    print(result)
    return result


def xgboost_predict(df):
    loaded_model = joblib.load('Xgboost_model')
    result = loaded_model.predict(df)
    print(result)
    return result


def rf_predict(df):
    loaded_model = joblib.load('RandomForest_model')
    result = loaded_model.predict(df)
    print(result)
    return result


def nb_predict(df):
    loaded_model = joblib.load('NaiveBayes_model')
    result = loaded_model.predict(df)
    print(result)
    return result


def lr_predict(df):
    loaded_model = joblib.load('LogisticRegression_model')
    result = loaded_model.predict(df)
    print(result)
    return result


def plt_con():
    y_true = test_y
    y_pred = stacking_predict(test_X)
    mat_con = (confusion_matrix(y_true, y_pred, labels=[0, 1]))

    # Setting the attributes
    fig, px = plt.subplots(figsize=(7.5, 7.5))
    px.matshow(mat_con, cmap=plt.cm.YlOrRd, alpha=0.5)
    for m in range(mat_con.shape[0]):
        for n in range(mat_con.shape[1]):
            px.text(x=m, y=n, s=mat_con[m, n],
                    va='center', ha='center', size='xx-large')

    # Sets the labels
    # plt.show()
    plt.xlabel('Predictions', fontsize=16)
    plt.ylabel('Actuals', fontsize=16)
    plt.title('Confusion Matrix', fontsize=15)
    plt.savefig('static/assets/img/confusion_matrix.jpg')
