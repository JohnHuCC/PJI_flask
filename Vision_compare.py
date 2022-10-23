import numpy as np
import pandas as pd
import plotly.express as px
import joblib
from sklearn import ensemble, preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

df_data = pd.read_csv(
    '/Users/johnnyhu/Desktop/PJI_Dataset/PJI_train.csv')

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
    PJI_X, PJI_y, test_size=0.3)

# 建立 random forest 模型
forest = ensemble.RandomForestClassifier(n_estimators=50, max_depth=5)
forest_fit = forest.fit(train_X, train_y)
joblib.dump(forest, 'RF_model')

# arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


def tran_df(arr):
    predict_data = pd.DataFrame(
        np.array([arr]), columns=['Age', 'Segment', 'HGB', 'PLATELET', 'Serum ', 'P.T', 'APTT', 'Total CCI', 'Total Elixhauser Groups per record', 'Primary,Revision,native hip', 'ASA_2',
                                  '2X positive culture', 'Serum CRP', 'Serum ESR', 'Synovial WBC', 'Single Positive culture', 'Synovial_PMN', 'Positive Histology', 'Purulence'
                                  ])
    print(predict_data)
    return predict_data


# predict_data = tran_df(arr)


def rf_predict(df):
    loaded_model = joblib.load('RF_model')
    result = loaded_model.predict(df)
    print(result)
    return result


def plt_con():
    y_true = test_y
    y_pred = rf_predict(test_X)
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
    plt.savefig('confusion_matrix.png')

# plt_con(test_y, rf_predict(test_X))
# rf_predict(predict_data)
