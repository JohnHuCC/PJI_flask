from sklearn import ensemble, preprocessing, metrics
from sklearn.model_selection import train_test_split
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

df_data = pd.read_csv(
    '/Users/johnnyhu/Desktop/PJI_Dataset/PJI_train.csv')
PJI_X = df_data[["Age", "Segment", "HGB", "PLATELET", "Serum ", "P.T", "APTT", "Total CCI", "Total Elixhauser Groups per record",
                "Primary,Revision,native hip", "ASA_2", "2X positive culture", "Serum CRP", "Serum ESR", "Synovial WBC",
                 "Single Positive culture", "Synovial_PMN", "Positive Histology", "Purulence"]]
PJI_y = df_data["Group"]

X_train, X_test, y_train, y_test = train_test_split(
    PJI_X, PJI_y, test_size=0.3)
print(X_test)
# 建立 random forest 模型
forest = ensemble.RandomForestClassifier(
    n_estimators=50, max_depth=5).fit(X_train, y_train)
print(forest.feature_importances_)
importances = forest.feature_importances_
indices = np.argsort(importances)
features = X_train.columns
plt.figure(figsize=(20, 16))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig(
    "/Users/johnnyhu/Desktop/PJI_flask/static/assets/img/feature_importances.png")
