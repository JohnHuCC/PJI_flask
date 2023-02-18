from sklearn import ensemble, preprocessing, metrics
from sklearn.model_selection import train_test_split
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

df_data = pd.read_csv(
    '/Users/johnnyhu/Desktop/PJI_Dataset/PJI_all.csv')
PJI_X = df_data[["Age", "Segment", "HGB", "PLATELET", "Serum ", "P.T", "APTT", "Total CCI", "Total Elixhauser",
                "Primary,Revision", "ASA_2", "2X positive culture", "Serum CRP", "Serum ESR", "Synovial WBC",
                 "Single Positive culture", "Synovial_PMN", "Positive Histology", "Purulence"]]
PJI_y = df_data["Group"]

X_train, X_test, y_train, y_test = train_test_split(
    PJI_X, PJI_y, test_size=0.1)
# 建立 random forest 模型

forest = ensemble.RandomForestClassifier(
    n_estimators=50, max_depth=5).fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)
features = X_train.columns
fig, ax = plt.subplots(figsize=(36, 16))
bars = plt.barh(range(len(indices)),
                importances[indices], color='#ffc2c3', align='center')
ax.bar_label(bars, size=20)
plt.title('Feature Importance', size=48)
plt.yticks(range(len(indices)), [features[i] for i in indices], size=24)
plt.xlabel('Relative Importance', size=32)
plt.savefig(
    "static/assets/img/feature_importances.png")
