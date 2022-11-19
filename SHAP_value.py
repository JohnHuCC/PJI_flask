from sklearn import ensemble, preprocessing, metrics
from sklearn.model_selection import train_test_split
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import xgboost

df_data = pd.read_csv(
    '/Users/johnnyhu/Desktop/PJI_Dataset/PJI_train.csv')

# PJI_X = pd.DataFrame([df_data["Age"],
#                       df_data["Segment"],
#                       df_data["HGB"],
#                       df_data["PLATELET"],
#                       df_data["Serum "],
#                       df_data["P.T"],
#                       df_data["APTT"],
#                       df_data["Total CCI"],
#                       df_data["Total Elixhauser Groups per record"],
#                       df_data["Primary,Revision,native hip"],
#                       df_data["ASA_2"],
#                       df_data["2X positive culture"],
#                       df_data["Serum CRP"],
#                       df_data["Serum ESR"],
#                       df_data["Synovial WBC"],
#                       df_data["Single Positive culture"],
#                       df_data["Synovial_PMN"],
#                       df_data["Positive Histology"],
#                       df_data["Purulence"],
#                       ]).T
PJI_X = df_data[["Age", "Segment", "HGB", "PLATELET", "Serum ", "P.T", "APTT", "Total CCI", "Total Elixhauser Groups per record",
                "Primary,Revision,native hip", "ASA_2", "2X positive culture", "Serum CRP", "Serum ESR", "Synovial WBC",
                 "Single Positive culture", "Synovial_PMN", "Positive Histology", "Purulence"]]
PJI_y = df_data["Group"]
# print(PJI_X)

X_train, X_test, y_train, y_test = train_test_split(
    PJI_X, PJI_y, test_size=0.3)
# print(X_test)
# print(y_test)
# 建立 random forest 模型
# forest = ensemble.RandomForestClassifier(
#     n_estimators=50, max_depth=5).fit(X_train, y_train)
# # print(forest.feature_importances_)
# importances = forest.feature_importances_
# indices = np.argsort(importances)
# features = X_train.columns
# plt.figure(figsize=(6, 8))
# plt.title('Feature Importances')
# plt.barh(range(len(indices)), importances[indices], color='b', align='center')
# plt.yticks(range(len(indices)), [features[i] for i in indices])
# plt.xlabel('Relative Importance')

# plt.savefig("feature_Importances.png")
# plt.show()

# import shap
# import sklearn
# from sklearn.model_selection import train_test_split

# X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
# print(X_train)
# print(Y_train)

# knn = sklearn.neighbors.KNeighborsClassifier()
# knn.fit(X_train, Y_train)

# svc_linear = sklearn.svm.SVC(kernel='linear', probability=True)
# svc_linear.fit(X_train, Y_train)

# rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
# stacking_model = sklearn.ensemble.StackingClassifier([('knn', knn), ('svc', svc_linear)], rf)
# stacking_model.fit(X_train, Y_train)
# X, y = shap.datasets.adult()

# model = xgboost.XGBClassifier()
# model.fit(X, y)
# print(X[:100])
# print(y)
# print(X_train)
# print(y_train)
# explainer = shap.explainers.Exact(model.predict_proba, X)
# shap_values = explainer(X[:100])
# print(shap_values)

shap.initjs()
loaded_model = joblib.load('XGB_model')
explainer = shap.explainers.Exact(loaded_model.predict_proba, X_train)
# explainer = shap.KernelExplainer(loaded_model, X_train)
shap_values = explainer(X_train[:100])
print(shap_values)
# shap.force_plot(explainer.expected_value[0], shap_values[0], X_test.iloc[0, :])
# plt.savefig("force_plot.png")
# X_temp = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
#            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]]


# print(loaded_model.score(X_test, y_test))

# explainer = shap.TreeExplainer(loaded_model)
# choosen_instance = X_test.loc[[120]]
# shap_values = explainer.shap_values(choosen_instance)
# shap.initjs()
# shap.force_plot(explainer.expected_value[1], shap_values[1],
#                 choosen_instance, show=False, matplotlib=True)
# plt.savefig("force_plot.png")
