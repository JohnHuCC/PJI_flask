from sklearn import ensemble, preprocessing, metrics
from sklearn.model_selection import train_test_split
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import xgboost

df_data = pd.read_csv(
    'PJI_Dataset/PJI_train.csv')
PJI_X = df_data[["Age", "Segment", "HGB", "PLATELET", "Serum ", "P.T", "APTT", "Total CCI", "Total Elixhauser Groups per record",
                "Primary,Revision,native hip", "ASA_2", "2X positive culture", "Serum CRP", "Serum ESR", "Synovial WBC",
                 "Single Positive culture", "Synovial_PMN", "Positive Histology", "Purulence"]]
PJI_y = df_data["Group"]
# print(PJI_X)

X_train, X_test, y_train, y_test = train_test_split(
    PJI_X, PJI_y, test_size=0.3)

shap.initjs()
loaded_model = joblib.load('Stacking_model')
X_train_summary = shap.sample(X_train, 15)

explainer = shap.KernelExplainer(
    loaded_model.predict, X_train_summary, keep_index=True)
shap_values = explainer.shap_values(X_train_summary)
# print(shap_values)
shap.summary_plot(shap_values, X_train_summary, show=False)
plt.savefig(
    "PJI_flask/static/assets/img/summary_plot.png")
# shap.force_plot(explainer.expected_value, shap_values,
#                 X_train_summary)
shap.force_plot(
    explainer.expected_value, shap_values, X_train_summary)
plt.savefig(
    "PJI_flask/static/assets/img/force_plot.png")
# shap.force_plot(explainer.expected_value[0], shap_values[0], X_test.iloc[0, :])


# print(loaded_model.score(X_test, y_test))

# explainer = shap.TreeExplainer(loaded_model)
# choosen_instance = X_test.loc[[120]]
# shap_values = explainer.shap_values(choosen_instance)
# shap.initjs()
# shap.force_plot(explainer.expected_value[1], shap_values[1],
#                 choosen_instance, show=False, matplotlib=True)
# plt.savefig("force_plot.png")
