from collections import Counter
from func_timeout import func_set_timeout
import func_timeout
import subprocess
from threading import Timer
import threading
import os
import re
import json
import itertools
from sympy.logic.boolalg import to_dnf
from sympy.logic import simplify_logic, SOPform
from sympy import symbols
from sympy import Symbol
import warnings
from IPython.display import Image
from subprocess import call
import joblib
from operator import itemgetter
import xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from mixed_naive_bayes import MixedNB
from mlxtend.classifier import StackingClassifier
import six
import sys
from sklearn.tree import export_graphviz
import time
import eventlet
from imblearn.over_sampling import SMOTE
import personal_DecisionPath2
sys.modules['sklearn.externals.six'] = six
##
warnings.filterwarnings("ignore")

uncorr_diseases = [
    'Pulmonary Circulation Disorders',
    'Peripheral Vascular Disorders',
    'Hypothyroidism',
    'AIDS/HIV', 'Lymphoma', 'Metastatic Cancer',
    'Weight Loss', 'Fluid and Electrolyte Disorders',
]

combine_disease = [
    'Hypertension Uncomplicated', 'Hypertension Complicated',
    'Paralysis', 'Other Neurological Disorders',
    'Diabetes Uncomplicated', 'Diabetes Complicated',
    'Blood Loss Anemia', 'Deficiency Anemia'
]


def Comorbidity(df):
    # drop 不相關的病症
    # 針對共病症再整併處理
    df.drop(columns=uncorr_diseases, inplace=True)
    df['Hypertension'] = df['Hypertension Uncomplicated'] + \
        df['Hypertension Complicated']
    df['Paralysis or neurological disorders'] = df['Paralysis'] + \
        df['Other Neurological Disorders']
    df['Diabetes mellitus'] = df['Diabetes Uncomplicated'] + \
        df['Diabetes Complicated']
    df['Anemia'] = df['Blood Loss Anemia'] + df['Deficiency Anemia']
    df.drop(columns=combine_disease, inplace=True)
    return df


# In[6]: Main parameter setting
# 主體參數設定
debug_model = 0
pID_idx = 5
start_a = time.time()
explainers_counter = 5  # 找出 n 組候選 explainers
CONDITIOS_AvgFidelity = {}
df = pd.read_excel(
    '/Users/johnnyhu/Desktop/Revision PJI For交大 V9(6月信Validation).xlsx')

df.drop(columns=['Name', 'CTNO', 'CSN',
                 'Turbidity', 'Color'], inplace=True)
df['Laterality '].replace(['R', 'L'], [0, 1], inplace=True)
df['Joint'].replace(['H', 'K'], [0, 1], inplace=True)
# 將'group', 'gender' 資料分為 0, 1 兩類
df.Group.replace(2, 0, inplace=True)
df.Gender.replace(2, 0, inplace=True)

# 6.2 滑膜白細胞酯酶，將內容 "Negative, 1+, 2+, 3+ 及Trace" 轉碼
df['synovial Leukocyte Esterase'].replace(
    ['Negative', '1+', '2+', '3+', 'Trace'], [0, 1, 2, 3, np.nan], inplace=True)

# 6.3 將 {1, 2} 轉碼為 {0, 1}, {3} 與 {na} 後續將因為處理空值 'Total Score', '2nd ICM' 會被刪除
# df['Primary, Revision\nnative hip'].value_counts(), 可顯示資料統計
# Primary, Revision\nnative hip {1, 2}
df['Primary, Revision\nnative hip'].replace(2, 0, inplace=True)
if (debug_model == 1):
    print(df.shape)

# 6.4 刪除['Total Score', '2nd ICM']空值記錄後,剩餘的感染與非感染的病患比例
# MM = df[feature_selection2]
# 將有空值的記錄刪除
df = df.dropna(subset=['Total Score', '2nd ICM']).reset_index(drop=True)
if (debug_model == 1):
    print(df.shape)

pd.set_option('display.max_columns', None)
# plt.style.use('ggplot')

t_ = df['Group'].value_counts().sort_index()

# 6.5 重新修訂 '2nd ICM' 數值
# np.where(condition, x, y) # 滿足條件(condition)，輸出x，不滿足輸出y。
# 計算 'total score'
df['Total Score'] = np.where((df['Serum CRP'] >= 10) | (df['D_dimer'] >= 860), 2, 0) + \
    np.where(df['Serum ESR'] >= 30, 1, 0) + \
    np.where((df['Synovial WBC'] >= 3000) | (df['synovial Leukocyte Esterase'] >= 2), 3, 0) + \
    np.where(df['Synovial Neutrophil'] >= 70, 2, 0) + \
    np.where(df['Single Positive culture'] == 1, 2, 0) + \
    np.where(df['Positive Histology'] == 1, 3, 0) + \
    np.where(df['Pulurence'] == 1, 3, 0)

# 6.6 重新修訂 2018 ICM: (1) >= 6 Infected, (2) 2-5 Possibly Infected, (3) 0-1 Not Infected
df['2nd ICM'] = np.where(df['Total Score'] >= 6, 1, 0)

# 6.7 修訂 2018 ICM 欄位, '2X positive cultures' =1 or Sinus Tract = 1 的患者 '2nd ICM'  = 1
df.loc[(df['2X positive culture'] == 1) | (
    df['Sinus Tract'] == 1), '2nd ICM'] = 1

# 6.8 刪除 missing rate < 0.619的 cols, 並 繪製圖表與列出 drop.cols_list
THRESHOLD = 0  # for missing rate: 200 / 323 = 0.619

# 6.9 忽略 df['cols'] 索引:32 以後的 cols，同時統計每一個 col 'notnull' 的個數
# 並列表為 table
table = df.notnull().sum()[:-32]  # 不看綜合病症

if (debug_model == 1):
    print("Columns should be drop out: \n{}".format(
        table[table.values < THRESHOLD].index.tolist()))
df.drop(columns=table[table.values <
        THRESHOLD].index.tolist(), inplace=True)

temp_col = [
    'No. ', 'Group',
    'PJI/Revision Date',
    'Total Score', '2nd ICM',
    'Minor ICM Criteria Total', '1st ICM',
    'Minor MSIS Criteria Total', 'MSIS final classification'
]

# 6.10 補值前處理：
# a. 把可能造成overfitting 的 outcome (temp_col) 先移除，再補值
# b. 補值後再行合併
# c. df.copy(), 複製此對象的索引和數據, 同時 reset_index, 重組index,避免產生莫名的邏輯錯誤
no_group = list(df['No.Group'])
internal = df.copy().reset_index(drop=True)
internal_temp_df = internal[temp_col].copy()
internal.drop(columns=['Date of first surgery ',
                       'Date of last surgery '] + temp_col, inplace=True)

if (debug_model == 1):
    print(internal.shape)
internal.tail()

# d. MICE 補值，based estimator 為 BayesianRidge
imputer_bayes = IterativeImputer(estimator=BayesianRidge(),
                                 max_iter=50,
                                 random_state=0)

# imputation by bayes
imputer_bayes.fit(internal)
impute_internal = pd.DataFrame(
    data=imputer_bayes.transform(internal),
    columns=internal.columns
)

float_col = [
    'Height (m)', 'BW (kg)',
    'BMI', 'Serum WBC  (10¬3/uL)',
    'HGB (g/dL)', 'Serum CRP (mg/L)',
    'CR(B)  (mg/dL)'
]

# 6.11 補值後之資料修補
# a: 負數轉正
# b: 將"float" cols 轉換為 int
# c: 修正 BMI
# d: 修正'Total Elixhauser Groups per record',  往前推31個欄位加總
# e: concat程序: 補值後再與 'temp_col' cols 合併
impute_internal = impute_internal.abs()
impute_internal[impute_internal.columns[~np.isin(impute_internal.columns, float_col)]] = impute_internal[
    impute_internal.columns[~np.isin(impute_internal.columns, float_col)]].round(0).astype(int)

impute_internal['BMI'] = impute_internal['BW (kg)'] / (
    impute_internal['Height (m)'] * impute_internal['Height (m)'])
impute_internal['Total Elixhauser Groups per record'] = impute_internal[impute_internal.columns[-32:-1]
                                                                        ].sum(axis=1)
impute_internal = pd.concat(
    [internal_temp_df, impute_internal], sort=False, ignore_index=False, axis=1)
impute_internal.tail()

# 6.12 將各項資料屬性，按cols進行分類
s1 = {
    # [0, 1]
    'check_Primary': ['Primary, Revision\nnative hip'],

    # [1, 2, 3, 4, 5]   #ps. 只有[1, 2, 3, 4]
    'check_ASA': ['ASA'],

    # [0, 1] : Outlier > 1 or Outlier < 0
    'check_label': [
        'Laterality ', 'Joint', 'Gender', 'Positive Histology',
        '2X positive culture', 'Sinus Tract', 'Pulurence', 'Single Positive culture',
        'Congestive Heart Failure', 'Cardiac Arrhythmia', 'Valvular Disease',
        'Pulmonary Circulation Disorders', 'Peripheral Vascular Disorders',
        'Hypertension Uncomplicated', 'Hypertension Complicated', 'Paralysis',
        'Other Neurological Disorders', 'Chronic Pulmonary Disease',
        'Diabetes Uncomplicated', 'Diabetes Complicated', 'Hypothyroidism',
        'Renal Failure', 'Liver Disease',
        'Peptic Ulcer Disease excluding bleeding', 'AIDS/HIV', 'Lymphoma',
        'Metastatic Cancer', 'Solid Tumor without Metastasis',
        'Rheumatoid Arthritis/collagen', 'Coagulopathy', 'Obesity',
        'Weight Loss', 'Fluid and Electrolyte Disorders', 'Blood Loss Anemia',
        'Deficiency Anemia', 'Alcohol Abuse', 'Drug Abuse', 'Psychoses',
        'Depression'
    ],

    # [numeric data] : Outlier > Q3 + 1.5(IQR) or Outlier < Q1 - 1.5(IQR)
    'check_numeric': [
        'Age', 'Height (m)', 'BW (kg)', 'BMI', 'Serum ESR',
        'Serum WBC ', 'HGB', 'PLATELET', 'P.T', 'APTT',
        'Serum CRP', 'CR(B)', 'AST', 'ALT', 'Synovial WBC',
        'Total CCI'
    ],

    # [percentage data] : Outlier > 100
    'percentage_data': ['Segment (%)', 'Synovial Neutrophil']
}

# 6.13 修正
# a. 將ASA儲存為 str, for OneHotEncoder
# b. 將 'Synovial Neutrophil' 上限設100
impute_internal['ASA'] = impute_internal['ASA'].astype(str)
impute_internal.loc[impute_internal['Synovial Neutrophil']
                    > 100, 'Synovial Neutrophil'] = 100

# 6.14 補值後 drop 'outcome' 以外的其他cols 做為建模的基礎
internal_X, internal_y = impute_internal[impute_internal.columns[9:]
                                         ], impute_internal['Group']
print('impute_intermal:')
print(internal_X)
# print(
#     pd.Index(list(impute_internal.columns[0])+list(impute_internal.columns[10:])))
# internal_y.value_counts().sort_index().plot(
#     kind='bar', color=['r', 'b'], title="Training dataset", rot=0)
# plt.show()

# 6.15 Comorbidity 程序處理
# 先副本處理再寫回原變數
internal_X = Comorbidity(internal_X.copy())

# 5.16 將ASA 轉為 OneHotEncoder格式, 並整合至 internal_X
ohe = OneHotEncoder(handle_unknown='ignore')
ohe.fit(internal_X[['ASA']].values)
ASA_data = ohe.transform(internal_X[['ASA']].values).toarray().astype(int)
for idx, class_name in enumerate(ohe.categories_[0]):
    internal_X['ASA_{}'.format(class_name)] = ASA_data[:, idx]
internal_X.drop(columns=['ASA'], inplace=True)
internal_X.tail()

# 6.16 修正 columns name
internal_X = internal_X.rename(columns={"Synovial Neutrophil": "Synovial_PMN",
                                        "Pulurence": "Purulence"})

# In[8]: feature_selection2 by feature importance from PJI-PI-01-02-2021.docx (Table 4)
feature_selection2 = [
    'No.Group',
    'Age',
    'Segment (%)',  # Neutrophil Segment'
    'HGB',          # Hemoglobin
    'PLATELET',
    'Serum WBC ',
    'P.T',          # Prothrombin Time
    'APTT',         # Activated Partial Thromboplastin Time
    'Total CCI',    # Charlson Comorbidity Index
    'Total Elixhauser Groups per record',  # Elixhauser Comorbidity Index
    # Surgery (primary/revision), category
    'Primary, Revision\nnative hip',
    'ASA_2',        # American Society of Anesthesiologists, category
    '2X positive culture',
    'Serum CRP',
    'Serum ESR',
    'Synovial WBC',
    'Single Positive culture',  # category
    'Synovial_PMN',  # 'Synovial Neutrophil',
    'Positive Histology',  # category
    'Purulence',    # 'Pulurence' , category
]
end_a = time.time()
### 補充:  iloc vs loc 功能說明 ###
# iloc，即index locate 用index索引進行定位，所以引數是整型，如：df.iloc[10:20, 3:5]
# loc，則可以使用column名和index名進行定位，如：df.loc[‘image1’:‘image10’, ‘age’:‘score’]
internal_X = internal_X.loc[:, feature_selection2].copy()
df_X = internal_X[0:50]
df_X1 = internal_X[79:89]
frames = [df_X, df_X1]
frames_x = pd.concat(frames)
df_y = internal_y[0:50]
df_y1 = internal_y[79:89]
frames = [df_y, df_y1]
frames_y = pd.concat(frames)
frames_x = frames_x.reset_index().drop("index", axis=1)
frames_y = frames_y.reset_index().drop("index", axis=1)
pretime = end_a - start_a
print("Preprocessing Start time = {}".format(start_a))
print("Preprocessing End time = {}".format(end_a))
print("Preprocessing  time = {}".format(pretime))

y = frames_y["Group"]
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(frames_x, y)
counter = Counter(y_res)
print(X_res)
X_res.to_csv('New_data_x_test.csv', encoding='utf-8', index=False)
y_res.to_csv('New_data_y_test.csv', encoding='utf-8', index=False)
internal_X.to_csv('internal_x_test.csv', encoding='utf-8', index=False)
internal_y.to_csv('internal_y_test.csv', encoding='utf-8', index=False)