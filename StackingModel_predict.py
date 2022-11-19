# In[1]: Import Library
# 引用適當的 Library
##
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
sys.modules['sklearn.externals.six'] = six
##
warnings.filterwarnings("ignore")

# In[6]: Main parameter setting
# 主體參數設定
debug_model = 0
Explainer_depth = 12  # The depth of Explainer Model
pID_idx = 5
pID = [11, 212, 51, 210, 79, 159]
PID = pID[pID_idx]
explainers_counter = 5  # 找出 n 組候選 explainers
CONDITIOS_AvgFidelity = {}
# 根據ground_truth & Meta Learner 調節
if pID_idx >= 3:
    rule_1, rule_2 = 0, 1  # ground_truth: N, Meta: N
else:
    rule_1, rule_2 = 1, 0  # ground_truth: I, Meta: I

# In[7]: File reading and pre-processing
# 6.1 讀檔與前處理作業
df = pd.read_excel(
    '/Users/johnnyhu/Desktop/Revision PJI For交大 V9(6月信Validation).xlsx')
df.drop(columns=['Name', 'CTNO', 'CSN', 'Turbidity', 'Color'], inplace=True)
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
df.tail()

# 6.4 刪除['Total Score', '2nd ICM']空值記錄後,剩餘的感染與非感染的病患比例
# MM = df[feature_selection2]
# 將有空值的記錄刪除
df = df.dropna(subset=['Total Score', '2nd ICM']).reset_index(drop=True)
if (debug_model == 1):
    print(df.shape)

pd.set_option('display.max_columns', None)
# plt.style.use('ggplot')

t_ = df['Group'].value_counts().sort_index()
# t_.plot.bar(rot=0, color=['r', 'b'], alpha=0.7, fontsize=14)
# for idx, v_ in enumerate(t_):
#     plt.text(idx - 0.07, v_ - 35, "{}".format(v_), fontsize=14)

# plt.xticks([0, 1], ['Non infection', 'infection'], fontsize=14)
# plt.xlabel("Group", fontsize=14)
# plt.ylabel("Patient", fontsize=14)
# plt.show()

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
THRESHOLD = 200  # for missing rate: 200 / 323 = 0.619

# 6.9 忽略 df['cols'] 索引:32 以後的 cols，同時統計每一個 col 'notnull' 的個數
# 並列表為 table
table = df.notnull().sum()[:-32]  # 不看綜合病症

if (debug_model == 1):
    print("Columns should be drop out: \n{}".format(
        table[table.values < THRESHOLD].index.tolist()))
df.drop(columns=table[table.values < THRESHOLD].index.tolist(), inplace=True)


temp_col = [
    'No.Group', 'No. ', 'Group',
    'PJI/Revision Date',
    'Total Score', '2nd ICM',
    'Minor ICM Criteria Total', '1st ICM',
    'Minor MSIS Criteria Total', 'MSIS final classification'
]

# 6.10 補值前處理：
# a. 把可能造成overfitting 的 outcome (temp_col) 先移除，再補值
# b. 補值後再行合併
# c. df.copy(), 複製此對象的索引和數據, 同時 reset_index, 重組index,避免產生莫名的邏輯錯誤
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
###
# Review for profiles
#
# 目前送交論文的版本為插補後四捨五入的版本 (未還原原始資料)。
# 若分析時需比對插補資料與四捨五入前後的差異，可將必要欄位進行備份，待四捨五入後再行回復
# [Line: 321-329, Line: 332-340]
impute_internal = impute_internal.abs()
# Height_ = impute_internal['Height (m)']
# BW_ = impute_internal['BW (kg)']
# SerumWBC_ = impute_internal['Serum WBC ']
# HGB_ = impute_internal['HGB']
# SerumCRP_ = impute_internal['Serum CRP']
# CR_ = impute_internal['CR(B)']
# Segment_ = impute_internal['Segment (%)']
# PT_ = impute_internal['P.T']
# APTT_ = impute_internal['APTT']
impute_internal[impute_internal.columns[~np.isin(impute_internal.columns, float_col)]] = impute_internal[
    impute_internal.columns[~np.isin(impute_internal.columns, float_col)]].round(0).astype(int)
# impute_internal['Height (m)'] = Height_
# impute_internal['BW (kg)'] = BW_
# impute_internal['Serum WBC '] = SerumWBC_
# impute_internal['HGB'] = HGB_
# impute_internal['Serum CRP'] = SerumCRP_
# impute_internal['CR(B)'] = CR_
# impute_internal['Segment (%)'] = Segment_
# impute_internal['P.T'] = PT_
# impute_internal['APTT'] = APTT_
# Review for profiles

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
internal_X, internal_y = impute_internal[impute_internal.columns[10:]
                                         ], impute_internal['Group']
internal_y.value_counts().sort_index().plot(
    kind='bar', color=['r', 'b'], title="Training dataset", rot=0)
# plt.show()

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
    'Age',
    'Segment (%)',  # Neutrophil Segment'
    'HGB',          # Hemoglobin
    'PLATELET',
    'Serum WBC ',
    'P.T',          # Prothrombin Time
    'APTT',         # Activated Partial Thromboplastin Time
    'Total CCI',    # Charlson Comorbidity Index
    'Total Elixhauser Groups per record',  # Elixhauser Comorbidity Index
    'Primary, Revision\nnative hip',  # Surgery (primary/revision), category
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

### 補充:  iloc vs loc 功能說明 ###
# iloc，即index locate 用index索引進行定位，所以引數是整型，如：df.iloc[10:20, 3:5]
# loc，則可以使用column名和index名進行定位，如：df.loc[‘image1’:‘image10’, ‘age’:‘score’]
internal_X = internal_X.loc[:, feature_selection2].copy()
if (debug_model == 1):
    print(internal_X.shape)
internal_X.tail()
internal_X.columns

# 7.1 Get the specific patient profile by PID
X_train, y_train = internal_X.drop(index=PID), internal_y.drop(index=PID)

X_test, y_test = internal_X.iloc[PID:PID + 1], internal_y.iloc[PID:PID + 1]

if (debug_model == 1):
    print("Tr shape: {}, Ts shape: {}".format(X_train.shape, X_test.shape))


# 7.2 Split dataset to tr (80%) and val (20%)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=666, shuffle=True)

# df_x_train_sp = pd.DataFrame(X_tr)
# df_x_validation_sp = pd.DataFrame(X_val)
# df_y_train_sp = pd.DataFrame(y_tr)
# df_y_validation_sp = pd.DataFrame(y_val)

# df_x_train_sp.to_csv(
#     r"/Users/johnnyhu/Desktop/PJI_Dataset/PJI_x_train_sp.csv", index=False, sep=',')
# df_x_validation_sp.to_csv(
#     r"/Users/johnnyhu/Desktop/PJI_Dataset/PJI_x_val_sp.csv", index=False, sep=',')
# df_y_train_sp.to_csv(
#     r"/Users/johnnyhu/Desktop/PJI_Dataset/PJI_y_train_sp.csv", index=False, sep=',')
# df_y_validation_sp.to_csv(
#     r"/Users/johnnyhu/Desktop/PJI_Dataset/PJI_y_validation_sp.csv", index=False, sep=',')

if (debug_model == 1):
    print("Val shape: {}".format(X_val.shape))

# ### 7.3 Plot the Decision tree grpah
##
# from sklearn import tree
# from sklearn.externals.six import StringIO
# from sklearn.tree import DecisionTreeClassifier
# import os
# import pydotplus

# clf = DecisionTreeClassifier(random_state=123, max_depth=5)
# dt = clf.fit(X_tr, y_tr)
# os.environ["PATH"] += os.pathsep + 'C:/Graphviz/bin'
# dot_data = StringIO()
# tree.export_graphviz(dt #模型
#                     ,feature_names=X_tr.columns  #tez
#                     ,class_names=["Infection","Not infected"] #類別名
#                     ,filled=True    #由顏色標識不純度
#                     ,rounded=True   #樹節點為圓角矩形
#                     ,out_file=dot_data
#                 )
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("./output/case" + str(pID_idx+1) + "_dt.pdf")

# In[9]: Stacking Modeling
# 8.1 Construct Base Classifier
xgb = xgboost.XGBClassifier(learning_rate=0.01, max_depth=3, n_estimators=50, scale_pos_weight=0.85,
                            enable_experimental_json_serialization=True, tree_method='hist', random_state=0)
rf = RandomForestClassifier(n_estimators=50, random_state=123, max_depth=5)
nb_pipe = MixedNB(categorical_features=[9, 10, 11, 15, 17, 18])
lr_pipe = make_pipeline(StandardScaler(), LogisticRegression(
    solver='lbfgs', max_iter=500, random_state=123))
svc_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(probability=True, random_state=123, C=10, gamma=0.01))
])

model_xgb = xgb.fit(X_tr.values, y_tr.values)
xgb_predict = model_xgb.predict(X_val)

# 8.2 Stacking Model from 80% dataset   ?
stacking_model = StackingClassifier(
    classifiers=[xgb, rf, lr_pipe, nb_pipe],
    use_probas=True,
    average_probas=True,
    use_features_in_secondary=True,
    meta_classifier=svc_pipe
)

# df_x_train = pd.DataFrame(X_train)
# df_y_train = pd.DataFrame(y_train)
# # print(df_x_train.columns.values.tolist())
# df_x_train.to_csv(r"C:\Users\JohnnyHu\Desktop\PJI_x_train.csv",
#                   index=False, sep=',')
# # print(df_y_train)
# df_y_train.to_csv(r"C:\Users\JohnnyHu\Desktop\PJI_y_train.csv",
#                   index=False, sep=',')

# 8.3 Stacking Model from 100% dataset
stacking_model.fit(X_train.values, y_train.values)
joblib.dump(stacking_model, 'Stacking_model')

loaded_model = joblib.load('Stacking_model')

print(loaded_model.predict(X_test))

target = ['0', '1']
