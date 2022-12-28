#!/usr/bin/env python
### 
# @1 Library 引用注意事項
#
# mlxtend.__version__, 0.13.0 家承 (0.18.0)
# sklearn. 0.21.3 家承 (0.23.2)
###
# @2 本檔將以下檔案進行合併。
# 
# 2.1 ./PJI_20210128/main.1.20210103.case*.py 
# 2.2 ./PJI_20210108/dnf_case1_trace.000_20210108.py 
# ---------------------------------------------------
# 2.1.1 蒐集與 stacking 相同答案的 Decision Tree, τ。
# 2.1.1 計算 τ 的 fidelity_scores
# ---------------------------------------------------
# 2.2.1 計算 decision_path of τ 的 fidelity_scores
# 請注意，2.2.1 fidelity_scores of decision_path 與 fidelity_scores of τ 不同
###
#
#
#

# In[1]: Import Library
# 引用適當的 Library 
##
import six
import sys
sys.modules['sklearn.externals.six'] = six
from mlxtend.classifier import StackingClassifier
##
from mixed_naive_bayes import MixedNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.utils import resample
from operator import itemgetter 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost


# In[2]: Collect Decision Tree with the same prediction as the stacking model
### Description: Collect Decision Tree with the same prediction as the stacking model
def getCandidate(X_train, y_train, test_X, stacking_model, Explainer_depth,
                 explainer_count):
    """
    input: 
        X_train, X of training data of the LOO
        y_train, y of training data of the LOO
        test_X, should be dataframe like. X of test data of the LOO
        stacking_model,   
        Explainer_depth, the depth of the Explainer
        explainer_count, the number of randomly generated explainers
    output:
        explainers, five candidate explainers (RF)
        tree_candidates, the tree_candidates of the five candidate explainers
    """
    explainers = {}
    tree_candidates = {}
    while_flag = 0 # the flag of available explainer
    i = 0          # the index of explainer
    while (True):
        i = i + 1
        if while_flag >= explainer_count:
            break
        
        explainers[i-1] = RandomForestClassifier(max_depth=Explainer_depth,
                                               n_estimators=100, random_state=i-1)
        explainers[i-1].fit(X_train, y_train)
        stacking_pred = stacking_model.predict(test_X)
        explainers_pred = explainers[i-1].predict(test_X)
        
        if stacking_pred == explainers_pred:
            # print("(stacking_pred == explainers_pred) + 1")
            tree_candidate = []
            for tree_idx in range(explainers[i-1].n_estimators):
                tree_model = explainers[i-1].estimators_[tree_idx]
                tree_pred = tree_model.predict(test_X)
                # print (stacking_pred == tree_pred)
                if stacking_pred == tree_pred:
                    # print("(stacking_pred == tree_pred) + 1")
                    tree_candidate.append(tree_idx)
            tree_candidates[i-1] = tree_candidate
            while_flag = while_flag + 1
        else:
            del explainers[i-1] # 不存在當下可解釋器，刪除指標
            continue

    return explainers, tree_candidates

# In[3]: Enumerate the decision path of the specific sample
# 列舉特定樣本的決策路徑
def interpret(sample, estimator, feature_names):
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    # Initialize
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    X_test = sample[feature_names].values.reshape(1, -1)
    result = {}
    result['prediction'] = estimator.predict(X_test)
    node_indicator = estimator.decision_path(X_test)
    leave_id = estimator.apply(X_test)
    sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]
    result['info'] = []

    for node_id in node_index:
        if leave_id[sample_id] == node_id:
            continue

        if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"
        result['info'].append([feature_names[feature[node_id]], threshold_sign, round(threshold[node_id], 2)])

    return result

# In[4]: Comorbidity analysis
def Comorbidity(df):
    # drop 不相關的病症
    # 針對共病症再整併處理
    df.drop(columns=uncorr_diseases, inplace=True)
    df['Hypertension'] = df['Hypertension Uncomplicated'] + df['Hypertension Complicated']
    df['Paralysis or neurological disorders'] = df['Paralysis'] + df['Other Neurological Disorders']
    df['Diabetes mellitus'] = df['Diabetes Uncomplicated'] + df['Diabetes Complicated']
    df['Anemia'] = df['Blood Loss Anemia'] + df['Deficiency Anemia']
    df.drop(columns=combine_disease, inplace=True)
    return df

# In[5]: getTopN_Fidelity
def getTopN_Fidelity(fidelity_list, top_N_indices, top_N):
    # fidelity_list, candidate_decision_path & the fidelity
    # top_N_indices: the indices of top_N of the decision path.
    fidelity_list_ = []
    rules_list_ = []
    for i in range(len(top_N_indices)):
        fidelity_list_ = fidelity_list_ + fidelity_list[top_N_indices[i], 'fidelity']
        rules_list_    = rules_list_ + fidelity_list[top_N_indices[i], 'rules']
        
    top_n_fidelity_i = sorted(range(len(fidelity_list_)), key = lambda k : fidelity_list_[k])[-top_N:]
    
    # fidelity_list & decision_path
    return list(top_n_fidelity_i), \
           list(itemgetter(*top_n_fidelity_i)(fidelity_list_)), \
           list(itemgetter(*top_n_fidelity_i)(rules_list_))

# In[6]: Main parameter setting
# 主體參數設定
debug_model = 0
Explainer_depth = 12 # The depth of Explainer Model
pID_idx = 5
pID = [11, 212, 51, 210, 79, 159]
PID = pID[pID_idx]
explainers_counter = 5 # 找出 n 組候選 explainers 
topN = 13              # top_N of fidelity of decision path of candidate explainers 
CONDITIOS_AvgFidelity = {}
## 根據ground_truth & Meta Learner 調節
if pID_idx >= 3:
    rule_1, rule_2 = 0, 1  # ground_truth: N, Meta: N
else:
    rule_1, rule_2 = 1, 0  # ground_truth: I, Meta: I

# In[7]: File reading and pre-processing
### 6.1 讀檔與前處理作業
df = pd.read_excel('./PJI/Revision PJI For交大 V9(6月信Validation).xlsx')
df.drop(columns=['Name', 'CTNO', 'CSN', 'Turbidity', 'Color'], inplace=True)
df['Laterality '].replace(['R', 'L'], [0, 1], inplace=True)
df['Joint'].replace(['H', 'K'], [0, 1], inplace=True)
# 將'group', 'gender' 資料分為 0, 1 兩類
df.Group.replace(2, 0, inplace=True)
df.Gender.replace(2, 0, inplace=True)

### 6.2 滑膜白細胞酯酶，將內容 "Negative, 1+, 2+, 3+ 及Trace" 轉碼
df['synovial Leukocyte Esterase'].replace(['Negative', '1+', '2+', '3+', 'Trace'], [0, 1, 2, 3, np.nan], inplace=True)

### 6.3 將 {1, 2} 轉碼為 {0, 1}, {3} 與 {na} 後續將因為處理空值 'Total Score', '2nd ICM' 會被刪除
# df['Primary, Revision\nnative hip'].value_counts(), 可顯示資料統計
# Primary, Revision\nnative hip {1, 2}
df['Primary, Revision\nnative hip'].replace(2, 0, inplace=True)
if (debug_model == 1):
    print(df.shape)
df.tail()

### 6.4 刪除['Total Score', '2nd ICM']空值記錄後,剩餘的感染與非感染的病患比例
# MM = df[feature_selection2]
# 將有空值的記錄刪除
df = df.dropna(subset=['Total Score', '2nd ICM']).reset_index(drop=True)
if (debug_model == 1):
    print(df.shape)

pd.set_option('display.max_columns', None)
plt.style.use('ggplot')

t_ = df['Group'].value_counts().sort_index()
t_.plot.bar(rot=0, color=['r', 'b'], alpha=0.7, fontsize=14)
for idx, v_ in enumerate(t_):
    plt.text(idx - 0.07, v_ - 35, "{}".format(v_), fontsize=14)

plt.xticks([0, 1], ['Non infection', 'infection'], fontsize=14)
plt.xlabel("Group", fontsize=14)
plt.ylabel("Patient", fontsize=14)
plt.show()

### 6.5 重新修訂 '2nd ICM' 數值
# np.where(condition, x, y) # 滿足條件(condition)，輸出x，不滿足輸出y。
# 計算 'total score'
df['Total Score'] = np.where((df['Serum CRP'] >= 10) | (df['D_dimer'] >= 860), 2, 0) + \
                    np.where(df['Serum ESR'] >= 30, 1, 0) + \
                    np.where((df['Synovial WBC'] >= 3000) | (df['synovial Leukocyte Esterase'] >= 2), 3, 0) + \
                    np.where(df['Synovial Neutrophil'] >= 70, 2, 0) + \
                    np.where(df['Single Positive culture'] == 1, 2, 0) + \
                    np.where(df['Positive Histology'] == 1, 3, 0) + \
                    np.where(df['Pulurence'] == 1, 3, 0)

### 6.6 重新修訂 2018 ICM: (1) >= 6 Infected, (2) 2-5 Possibly Infected, (3) 0-1 Not Infected
df['2nd ICM'] = np.where(df['Total Score'] >= 6, 1, 0)

### 6.7 修訂 2018 ICM 欄位, '2X positive cultures' =1 or Sinus Tract = 1 的患者 '2nd ICM'  = 1
df.loc[(df['2X positive culture'] == 1) | (df['Sinus Tract'] == 1), '2nd ICM'] = 1

### 6.8 刪除 missing rate < 0.619的 cols, 並 繪製圖表與列出 drop.cols_list
THRESHOLD = 200  # for missing rate: 200 / 323 = 0.619

### 6.9 忽略 df['cols'] 索引:32 以後的 cols，同時統計每一個 col 'notnull' 的個數
# 並列表為 table
table = df.notnull().sum()[:-32]  # 不看綜合病症

plt.figure(figsize=(8, 12))
plt.barh(table[table.values < THRESHOLD].index, table[table.values < THRESHOLD].values, alpha=.8,
         label='drop feature < threshold')
plt.barh(table[table.values >= THRESHOLD].index, table[table.values >= THRESHOLD].values, alpha=.8,
         label='more than threshold data')
plt.margins(y=0.008, x=0.3)
plt.legend(bbox_to_anchor=(1, 0.06), fancybox=True, shadow=True)
plt.ylabel('Feature')
plt.xlabel('Patients')
plt.xlim(0, 350)
plt.show()

if (debug_model == 1):
    print("Columns should be drop out: \n{}".format(table[table.values < THRESHOLD].index.tolist()))
df.drop(columns=table[table.values < THRESHOLD].index.tolist(), inplace=True)


temp_col = [
    'No.Group', 'No. ', 'Group',
    'PJI/Revision Date',
    'Total Score', '2nd ICM',
    'Minor ICM Criteria Total', '1st ICM',
    'Minor MSIS Criteria Total', 'MSIS final classification'
]

### 6.10 補值前處理：
# a. 把可能造成overfitting 的 outcome (temp_col) 先移除，再補值
# b. 補值後再行合併
# c. df.copy(), 複製此對象的索引和數據, 同時 reset_index, 重組index,避免產生莫名的邏輯錯誤
internal = df.copy().reset_index(drop=True)
internal_temp_df = internal[temp_col].copy()
internal.drop(columns=['Date of first surgery ', 'Date of last surgery '] + temp_col, inplace=True)
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

### 6.11 補值後之資料修補
# a: 負數轉正
# b: 將"float" cols 轉換為 int 
# c: 修正 BMI
# d: 修正'Total Elixhauser Groups per record',  往前推31個欄位加總
# e: concat程序: 補值後再與 'temp_col' cols 合併
###
# Review for profiles
# 原始資料有進行插補作業與四捨五入。
# 目前送交論文的版本為插補後四捨五入的版本 (未還原原始資料)。
# 若分析時需比對插補資料與四捨五入前後的差異，可將必要欄位進行備份，待四捨五入後再行回復
# [Line: 281-289, Line: 292-301]
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
## Review for profiles

impute_internal['BMI'] = impute_internal['BW (kg)'] / (impute_internal['Height (m)'] * impute_internal['Height (m)'])
impute_internal['Total Elixhauser Groups per record'] = impute_internal[impute_internal.columns[-32:-1]].sum(axis=1)
impute_internal = pd.concat([internal_temp_df, impute_internal], sort=False, ignore_index=False, axis=1)
impute_internal.tail()

### 6.12 將各項資料屬性，按cols進行分類
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

### 6.13 修正
# a. 將ASA儲存為 str, for OneHotEncoder
# b. 將 'Synovial Neutrophil' 上限設100
impute_internal['ASA'] = impute_internal['ASA'].astype(str)
impute_internal.loc[impute_internal['Synovial Neutrophil'] > 100, 'Synovial Neutrophil'] = 100

### 6.14 補值後 drop 'outcome' 以外的其他cols 做為建模的基礎
internal_X, internal_y = impute_internal[impute_internal.columns[10:]], impute_internal['Group']
internal_y.value_counts().sort_index().plot(kind='bar', color=['r', 'b'], title="Training dataset", rot=0)
plt.show()

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

### 6.15 Comorbidity 程序處理
# 先副本處理再寫回原變數
internal_X = Comorbidity(internal_X.copy())

### 5.16 將ASA 轉為 OneHotEncoder格式, 並整合至 internal_X
ohe = OneHotEncoder(handle_unknown='ignore')
ohe.fit(internal_X[['ASA']].values)
ASA_data = ohe.transform(internal_X[['ASA']].values).toarray().astype(int)
for idx, class_name in enumerate(ohe.categories_[0]):
    internal_X['ASA_{}'.format(class_name)] = ASA_data[:, idx]
internal_X.drop(columns=['ASA'], inplace=True)
internal_X.tail()

### 6.16 修正 columns name
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
    'Synovial_PMN', # 'Synovial Neutrophil',
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

### 7.1 Get the specific patient profile by PID
X_train, y_train = internal_X.drop(index=PID), internal_y.drop(index=PID)
X_test, y_test = internal_X.iloc[PID:PID + 1], internal_y.iloc[PID:PID + 1]
if (debug_model == 1):
    print("Tr shape: {}, Ts shape: {}".format(X_train.shape, X_test.shape))


### 7.2 Split dataset to tr (80%) and val (20%)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=666, shuffle=True)
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
### 8.1 Construct Base Classifier
xgb = xgboost.XGBClassifier(learning_rate=0.01, max_depth=3, n_estimators=50, scale_pos_weight=0.85,
                            enable_experimental_json_serialization=True, tree_method='hist', random_state=0)
rf = RandomForestClassifier(n_estimators=50, random_state=123, max_depth=5)
nb_pipe = MixedNB(categorical_features=[9, 10, 11, 15, 17, 18])
lr_pipe = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=500, random_state=123))
svc_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(probability=True, random_state=123, C=10, gamma=0.01))
])

### 8.2 Stacking Model from 80% dataset
stacking_model = StackingClassifier(
    classifiers=[xgb, rf, lr_pipe, nb_pipe],
    use_probas=True,
    average_probas=True,
    use_features_in_secondary=True,
    meta_classifier=svc_pipe
)

### 8.3 Stacking Model from 100% dataset
stacking_model.fit(X_train, y_train)

### 8.4 Explainer Modeling from 100% dataset
explainer = RandomForestClassifier(max_depth=Explainer_depth, n_estimators=100, random_state=123)
explainer.fit(X_train, y_train)

### 8.5 Check whether the prediction result of RandomForest of ts case is equal to Stacking Modeling 
## 檢查 ts case 的 RF 預測是否等同 Stacking 預測
#
## Print the accuracy with stacking model 
# For Linux
# print("rf vs. stacking compare 結果: {}".format(accuracy_score( y_test.values,stacking_model.predict(X_test.values))))
# For windows
if (debug_model == 1):
    print("GroundTruth vs. stacking compare 結果: {}".format(accuracy_score(y_test, stacking_model.predict(X_test))))
    print("GroundTruth vs. RF compare 結果: {}".format(accuracy_score(y_test, explainer.predict(X_test))))
#
## Print the Prediction & Infection Probability with Stacking model
# For Linux
# print("Stacking Prediction : {}".format(stacking_model.predict(X_test.values)[0]))
# print("Stacking Infection Proba : {}".format(stacking_model.predict_proba(X_test.values)[:, 1][0]))
# For Windows
if (debug_model == 1):
    print("Stacking Prediction : {}".format(stacking_model.predict(X_test)[0]))
    print("Stacking Infection Proba : {}".format(stacking_model.predict_proba(X_test)[:, 1][0]))


# In[10]: PID_Trace
impute_internal_ = impute_internal.copy()
X_test_val = impute_internal_.iloc[PID:PID + 1]
X_test_val[['2nd ICM', '2X positive culture', 'Sinus Tract', 'Minor ICM Criteria Total']].T
if (debug_model == 1):
    print('PID:{}'.format(PID))
    print('2X positive culture:{}'.format(X_test_val[['2X positive culture']].values[0][0]))
    print('Sinus Tract:{}'.format(X_test_val[['Sinus Tract']].values[0][0]))
    print("True Class:{}".format(y_test.values[0]))
    print("Diagnosis of Meta Learner:{}".format(stacking_model.predict(X_test)[0]))
    print("Diagnosis of 2018 ICM:{}".format(X_test_val[['2nd ICM']].values[0][0]))
    print("2018 ICM Major Criteria:{}".format(
        X_test_val[['2X positive culture']].values[0][0] | X_test_val[['Sinus Tract']].values[0][0]))
    print("2018 ICM Minor Criteria:{}".format(X_test_val[['Total Score']].values[0][0]))

# In[11]: Randomly generate random forest and candidate tree
explainers, tree_candidates = getCandidate(X_train, y_train,
                                           X_test, stacking_model, 
                                           Explainer_depth, explainers_counter)

### 10.1 Prepare the val_df (size = 10) for calculate fidelity scores
VAL_SIZE = 10
VAL_DATASET = []
Y_VAL_DATASET = []
for i in range(VAL_SIZE):
    VAL_DATASET.append(resample(X_val, n_samples=55, replace=False, random_state=i))
    Y_VAL_DATASET.append(resample(y_val, n_samples=55, replace=False, random_state=i))

### 10.2 Calculate the fidelity by explain_i
#explain_i generate the tree_candidates[explain_i]
for explain_i in list(explainers.keys()):

    VAL_list = []
    rules = []
    top_n_rank = {}
    top_n_tree_idx = {}
    for val_df, y_val_df in zip(VAL_DATASET, Y_VAL_DATASET):
        ACC_list = []
        for tree_idx in tree_candidates[explain_i]:
            tree_pred = explainers[explain_i].estimators_[tree_idx].predict(val_df)
            stack_pred = stacking_model.predict(val_df)
            ACC_list.append(accuracy_score(stack_pred, tree_pred))
        VAL_list.append(ACC_list)
    
    fidelity_scores = np.array(VAL_list).reshape(VAL_SIZE, -1).mean(axis=0)
    rank = np.argsort(-1 * fidelity_scores)
    top_n_rank[explain_i] = fidelity_scores[rank][:10]
    top_n_tree_idx[explain_i] = np.array(tree_candidates[explain_i])[rank][:10]
    
    
    ### 10.3 Enumerate the decision path of the explain_i
    res_combined = []
    for num, (idx, score) in enumerate(zip(top_n_tree_idx[explain_i], top_n_rank[explain_i])):
        # if (debug_model == 1):
        #     print("Decision Path_{} : ".format(num+1))
        res = interpret(X_test, explainers[explain_i].estimators_[idx], feature_selection2)
        rule = " and ".join([" ".join([str(w_) for w_ in r_]) for r_ in res['info']])
        rules.append(rule)
        res_combined = res_combined + [" ".join([str(w_) for w_ in r_]) for r_ in res['info']]
        # if (debug_model == 1):
        #     print("  {}".format(rule))
        #     print("  Fidelity of decision path to prediction : {:.4f}\n".format(score))
    
    ### 10.4 Fixed the decision path (rules) with condition 
    rules_ = rules            
    rules_ = [w_.replace('Positive Histology > 0.5', 'Positive_Histology == True') for w_ in rules_]
    rules_ = [w_.replace('Positive Histology <= 0.5', 'Positive_Histology == False') for w_ in rules_]
    rules_ = [w_.replace('Primary, Revision\nnative hip > 0.5', 'Surgery == True') for w_ in rules_]
    rules_ = [w_.replace('Primary, Revision\nnative hip <= 0.5', 'Surgery == False') for w_ in rules_]
    rules_ = [w_.replace('Purulence <= 0.5', 'Purulence == False') for w_ in rules_]
    rules_ = [w_.replace('Purulence > 0.5', 'Purulence == True') for w_ in rules_]
    rules_ = [w_.replace('Serum ESR', 'Serum_ESR') for w_ in rules_]
    rules_ = [w_.replace('Serum CRP', 'Serum_CRP') for w_ in rules_]    
    rules_ = [w_.replace('BW (kg)', 'BW') for w_ in rules_]    
    rules_ = [w_.replace('Segment (%)', 'Segment') for w_ in rules_]
    rules_ = [w_.replace('Synovial WBC', 'Synovial_WBC') for w_ in rules_]
    rules_ = [w_.replace('Height (m)', 'Height') for w_ in rules_]
    rules_ = [w_.replace('2X positive culture <= 0.5', 'two_positive_culture == False') for w_ in rules_]
    rules_ = [w_.replace('2X positive culture > 0.5', 'two_positive_culture == True') for w_ in rules_]
    rules_ = [w_.replace('Synovial Neutrophil', 'Synovial_PMN') for w_ in rules_]
    rules_ = [w_.replace('Serum WBC ', 'Serum_WBC_') for w_ in rules_]
    rules_ = [w_.replace('Single Positive culture', 'Single_Positive_culture') for w_ in rules_]
    rules_ = [w_.replace('P.T', 'P_T') for w_ in rules_]
    rules_ = [w_.replace('Total Elixhauser Groups per record', 'Total_Elixhauser_Groups_per_record') for w_ in rules_]     
    rules_ = [w_.replace('Total CCI', 'Total_CCI') for w_ in rules_]
    rules_ = [w_.replace('HGB', 'Hb') for w_ in rules_]
    rules_
    
    # In[12]: d_path by feature importance from PJI-PI-01-02-2021.docx (Table 4) 
    # Modify to suitable names for the parser
    d_path = {
        '2X positive culture': 'two_positive_culture',
        'APTT': 'APTT',
        'ASA_2': 'ASA_2',
        'Age': 'Age',
        'HGB': 'Hb',
        'P.T': 'P_T',
        'PLATELET': 'PLATELET',
        'Positive Histology': 'Positive_Histology',
        'Primary, Revision\nnative hip': 'Surgery',
        'Pulurence': 'Purulence',  # 膿:Purulence
        #'Rheumatoid Arthritis/collagen': 'Rheumatoid_Arthritis/collagen',
        'Segment (%)': 'Segment',
        'Serum CRP': 'Serum_CRP',
        'Serum ESR': 'Serum_ESR',
        'Serum WBC ': 'Serum_WBC_',
        'Single Positive culture': 'Single_Positive_culture',
        'Synovial WBC': 'Synovial_WBC',
        'Synovial_PMN': 'Synovial_PMN',
        'Total CCI': 'Total_CCI',
        'Total Elixhauser Groups per record': 'Total_Elixhauser_Groups_per_record',
    }
    
    # In[13]: Enumerate the mean_fidelitys of decision path and decision tree
    condition_i = 0
    AVG_FIDELITYS = []
    CONDITIOS = rules_
    if (debug_model == 1):
        print("Enumerate the decision path of the explain[{n}]".format(n=explain_i))
    for condition in rules_: 
        fidelity2, fidelity, mean_fidelity = [], [], []
        ACC_list, AUC_list, F1_list, Recall_list,Precision_list, MCC_list = [],[],[],[],[],[]
        #for val_df in VAL_DATASET:
        for val_df, y_val_df in zip(VAL_DATASET, Y_VAL_DATASET):
            stack_pred = stacking_model.predict(val_df)
            
            #debug
            for tree_idx in tree_candidates[explain_i]:
                tree_model = explainers[explain_i].estimators_[tree_idx]
                res = interpret(X_test, explainers[explain_i].estimators_[tree_idx], feature_selection2)
                # rule = " and ".join([" ".join([str(w_) for w_ in r_]) for r_ in res['info']])
                tree_pred = tree_model.predict(val_df)
                
                merge_pred = np.where(val_df.rename(columns=d_path).eval(condition), rule_1, rule_2)
                # 
                fidelity.append(accuracy_score(stack_pred, merge_pred))
                fidelity2.append(accuracy_score(stack_pred, tree_pred))
            
            # ACC_list.append(accuracy_score(y_val_df, merge_pred))
            # fpr, tpr, thresholds = roc_curve(y_val_df, merge_pred, pos_label=1)
            # AUC_list.append(auc(fpr, tpr))
            # F1_list.append(f1_score(y_val_df, merge_pred, average='weighted'))
            # Recall_list.append(recall_score(y_val_df, merge_pred, average='weighted'))
            # Precision_list.append(precision_score(y_val_df, merge_pred, average='weighted'))
            # MCC_list.append(matthews_corrcoef(y_val_df.values, merge_pred))
        mean_fidelity = round(np.mean(fidelity),3)
        AVG_FIDELITYS.append(mean_fidelity)
        if (debug_model == 1):
            print("Decision_path[{n}]:{f}".format(n=condition_i, f=mean_fidelity))
            print("{s}".format(s=condition))
        condition_i = condition_i + 1
        #print("mean_fidelity_of decision_path: {:.3f}".format(np.mean(fidelity)))
        #print("mean_fidelity_of decision_tree: {:.3f}".format(np.mean(fidelity2)))
    CONDITIOS_AvgFidelity[explain_i, 'rules'] = CONDITIOS
    CONDITIOS_AvgFidelity[explain_i, 'fidelity'] = AVG_FIDELITYS
    
# In[14]: Concatenate multi lists for CONDITIOS_AvgFidelity
ind_, fidelity_list, rules_list = getTopN_Fidelity(CONDITIOS_AvgFidelity, list(explainers.keys()), topN)