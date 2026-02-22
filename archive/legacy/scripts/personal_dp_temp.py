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
import auto_pick_boundary_v1
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
final_singleton = set()
Explainer_depth = 12  # The depth of Explainer Model
explainer = RandomForestClassifier(
    max_depth=Explainer_depth, n_estimators=100, random_state=123)

# In[2]: Collect Decision Tree with the same prediction as the stacking model
# Description: Collect Decision Tree with the same prediction as the stacking model


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
    while_flag = 0  # the flag of available explainer
    i = 0          # the index of explainer
    while (True):
        i = i + 1
        if while_flag >= explainer_count:
            break

        explainers[i-1] = RandomForestClassifier(max_depth=Explainer_depth,
                                                 n_estimators=100, random_state=i-1)
        explainers[i-1].fit(X_train.values, y_train.values)
        stacking_pred = stacking_model.predict(test_X)
        explainers_pred = explainers[i-1].predict(test_X)
        # print("train:"+str(i))
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
            del explainers[i-1]  # 不存在當下可解釋器，刪除指標
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
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]
        :node_indicator.indptr[sample_id + 1]]
    result['info'] = []

    for node_id in node_index:
        if leave_id[sample_id] == node_id:
            continue

        if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"
        result['info'].append(
            [feature_names[feature[node_id]], threshold_sign, round(threshold[node_id], 2)])

    return result

# In[4]: Comorbidity analysis


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

# In[5]: getTopN_Fidelity


def getTopN_Fidelity(fidelity_list, top_N_indices, top_N):
    # fidelity_list, candidate_decision_path & the fidelity
    # top_N_indices: the indices of top_N of the decision path.
    fidelity_list_ = []
    rules_list_ = []
    for idx in top_N_indices:
        fidelity_list_ = fidelity_list_ + \
            fidelity_list[idx, 'fidelity']
        rules_list_ = rules_list_ + fidelity_list[idx, 'rules']

    top_n_fidelity_i = sorted(
        range(len(fidelity_list_)), key=lambda k: fidelity_list_[k])[-top_N:]
    # print(len(fidelity_list_))
    # print(len(rules_list_))
    # print(sorted(fidelity_list_))
    return list(itemgetter(*top_n_fidelity_i)(rules_list_))


# In[15]: Import Library
# 引用適當的 Library
# from sympy.logic import simplify_logic


# In[16]: Declare the function for clasification with ICM & nonICM
# 原始推導流程參閱掃描筆記檔案
def Non_ICM_1(x):
    return {
        'mu(I)_delta_mu(N)': ['<=', 'delta', 'mu(N)'],
        'mu(I)_mu(N)_delta': ['NotUsed'],
        'delta_mu(N)_mu(I)': ['NotUsed'],
        'delta_mu(I)_mu(N)': ['<=', 'delta', 'mu(N)'],
        'mu(N)_delta_mu(I)': ['>=', 'mu(N)', 'delta'],
        'mu(N)_mu(I)_delta': ['>=', 'mu(N)', 'delta'],

    }.get(x, 'Error')


def Non_ICM_0(x):
    return {
        'delta_mu(N)_mu(I)': ['<=', 'delta', 'mu(I)'],
        'delta_mu(I)_mu(N)': ['NotUsed'],
        'mu(N)_mu(I)_delta': ['NotUsed'],
        'mu(N)_delta_mu(I)': ['<=', 'delta', 'mu(I)'],
        'mu(I)_delta_mu(N)': ['>=', 'mu(I)', 'delta'],
        'mu(I)_mu(N)_delta': ['>=', 'mu(I)', 'delta'],
    }.get(x, 'Error')


def ICM_1(x):
    return {

        'ICM_Q_delta': ['>=', 'ICM', 'delta'],
        'ICM_delta_Q': ['>=', 'ICM', 'delta'],
        'Q_ICM_delta': ['>=', 'ICM', 'delta'],
        'Q_delta_ICM': ['>=', 'Q', 'delta'],
        'delta_ICM_Q': ['NotUsed'],
        'delta_Q_ICM': ['NotUsed'],

    }.get(x, 'Error')


def ICM_0(x):
    return {
        'delta_Q_ICM': ['<=', 'delta', 'ICM'],
        'delta_ICM_Q': ['<=', 'delta', 'ICM'],
        'ICM_delta_Q': ['<=', 'delta', 'Q'],
        'ICM_Q_delta': ['NotUsed'],
        'Q_ICM_delta': ['NotUsed'],
        'Q_delta_ICM': ['<=', 'delta', 'ICM'],

    }.get(x, 'Error')


# In[17]: Get the upperbound & lowerbound of sintletons
# 注意：此處須匯入兩個檔案: Non2018ICM.xlsx, 2018ICM.xlsx
def singleton_opt(X_test):
    non2018ICM = pd.read_excel(
        "/Users/johnnyhu/Desktop/PJI_20210428_InterpretableML/Non2018ICM.xlsx")
    _2018ICM = pd.read_excel(
        "/Users/johnnyhu/Desktop/PJI_20210428_InterpretableML/2018ICM.xlsx")

    _2018ICM_ = _2018ICM[['variable', 'threshold']]
    non2018ICM_ = non2018ICM[['variable', 'mu(N)', 'mu(I)']]

    # non2018ICM_.iloc[0, 1]

    X_test_ = X_test.rename(
        columns={'Serum WBC ': 'Serum_WBC_',
                 'Segment (%)': 'Segment',
                 'HGB': 'Hb',
                 'P.T': 'P_T',
                 'Total CCI': 'Total_CCI',
                 'Total Elixhauser Groups per record': 'Total_Elixhauser_Groups_per_record',
                 'Serum CRP': 'Serum_CRP',
                 'Serum ESR': 'Serum_ESR',
                 'Synovial WBC': 'Synovial_WBC',

                 }
    )

    decision_list = {}
    decision_list_file = open("decision_list_file.txt", "w")
    for val in list(final_singleton):
        NotUsed_flag = 0

        if ("==") in val:
            continue
        elif ("<=") in val:
            variable = val[:val.find("<=")-1]
            operator = val[val.find("<="):val.find("<=")+2]
            Q = val[val.find("<=")+3:]
        elif ("< ") in val:
            variable = val[:val.find("<")-1]
            operator = val[val.find("<"):val.find("<")+1]
            Q = val[val.find("<")+2:]
        elif (">=") in val:
            variable = val[:val.find(">=")-1]
            operator = val[val.find(">="):val.find(">=")+2]
            Q = val[val.find(">=")+3:]
        elif ("> ") in val:
            variable = val[:val.find(">")-1]
            operator = val[val.find(">"):val.find(">")+1]
            Q = val[val.find(">")+2:]

        # print (variable, operator, Q)

        try:
            if variable in list(non2018ICM_['variable']):  # non2018ICM List
                index = list(non2018ICM_['variable']).index(variable)
                mu_N = non2018ICM_['mu(N)'][index]
                mu_I = non2018ICM_['mu(I)'][index]
                delta = float(X_test_[variable])

                nonICM = {"mu(N)": mu_N, "mu(I)": mu_I, "delta": delta}
                # nonICM_Sorting = (sorted(nonICM.items(), key=lambda x:x[1]))
                nonICM_Sorting = sorted(nonICM, key=nonICM.get)
                concate_nonICM_Sorting = '_'.join(nonICM_Sorting)

                if explainer.predict(X_test) == 0:   # Meta(I) = 0
                    result = Non_ICM_0(concate_nonICM_Sorting)
                    if len(result) == 3:
                        operator, _lower_bound, _upper_bound = result
                        lower_bound, upper_bound = nonICM[_lower_bound], nonICM[_upper_bound]

                    else:  # 不考慮
                        NotUsed_flag = 1

                elif explainer.predict(X_test) == 1:  # Meta(I) = 1
                    result = Non_ICM_1(concate_nonICM_Sorting)
                    if len(result) == 3:
                        operator, _lower_bound, _upper_bound = result
                        lower_bound, upper_bound = nonICM[_lower_bound], nonICM[_upper_bound]

                    else:  # 不考慮
                        NotUsed_flag = 1

            elif variable in list(_2018ICM_['variable']):  # 2018ICM List
                index = list(_2018ICM_['variable']).index(variable)
                ICM_threshold = _2018ICM_['threshold'][index]
                delta = float(X_test_[variable])

                ICM = {"ICM": ICM_threshold, "Q": float(Q), "delta": delta}
                ICM_Sorting = sorted(ICM, key=ICM.get)
                concate_ICM_Sorting = '_'.join(ICM_Sorting)

                if explainer.predict(X_test) == 0:   # Meta(I) = 0
                    result = ICM_0(concate_ICM_Sorting)
                    if len(result) == 3:
                        operator, _lower_bound, _upper_bound = result
                        lower_bound, upper_bound = ICM[_lower_bound], ICM[_upper_bound]

                    else:  # 不考慮
                        NotUsed_flag = 1

                elif explainer.predict(X_test) == 1:   # Meta(I) = 1
                    result = ICM_1(concate_ICM_Sorting)
                    if len(result) == 3:
                        operator, _lower_bound, _upper_bound = result
                        lower_bound, upper_bound = ICM[_lower_bound], ICM[_upper_bound]

                    else:  # 不考慮
                        NotUsed_flag = 1

            if NotUsed_flag == 1:
                decision_list[val] = result
            else:
                decision_list[val] = [
                    variable, operator, lower_bound, upper_bound]

            decision_list_file.write(str(val)+" "+str(decision_list[val])+"\n")

        except Exception as e:
            print("You got an Exception.", str(e))

    decision_list_file.close()
    return decision_list


# In[18]: Transfer the decision_path to POS Form
def transPOSForm(Candidate_DecisionPath):

    # In[19]: 讀取 rules_list, data
    # data = [
    # # '(Serum_ESR > 48.5 and Segment > 64.5) | (Serum_ESR > 49.0 and Serum_CRP > 13.5 and APTT > 29.5 and Synovial_PMN > 52.0) | (Serum_ESR > 39.5 and Synovial_PMN > 69.0 and Age > 32.5 and Synovial_WBC > 367.5) | (Serum_ESR > 55.0 and P_T > 5.0 and Age > 33.0 and Hb > 8.5) | (Serum_ESR > 41.0 and Synovial_PMN > 41.5 and P_T > 5.0 and APTT > 23.5 and Age > 31.5 and Segment > 64.5)',
    # '(Positive_Histology == True and Serum_CRP > 3.0 and Hb <= 13.5) | (Positive_Histology == True and Serum_CRP > 2.5 and APTT > 23.5 and Hb <= 13.5) | (Serum_ESR > 39.5 and APTT > 23.5 and Serum_CRP > 4.0 and Positive_Histology == True) | (Positive_Histology == True and Serum_CRP > 3.0) | (Positive_Histology == True and P_T > 5.0 and Segment > 65.5) | (Serum_ESR > 28.5 and Synovial_WBC > 2483.5 and APTT > 26.5)'
    # ]
    data = Candidate_DecisionPath
    res_combined = []
    # In[20] 拆解所有的 decision path 元素
    # input: all_path
    # output: all_singleton
    for element in data:
        element = element.replace(') | (', ' and ')
        element = element.replace('(', '')
        element = element.replace(')', '')
        element = element.replace('&', 'and')

        res_combined = res_combined + \
            [x.strip(' ') for x in element.split('and')]

    _singleton_set = set()
    for atom in res_combined:
        _singleton_set.add(atom)

    _singleton_list = list()
    for atom in _singleton_set:
        _singleton_list.append(atom)

    # In[21]: 列舉 all_singleton 的 features by Set()
    singleton_f = set()
    for i, (val) in enumerate(_singleton_list):

        if val.find("<=") > 0:
            singleton_f.add(val[:val.find("<=")-1])

        elif val.find("<") > 0:
            singleton_f.add(val[:val.find("<")-1])

        elif val.find("==") > 0:
            singleton_f.add(val[:val.find("==")-1])

        elif val.find(">=") > 0:
            singleton_f.add(val[:val.find(">=")-1])

        elif val.find(">") > 0:
            singleton_f.add(val[:val.find(">")-1])

    # In[22]: 列舉 all_singleton 的 features by List()
    _singleton_list_f = list()
    for i, (val) in enumerate(_singleton_list):

        if val.find("<=") > 0:
            _singleton_list_f.append(val[:val.find("<=")-1])
        elif val.find("<") > 0:
            _singleton_list_f.append(val[:val.find("<")-1])
        elif val.find("==") > 0:
            _singleton_list_f.append(val[:val.find("==")-1])
        elif val.find(">=") > 0:
            _singleton_list_f.append(val[:val.find(">=")-1])
        elif val.find(">") > 0:
            _singleton_list_f.append(val[:val.find(">")-1])

    # In[23]: 列舉所有 decision path的 singleton constraints
    _singleton_list_ = {}
    final_singleton = set()
    for i, (val) in enumerate(singleton_f):
        index = [n for n, x in enumerate(_singleton_list_f) if x == val]
        val_list = []

        if len(index) == 1:
            # print(_singleton_list[index[0]])
            final_singleton.add(_singleton_list[index[0]])
        else:
            if (_singleton_list[index[0]].find('<=') > 0):
                for val2 in index:  # index):

                    if _singleton_list[val2].find("<") > 1:
                        val_list.append(
                            float(_singleton_list[val2][_singleton_list[val2].find("<")+2:]))

                    elif _singleton_list[val2].find("<=") > 1:
                        val_list.append(
                            float(_singleton_list[val2][_singleton_list[val2].find("<=")+3:]))

                    elif _singleton_list[val2].find(">") > 1:
                        val_list.append(
                            float(_singleton_list[val2][_singleton_list[val2].find(">")+2:]))

                    elif _singleton_list[val2].find(">=") > 1:
                        val_list.append(
                            float(_singleton_list[val2][_singleton_list[val2].find(">=")+3:]))

                min_value = min(val_list)
                min_index = val_list.index(min_value)
                final_singleton.add(_singleton_list[index[min_index]])

                for i in index:
                    _singleton_list_[_singleton_list[i]
                                     ] = _singleton_list[index[min_index]]

            elif (_singleton_list[index[0]].find('<') > 0):
                for val2 in index:  # index):
                    if _singleton_list[val2].find("<") > 1:
                        val_list.append(
                            float(_singleton_list[val2][_singleton_list[val2].find("<")+2:]))

                    elif _singleton_list[val2].find("<=") > 1:
                        val_list.append(
                            float(_singleton_list[val2][_singleton_list[val2].find("<=")+3:]))

                    elif _singleton_list[val2].find(">") > 1:
                        val_list.append(
                            float(_singleton_list[val2][_singleton_list[val2].find(">")+2:]))

                    elif _singleton_list[val2].find(">=") > 1:
                        val_list.append(
                            float(_singleton_list[val2][_singleton_list[val2].find(">=")+3:]))
                min_value = min(val_list)
                min_index = val_list.index(min_value)
                final_singleton.add(_singleton_list[index[min_index]])

                for i in index:
                    _singleton_list_[_singleton_list[i]
                                     ] = _singleton_list[index[min_index]]

            elif (_singleton_list[index[0]].find('>=') > 0):
                for val2 in index:
                    if _singleton_list[val2].find("<") > 1:
                        val_list.append(
                            float(_singleton_list[val2][_singleton_list[val2].find("<")+2:]))

                    elif _singleton_list[val2].find("<=") > 1:
                        val_list.append(
                            float(_singleton_list[val2][_singleton_list[val2].find("<=")+3:]))

                    elif _singleton_list[val2].find(">") > 1:
                        val_list.append(
                            float(_singleton_list[val2][_singleton_list[val2].find(">")+2:]))

                    elif _singleton_list[val2].find(">=") > 1:
                        val_list.append(
                            float(_singleton_list[val2][_singleton_list[val2].find(">=")+3:]))
                max_value = max(val_list)
                max_index = val_list.index(max_value)
                final_singleton.add(_singleton_list[index[max_index]])

                for i in index:
                    _singleton_list_[_singleton_list[i]
                                     ] = _singleton_list[index[max_index]]

            elif (_singleton_list[index[0]].find('>') > 0):
                for val2 in index:
                    if _singleton_list[val2].find("<") > 1:
                        val_list.append(
                            float(_singleton_list[val2][_singleton_list[val2].find("<")+2:]))

                    elif _singleton_list[val2].find("<=") > 1:
                        val_list.append(
                            float(_singleton_list[val2][_singleton_list[val2].find("<=")+3:]))

                    elif _singleton_list[val2].find(">") > 1:
                        val_list.append(
                            float(_singleton_list[val2][_singleton_list[val2].find(">")+2:]))

                    elif _singleton_list[val2].find(">=") > 1:
                        val_list.append(
                            float(_singleton_list[val2][_singleton_list[val2].find(">=")+3:]))
                max_value = max(val_list)
                max_index = val_list.index(max_value)
                final_singleton.add(_singleton_list[index[max_index]])

                for i in index:
                    _singleton_list_[_singleton_list[i]
                                     ] = _singleton_list[index[max_index]]

    # In[24]: 宣告 Symbols by ns[] for symbols
    sym = "a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x"
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x = symbols(
        sym)
    sym_array = [a, b, c, d, e, f, g, h, i, j,
                 k, l, m, n, o, p, q, r, s, t, u, v, w, x]
    ns = {}
    for s_, singleton_ in zip(sym_array, final_singleton):
        ns[s_] = Symbol(singleton_)

    # In[25]: decision path 同質性簡化
    data_ = data
    for val in list(_singleton_list_.keys()):
        data_ = [w_.replace(val, _singleton_list_[val]) for w_ in data_]

    ironmen_dict = {"featureSet": data_}
    # 建立 data frame
    df = pd.DataFrame(ironmen_dict)

    # logic_path = set()
    transPOSForm = []
    for line in df['featureSet']:
        string = line
        for s_, singleton_ in zip(list(['ns[a]', 'ns[b]', 'ns[c]', 'ns[d]', 'ns[e]',
                                        'ns[f]', 'ns[g]', 'ns[h]', 'ns[i]', 'ns[j]',
                                        'ns[k]', 'ns[l]', 'ns[m]', 'ns[n]', 'ns[o]',
                                        'ns[p]', 'ns[q]', 'ns[r]', 'ns[s]', 'ns[t]',
                                        'ns[u]', 'ns[v]', 'ns[w]', 'ns[x]'
                                        ]), list(final_singleton)):

            string = string.replace(singleton_, s_)

        string = string.replace('and', '&')
        string = string.replace('or', '|')

        out = to_dnf(eval(string))
        transPOSForm.append(out)

    return list(set(transPOSForm)), list(final_singleton)


# In[6]: Main parameter setting
# 主體參數設定
debug_model = 0
pID_idx = 5
# pID = [11, 212, 51, 210, 79, 159]
# PID = pID[pID_idx]

explainers_counter = 5  # 找出 n 組候選 explainers
CONDITIOS_AvgFidelity = {}

# In[7]: File reading and pre-processing
# 6.1 讀檔與前處理作業
df = pd.read_excel(
    '/Users/johnnyhu/Desktop/xray_outside_data.xlsx')
# df = pd.read_excel('/Users/johnnyhu/Desktop/Revision_PJI_main.xlsx')

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
THRESHOLD = 0.065  # for missing rate: 200 / 323 = 0.619

# 6.9 忽略 df['cols'] 索引:32 以後的 cols，同時統計每一個 col 'notnull' 的個數
# 並列表為 table
table = df.notnull().sum()[:-32]  # 不看綜合病症

if (debug_model == 1):
    print("Columns should be drop out: \n{}".format(
        table[table.values < THRESHOLD].index.tolist()))
df.drop(columns=table[table.values <
                      THRESHOLD].index.tolist(), inplace=True)

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
###
# Review for profiles
#
# 目前送交論文的版本為插補後四捨五入的版本 (未還原原始資料)。
# 若分析時需比對插補資料與四捨五入前後的差異，可將必要欄位進行備份，待四捨五入後再行回復
# [Line: 321-329, Line: 332-340]
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
internal_X, internal_y = impute_internal[impute_internal.columns[10:]
                                         ], impute_internal['Group']
internal_y.value_counts().sort_index().plot(
    kind='bar', color=['r', 'b'], title="Training dataset", rot=0)
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

### 補充:  iloc vs loc 功能說明 ###
# iloc，即index locate 用index索引進行定位，所以引數是整型，如：df.iloc[10:20, 3:5]
# loc，則可以使用column名和index名進行定位，如：df.loc[‘image1’:‘image10’, ‘age’:‘score’]
internal_X = internal_X.loc[:, feature_selection2].copy()

if (debug_model == 1):
    print(internal_X.shape)
internal_X.tail()
internal_X.columns

internal_X.to_csv('PJI_Dataset/outside_x_all.csv',
                  encoding='utf-8', index=False)
internal_y.to_csv('PJI_Dataset/outside_y_all.csv',
                  encoding='utf-8', index=False)


def personalDP(PID):
    start_c = time.time()
    X_res_test = pd.read_csv(
        'PJI_Dataset/outside_x_test.csv', encoding='utf-8')

    internal_X = pd.read_csv('PJI_Dataset/outside_x_all.csv', encoding='utf-8')
    internal_y = pd.read_csv('PJI_Dataset/outside_y_all.csv', encoding='utf-8')

    no_group = list(X_res_test['No.Group'])
    PID_index = no_group.index(PID)
    print('PID:', PID)
    print('PID_index:', PID_index)

    # # 7.1 Get the specific patient profile by PID
    X_train, y_train = internal_X.drop(
        index=PID_index), internal_y.drop(index=PID_index)
    # X_train, y_train = internal_X, internal_y

    X_test, y_test = internal_X.iloc[PID_index:PID_index +
                                     1], internal_y.iloc[PID_index:PID_index + 1]

    # Split dataset to tr (80%) and val (20%)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=666, shuffle=True)

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

    model_xgb = xgb.fit(X_train.values, y_train.values)

    # 8.2 Stacking Model from 80% dataset   ?
    stacking_model = StackingClassifier(
        classifiers=[xgb, rf, lr_pipe],
        use_probas=True,
        average_probas=True,
        use_features_in_secondary=True,
        meta_classifier=svc_pipe
    )
    stacking_model.fit(X_train, y_train)
    print('result:', stacking_model.predict(X_test))
    # # 8.3 Stacking Model from 100% dataset
    # if os.path.exists('PJI_model/Stacking_model_'+str(PID)):
    #     print("model exists, loading...")
    #     stacking_model = joblib.load('PJI_model/Stacking_model_'+str(PID))
    # else:
    #     print("model missing, training...")
    #     stacking_model.fit(X_train, y_train)
    #     joblib.dump(stacking_model, 'PJI_model/Stacking_model_'+str(PID))
    # personal_result = stacking_model.predict(X_test.values)[0]
    # # 根據ground_truth & Meta Learner 調節
    # if personal_result == 0:
    #     rule_1, rule_2 = 0, 1  # ground_truth: N, Meta: N
    # else:
    #     rule_1, rule_2 = 1, 0  # ground_truth: I, Meta: I

    # print("Stacking Prediction : {}".format(personal_result))

    # # '''test for export the progress'''
    # # tempfile = open("progress.tmp", "w")
    # # tempfile.write(personal_result)
    # # tempfile.close()
    # # '''end test'''

    # if os.path.exists('PJI_model/Stacking_model_'+str(PID)) & os.path.exists("Decision_rule/decision_rule_"+str(PID)+".json"):
    #     print("model exists, skip pruning...")
    # else:
    #     print("model missing, pruning...")
    #     # 8.4 Explainer Modeling from 100% dataset   ?
    #     explainer.fit(X_train.values, y_train.values)
    #     estimator = explainer.estimators_[5]

    #     if (debug_model == 1):
    #         print("GroundTruth vs. stacking compare 結果: {}".format(
    #             accuracy_score(y_test, stacking_model.predict(X_test))))
    #         print("GroundTruth vs. RF compare 結果: {}".format(
    #             accuracy_score(y_test, explainer.predict(X_test))))

    #     if (debug_model == 1):
    #         print("Stacking Prediction : {}".format(
    #             stacking_model.predict(X_test)[0]))
    #         print("Stacking Infection Proba : {}".format(
    #             stacking_model.predict_proba(X_test)[:, 1][0]))

    #     # In[10]: PID_Trace
    #     impute_internal_ = impute_internal.copy()
    #     X_test_val = impute_internal_.iloc[PID:PID + 1]
    #     X_test_val[['2nd ICM', '2X positive culture',
    #                 'Sinus Tract', 'Minor ICM Criteria Total']].T

    #     loaded_model = joblib.load('Stacking_model/stacking_model.pkl')
    #     # In[11]: Randomly generate random forest and candidate tree
    #     explainers, tree_candidates = getCandidate(X_train, y_train,
    #                                                X_test, loaded_model,
    #                                                Explainer_depth, explainers_counter)

    #     # 10.1 Prepare the val_df (size = 10) for calculate fidelity scores
    #     VAL_SIZE = 1
    #     VAL_DATASET = []
    #     Y_VAL_DATASET = []
    #     for i in range(VAL_SIZE):
    #         VAL_DATASET.append(resample(X_val, n_samples=33,
    #                                     replace=False, random_state=i))
    #         Y_VAL_DATASET.append(resample(y_val, n_samples=33,
    #                                       replace=False, random_state=i))

    #     # 10.2 Calculate the fidelity by explain_i
    #     # explain_i generate the tree_candidates[explain_i]
    #     for explain_i in list(explainers.keys()):
    #         VAL_list = []
    #         rules = []
    #         top_n_rank = {}
    #         top_n_tree_idx = {}
    #         for val_df, y_val_df in zip(VAL_DATASET, Y_VAL_DATASET):
    #             ACC_list = []
    #             for tree_idx in tree_candidates[explain_i]:
    #                 tree_pred = explainers[explain_i].estimators_[
    #                     tree_idx].predict(val_df)
    #                 stack_pred = stacking_model.predict(val_df)
    #                 ACC_list.append(accuracy_score(stack_pred, tree_pred))
    #             VAL_list.append(ACC_list)

    #         fidelity_scores = np.array(VAL_list).reshape(
    #             VAL_SIZE, -1).mean(axis=0)
    #         rank = np.argsort(-1 * fidelity_scores)
    #         top_n_rank[explain_i] = fidelity_scores[rank][:10]
    #         top_n_tree_idx[explain_i] = np.array(
    #             tree_candidates[explain_i])[rank][:10]

    #         # 10.3 Enumerate the decision path of the explain_i
    #         res_combined = []
    #         for num, (idx, score) in enumerate(zip(top_n_tree_idx[explain_i], top_n_rank[explain_i])):
    #             # if (debug_model == 1):
    #             #     print("Decision Path_{} : ".format(num+1))
    #             res = interpret(X_test, explainers[explain_i].estimators_[
    #                             idx], feature_selection2)
    #             rule = " and ".join([" ".join([str(w_) for w_ in r_])
    #                                  for r_ in res['info']])
    #             rules.append(rule)
    #             res_combined = res_combined + \
    #                 [" ".join([str(w_) for w_ in r_]) for r_ in res['info']]

    #         # 10.4 Fixed the decision path (rules) with condition
    #         rules_ = rules
    #         rules_ = [w_.replace('Positive Histology > 0.5',
    #                              'Positive_Histology == True') for w_ in rules_]
    #         rules_ = [w_.replace('Positive Histology <= 0.5',
    #                              'Positive_Histology == False') for w_ in rules_]
    #         rules_ = [w_.replace('Primary, Revision\nnative hip > 0.5',
    #                              'Surgery == True') for w_ in rules_]
    #         rules_ = [w_.replace('Primary, Revision\nnative hip <= 0.5',
    #                              'Surgery == False') for w_ in rules_]
    #         rules_ = [w_.replace('Purulence <= 0.5', 'Purulence == False')
    #                   for w_ in rules_]
    #         rules_ = [w_.replace('Purulence > 0.5', 'Purulence == True')
    #                   for w_ in rules_]
    #         rules_ = [w_.replace('Serum ESR', 'Serum_ESR') for w_ in rules_]
    #         rules_ = [w_.replace('Serum CRP', 'Serum_CRP') for w_ in rules_]
    #         rules_ = [w_.replace('BW (kg)', 'BW') for w_ in rules_]
    #         rules_ = [w_.replace('Segment (%)', 'Segment') for w_ in rules_]
    #         rules_ = [w_.replace('Synovial WBC', 'Synovial_WBC')
    #                   for w_ in rules_]
    #         rules_ = [w_.replace('Height (m)', 'Height') for w_ in rules_]
    #         rules_ = [w_.replace('2X positive culture <= 0.5',
    #                              'two_positive_culture == False') for w_ in rules_]
    #         rules_ = [w_.replace('2X positive culture > 0.5',
    #                              'two_positive_culture == True') for w_ in rules_]
    #         rules_ = [w_.replace('Synovial Neutrophil', 'Synovial_PMN')
    #                   for w_ in rules_]
    #         rules_ = [w_.replace('Serum WBC ', 'Serum_WBC_') for w_ in rules_]
    #         rules_ = [w_.replace('Single Positive culture',
    #                              'Single_Positive_culture') for w_ in rules_]
    #         rules_ = [w_.replace('P.T', 'P_T') for w_ in rules_]
    #         rules_ = [w_.replace('Total Elixhauser Groups per record',
    #                              'Total_Elixhauser_Groups_per_record') for w_ in rules_]
    #         rules_ = [w_.replace('Total CCI', 'Total_CCI') for w_ in rules_]
    #         rules_ = [w_.replace('HGB', 'Hb') for w_ in rules_]

    #         # rules_
    #         # In[12]: d_path by feature importance from PJI-PI-01-02-2021.docx (Table 4)
    #         # Modify to suitable names for the parser
    #         d_path = {
    #             '2X positive culture': 'two_positive_culture',
    #             'APTT': 'APTT',
    #             'ASA_2': 'ASA_2',
    #             'Age': 'Age',
    #             'HGB': 'Hb',
    #             'P.T': 'P_T',
    #             'PLATELET': 'PLATELET',
    #             'Positive Histology': 'Positive_Histology',
    #             'Primary, Revision\nnative hip': 'Surgery',
    #             'Pulurence': 'Purulence',  # 膿:Purulence
    #             # 'Rheumatoid Arthritis/collagen': 'Rheumatoid_Arthritis/collagen',
    #             'Segment (%)': 'Segment',
    #             'Serum CRP': 'Serum_CRP',
    #             'Serum ESR': 'Serum_ESR',
    #             'Serum WBC ': 'Serum_WBC_',
    #             'Single Positive culture': 'Single_Positive_culture',
    #             'Synovial WBC': 'Synovial_WBC',
    #             'Synovial_PMN': 'Synovial_PMN',
    #             'Total CCI': 'Total_CCI',
    #             'Total Elixhauser Groups per record': 'Total_Elixhauser_Groups_per_record',
    #         }

    #         condition_i = 0
    #         AVG_FIDELITYS = []
    #         CONDITIOS = rules_
    #         # if (debug_model == 1):
    #         #     print("Enumerate the decision path of the explain[{n}]".format(
    #         #         n=explain_i))
    #         print("Enumerate the decision path of the explain[{n}]".format(
    #             n=explain_i))
    #         for condition in rules_:
    #             fidelity = []
    #             for val_df in VAL_DATASET:
    #                 # val_df = val_df.to_frame
    #                 # print(val_df)
    #                 stack_pred = stacking_model.predict(val_df)
    #                 val_df = val_df.rename(columns=d_path)
    #                 merge_pred = np.where(
    #                     val_df.eval(condition), rule_1, rule_2)
    #                 for tree_idx in tree_candidates[explain_i]:
    #                     fidelity.append(accuracy_score(stack_pred, merge_pred))
    #             AVG_FIDELITYS.append(round(np.mean(fidelity), 3))
    #         # print(AVG_FIDELITYS)
    #         # validate fidelity
    #         CONDITIOS_AvgFidelity[explain_i, 'rules'] = CONDITIOS
    #         CONDITIOS_AvgFidelity[explain_i, 'fidelity'] = AVG_FIDELITYS

    #     end_c = time.time()
    #     # In[14]: Concatenate multi lists for CONDITIOS_AvgFidelity
    #     rules_list = getTopN_Fidelity(
    #         CONDITIOS_AvgFidelity, list(explainers.keys()), 7)
    #     print('TOP7 rule list:')
    #     print(rules_list)
    #     # In[26]: Call the transPOSForm func.
    #     POS_Form, final_singleton = transPOSForm(rules_list)
    #     print('final_singleton:')
    #     print(final_singleton)
    #     ## 20210531 ##
    #     # 套用演算法分類 ICM & non_ICM 分類
    #     # 原始推導資料請參閱文件
    #     ## singleton_opt == parser.py
    #     # In[27]: Call the singleton_opt function
    #     rules_list = auto_pick_boundary_v1.singleton_opt(PID, final_singleton)
    #     print('rules list after auto bound:'+str(PID))
    #     print(rules_list)
    #     # print('opt_decision_list:')
    #     # print(opt_decision_list)
    #     # In[28]: return the indices of the each rule with in truth table
    #     rule_i_ = []
    #     for i in range(len(POS_Form)):
    #         rule = list()
    #         rule_i = list()
    #         for j in range(len(final_singleton)):
    #             if final_singleton[j] in str(POS_Form[i]):
    #                 rule.append(str(final_singleton[j]))
    #                 rule_i.append(j)
    #         if (debug_model == 1):
    #             print(rule, rule_i)
    #         rule_i_.append(rule_i)
    #     # print(rule_i_)
    #     ###
    #     # Init the truth table
    #     init_data = np.array([], dtype=np.int64).reshape(
    #         0, len(final_singleton))
    #     lst_all = pd.DataFrame(init_data, columns=final_singleton)
    #     for index_rule in rule_i_:
    #         # print(i)
    #         M = len(final_singleton)-len(index_rule)
    #         lst = [list(a) for a in itertools.product([0, 1], repeat=M)]

    #         fs = final_singleton.copy()
    #         # transfer to the formate of pandas.dataframe
    #         for remove_ind in index_rule:
    #             fs.remove(final_singleton[remove_ind])

    #         df_fs = pd.DataFrame(lst, columns=fs)

    #         # for j in range(len(lst)):
    #         for i_ in index_rule:
    #             insert_column = pd.DataFrame([1]*2**M)
    #             df_fs.insert(i_, final_singleton[i_], insert_column)

    #         lst_all = pd.concat([df_fs, lst_all])
    #         lst_all = lst_all.reset_index(drop=True)

    #     # Get the unique elements of the truth table
    #     lst_all_ = set()
    #     for i in lst_all.values:
    #         lst_all_.add(tuple(i))

    #     # transfer the set of truth_tables to the narray
    #     minterms_ = []
    #     for i in list(lst_all_):
    #         minterms_.append(list(np.asarray(i).data))
    #     print('minterms:')
    #     print(minterms_)

    #     # In[29]: POS_minterm process
    #     sym = "a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x"
    #     a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x = symbols(
    #         sym)
    #     sym_array = [a, b, c, d, e, f, g, h, i, j,
    #                  k, l, m, n, o, p, q, r, s, t, u, v, w, x]
    #     ns_all = list()

    #     for s_, singleton_ in zip(sym_array, final_singleton):
    #         ns_all.append(Symbol(singleton_))

    #     # print('ns_all:')
    #     # print(ns_all)
    #     # In[30]: call the simplify_logic and get the Simplify_decisionRules
    #     Simplify_DecisionRules = "{}".format(
    #         simplify_logic(SOPform(ns_all, minterms_), 'dnf'))

    #     rule_str = Simplify_DecisionRules
    #     print('rule_str:')
    #     print(Simplify_DecisionRules)
    #     print('final_singleton:')
    #     print(final_singleton)
    #     j = 65
    #     for i in range(len(final_singleton)):
    #         a = j + i
    #         rule_str = rule_str.replace(final_singleton[i], str(chr(a)))

    #     regex = re.compile('[A-Za-z]')
    #     rule_str_sp = rule_str.split('|')

    #     decision_rule = {}
    #     for i in range(len(rule_str_sp)):
    #         temp = regex.findall(rule_str_sp[i])
    #         decision_rule.setdefault(i, temp)

    #     nodeCounter = {}
    #     for k, v in decision_rule.items():
    #         for node in v:
    #             if node not in nodeCounter:
    #                 nodeCounter[node] = 1
    #             else:
    #                 nodeCounter[node] += 1

    #     nodeCounter = dict(
    #         sorted(nodeCounter.items(), key=lambda x: x[1], reverse=True))
    #     nodeOrder = list(nodeCounter.keys())
    #     for k, v in decision_rule.items():
    #         decision_rule[k] = list(
    #             sorted(v, key=lambda x: nodeOrder.index(x)))
    #     print("nodeOrder:")
    #     print(nodeOrder)
    #     with open("Decision_rule/decision_rule_"+str(PID)+".json", "w") as decision_rule_file:
    #         json.dump(decision_rule, decision_rule_file, indent=4)

    #     decision_rule_map = {}
    #     for i in range(len(final_singleton)):
    #         a = j + i
    #         decision_rule_map.setdefault(str(chr(a)), final_singleton[i])

    #     with open("Decision_rule/decision_rule_map_"+str(PID)+".json", "w") as decision_rule_file:
    #         json.dump(decision_rule_map, decision_rule_file, indent=4)

    #     rule_str_sp = Simplify_DecisionRules
    #     rule_str_sp = rule_str_sp.split('|')
    #     for i in range(len(rule_str_sp)):
    #         rule_str_sp[i] = rule_str_sp[i].replace('&', 'and').replace(
    #             ' (', '').replace('(', '').replace(') ', '').replace(')', '')

    #     AV_FIDELITYS = []
    #     for condition in rule_str_sp:
    #         fidelity = []
    #         for val_df in VAL_DATASET:
    #             stack_pred = stacking_model.predict(val_df)
    #             val_df = val_df.rename(columns=d_path)
    #             merge_pred = np.where(val_df.eval(condition), rule_1, rule_2)
    #             for tree_idx in tree_candidates[explain_i]:
    #                 fidelity.append(accuracy_score(stack_pred, merge_pred))
    #         AV_FIDELITYS.append(round(np.mean(fidelity), 3))
    #     # print(AV_FIDELITYS)

    #     cytime = end_c - start_c
    #     print("Start time = {}".format(start_c))
    #     print("End time = {}".format(end_c))
    #     print("Cython time = {}".format(cytime))
    # return (personal_result)


if __name__ == "__main__":
    run_id = [631, 8131, 8381, 8631, 9131, 10221, 11121, 12601, 12751, 12331,
              11291, 42, 1302, 18722, 23172, 23852, 24092, 24302, 24362, 24402, 24462]
    for i in range(len(run_id)):
        personalDP(run_id[i])
